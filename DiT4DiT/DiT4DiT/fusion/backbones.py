"""Frozen observation encoders and the two independent Cosmos forward paths.

Future targets are accepted only by video_loss(), never by features().
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from diffusers import AutoencoderKLWan, CosmosTransformer3DModel, UniPCMultistepScheduler
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration


def current_views(example):
    image = example["image"][0]
    image = torch.as_tensor(np.asarray(image).copy() if not torch.is_tensor(image) else image)
    if image.shape[-1] == 3:
        image = image.permute(2, 0, 1)
    if image.shape != (3, 224, 448):
        raise ValueError(f"Expected current primary/wrist mosaic [3,224,448], got {image.shape}")
    byte_pixels = image.dtype == torch.uint8
    image = image.float()
    if byte_pixels or image.max() > 1:
        image = image / 255
    # The same uint8 conversion is used during cache preparation and live inference.
    return [(part.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            for part in image.split(224, dim=-1)]


class FrozenObservationEncoder(nn.Module):
    def __init__(self, cfg, device="cpu", load_qwen=True):
        super().__init__()
        if load_qwen:
            self.qwen, self.processor = load_qwen_backbone(cfg)
        self.vae = AutoencoderKLWan.from_pretrained(
            cfg.framework.cosmos25.base_model, subfolder="vae", torch_dtype=torch.bfloat16,
            local_files_only=True)
        self.requires_grad_(False)
        self.to(device).eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def qwen_features(self, examples):
        messages = [[{"role": "user", "content": [
            *[{"type": "image", "image": image} for image in current_views(ex)],
            {"type": "text", "text": ex["lang"]}]}] for ex in examples]
        batch = self.processor.apply_chat_template(
            messages, tokenize=True, padding=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(next(self.qwen.parameters()).device)
        result = self.qwen(**batch, use_cache=False, output_hidden_states=False, return_dict=True)
        return result.last_hidden_state.detach(), batch["attention_mask"].bool()

    @torch.no_grad()
    def posteriors(self, examples, include_target=True):
        device = next(self.vae.parameters()).device
        results = []
        for ex in examples:
            frames = torch.stack([x if torch.is_tensor(x) else torch.from_numpy(np.asarray(x).copy())
                                  for x in ex["image"]]).float()
            if frames.shape[-1] == 3:
                frames = frames.permute(0, 3, 1, 2)
            if frames.max() > 1:
                frames = frames / 255
            if frames.shape[1:] != (3, 224, 448):
                raise ValueError("VAE input must be [T,3,224,448]")
            cond = torch.zeros((1, 3, 5, 224, 448), device=device, dtype=self.vae.dtype)
            cond[:, :, 0] = frames[0].to(device) * 2 - 1
            posterior = self.vae.encode(cond).latent_dist
            item = {"cond_mean": posterior.mean[0].float(), "cond_logvar": posterior.logvar[0].float()}
            if include_target:
                if frames.shape[0] != 5:
                    raise ValueError("Video supervision needs exactly five observed/target frames")
                target = frames.permute(1, 0, 2, 3).unsqueeze(0).to(device, self.vae.dtype) * 2 - 1
                item["target_mean"] = self.vae.encode(target).latent_dist.mean[0].float()
            results.append(item)
        return results

    @torch.no_grad()
    def encode(self, examples, include_target=True):
        hidden, mask = self.qwen_features(examples)
        posterior = self.posteriors(examples, include_target)
        return [{**p, "qwen": hidden[i, mask[i]].detach().cpu().contiguous(),
                 **{k: torch.as_tensor(ex[k]) for k in ("action", "state", "action_mask") if k in ex},
                 "lang": ex["lang"]} for i, (p, ex) in enumerate(zip(posterior, examples))]


def load_qwen_backbone(cfg):
    qpath = cfg.framework.qwenvl.base_vlm
    full, info = Qwen3VLForConditionalGeneration.from_pretrained(
        qpath, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
        output_loading_info=True)
    if info["missing_keys"] or info["unexpected_keys"] or info.get("mismatched_keys"):
        raise ValueError(f"Qwen checkpoint failed strict initialization: {info}")
    processor = AutoProcessor.from_pretrained(qpath, local_files_only=True)
    processor.image_processor.size = {"shortest_edge": 224**2, "longest_edge": 224**2}
    processor.tokenizer.padding_side = "right"
    return full.model, processor


def qwen_inputs(processor, examples, device):
    messages = [[{"role": "user", "content": [
        *[{"type": "image", "image": image} for image in current_views(ex)],
        {"type": "text", "text": ex["lang"]}]}] for ex in examples]
    return processor.apply_chat_template(messages, tokenize=True, padding=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt").to(device)


@torch.no_grad()
def encode_texts(cfg, instructions, device):
    root = cfg.framework.cosmos25.base_model
    tokenizer = AutoTokenizer.from_pretrained(root, subfolder="tokenizer", local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        root, subfolder="text_encoder", dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True).to(device).eval()
    result = {}
    for text in sorted(set(instructions)):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant who will provide prompts to an image generator."}]},
            {"role": "user", "content": [{"type": "text", "text": text}]}]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
            add_vision_id=False, max_length=512, truncation=True, padding="max_length")
        output = model(torch.tensor([ids], device=device), output_hidden_states=True)
        # Match upstream: normalization is performed in the encoder output dtype.
        layers = [(h - h.mean(-1, keepdim=True)) / (h.std(-1, keepdim=True) + 1e-8)
                  for h in output.hidden_states[1:]]
        result[text] = torch.cat(layers, dim=-1)[0].cpu().contiguous()
    del model
    torch.cuda.empty_cache()
    return result


class JointCosmos(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        root = Path(cfg.framework.cosmos25.base_model)
        self.transformer = CosmosTransformer3DModel.from_pretrained(
            root / "transformer", torch_dtype=torch.bfloat16, local_files_only=True)
        vae_cfg = json.loads((root / "vae/config.json").read_text())
        self.register_buffer("latent_mean", torch.tensor(vae_cfg["latents_mean"]).view(1, 16, 1, 1, 1), persistent=False)
        self.register_buffer("latent_std", torch.tensor(vae_cfg["latents_std"]).view(1, 16, 1, 1, 1), persistent=False)
        self.normalization_values = (vae_cfg["latents_mean"], vae_cfg["latents_std"])
        scheduler = UniPCMultistepScheduler.from_pretrained(root / "scheduler")
        scheduler.set_timesteps(1)
        self.feature_sigma = float(scheduler.sigmas[0])
        self.conditional_t = float(cfg.framework.cosmos25.conditional_frame_timestep)
        self.capture = False
        self.hidden = None
        self.action_gradients = bool(cfg.framework.cosmos25.get("action_gradients", False))
        self.transformer.transformer_blocks[int(cfg.framework.cosmos25.extract_layer)].register_forward_hook(self._capture)
        if cfg.trainer.get("gradient_checkpointing", True):
            self.transformer.enable_gradient_checkpointing()

    def _capture(self, module, inputs, output):
        if self.capture:
            hidden = output[0] if isinstance(output, tuple) else output
            self.hidden = hidden if self.action_gradients else hidden.detach()

    def normalize(self, x):
        # DeepSpeed casts module buffers to BF16; retain the original FP32 constants.
        mean, std = [torch.tensor(v, device=x.device, dtype=torch.float32).view(1, 16, 1, 1, 1)
                     for v in self.normalization_values]
        return (x.float() - mean) / std

    def sample_condition(self, mean, logvar, generator=None):
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        # VAE posterior sampling is BF16 upstream, then converted to FP32 for normalization.
        dtype = self.transformer.dtype
        raw = mean.to(dtype) + torch.exp(logvar.to(dtype) * 0.5) * noise.to(dtype)
        return self.normalize(raw)

    def _call(self, x, cond, t, text):
        mask = torch.zeros_like(x[:, :1])
        mask[:, :, :1] = 1
        timestep = t.reshape(-1, 1, 1, 1, 1).expand(x.shape[0], 1, x.shape[2], 1, 1).clone()
        timestep[:, :, :1] = self.conditional_t
        dtype = self.transformer.dtype
        return self.transformer(
            hidden_states=(mask * cond + (1 - mask) * x).to(dtype),
            condition_mask=mask.to(dtype), timestep=timestep.to(dtype),
            encoder_hidden_states=text.to(dtype),
            padding_mask=x.new_zeros(1, 1, 224, 448, dtype=dtype), return_dict=False)[0]

    def features(self, cond, text, noise=None, generator=None):
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.action_gradients):
            return self._features(cond, text, noise, generator)

    def _features(self, cond, text, noise=None, generator=None):
        if noise is None:
            noise = torch.randn(cond.shape, device=cond.device, dtype=torch.float32, generator=generator)
        self.capture = True
        self.hidden = None
        try:
            self._call(noise, cond, torch.full((cond.shape[0],), self.feature_sigma, device=cond.device), text)
            if self.hidden is None:
                raise RuntimeError("Cosmos extraction hook did not run")
            return self.hidden
        finally:
            self.capture = False
            self.hidden = None

    def video_loss(self, cond, target_mean, text, noise=None, times=None):
        target = self.normalize(target_mean)
        noise = torch.randn_like(target) if noise is None else noise
        times = torch.rand(target.shape[0], device=target.device) if times is None else times
        xt = (1 - times[:, None, None, None, None]) * target + times[:, None, None, None, None] * noise
        prediction = self._call(xt, cond, times, text)
        return F.mse_loss(prediction[:, :, 1:].float(), (noise - target)[:, :, 1:].float())
