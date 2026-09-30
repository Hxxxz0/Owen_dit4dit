"""Cosmos/Qwen feature concatenation with explicit intervention boundaries."""
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn

from DiT4DiT.model.framework.base_framework import baseframework
from DiT4DiT.model.tools import FRAMEWORK_REGISTRY
from DiT4DiT.model.modules.action_model.ActionDiT import FlowmatchingActionHead
from DiT4DiT.fusion.backbones import FrozenObservationEncoder, JointCosmos
from DiT4DiT.fusion.common import CONDITIONS, fingerprint, generator
from DiT4DiT.fusion.interventions import freeze_encoded_observations, intervene_encoded_observations


class FeatureConcat(nn.Module):
    def __init__(self, cosmos_dim=2048, qwen_dim=2560, output_dim=2048, detach_inputs=True):
        super().__init__()
        self.detach_inputs = detach_inputs
        self.cosmos = nn.Sequential(nn.LayerNorm(cosmos_dim), nn.Linear(cosmos_dim, output_dim))
        self.qwen = nn.Sequential(nn.LayerNorm(qwen_dim), nn.Linear(qwen_dim, output_dim))
        self.source = nn.Parameter(torch.zeros(2, output_dim))
        nn.init.normal_(self.source, std=0.02)

    def forward(self, cosmos, qwen, qwen_mask, condition="full", donor=None):
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown intervention: {condition}")
        if condition.startswith("swap"):
            if donor is None:
                raise ValueError("Swap requires an explicit donor feature pair")
            dc, dq, dm = donor
            if condition == "swap_cosmos":
                cosmos = dc
            else:
                qwen, qwen_mask = dq, dm
        if self.detach_inputs:
            cosmos, qwen = cosmos.detach(), qwen.detach()
        c, q = self.cosmos(cosmos) + self.source[0], self.qwen(qwen) + self.source[1]
        cmask = torch.ones(c.shape[:2], device=c.device, dtype=torch.bool)
        qmask = qwen_mask.to(device=q.device, dtype=torch.bool).clone()
        if condition == "mask_cosmos":
            cmask.zero_()
        elif condition == "mask_qwen":
            qmask.zero_()
        mask = torch.cat([cmask, qmask], dim=1)
        if not mask.any(dim=1).all():
            raise ValueError("Cross-attention requires at least one valid context token")
        # Additive mask works consistently with diffusers attention processors.
        attention_mask = torch.zeros_like(mask, dtype=c.dtype).masked_fill(~mask, -torch.inf)[:, None, :]
        return torch.cat([c, q], dim=1), attention_mask


@FRAMEWORK_REGISTRY.register("DiT4DiTQwenConcat")
class DiT4DiTQwenConcat(baseframework):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cosmos = JointCosmos(config)
        self.fusion = FeatureConcat()
        self.action_model = FlowmatchingActionHead(config)
        self.action_model.model.gradient_checkpointing = bool(config.trainer.get("gradient_checkpointing", True))
        root = Path(config.cache.root)
        if json.loads((root / "provenance.json").read_text())["fingerprint"] != fingerprint(config):
            raise ValueError("Text cache provenance mismatch")
        self.text_cpu = torch.load(root / "text.pt", mmap=True, weights_only=True)
        self.text_device = {}
        self.observation_encoder = None

    @property
    def device(self):
        return next(self.cosmos.parameters()).device

    def texts(self, samples):
        result = []
        for sample in samples:
            key = sample["lang"]
            if key not in self.text_cpu:
                raise KeyError(f"Instruction not in the versioned LIBERO text cache: {key}")
            if key not in self.text_device or self.text_device[key].device != self.device:
                self.text_device[key] = self.text_cpu[key].to(self.device)
            result.append(self.text_device[key])
        return torch.stack(result)

    def tensor_batch(self, samples, key):
        return torch.stack([torch.as_tensor(s[key]) for s in samples]).to(self.device)

    def observations(self, samples, include_target=False):
        if "cond_mean" in samples[0]:
            return samples
        if self.observation_encoder is None:
            self.observation_encoder = FrozenObservationEncoder(self.config, self.device)
        return self.observation_encoder.encode(samples, include_target=include_target)

    def context_features(self, samples, seed=None):
        samples = self.observations(samples)
        text = self.texts(samples)
        mean, logvar = self.tensor_batch(samples, "cond_mean"), self.tensor_batch(samples, "cond_logvar")
        cond_gen = generator(self.device, seed, "condition") if seed is not None else None
        video_gen = generator(self.device, seed, "video_noise") if seed is not None else None
        cond = self.cosmos.sample_condition(mean, logvar, cond_gen)
        c = self.cosmos.features(cond, text, generator=video_gen)
        lengths = [len(s["qwen"]) for s in samples]
        q = torch.zeros(len(samples), max(lengths), 2560, dtype=c.dtype, device=self.device)
        mask = torch.zeros(len(samples), max(lengths), dtype=torch.bool, device=self.device)
        for i, s in enumerate(samples):
            q[i, :lengths[i]] = s["qwen"].to(self.device)
            mask[i, :lengths[i]] = True
        return c, q, mask, cond, text

    def forward(self, examples, **kwargs):
        samples = self.observations(examples, include_target=True)
        c, q, mask, cond, text = self.context_features(samples)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            context, attention_mask = self.fusion(c, q, mask)
            repeat = int(self.config.trainer.repeated_diffusion_steps)
            action_loss = self.action_model(
                context.repeat(repeat, 1, 1), self.tensor_batch(samples, "action").to(context.dtype).repeat(repeat, 1, 1),
                self.tensor_batch(samples, "action_mask").repeat(repeat, 1, 1),
                self.tensor_batch(samples, "state").to(context.dtype).repeat(repeat, 1, 1),
                encoder_attention_mask=attention_mask.repeat(repeat, 1, 1))
            video_loss = self.cosmos.video_loss(cond, self.tensor_batch(samples, "target_mean"), text)
        return {"action_loss": action_loss, "future_video_loss": video_loss}

    @torch.inference_mode()
    def action_context(self, examples, intervention="full", donor_examples=None,
                       feature_seed=42, noise_std=0.1, initial_examples=None, initial_encoded=None):
        """Encode visual/text context independently of the live proprioceptive state."""
        samples = self.observations(examples)
        fusion_intervention = intervention
        if intervention in ("freeze_cosmos", "freeze_qwen"):
            if initial_encoded is None:
                if initial_examples is None:
                    raise ValueError("Branch freezing requires the episode's initial image")
                initial_encoded = self.observations(initial_examples)
            samples = freeze_encoded_observations(samples, initial_encoded, intervention)
            fusion_intervention = "full"
        if intervention in ("noise_cosmos", "noise_qwen"):
            samples = intervene_encoded_observations(
                self.observation_encoder, examples, samples, intervention, feature_seed, noise_std)
            fusion_intervention = "full"
        c, q, mask, _, _ = self.context_features(samples, seed=feature_seed)
        donor = None
        if intervention.startswith("swap"):
            if donor_examples is None:
                raise ValueError("Donor observations are required for a swap")
            donor = self.context_features(donor_examples, seed=feature_seed)[:3]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            context, attention_mask = self.fusion(c, q, mask, fusion_intervention, donor)
        return context, attention_mask

    @torch.inference_mode()
    def predict_action(self, examples, intervention="full", donor_examples=None,
                       feature_seed=42, action_seed=42, initial_noise=None, noise_std=0.1,
                       initial_examples=None, initial_encoded=None, **kwargs):
        if isinstance(examples, dict):
            examples = [examples]
        context, attention_mask = self.action_context(
            examples, intervention, donor_examples, feature_seed, noise_std, initial_examples, initial_encoded)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.tensor_batch(examples, "state").to(context.dtype)
            if initial_noise is None:
                initial_noise = torch.randn((len(examples), 8, 8), device=self.device, dtype=context.dtype,
                    generator=generator(self.device, action_seed, "action"))
            else:
                initial_noise = torch.as_tensor(initial_noise, device=self.device)
            actions = self.action_model.predict_action(context, state, attention_mask, initial_noise)
        return {"normalized_actions": actions.float().cpu().numpy()}
