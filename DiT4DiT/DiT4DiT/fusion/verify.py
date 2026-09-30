"""Unit invariants and real-model checks. Writes evidence; fails on any violation."""
import argparse
import copy
from pathlib import Path
import time

import torch
from omegaconf import OmegaConf

from DiT4DiT.model.framework.QwenConcat import DiT4DiTQwenConcat, FeatureConcat
from DiT4DiT.model.modules.action_model.ActionDiT import FlowmatchingActionHead
from .backbones import FrozenObservationEncoder
from .common import atomic_json, load_config, require_ppu
from .data import IndexedLibero


def unit_checks(cfg):
    torch.manual_seed(42)
    fusion = FeatureConcat(4, 6, 2048).cuda()
    c = torch.randn(2, 7, 4, device="cuda", requires_grad=True)
    q = torch.randn(2, 5, 6, device="cuda", requires_grad=True)
    qm = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], device="cuda").bool()
    small = copy.deepcopy(cfg)
    small.framework.action_model.diffusion_model_cfg.num_layers = 2
    head = FlowmatchingActionHead(small).cuda().eval()
    state = torch.randn(2, 1, 16, device="cuda")
    noise = torch.randn(2, 8, 8, device="cuda")
    for condition in ("mask_cosmos", "mask_qwen"):
        h, m = fusion(c, q, qm, condition)
        before = head.predict_action(h, state, m, noise)
        cc = c * 100 + 50 if condition == "mask_cosmos" else c
        qq = q * 100 + 50 if condition == "mask_qwen" else q
        h2, m2 = fusion(cc, qq, qm, condition)
        after = head.predict_action(h2, state, m2, noise)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    qpad = q.detach().clone()
    qpad[~qm] = 1000
    h, m = fusion(c, q, qm)
    hp, mp = fusion(c, qpad, qm)
    torch.testing.assert_close(head.predict_action(h, state, m, noise), head.predict_action(hp, state, mp, noise), rtol=0, atol=0)
    head.train()
    actions = torch.randn(2, 8, 8, device="cuda")
    am = torch.ones_like(actions, dtype=torch.bool)
    am[:, :, 7] = False
    head.model.gradient_checkpointing = True
    loss = head(h, actions, am, state, encoder_attention_mask=m)
    loss.backward()
    assert c.grad is None and q.grad is None, "Backbone features must be detached"
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fusion.parameters())
    head.eval()
    # Checkpoint recomputation must preserve dropout and gradients.
    checked = copy.deepcopy(head).train()
    plain = copy.deepcopy(head).train()
    checked.model.gradient_checkpointing = True
    plain.model.gradient_checkpointing = False
    for module in (checked, plain):
        module.zero_grad(set_to_none=True)
        torch.manual_seed(912)
        module(h.detach(), actions, am, state, encoder_attention_mask=m).backward()
    for a, b in zip(checked.parameters(), plain.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=2e-4)
    return {"mask_invariance": True, "padding_invariance": True,
            "adapter_gradients": True, "detached_features": True, "checkpoint_gradients": True}


def full_checks(cfg):
    raw = IndexedLibero(cfg)
    examples = [raw[start] for start, _ in raw.suite_ranges]
    frozen = FrozenObservationEncoder(cfg, "cuda")
    samples = frozen.encode(examples)
    for s in samples:
        assert s["qwen"].shape[1] == 2560
        assert s["cond_mean"].shape == (16, 2, 28, 56)
    model = DiT4DiTQwenConcat(cfg).cuda()
    model.observation_encoder = frozen
    model.eval()
    before = model.predict_action([samples[0]], action_seed=18)["normalized_actions"]
    online = model.predict_action([examples[0]], action_seed=18)["normalized_actions"]
    # BF16 GEMM can round differently for a padded batch and a single example.
    torch.testing.assert_close(torch.from_numpy(before), torch.from_numpy(online), rtol=1e-2, atol=5e-3)
    single = frozen.encode([examples[0]])
    single_pred = model.predict_action(single, action_seed=18)["normalized_actions"]
    torch.testing.assert_close(torch.from_numpy(single_pred), torch.from_numpy(online), rtol=0, atol=0)
    replaced = copy.copy(examples[0])
    replaced["image"] = [examples[0]["image"][0]] + [torch.zeros_like(f) for f in examples[0]["image"][1:]]
    after = model.predict_action([replaced], action_seed=18)["normalized_actions"]
    torch.testing.assert_close(torch.from_numpy(online), torch.from_numpy(after), rtol=0, atol=0)
    poisoned = dict(samples[0], target_mean=torch.full_like(samples[0]["target_mean"], 1000))
    pred = model.predict_action([poisoned], action_seed=18)["normalized_actions"]
    torch.testing.assert_close(torch.from_numpy(before), torch.from_numpy(pred), rtol=0, atol=0)
    # Match the upstream Cosmos feature call using the same latent and text inputs.
    from DiT4DiT.model.modules.vlm.Cosmos25 import Cosmos25FeatureExtractor
    from diffusers import UniPCMultistepScheduler
    ref = Cosmos25FeatureExtractor.__new__(Cosmos25FeatureExtractor)
    torch.nn.Module.__init__(ref)
    ref.transformer = model.cosmos.transformer
    ref.vae = frozen.vae
    ref.config = cfg
    ref.vae_scale_factor_temporal, ref.vae_scale_factor_spatial = 4, 8
    ref.max_sequence_length, ref.extract_layer = 512, 17
    ref.scheduler = UniPCMultistepScheduler.from_pretrained(Path(cfg.framework.cosmos25.base_model) / "scheduler")
    ref._cached_hidden, ref._capture_hidden_enabled, ref._hook_handle = [], True, None
    ref._register_hook()
    c, _, _, cond, text = model.context_features([samples[0]], seed=42)
    noise = torch.randn_like(cond)
    mask = torch.zeros_like(cond[:, :1]); mask[:, :, 0] = 1
    indicator = mask[:, :, :, :1, :1]
    ref.encode_prompt = lambda **kwargs: (text, None)
    ref.prepare_latents = lambda **kwargs: (noise, cond, mask, indicator)
    checkpointing = ref.transformer.gradient_checkpointing
    # The upstream wrapper forcibly enables grad even inside inference_mode.
    ref.transformer.gradient_checkpointing = False
    with torch.inference_mode():
        expected = ref(prompts=[samples[0]["lang"]], videos=torch.zeros(1, 3, 1, 224, 448, device="cuda"),
                       height=224, width=448, num_frames_out=5, conditional_frame_timestep=0.0001)
        actual = model.cosmos.features(cond, text, noise=noise)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    ref._hook_handle.remove()
    ref.transformer.gradient_checkpointing = checkpointing
    model.train()
    losses = model([samples[0]])
    losses["action_loss"].backward()
    assert all(p.grad is None for p in model.cosmos.parameters())
    for part in (model.fusion.cosmos, model.fusion.qwen, model.action_model):
        assert any(p.grad is not None and p.grad.abs().max() > 0 for p in part.parameters())
    losses["future_video_loss"].backward()
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.cosmos.parameters())
    assert all(p.grad is None and not p.requires_grad for p in frozen.parameters())
    assert torch.isfinite(losses["action_loss"]) and torch.isfinite(losses["future_video_loss"])
    return {"real_suites": len(examples), "cosmos_tokens": int(c.shape[1]),
        "qwen_tokens": [len(s["qwen"]) for s in samples], "cache_online_equal": True,
        "batch_rounding_max_action_difference": float(abs(before-online).max()),
        "future_isolation": True, "upstream_cosmos_feature_parity": True,
        "gradient_routes": True, "action_loss": float(losses["action_loss"]),
        "video_loss": float(losses["future_video_loss"]), "peak_gib": torch.cuda.max_memory_allocated()/1024**3}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--full", action="store_true")
    p.add_argument("--output", default="logs/concat_validation.json")
    args = p.parse_args()
    require_ppu()
    cfg = load_config(args.config)
    started = time.monotonic()
    results = {"unit": unit_checks(cfg)}
    torch.cuda.empty_cache()
    if args.full:
        results["full"] = full_checks(cfg)
    results.update(passed=True, seconds=time.monotonic()-started)
    atomic_json(args.output, results)
    print(results, flush=True)


if __name__ == "__main__":
    main()
