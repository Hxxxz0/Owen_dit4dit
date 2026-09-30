"""Full Qwen/Cosmos joint training with online current-image Qwen features."""
import torch

from DiT4DiT.model.tools import FRAMEWORK_REGISTRY
from DiT4DiT.model.framework.QwenConcat import DiT4DiTQwenConcat
from DiT4DiT.fusion.backbones import FrozenObservationEncoder, load_qwen_backbone, qwen_inputs
from DiT4DiT.fusion.common import generator
from DiT4DiT.fusion.interventions import noisy_observations


@FRAMEWORK_REGISTRY.register("DiT4DiTQwenJoint")
class DiT4DiTQwenJoint(DiT4DiTQwenConcat):
    def __init__(self, config):
        super().__init__(config)
        if not self.cosmos.action_gradients:
            raise ValueError("Joint training requires Cosmos action_gradients=true")
        self.fusion.detach_inputs = False
        # Load after the old modules so their seeded initialization is unchanged.
        self.qwen, self.qwen_processor = load_qwen_backbone(config)
        self.qwen.requires_grad_(True)
        if config.framework.qwenvl.get("gradient_checkpointing", True):
            self.qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def observations(self, samples, include_target=False):
        if any("qwen" in sample for sample in samples):
            raise ValueError("Joint Qwen cannot consume stale cached Qwen features")
        if any("image" not in sample or len(sample["image"]) != 1 for sample in samples):
            raise ValueError("Joint Qwen requires exactly one current image per sample")
        if all("cond_mean" in s and "cond_logvar" in s for s in samples):
            if include_target and any("target_mean" not in s for s in samples):
                raise ValueError("Joint video supervision requires frozen VAE targets")
            return samples
        if include_target:
            raise ValueError("Training requires the validated frozen VAE target cache")
        if self.observation_encoder is None:
            self.observation_encoder = FrozenObservationEncoder(self.config, self.device, load_qwen=False)
        posteriors = self.observation_encoder.posteriors(samples, include_target=False)
        return [{**sample, **posterior} for sample, posterior in zip(samples, posteriors)]

    def context_features(self, samples, seed=None):
        samples = self.observations(samples)
        text = self.texts(samples)
        mean, logvar = self.tensor_batch(samples, "cond_mean"), self.tensor_batch(samples, "cond_logvar")
        cond_gen = generator(self.device, seed, "condition") if seed is not None else None
        video_gen = generator(self.device, seed, "video_noise") if seed is not None else None
        cond = self.cosmos.sample_condition(mean, logvar, cond_gen)
        c = self.cosmos.features(cond, text, generator=video_gen)
        batch = qwen_inputs(self.qwen_processor, samples, self.device)
        q = self.qwen(**batch, use_cache=False, output_hidden_states=False, return_dict=True).last_hidden_state
        return c, q, batch["attention_mask"].bool(), cond, text

    @torch.inference_mode()
    def action_context(self, examples, intervention="full", donor_examples=None,
                       feature_seed=42, noise_std=0.1, initial_examples=None, initial_encoded=None):
        samples = self.observations(examples)
        fusion_intervention = intervention
        if intervention in ("freeze_cosmos", "freeze_qwen"):
            if initial_encoded is None:
                if initial_examples is None:
                    raise ValueError("Branch freezing requires the episode's initial image")
                initial_encoded = self.observations(initial_examples)
            if len(samples) != len(initial_encoded):
                raise ValueError("Current and initial observation batches must match")
            keys = ("cond_mean", "cond_logvar") if intervention == "freeze_cosmos" else ("image",)
            samples = [{**s, **{k: initial[k] for k in keys}} for s, initial in zip(samples, initial_encoded)]
            fusion_intervention = "full"
        elif intervention in ("noise_cosmos", "noise_qwen"):
            noisy = noisy_observations(examples, feature_seed, noise_std)
            if intervention == "noise_qwen":
                samples = [{**s, "image": n["image"]} for s, n in zip(samples, noisy)]
            else:
                if self.observation_encoder is None:
                    self.observation_encoder = FrozenObservationEncoder(self.config, self.device, load_qwen=False)
                posteriors = self.observation_encoder.posteriors(noisy, include_target=False)
                samples = [{**s, **p} for s, p in zip(samples, posteriors)]
            fusion_intervention = "full"
        c, q, mask, _, _ = self.context_features(samples, seed=feature_seed)
        donor = None
        if intervention.startswith("swap"):
            if donor_examples is None:
                raise ValueError("Current donor images are required for joint-model swaps")
            donor = self.context_features(donor_examples, seed=feature_seed)[:3]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.fusion(c, q, mask, fusion_intervention, donor)
