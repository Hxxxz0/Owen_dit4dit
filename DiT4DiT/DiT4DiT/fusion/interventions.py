"""Paired image interventions at one observation-encoder boundary."""
import numpy as np

from .protocol import CONDITIONS, seed_for


ATTRIBUTION_CONDITIONS = (*CONDITIONS, "freeze_cosmos", "freeze_qwen", "noise_cosmos", "noise_qwen")


def freeze_encoded_observations(encoded, initial_encoded, condition):
    """Use initial-image features only for the selected branch; state stays live."""
    if condition not in ("freeze_cosmos", "freeze_qwen"):
        raise ValueError(f"Not a branch-freeze intervention: {condition}")
    if len(encoded) != len(initial_encoded):
        raise ValueError("Current and initial observation batches must match")
    keys = ("cond_mean", "cond_logvar") if condition == "freeze_cosmos" else ("qwen",)
    return [{**current, **{key: initial[key] for key in keys}}
            for current, initial in zip(encoded, initial_encoded)]


def noisy_observations(examples, seed, noise_std):
    """Corrupt the same uint8 pixels for either branch; retain language/state."""
    if not 0 <= noise_std <= 1:
        raise ValueError("noise_std must be between zero and one of the pixel range")
    rng = np.random.default_rng(seed_for(seed, "paired_image_noise"))
    result = []
    for example in examples:
        if len(example["image"]) != 1:
            raise ValueError("Attribution inference requires exactly one current frame")
        image = np.asarray(example["image"][0])
        if image.shape != (224, 448, 3) or image.dtype != np.uint8:
            raise ValueError("Expected a uint8 primary/wrist mosaic [224,448,3]")
        noise = rng.normal(0, noise_std * 255, image.shape)
        corrupted = np.clip(np.rint(image.astype(np.float32) + noise), 0, 255).astype(np.uint8)
        result.append({**example, "image": [corrupted]})
    return result


def intervene_encoded_observations(encoder, examples, encoded, condition, seed, noise_std):
    """Replace only the targeted branch; no in-place edits to clean features."""
    if condition not in ("noise_cosmos", "noise_qwen"):
        raise ValueError(f"Not an image-noise intervention: {condition}")
    noisy = noisy_observations(examples, seed, noise_std)
    if condition == "noise_cosmos":
        posteriors = encoder.posteriors(noisy, include_target=False)
        return [{**sample, "cond_mean": posterior["cond_mean"],
                 "cond_logvar": posterior["cond_logvar"]}
                for sample, posterior in zip(encoded, posteriors)]
    hidden, mask = encoder.qwen_features(noisy)
    return [{**sample, "qwen": hidden[i, mask[i]].detach().cpu().contiguous()}
            for i, sample in enumerate(encoded)]
