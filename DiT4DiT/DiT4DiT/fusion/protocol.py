"""Shared preprocessing without importing the policy training environment."""
import hashlib
import json
import numpy as np

CONDITIONS = ("full", "mask_cosmos", "mask_qwen", "swap_cosmos", "swap_qwen")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
LIMITS = dict(zip(SUITES, (220, 280, 300, 520)))


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8], "little") % (2**63-1)


def preprocess_observation(obs, instruction):
    import cv2
    primary = cv2.resize(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), (224,224), interpolation=cv2.INTER_AREA)
    wrist = cv2.resize(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), (224,224), interpolation=cv2.INTER_AREA)
    quat = np.asarray(obs["robot0_eef_quat"]).copy()
    quat[3] = np.clip(quat[3], -1, 1)
    den = np.sqrt(1 - quat[3]**2)
    axis = np.zeros(3) if np.isclose(den, 0) else quat[:3] * 2 * np.arccos(quat[3])/den
    state = np.concatenate([obs["robot0_eef_pos"], axis, obs["robot0_gripper_qpos"]])
    state = np.stack([np.sin(state), np.cos(state)], axis=-1).reshape(1, 16).astype(np.float32)
    return {"image": [np.concatenate([primary, wrist], axis=1)], "state": state, "lang": instruction}


def unnormalize(actions, stats):
    actions = np.clip(np.asarray(actions)[..., :7].copy(), -1, 1)
    actions[..., 6] = np.where(actions[..., 6] < 0.5, 0, 1)
    mask = np.asarray(stats.get("mask", np.ones(7, dtype=bool)))[:7]
    low, high = np.asarray(stats["min"])[:7], np.asarray(stats["max"])[:7]
    actions = np.where(mask, 0.5*(actions+1)*(high-low)+low, actions)
    actions[..., 6] = 1 - 2*(actions[..., 6] > 0.5)
    return actions
