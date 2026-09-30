import unittest
from unittest.mock import Mock

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from DiT4DiT.fusion.backbones import JointCosmos
from DiT4DiT.fusion.joint_checks import optimizer_groups
from DiT4DiT.model.framework.QwenConcat import FeatureConcat
from DiT4DiT.model.framework.QwenJoint import DiT4DiTQwenJoint


class JointTrainingTests(unittest.TestCase):
    def test_dark_uint8_images_keep_pixel_scale(self):
        import numpy as np
        from DiT4DiT.fusion.backbones import current_views
        image = np.ones((224, 448, 3), dtype=np.uint8)
        views = current_views({"image": [image]})
        self.assertTrue(np.array_equal(np.concatenate(views, axis=1), image))

    def test_action_fusion_gradient_routes_and_legacy_default(self):
        for detach in (True, False):
            fusion = FeatureConcat(4, 6, 8, detach_inputs=detach)
            c = torch.randn(2, 3, 4, requires_grad=True)
            q = torch.randn(2, 5, 6, requires_grad=True)
            context, _ = fusion(c, q, torch.ones(2, 5, dtype=torch.bool))
            context.square().mean().backward()
            for x in (c, q):
                self.assertEqual(x.grad is None, detach)
                if not detach:
                    self.assertGreater(x.grad.abs().max(), 0)

    def test_optimizer_rejects_missing_or_duplicated_parameters(self):
        model = nn.Module()
        model.qwen = nn.Linear(2, 3)
        model.cosmos = nn.Linear(2, 3)
        with self.assertRaisesRegex(ValueError, "missing"):
            optimizer_groups(model, {"cosmos": 1e-5})
        groups = optimizer_groups(model, {"qwen": 1e-5, "cosmos": 1e-5})
        self.assertEqual(sum(len(g["params"]) for g in groups), 4)
        model.alias = model.qwen
        with self.assertRaisesRegex(ValueError, "multiple"):
            optimizer_groups(model, {"qwen": 1e-5, "cosmos": 1e-5, "alias": 1e-5})

    def test_joint_inputs_reject_frozen_qwen_and_future_images(self):
        model = object.__new__(DiT4DiTQwenJoint)
        sample = {"image": [torch.zeros(224, 448, 3)], "cond_mean": 1, "cond_logvar": 1, "target_mean": 2}
        self.assertIs(model.observations([sample], include_target=True)[0], sample)
        with self.assertRaisesRegex(ValueError, "stale"):
            model.observations([{**sample, "qwen": torch.ones(1)}])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            model.observations([{**sample, "image": sample["image"] * 5}])

    def test_cosmos_hook_keeps_graph_with_checkpoint_recomputation(self):
        cosmos = JointCosmos.__new__(JointCosmos)
        nn.Module.__init__(cosmos)
        cosmos.action_gradients = True
        cosmos.capture = False
        cosmos.hidden = None
        cosmos.feature_sigma = 1.0
        block = nn.Sequential(nn.Linear(4, 4), nn.SiLU())
        block.register_forward_hook(cosmos._capture)
        cosmos._call = lambda x, cond, t, text: checkpoint(block, x + cond, use_reentrant=False)
        cond = torch.ones(1, 4)
        features = cosmos.features(cond, torch.zeros(1), noise=torch.ones(1, 4))
        # A separate video pass must not overwrite the saved action graph.
        video = cosmos._call(cond, cond, None, None).square().sum()
        action_grad = torch.autograd.grad(features.square().sum(), block[0].weight, retain_graph=True)[0]
        self.assertGreater(action_grad.abs().max(), 0)
        (features.square().sum()+video).backward()
        self.assertTrue(torch.isfinite(block[0].weight.grad).all())
        cosmos.action_gradients = False
        self.assertFalse(cosmos.features(cond, torch.zeros(1)).requires_grad)


if __name__ == "__main__":
    unittest.main()
