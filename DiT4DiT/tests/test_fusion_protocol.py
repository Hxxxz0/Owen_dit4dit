"""Regression checks for sampling, LR floors, and paired scientific reporting."""
import copy
import unittest
from unittest.mock import Mock, call

import torch

from DiT4DiT.fusion.common import CONDITIONS, SUITES
from DiT4DiT.fusion.data import MixtureBatches
from DiT4DiT.fusion.report import OFFLINE_METRICS, summarize, summarize_full, summarize_offline
from DiT4DiT.fusion.train import make_scheduler


class FusionProtocolTests(unittest.TestCase):
    def test_indexed_cache_never_randomly_substitutes_failed_sample(self):
        from DiT4DiT.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset
        mixture = object.__new__(LeRobotMixtureDataset)
        mixture.retry_same_index = True
        mixture.sample_step = Mock(side_effect=ValueError("invalid source frame"))
        with self.assertRaisesRegex(ValueError, "invalid source frame"):
            mixture[7]
        self.assertEqual(mixture.sample_step.call_args_list, [call(7)]*10)

    def test_resume_sampler_ignores_prefetch(self):
        options = dict(ranges=[(0, 100), (100, 180), (180, 500), (500, 1000)],
                       batch_size=4, microsteps=12, seed=42, rank=7)
        full = list(MixtureBatches(**options))
        self.assertEqual(full[6:], list(MixtureBatches(**options, start=6)))
        self.assertNotEqual(full, list(MixtureBatches(**{**options, "rank": 8})))

    def test_absolute_lr_floor_for_different_groups(self):
        optimizer = torch.optim.AdamW([
            {"params": [torch.nn.Parameter(torch.ones(()))], "lr": lr} for lr in (1e-5, 1e-4)])
        scheduler = make_scheduler(optimizer, steps=160, warmup=10, minimum=5e-7)
        self.assertEqual(scheduler.get_last_lr(), [0., 0.])
        for step in range(1, 161):
            optimizer.step()
            scheduler.step()
            if step == 10:
                self.assertEqual(scheduler.get_last_lr(), [1e-5, 1e-4])
        for lr in scheduler.get_last_lr():
            self.assertAlmostEqual(lr, 5e-7, places=15)

    @staticmethod
    def episodes():
        return [{"suite": s, "task": t, "trial": i, "condition": c,
                 "checkpoint_sha256": "same", "episode_seed": i,
                 "success": c != "mask_cosmos"}
                for s in SUITES for t in range(10) for i in range(2) for c in CONDITIONS]

    def test_paired_report_uses_complete_rounds(self):
        rows = self.episodes()
        report = summarize(rows, bootstrap=200)
        self.assertEqual(report["trials_per_task"], 2)
        self.assertEqual(report["conditions"]["mask_cosmos"]["paired_ci95"], [1., 1.])
        self.assertEqual(report["conditions"]["mask_qwen"]["drop"], 0.)
        self.assertEqual(summarize(rows[:-1], bootstrap=200)["trial_indices"], [0])

    def test_report_rejects_mixed_checkpoints_and_unpaired_seeds(self):
        rows = self.episodes()
        rows[0]["checkpoint_sha256"] = "other"
        with self.assertRaises(ValueError):
            summarize(rows)
        rows = self.episodes()
        rows[0]["episode_seed"] = 19
        with self.assertRaises(ValueError):
            summarize(rows)

    def test_intermediate_report_requires_all_tasks(self):
        rows = [r for r in self.episodes() if r["condition"] == "full"]
        self.assertEqual(summarize_full(rows)["success_rate"], 1.)
        self.assertEqual(summarize_full(rows[:-1])["trials_per_task"], 1)

    def test_offline_pairing_and_equal_task_weights(self):
        rows = [{"checkpoint_sha256": "same", "lang": lang, "sample_index": index,
                 "condition": condition, **{m: value for m in OFFLINE_METRICS}}
                for lang, value in (("task_a", 1.), ("task_b", 3.))
                for index in range(2) for condition in CONDITIONS]
        report = summarize_offline(rows, expected_tasks=2, expected_per_task=2)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["conditions"]["full"]["translation_delta_l2"], 2.)
        self.assertEqual(summarize_offline(rows[:-1], 2, 2)["status"], "incomplete")
        with self.assertRaises(ValueError):
            summarize_offline(rows + [copy.deepcopy(rows[0])], 2, 2)


if __name__ == "__main__":
    unittest.main()
