"""Scientific controls: change one branch, preserve paired inputs and samples."""
import copy
import unittest

import numpy as np
import torch

from DiT4DiT.fusion.interventions import (
    ATTRIBUTION_CONDITIONS, freeze_encoded_observations,
    intervene_encoded_observations, noisy_observations,
)
from DiT4DiT.fusion.paired_evaluate import paired_jobs
from DiT4DiT.fusion.report import summarize


class AttributionTests(unittest.TestCase):
    def setUp(self):
        self.example = {"image": [np.full((224, 448, 3), 128, dtype=np.uint8)],
                        "state": np.ones((1, 16)), "lang": "task"}
        self.live = {"qwen": torch.ones(3, 2), "cond_mean": torch.ones(2),
                     "cond_logvar": torch.ones(2), "state": torch.ones(16), "lang": "task"}
        self.initial = {key: value * 7 if torch.is_tensor(value) else value
                        for key, value in self.live.items()}

    def test_freezing_one_branch_keeps_the_other_and_live_state(self):
        for condition, replaced in [("freeze_cosmos", {"cond_mean", "cond_logvar"}),
                                    ("freeze_qwen", {"qwen"})]:
            frozen = freeze_encoded_observations([self.live], [self.initial], condition)[0]
            for key in self.live:
                self.assertIs(frozen[key], self.initial[key] if key in replaced else self.live[key])
        self.assertTrue(torch.equal(self.live["qwen"], torch.ones(3, 2)))

    def test_noise_uses_identical_pixels_for_both_branches_without_mutation(self):
        captures = []

        class Encoder:
            def posteriors(inner, examples, include_target):
                self.assertFalse(include_target)
                captures.append(examples[0]["image"][0])
                return [{"cond_mean": torch.zeros(2), "cond_logvar": torch.zeros(2)}]

            def qwen_features(inner, examples):
                captures.append(examples[0]["image"][0])
                return torch.zeros(1, 3, 2), torch.ones(1, 3, dtype=torch.bool)

        cosmos = intervene_encoded_observations(Encoder(), [self.example], [self.live], "noise_cosmos", 42, .1)[0]
        qwen = intervene_encoded_observations(Encoder(), [self.example], [self.live], "noise_qwen", 42, .1)[0]
        np.testing.assert_array_equal(captures[0], captures[1])
        self.assertFalse(np.array_equal(captures[0], self.example["image"][0]))
        self.assertIs(cosmos["qwen"], self.live["qwen"])
        self.assertIs(qwen["cond_mean"], self.live["cond_mean"])
        self.assertIs(qwen["state"], self.live["state"])
        self.assertTrue((self.example["image"][0] == 128).all())

    def test_future_images_are_rejected_and_zero_noise_is_identity(self):
        zero = noisy_observations([self.example], 42, 0)[0]
        np.testing.assert_array_equal(zero["image"][0], self.example["image"][0])
        with self.assertRaises(ValueError):
            noisy_observations([{**self.example, "image": self.example["image"] * 2}], 42, .1)

    def test_all_nine_conditions_are_paired_on_one_of_sixteen_devices(self):
        jobs = paired_jobs(0, 2, ATTRIBUTION_CONDITIONS, 16)
        self.assertEqual(sum(map(len, jobs)), 40 * 2 * 9)
        self.assertTrue(all(jobs))
        owners, groups = {}, {}
        for rank, assigned in enumerate(jobs):
            for job in assigned:
                key = job["suite"], job["task"], job["trial"]
                self.assertEqual(owners.setdefault(key, rank), rank)
                groups.setdefault(key, []).append(job["condition"])
        self.assertTrue(all(set(rows) == set(ATTRIBUTION_CONDITIONS) and len(rows) == 9 for rows in groups.values()))
        self.assertEqual(jobs, paired_jobs(0, 2, ATTRIBUTION_CONDITIONS, 16))

    def test_report_rejects_changed_initial_observations_and_protocols(self):
        rows = [{**job, "checkpoint_sha256": "same", "protocol_sha256": "protocol",
                 "episode_seed": job["trial"], "initial_state_sha256": "state",
                 "first_observation_sha256": "image", "success": job["condition"] != "freeze_cosmos"}
                for jobs in paired_jobs(0, 2, ATTRIBUTION_CONDITIONS, 16) for job in jobs]
        result = summarize(rows, bootstrap=100, conditions=ATTRIBUTION_CONDITIONS)
        self.assertEqual(result["branch_comparisons"]["freeze"]["cosmos_minus_qwen_drop"], 1.)
        self.assertEqual(result["trials_per_task"], 2)
        for field in ("protocol_sha256", "initial_state_sha256", "first_observation_sha256"):
            changed = copy.deepcopy(rows)
            changed[0][field] = "different"
            with self.assertRaises(ValueError):
                summarize(changed, bootstrap=100, conditions=ATTRIBUTION_CONDITIONS)

    def test_concurrent_simulators_preserve_same_device_pairing(self):
        jobs = paired_jobs(0, 5, ATTRIBUTION_CONDITIONS, 16, slots_per_device=3)
        serial = paired_jobs(0, 5, ATTRIBUTION_CONDITIONS, 16)
        device_for_pair = {(job["suite"], job["task"], job["trial"]): rank
                           for rank, assigned in enumerate(serial) for job in assigned}
        self.assertEqual(len(jobs), 48)
        self.assertEqual(sum(map(len, jobs)), 40 * 5 * 9)
        owners = {}
        for worker, assigned in enumerate(jobs):
            for job in assigned:
                key = job["suite"], job["task"], job["trial"]
                self.assertEqual(owners.setdefault(key, worker), worker)
                self.assertEqual(worker // 3, device_for_pair[key])


if __name__ == "__main__":
    unittest.main()
