"""Batch repartition must preserve samples, schedule, and strict resume guards."""
import copy
from pathlib import Path
import unittest

import numpy as np
from omegaconf import OmegaConf

from DiT4DiT.fusion.common import seed_for
from DiT4DiT.fusion.data import MixtureBatches
from DiT4DiT.fusion.resume import validate_resume_config


class RepartitionTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[2] / "configs/qwen_concat_ppu16.yaml"
        self.saved = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        self.current = copy.deepcopy(self.saved)
        self.current["datasets"]["vla_data"]["per_device_batch_size"] = 16
        self.current["trainer"]["gradient_accumulation_steps"] = 1

    def test_change_requires_explicit_flag_and_preserves_global_batch(self):
        with self.assertRaisesRegex(ValueError, "Resume configuration changed"):
            validate_resume_config(self.saved, self.current)
        self.assertTrue(validate_resume_config(self.saved, self.current, True)["changed"])
        self.current["trainer"]["gradient_accumulation_steps"] = 2
        with self.assertRaisesRegex(ValueError, "effective batch"):
            validate_resume_config(self.saved, self.current, True)

    def test_flag_cannot_change_model_schedule_data_or_seed(self):
        mutations = [("trainer", "max_train_steps", 80000),
                     ("trainer", "gradient_checkpointing", False), ("seed", None, 123),
                     ("framework", "name", "other")]
        for section, key, value in mutations:
            cfg = copy.deepcopy(self.current)
            if key is None:
                cfg[section] = value
            else:
                cfg[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_resume_config(self.saved, cfg, True)
        cfg = copy.deepcopy(self.current)
        cfg["datasets"]["vla_data"]["data_mix"] = "subset"
        with self.assertRaises(ValueError):
            validate_resume_config(self.saved, cfg, True)

    def test_legacy_samples_and_resumed_updates_match_on_every_rank(self):
        ranges = [(0, 101), (101, 301), (301, 607), (607, 1000)]
        for rank in range(16):
            # Independent copy of the original sampler, including RNG draw order.
            original = []
            for micro in range(2000, 2020):
                rng = np.random.default_rng(seed_for(42, rank, micro, "data"))
                original.extend(int(rng.integers(*ranges[i])) for i in rng.integers(4, size=8))
            resumed = list(MixtureBatches(ranges, 16, 1010, 42, rank, 1000, chunk_size=8))
            self.assertEqual(original, [index for batch in resumed for index in batch])
            resumed_again = list(MixtureBatches(ranges, 16, 1010, 42, rank, 1005, chunk_size=8))
            self.assertEqual(resumed[5:], resumed_again)
            split = list(MixtureBatches(ranges, 4, 4040, 42, rank, 4000, chunk_size=8))
            self.assertEqual(original, [index for batch in split for index in batch])

    def test_same_config_needs_no_exception(self):
        self.assertFalse(validate_resume_config(self.current, self.current)["changed"])


if __name__ == "__main__":
    unittest.main()
