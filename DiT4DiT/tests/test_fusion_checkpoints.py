import json
from pathlib import Path
import tempfile
import unittest

from DiT4DiT.fusion.checkpoints import publish_recovery, prune_recovery, recovery_manifest, reconcile_metrics
from DiT4DiT.fusion.common import atomic_json


class CheckpointRecoveryTests(unittest.TestCase):
    def make_state(self, root, step, world=2):
        state = root / "training_state" / f"step_{step:06d}"
        state.mkdir(parents=True)
        names = ["mp_rank_00_model_states.pt"]
        names += [f"rng_{rank}.pt" for rank in range(world)]
        names += [f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt" for rank in range(world)]
        for name in names:
            (state / name).write_bytes(b"placeholder")
        return state

    def test_partial_save_cannot_replace_last_complete_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_state(root, 100)
            publish_recovery(root, 100, 2)
            state = self.make_state(root, 500)
            (state / "rng_1.pt").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                publish_recovery(root, 500, 2)
            self.assertEqual(recovery_manifest(root)["step"], 100)

    def test_rolling_recovery_does_not_publish_inference_or_count_partial_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self.make_state(root, 5000)
            publish_recovery(root, 5000, 2)
            weight = root / "steps_5000.pt"
            weight.write_bytes(b"milestone")
            atomic_json(root / "latest.json", {"step": 5000, "checkpoint": str(weight), "training_state": str(first)})
            for step in (5500, 6000):
                self.make_state(root, step)
                publish_recovery(root, step, 2)
            self.make_state(root, 6500)  # interrupted before publication
            prune_recovery(root)
            self.assertFalse(first.exists())
            self.assertEqual(len(list((root / "training_state").glob("*/complete.json"))), 2)
            self.assertEqual(recovery_manifest(root)["step"], 6000)
            self.assertTrue(weight.exists())
            self.assertEqual(json.loads((root / "latest.json").read_text())["step"], 5000)
            self.assertNotIn("training_state", json.loads((root / "latest.json").read_text()))

    def test_replay_archives_unsaved_updates_and_partial_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metrics.jsonl").write_text('{"step": 1}\n{"step": 2}\n{"step":')
            reconcile_metrics(root, 1)
            self.assertEqual((root / "metrics.jsonl").read_text(), '{"step": 1}\n')
            self.assertEqual(len(list(root.glob("unsaved_metrics*.jsonl"))), 1)


if __name__ == "__main__":
    unittest.main()
