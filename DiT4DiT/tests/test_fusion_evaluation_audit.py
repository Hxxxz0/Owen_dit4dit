import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from DiT4DiT.fusion.common import file_sha
from DiT4DiT.fusion.evaluation_audit import audit_round
from DiT4DiT.fusion.interventions import ATTRIBUTION_CONDITIONS
from DiT4DiT.fusion.protocol import SUITES, seed_for


class SavedEvaluationAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "new"
        reference = Path(self.directory.name) / "reference"
        (self.root / "episodes").mkdir(parents=True)
        (reference / "episodes").mkdir(parents=True)
        self.protocol = {"reference_evaluation": str(reference),
                         "checkpoint_sha256": "new-checkpoint", "conditions": ATTRIBUTION_CONDITIONS}
        (self.root / "protocol.json").write_text(json.dumps(self.protocol))
        for suite in SUITES:
            for task in range(10):
                identity = {"initial_state_sha256": f"state-{suite}-{task}",
                            "first_observation_sha256": f"image-state-{suite}-{task}",
                            "episode_seed": seed_for(suite, task, 0, "episode") % (2**31-1)}
                (reference / "episodes" / f"{suite}_{task:02d}_00_full.json").write_text(json.dumps(identity))
                for condition in ATTRIBUTION_CONDITIONS:
                    path = self.root / "episodes" / f"{suite}_{task:02d}_00_{condition}.json"
                    path.write_text(json.dumps({**identity, "steps": 8, "success": True,
                        "checkpoint_sha256": "new-checkpoint", "protocol_sha256": file_sha(self.root / "protocol.json")}))
                    np.save(path.with_suffix(".actions.npy"), np.zeros((8, 7)))
                    path.with_suffix(".inputs.jsonl").write_text(json.dumps({"chunk": 0,
                        "action_seed": seed_for(suite, task, 0, 0, "action"),
                        "feature_seed": seed_for(suite, task, 0, 0, "features"),
                        "current_image_sha256": "pixels", "initial_image_sha256": "pixels"}) + "\n")

    def test_complete_paired_round_passes(self):
        result = audit_round(self.root, 1, self.protocol)
        self.assertTrue(result["passed"])
        self.assertEqual(result["episodes"], 360)

    def test_freeze_first_action_mismatch_is_rejected(self):
        path = self.root / "episodes" / f"{SUITES[0]}_00_00_freeze_qwen.actions.npy"
        actions = np.load(path)
        actions[0, 0] = 1
        np.save(path, actions)
        with self.assertRaisesRegex(ValueError, "First action chunk differs"):
            audit_round(self.root, 1, self.protocol)

    def test_changed_historical_initial_state_is_rejected(self):
        path = self.root / "episodes" / f"{SUITES[0]}_00_00_full.json"
        row = json.loads(path.read_text())
        row["initial_state_sha256"] = "different"
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "Pairing differs"):
            audit_round(self.root, 1, self.protocol)

    def test_wrong_action_seed_is_rejected(self):
        path = self.root / "episodes" / f"{SUITES[0]}_00_00_full.inputs.jsonl"
        row = json.loads(path.read_text())
        row["action_seed"] += 1
        path.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "Incorrect paired random seeds"):
            audit_round(self.root, 1, self.protocol)


if __name__ == "__main__":
    unittest.main()
