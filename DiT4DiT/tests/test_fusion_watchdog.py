"""No real cloud calls: eviction retry and uncertain-response reconciliation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch


path = Path(__file__).resolve().parents[2] / "dlc/watch_joint_job.py"
spec = importlib.util.spec_from_file_location("joint_watchdog", path)
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


class WatchdogTests(unittest.TestCase):
    def prepare(self, directory):
        root = Path(directory)
        (root / "config256.yaml").write_text(f"run_root_dir: {directory}\nrun_id: test\n")
        for name, data in {
            "training_job.json": {"JobId": "old"},
            "training_request.json": {"DisplayName": "experiment", "UserCommand": "python train.py"},
            "auto_restart_policy.json": {"expires_unix": time.time()+600, "max_restarts": 6},
            "source_sha256.json": {},
        }.items():
            (root / name).write_text(json.dumps(data))
        return root

    def test_uncertain_create_is_reconciled_without_duplicate_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.prepare(tmp)
            calls, list_count = [], 0
            def fake_api(operation, *args):
                nonlocal list_count
                calls.append(operation)
                if operation == "GetJob":
                    return {"Status": "Failed", "ReasonCode": "JobEvicted"}
                if operation == "ListJobs":
                    list_count += 1
                    return [] if list_count == 1 else [{"JobId": "new", "DisplayName": "experiment_retry01", "Status": "Running"}]
                if operation == "CreateJob":
                    raise subprocess.TimeoutExpired("CreateJob", 120)
                self.fail(operation)
            with patch.object(watchdog.sys, "argv", ["watch", "--bundle", str(root)]), \
                 patch.object(watchdog, "api", side_effect=fake_api), \
                 patch.object(watchdog.time, "sleep", side_effect=[None, RuntimeError("end test")]), \
                 patch("DiT4DiT.fusion.checkpoints.recovery_manifest", return_value={"step": 500}):
                with self.assertRaisesRegex(RuntimeError, "end test"):
                    watchdog.main()
            self.assertEqual(calls.count("CreateJob"), 1)
            self.assertEqual(json.loads((root / "active_job.json").read_text())["JobId"], "new")

    def test_user_stop_and_model_failures_never_restart(self):
        for job in ({"Status": "Stopped"}, {"Status": "Failed", "ReasonMessage": "CUDA out of memory"}):
            with tempfile.TemporaryDirectory() as tmp:
                root = self.prepare(tmp)
                with patch.object(watchdog.sys, "argv", ["watch", "--bundle", str(root)]), \
                     patch.object(watchdog, "api", return_value=job) as api:
                    watchdog.main()
                self.assertEqual([c.args[0] for c in api.call_args_list], ["GetJob"])


if __name__ == "__main__":
    unittest.main()
