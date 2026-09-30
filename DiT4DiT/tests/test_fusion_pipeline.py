"""Exercise OOM fallback and restart without launching accelerator jobs."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf


WORKSPACE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("concat_pipeline", WORKSPACE / "run_concat_pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class PipelineTests(unittest.TestCase):
    def exercise(self, failure):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = OmegaConf.load(WORKSPACE / "configs/qwen_concat_ppu16.yaml")
            cfg.run_root_dir = str(root / "results")
            config = root / "requested.yaml"
            OmegaConf.save(cfg, config)
            if failure == "repartition":
                state_dir = root / "results/test_run_pipeline"
                state_dir.mkdir(parents=True)
                (state_dir / "batch_repartition.json").write_text(json.dumps({"batch_size": 16, "accumulation": 1}))
                (state_dir / "state.json").write_text(json.dumps({"run_id": "test_run", "completed": [],
                                                                 "train_first_elapsed_seconds": 3600}))
                training = root / "results/test_run"
                training.mkdir()
                (training / "latest.json").write_text(json.dumps({"checkpoint": "saved.pt"}))
            seen = []
            fail_once = failure == "error"

            def command(args, **kwargs):
                nonlocal fail_once
                if "DiT4DiT.fusion.train" in args and "--profile" in args:
                    batch = int(next(x.split("=", 1)[1] for x in args if x.startswith("datasets.vla_data.per_device_batch_size=")))
                    run = next(x.split("=", 1)[1] for x in args if x.startswith("run_id="))
                    seen.append(run)
                    if fail_once:
                        fail_once = False
                        kwargs["stdout"].write("invalid data shape\n")
                        return SimpleNamespace(returncode=1)
                    if batch == 4 or (failure == "fallback" and batch in (8, 16)):
                        kwargs["stdout"].write("torch.OutOfMemoryError: PPU out of memory\n")
                        return SimpleNamespace(returncode=1)
                    path = root / "results" / run
                    path.mkdir(parents=True)
                    (path / "profile.json").write_text(json.dumps({"batch_size": batch, "accumulation": 16//batch,
                        "samples_per_second": batch*10, "max_reserved_gib": 90 if batch == 16 else 70,
                        "budget_steps": 2000}))
                elif "DiT4DiT.fusion.train" in args:
                    if failure == "repartition":
                        self.assertIn("--resume", args)
                        self.assertIn("--allow-batch-repartition", args)
                        self.assertEqual(args[args.index("--pause-after-hours") + 1], "17.0")
                        selected_cfg = OmegaConf.load(args[args.index("--config") + 1])
                        self.assertEqual(selected_cfg.datasets.vla_data.per_device_batch_size, 16)
                        self.assertEqual(selected_cfg.trainer.gradient_accumulation_steps, 1)
                    path = root / "results/test_run"
                    path.mkdir(parents=True, exist_ok=True)
                    (path / "latest.json").write_text(json.dumps({"checkpoint": str(path / "checkpoint.pt")}))
                    (path / "complete.json").write_text("{}")
                return SimpleNamespace(returncode=0)

            cwd = Path.cwd()
            try:
                with patch.object(pipeline, "ROOT", root), patch.object(pipeline.subprocess, "run", side_effect=command), \
                     patch("sys.argv", ["pipeline", "--run-id", "test_run", "--config", str(config)] +
                           (["--train-now"] if failure in ("direct", "repartition") else [])):
                    if failure == "error":
                        with self.assertRaises(RuntimeError):
                            pipeline.main()
                        state = json.loads((root / "results/test_run_pipeline/state.json").read_text())
                        self.assertEqual(state["status"], "failed")
                    pipeline.main()
                    state = json.loads((root / "results/test_run_pipeline/state.json").read_text())
                    selected = json.loads((root / "results/test_run_pipeline/selection.json").read_text())["selected"]
                    self.assertEqual(state["status"], "complete")
                    self.assertEqual(selected["batch_size"], 2 if failure == "fallback" else 16 if failure == "repartition" else 8)
                    before = list(seen)
                    if failure in ("direct", "repartition"):
                        self.assertEqual(seen, [])
                        pipeline.sys.argv.remove("--train-now")
                    pipeline.main()
                    self.assertEqual(seen, before, "Completed stages must not execute twice")
                    if failure == "error":
                        self.assertIn("test_run_profile_b4_attempt2", seen)
            finally:
                os.chdir(cwd)

    def test_skip_oom_and_reject_over_budget_candidate(self):
        self.exercise("oom")

    def test_direct_training_and_restart_skip_profiles(self):
        self.exercise("direct")

    def test_repartition_persists_on_restart_with_remaining_time_budget(self):
        self.exercise("repartition")

    def test_fallback_preserves_global_batch(self):
        self.exercise("fallback")

    def test_non_oom_failure_stops_and_restart_uses_new_attempt(self):
        self.exercise("error")


if __name__ == "__main__":
    unittest.main()
