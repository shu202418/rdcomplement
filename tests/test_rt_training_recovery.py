"""Tiny CPU integration tests for crash recovery and skipped gradient updates."""

import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from rt_completion import pipeline
from rt_completion.data import RTSequenceDataset
from rt_completion.storage import close_read_envs
from tests import test_rt_completion as fixture_module


ROOT = Path(__file__).resolve().parents[1]


class SimulatedInterruption(RuntimeError):
    pass


class TrainingRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def tearDown(self):
        close_read_envs()

    def fixture(self, root):
        baseline, config_path, labels, splits = fixture_module.CompletionTests().fixture(root)
        with (ROOT / "configs/rt_completion_s2_v2.yaml").open() as stream:
            config = yaml.safe_load(stream)
        config["data"] = baseline["data"]
        config["export"] = baseline["export"]
        config["model"]["base_channels"] = 2
        config["validation"]["groups"] = ["random"]
        config["train"].update(epochs=1, batch_size=1, amp=False, log_every=1000,
                                checkpoint_every_steps=2, retry_fp32=True,
                                max_nonfinite_batches=20, max_consecutive_nonfinite=3,
                                max_failure_snapshots=3, amp_dtype="auto")
        with config_path.open("w") as stream:
            yaml.safe_dump(config, stream)
        args = argparse.Namespace(config=str(config_path), labels=str(labels),
                                  output=str(root / "run"), device="cpu", resume=None,
                                  epochs=None, batch_size=None, workers=0,
                                  limit_train=None, limit_val=None, max_steps=None)
        return config, args, splits

    @staticmethod
    def load_checkpoint(path):
        return torch.load(path, map_location="cpu", weights_only=True)

    def assert_tree_equal(self, left, right, path="state"):
        if isinstance(left, torch.Tensor):
            self.assertIsInstance(right, torch.Tensor, path)
            self.assertTrue(torch.equal(left, right), path)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right), path)
            for key in left:
                self.assert_tree_equal(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right), path)
            for index, (value, other) in enumerate(zip(left, right)):
                self.assert_tree_equal(value, other, f"{path}[{index}]")
        else:
            self.assertEqual(left, right, path)

    @staticmethod
    def recording_getitem(reads):
        original = RTSequenceDataset.__getitem__

        def getitem(dataset, index):
            if dataset.split == "train":
                reads.append(tuple(dataset.indicators[index]))
            return original(dataset, index)

        return getitem

    def test_intra_epoch_resume_matches_model_optimizer_metrics_and_sample_order(self):
        with tempfile.TemporaryDirectory(prefix="rt-recovery-test-") as temporary:
            root = Path(temporary)
            _, args, splits = self.fixture(root)
            control_args = copy.copy(args)
            control_args.output = str(root / "control")
            control_reads = []
            with patch.object(RTSequenceDataset, "__getitem__", self.recording_getitem(control_reads)):
                pipeline.train(control_args)
            expected = self.load_checkpoint(root / "control" / "last.pt")

            original_save = pipeline.save_checkpoint
            interrupted = False

            def interrupt_after_second_batch(path, payload):
                nonlocal interrupted
                original_save(path, payload)
                if (not interrupted and Path(path).name == "progress.pt"
                        and payload.get("epoch_complete") is False
                        and payload.get("progress", {}).get("next_step") == 2):
                    interrupted = True
                    raise SimulatedInterruption("stop after a durable step checkpoint")

            resumed_reads = []
            with patch.object(RTSequenceDataset, "__getitem__", self.recording_getitem(resumed_reads)), \
                    patch.object(pipeline, "save_checkpoint", interrupt_after_second_batch):
                with self.assertRaisesRegex(SimulatedInterruption, "durable step checkpoint"):
                    pipeline.train(args)
            self.assertTrue(interrupted)
            partial = self.load_checkpoint(root / "run" / "progress.pt")
            self.assertFalse(partial["epoch_complete"])
            self.assertEqual(partial["epoch"], 0)
            self.assertEqual(partial["progress"]["next_step"], 2)
            self.assertEqual(partial["progress"]["successful_updates"], 2)

            args.resume = str(root / "run" / "progress.pt")
            with patch.object(RTSequenceDataset, "__getitem__", self.recording_getitem(resumed_reads)):
                pipeline.train(args)
            actual = self.load_checkpoint(root / "run" / "last.pt")
            self.assertTrue(actual["epoch_complete"])
            self.assertEqual(resumed_reads, control_reads)
            self.assertEqual(len(resumed_reads), len(splits["train"]))
            self.assertEqual(len(set(resumed_reads)), len(splits["train"]))
            for name in ("model", "optimizer", "metrics", "best_score"):
                self.assert_tree_equal(expected[name], actual[name], name)
            reports = (root / "run" / "metrics.jsonl").read_text().splitlines()
            self.assertEqual(len(reports), 1)
            self.assertEqual(json.loads(reports[0])["epoch"], 1)

    def test_legacy_epoch_checkpoint_starts_at_following_epoch(self):
        with tempfile.TemporaryDirectory(prefix="rt-legacy-recovery-test-") as temporary:
            root = Path(temporary)
            _, args, splits = self.fixture(root)
            pipeline.train(args)
            last = root / "run" / "last.pt"
            legacy = self.load_checkpoint(last)
            for name in ("epoch_complete", "progress", "rng_state"):
                legacy.pop(name, None)
            for name in ("checkpoint_every_steps", "max_nonfinite_batches", "max_consecutive_nonfinite",
                         "retry_fp32", "max_failure_snapshots", "amp_dtype"):
                legacy["config"]["train"].pop(name, None)
            torch.save(legacy, last)
            args.resume, args.epochs = str(last), 2
            reads = []
            with patch.object(RTSequenceDataset, "__getitem__", self.recording_getitem(reads)):
                pipeline.train(args)
            actual = self.load_checkpoint(last)
            self.assertEqual(actual["epoch"], 1)
            self.assertTrue(actual["epoch_complete"])
            self.assertEqual(len(reads), len(splits["train"]))
            reports = [json.loads(line) for line in (root / "run" / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([item["epoch"] for item in reports], [1, 2])

    def test_validation_failure_keeps_train_progress_and_resume_does_not_retrain(self):
        with tempfile.TemporaryDirectory(prefix="rt-val-recovery-test-") as temporary:
            root = Path(temporary)
            _, args, splits = self.fixture(root)
            original_epoch = pipeline.run_epoch

            def fail_validation(model, batches, config, device, epoch, optimizer=None, *extra, **kwargs):
                if optimizer is None:
                    raise SimulatedInterruption("validation temporarily unavailable")
                return original_epoch(model, batches, config, device, epoch, optimizer, *extra, **kwargs)

            with patch.object(pipeline, "run_epoch", fail_validation):
                with self.assertRaisesRegex(SimulatedInterruption, "validation temporarily unavailable"):
                    pipeline.train(args)
            partial = self.load_checkpoint(root / "run" / "progress.pt")
            self.assertFalse(partial["epoch_complete"])
            self.assertEqual(partial["progress"]["next_step"], len(splits["train"]))
            with self.assertRaisesRegex(ValueError, "Incomplete training checkpoint"):
                pipeline.export(argparse.Namespace(checkpoint=str(root / "run" / "progress.pt")))
            saved_model = copy.deepcopy(partial["model"])
            saved_optimizer = copy.deepcopy(partial["optimizer"])
            args.resume = str(root / "run" / "progress.pt")
            reads = []
            with patch.object(RTSequenceDataset, "__getitem__", self.recording_getitem(reads)):
                pipeline.train(args)
            self.assertEqual(reads, [])
            actual = self.load_checkpoint(root / "run" / "last.pt")
            self.assert_tree_equal(saved_model, actual["model"], "model")
            self.assert_tree_equal(saved_optimizer, actual["optimizer"], "optimizer")
            self.assertTrue(actual["epoch_complete"])
            self.assertEqual(self.load_checkpoint(root / "run" / "progress.pt")["epoch_complete"], True)

    def test_one_nonfinite_gradient_batch_is_skipped_without_optimizer_pollution(self):
        with tempfile.TemporaryDirectory(prefix="rt-nonfinite-recovery-test-") as temporary:
            root = Path(temporary)
            config, args, splits = self.fixture(root)
            config["train"]["retry_fp32"] = False
            with Path(args.config).open("w") as stream:
                yaml.safe_dump(config, stream)
            original_model = pipeline.RTCompletionNet
            injected = False
            observed_progress = []
            original_save = pipeline.save_checkpoint

            def model_with_bad_gradient(*values, **kwargs):
                model = original_model(*values, **kwargs)

                def corrupt_once(gradient):
                    nonlocal injected
                    if not injected:
                        injected = True
                        return torch.full_like(gradient, float("inf"))
                    return gradient

                next(model.parameters()).register_hook(corrupt_once)
                return model

            def capture_progress(path, payload):
                if Path(path).name == "progress.pt" and not payload.get("epoch_complete"):
                    observed_progress.append(copy.deepcopy(payload["progress"]))
                return original_save(path, payload)

            with patch.object(pipeline, "RTCompletionNet", model_with_bad_gradient), \
                    patch.object(pipeline, "save_checkpoint", capture_progress):
                pipeline.train(args)
            self.assertTrue(injected)
            self.assertTrue(observed_progress)
            self.assertEqual(observed_progress[-1]["skipped_batches"], 1)
            self.assertEqual(observed_progress[-1]["successful_updates"], len(splits["train"]) - 1)
            actual = self.load_checkpoint(root / "run" / "last.pt")
            self.assertTrue(all(torch.isfinite(value).all() for value in actual["model"].values()))
            steps = [int(state["step"]) for state in actual["optimizer"]["state"].values()]
            self.assertTrue(steps)
            self.assertEqual(set(steps), {len(splits["train"]) - 1})
            for state in actual["optimizer"]["state"].values():
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        self.assertTrue(torch.isfinite(value).all())

    def test_persistent_bad_gradients_stop_with_safe_checkpoint_not_best_model(self):
        with tempfile.TemporaryDirectory(prefix="rt-persistent-nonfinite-") as temporary:
            root = Path(temporary)
            config, args, _ = self.fixture(root)
            config["train"]["retry_fp32"] = False
            with Path(args.config).open("w") as stream:
                yaml.safe_dump(config, stream)
            original_model = pipeline.RTCompletionNet
            def always_bad(*values, **kwargs):
                model = original_model(*values, **kwargs)
                next(model.parameters()).register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
                return model
            with patch.object(pipeline, "RTCompletionNet", always_bad):
                with self.assertRaisesRegex(FloatingPointError, "Too many non-finite batches"):
                    pipeline.train(args)
            partial = self.load_checkpoint(root / "run" / "progress.pt")
            self.assertEqual(partial["progress"]["next_step"], 3)
            self.assertEqual(partial["progress"]["successful_updates"], 0)
            self.assertEqual(partial["progress"]["skipped_batches"], 3)
            self.assertEqual(partial["optimizer"]["state"], {})
            self.assertFalse((root / "run" / "best.pt").exists())
            self.assertTrue(all(torch.isfinite(value).all() for value in partial["model"].values()))
            with (root / "run" / "numerics_events.jsonl").open() as stream:
                self.assertEqual(len([json.loads(line) for line in stream]), 3)

    def test_short_all_skipped_epoch_cannot_select_a_best_model(self):
        with tempfile.TemporaryDirectory(prefix="rt-empty-safe-epoch-") as temporary:
            root = Path(temporary)
            _, args, _ = self.fixture(root)
            args.max_steps = 1
            original_model = pipeline.RTCompletionNet
            def always_bad(*values, **kwargs):
                model = original_model(*values, **kwargs)
                next(model.parameters()).register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
                return model
            with patch.object(pipeline, "RTCompletionNet", always_bad):
                with self.assertRaisesRegex(FloatingPointError, "No successful training batches"):
                    pipeline.train(args)
            partial = self.load_checkpoint(root / "run" / "progress.pt")
            self.assertEqual(partial["progress"]["next_step"], 1)
            self.assertEqual(partial["progress"]["skipped_batches"], 1)
            self.assertFalse((root / "run" / "best.pt").exists())


if __name__ == "__main__":
    unittest.main()
