"""Teacher caching integration must preserve losses, validation and recovery."""

import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from rt_completion import pipeline
from rt_completion.geometry import temporal_target_v2
from rt_completion.losses import completion_loss
from rt_completion.storage import close_read_envs
from rt_completion.teacher_cache import build_teacher_cache
from tests import test_rt_completion as fixture_module
from tests import test_rt_mask_loss_v2 as mask_fixture
from tests import test_rt_pipeline_v2 as pipeline_fixture


class SpeedPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def tearDown(self):
        close_read_envs()

    def test_explicit_cached_teacher_matches_online_loss_and_gradients(self):
        raw, masked, joints, history, output, config, pseudo = mask_fixture.MaskLossV2Tests().loss_fixture()
        cached_output = {key: value.detach().clone().requires_grad_(True) for key, value in output.items()}
        with patch("rt_completion.geometry.temporal_target_v2", return_value=pseudo) as builder:
            loss, metrics = completion_loss(output, raw, masked, joints, history, mask_fixture.DummyModel(), config)
            cached_loss, cached_metrics = completion_loss(cached_output, raw, masked, joints, history,
                                                          mask_fixture.DummyModel(), config, temporal_teacher=pseudo)
            self.assertEqual(builder.call_count, 1)
        torch.testing.assert_close(loss, cached_loss, rtol=0, atol=0)
        self.assertEqual(metrics.keys(), cached_metrics.keys())
        for key in metrics:
            torch.testing.assert_close(metrics[key], cached_metrics[key], rtol=0, atol=0)
        loss.backward()
        cached_loss.backward()
        for key in output:
            torch.testing.assert_close(output[key].grad, cached_output[key].grad, rtol=0, atol=0)
        self.assertIsNone(pseudo["intensity"].grad)
        self.assertIsNone(pseudo["confidence"].grad)
        broken = dict(pseudo, intensity=torch.zeros(1, 1, 2, 2, 2))
        with self.assertRaisesRegex(ValueError, "shape/key"):
            completion_loss(cached_output, raw, masked, joints, history, mask_fixture.DummyModel(), config,
                            temporal_teacher=broken)

    def test_online_validation_builds_teacher_once_per_batch_not_per_group(self):
        with tempfile.TemporaryDirectory(prefix="rt-online-teacher-test-") as temporary:
            root = Path(temporary)
            _, config_path, labels, _ = fixture_module.CompletionTests.fixture(
                pipeline_fixture.ReliabilityPipelineTests(), root)
            args = argparse.Namespace(config=str(config_path), labels=str(labels), output=str(root / "run"),
                                      device="cpu", resume=None, epochs=1, batch_size=1, workers=0,
                                      limit_train=1, limit_val=1, max_steps=1)
            with patch("rt_completion.geometry.temporal_target_v2", wraps=temporal_target_v2) as builder, \
                    redirect_stdout(io.StringIO()):
                pipeline.train(args)
                self.assertEqual(builder.call_count, 2)  # one train + one shared four-group val
            checkpoint = torch.load(root / "run" / "last.pt", weights_only=True)
            self.assertEqual(checkpoint["metrics"]["train"]["diagnostics_scope"], "disabled")
            for group in checkpoint["metrics"]["val"]["groups"].values():
                self.assertEqual(group["diagnostics_scope"], "full")
                self.assertIn("risk_coverage", group)

    def test_add_cache_to_existing_progress_and_keep_model_optimizer_and_val(self):
        with tempfile.TemporaryDirectory(prefix="rt-cached-resume-test-") as temporary:
            root = Path(temporary)
            config, config_path, labels, _ = fixture_module.CompletionTests.fixture(
                pipeline_fixture.ReliabilityPipelineTests(), root)
            args = argparse.Namespace(config=str(config_path), labels=str(labels), output=str(root / "run"),
                                      device="cpu", resume=None, epochs=1, batch_size=1, workers=0,
                                      limit_train=None, limit_val=None, max_steps=None)
            save_checkpoint = pipeline.save_checkpoint

            def stop_after_first_batch(path, payload):
                save_checkpoint(path, payload)
                if Path(path).name == "progress.pt" and payload.get("progress", {}).get("next_step") == 1:
                    raise InterruptedError("simulated interruption")

            with redirect_stdout(io.StringIO()):
                with patch.object(pipeline, "save_checkpoint", side_effect=stop_after_first_batch):
                    with self.assertRaisesRegex(InterruptedError, "simulated"):
                        pipeline.train(args)
                progress = root / "run" / "progress.pt"
                # Resume once without caching to obtain a same-checkpoint control.
                original_progress = torch.load(progress, weights_only=True)
                args.resume = str(progress)
                pipeline.train(args)
                expected = torch.load(root / "run" / "last.pt", weights_only=True)
                # Restore only the temporary fixture's checkpoint. Real experiments are untouched.
                torch.save(original_progress, progress)
                cache = root / "teacher"
                build_teacher_cache(config, labels, cache, batch_size=2)
                args.teacher_cache = str(cache)
                with patch("rt_completion.geometry.temporal_target_v2", side_effect=AssertionError("cache miss")):
                    pipeline.train(args)
                actual = torch.load(root / "run" / "last.pt", weights_only=True)
            for key in expected["model"]:
                torch.testing.assert_close(expected["model"][key], actual["model"][key], rtol=0, atol=0)
            for index in expected["optimizer"]["state"]:
                for key in expected["optimizer"]["state"][index]:
                    torch.testing.assert_close(expected["optimizer"]["state"][index][key],
                                               actual["optimizer"]["state"][index][key], rtol=0, atol=0)
            self.assertEqual(expected["metrics"], actual["metrics"])
            self.assertEqual(expected["best_score"], actual["best_score"])


if __name__ == "__main__":
    unittest.main()
