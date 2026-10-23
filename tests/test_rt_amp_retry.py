"""CPU BF16 integration coverage for safe, unscaled FP32 batch retries."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import torch
import yaml

from rt_completion import pipeline
from rt_completion.model import RTCompletionNet
from rt_completion.numerics import make_grad_scaler, precision_policy


ROOT = Path(__file__).resolve().parents[1]


class ToyCompletion(torch.nn.Module):
    """Keep the real autocast/optimizer path while isolating failure modes."""
    def __init__(self):
        super().__init__()
        self.project = torch.nn.Linear(4, 1)
        self.before_forward = None

    def forward(self, raw, history_valid=None):
        if self.before_forward is not None:
            self.before_forward()
        return {"value": self.project(raw.mean(dim=(2, 3, 4)))}


class AmpRetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def inputs(self):
        raw = torch.zeros(1, 4, 9, 11, 9)
        raw[:, :, 4, 5, 4] = 1.
        raw[:, :, 5, 5, 4] = .3
        raw[:, :, 4, 4, 4] = .2
        masked = torch.zeros_like(raw[:, -1:], dtype=torch.bool)
        masked[:, :, 4, 5, 4] = True
        corrupted = raw.clone()
        corrupted[:, -1, 4, 5, 4] = 0.
        joints = torch.zeros(1, 4, 22, 3)
        joints[..., 0] = torch.linspace(-.2, .2, 22)
        return corrupted, raw, masked, joints, torch.ones(1, 4), torch.tensor([[1, 1, 7]])

    def config(self):
        with (ROOT / "configs/rt_completion_s2_v2.yaml").open() as stream:
            config = yaml.safe_load(stream)
        config["data"].update(shape=[9, 11, 9], bounds=[[-1., 1.]] * 3)
        config["model"]["base_channels"] = 2
        config["train"].update(epochs=1, amp=True, amp_dtype="bfloat16",
                                retry_fp32=True, max_failure_snapshots=0)
        return config

    def prepare(self, model, config):
        policy = precision_policy(config, "cpu")
        self.assertTrue(policy["enabled"])
        self.assertEqual(policy["dtype"], torch.bfloat16)
        model._training_scaler = make_grad_scaler(policy, "cpu")
        self.assertFalse(model._training_scaler.is_enabled())
        return torch.optim.AdamW(model.parameters(), lr=.001), policy

    def run_batch(self, model, optimizer, config, policy, diagnostic_dir):
        with torch.backends.mkldnn.flags(enabled=False), redirect_stdout(io.StringIO()):
            return pipeline._training_batch(model, optimizer, *self.inputs(), config, policy,
                                            epoch=0, step=0, diagnostic_dir=diagnostic_dir)

    def assert_single_update(self, optimizer):
        steps = [int(state["step"]) for state in optimizer.state.values()]
        self.assertTrue(steps)
        self.assertEqual(set(steps), {1})
        for state in optimizer.state.values():
            self.assertTrue(all(torch.isfinite(value).all() for value in state.values()
                                if isinstance(value, torch.Tensor)))

    @staticmethod
    def read_event(directory):
        def invalid_constant(value):
            raise AssertionError(f"Non-standard JSON number: {value}")
        lines = (Path(directory) / "numerics_events.jsonl").read_text().splitlines()
        assert len(lines) == 1
        return json.loads(lines[0], parse_constant=invalid_constant)

    def test_real_v2_bfloat16_bad_gradient_retries_fp32_exactly_once(self):
        torch.manual_seed(117)
        config = self.config()
        model = RTCompletionNet(config["data"], config["model"])
        optimizer, policy = self.prepare(model, config)
        backward_calls = 0

        def corrupt_first_gradient(gradient):
            nonlocal backward_calls
            backward_calls += 1
            return torch.full_like(gradient, float("inf")) if backward_calls == 1 else gradient

        handle = next(model.parameters()).register_hook(corrupt_first_gradient)
        with tempfile.TemporaryDirectory(prefix="rt-bf16-retry-") as temporary:
            with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                prediction, loss, metrics, status, recovered = self.run_batch(
                    model, optimizer, config, policy, temporary)
            self.assertEqual(backward_calls, 2)
            self.assertEqual(step.call_count, 1)
            self.assertTrue(recovered)
            self.assertTrue(status["updated"])
            self.assertEqual(status["scale_before"], 1.)
            self.assertEqual(status["scale_after"], 1.)
            self.assertEqual(model._training_scaler.get_scale(), 1.)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(pipeline.finite_prediction(prediction))
            self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))
            self.assert_single_update(optimizer)
            event = self.read_event(temporary)
            self.assertEqual(event["initial_update"]["reason"], "nonfinite_gradient")
            self.assertEqual(event["action"], "recovered_fp32")
            self.assertTrue(event["retry_update"]["updated"])
        handle.remove()

    def test_nonfinite_forward_releases_initial_graph_and_retries_without_scaling(self):
        torch.manual_seed(118)
        config = self.config()
        model = ToyCompletion()
        optimizer, policy = self.prepare(model, config)
        initial_prediction = None
        calls = 0
        precision_seen = []

        def check_old_graph_released():
            if calls == 1:
                self.assertIsNotNone(initial_prediction)
                self.assertIsNone(initial_prediction(), "Initial AMP graph retained during FP32 retry")

        def controlled_loss(prediction, *unused):
            nonlocal calls, initial_prediction
            calls += 1
            value = prediction["value"]
            precision_seen.append(value.dtype)
            if calls == 1:
                initial_prediction = weakref.ref(value)
                loss = value.square().mean() * float("inf")
                return loss, {"loss": loss.detach(), "diagnostic_nan": value.new_tensor(float("nan"))}
            loss = value.float().square().mean()
            return loss, {"loss": loss.detach()}

        model.before_forward = check_old_graph_released
        with tempfile.TemporaryDirectory(prefix="rt-forward-retry-") as temporary, \
                patch.object(pipeline, "completion_loss", new=controlled_loss), \
                patch.object(optimizer, "step", wraps=optimizer.step) as step:
            prediction, loss, _, status, recovered = self.run_batch(
                model, optimizer, config, policy, temporary)
            self.assertEqual(calls, 2)
            self.assertEqual(precision_seen, [torch.bfloat16, torch.float32])
            self.assertEqual(step.call_count, 1)
            self.assertTrue(recovered)
            self.assertTrue(status["updated"])
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(pipeline.finite_prediction(prediction))
            self.assert_single_update(optimizer)
            event = self.read_event(temporary)
            self.assertEqual(event["initial_update"]["reason"], "nonfinite_loss")
            self.assertIsNone(event["metrics"]["diagnostic_nan"])
            self.assertEqual(event["action"], "recovered_fp32")

    def test_both_backward_attempts_bad_leave_weights_and_adam_state_unchanged(self):
        torch.manual_seed(119)
        config = self.config()
        model = ToyCompletion()
        optimizer, policy = self.prepare(model, config)
        original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        backward_calls = 0

        def always_bad_gradient(gradient):
            nonlocal backward_calls
            backward_calls += 1
            return torch.full_like(gradient, float("nan"))

        def finite_loss(prediction, *unused):
            loss = prediction["value"].float().square().mean()
            return loss, {"loss": loss.detach()}

        handle = next(model.parameters()).register_hook(always_bad_gradient)
        with tempfile.TemporaryDirectory(prefix="rt-unrecoverable-") as temporary, \
                patch.object(pipeline, "completion_loss", new=finite_loss), \
                patch.object(optimizer, "step", wraps=optimizer.step) as step:
            prediction, loss, metrics, status, recovered = self.run_batch(
                model, optimizer, config, policy, temporary)
            self.assertEqual(backward_calls, 2)
            self.assertEqual(step.call_count, 0)
            self.assertFalse(recovered)
            self.assertFalse(status["updated"])
            self.assertIsNone(prediction)
            self.assertFalse(loss.requires_grad)
            self.assertTrue(all(not value.requires_grad for value in metrics.values()))
            self.assertFalse(optimizer.state)
            for name, parameter in model.named_parameters():
                self.assertTrue(torch.equal(parameter, original[name]), name)
                self.assertIsNone(parameter.grad, name)
            event = self.read_event(temporary)
            self.assertEqual(event["action"], "skipped_batch")
            self.assertEqual(event["retry_update"]["reason"], "nonfinite_gradient")
        handle.remove()


if __name__ == "__main__":
    unittest.main()
