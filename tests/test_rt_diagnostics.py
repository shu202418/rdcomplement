"""Safe bounded artifacts for non-finite RT training batches."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from rt_completion.diagnostics import tensor_statistics, write_numerics_event


class NumericsDiagnosticsTests(unittest.TestCase):
    def test_nonfinite_metrics_and_tensor_ranges_are_strict_json(self):
        with tempfile.TemporaryDirectory(prefix="rt-numerics-") as temporary:
            event = {"epoch": 1, "step": 3, "reason": "bad_gradients", "scale": float("inf"),
                     "metrics": {"loss": torch.tensor(float("nan")), "finite": 2.5},
                     "bad_parameters": ["encoder.weight"]}
            raw = torch.tensor([0., 1., float("nan"), float("inf")])
            path = write_numerics_event(temporary, event, {"rt": raw}, max_snapshots=0)
            line = path.read_text().strip()
            self.assertNotIn("NaN", line)
            self.assertNotIn("Infinity", line)
            record = json.loads(line)
            self.assertIsNone(record["scale"])
            self.assertIsNone(record["metrics"]["loss"])
            self.assertEqual(record["metrics"]["finite"], 2.5)
            self.assertEqual(record["tensor_stats"]["raw"]["finite_count"], 2)
            self.assertEqual(record["tensor_stats"]["raw"]["max"], 1.)
            self.assertEqual(record["snapshot_skipped"], "snapshot_limit")
            self.assertTrue(torch.isnan(raw[2]))

    def test_snapshots_are_bounded_never_overwrite_and_weights_only_safe(self):
        with tempfile.TemporaryDirectory(prefix="rt-numerics-") as temporary:
            root = Path(temporary)
            event = {"epoch": 2, "step": 100, "reason": "nonfinite_gradient", "amp_dtype": "bfloat16"}
            batch = {"rt": torch.ones(2, 4, 3, 3, 3), "corrupted": torch.zeros(2, 4, 3, 3, 3),
                     "masked": torch.ones(2, 1, 3, 3, 3, dtype=torch.bool),
                     "joints": torch.zeros(2, 4, 22, 3), "history_valid": torch.ones(2, 4, dtype=torch.bool),
                     "indicator": torch.tensor([[1, 2, 3], [1, 2, 4]]),
                     "dense_mesh": torch.zeros(100, 3)}
            first_path = write_numerics_event(root, event, batch, max_snapshots=2)
            first_record = json.loads(first_path.read_text().splitlines()[0])
            first_snapshot = root / first_record["snapshot"]
            initial_bytes = first_snapshot.read_bytes()
            write_numerics_event(root, event, batch, max_snapshots=2)
            write_numerics_event(root, event, batch, max_snapshots=2)
            records = [json.loads(line) for line in first_path.read_text().splitlines()]
            self.assertEqual(len(records), 3)
            self.assertEqual(len({record["event_id"] for record in records}), 3)
            self.assertNotEqual(records[0]["snapshot"], records[1]["snapshot"])
            self.assertEqual(len(list((root / "debug_failures").glob("failure_*.pt"))), 2)
            self.assertEqual(first_snapshot.read_bytes(), initial_bytes)
            self.assertEqual(records[2]["snapshot_skipped"], "snapshot_limit")
            loaded = torch.load(first_snapshot, weights_only=True)
            self.assertEqual(set(loaded), {"event", "batch"})
            self.assertNotIn("dense_mesh", loaded["batch"])
            self.assertTrue(torch.equal(loaded["batch"]["raw"], batch["rt"]))
            self.assertEqual(loaded["batch"]["raw"].device.type, "cpu")
            self.assertTrue(loaded["batch"]["mask"].dtype == torch.bool)

    def test_no_batch_and_oversize_batch_still_record_events(self):
        with tempfile.TemporaryDirectory(prefix="rt-numerics-") as temporary:
            event = {"epoch": 1, "step": 1}
            path = write_numerics_event(temporary, event)
            write_numerics_event(temporary, event, {"raw": torch.ones(10)}, max_snapshot_bytes=1)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(records[0]["snapshot_skipped"], "no_batch")
            self.assertEqual(records[1]["snapshot_skipped"], "byte_limit")
            self.assertEqual(records[1]["tensor_stats"]["raw"]["numel"], 10)

    def test_empty_and_all_nonfinite_tensor_statistics(self):
        self.assertEqual(tensor_statistics(torch.empty(0))["finite_count"], 0)
        stats = tensor_statistics(torch.tensor([float("nan"), float("-inf")]))
        self.assertEqual(stats["nonfinite_count"], 2)
        self.assertIsNone(stats["min"])
        self.assertIsNone(stats["mean"])
        self.assertEqual(tensor_statistics(torch.tensor([True, False]))["max"], 1.)


if __name__ == "__main__":
    unittest.main()
