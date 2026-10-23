"""CPU coverage for safe, exact intra-epoch progress state."""

import io
import random
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rt_completion.evaluation import GroupMetrics, HeldoutDiagnostics
from rt_completion.progress import (DeterministicBatchSampler, capture_rng_state,
                                    group_metrics_state, restore_group_metrics_state,
                                    restore_rng_state)


class RecordingDataset(Dataset):
    def __init__(self, length):
        self.length, self.reads = length, []

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self.reads.append(index)
        return index


class ProgressTests(unittest.TestCase):
    def test_resumed_order_matches_uninterrupted_and_keeps_global_rng(self):
        original_rng = torch.get_rng_state().clone()
        full = list(DeterministicBatchSampler(23, 4, 717))
        before = list(DeterministicBatchSampler(23, 4, 717))[:3]
        remainder = list(DeterministicBatchSampler(23, 4, 717, start_step=3))
        self.assertEqual(before + remainder, full)
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
        self.assertEqual(len(DeterministicBatchSampler(23, 4, 717, 3)), 3)
        self.assertEqual(len(full[-1]), 3)
        self.assertEqual(list(DeterministicBatchSampler(23, 4, 717, 6)), [])
        self.assertEqual(list(DeterministicBatchSampler(0, 4, 717)), [])
        self.assertNotEqual(full, list(DeterministicBatchSampler(23, 4, 718)))
        dropped = list(DeterministicBatchSampler(23, 4, 717, drop_last=True))
        self.assertEqual(dropped, full[:-1])

    def test_resume_does_not_read_skipped_dataset_items(self):
        dataset = RecordingDataset(31)
        full = list(DeterministicBatchSampler(len(dataset), 3, 29))
        loader = DataLoader(dataset, batch_sampler=DeterministicBatchSampler(31, 3, 29, 4),
                            generator=torch.Generator().manual_seed(9), num_workers=0)
        actual = [batch.tolist() for batch in loader]
        self.assertEqual(actual, full[4:])
        self.assertEqual(dataset.reads, [item for batch in full[4:] for item in batch])
        self.assertFalse(set(dataset.reads) & set(item for batch in full[:4] for item in batch))

    def test_invalid_sampler_settings(self):
        for arguments in ((-1, 2, 1), (2, 0, 1), (2, 1, 1, -1), (2, 1, 1, 3)):
            with self.assertRaises(ValueError):
                DeterministicBatchSampler(*arguments)

    def test_rng_round_trip_is_weights_only_safe(self):
        initial = capture_rng_state()
        try:
            random.seed(11)
            np.random.seed(22)
            np.random.normal()  # Preserve the cached Gaussian as well as MT keys.
            torch.manual_seed(33)
            mask_generator = torch.Generator().manual_seed(44)
            snapshot = capture_rng_state(mask_generator)
            buffer = io.BytesIO()
            torch.save(snapshot, buffer)
            buffer.seek(0)
            loaded = torch.load(buffer, weights_only=True)

            def draw():
                return (random.random(), np.random.normal(size=7), torch.rand(7),
                        torch.rand(7, generator=mask_generator))

            expected = draw()
            draw()
            restore_rng_state(loaded, mask_generator)
            actual = draw()
            self.assertEqual(expected[0], actual[0])
            np.testing.assert_array_equal(expected[1], actual[1])
            self.assertTrue(torch.equal(expected[2], actual[2]))
            self.assertTrue(torch.equal(expected[3], actual[3]))
        finally:
            restore_rng_state(initial)

    @staticmethod
    def populate(metrics):
        metrics.samples += 2
        metrics.totals["loss"] = metrics.totals.get("loss", 0.) + 1.75
        error = torch.tensor([[[.1, .5, .8]], [[.01, .2, .4]]])
        sigma = torch.tensor([[[.2, .4, 1.]], [[.02, .3, .6]]])
        masked = torch.tensor([[[True, True, False]], [[True, False, True]]])
        accepted = torch.tensor([[[True, False, False]], [[True, False, False]]])
        metrics.heldout.update(error, sigma, masked, accepted, error * 1.5)
        part = metrics.parts.setdefault("6", HeldoutDiagnostics())
        part.update(error[:1], sigma[:1], masked[:1], accepted[:1], error[:1] * 1.5)

    def test_streaming_metrics_round_trip_and_continued_updates(self):
        original = GroupMetrics()
        self.populate(original)
        buffer = io.BytesIO()
        torch.save(group_metrics_state(original), buffer)
        buffer.seek(0)
        restored = restore_group_metrics_state(torch.load(buffer, weights_only=True))
        self.assertEqual(original.summary(), restored.summary())
        self.populate(original)
        self.populate(restored)
        self.assertEqual(original.summary(), restored.summary())
        self.assertEqual(original.samples, restored.samples)
        np.testing.assert_array_equal(original.heldout.bin_count, restored.heldout.bin_count)
        # Restored state owns its arrays, rather than aliasing a checkpoint tensor.
        state = group_metrics_state(original)
        restored = restore_group_metrics_state(state)
        restored.heldout.bin_count[0] += 1
        self.assertNotEqual(restored.heldout.bin_count[0], state["heldout"]["bin_count"][0])


if __name__ == "__main__":
    unittest.main()
