"""Fixed-mask grouping and the v2 offline train/evaluate/export contract."""

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml

from tests import test_rt_completion as fixture_module
from rt_completion.data import RTSequenceDataset
from rt_completion.evaluation import MASK_GROUPS, fixed_corruption
from rt_completion.pipeline import evaluate, export, load_config, train
from rt_completion.storage import CompletionCache, close_read_envs, read_json


ROOT = Path(__file__).resolve().parents[1]


class ReliabilityPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def tearDown(self):
        close_read_envs()

    def config(self):
        original = fixture_module.CompletionTests.config(self)
        with (ROOT / "configs/rt_completion_s2_v2.yaml").open() as stream:
            config = yaml.safe_load(stream)
        config["data"] = original["data"]
        config["model"]["base_channels"] = 2
        config["train"].update(epochs=1, batch_size=1, amp=False, log_every=1)
        config["export"] = original["export"]
        return config

    def test_config_reuses_same_label_data_signature(self):
        baseline = load_config(ROOT / "configs/rt_completion_s2.yaml")
        improved = load_config(ROOT / "configs/rt_completion_s2_v2.yaml")
        self.assertEqual(baseline["data"], improved["data"])
        self.assertEqual(improved["model"]["architecture"], "reliability_v2")

    def test_fixed_masks_independent_of_batching_and_keep_original(self):
        config = self.config()
        raw = torch.zeros(3, 4, 9, 11, 9)
        raw[:, :, 4, 5, 4] = 1
        raw[:, :, 5, 5, 4] = .01
        original = raw.clone()
        joints = torch.zeros(3, 4, 22, 3)
        joints[..., 0] = torch.linspace(-.5, .5, 22)
        indicators = torch.tensor([[1, 1, 7], [1, 1, 8], [2, 1, 9]])
        for group in MASK_GROUPS:
            together = fixed_corruption(raw, joints, indicators, config["mask"],
                                         config["data"]["bounds"], 100042, group)
            separately = [fixed_corruption(raw[i:i+1], joints[i:i+1], indicators[i:i+1],
                                            config["mask"], config["data"]["bounds"], 100042, group)
                          for i in range(3)]
            self.assertTrue(torch.equal(together[0], torch.cat([value[0] for value in separately])), group)
            self.assertTrue(torch.equal(together[1], torch.cat([value[1] for value in separately])), group)
            self.assertFalse((together[1] & (raw[:, -1:] == 0)).any())
            self.assertEqual(together[2]["kind"], [group] * 3)
        self.assertTrue(torch.equal(raw, original))

    def test_ineligible_body_group_not_replaced_with_random(self):
        config = self.config()
        raw = torch.ones(1, 4, 9, 11, 9)
        joints = torch.full((1, 4, 22, 3), 50.)
        for group in ("single_part", "continuous_part"):
            corrupted, masked, info = fixed_corruption(raw, joints, torch.tensor([[1, 1, 4]]),
                                                       config["mask"], config["data"]["bounds"], 1, group)
            self.assertFalse(masked.any())
            self.assertFalse(info["eligible"].any())
            self.assertTrue(torch.equal(corrupted, raw))
            self.assertEqual(info["kind"], [group])

    def test_v2_train_evaluate_export_resume_and_radar_only_reload(self):
        with tempfile.TemporaryDirectory(prefix="rt-v2-pipeline-") as temporary:
            root = Path(temporary)
            config, config_path, labels, splits = fixture_module.CompletionTests.fixture(self, root)
            args = argparse.Namespace(config=str(config_path), labels=str(labels), output=str(root / "run"),
                                      device="cpu", resume=None, epochs=1, batch_size=1, workers=0,
                                      limit_train=3, limit_val=2, max_steps=1)
            progress = io.StringIO()
            with redirect_stdout(progress):
                train(args)
            self.assertIn("Epoch [1/1] started", progress.getvalue())
            self.assertIn("Epoch [1/1] train started", progress.getvalue())
            self.assertIn("Epoch [1/1] val started", progress.getvalue())
            self.assertIn("Epoch [1/1] train step 1:", progress.getvalue())
            self.assertIn("Epoch [1/1] val grouped step 1:", progress.getvalue())
            self.assertIn("Epoch [1/1] completed", progress.getvalue())
            checkpoint_path = root / "run" / "best.pt"
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            self.assertTrue(any(key.startswith("v2.") for key in checkpoint["model"]))
            self.assertEqual(set(checkpoint["metrics"]["val"]["groups"]), set(MASK_GROUPS))
            self.assertEqual(checkpoint["metrics"]["val"]["selection_aggregation"],
                             "macro_over_nonempty_mask_groups")
            evaluation = argparse.Namespace(checkpoint=str(checkpoint_path), labels=str(labels),
                                            output=str(root / "evaluation.json"), device="cpu",
                                            groups=None, limit=2, max_steps=None, batch_size=2, workers=0)
            report = evaluate(evaluation)
            self.assertEqual(report["split"], "val")
            self.assertEqual(set(report["metrics"]["groups"]), set(MASK_GROUPS))
            for group in report["metrics"]["groups"].values():
                self.assertIn("risk_coverage", group)
                self.assertIn("accepted_heldout_coverage", group)
                self.assertEqual(group["frames"], 2)
            with self.assertRaises(FileExistsError):
                evaluate(evaluation)
            # No params/calib LMDB or SMPL-X model exists in this fixture.
            export_args = argparse.Namespace(checkpoint=str(checkpoint_path), output=str(root / "completed"),
                                             device="cpu", splits=["train", "val", "test"], limit=None,
                                             batch_size=2, workers=0, probability_threshold=None,
                                             max_added_voxels=None, resume=False)
            export(export_args)
            self.assertEqual(read_json(root / "completed" / "manifest.json")["state"], "complete")
            reader = CompletionCache(root / "completed", config["data"]["cache_path"],
                                      "p1", "s2", "test", (9, 11, 9))
            dataset = RTSequenceDataset(config["data"], "test")
            for key in splits["test"]:
                raw = dataset._radar(key)
                completed = reader.apply(raw, key)
                np.testing.assert_array_equal(completed[raw != 0], raw[raw != 0])
            close_read_envs()
            args.resume = str(root / "run" / "last.pt")
            args.epochs = 2
            resumed_progress = io.StringIO()
            with redirect_stdout(resumed_progress):
                train(args)
            self.assertIn("Epoch [2/2] started", resumed_progress.getvalue())
            self.assertIn("Epoch [2/2] completed", resumed_progress.getvalue())
            self.assertNotIn("Epoch [1/2] started", resumed_progress.getvalue())
            with (root / "run" / "metrics.jsonl").open() as stream:
                reports = [json.loads(line) for line in stream]
            self.assertEqual([(report["epoch"], report["total_epochs"]) for report in reports],
                             [(1, 1), (2, 2)])
            self.assertEqual(torch.load(args.resume, weights_only=True)["epoch"], 1)


if __name__ == "__main__":
    unittest.main()
