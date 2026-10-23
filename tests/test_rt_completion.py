"""CPU tests; no pytest, CUDA, real dataset or SMPL-X model files needed."""

import argparse
import gzip
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import lmdb
import msgpack
import numpy as np
import torch
import yaml

from dataset.lmdb_utils import pack_dict_np, pack_ndarray
from rt_completion.data import RTSequenceDataset, causal_window
from rt_completion.geometry import temporal_target, warp_by_root
from rt_completion.losses import completion_loss, corrupt_rt
from rt_completion.model import RTCompletionNet
from rt_completion.pipeline import export, init_manifest, train, write_m4human_config
from rt_completion.storage import (CompletionCache, SparseWriter, apply_additions,
                                   atomic_json, close_read_envs, frame_key, read_json,
                                   source_identity)


ROOT = Path(__file__).resolve().parents[1]


class CompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def tearDown(self):
        close_read_envs()

    def config(self):
        with (ROOT / "configs/rt_completion_s2.yaml").open() as stream:
            config = yaml.safe_load(stream)
        config["data"].update(shape=[9, 11, 9], bounds=[[-1., 1.]] * 3)
        config["model"]["base_channels"] = 2
        config["train"].update(epochs=1, batch_size=1, amp=False)
        config["export"].update(probability_threshold=0., body_threshold=0.,
                                min_intensity=0., max_sigma_log=100., max_added_voxels=7)
        return config

    def test_window_causal_gap_and_sequence(self):
        allowed = {(1, 1, 0), (1, 1, 1), (1, 1, 3), (2, 1, 2), (1, 2, 2)}
        self.assertEqual(causal_window((1, 1, 1), allowed, 4), [(1, 1, 0)] * 3 + [(1, 1, 1)])
        self.assertEqual(causal_window((1, 1, 3), allowed, 4), [(1, 1, 3)] * 4)
        with self.assertRaises(KeyError):
            causal_window((1, 1, 2), allowed, 4)

    def test_warp_axis_order_and_direction(self):
        for axis in range(3):
            volume = torch.zeros(1, 1, 9, 9, 9)
            volume[0, 0, 4, 4, 4] = 1.
            current = torch.zeros(1, 3)
            current[0, axis] = .25
            result = warp_by_root(volume, torch.zeros(1, 3), current, [[-1, 1]] * 3)
            expected = [4, 4, 4]
            expected[axis] += 1
            self.assertAlmostEqual(float(result[(0, 0, *expected)]), 1.)
            self.assertAlmostEqual(float(result.sum()), 1.)

    def test_padded_frames_do_not_supply_temporal_evidence(self):
        raw = torch.zeros(1, 4, 9, 9, 9)
        raw[:, :3, 4, 4, 4] = 1
        joints = torch.zeros(1, 4, 22, 3)
        _, empty = temporal_target(raw, joints, torch.zeros(1, 4), [[-1, 1]] * 3, .3, 2)
        self.assertFalse(empty.any())
        target, supported = temporal_target(raw, joints, torch.tensor([[0., 1., 1., 0.]]), [[-1, 1]] * 3, .3, 2)
        self.assertTrue(supported[0, 0, 4, 4, 4])
        self.assertAlmostEqual(float(target[0, 0, 4, 4, 4]), 1.)

    def test_forward_mask_loss_and_backward(self):
        config = self.config()
        model = RTCompletionNet(config["data"], config["model"])
        raw = torch.zeros(1, 4, 9, 11, 9)
        raw[:, :, 4, 5, 4] = 1
        raw[:, :, 5, 5, 4] = .01  # weak observation must also be treated as real
        mask_config = {**config["mask"], "drop_probability": 1.}
        corrupted, mask = corrupt_rt(raw, mask_config, torch.Generator().manual_seed(1))
        self.assertEqual(mask.sum().item(), 2)
        result = model(corrupted)
        self.assertEqual(result["intensity"].shape, (1, 1, 9, 11, 9))
        self.assertEqual(result["joints"].shape, (1, 22, 3))
        joints = torch.zeros(1, 4, 22, 3)
        loss, metrics = completion_loss(result, raw, mask, joints, torch.tensor([[1., 1., 1., 0.]]), model, config["loss"])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(metrics["masked_voxels"]), 2)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_merge_preserves_all_original_returns(self):
        raw = np.zeros((9, 9, 9), np.float32)
        raw[4, 4, 4] = 1e-7
        original = raw.copy()
        record = {"indices": np.array([0, 2], np.int32), "values": np.array([.5, .25], np.float32)}
        completed = apply_additions(raw, record)
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(completed[raw != 0], raw[raw != 0])
        self.assertEqual(completed.flat[0], .5)
        self.assertEqual(np.count_nonzero(completed), 3)
        with self.assertRaises(ValueError):
            apply_additions(raw, {"indices": np.flatnonzero(raw), "values": np.array([9.])})

    def fixture(self, root):
        config = self.config()
        source = root / "rf3dpose_all"
        source.mkdir()
        config["data"]["cache_path"] = str(source)
        split = {"train": [[1, 1, i] for i in range(5)],
                 "val": [[2, 1, i] for i in range(3)], "test": [[3, 1, i] for i in range(3)]}
        with gzip.open(source / "indeces.pkl.gz", "wb") as stream:
            pickle.dump({"p1": {"s2": split}}, stream)
        env = lmdb.open(str(source / "radar_comp.lmdb"), subdir=False, map_size=2**20)
        with env.begin(write=True) as txn:
            for values in split.values():
                for key in values:
                    txn.put(frame_key(key), pack_ndarray(np.array([[4, 5, 4, 1.], [5, 5, 4, .01]], np.float32)))
        env.close()
        labels = root / "labels"
        labels.mkdir()
        signature = {"data": config["data"], "source": source_identity(source)}
        init_manifest(labels, "rt_joint_labels", signature, False)
        writer = SparseWriter(labels / "joints.lmdb", map_size_gb=.01)
        writer.put_many([(key, {"joints": np.zeros((22, 3), np.float32)})
                         for name in ("train", "val") for key in split[name]])
        writer.close()
        config_path = root / "config.yaml"
        with config_path.open("w") as stream:
            yaml.safe_dump(config, stream)
        return config, config_path, labels, split

    def test_original_dataset_uses_each_temporal_frames_additions(self):
        # Exercise the real RF3DPoseDataset.__getitem__ integration. Only the
        # unrelated mesh/calibration work is mocked; LMDB and RT merge are real.
        from dataset.m4human_dataset import RF3DPoseDataset
        with tempfile.TemporaryDirectory(prefix="rt-dataset-test-") as temporary:
            root = Path(temporary)
            config, _, _, splits = self.fixture(root)
            data = config["data"]
            keys = sum(splits.values(), [])
            payloads = {
                "radar_pc": lambda key: pack_ndarray(np.ones((2, 4), np.float32)),
                "params": lambda key: pack_dict_np({"gender": 0, "trans": np.zeros(3)}),
                "calib": lambda key: pack_dict_np({"radar_to_cam_rotmatrix": np.eye(3),
                                                    "radar_to_cam_tvec": np.zeros(3)}),
                "indicator": lambda key: msgpack.packb(key),
            }
            for name, payload in payloads.items():
                env = lmdb.open(str(Path(data["cache_path"]) / f"{name}.lmdb"), subdir=False, map_size=2**20)
                with env.begin(write=True) as txn:
                    for key in keys:
                        txn.put(frame_key(key), payload(key))
                env.close()
            cache_path = root / "completed"
            cache_path.mkdir()
            manifest = init_manifest(cache_path, "rt_additions", {
                "source": source_identity(data["cache_path"]), "data": data,
                "geometry": "predicted_from_rt_only",
            }, False)
            writer = SparseWriter(cache_path / "additions.lmdb", .01)
            writer.put_many([(key, {"indices": np.array([0], np.int32),
                                     "values": np.array([.25 + key[2]], np.float32)}) for key in keys])
            writer.close()
            manifest.update(state="complete", partitions={s: {"complete": True} for s in splits})
            atomic_json(cache_path / "manifest.json", manifest)
            def dummy_models(dataset, path):
                dataset.smplx_models = {}
            module = "dataset.m4human_dataset."
            with patch.object(RF3DPoseDataset, "_load_smplx_model", dummy_models), \
                    patch(module + "calibrate_param_to_radar", side_effect=lambda p, c: p), \
                    patch(module + "generate_vertices_in_radar_space", return_value=np.zeros((22, 3))), \
                    patch(module + "compute_joints_from_vertices", return_value=np.zeros((22, 3))), \
                    patch(module + "aabb_from_smplx_params", return_value={}), \
                    patch(module + "project_3d_to_2d", return_value=np.zeros((22, 2))):
                dataset = RF3DPoseDataset([], cache_dir=str(root), split="train", radar_shape=data["shape"],
                                          main_modality="rawImage_XYZ", scale_id="p1", split_id="s2",
                                          rt_completion_cache=str(cache_path))
                sample = dataset[4]
                self.assertEqual(sample["rawImage_XYZ"].shape, (4, 9, 11, 9))
                np.testing.assert_array_equal(sample["rawImage_XYZ"][:, 0, 0, 0], [1.25, 2.25, 3.25, 4.25])
                np.testing.assert_array_equal(sample["rawImage_XYZ"][:, 4, 5, 4], [1.] * 4)
                dataset.rt_completion = None
                baseline = dataset[4]
                np.testing.assert_array_equal(baseline["rawImage_XYZ"][:, 0, 0, 0], [0.] * 4)
                for env in dataset.lmdb_envs.values():
                    env.close()
            close_read_envs()

    def test_offline_pipeline_train_export_resume_and_reload(self):
        with tempfile.TemporaryDirectory(prefix="rt-completion-test-") as temporary:
            root = Path(temporary)
            config, config_path, labels, split = self.fixture(root)
            dataset = RTSequenceDataset(config["data"], "train", labels)
            self.assertEqual(dataset[0]["history_valid"].sum().item(), 0)
            self.assertEqual(dataset[4]["history_valid"].sum().item(), 3)
            train_args = argparse.Namespace(config=str(config_path), labels=str(labels), output=str(root / "run"),
                                            device="cpu", resume=None, epochs=1, batch_size=1, workers=0,
                                            limit_train=3, limit_val=2, max_steps=1)
            train(train_args)
            checkpoint = root / "run" / "best.pt"
            self.assertTrue(checkpoint.exists())
            export_args = argparse.Namespace(checkpoint=str(checkpoint), output=str(root / "completed"),
                                             device="cpu", splits=["train", "val", "test"], limit=1,
                                             batch_size=2, workers=0, probability_threshold=None,
                                             max_added_voxels=None, resume=False)
            # The fixture deliberately has NO params/calib LMDB or SMPL-X model.
            # Successful export proves the deployment path is radar-only.
            export(export_args)
            self.assertEqual(read_json(root / "completed" / "manifest.json")["state"], "partial")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                CompletionCache(root / "completed", config["data"]["cache_path"], "p1", "s2", "train", (9, 11, 9))
            export_args.resume = True
            export_args.limit = None
            export(export_args)
            export(export_args)  # idempotent resume; no duplicate writes
            manifest = read_json(root / "completed" / "manifest.json")
            self.assertEqual(manifest["state"], "complete")
            reader = CompletionCache(root / "completed", config["data"]["cache_path"], "p1", "s2", "train", (9, 11, 9))
            for key in split["train"]:
                raw = dataset._radar(key)
                record = reader.get(key)
                completed = reader.apply(raw, key)
                self.assertEqual(len(record["indices"]), 7)
                np.testing.assert_array_equal(raw[raw != 0], completed[raw != 0])
            with self.assertRaisesRegex(ValueError, "mismatch"):
                CompletionCache(root / "completed", config["data"]["cache_path"], "p1", "s1", "train", (9, 11, 9))
            # Closing reader handles is required before opening a same-process writer.
            close_read_envs()
            export_args.probability_threshold = .5
            with self.assertRaisesRegex(ValueError, "signature mismatch"):
                export(export_args)
            base = root / "base.yaml"
            with base.open("w") as stream:
                yaml.safe_dump({"paths": {}, "dataset": {}, "model": {}, "train": {}, "eval": {}}, stream)
            make_args = argparse.Namespace(base=str(base), cache=str(root / "completed"), output=str(root / "downstream.yaml"))
            write_m4human_config(make_args)
            with open(make_args.output) as stream:
                downstream = yaml.safe_load(stream)
            self.assertEqual(downstream["paths"]["rt_completion_cache"], str(root / "completed"))
            self.assertEqual(downstream["dataset"]["split"], "s2")
            with self.assertRaises(FileExistsError):
                write_m4human_config(make_args)
            train_args.resume = str(root / "run" / "last.pt")
            train_args.epochs = 2
            train(train_args)
            self.assertEqual(torch.load(train_args.resume, weights_only=True)["epoch"], 1)
            close_read_envs()


if __name__ == "__main__":
    unittest.main()
