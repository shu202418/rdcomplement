"""Lossless sparse temporal-teacher cache and resume tests, CPU only."""

import copy
import gzip
from pathlib import Path
import pickle
import socket
import tempfile
import unittest
from unittest.mock import patch

import lmdb
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.lmdb_utils import pack_ndarray, unpack_dict_np
from rt_completion.data import RTSequenceDataset
from rt_completion.geometry import temporal_target_v2
from rt_completion.storage import (SparseWriter, atomic_json, close_read_envs, frame_key,
                                   read_env, read_json, source_identity)
from rt_completion.teacher_cache import (TeacherCache, TeacherDataset, build_teacher_cache,
                                        decode_teacher, encode_teacher, teacher_signature)


class TeacherCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def tearDown(self):
        close_read_envs()

    def teacher(self):
        teacher = {"intensity": torch.zeros(2, 1, 3, 4, 5),
                   "valid_mask": torch.zeros(2, 1, 3, 4, 5, dtype=torch.bool),
                   "confidence": torch.zeros(2, 1, 3, 4, 5),
                   "support_count": torch.zeros(2, 1, 3, 4, 5, dtype=torch.int64)}
        for batch in range(2):
            teacher["valid_mask"][batch, 0, 1, 2, 3] = True
            teacher["intensity"][batch, 0, 1, 2, 3] = 1.234567 + batch
            teacher["confidence"][batch, 0, 1, 2, 3] = .876543
            teacher["support_count"][batch, 0, 1, 2, 3] = 3
            teacher["support_count"][batch, 0, 0, 0, 0] = 1  # outside valid_mask
        return teacher

    def assert_teacher_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for name in actual:
            self.assertTrue(torch.equal(actual[name], expected[name]), name)
            self.assertEqual(actual[name].dtype, expected[name].dtype, name)
            self.assertFalse(actual[name].requires_grad)

    def test_codec_lossless_all_fields_including_invalid_support_counts(self):
        teacher = self.teacher()
        for batch in range(2):
            record = encode_teacher(teacher, batch)
            self.assertEqual(record["indices"].dtype, np.uint32)
            self.assertEqual(record["intensity"].dtype, np.float32)
            self.assertEqual(record["confidence"].dtype, np.float32)
            self.assertEqual(record["support_count"].dtype, np.uint8)
            decoded = decode_teacher(record, (3, 4, 5))
            self.assert_teacher_equal(decoded, {name: value[batch] for name, value in teacher.items()})

    def test_empty_targets_nonempty_counts_and_large_count_roundtrip(self):
        teacher = self.teacher()
        teacher["valid_mask"].zero_()
        teacher["intensity"].zero_()
        teacher["confidence"].zero_()
        teacher["support_count"][0, 0, 0, 0, 0] = 300
        record = encode_teacher(teacher)
        self.assertEqual(record["indices"].size, 0)
        self.assertEqual(record["support_count"].dtype, np.uint16)
        self.assert_teacher_equal(decode_teacher(record, (3, 4, 5)),
                                  {name: value[0] for name, value in teacher.items()})
        for value in teacher.values():
            value.zero_()
        self.assert_teacher_equal(decode_teacher(encode_teacher(teacher), (3, 4, 5)),
                                  {name: value[0] for name, value in teacher.items()})

    def test_codec_rejects_malformed_or_quantized_targets(self):
        cases = []
        teacher = self.teacher()
        teacher["intensity"] = teacher["intensity"].half()
        cases.append(teacher)
        teacher = self.teacher()
        teacher["confidence"][0, 0, 1, 2, 3] = float("nan")
        cases.append(teacher)
        teacher = self.teacher()
        teacher["intensity"][0, 0, 0, 0, 0] = 1  # would be silently lost
        cases.append(teacher)
        teacher = self.teacher()
        teacher["support_count"][0, 0, 1, 2, 3] = -1
        cases.append(teacher)
        for teacher in cases:
            with self.assertRaises(ValueError):
                encode_teacher(teacher)
        record = encode_teacher(self.teacher())
        for name, value in (("indices", np.array([999], np.uint32)),
                            ("indices", np.array([1, 1], np.uint32)),
                            ("support_count", np.array([0, 3], np.uint8)),
                            ("confidence", np.array([2.], np.float32))):
            malformed = {**record, name: value}
            with self.assertRaises(ValueError):
                decode_teacher(malformed, (3, 4, 5))
        with self.assertRaises(ValueError):
            decode_teacher(record, (3, 4, 6))

    def fixture(self, root):
        source = root / "source"
        source.mkdir()
        config = {
            "data": {"cache_path": str(source), "scale": "p1", "protocol": "s2",
                     "shape": [9, 9, 9], "bounds": [[-1., 1.]] * 3,
                     "window": 4, "intensity_scale": 1., "intensity_cap": 10.},
            "model": {"architecture": "reliability_v2", "capsule_radius_m": .3},
            "loss": {"temporal_alignment": "articulated", "temporal_weight": .1,
                     "temporal_min_support": 2},
        }
        splits = {"train": [[1, 1, index] for index in range(5)],
                  "val": [[2, 1, index] for index in range(4)],
                  "test": [[3, 1, index] for index in range(2)]}
        with gzip.open(source / "indeces.pkl.gz", "wb") as stream:
            pickle.dump({"p1": {"s2": splits}}, stream)
        env = lmdb.open(str(source / "radar_comp.lmdb"), subdir=False, map_size=2**20)
        with env.begin(write=True) as txn:
            for values in splits.values():
                for key in values:
                    points = [[4, 4, 4, 1. + .1 * key[2]]] if key[2] < 3 else [[5, 4, 4, 2.]]
                    txn.put(frame_key(key), pack_ndarray(np.asarray(points, np.float32)))
        env.close()
        labels = root / "labels"
        labels.mkdir()
        joints = np.zeros((22, 3), np.float32)
        joints[:, 0] = np.linspace(-.7, .7, 22)
        writer = SparseWriter(labels / "joints.lmdb", map_size_gb=.01)
        writer.put_many([(key, {"joints": joints})
                         for split in ("train", "val") for key in splits[split]])
        writer.close()
        atomic_json(labels / "manifest.json", {
            "format_version": 1, "kind": "rt_joint_labels", "state": "complete",
            "signature": {"source": source_identity(source), "data": config["data"]},
            "partitions": {split: {"complete": True} for split in ("train", "val")},
        })
        return config, labels, splits

    def test_full_build_reader_worker_wrapper_and_online_target_equivalence(self):
        with tempfile.TemporaryDirectory(prefix="rt-teacher-") as temporary:
            root = Path(temporary)
            config, labels, splits = self.fixture(root)
            output = root / "teacher"
            manifest = build_teacher_cache(config, labels, output, batch_size=2, map_size_gb=.01)
            self.assertEqual(manifest["state"], "complete")
            self.assertEqual(manifest["storage"]["record_count"], 9)
            self.assertEqual(manifest["storage"]["lmdb_size_bytes"], (output / "teacher.lmdb").stat().st_size)
            self.assertEqual(read_json(output / "manifest.json"), manifest)
            for split in ("train", "val"):
                cache = TeacherCache(output, config, labels, split)
                dataset = RTSequenceDataset(config["data"], split, labels)
                wrapped = TeacherDataset(dataset, cache)
                for index in range(len(dataset)):
                    original = dataset[index]
                    online = temporal_target_v2(original["rt"][None], original["joints"][None],
                                                original["history_valid"][None], config["data"]["bounds"],
                                                .3, config["loss"])
                    stored = wrapped[index]["teacher"]
                    self.assert_teacher_equal(stored, {name: value[0] for name, value in online.items()})
                # Pickling does not serialize any open LMDB environment.
                restored = pickle.loads(pickle.dumps(wrapped))
                batch = next(iter(DataLoader(restored, batch_size=2, num_workers=0)))
                self.assertEqual(batch["teacher"]["intensity"].shape, (2, 1, 9, 9, 9))
                self.assertEqual(len(restored.indicators), len(splits[split]))
            with self.assertRaisesRegex(ValueError, "test"):
                TeacherCache(output, config, labels, "test")
            with self.assertRaisesRegex(KeyError, "no silent fallback"):
                cache.get((9, 9, 9))

    def test_limit_always_partial_resume_skips_existing_and_can_finish(self):
        with tempfile.TemporaryDirectory(prefix="rt-teacher-resume-") as temporary:
            root = Path(temporary)
            config, labels, _ = self.fixture(root)
            output = root / "teacher"
            partial = build_teacher_cache(config, labels, output, limit=1, map_size_gb=.01)
            self.assertEqual(partial["state"], "partial")
            self.assertEqual(partial["storage"]["record_count"], 2)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                TeacherCache(output, config, labels, "train")
            with self.assertRaises(FileExistsError):
                build_teacher_cache(config, labels, output, map_size_gb=.01)
            before = TeacherCache(output, config, labels, "train", require_complete=False).get((1, 1, 0))
            # An already-open lazy reader must not block opening the writer.
            with patch("rt_completion.geometry.temporal_target_v2", wraps=temporal_target_v2) as teacher:
                full = build_teacher_cache(config, labels, output, resume=True, batch_size=2, map_size_gb=.01)
            self.assertEqual(teacher.call_count, 4)  # 4 train pending + 3 val pending
            self.assertEqual(full["state"], "complete")
            self.assertEqual(full["storage"]["record_count"], 9)
            self.assert_teacher_equal(TeacherCache(output, config, labels, "train").get((1, 1, 0)), before)

    def test_debug_limit_covering_every_frame_still_rejected(self):
        with tempfile.TemporaryDirectory(prefix="rt-teacher-debug-") as temporary:
            root = Path(temporary)
            config, labels, _ = self.fixture(root)
            output = root / "teacher"
            manifest = build_teacher_cache(config, labels, output, limit=100, map_size_gb=.01)
            self.assertTrue(all(part["complete"] for part in manifest["partitions"].values()))
            self.assertEqual(manifest["state"], "partial")
            with self.assertRaises(ValueError):
                TeacherCache(output, config, labels, "train")

    def test_failed_build_records_partial_and_resume_preserves_committed_batches(self):
        with tempfile.TemporaryDirectory(prefix="rt-teacher-interrupt-") as temporary:
            root = Path(temporary)
            config, labels, _ = self.fixture(root)
            output = root / "teacher"
            calls = 0

            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated interruption")
                return temporal_target_v2(*args, **kwargs)

            with patch("rt_completion.geometry.temporal_target_v2", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    build_teacher_cache(config, labels, output, batch_size=2, map_size_gb=.01)
            partial = read_json(output / "manifest.json")
            self.assertEqual(partial["state"], "partial")
            self.assertEqual(partial["storage"]["record_count"], 2)
            self.assertEqual(partial["partitions"]["train"]["written"], 2)
            with patch("rt_completion.geometry.temporal_target_v2", wraps=temporal_target_v2) as teacher:
                full = build_teacher_cache(config, labels, output, resume=True, batch_size=2, map_size_gb=.01)
            self.assertEqual(teacher.call_count, 4)
            self.assertEqual(full["state"], "complete")

    def test_reader_dataset_supports_spawn_worker(self):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except PermissionError:
            self.skipTest("Sandbox prohibits the local IPC sockets required by DataLoader workers")
        with tempfile.TemporaryDirectory(prefix="rt-teacher-worker-") as temporary:
            root = Path(temporary)
            config, labels, _ = self.fixture(root)
            output = root / "teacher"
            build_teacher_cache(config, labels, output, map_size_gb=.01)
            dataset = TeacherDataset(RTSequenceDataset(config["data"], "train", labels),
                                     TeacherCache(output, config, labels, "train"))
            batches = DataLoader(dataset, batch_size=2, num_workers=1,
                                 multiprocessing_context="spawn", timeout=30)
            observed = 0
            for batch in batches:
                observed += batch["teacher"]["intensity"].shape[0]
                self.assertFalse(batch["teacher"]["intensity"].is_cuda)
            self.assertEqual(observed, 5)

    def test_wrong_provenance_rejected_by_read_and_resume(self):
        with tempfile.TemporaryDirectory(prefix="rt-teacher-signature-") as temporary:
            root = Path(temporary)
            config, labels, _ = self.fixture(root)
            output = root / "teacher"
            build_teacher_cache(config, labels, output, map_size_gb=.01)
            changed = copy.deepcopy(config)
            changed["loss"]["temporal_min_support"] = 1
            with self.assertRaisesRegex(ValueError, "provenance"):
                TeacherCache(output, changed, labels, "train")
            with self.assertRaisesRegex(ValueError, "signature"):
                build_teacher_cache(changed, labels, output, resume=True, map_size_gb=.01)
            changed = copy.deepcopy(config)
            changed["model"]["capsule_radius_m"] = .31
            with self.assertRaises(ValueError):
                TeacherCache(output, changed, labels, "train")
            signature = teacher_signature(config, labels)
            self.assertIn("geometry_sha256", signature["teacher"])
            self.assertIn("lmdb_mtime_ns", signature["labels"])

    def test_test_build_output_overlap_and_missing_records_rejected(self):
        with tempfile.TemporaryDirectory(prefix="rt-teacher-safety-") as temporary:
            root = Path(temporary)
            config, labels, _ = self.fixture(root)
            with self.assertRaises(ValueError):
                build_teacher_cache(config, labels, root / "teacher", splits=("test",), map_size_gb=.01)
            with self.assertRaises(ValueError):
                build_teacher_cache(config, labels, labels / "teacher", map_size_gb=.01)
            output = root / "teacher"
            build_teacher_cache(config, labels, output, map_size_gb=.01)
            # Check stored bytes are actually sparse, then remove a record to
            # prove even a stale "complete" manifest cannot cause fallback.
            with read_env(output / "teacher.lmdb").begin() as txn:
                record = unpack_dict_np(txn.get(frame_key((1, 1, 3))))
                self.assertGreater(record["indices"].size, 0)
                self.assertLess(record["indices"].nbytes, 9**3)
            close_read_envs()
            env = lmdb.open(str(output / "teacher.lmdb"), subdir=False)
            with env.begin(write=True) as txn:
                txn.delete(frame_key((1, 1, 3)))
            env.close()
            with self.assertRaises(KeyError):
                TeacherCache(output, config, labels, "train").get((1, 1, 3))


if __name__ == "__main__":
    unittest.main()
