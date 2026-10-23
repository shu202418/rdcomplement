"""Lossless, training-only sparse caches for the fixed temporal RT teacher.

No learned weights or test labels are used.  A cache is tied to the source,
joint-label LMDB, geometry implementation, and teacher configuration.  Missing
records and partial/debug caches are errors, never a silent on-the-fly fallback.
"""

import copy
import math
import operator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.lmdb_utils import unpack_dict_np
from . import FORMAT_VERSION
from .storage import (SparseWriter, atomic_json, close_read_envs, frame_key, read_env, read_json,
                      sha256_file, source_identity)


TEACHER_VERSION = "articulated-v2-fp32-1"
CODEC_VERSION = 1
TEACHER_FIELDS = ("intensity", "valid_mask", "confidence", "support_count")


def _shape(shape):
    try:
        dimensions = tuple(operator.index(value) for value in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("Teacher shape must contain three positive integers") from exc
    if len(dimensions) != 3 or any(value <= 0 for value in dimensions):
        raise ValueError("Teacher shape must contain three positive integers")
    if math.prod(dimensions) > np.iinfo(np.uint32).max:
        raise ValueError("Teacher sparse codec requires fewer than 2**32 voxels")
    return dimensions


def encode_teacher(teacher, batch_index=0):
    """Encode one B,1,X,Y,Z teacher item with no FP32 quantization.

    The nonzero support-count map is saved separately, including counts outside
    valid_mask.  Thus decoding reconstructs all four existing teacher outputs,
    not just the values that currently contribute to the loss.
    """
    if any(name not in teacher for name in TEACHER_FIELDS):
        raise ValueError("Temporal teacher is missing required fields")
    first = teacher["intensity"]
    if not isinstance(first, torch.Tensor) or first.ndim != 5 or first.shape[1] != 1:
        raise ValueError("Teacher tensors must have shape B,1,X,Y,Z")
    shape = _shape(first.shape[-3:])
    if not 0 <= batch_index < first.shape[0]:
        raise ValueError("Teacher batch index out of bounds")
    arrays = {}
    for name in TEACHER_FIELDS:
        value = teacher[name]
        if not isinstance(value, torch.Tensor) or value.shape != first.shape:
            raise ValueError("Teacher tensor shapes differ")
        arrays[name] = value[batch_index, 0].detach().cpu().numpy().reshape(-1)
    intensity, valid, confidence, count = (arrays[name] for name in TEACHER_FIELDS)
    if intensity.dtype != np.float32 or confidence.dtype != np.float32 or valid.dtype != np.bool_:
        raise ValueError("Teacher intensity/confidence must be FP32 and valid_mask boolean")
    if count.dtype.kind not in "iu" or (count < 0).any():
        raise ValueError("Teacher support counts must be nonnegative integers")
    if (not np.isfinite(intensity).all() or (intensity < 0).any()
            or not np.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any()):
        raise ValueError("Invalid teacher intensity or confidence")
    if np.any(intensity[~valid] != 0) or np.any(confidence[~valid] != 0):
        raise ValueError("Teacher intensity/confidence outside valid_mask must be zero")
    if np.any(count[valid] == 0) or (count.size and count.max() > np.iinfo(np.uint32).max):
        raise ValueError("Invalid teacher support counts")
    indices = np.flatnonzero(valid).astype(np.uint32)
    count_indices = np.flatnonzero(count).astype(np.uint32)
    maximum = int(count.max()) if count.size else 0
    count_dtype = np.uint8 if maximum <= 255 else np.uint16 if maximum <= 65535 else np.uint32
    return {
        "codec_version": CODEC_VERSION,
        "shape": np.asarray(shape, dtype=np.uint32),
        "indices": indices,
        "intensity": intensity[indices].copy(),
        "confidence": confidence[indices].copy(),
        "count_indices": count_indices,
        "support_count": count[count_indices].astype(count_dtype),
    }


def _indices(record, name, size):
    values = np.asarray(record[name])
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError(f"Teacher {name} must be one-dimensional integer indices")
    if len(values) and ((values < 0).any() or values.max() >= size
                        or np.any(values[1:] <= values[:-1])):
        raise ValueError(f"Teacher {name} are out of bounds, duplicate, or unsorted")
    return values.astype(np.int64, copy=False)


def decode_teacher(record, shape):
    """Decode one sparse record into CPU tensors with shape 1,X,Y,Z."""
    shape = _shape(shape)
    if record.get("codec_version") != CODEC_VERSION:
        raise ValueError("Unsupported teacher sparse codec")
    if _shape(np.asarray(record["shape"]).tolist()) != shape:
        raise ValueError("Teacher record grid shape mismatch")
    size = math.prod(shape)
    indices = _indices(record, "indices", size)
    count_indices = _indices(record, "count_indices", size)
    intensity = np.asarray(record["intensity"])
    confidence = np.asarray(record["confidence"])
    counts = np.asarray(record["support_count"])
    if (intensity.dtype != np.float32 or confidence.dtype != np.float32
            or intensity.shape != indices.shape or confidence.shape != indices.shape):
        raise ValueError("Invalid FP32 teacher sparse value arrays")
    if (not np.isfinite(intensity).all() or (intensity < 0).any()
            or not np.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any()):
        raise ValueError("Invalid teacher sparse intensity/confidence")
    if counts.ndim != 1 or counts.shape != count_indices.shape or counts.dtype.kind not in "iu":
        raise ValueError("Invalid teacher sparse support-count array")
    if len(counts) and ((counts <= 0).any() or counts.max() > np.iinfo(np.uint32).max):
        raise ValueError("Teacher stored support counts must be positive uint32-range integers")
    result = {
        "intensity": torch.zeros((1, *shape), dtype=torch.float32),
        "valid_mask": torch.zeros((1, *shape), dtype=torch.bool),
        "confidence": torch.zeros((1, *shape), dtype=torch.float32),
        "support_count": torch.zeros((1, *shape), dtype=torch.int64),
    }
    result["valid_mask"].view(-1)[indices] = True
    result["intensity"].view(-1)[indices] = torch.from_numpy(intensity.copy())
    result["confidence"].view(-1)[indices] = torch.from_numpy(confidence.copy())
    result["support_count"].view(-1)[count_indices] = torch.from_numpy(counts.astype(np.int64))
    if len(indices) and bool((result["support_count"].view(-1)[indices] == 0).any()):
        raise ValueError("Valid teacher targets have no historical support")
    return result


def teacher_signature(config, labels_path):
    """Conservative provenance; loss/model changes never silently reuse a cache."""
    if config["model"].get("architecture") != "reliability_v2":
        raise ValueError("Temporal-teacher preparation requires architecture=reliability_v2")
    if float(config["loss"].get("temporal_weight", 0.)) <= 0:
        raise ValueError("Temporal-teacher preparation requires positive temporal_weight")
    data = copy.deepcopy(config["data"])
    data["cache_path"] = str(Path(data["cache_path"]).resolve())
    labels = Path(labels_path).resolve()
    manifest = read_json(labels / "manifest.json")
    if manifest.get("kind") != "rt_joint_labels" or manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported joint-label cache for temporal teacher")
    source = source_identity(data["cache_path"])
    if manifest["signature"].get("source") != source or manifest["signature"].get("data") != data:
        raise ValueError("Temporal-teacher joint labels/source/protocol mismatch")
    stat = (labels / "joints.lmdb").stat()
    alignment = config["loss"].get("temporal_alignment", "articulated")
    if alignment != "articulated":
        raise ValueError("Sparse temporal-teacher cache supports articulated alignment only")
    return {
        "source": source,
        "data": data,
        "labels": {"path": str(labels), "signature": copy.deepcopy(manifest["signature"]),
                   "lmdb_size": stat.st_size, "lmdb_mtime_ns": stat.st_mtime_ns},
        "loss": copy.deepcopy(config["loss"]),
        "capsule_radius_m": float(config["model"]["capsule_radius_m"]),
        "teacher": {"version": TEACHER_VERSION, "codec_version": CODEC_VERSION,
                    "geometry_sha256": sha256_file(Path(__file__).with_name("geometry.py")),
                    "precision": "float32", "geometry": "training_gt_joints_only"},
    }


class TeacherCache:
    """Pickle-safe lazy reader; no LMDB environment is retained on the object."""

    def __init__(self, path, config, labels_path, split, require_complete=True):
        if split not in ("train", "val"):
            raise ValueError("Temporal-teacher caches never load test labels")
        self.path = str(Path(path).resolve())
        self.split = split
        self.manifest = read_json(Path(self.path) / "manifest.json")
        manifest = self.manifest
        if manifest.get("kind") != "rt_temporal_teacher" or manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported temporal-teacher cache format")
        if manifest.get("signature") != teacher_signature(config, labels_path):
            raise ValueError("Temporal-teacher cache provenance mismatch")
        self.shape = _shape(config["data"]["shape"])
        partition = manifest.get("partitions", {}).get(split, {})
        if require_complete and (manifest.get("state") != "complete"
                                 or not partition.get("complete")
                                 or partition.get("written", 0) != partition.get("expected", -1)
                                 or manifest.get("debug_limits", {}).get("limit") is not None):
            raise ValueError("Temporal-teacher cache is incomplete/debug-only; resume preparation first")
        # Validate records on access rather than keeping a large in-memory key set.
        self.data = copy.deepcopy(manifest["signature"]["data"])

    def get(self, indicator):
        with read_env(Path(self.path) / "teacher.lmdb").begin() as txn:
            blob = txn.get(frame_key(indicator))
        if blob is None:
            raise KeyError(f"Temporal teacher missing for {indicator}; no silent fallback")
        return decode_teacher(unpack_dict_np(blob), self.shape)


class TeacherDataset(Dataset):
    """Attach detached dense CPU teacher tensors, allowing ordinary DataLoader collation."""

    def __init__(self, dataset, cache):
        if dataset.split != cache.split or dataset.split not in ("train", "val"):
            raise ValueError("Temporal teacher and dataset split mismatch")
        data = copy.deepcopy(dataset.data)
        data["cache_path"] = str(Path(data["cache_path"]).resolve())
        if data != cache.data:
            raise ValueError("Temporal teacher and dataset configuration mismatch")
        self.dataset, self.cache = dataset, cache

    @property
    def indicators(self):
        return self.dataset.indicators

    @property
    def split(self):
        return self.dataset.split

    @property
    def data(self):
        return self.dataset.data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        item["teacher"] = self.cache.get(item["indicator"])
        return item


def build_teacher_cache(config, labels_path, output, *, device="cpu", batch_size=2,
                        workers=0, splits=("train", "val"), limit=None, resume=False,
                        map_size_gb=32, log_every=100):
    """Build/resume a model-free teacher cache. Limited builds remain incomplete.

    Existing complete records are filtered before DataLoader construction, so
    resume does not reload or recompute those windows. All requested frames keep
    their full original split as temporal context even for a limited build.
    """
    # Local imports keep this codec/reader usable from pipeline and spawned workers.
    from .data import RTSequenceDataset, load_splits
    from .geometry import temporal_target_v2
    from .pipeline import init_manifest, loader, output_lock

    splits = tuple(splits)
    if not splits or len(set(splits)) != len(splits) or any(split not in ("train", "val") for split in splits):
        raise ValueError("Temporal-teacher preparation supports distinct train/val splits only")
    if batch_size <= 0 or workers < 0 or map_size_gb <= 0 or log_every <= 0 or (limit is not None and limit <= 0):
        raise ValueError("Invalid teacher-cache batch/workers/map-size/limit settings")
    signature = teacher_signature(config, labels_path)
    data = signature["data"]
    output_path, labels = Path(output).resolve(), Path(labels_path).resolve()
    if output_path == labels or output_path in labels.parents or labels in output_path.parents:
        raise ValueError("Teacher output must be separate from the original joint-label cache")
    device = torch.device(device)
    all_splits = load_splits(data)
    split_keys = {split: {frame_key(key) for key in all_splits[split]} for split in ("train", "val")}
    allowed_keys = set.union(*split_keys.values())
    with output_lock(output_path, data["cache_path"]) as output_path:
        manifest = init_manifest(output_path, "rt_temporal_teacher", signature, resume)
        # Standalone preparation must not open a second same-process LMDB handle
        # over a lazy reader from an earlier evaluation/preparation invocation.
        close_read_envs()
        writer = SparseWriter(output_path / "teacher.lmdb", map_size_gb=map_size_gb)
        existing = writer.keys()
        if existing - allowed_keys:
            writer.close()
            raise ValueError("Temporal teacher contains keys outside train/val")
        written_counts = {split: len(existing & keys) for split, keys in split_keys.items()}

        def update_manifest(state="partial"):
            manifest["partitions"] = {
                split: {"expected": len(all_splits[split]),
                        "written": written_counts[split]}
                for split in ("train", "val")
            }
            for partition in manifest["partitions"].values():
                partition["complete"] = partition["written"] == partition["expected"]
            complete = all(partition["complete"] for partition in manifest["partitions"].values())
            if state == "complete":
                manifest["state"] = "complete" if complete and limit is None else "partial"
            else:
                manifest["state"] = state
            manifest["debug_limits"] = {"limit": limit}
            manifest["storage"] = {"record_count": len(existing),
                                   "lmdb_size_bytes": (output_path / "teacher.lmdb").stat().st_size}
            atomic_json(output_path / "manifest.json", manifest)

        try:
            update_manifest("writing")
            for split in splits:
                dataset = RTSequenceDataset(data, split, labels_path)
                requested = dataset.indicators[:limit] if limit is not None else dataset.indicators
                dataset.indicators = [key for key in requested if frame_key(key) not in existing]
                print(f"Teacher {split}: {len(dataset.indicators)} pending / {len(requested)} requested", flush=True)
                if not dataset.indicators:
                    continue
                batches = loader(dataset, batch_size, workers, device=device)
                done = 0
                with torch.inference_mode():
                    for step, batch in enumerate(batches):
                        raw = batch["rt"].to(device, non_blocking=True)
                        joints = batch["joints"].to(device, non_blocking=True)
                        history = batch["history_valid"].to(device, non_blocking=True)
                        teacher = temporal_target_v2(raw, joints, history, data["bounds"],
                                                     signature["capsule_radius_m"], config["loss"])
                        keys = batch["indicator"].tolist()
                        records = [(key, encode_teacher(teacher, index)) for index, key in enumerate(keys)]
                        writer.put_many(records)
                        existing.update(frame_key(key) for key in keys)
                        written_counts[split] += len(records)
                        done += len(records)
                        if step == 0 or (step + 1) % log_every == 0:
                            print(f"  Teacher {split}: {done}/{len(dataset.indicators)}", flush=True)
                            update_manifest("writing")
                update_manifest("writing")
            update_manifest("complete")
        except BaseException:
            update_manifest("partial")
            raise
        finally:
            writer.close()
    print(f"Temporal teacher: {output_path} ({manifest['state']}); train/val only, no model weights", flush=True)
    return manifest
