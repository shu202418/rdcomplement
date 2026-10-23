"""Small LMDB-only dataset, without images, point clouds or inference-time GT."""

import gzip
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.lmdb_utils import unpack_dict_np, unpack_ndarray
from .storage import frame_key, read_env, read_json, source_identity
from . import FORMAT_VERSION


# Kept consistent with RF3DPoseDataset without importing its SMPL-X/plotting code.
INVALID = {(1, 12, 58), (1, 13, 491), (1, 31, 324), (1, 36, 284), (1, 37, 43),
           (5, 43, 551), (5, 43, 552), (5, 47, 136), (5, 47, 140)}


def load_splits(data):
    # This is the project's existing, trusted local split pickle.
    with gzip.open(Path(data["cache_path"]) / "indeces.pkl.gz", "rb") as stream:
        split_file = pickle.load(stream)
    splits = {
        name: [tuple(map(int, x)) for x in values if tuple(x) not in INVALID]
        for name, values in split_file[data["scale"]][data["protocol"]].items()
        if name in ("train", "val", "test")
    }
    sets = {name: set(values) for name, values in splits.items()}
    for name, values in splits.items():
        if len(values) != len(sets[name]):
            raise ValueError(f"Duplicate keys in {name}")
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise ValueError("Train/val/test split overlap")
    return splits


def causal_window(indicator, allowed, length):
    """Only contiguous past frames within this split/sequence; stop at any gap."""
    if length < 1:
        raise ValueError("Temporal window must be positive")
    current = tuple(indicator)
    if current not in allowed:
        raise KeyError(current)
    newest_first = [current]
    for offset in range(1, length):
        previous = (current[0], current[1], current[2] - offset)
        if previous not in allowed:
            break
        newest_first.append(previous)
    newest_first.extend([newest_first[-1]] * (length - len(newest_first)))
    return newest_first[::-1]


class RTSequenceDataset(Dataset):
    def __init__(self, data, split, labels_path=None, limit=None):
        self.data = data
        self.split = split
        self.indicators = load_splits(data)[split]
        self.allowed = set(self.indicators)
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive")
            self.indicators = self.indicators[:limit]
        self.labels_path = labels_path
        if labels_path:
            metadata = read_json(Path(labels_path) / "manifest.json")
            if metadata.get("kind") != "rt_joint_labels" or metadata.get("format_version") != FORMAT_VERSION:
                raise ValueError("Unsupported joint-label cache format")
            if split == "test":
                raise ValueError("Completion test/export data must not load ground-truth joints")
            if metadata["signature"]["source"] != source_identity(data["cache_path"]):
                raise ValueError("Joint-label cache/source mismatch")
            if metadata["signature"]["data"] != data:
                raise ValueError("Joint-label cache/protocol configuration mismatch")
        if not self.indicators:
            raise ValueError(f"Empty {split} dataset")

    def __len__(self):
        return len(self.indicators)

    def window(self, indicator):
        return causal_window(indicator, self.allowed, self.data["window"])

    def _radar(self, indicator):
        with read_env(Path(self.data["cache_path"]) / "radar_comp.lmdb").begin() as txn:
            blob = txn.get(frame_key(indicator))
        if blob is None:
            raise KeyError(f"Source RT missing: {indicator}")
        packed = unpack_ndarray(blob)
        if packed.ndim != 2 or packed.shape[1] != 4 or not np.isfinite(packed).all():
            raise ValueError(f"Invalid source RT: {indicator}")
        coords = packed[:, :3].astype(np.int64)
        if (coords < 0).any() or (coords >= np.array(self.data["shape"])).any() or (packed[:, 3] < 0).any():
            raise ValueError(f"Out-of-bounds/negative RT: {indicator}")
        volume = np.zeros(self.data["shape"], dtype=np.float32)
        volume[tuple(coords.T)] = packed[:, 3]
        return volume

    def __getitem__(self, index):
        indicator = self.indicators[index]
        window = self.window(indicator)
        unique = {key: self._radar(key) for key in set(window)}
        # Duplicated boundary frames may enter the GRU, but NEVER count as
        # independent evidence in a temporal pseudo target.
        seen = {indicator}
        history = []
        for key in reversed(window):
            history.append(float(key not in seen))
            seen.add(key)
        result = {
            "rt": torch.from_numpy(np.stack([unique[key] for key in window])),
            "indicator": torch.tensor(indicator, dtype=torch.long),
            "history_valid": torch.tensor(history[::-1]),
        }
        if self.labels_path is not None:
            joints = []
            with read_env(Path(self.labels_path) / "joints.lmdb").begin() as txn:
                for key in window:
                    blob = txn.get(frame_key(key))
                    if blob is None:
                        raise KeyError(f"Joint label missing: {key}; run prepare-labels for this split")
                    joints.append(unpack_dict_np(blob)["joints"])
            array = np.stack(joints).astype(np.float32)
            if array.shape != (self.data["window"], 22, 3) or not np.isfinite(array).all():
                raise ValueError(f"Invalid joint supervision: {indicator}")
            result["joints"] = torch.from_numpy(array)
        return result
