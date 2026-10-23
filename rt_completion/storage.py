"""Versioned sparse-addition caches; the original radar LMDB is never modified."""

import hashlib
import json
import os
from pathlib import Path

import lmdb
import numpy as np

from dataset.lmdb_utils import pack_dict_np, unpack_dict_np
from . import FORMAT_VERSION


_READ_ENVS = {}
_OWNER_PID = os.getpid()


def read_env(path):
    # Share handles across train/val datasets in one process, reopen after fork.
    global _OWNER_PID
    if _OWNER_PID != os.getpid():
        close_read_envs()
        _OWNER_PID = os.getpid()
    path = str(Path(path).resolve())
    if path not in _READ_ENVS:
        _READ_ENVS[path] = lmdb.open(
            path, readonly=True, create=False, lock=False, readahead=False,
            subdir=False, max_readers=4096,
        )
    return _READ_ENVS[path]


def close_read_envs():
    for env in _READ_ENVS.values():
        env.close()
    _READ_ENVS.clear()


def frame_key(indicator):
    return str([int(x) for x in indicator]).encode()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(cache_path):
    """Do not hash the huge radar file, or read SMPL-X labels at export time."""
    path = Path(cache_path).resolve()
    stat = (path / "radar_comp.lmdb").stat()
    return {
        "cache_path": str(path),
        "radar_size": stat.st_size,
        "radar_mtime_ns": stat.st_mtime_ns,
        "indices_sha256": sha256_file(path / "indeces.pkl.gz"),
    }


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


class SparseWriter:
    """A single writer; LMDB map grows on demand, never replaces existing keys."""

    def __init__(self, path, map_size_gb=8):
        self.env = lmdb.open(str(path), subdir=False, map_size=int(map_size_gb * 2**30))

    def keys(self):
        with self.env.begin() as txn:
            return set(txn.cursor().iternext(values=False))

    def put_many(self, records):
        encoded = [(frame_key(key), pack_dict_np(value)) for key, value in records]
        while True:
            try:
                with self.env.begin(write=True) as txn:
                    for key, value in encoded:
                        if not txn.put(key, value, overwrite=False):
                            raise ValueError(f"Refusing to overwrite cached frame {key!r}")
                return
            except lmdb.MapFullError:
                self.env.set_mapsize(self.env.info()["map_size"] * 2)

    def close(self):
        self.env.sync()
        self.env.close()


def apply_additions(raw, record):
    """Never change even weak nonzero observations; stored zeros are UNKNOWN."""
    indices = np.asarray(record["indices"], dtype=np.int64)
    values = np.asarray(record["values"], dtype=np.float32)
    if indices.ndim != 1 or values.shape != indices.shape:
        raise ValueError("Invalid addition array shape")
    if len(indices) and (indices.min() < 0 or indices.max() >= raw.size):
        raise ValueError("Addition index out of bounds")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid generated intensities")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Duplicate addition indices")
    completed = np.array(raw, copy=True, order="C")
    flat = completed.reshape(-1)
    if np.any(flat[indices] != 0):
        raise ValueError("Cache attempts to overwrite original nonzero RT")
    flat[indices] = values
    return completed


class CompletionCache:
    def __init__(self, path, source_path, scale, protocol, split, radar_shape):
        self.path = str(Path(path).resolve())
        self.manifest = read_json(Path(path) / "manifest.json")
        manifest = self.manifest
        if manifest.get("format_version") != FORMAT_VERSION or manifest.get("kind") != "rt_additions":
            raise ValueError("Unsupported RT completion cache format")
        if manifest["signature"]["source"] != source_identity(source_path):
            raise ValueError("RT completion cache belongs to a different/modified source dataset")
        data = manifest["signature"]["data"]
        if (data["scale"], data["protocol"], tuple(data["shape"])) != (scale, protocol, tuple(radar_shape)):
            raise ValueError("RT completion cache scale/protocol/shape mismatch")
        part = manifest.get("partitions", {}).get(split, {})
        if manifest.get("state") != "complete" or not part.get("complete", False):
            raise ValueError(f"RT completion cache is incomplete for {split}; finish/resume export first")
        if manifest["signature"].get("geometry") != "predicted_from_rt_only":
            raise ValueError("Ground-truth geometry caches are not accepted for radar-only evaluation")

    def get(self, indicator):
        with read_env(Path(self.path) / "additions.lmdb").begin() as txn:
            blob = txn.get(frame_key(indicator))
        if blob is None:
            raise KeyError(f"Completed RT missing for {indicator}; no silent raw-RT fallback")
        return unpack_dict_np(blob)

    def apply(self, raw, indicator):
        return apply_additions(raw, self.get(indicator))
