"""Derive conservative RT-addition caches without rerunning the network."""

import copy
from pathlib import Path

import numpy as np

from dataset.lmdb_utils import unpack_dict_np
from .pipeline import init_manifest, load_splits, output_lock
from .storage import (SparseWriter, atomic_json, frame_key, read_env, read_json,
                      sha256_file)


def prune_record(record, max_added_voxels, max_sigma_log, intensity_multiplier):
    """Filter/rank one exported record while preserving its provenance fields."""
    indices = np.asarray(record["indices"], dtype=np.int32)
    values = np.asarray(record["values"], dtype=np.float32)
    probability = np.asarray(record["probability"], dtype=np.float32)
    sigma = np.asarray(record["sigma_log"], dtype=np.float32)
    if not (indices.ndim == values.ndim == probability.ndim == sigma.ndim == 1):
        raise ValueError("Completion record arrays must be one-dimensional")
    if not (len(indices) == len(values) == len(probability) == len(sigma)):
        raise ValueError("Completion record arrays have inconsistent lengths")
    if not (np.isfinite(values).all() and np.isfinite(probability).all()
            and np.isfinite(sigma).all()):
        raise ValueError("Completion record contains non-finite values")

    keep = np.flatnonzero((sigma <= max_sigma_log) & (values > 0))
    if len(keep) > max_added_voxels:
        score = probability[keep] * values[keep] / np.maximum(sigma[keep], 1e-4)
        # argpartition avoids sorting every candidate merely to retain a small top-k.
        chosen = np.argpartition(score, -max_added_voxels)[-max_added_voxels:]
        keep = keep[chosen]
    keep = keep[np.argsort(indices[keep], kind="stable")]
    result = {
        "indices": indices[keep].astype(np.int32, copy=True),
        "values": (values[keep] * intensity_multiplier).astype(np.float32, copy=False),
        "probability": probability[keep].astype(np.float16),
        "sigma_log": sigma[keep].astype(np.float16),
    }
    if "joints" in record:
        result["joints"] = np.asarray(record["joints"], dtype=np.float32).copy()
    return result


def prune_cache(args):
    source_path = Path(args.source_cache).resolve()
    output_path = Path(args.output).resolve()
    if source_path == output_path or source_path in output_path.parents:
        raise ValueError("Variant output must not be the source cache or one of its children")
    if args.max_added_voxels <= 0:
        raise ValueError("--max-added-voxels must be positive")
    if not np.isfinite(args.max_sigma_log) or args.max_sigma_log <= 0:
        raise ValueError("--max-sigma-log must be positive")
    if not np.isfinite(args.intensity_multiplier) or not 0 < args.intensity_multiplier <= 1:
        raise ValueError("--intensity-multiplier must be in (0, 1]")

    source_manifest_path = source_path / "manifest.json"
    source_manifest = read_json(source_manifest_path)
    if (source_manifest.get("kind") != "rt_additions"
            or source_manifest.get("state") != "complete"):
        raise ValueError("Source must be a complete RT-additions cache")
    for split in ("train", "val", "test"):
        if not source_manifest.get("partitions", {}).get(split, {}).get("complete", False):
            raise ValueError(f"Source completion cache is incomplete for {split}")

    signature = copy.deepcopy(source_manifest["signature"])
    source_settings = signature.setdefault("settings", {})
    source_settings["max_added_voxels"] = int(args.max_added_voxels)
    source_settings["max_sigma_log"] = min(
        float(source_settings.get("max_sigma_log", float("inf"))),
        float(args.max_sigma_log),
    )
    source_settings["post_intensity_multiplier"] = float(args.intensity_multiplier)
    signature["derived_from"] = {
        "cache": str(source_path),
        "manifest_sha256": sha256_file(source_manifest_path),
        "operation": "confidence_budget_intensity_prune",
    }
    data = signature["data"]
    splits = load_splits(data)

    with output_lock(output_path, data["cache_path"]) as output:
        manifest = init_manifest(output, "rt_additions", signature, args.resume)
        manifest["training"] = copy.deepcopy(source_manifest.get("training", {}))
        manifest["postprocess"] = {
            "max_added_voxels": int(args.max_added_voxels),
            "max_sigma_log": float(args.max_sigma_log),
            "intensity_multiplier": float(args.intensity_multiplier),
        }
        manifest["state"] = "writing"
        atomic_json(output / "manifest.json", manifest)
        writer = SparseWriter(output / "additions.lmdb", args.map_size_gb)
        source_env = read_env(source_path / "additions.lmdb")
        try:
            existing = writer.keys()
            for split in ("train", "val", "test"):
                pending = [key for key in splits[split] if frame_key(key) not in existing]
                print(f"Prune {split}: {len(pending)} pending frames", flush=True)
                written, kept, chunk = 0, 0, []
                with source_env.begin() as transaction:
                    for key in pending:
                        blob = transaction.get(frame_key(key))
                        if blob is None:
                            raise KeyError(f"Source completion cache is missing {key}")
                        record = prune_record(
                            unpack_dict_np(blob), args.max_added_voxels,
                            args.max_sigma_log, args.intensity_multiplier,
                        )
                        chunk.append((key, record))
                        kept += len(record["indices"])
                        if len(chunk) >= 1000:
                            writer.put_many(chunk)
                            existing.update(frame_key(item) for item, _ in chunk)
                            written += len(chunk)
                            chunk.clear()
                            print(
                                f"  {split}: {written}/{len(pending)}, "
                                f"mean kept={kept / written:.1f}", flush=True,
                            )
                    if chunk:
                        writer.put_many(chunk)
                        existing.update(frame_key(item) for item, _ in chunk)
                        written += len(chunk)
                done = sum(frame_key(key) in existing for key in splits[split])
                manifest["partitions"][split] = {
                    "expected": len(splits[split]), "written": done,
                    "complete": done == len(splits[split]),
                }
                atomic_json(output / "manifest.json", manifest)
            manifest["state"] = "complete" if all(
                manifest["partitions"].get(split, {}).get("complete", False)
                for split in ("train", "val", "test")
            ) else "partial"
            atomic_json(output / "manifest.json", manifest)
        finally:
            writer.close()
    print(f"Pruned RT cache: {output_path}; state={manifest['state']}", flush=True)

