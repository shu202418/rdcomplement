"""Offline commands. Export never constructs SMPL-X or opens joint labels."""

import contextlib
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from dataset.lmdb_utils import unpack_dict_np
from . import FORMAT_VERSION
from .data import RTSequenceDataset, causal_window, load_splits
from .diagnostics import write_numerics_event
from .evaluation import GroupMetrics, MASK_GROUPS, fixed_corruption
from .losses import completion_loss, corrupt_rt
from .model import RTCompletionNet
from .numerics import (graceful_load_scaler, make_grad_scaler, precision_policy,
                       safe_optimizer_step)
from .progress import (DeterministicBatchSampler, capture_rng_state, group_metrics_state,
                       restore_group_metrics_state, restore_rng_state)
from .selection import select_additions
from .storage import (SparseWriter, atomic_json, frame_key, read_env, read_json,
                      sha256_file, source_identity)


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    data = config["data"]
    data["cache_path"] = str(Path(data["cache_path"]).resolve())
    if len(data["shape"]) != 3 or any(n < 9 for n in data["shape"]):
        raise ValueError("Expected three grid dimensions >= 9")
    if len(data["bounds"]) != 3 or any(lo >= hi for lo, hi in data["bounds"]):
        raise ValueError("Invalid X/Y/Z bounds")
    if data["window"] < 1 or data["intensity_scale"] <= 0 or data["intensity_cap"] <= 0:
        raise ValueError("Invalid normalization or temporal window")
    loss = config["loss"]
    if loss["temporal_weight"] > 0 and not (1 <= loss["temporal_min_support"] < data["window"]):
        raise ValueError("temporal_min_support must be smaller than the temporal window")
    architecture = config["model"].get("architecture", "baseline")
    if architecture not in ("baseline", "reliability_v2"):
        raise ValueError(f"Unknown completion architecture: {architecture}")
    groups = config.get("validation", {}).get("groups", [])
    if groups and (len(set(groups)) != len(groups) or any(group not in MASK_GROUPS for group in groups)):
        raise ValueError("validation.groups must contain distinct supported mask groups")
    if config["mask"].get("mode") == "mixed_parts":
        weights = config["mask"].get("group_weights", {})
        if weights and (set(weights) - set(MASK_GROUPS)
                        or any(float(value) < 0 for value in weights.values())
                        or sum(float(value) for value in weights.values()) <= 0):
            raise ValueError("Invalid mixed mask group weights")
    settings = config["train"]
    for key, default, minimum in (("checkpoint_every_steps", 1000, 0),
                                  ("detailed_metrics_every", 0, 0),
                                  ("max_nonfinite_batches", 20, 0),
                                  ("max_consecutive_nonfinite", 3, 1),
                                  ("max_failure_snapshots", 3, 0)):
        value = settings.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"train.{key} must be an integer >= {minimum}")
    if settings.get("amp_dtype", "auto") not in ("auto", "float32", "float16", "bfloat16"):
        raise ValueError("Invalid train.amp_dtype")
    return config


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; use --device cpu for smoke tests")
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def output_lock(path, source):
    path, source = Path(path).resolve(), Path(source).resolve()
    if path == source or path in source.parents or source in path.parents:
        raise ValueError("Output must be a separate directory, not the source cache or its parent/child")
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".writer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another writer is using {path}") from exc
        yield path


def init_manifest(path, kind, signature, resume):
    manifest_path = path / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"{path} already exists; use --resume with identical settings")
        manifest = read_json(manifest_path)
        if (manifest.get("kind") != kind or manifest.get("format_version") != FORMAT_VERSION
                or manifest.get("signature") != signature):
            raise ValueError("Resume signature mismatch: source, protocol, checkpoint or settings changed")
        return manifest
    if any(item.name != ".writer.lock" for item in path.iterdir()):
        raise FileExistsError(f"Refusing to use non-empty output directory {path}")
    manifest = {"format_version": FORMAT_VERSION, "kind": kind, "signature": signature,
                "partitions": {}, "state": "writing"}
    atomic_json(manifest_path, manifest)
    return manifest


def loader(dataset, batch_size, workers, shuffle=False, seed=42, device=None, start_step=0):
    # Do not fork an open export writer/file lock or an initialized CUDA model
    # into workers. Only the lightweight, pickleable dataset enters each worker.
    process_options = {"multiprocessing_context": "spawn", "timeout": 60} if workers else {}
    common = dict(num_workers=workers, pin_memory=device is not None and device.type == "cuda",
                  generator=torch.Generator().manual_seed(seed), persistent_workers=False,
                  **process_options)
    if shuffle:
        sampler = DeterministicBatchSampler(len(dataset), batch_size, seed, start_step)
        return DataLoader(dataset, batch_sampler=sampler, **common)
    if start_step:
        raise ValueError("Only training loaders support a resumed batch offset")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **common)


def prepare_labels(args):
    # Import the project's calibration only here: export has no access to GT.
    import smplx
    from dataset.data_loader_camera_calibration import calibrate_param_to_radar

    config = load_config(args.config)
    data = config["data"]
    device = choose_device(args.device)
    smplx_root = Path(args.smplx_models).resolve()
    signature = {"source": source_identity(data["cache_path"]), "data": data,
                 "smplx_models": str(smplx_root), "units": "radar_space_meters"}
    # Calibration/parameter changes invalidate label-resume as well.
    signature["label_sources"] = {}
    for path in [Path(data["cache_path"]) / name for name in ("params.lmdb", "calib.lmdb")] + [
        smplx_root / "smplx" / f"SMPLX_{sex.upper()}.pkl" for sex in ("male", "female")
    ]:
        stat = path.stat()
        signature["label_sources"][str(path)] = [stat.st_size, stat.st_mtime_ns]
    splits = load_splits(data)
    models = {sex: smplx.create(str(smplx_root), model_type="smplx", gender=sex).to(device).eval()
              for sex in ("male", "female")}
    male_subjects = {2, 5, 6, 9, 11, 12, 14, 16, 17, 18, 19, 20}
    with output_lock(args.output, data["cache_path"]) as output:
        manifest = init_manifest(output, "rt_joint_labels", signature, args.resume)
        writer = SparseWriter(output / "joints.lmdb", map_size_gb=1)
        try:
            existing = writer.keys()
            for split in ("train", "val"):
                allowed = set(splits[split])
                targets = splits[split][:args.limit] if args.limit else splits[split]
                needed = sorted({key for target in targets for key in causal_window(target, allowed, data["window"])})
                pending = [key for key in needed if frame_key(key) not in existing]
                print(f"Labels {split}: {len(pending)} pending / {len(needed)} requested", flush=True)
                for start in range(0, len(pending), args.batch_size):
                    keys = pending[start:start + args.batch_size]
                    groups = {"male": [], "female": []}
                    with read_env(Path(data["cache_path"]) / "params.lmdb").begin() as params_txn, \
                            read_env(Path(data["cache_path"]) / "calib.lmdb").begin() as calib_txn:
                        for key in keys:
                            p = params_txn.get(frame_key(key))
                            c = calib_txn.get(frame_key(key))
                            if p is None or c is None:
                                raise KeyError(f"Missing SMPL-X parameters/calibration: {key}")
                            calibrated = calibrate_param_to_radar(unpack_dict_np(p), unpack_dict_np(c))
                            sex = "male" if key[0] in male_subjects else "female"
                            groups[sex].append((key, calibrated))
                    records = []
                    for sex, group in groups.items():
                        if not group:
                            continue
                        parameters = {
                            field: torch.as_tensor(np.stack([p[source] for _, p in group]),
                                                   dtype=torch.float32, device=device)
                            for field, source in (("betas", "betas"), ("body_pose", "pose_body"),
                                                  ("global_orient", "root_orient"), ("transl", "trans"))
                        }
                        # smplx.create defaults to batch_size=1. Explicitly
                        # broadcast its default hand/face parameters for variable
                        # batches (including the final short batch).
                        for field in ("left_hand_pose", "right_hand_pose", "jaw_pose",
                                      "leye_pose", "reye_pose", "expression"):
                            parameters[field] = getattr(models[sex], field).expand(len(group), -1)
                        with torch.inference_mode():
                            mesh = models[sex](**parameters)
                            # Same J_regressor[:22] @ vertices as RF3DPoseDataset,
                            # not SMPL-X's differently ordered output.joints list.
                            joints = torch.einsum("jv,bvc->bjc", models[sex].J_regressor[:22], mesh.vertices)
                        for (key, _), array in zip(group, joints.cpu().numpy()):
                            if not np.isfinite(array).all():
                                raise ValueError(f"Non-finite SMPL-X joints for {key}")
                            records.append((key, {"joints": array.astype(np.float32)}))
                    writer.put_many(records)
                    existing.update(frame_key(key) for key, _ in records)
                    if start == 0 or (start // args.batch_size + 1) % 100 == 0:
                        print(f"  {split}: {min(start + args.batch_size, len(pending))}/{len(pending)}", flush=True)
                done = sum(frame_key(key) in existing for key in splits[split])
                manifest["partitions"][split] = {"expected": len(splits[split]), "written": done,
                                                  "complete": done == len(splits[split])}
                atomic_json(output / "manifest.json", manifest)
            manifest["state"] = "complete" if all(p["complete"] for p in manifest["partitions"].values()) else "partial"
            atomic_json(output / "manifest.json", manifest)
        finally:
            writer.close()
    print(f"Joint labels: {output} ({manifest['state']}); no test labels generated", flush=True)


def _batch_forward(model, corrupted, raw, masked, joints, history, config, policy, temporal_teacher=None):
    with torch.autocast(device_type=raw.device.type, dtype=policy["dtype"], enabled=policy["enabled"]):
        prediction = model(corrupted, history_valid=history)
    teacher_options = {"temporal_teacher": temporal_teacher} if temporal_teacher is not None else {}
    loss, metrics = completion_loss(prediction, raw, masked, joints, history, model, config["loss"],
                                    **teacher_options)
    return prediction, loss, metrics


def _training_batch(model, optimizer, corrupted, raw, masked, joints, history, indicators,
                    config, policy, epoch, step, diagnostic_dir, temporal_teacher=None):
    """Never step with bad gradients; optionally retry the same inputs in FP32."""
    scaler = model._training_scaler
    optimizer.zero_grad(set_to_none=True)
    prediction, loss, metrics = _batch_forward(model, corrupted, raw, masked, joints, history, config, policy,
                                             temporal_teacher)
    if bool(torch.isfinite(loss)):
        scaler.scale(loss).backward()
        status = safe_optimizer_step(model, optimizer, scaler, config["train"]["grad_clip"])
    else:
        status = {"updated": False, "grad_norm": None, "reason": "nonfinite_loss",
                  "nonfinite_parameters": [], "overflow": False,
                  "scale_before": float(scaler.get_scale()), "scale_after": float(scaler.get_scale())}
    if status["updated"]:
        return prediction, loss, metrics, status, False
    first_failure = dict(status)
    failed_metrics = {key: value.detach() for key, value in metrics.items()}
    recovered = False
    if policy["enabled"] and config["train"].get("retry_fp32", True):
        # GradScaler has already handled any FP16 overflow. A disabled scaler
        # here avoids applying the same oversized scale to the FP32 retry.
        optimizer.zero_grad(set_to_none=True)
        # In particular, a nonfinite forward loss has not run backward yet.
        # Release its full AMP graph before allocating the FP32 retry graph.
        del prediction, loss, metrics
        full_precision = {"enabled": False, "dtype": torch.float32}
        prediction, loss, metrics = _batch_forward(
            model, corrupted, raw, masked, joints, history, config, full_precision, temporal_teacher)
        if bool(torch.isfinite(loss)):
            loss.backward()
            retry_scaler = torch.amp.GradScaler(raw.device.type, enabled=False)
            status = safe_optimizer_step(model, optimizer, retry_scaler, config["train"]["grad_clip"])
            recovered = status["updated"]
        else:
            status = {**status, "updated": False, "reason": "nonfinite_loss_fp32"}
    event = {"epoch": epoch + 1, "step": step + 1,
             "indicators": indicators.detach().cpu().tolist(), "amp_dtype": policy["name"],
             "reason": first_failure["reason"], "initial_update": first_failure,
             "retry_update": status, "recovered_fp32": recovered,
             "action": "recovered_fp32" if recovered else "skipped_batch", "metrics": failed_metrics}
    if diagnostic_dir is not None:
        write_numerics_event(diagnostic_dir, event,
                             {"raw": raw, "corrupted": corrupted, "mask": masked, "joints": joints,
                              "history_valid": history, "indicators": indicators},
                             max_snapshots=config["train"].get("max_failure_snapshots", 3))
    print(f"WARNING: Epoch [{epoch + 1}/{config['train']['epochs']}] step {step + 1}: "
          f"{first_failure['reason']}; {'recovered in FP32' if recovered else 'update skipped'}; "
          f"scale {first_failure['scale_before']:.4g}->{first_failure['scale_after']:.4g}", flush=True)
    if not recovered:
        optimizer.zero_grad(set_to_none=True)
        prediction = None
        loss = loss.detach()
        metrics = {key: value.detach() for key, value in metrics.items()}
    return prediction, loss, metrics, status, recovered


def _training_counts(state=None):
    state = state or {}
    return {key: int(state.get(key, 0)) for key in
            ("successful_updates", "skipped_batches", "recovered_batches", "consecutive_nonfinite")}


def _record_update(counts, status, recovered):
    if status["updated"]:
        counts["successful_updates"] += 1
        counts["consecutive_nonfinite"] = 0
        counts["recovered_batches"] += int(recovered)
    else:
        counts["skipped_batches"] += 1
        counts["consecutive_nonfinite"] += 1


def _too_many_anomalies(counts, config):
    return (counts["skipped_batches"] > config["train"].get("max_nonfinite_batches", 20)
            or counts["consecutive_nonfinite"] >= config["train"].get("max_consecutive_nonfinite", 3))


def _checkpoint_due(next_step, config):
    every = config["train"].get("checkpoint_every_steps", 1000)
    return next_step == 1 or (every > 0 and next_step % every == 0)


def _batch_teacher(batch, raw, joints, history, model, config):
    """One immutable teacher per original batch, shared across all val masks/retries."""
    if (config["model"].get("architecture") != "reliability_v2"
            or config["loss"].get("temporal_weight", 0.) <= 0
            or config["loss"].get("temporal_alignment", "articulated") != "articulated"):
        if "teacher" in batch:
            raise ValueError("Teacher caches require active V2 articulated temporal supervision")
        return None
    if "teacher" in batch:
        return {name: value.to(raw.device, non_blocking=True).detach()
                for name, value in batch["teacher"].items()}
    from .geometry import temporal_target_v2
    return temporal_target_v2(raw, joints.detach(), history, model.bounds, model.radius, config["loss"])


def _with_teacher_cache(dataset, args, config):
    cache_path = getattr(args, "teacher_cache", None)
    if not cache_path:
        return dataset
    from .teacher_cache import TeacherCache, TeacherDataset
    return TeacherDataset(dataset, TeacherCache(cache_path, config, args.labels, dataset.split))


def prepare_teacher(args):
    """Cache weak train/val supervision, never completed RT or test geometry."""
    from .teacher_cache import build_teacher_cache
    config = load_config(args.config)
    return build_teacher_cache(config, args.labels, args.output, device=choose_device(args.device),
                               batch_size=args.batch_size, workers=args.workers, splits=args.splits,
                               limit=args.limit, resume=args.resume, map_size_gb=args.map_size_gb)


def run_epoch(model, batches, config, device, epoch, optimizer=None, max_steps=None,
              *, progress_callback=None, resume_state=None, diagnostic_dir=None):
    if (config["model"].get("architecture") == "reliability_v2"
            or config.get("validation", {}).get("groups")):
        return run_grouped_epoch(model, batches, config, device, epoch, optimizer, max_steps,
                                 progress_callback=progress_callback, resume_state=resume_state,
                                 diagnostic_dir=diagnostic_dir)
    training = optimizer is not None
    model.train(training)
    mask_rng = torch.Generator(device=device).manual_seed(config["train"]["seed"] + (epoch if training else 100000))
    policy = precision_policy(config, device)
    totals, samples = {}, 0
    start_step = int((resume_state or {}).get("next_step", 0))
    counts = _training_counts(resume_state)
    if resume_state:
        totals = dict(resume_state["accumulator"]["totals"])
        samples = int(resume_state["accumulator"]["samples"])
        restore_rng_state(resume_state["rng_state"], mask_rng)
    next_step = start_step
    def checkpoint():
        if progress_callback is not None:
            progress_callback({"next_step": next_step, "accumulator": {"totals": totals, "samples": samples},
                               "mask_mix": {}, "rng_state": capture_rng_state(mask_rng), **counts})
    for step, batch in enumerate(batches, start=start_step):
        if max_steps is not None and step >= max_steps:
            break
        raw = batch["rt"].to(device, non_blocking=True)
        joints = batch["joints"].to(device, non_blocking=True)
        history = batch["history_valid"].to(device, non_blocking=True)
        corrupted, masked = corrupt_rt(raw, config["mask"], mask_rng)
        with torch.set_grad_enabled(training):
            if training:
                prediction, loss, metrics, status, recovered = _training_batch(
                    model, optimizer, corrupted, raw, masked, joints, history, batch["indicator"],
                    config, policy, epoch, step, diagnostic_dir)
                _record_update(counts, status, recovered)
            else:
                prediction, loss, metrics = _batch_forward(
                    model, corrupted, raw, masked, joints, history, config, policy)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite completion validation loss at step {step}")
        next_step = step + 1
        if training and not status["updated"]:
            if _too_many_anomalies(counts, config):
                checkpoint()
                raise FloatingPointError("Too many non-finite batches; safe progress checkpoint saved")
            if _checkpoint_due(next_step, config):
                checkpoint()
            continue
        batch_size = raw.shape[0]
        samples += batch_size
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        if training and _checkpoint_due(next_step, config):
            checkpoint()
        if (step + 1) % config["train"]["log_every"] == 0:
            print(f"  Epoch [{epoch + 1}/{config['train']['epochs']}] "
                  f"{'train' if training else 'val'} step {step + 1}: loss={float(loss):.5f}", flush=True)
    if training:
        checkpoint()
    if not samples:
        raise ValueError("No batches processed")
    result = {key: value / samples for key, value in totals.items()}
    if training:
        result.update(counts)
    return result


def run_grouped_epoch(model, batches, config, device, epoch, optimizer=None, max_steps=None,
                      *, progress_callback=None, resume_state=None, diagnostic_dir=None):
    """Train mixed masks; validate fixed groups without any GT forward input."""
    training = optimizer is not None
    model.train(training)
    rng = torch.Generator(device=device).manual_seed(config["train"]["seed"] + epoch)
    groups = ["mixed"] if training else config.get("validation", {}).get("groups", list(MASK_GROUPS))
    if not groups:
        groups = list(MASK_GROUPS)
    collectors = {group: GroupMetrics() for group in groups}
    mask_mix = {}
    policy = precision_policy(config, device)
    start_step = int((resume_state or {}).get("next_step", 0))
    counts = _training_counts(resume_state)
    if resume_state:
        collectors["mixed"] = restore_group_metrics_state(resume_state["accumulator"])
        mask_mix = dict(resume_state["mask_mix"])
        restore_rng_state(resume_state["rng_state"], rng)
    steps = start_step
    def checkpoint():
        if progress_callback is not None:
            progress_callback({"next_step": steps, "accumulator": group_metrics_state(collectors["mixed"]),
                               "mask_mix": mask_mix, "rng_state": capture_rng_state(rng), **counts})
    for step, batch in enumerate(batches, start=start_step):
        if max_steps is not None and step >= max_steps:
            break
        raw = batch["rt"].to(device, non_blocking=True)
        joints = batch["joints"].to(device, non_blocking=True)
        history = batch["history_valid"].to(device, non_blocking=True)
        teacher = _batch_teacher(batch, raw, joints, history, model, config)
        for group in groups:
            if training:
                corrupted, masked, info = corrupt_rt(
                    raw, config["mask"], rng, joints=joints, bounds=config["data"]["bounds"],
                    return_info=True)
                for kind in info["kind"]:
                    mask_mix[kind] = mask_mix.get(kind, 0) + 1
            else:
                corrupted, masked, info = fixed_corruption(
                    raw, joints, batch["indicator"], config["mask"], config["data"]["bounds"],
                    config.get("validation", {}).get("seed", config["train"]["seed"] + 100000), group)
            with torch.set_grad_enabled(training):
                if training:
                    prediction, loss, metrics, status, recovered = _training_batch(
                        model, optimizer, corrupted, raw, masked, joints, history, batch["indicator"],
                        config, policy, epoch, step, diagnostic_dir, teacher)
                    _record_update(counts, status, recovered)
                else:
                    prediction, loss, metrics = _batch_forward(
                        model, corrupted, raw, masked, joints, history, config, policy, teacher)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite {group} validation loss at step {step}")
            if not training or status["updated"]:
                every = config["train"].get("detailed_metrics_every", 0)
                detailed = not training or (every > 0 and (step + 1) % every == 0)
                collectors[group].update(metrics, model, raw, corrupted, masked, prediction, info,
                                         config["export"], detailed=detailed)
        steps = step + 1
        if training:
            if _too_many_anomalies(counts, config):
                checkpoint()
                raise FloatingPointError("Too many non-finite batches; safe progress checkpoint saved")
            if _checkpoint_due(steps, config):
                checkpoint()
        if (step + 1) % config["train"]["log_every"] == 0:
            print(f"  Epoch [{epoch + 1}/{config['train']['epochs']}] "
                  f"{'train' if training else 'val grouped'} step {step + 1}: loss={float(loss):.5f}", flush=True)
    if training:
        checkpoint()
    if not steps:
        raise ValueError("No batches processed")
    reports = {group: collector.summary() for group, collector in collectors.items()}
    if training:
        result = reports["mixed"]
        if not collectors["mixed"].samples:
            raise FloatingPointError("No successful training batches; safe progress checkpoint saved")
        result["mask_mix"] = mask_mix
        result.update(counts)
        # Training batches may occasionally contain no eligible held-out point.
        return result
    eligible = [report for report in reports.values() if report["total_masked_voxels"]]
    if not eligible:
        raise ValueError("No eligible held-out observations in any validation mask group")
    # Each nonempty group has equal influence; dense groups cannot dominate.
    result = {key: sum(report[key] for report in eligible) / len(eligible)
              for key in ("masked_mae_log", "joint_mpjpe_m", "loss")}
    result.update(groups=reports, selection_aggregation="macro_over_nonempty_mask_groups",
                  evaluated_groups=len(reports), eligible_groups=len(eligible))
    return result


def save_checkpoint(path, payload):
    temporary = str(path) + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(args):
    config = load_config(args.config)
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    seed_everything(config["train"]["seed"])
    device = choose_device(args.device)
    identity = source_identity(config["data"]["cache_path"])
    datasets = {split: RTSequenceDataset(config["data"], split, args.labels,
                                         args.limit_train if split == "train" else args.limit_val)
                for split in ("train", "val")}
    datasets = {split: _with_teacher_cache(dataset, args, config) for split, dataset in datasets.items()}
    label_signature = read_json(Path(args.labels) / "manifest.json")["signature"]
    model = RTCompletionNet(config["data"], config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"]["lr"]),
                                  weight_decay=float(config["train"]["weight_decay"]))
    policy = precision_policy(config, device)
    model._training_scaler = make_grad_scaler(policy, device)
    best, first_epoch = float("inf"), 0
    resume_training_state = None
    debug_limits = {"train": args.limit_train, "val": args.limit_val, "steps": args.max_steps}
    if any(value is not None for value in debug_limits.values()):
        print("WARNING: debug limits enabled; this run is not a full-data training result", flush=True)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["source"] != identity or checkpoint["config"]["data"] != config["data"]:
            raise ValueError("Resume source/protocol mismatch")
        if checkpoint["label_signature"] != label_signature:
            raise ValueError("Resume joint-supervision provenance changed")
        for section in ("model", "loss", "mask"):
            if checkpoint["config"][section] != config[section]:
                raise ValueError(f"Resume {section} settings changed")
        if checkpoint["config"].get("validation") != config.get("validation"):
            raise ValueError("Resume validation mask/selection settings changed")
        if checkpoint["debug_limits"] != debug_limits:
            raise ValueError("Resume smoke-test limits changed; use the same limits")
        old_train = dict(checkpoint["config"]["train"])
        new_train = dict(config["train"])
        # Operational safeguards can be introduced to old checkpoints without
        # changing model/data/loss or optimization hyperparameters.
        operational = ("epochs", "amp_dtype", "checkpoint_every_steps", "max_nonfinite_batches",
                       "max_consecutive_nonfinite", "retry_fp32", "max_failure_snapshots",
                       "detailed_metrics_every")
        for key in operational:
            old_train.pop(key, None)
            new_train.pop(key, None)
        if old_train != new_train:
            raise ValueError("Resume training settings changed (only epochs may increase)")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        graceful_load_scaler(model._training_scaler, checkpoint.get("scaler"))
        best = checkpoint["best_score"]
        if checkpoint.get("epoch_complete", True):
            first_epoch = checkpoint["epoch"] + 1
            if checkpoint.get("rng_state"):
                restore_rng_state(checkpoint["rng_state"])
        else:
            first_epoch = checkpoint["epoch"]
            resume_training_state = checkpoint.get("progress")
            if not resume_training_state or "next_step" not in resume_training_state:
                raise ValueError("Incomplete checkpoint lacks a resumable training position")
    with output_lock(args.output, config["data"]["cache_path"]) as output:
        if any(item.name != ".writer.lock" for item in output.iterdir()) and not args.resume:
            raise FileExistsError(f"Training output is not empty: {output}; choose a new directory or --resume")
        if args.resume and Path(args.resume).resolve().parent != output:
            raise ValueError("Resume into the original checkpoint directory to preserve best.pt/history")
        atomic_json(output / "config.json", config)
        architecture = config["model"].get("architecture", "baseline")
        print(f"Training {architecture} on {device}: {sum(p.numel() for p in model.parameters()):,} parameters; "
              f"{len(datasets['train'])} train / {len(datasets['val'])} val frames", flush=True)
        print(f"Precision: {policy['name']}; GradScaler={'enabled' if policy['scaling'] else 'disabled'}; "
              f"step checkpoint interval={config['train'].get('checkpoint_every_steps', 1000)}", flush=True)
        if architecture == "reliability_v2":
            cache_path = getattr(args, "teacher_cache", None)
            print(f"Temporal teacher: {Path(cache_path).resolve() if cache_path else 'online (shared across val groups)'}; "
                  f"train detailed_metrics_every={config['train'].get('detailed_metrics_every', 0)}; "
                  "validation diagnostics=full", flush=True)
        def payload_for(epoch, metrics, *, complete, progress=None):
            return {
                "format_version": FORMAT_VERSION, "epoch": epoch, "epoch_complete": complete,
                "config": config, "source": identity, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scaler": model._training_scaler.state_dict(),
                "best_score": best, "debug_limits": debug_limits,
                "train_frames": len(datasets["train"]), "label_signature": label_signature,
                "val_frames": len(datasets["val"]), "metrics": metrics,
                "progress": progress, "rng_state": capture_rng_state(),
                "precision": {"name": policy["name"], "scaling": policy["scaling"]},
                "training_runtime_version": 3,
                "teacher_cache": (str(Path(args.teacher_cache).resolve())
                                  if getattr(args, "teacher_cache", None) else None),
            }
        for epoch in range(first_epoch, config["train"]["epochs"]):
            epoch_label = f"Epoch [{epoch + 1}/{config['train']['epochs']}]"
            print(f"{epoch_label} started", flush=True)
            started = time.monotonic()
            metrics = {}
            def save_progress(state):
                save_checkpoint(output / "progress.pt", payload_for(epoch, {}, complete=False, progress=state))
                print(f"{epoch_label} saved progress.pt at step {state['next_step']}", flush=True)
            for split in ("train", "val"):
                print(f"{epoch_label} {split} started", flush=True)
                state = resume_training_state if split == "train" else None
                start_step = state["next_step"] if state else 0
                if state:
                    print(f"{epoch_label} resuming: {start_step} training batches already processed", flush=True)
                batches = loader(datasets[split], config["train"]["batch_size"], args.workers,
                                 shuffle=split == "train", seed=config["train"]["seed"] + epoch,
                                 device=device, start_step=start_step)
                metrics[split] = run_epoch(model, batches, config, device, epoch,
                                           optimizer if split == "train" else None, args.max_steps,
                                           progress_callback=save_progress if split == "train" else None,
                                           resume_state=state, diagnostic_dir=output)
                if split == "train":
                    resume_training_state = None
            # Select on validation only, never on test labels or downstream test MPJPE.
            score = metrics["val"]["masked_mae_log"] + config["train"]["selection_joint_weight"] * metrics["val"]["joint_mpjpe_m"]
            improved = score < best
            best = min(score, best)
            payload = payload_for(epoch, metrics, complete=True)
            save_checkpoint(output / "last.pt", payload)
            if improved:
                save_checkpoint(output / "best.pt", payload)
            # Keep the advertised recovery path current after validation too.
            save_checkpoint(output / "progress.pt", payload)
            report = {"epoch": epoch + 1, "total_epochs": config["train"]["epochs"],
                      "seconds": time.monotonic() - started,
                      "selection_score": score, **metrics}
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report, allow_nan=False) + "\n")
            if architecture == "reliability_v2":
                # Full per-part histograms remain in metrics.jsonl; printing
                # every bin every epoch would obscure the useful progress.
                console = {key: report[key] for key in ("epoch", "total_epochs", "seconds", "selection_score")}
                console["train"] = {key: metrics["train"].get(key)
                                    for key in ("loss", "masked_mae_log", "joint_mpjpe_m",
                                                "temporal_confidence", "history_null", "mask_mix",
                                                "successful_updates", "skipped_batches", "recovered_batches")}
                console["val"] = {key: metrics["val"][key]
                                  for key in ("masked_mae_log", "joint_mpjpe_m")}
                console["val"]["groups"] = {
                    name: {key: values.get(key) for key in
                           ("masked_mae_log", "eligible_frame_fraction", "masked_coverage68",
                            "masked_coverage95", "accepted_heldout_coverage")}
                    for name, values in metrics["val"]["groups"].items()}
            else:
                console = report
            print(json.dumps(console, ensure_ascii=False, allow_nan=False), flush=True)
            print(f"{epoch_label} completed in {report['seconds']:.1f}s", flush=True)
    print(f"Training finished: {output / 'best.pt'}", flush=True)


@torch.no_grad()
def prediction_to_records(model, raw, prediction, config):
    selected = select_additions(model, raw, prediction, config)
    probability, sigma, values = selected["probability"], selected["sigma"], selected["values"]
    records = []
    for batch in range(raw.shape[0]):
        indices = selected["accepted"][batch].flatten().nonzero().flatten()
        indices = indices.sort().values
        records.append({
            "indices": indices.cpu().numpy().astype(np.int32),
            "values": values[batch].flatten()[indices].cpu().numpy().astype(np.float32),
            "probability": probability[batch].flatten()[indices].cpu().numpy().astype(np.float16),
            "sigma_log": sigma[batch].flatten()[indices].cpu().numpy().astype(np.float16),
            "joints": prediction["joints"][batch].cpu().numpy().astype(np.float32),
        })
    return records


def finite_prediction(value):
    """V2 additionally returns nested multi-scale diagnostic tensors."""
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_prediction(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_prediction(item) for item in value)
    return True


def evaluate(args):
    """Read-only checkpoint evaluation; writes a NEW val report, never test GT."""
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation report {output}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("Not an offline RT completion checkpoint")
    config = copy.deepcopy(checkpoint["config"])
    identity = source_identity(config["data"]["cache_path"])
    if identity != checkpoint["source"]:
        raise ValueError("Checkpoint radar source/protocol changed")
    groups = getattr(args, "groups", None) or config.get("validation", {}).get("groups", list(MASK_GROUPS))
    if len(set(groups)) != len(groups) or any(group not in MASK_GROUPS for group in groups):
        raise ValueError("Evaluation groups must be distinct supported mask groups")
    config["validation"] = {"groups": groups, "seed": config.get("validation", {}).get(
        "seed", config["train"]["seed"] + 100000)}
    dataset = RTSequenceDataset(config["data"], "val", args.labels, getattr(args, "limit", None))
    dataset = _with_teacher_cache(dataset, args, config)
    signature = read_json(Path(args.labels) / "manifest.json")["signature"]
    if signature != checkpoint["label_signature"]:
        raise ValueError("Evaluation joint-supervision provenance changed")
    device = choose_device(args.device)
    seed_everything(config["train"]["seed"])
    model = RTCompletionNet(config["data"], config["model"]).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    batches = loader(dataset, args.batch_size, args.workers, device=device)
    metrics = run_epoch(model, batches, config, device, checkpoint["epoch"],
                        max_steps=getattr(args, "max_steps", None))
    report = {"checkpoint": str(Path(args.checkpoint).resolve()),
              "checkpoint_sha256": sha256_file(args.checkpoint), "epoch": checkpoint["epoch"] + 1,
              "epoch_complete": checkpoint.get("epoch_complete", True),
              "split": "val", "source": identity, "validation": config["validation"],
              "debug_limits": {"frames": getattr(args, "limit", None),
                               "steps": getattr(args, "max_steps", None)},
              "training_debug_limits": checkpoint.get("debug_limits"),
              "calibration_scope": "synthetically held-out measured returns; not natural unknown voxels",
              "metrics": metrics}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    print(f"Fixed-mask validation report: {output}", flush=True)
    return report


def export(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("Not an offline RT completion checkpoint")
    if not checkpoint.get("epoch_complete", True):
        raise ValueError("Incomplete training checkpoint: resume progress.pt first; export a validated best.pt/last.pt")
    config = checkpoint["config"]
    if any(value is not None for value in checkpoint["debug_limits"].values()):
        print("WARNING: exporting a limited/debug training checkpoint; do not interpret it as a validated model", flush=True)
    data = config["data"]
    settings = dict(config["export"])
    if args.probability_threshold is not None:
        settings["probability_threshold"] = args.probability_threshold
    if args.max_added_voxels is not None:
        settings["max_added_voxels"] = args.max_added_voxels
    if not 0 <= settings["probability_threshold"] <= 1 or settings["max_added_voxels"] < 0:
        raise ValueError("Invalid export thresholds")
    identity = source_identity(data["cache_path"])
    if identity != checkpoint["source"]:
        raise ValueError("Checkpoint was trained with a different/modified radar source or split file")
    device = choose_device(args.device)
    model = RTCompletionNet(data, config["model"]).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    signature = {"source": identity, "data": data, "settings": settings,
                 "checkpoint_sha256": sha256_file(args.checkpoint), "geometry": "predicted_from_rt_only"}
    splits = load_splits(data)
    with output_lock(args.output, data["cache_path"]) as output:
        manifest = init_manifest(output, "rt_additions", signature, args.resume)
        manifest["training"] = {"checkpoint": str(Path(args.checkpoint).resolve()), "epoch": checkpoint["epoch"] + 1,
                                "train_frames": checkpoint["train_frames"], "debug_limits": checkpoint["debug_limits"]}
        manifest["state"] = "writing"
        atomic_json(output / "manifest.json", manifest)
        writer = SparseWriter(output / "additions.lmdb", settings["map_size_gb"])
        try:
            existing = writer.keys()
            for split in args.splits:
                dataset = RTSequenceDataset(data, split, labels_path=None, limit=args.limit)
                # Keep the full allowed set for context even when targets are
                # restricted for a dry run or removed by resumable export.
                dataset.indicators = [key for key in dataset.indicators if frame_key(key) not in existing]
                batches = loader(dataset, args.batch_size, args.workers, device=device)
                print(f"Export {split}: {len(dataset)} pending frames; GT/SMPL-X disabled", flush=True)
                done_now, added_now = 0, 0
                with torch.inference_mode():
                    for batch in batches:
                        raw = batch["rt"].to(device, non_blocking=True)
                        history = batch["history_valid"].to(device, non_blocking=True)
                        prediction = model(raw, history_valid=history)
                        if not finite_prediction(prediction):
                            raise FloatingPointError("Non-finite completion prediction")
                        records = prediction_to_records(model, raw, prediction, settings)
                        pairs = list(zip(batch["indicator"].tolist(), records))
                        writer.put_many(pairs)
                        existing.update(frame_key(key) for key, _ in pairs)
                        done_now += len(records)
                        added_now += sum(len(record["indices"]) for record in records)
                        if done_now == len(records) or done_now % (args.batch_size * 100) == 0:
                            print(f"  {split}: {done_now}/{len(dataset)}, mean additions={added_now / done_now:.1f}", flush=True)
                done = sum(frame_key(key) in existing for key in splits[split])
                manifest["partitions"][split] = {"expected": len(splits[split]), "written": done,
                                                  "complete": done == len(splits[split])}
                atomic_json(output / "manifest.json", manifest)
            manifest["state"] = "complete" if all(manifest["partitions"].get(s, {}).get("complete", False)
                                                   for s in ("train", "val", "test")) else "partial"
            atomic_json(output / "manifest.json", manifest)
        finally:
            writer.close()
    print(f"Exported sparse additions: {output}; state={manifest['state']}", flush=True)


def write_m4human_config(args):
    manifest = read_json(Path(args.cache) / "manifest.json")
    if manifest.get("kind") != "rt_additions" or manifest.get("state") != "complete":
        raise ValueError("Complete all three export splits before making a training config")
    if Path(args.output).exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    with open(args.base, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    data = manifest["signature"]["data"]
    config["paths"]["cached_root"] = str(Path(data["cache_path"]).parent)
    config["paths"]["rt_completion_cache"] = str(Path(args.cache).resolve())
    config["dataset"].update(scale=data["scale"], split=data["protocol"])
    config["model"].update(name="RT-Mesh", modality="rawImage_XYZ")
    config["train"].update(resume_path="", resume_start_epoch=0)
    config["eval"].update(test_mode=False, test_model_path="")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    print(f"New RT-Mesh configuration: {args.output}; original configuration unchanged", flush=True)
