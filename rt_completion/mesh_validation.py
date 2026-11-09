"""Frozen RT-Mesh validation probe; deliberately never opens the test split."""

import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import yaml

from dataset.m4human_dataset import RF3DPoseDataset, ToTensor
from mmwave_models.Tensor_models.RTmesh.tpn_mul_attn5 import (
    Simple3DConvModelWithTripleCNNFPNAndAttention,
)
from sources.Train_and_model_loss import batch_rodrigues, compute_joints_from_vertices

from .storage import atomic_json, read_json, sha256_file


def _load_state(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = ("fpn.attention_pos", "fpn_m.attention_pos")
    bad_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(allowed_missing)
    ]
    if bad_missing or incompatible.unexpected_keys:
        raise ValueError(
            f"RT-Mesh checkpoint mismatch: missing={bad_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def _rotation_error(prediction, target):
    pred = batch_rodrigues(prediction.reshape(-1, 3)).reshape(prediction.shape[0], -1, 3, 3)
    truth = batch_rodrigues(target.reshape(-1, 3)).reshape(target.shape[0], -1, 3, 3)
    cosine = ((pred * truth).sum(dim=(-1, -2)) - 1) / 2
    return torch.rad2deg(torch.acos(cosine.clamp(-1, 1))).mean(-1)


def summarize_action_sums(sums, counts):
    total = int(counts.sum())
    if total == 0:
        raise ValueError("Validation subset contains no samples")
    overall = {name: float(values.sum() / total) for name, values in sums.items()}
    per_action = {}
    for action in range(50):
        if counts[action]:
            per_action[str(action + 1)] = {
                "frames": int(counts[action]),
                **{name: float(values[action] / counts[action]) for name, values in sums.items()},
            }
    return overall, per_action


def evaluate_mesh_val(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite validation report {output}")
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["model"]["name"] != "RT-Mesh" or config["model"]["modality"] != "rawImage_XYZ":
        raise ValueError("Frozen validation currently supports RT-Mesh/rawImage_XYZ only")

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu only for tiny smoke tests")

    cache = str(Path(args.cache).resolve()) if args.cache else None
    smplx_file = Path(config["paths"]["smplx"]["neutral"])
    if not smplx_file.is_absolute():
        smplx_file = Path.cwd() / smplx_file
    # smplx.create expects the directory containing the ``smplx`` folder,
    # whereas RT-Mesh itself receives the three explicit model files.
    smplx_model_root = smplx_file.resolve().parent.parent
    dataset = RF3DPoseDataset(
        [], cache_dir=config["paths"]["cached_root"], split="val",
        transform=transforms.Compose([ToTensor()]), load_save=True,
        smplx_model_path=str(smplx_model_root),
        main_modality="rawImage_XYZ", scale_id=config["dataset"]["scale"],
        split_id=config["dataset"]["split"], rt_completion_cache=cache,
        return_temporal_labels=False,
    )
    subset_indices = None
    if args.limit is not None and args.limit < len(dataset):
        generator = np.random.default_rng(seed)
        subset_indices = np.sort(generator.choice(len(dataset), args.limit, replace=False))
        dataset = Subset(dataset, subset_indices.tolist())
    workers = int(args.workers)
    loader_options = {"persistent_workers": True, "prefetch_factor": 1} if workers else {}
    batches = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda", **loader_options,
    )

    model = Simple3DConvModelWithTripleCNNFPNAndAttention(
        config["paths"]["smplx"], kinematic_refinement=False,
    ).to(device).eval()
    _load_state(model, args.checkpoint)
    metric_names = ("mve_mm", "joint_location_mm", "joint_rotation_deg",
                    "mesh_location_mm", "gender_accuracy")
    sums = {name: np.zeros(50, dtype=np.float64) for name in metric_names}
    counts = np.zeros(50, dtype=np.int64)
    started = time.monotonic()

    with torch.inference_mode():
        for step, batch in enumerate(batches, 1):
            radar = batch["rawImage_XYZ"].to(device, non_blocking=True)
            ground = batch["parameter"]
            betas = ground["betas"][:, :10].to(device, non_blocking=True)
            pose = ground["pose_body"].to(device, non_blocking=True)
            root = ground["root_orient"].to(device, non_blocking=True)
            translation = ground["trans"].to(device, non_blocking=True)
            gender = ground["gender"].to(device, non_blocking=True)
            pred_betas, pred_pose, pred_root, pred_translation, pred_gender, _ = model(radar)
            pred_vertices = model.get_smplx_output(
                pred_betas, pred_pose, pred_root, pred_translation, pred_gender,
            ).squeeze(1)
            true_vertices = model.get_smplx_output(
                betas, pose, root, translation, gender,
            ).squeeze(1)
            pred_joints = compute_joints_from_vertices(model, pred_vertices, gender)
            true_joints = compute_joints_from_vertices(model, true_vertices, gender)
            values = {
                "mve_mm": torch.linalg.vector_norm(pred_vertices - true_vertices, dim=-1).mean(-1),
                "joint_location_mm": torch.linalg.vector_norm(
                    pred_joints - true_joints, dim=-1).mean(-1),
                "joint_rotation_deg": _rotation_error(pred_pose, pose),
                "mesh_location_mm": torch.linalg.vector_norm(
                    pred_translation - translation, dim=-1) * 1000,
                "gender_accuracy": (pred_gender > .5).eq(gender > .5).float().reshape(-1),
            }
            actions = batch["indicator"][:, 1].cpu().numpy().astype(np.int64) - 1
            for index, action in enumerate(actions):
                counts[action] += 1
                for name in metric_names:
                    sums[name][action] += float(values[name][index])
            if step == 1 or step % 50 == 0 or step == len(batches):
                done = int(counts.sum())
                elapsed = time.monotonic() - started
                print(f"Validation: {done}/{len(dataset)}, {done / max(elapsed, 1e-6):.1f} frame/s", flush=True)

    overall, per_action = summarize_action_sums(sums, counts)
    subset_hash = None
    if subset_indices is not None:
        subset_hash = hashlib.sha256(subset_indices.astype(np.int64).tobytes()).hexdigest()
    cache_manifest = read_json(Path(cache) / "manifest.json") if cache else None
    report = {
        "kind": "frozen_rtmesh_validation_probe",
        "split": "val",
        "test_data_accessed": False,
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "completion_cache": cache,
        "completion_signature": cache_manifest.get("signature") if cache_manifest else None,
        "frames": int(counts.sum()),
        "full_validation_frames": len(dataset.dataset) if isinstance(dataset, Subset) else len(dataset),
        "subset_seed": seed if subset_indices is not None else None,
        "subset_indices_sha256": subset_hash,
        "elapsed_seconds": time.monotonic() - started,
        "overall": overall,
        "per_action": per_action,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    print(json.dumps({"output": str(output), "frames": report["frames"], **overall}, indent=2), flush=True)
    return report

