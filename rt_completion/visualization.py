"""Read-only, fixed-mask RT completion visualizations and sample diagnostics."""

import copy
from pathlib import Path

import numpy as np
import torch

from . import FORMAT_VERSION
from .data import RTSequenceDataset
from .evaluation import MASK_GROUPS, fixed_corruption
from .losses import ALL_MASK_GROUPS
from .model import RTCompletionNet
from .selection import select_additions
from .storage import atomic_json, read_json, sha256_file, source_identity


PROJECTIONS = ((2, "XY"), (1, "XZ"), (0, "YZ"))


def _safe_mean(values):
    return float(values.float().mean()) if values.numel() else None


def _project(volume, axis):
    return np.max(volume, axis=axis).T


def _plot_grid(path, volumes, bounds, title, cmaps=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmaps = cmaps or ["turbo"] * len(volumes)
    figure, axes = plt.subplots(3, len(volumes), figsize=(4.3 * len(volumes), 11),
                                constrained_layout=True)
    if len(volumes) == 1:
        axes = axes[:, None]
    extents = ([bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]],
               [bounds[0][0], bounds[0][1], bounds[2][0], bounds[2][1]],
               [bounds[1][0], bounds[1][1], bounds[2][0], bounds[2][1]])
    labels = (("x (m)", "y (m)"), ("x (m)", "z (m)"), ("y (m)", "z (m)"))
    for row, (axis, projection_name) in enumerate(PROJECTIONS):
        for column, (name, volume) in enumerate(volumes):
            image = _project(volume, axis)
            finite = image[np.isfinite(image)]
            positive = finite[finite > 0]
            vmax = float(np.percentile(positive, 99.5)) if positive.size else 1.
            rendered = axes[row, column].imshow(
                image, origin="lower", extent=extents[row], aspect="auto",
                cmap=cmaps[column], vmin=0., vmax=max(vmax, 1e-8))
            axes[row, column].set_title(f"{name} / {projection_name}")
            axes[row, column].set_xlabel(labels[row][0])
            axes[row, column].set_ylabel(labels[row][1])
            figure.colorbar(rendered, ax=axes[row, column], fraction=.046, pad=.03)
    figure.suptitle(title, fontsize=14)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _scatter_values(volume, bounds, mask=None, max_points=30000):
    """Return physical XYZ coordinates and values for a sparse 3-D field."""
    volume = np.asarray(volume)
    selected = np.asarray(mask, dtype=bool) if mask is not None else volume > 0
    indices = np.argwhere(selected)
    values = volume[tuple(indices.T)] if len(indices) else np.empty(0, np.float32)
    if len(indices) > max_points:
        # Preserve the strongest returns deterministically for readable plots.
        keep = np.argpartition(values, -max_points)[-max_points:]
        indices, values = indices[keep], values[keep]
    spans = np.asarray([high - low for low, high in bounds], dtype=np.float32)
    lows = np.asarray([low for low, _ in bounds], dtype=np.float32)
    denominators = np.maximum(np.asarray(volume.shape, dtype=np.float32) - 1, 1)
    coordinates = lows + indices.astype(np.float32) * spans / denominators
    return coordinates, values.astype(np.float32, copy=False)


def _format_3d_axis(axis, bounds, title):
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.set_xlim(*bounds[0])
    axis.set_ylim(*bounds[1])
    axis.set_zlim(*bounds[2])
    axis.set_box_aspect([high - low for low, high in bounds])
    axis.view_init(elev=22, azim=-58)


def _scatter_field(axis, volume, bounds, title, *, mask=None, cmap="turbo",
                   fixed_color=None, alpha=.75, size_scale=22):
    coordinates, values = _scatter_values(volume, bounds, mask=mask)
    if len(values):
        maximum = max(float(np.percentile(values, 99)), 1e-8)
        sizes = 3 + size_scale * np.clip(values / maximum, 0, 1)
        if fixed_color is None:
            points = axis.scatter(*coordinates.T, c=values, cmap=cmap, s=sizes,
                                  alpha=alpha, linewidths=0, depthshade=True)
        else:
            points = axis.scatter(*coordinates.T, color=fixed_color, s=sizes,
                                  alpha=alpha, linewidths=0, depthshade=True)
    else:
        points = None
    _format_3d_axis(axis, bounds, f"{title}\n{len(values)} voxels")
    return points


def _plot_teacher_3d(path, current, teacher, bounds, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    intensity = teacher["intensity"][0].numpy()
    valid = teacher["valid_mask"][0].numpy()
    confidence = teacher["confidence"][0].numpy()
    support = teacher["support_count"][0].float().numpy()
    current = current[0].numpy()
    figure = plt.figure(figsize=(18, 11), constrained_layout=True)
    axes = [figure.add_subplot(2, 3, index + 1, projection="3d") for index in range(6)]
    _scatter_field(axes[0], current, bounds, "Current measured RT")
    _scatter_field(axes[1], intensity, bounds, "Pseudo-GT intensity", mask=valid)
    _scatter_field(axes[2], valid.astype(np.float32), bounds, "Valid supervision mask",
                   mask=valid, cmap="gray", size_scale=9)
    _scatter_field(axes[3], confidence, bounds, "Confidence weight", mask=valid,
                   cmap="viridis", size_scale=15)
    _scatter_field(axes[4], support, bounds, "Historical support count",
                   mask=support > 0, cmap="cividis", size_scale=12)
    turbo = plt.colormaps["turbo"]
    current_color, teacher_color = turbo(.12), turbo(.82)
    _scatter_field(axes[5], current, bounds, "Current RT + pseudo-GT overlay",
                   fixed_color=current_color, alpha=.28, size_scale=13)
    _scatter_field(axes[5], intensity, bounds, "Current RT + pseudo-GT overlay",
                   mask=valid, fixed_color=teacher_color, alpha=.9, size_scale=22)
    axes[5].legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=current_color,
               markersize=7, label="Current measured RT"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=teacher_color,
               markersize=7, label="Temporal pseudo-GT"),
    ], loc="upper right", fontsize=8)
    figure.suptitle(title + "\nPhysical 3-D voxel coordinates", fontsize=14)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_history_3d(path, raw, bounds, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(19, 5.5), constrained_layout=True)
    names = ("History t-3", "History t-2", "History t-1", "Current t")
    for index, name in enumerate(names):
        axis = figure.add_subplot(1, 4, index + 1, projection="3d")
        _scatter_field(axis, raw[index].numpy(), bounds, name + " measured RT")
    figure.suptitle(title + "\nMeasured source frames before articulated alignment", fontsize=14)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_completion_3d(path, original, additions, accepted, completed, bounds, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure = plt.figure(figsize=(19, 5.8), constrained_layout=True)
    axes = [figure.add_subplot(1, 4, index + 1, projection="3d") for index in range(4)]
    _scatter_field(axes[0], original, bounds, "Before: original measured RT")
    _scatter_field(axes[1], additions, bounds, "Accepted predicted additions",
                   mask=accepted, cmap="turbo", alpha=.9, size_scale=24)
    _scatter_field(axes[2], completed, bounds, "After: completed RT",
                   cmap="turbo", alpha=.88, size_scale=24)
    turbo = plt.colormaps["turbo"]
    original_color, addition_color = turbo(.12), turbo(.82)
    _scatter_field(axes[3], original, bounds, "Before/after overlay",
                   fixed_color=original_color, alpha=.25, size_scale=12)
    _scatter_field(axes[3], additions, bounds, "Before/after overlay",
                   mask=accepted, fixed_color=addition_color, alpha=.9, size_scale=23)
    axes[3].legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=original_color,
               markersize=7, label="Original measured RT"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=addition_color,
               markersize=7, label="Accepted additions"),
    ], loc="upper right", fontsize=8)
    figure.suptitle(title + "\nRT-only deployment inference; no artificial mask or SMPL-X input",
                    fontsize=14)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _compact_validation(checkpoint):
    validation = checkpoint.get("metrics", {}).get("val", {})
    result = {key: validation.get(key) for key in
              ("loss", "masked_mae_log", "joint_mpjpe_m", "selection_aggregation")
              if key in validation}
    result["groups"] = {}
    for name, values in validation.get("groups", {}).items():
        result["groups"][name] = {key: values.get(key) for key in (
            "masked_mae_log", "masked_coverage68", "masked_coverage95",
            "masked_sigma_log", "accepted_heldout_coverage",
            "accepted_heldout_mae_log", "total_masked_voxels")}
    return result


def _plot_metric_summaries(output, validation, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = list(validation.get("groups", {}))
    if groups:
        x = np.arange(len(groups))
        masked = [validation["groups"][name]["masked_mae_log"] for name in groups]
        accepted = [validation["groups"][name]["accepted_heldout_mae_log"] for name in groups]
        coverage = [validation["groups"][name]["accepted_heldout_coverage"] for name in groups]
        cover68 = [validation["groups"][name]["masked_coverage68"] for name in groups]
        cover95 = [validation["groups"][name]["masked_coverage95"] for name in groups]
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
        width = .36
        turbo = plt.colormaps["turbo"]
        axes[0].bar(x - width / 2, masked, width, color=turbo(.18), label="Prediction mean MAE")
        axes[0].bar(x + width / 2, accepted, width, color=turbo(.78), label="Accepted-only MAE")
        axes[0].set_ylabel("MAE in log-intensity space (lower is better)")
        axes[0].legend()
        axes[1].bar(x - width, coverage, width, color=turbo(.08), label="Accepted held-out coverage")
        axes[1].bar(x, cover68, width, color=turbo(.48), label="Uncertainty coverage 68%")
        axes[1].bar(x + width, cover95, width, color=turbo(.88), label="Uncertainty coverage 95%")
        axes[1].set_ylabel("Fraction")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(fontsize=8)
        for axis in axes:
            axis.set_xticks(x, groups, rotation=18)
            axis.grid(axis="y", alpha=.25)
        figure.suptitle("Full validation metrics stored in checkpoint")
        figure.savefig(output / "full_validation_metrics.png", dpi=160)
        plt.close(figure)

    labels = [f"{item['dataset_index']}\n{item['mask_group']}" for item in records]
    x = np.arange(len(records))
    zero = [item["zero_fill_mae_log"] for item in records]
    prediction = [item["prediction_mean_mae_log"] for item in records]
    accepted_zero = [item["accepted_zero_fill_mae_log"] for item in records]
    accepted = [item["accepted_completion_mae_log"] for item in records]
    coverage = [item["accepted_heldout_coverage"] for item in records]
    figure, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    width = .36
    turbo = plt.colormaps["turbo"]
    axes[0].bar(x - width / 2, zero, width, color=turbo(.18), label="Zero fill")
    axes[0].bar(x + width / 2, prediction, width, color=turbo(.78), label="Prediction mean")
    axes[0].set_title("All held-out measured voxels")
    axes[1].bar(x - width / 2, accepted_zero, width, color=turbo(.18), label="Zero fill")
    axes[1].bar(x + width / 2, accepted, width, color=turbo(.78), label="Accepted completion")
    axes[1].set_title("Accepted held-out subset")
    axes[2].bar(x, coverage, color=turbo(.52))
    axes[2].set_title("Accepted held-out coverage")
    axes[2].set_ylim(0, 1.05)
    for axis in axes:
        axis.set_xticks(x, labels, rotation=18)
        axis.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("MAE in log-intensity space")
    axes[1].set_ylabel("MAE in log-intensity space")
    axes[2].set_ylabel("Fraction")
    axes[0].legend()
    axes[1].legend()
    figure.suptitle("Selected-sample completion versus zero-fill baseline")
    figure.savefig(output / "sample_metric_comparison.png", dpi=160)
    plt.close(figure)


def generate_visualizations(args):
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty visualization directory {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("Not an offline RT completion checkpoint")
    if not checkpoint.get("epoch_complete", True):
        raise ValueError("Use best.pt/last.pt from a completed epoch, not an in-epoch progress.pt")
    config = copy.deepcopy(checkpoint["config"])
    identity = source_identity(config["data"]["cache_path"])
    if identity != checkpoint["source"]:
        raise ValueError("Checkpoint radar source/protocol changed")
    label_signature = read_json(Path(args.labels) / "manifest.json")["signature"]
    if label_signature != checkpoint["label_signature"]:
        raise ValueError("Joint-label provenance changed")
    groups = list(args.groups)
    if len(set(groups)) != len(groups) or any(group not in ALL_MASK_GROUPS for group in groups):
        raise ValueError("Visualization groups must be distinct supported groups")
    dataset = RTSequenceDataset(config["data"], args.split, args.labels)
    for index in args.indices:
        if index >= len(dataset):
            raise IndexError(f"Dataset index {index} is outside {args.split} length {len(dataset)}")
    from .pipeline import choose_device, finite_prediction, seed_everything
    device = choose_device(args.device)
    seed_everything(config["train"]["seed"])
    model = RTCompletionNet(config["data"], config["model"]).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    seed = config.get("validation", {}).get("seed", config["train"]["seed"] + 100000)
    records = []
    for index in args.indices:
        item = dataset[index]
        raw = item["rt"].unsqueeze(0).to(device)
        joints = item["joints"].unsqueeze(0).to(device)
        history = item["history_valid"].unsqueeze(0).to(device)
        indicator = item["indicator"].unsqueeze(0)
        for group in groups:
            corrupted, masked, info = fixed_corruption(
                raw, joints, indicator, config["mask"], config["data"]["bounds"], seed, group)
            with torch.inference_mode():
                prediction = model(corrupted, history_valid=history)
                if not finite_prediction(prediction):
                    raise FloatingPointError("Non-finite completion prediction")
                selected = select_additions(model, corrupted, prediction, config["export"])
                current = raw[:, -1:]
                damaged = corrupted[:, -1:]
                completed = damaged.clone()
                completed[selected["accepted"]] = selected["values"][selected["accepted"]]
                target_log = model.normalize(current)
                damaged_error = (model.normalize(damaged) - target_log).abs()
                predicted_error = (prediction["intensity"] - target_log).abs()
                completed_error = (model.normalize(completed) - target_log).abs()
                accepted_heldout = selected["accepted"] & masked
                heldout = current * masked
                additions = selected["values"] * selected["accepted"]
                joint_mpjpe = (prediction["joints"] - joints[:, -1]).norm(dim=-1).mean()
            stem = f"{args.split}_{index:06d}_{group}"
            arrays = {
                "original": current[0, 0].cpu().numpy(),
                "masked_input": damaged[0, 0].cpu().numpy(),
                "heldout_measured": heldout[0, 0].cpu().numpy(),
                "accepted_additions": additions[0, 0].cpu().numpy(),
                "completed": completed[0, 0].cpu().numpy(),
                "heldout_error_log": (completed_error * masked)[0, 0].cpu().numpy(),
                "reflection_score": prediction["reflection_logits"].sigmoid()[0, 0].cpu().numpy(),
                "uncertainty_sigma_log": selected["sigma"][0, 0].cpu().numpy(),
                "accepted_mask": selected["accepted"][0, 0].float().cpu().numpy(),
            }
            heading = f"indicator={item['indicator'].tolist()}  mask={group}  epoch={checkpoint['epoch'] + 1}"
            _plot_grid(output / f"{stem}_comparison.png", [
                ("Original measured RT", arrays["original"]),
                ("Artificially masked input", arrays["masked_input"]),
                ("Held-out measured returns", arrays["heldout_measured"]),
                ("Accepted predicted additions", arrays["accepted_additions"]),
                ("Completed RT", arrays["completed"]),
            ], config["data"]["bounds"], heading)
            _plot_grid(output / f"{stem}_diagnostics.png", [
                ("Held-out abs error (log)", arrays["heldout_error_log"]),
                ("Reflection score", arrays["reflection_score"]),
                ("Uncertainty sigma (log)", arrays["uncertainty_sigma_log"]),
                ("Accepted voxel mask", arrays["accepted_mask"]),
            ], config["data"]["bounds"], heading,
                cmaps=["turbo"] * 4)
            number = int(masked.sum())
            accepted_number = int(accepted_heldout.sum())
            accepted_baseline = _safe_mean(damaged_error[accepted_heldout])
            accepted_mae = _safe_mean(completed_error[accepted_heldout])
            record = {
                "split": args.split, "dataset_index": index,
                "indicator": item["indicator"].tolist(), "mask_group": group,
                "mask_info": {key: value.tolist() for key, value in info.items()
                              if torch.is_tensor(value)},
                "heldout_measured_voxels": number,
                "zero_fill_mae_log": _safe_mean(damaged_error[masked]),
                "prediction_mean_mae_log": _safe_mean(predicted_error[masked]),
                "accepted_heldout_voxels": accepted_number,
                "accepted_heldout_coverage": accepted_number / number if number else None,
                "accepted_zero_fill_mae_log": accepted_baseline,
                "accepted_completion_mae_log": accepted_mae,
                "accepted_improvement_fraction": (
                    1. - accepted_mae / accepted_baseline
                    if accepted_mae is not None and accepted_baseline else None),
                "accepted_additions_total": int(selected["accepted"].sum()),
                "accepted_additions_at_natural_zeros": int((selected["accepted"] & ~masked).sum()),
                "joint_mpjpe_m": float(joint_mpjpe),
                "comparison_png": f"{stem}_comparison.png",
                "diagnostics_png": f"{stem}_diagnostics.png",
            }
            records.append(record)
            atomic_json(output / f"{stem}_metrics.json", record)
            print(f"Visualized {stem}: heldout={number}, accepted={accepted_number}", flush=True)
    report = {
        "scope": "fixed selected samples; natural-zero additions have no measured ground truth",
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
        "epoch": checkpoint["epoch"] + 1, "split": args.split,
        "indices": list(args.indices), "groups": groups,
        "full_validation_from_checkpoint": _compact_validation(checkpoint),
        "sample_diagnostics": records,
    }
    _plot_metric_summaries(output, report["full_validation_from_checkpoint"], records)
    atomic_json(output / "summary.json", report)
    print(f"Visualization report: {output / 'summary.json'}", flush=True)
    return report


def _teacher_statistics(teacher, current):
    valid = teacher["valid_mask"].bool()
    intensity = teacher["intensity"].float()
    confidence = teacher["confidence"].float()
    support = teacher["support_count"].long()
    valid_count = int(valid.sum())
    values = intensity[valid]
    weights = confidence[valid]
    counts = support[valid]
    histogram = {str(number): int((counts == number).sum())
                 for number in sorted(set(counts.tolist()))}
    return {
        "grid_voxels": int(current.numel()),
        "current_observed_voxels": int((current > 0).sum()),
        "teacher_valid_voxels": valid_count,
        "teacher_valid_fraction_of_grid": valid_count / current.numel(),
        "teacher_current_observation_overlap": int((valid & (current > 0)).sum()),
        "teacher_intensity_mean": _safe_mean(values),
        "teacher_intensity_median": float(values.median()) if values.numel() else None,
        "teacher_intensity_max": float(values.max()) if values.numel() else None,
        "teacher_confidence_mean": _safe_mean(weights),
        "teacher_confidence_min": float(weights.min()) if weights.numel() else None,
        "teacher_confidence_max": float(weights.max()) if weights.numel() else None,
        "teacher_support_mean": _safe_mean(counts),
        "teacher_support_histogram": histogram,
        "voxels_with_any_historical_support": int((support > 0).sum()),
    }


def generate_teacher_visualizations(args):
    """Visualize the immutable SMPL-X-joint-aligned temporal teacher cache."""
    from .pipeline import load_config
    from .teacher_cache import TeacherCache

    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty visualization directory {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    dataset = RTSequenceDataset(config["data"], args.split, args.labels)
    cache = TeacherCache(args.teacher_cache, config, args.labels, args.split)
    for index in args.indices:
        if index >= len(dataset):
            raise IndexError(f"Dataset index {index} is outside {args.split} length {len(dataset)}")
    records = []
    for index in args.indices:
        item = dataset[index]
        teacher = cache.get(item["indicator"])
        raw = item["rt"].float()
        current = raw[-1:].clone()
        valid = teacher["valid_mask"]
        # This sum is explanatory only. The temporal teacher is a training
        # target and is never itself exported as completed RT.
        illustrative = current + teacher["intensity"]
        stem = f"{args.split}_{index:06d}_temporal_teacher"
        indicator = item["indicator"].tolist()
        title = f"indicator={indicator}  articulated temporal pseudo-GT (training only)"
        _plot_grid(output / f"{stem}_sources.png", [
            ("History t-3 measured RT", raw[0].numpy()),
            ("History t-2 measured RT", raw[1].numpy()),
            ("History t-1 measured RT", raw[2].numpy()),
            ("Current t measured RT", raw[3].numpy()),
        ], config["data"]["bounds"], title)
        _plot_grid(output / f"{stem}_fields.png", [
            ("Current measured RT", current[0].numpy()),
            ("Pseudo-GT intensity", teacher["intensity"][0].numpy()),
            ("Valid supervision mask", valid[0].float().numpy()),
            ("Confidence weight", teacher["confidence"][0].numpy()),
            ("Historical support count", teacher["support_count"][0].float().numpy()),
            ("Current + pseudo-GT (illustrative)", illustrative[0].numpy()),
        ], config["data"]["bounds"], title,
            cmaps=["turbo"] * 6)
        _plot_history_3d(output / f"{stem}_sources_3d.png", raw,
                         config["data"]["bounds"], title)
        _plot_teacher_3d(output / f"{stem}_fields_3d.png", current, teacher,
                         config["data"]["bounds"], title)
        statistics = _teacher_statistics(teacher, current)
        record = {
            "split": args.split,
            "dataset_index": index,
            "indicator": indicator,
            "history_valid": item["history_valid"].tolist(),
            "history_observed_voxels": [int((frame > 0).sum()) for frame in raw[:-1]],
            **statistics,
            "sources_png": f"{stem}_sources.png",
            "fields_png": f"{stem}_fields.png",
            "sources_3d_png": f"{stem}_sources_3d.png",
            "fields_3d_png": f"{stem}_fields_3d.png",
            "interpretation": (
                "Pseudo-GT is training-only historical measured RT aligned with SMPL-X-derived "
                "joints. Current+pseudo-GT is illustrative and is not an exported completion."),
        }
        records.append(record)
        atomic_json(output / f"{stem}_metrics.json", record)
        print(f"Visualized {stem}: valid={statistics['teacher_valid_voxels']}, "
              f"confidence={statistics['teacher_confidence_mean']}", flush=True)
    report = {
        "scope": "training-only articulated temporal pseudo-GT; not network output or test GT",
        "config": str(Path(args.config).resolve()),
        "labels": str(Path(args.labels).resolve()),
        "teacher_cache": str(Path(args.teacher_cache).resolve()),
        "teacher_manifest": cache.manifest,
        "split": args.split,
        "indices": list(args.indices),
        "samples": records,
    }
    atomic_json(output / "summary.json", report)
    print(f"Temporal-teacher visualization: {output / 'summary.json'}", flush=True)
    return report


def generate_completion_visualizations(args):
    """Visualize actual RT-only completion without synthetic corruption or GT."""
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty visualization directory {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("Not an offline RT completion checkpoint")
    if not checkpoint.get("epoch_complete", True):
        raise ValueError("Use best.pt/last.pt from a completed epoch, not progress.pt")
    config = copy.deepcopy(checkpoint["config"])
    identity = source_identity(config["data"]["cache_path"])
    if identity != checkpoint["source"]:
        raise ValueError("Checkpoint radar source/protocol changed")
    dataset = RTSequenceDataset(config["data"], args.split, labels_path=None)
    for index in args.indices:
        if index >= len(dataset):
            raise IndexError(f"Dataset index {index} is outside {args.split} length {len(dataset)}")
    from .pipeline import choose_device, finite_prediction, seed_everything
    device = choose_device(args.device)
    seed_everything(config["train"]["seed"])
    model = RTCompletionNet(config["data"], config["model"]).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    records = []
    for index in args.indices:
        item = dataset[index]
        raw = item["rt"].unsqueeze(0).to(device)
        history = item["history_valid"].unsqueeze(0).to(device)
        with torch.inference_mode():
            prediction = model(raw, history_valid=history)
            if not finite_prediction(prediction):
                raise FloatingPointError("Non-finite completion prediction")
            selected = select_additions(model, raw, prediction, config["export"])
            original = raw[:, -1:].clone()
            completed = original.clone()
            completed[selected["accepted"]] = selected["values"][selected["accepted"]]
            additions = selected["values"] * selected["accepted"]
        indicator = item["indicator"].tolist()
        stem = f"{args.split}_{index:06d}_completion"
        title = f"indicator={indicator}  checkpoint epoch={checkpoint['epoch'] + 1}"
        arrays = {
            "original": original[0, 0].cpu().numpy(),
            "additions": additions[0, 0].cpu().numpy(),
            "accepted": selected["accepted"][0, 0].cpu().numpy(),
            "completed": completed[0, 0].cpu().numpy(),
        }
        _plot_completion_3d(output / f"{stem}_3d.png", arrays["original"],
                            arrays["additions"], arrays["accepted"], arrays["completed"],
                            config["data"]["bounds"], title)
        _plot_grid(output / f"{stem}_projections.png", [
            ("Before: original measured RT", arrays["original"]),
            ("Accepted predicted additions", arrays["additions"]),
            ("After: completed RT", arrays["completed"]),
        ], config["data"]["bounds"], title)
        accepted = selected["accepted"]
        number = int(accepted.sum())
        accepted_values = selected["values"][accepted]
        accepted_scores = selected["probability"][accepted]
        accepted_sigma = selected["sigma"][accepted]
        original_count = int((original > 0).sum())
        completed_count = int((completed > 0).sum())
        unchanged = bool(torch.equal(completed[original > 0], original[original > 0]))
        record = {
            "split": args.split,
            "dataset_index": index,
            "indicator": indicator,
            "history_valid": item["history_valid"].tolist(),
            "original_observed_voxels": original_count,
            "accepted_additions": number,
            "completed_observed_voxels": completed_count,
            "addition_fraction_of_completed": number / completed_count if completed_count else None,
            "original_observations_unchanged": unchanged,
            "accepted_intensity_mean": _safe_mean(accepted_values),
            "accepted_intensity_max": float(accepted_values.max()) if number else None,
            "accepted_reflection_score_mean": _safe_mean(accepted_scores),
            "accepted_uncertainty_sigma_mean": _safe_mean(accepted_sigma),
            "max_added_voxels": int(config["export"]["max_added_voxels"]),
            "budget_saturated": number == int(config["export"]["max_added_voxels"]),
            "comparison_3d_png": f"{stem}_3d.png",
            "projections_png": f"{stem}_projections.png",
            "ground_truth_note": (
                "Natural-zero additions have no complete RT ground truth; this is deployment-style "
                "structure/quantity visualization, not recovery-error evaluation."),
        }
        records.append(record)
        atomic_json(output / f"{stem}_metrics.json", record)
        print(f"Visualized {stem}: original={original_count}, additions={number}, "
              f"completed={completed_count}", flush=True)
    report = {
        "scope": "RT-only deployment-style completion without artificial mask, labels, or SMPL-X",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "epoch": checkpoint["epoch"] + 1,
        "source": identity,
        "split": args.split,
        "indices": list(args.indices),
        "samples": records,
    }
    atomic_json(output / "summary.json", report)
    print(f"Completion before/after visualization: {output / 'summary.json'}", flush=True)
    return report
