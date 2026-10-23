#!/usr/bin/env python3
"""Inspect two M4Human sequences and render reproducible diagnostic figures.

This script intentionally reads the processed LMDB files directly.  In this
release ``rgb_feat`` is a feature vector and ``item_depth1`` is a point cloud;
neither one is a display-ready image.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import imageio.v2 as imageio
import lmdb
import matplotlib.pyplot as plt
import msgpack
import numpy as np
import smplx

from dataset.data_loader_camera_calibration import calibrate_param_to_radar
from dataset.lmdb_utils import unpack_dict_np, unpack_ndarray
from dataset.m4human_dataset import gender_info
from dataset.m4human_utils import decompress_radar


DB_NAMES = (
    "indicator", "radar_comp", "radar_pc", "params", "calib",
    "bbox", "rgb_feat", "item_depth1",
)
CORE_NAMES = ("radar_comp", "radar_pc", "params", "calib")
BOUNDS = ((-3.0, 3.0), (0.5, 6.0), (0.0, 3.0))
RADAR_SHAPE = (121, 111, 31)
BODY_EDGES = (
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7),
    (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (9, 13),
    (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19),
    (18, 20), (19, 21),
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="MR-Mesh/rf3dpose_all")
    p.add_argument(
        "--smplx-root", default="models",
        help="Model root expected by smplx.create(); it must contain the smplx/ directory",
    )
    p.add_argument("--output", default="sequence_inspection")
    p.add_argument("--seed", type=int, default=20260916)
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--sequences", nargs="*", help="Optional values such as P14-A36 P1-A49")
    return p.parse_args()


def open_envs(root: Path):
    return {
        name: lmdb.open(
            str(root / f"{name}.lmdb"), readonly=True, subdir=False,
            lock=False, readahead=False, max_readers=64,
        )
        for name in DB_NAMES
    }


def read_obj(env, key, kind):
    with env.begin() as txn:
        raw = txn.get(str(key).encode())
    if raw is None:
        return None
    return unpack_dict_np(raw) if kind == "dict" else unpack_ndarray(raw)


def finite_stats(value):
    a = np.asarray(value)
    finite = np.isfinite(a) if np.issubdtype(a.dtype, np.number) else np.ones(a.shape, bool)
    good = a[finite]
    return {
        "shape": list(a.shape), "dtype": str(a.dtype),
        "min": float(good.min()) if good.size else None,
        "max": float(good.max()) if good.size else None,
        "mean": float(good.mean()) if good.size else None,
        "nan": int(np.isnan(a).sum()) if np.issubdtype(a.dtype, np.floating) else 0,
        "inf": int(np.isinf(a).sum()) if np.issubdtype(a.dtype, np.floating) else 0,
    }


def all_sequences(env):
    groups = collections.defaultdict(list)
    with env.begin() as txn:
        for _, raw in txn.cursor():
            ind = msgpack.unpackb(raw, raw=False)
            groups[(int(ind[0]), int(ind[1]))].append([int(x) for x in ind])
    for values in groups.values():
        values.sort(key=lambda x: x[2])
    return groups


def parse_sequence(text):
    text = text.upper().replace("P", "").replace("A", "")
    s, a = text.split("-")
    return int(s), int(a)


def camera_to_radar(points, calib):
    """Treat item_depth1 as camera XYZ (metres; Z behaves as optical depth).

    The processed release does not include the feature/depth-generation script,
    so this convention is inferred from the axis ranges and calibration layout.
    """
    r = np.asarray(calib["radar_to_cam_rotmatrix"], dtype=np.float64)
    t = np.asarray(calib["radar_to_cam_tvec"], dtype=np.float64)
    return (np.linalg.inv(r) @ (np.asarray(points) - t).T).T


class SmplRenderer:
    def __init__(self, root):
        self.models = {
            sex: smplx.create(root, model_type="smplx", gender=sex)
            for sex in ("male", "female")
        }

    def mesh_and_joints(self, params, calib, subject):
        p = calibrate_param_to_radar(params, calib)
        sex = "male" if gender_info[f"P{subject}"] > 0.5 else "female"
        model = self.models[sex]
        import torch
        with torch.no_grad():
            out = model(
                betas=torch.as_tensor(p["betas"][None], dtype=torch.float32),
                body_pose=torch.as_tensor(p["pose_body"][None], dtype=torch.float32),
                global_orient=torch.as_tensor(p["root_orient"][None], dtype=torch.float32),
                transl=torch.as_tensor(p["trans"][None], dtype=torch.float32),
            )
        verts = out.vertices[0].cpu().numpy()  # metres, radar coordinates
        reg = model.J_regressor[:22]
        joints = (reg @ out.vertices[0]).cpu().numpy()
        return verts, joints, np.asarray(model.faces)


def load_frame(envs, renderer, key):
    rt_sparse = read_obj(envs["radar_comp"], key, "array")
    rpc = read_obj(envs["radar_pc"], key, "array")
    params = read_obj(envs["params"], key, "dict")
    calib = read_obj(envs["calib"], key, "dict")
    rgb_feat = read_obj(envs["rgb_feat"], key, "array")
    depth_cam = read_obj(envs["item_depth1"], key, "array")
    bbox = read_obj(envs["bbox"], key, "dict")
    if any(x is None for x in (rt_sparse, rpc, params, calib)):
        raise RuntimeError(f"Core modality missing at {key}")
    rt = decompress_radar(rt_sparse, RADAR_SHAPE)
    verts, joints, faces = renderer.mesh_and_joints(params, calib, key[0])
    depth_radar = camera_to_radar(depth_cam, calib) if depth_cam is not None else None
    return {
        "key": key, "rt_sparse": rt_sparse, "rt": rt,
        "xy": rt.max(axis=2).T, "xz": rt.max(axis=1).T,
        "yz": rt.max(axis=0).T, "rpc": rpc, "params": params,
        "calib": calib, "rgb_feat": rgb_feat, "depth_cam": depth_cam,
        "depth_radar": depth_radar, "bbox": bbox,
        "verts": verts, "joints": joints, "faces": faces,
    }


def draw_skeleton(ax, joints, color="black", linewidth=1.5):
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=9, c=color)
    for i, j in BODY_EDGES:
        ax.plot(joints[[i, j], 0], joints[[i, j], 1], joints[[i, j], 2],
                color=color, linewidth=linewidth)


def set_radar_axes(ax, title):
    ax.set_xlim(*BOUNDS[0]); ax.set_ylim(*BOUNDS[1]); ax.set_zlim(*BOUNDS[2])
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.view_init(elev=20, azim=-55)


def render_overview(frame, path, vmin, vmax):
    fig = plt.figure(figsize=(20, 15), constrained_layout=True)
    key = frame["key"]
    fig.suptitle(f"M4Human P{key[0]}-A{key[1]} frame {key[2]} (processed modalities)", fontsize=16)

    for pos, arr, title, extent in (
        (1, frame["xy"], "RT max projection: XY / BEV", (-3, 3, .5, 6)),
        (2, frame["xz"], "RT max projection: XZ", (-3, 3, 0, 3)),
        (3, frame["yz"], "RT max projection: YZ", (.5, 6, 0, 3)),
    ):
        ax = fig.add_subplot(3, 4, pos)
        im = ax.imshow(arr, origin="lower", extent=extent, aspect="auto", cmap="turbo", vmin=vmin, vmax=vmax)
        ax.set_title(title); fig.colorbar(im, ax=ax, shrink=.75, label="radar response")

    ax = fig.add_subplot(3, 4, 4, projection="3d")
    nz = np.argwhere(frame["rt"] > 0)
    if len(nz) > 5000:
        nz = nz[np.linspace(0, len(nz) - 1, 5000).astype(int)]
    xyz = np.column_stack((
        -3 + nz[:, 0] * .05, .5 + nz[:, 1] * .05, nz[:, 2] * .1,
    ))
    colors = frame["rt"][tuple(nz.T)]
    ax.scatter(*xyz.T, c=colors, s=2, cmap="turbo", vmin=vmin, vmax=vmax)
    set_radar_axes(ax, "RT non-zero voxels")

    ax = fig.add_subplot(3, 4, 5)
    vals = frame["rt"][frame["rt"] > 0]
    ax.hist(vals, bins=60, color="steelblue"); ax.set_yscale("log")
    ax.set_title("RT response histogram"); ax.set_xlabel("response")

    ax = fig.add_subplot(3, 4, 6, projection="3d")
    rpc = frame["rpc"]
    sc = ax.scatter(rpc[:, 0], rpc[:, 1], rpc[:, 2], c=rpc[:, 3], s=5, cmap="viridis")
    fig.colorbar(sc, ax=ax, shrink=.6, label="RPC intensity")
    set_radar_axes(ax, "RPC XYZ, coloured by intensity")

    ax = fig.add_subplot(3, 4, 7)
    for i, label in enumerate(("X", "Y", "Z", "intensity")):
        ax.hist(rpc[:, i], bins=40, alpha=.45, label=label)
    ax.legend(fontsize=8); ax.set_title(f"RPC distributions (N={len(rpc)})")

    ax = fig.add_subplot(3, 4, 8)
    feat = frame["rgb_feat"]
    if feat is None:
        ax.text(.5, .5, "rgb_feat missing", ha="center")
    else:
        ax.plot(feat, linewidth=.6); ax.set_xlabel("feature dimension")
        ax.set_ylabel("activation")
    ax.set_title("RGB feature vector (not an RGB image)")

    ax = fig.add_subplot(3, 4, 9, projection="3d")
    verts, faces = frame["verts"], frame["faces"]
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                    color="lightsalmon", alpha=.75, linewidth=0)
    draw_skeleton(ax, frame["joints"])
    set_radar_axes(ax, "GT SMPL-X surface + 22-joint skeleton")

    ax = fig.add_subplot(3, 4, 10, projection="3d")
    ax.scatter(rpc[:, 0], rpc[:, 1], rpc[:, 2], c=rpc[:, 3], cmap="viridis", s=3, alpha=.7)
    ax.scatter(verts[::8, 0], verts[::8, 1], verts[::8, 2], c="tomato", s=1)
    draw_skeleton(ax, frame["joints"], color="red")
    set_radar_axes(ax, "RPC + GT mesh/skeleton")

    ax = fig.add_subplot(3, 4, 11, projection="3d")
    depth = frame["depth_radar"]
    if depth is not None:
        ax.scatter(depth[:, 0], depth[:, 1], depth[:, 2], s=1, c=depth[:, 2], cmap="cividis", alpha=.45)
        ax.scatter(verts[::8, 0], verts[::8, 1], verts[::8, 2], c="red", s=1)
    set_radar_axes(ax, "Depth-derived points (camera→radar) + mesh")

    ax = fig.add_subplot(3, 4, 12)
    depth_cam = frame["depth_cam"]
    if depth_cam is not None:
        ax.scatter(depth_cam[:, 0], depth_cam[:, 1], c=depth_cam[:, 2], s=2, cmap="cividis")
        ax.invert_yaxis(); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("camera X (m)"); ax.set_ylabel("camera Y (m)")
    ax.set_title("Depth point cloud, coloured by camera Z")

    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_animation(frames, path, fps, vmin, vmax):
    roots = np.asarray([x["joints"][0] for x in frames])
    with imageio.get_writer(path, mode="I", duration=1000 / fps, loop=0) as writer:
        for idx, frame in enumerate(frames):
            fig = plt.figure(figsize=(14, 8), constrained_layout=True)
            key = frame["key"]
            fig.suptitle(f"P{key[0]}-A{key[1]} frame={key[2]}  ({idx + 1}/{len(frames)})")
            ax = fig.add_subplot(2, 2, 1)
            ax.imshow(frame["xy"], origin="lower", extent=(-3, 3, .5, 6),
                      aspect="auto", cmap="turbo", vmin=vmin, vmax=vmax)
            ax.scatter(frame["joints"][:, 0], frame["joints"][:, 1], s=5, c="white")
            ax.set_title("RT BEV + GT joints"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

            ax = fig.add_subplot(2, 2, 2, projection="3d")
            rpc = frame["rpc"]
            ax.scatter(rpc[:, 0], rpc[:, 1], rpc[:, 2], c=rpc[:, 3], s=3, cmap="viridis")
            v = frame["verts"]
            ax.scatter(v[::12, 0], v[::12, 1], v[::12, 2], c="red", s=.7)
            draw_skeleton(ax, frame["joints"], color="red", linewidth=1)
            set_radar_axes(ax, "RPC + GT")

            ax = fig.add_subplot(2, 2, 3, projection="3d")
            d = frame["depth_radar"]
            if d is not None:
                ax.scatter(d[:, 0], d[:, 1], d[:, 2], c=d[:, 2], s=1, cmap="cividis", alpha=.5)
                ax.scatter(v[::12, 0], v[::12, 1], v[::12, 2], c="red", s=.7)
            set_radar_axes(ax, "Depth points + GT")

            ax = fig.add_subplot(2, 2, 4)
            ax.plot(roots[:idx + 1, 0], roots[:idx + 1, 1], "o-", markersize=2)
            ax.set_xlim(*BOUNDS[0]); ax.set_ylim(*BOUNDS[1]); ax.set_aspect("equal", adjustable="box")
            ax.set_title("GT root trajectory"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(rgba[:, :, :3])
            plt.close(fig)


def make_report(sequence, all_keys, selected, frames, envs):
    ids = [x[2] for x in all_keys]
    expected = set(range(min(ids), max(ids) + 1))
    report = {
        "sequence": f"P{sequence[0]}-A{sequence[1]}",
        "sequence_frames": len(all_keys), "frame_id_range": [min(ids), max(ids)],
        "missing_frame_ids": sorted(expected - set(ids)),
        "duplicate_frame_ids": sorted(k for k, n in collections.Counter(ids).items() if n > 1),
        "visualized_keys": selected,
        "database_entries": {name: env.stat()["entries"] for name, env in envs.items()},
        "representative_frame": {},
        "notes": {
            "radar_comp": "Sparse rows [ix, iy, iz, response]; decompressed to 121x111x31.",
            "radar_pc": "Rows [x_m, y_m, z_m, intensity].",
            "rgb_feat": "1024-D learned/encoded feature vector; not pixels and not directly invertible to RGB.",
            "item_depth1": "2000 XYZ samples, treated as camera coordinates in metres from axis ranges/calibration; not a 2-D depth raster. Its generation script is absent from this release.",
            "smplx": "GT mesh generated from betas/body pose/root orientation/translation and calibrated to radar coordinates.",
        },
    }
    f = frames[len(frames) // 2]
    for name in ("rt_sparse", "rt", "rpc", "rgb_feat", "depth_cam", "verts", "joints"):
        if f[name] is not None:
            report["representative_frame"][name] = finite_stats(f[name])
    report["representative_frame"]["params"] = {
        k: finite_stats(v) if isinstance(v, np.ndarray) else v for k, v in f["params"].items()
    }
    return report


def main():
    args = parse_args()
    root, out = Path(args.data_root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    envs = open_envs(root)
    groups = all_sequences(envs["indicator"])
    eligible = sorted(k for k, v in groups.items() if len(v) >= args.frames)
    if args.sequences:
        choices = [parse_sequence(x) for x in args.sequences]
    else:
        choices = random.Random(args.seed).sample(eligible, 2)
    renderer = SmplRenderer(args.smplx_root)
    manifest = {"seed": args.seed, "choices": [], "frames_per_sequence": args.frames}

    for sequence in choices:
        if sequence not in groups or len(groups[sequence]) < args.frames:
            raise ValueError(f"Sequence {sequence} is missing or shorter than {args.frames} frames")
        seq_out = out / f"P{sequence[0]}-A{sequence[1]}"
        seq_out.mkdir(exist_ok=True)
        # Use a contiguous valid-key window, reproducibly selected within each sequence.
        keys = groups[sequence]
        local_rng = random.Random(args.seed + sequence[0] * 1000 + sequence[1])
        start = local_rng.randrange(0, len(keys) - args.frames + 1)
        selected = keys[start:start + args.frames]
        print(f"Loading P{sequence[0]}-A{sequence[1]} frames {selected[0][2]}..{selected[-1][2]}", flush=True)
        frames = [load_frame(envs, renderer, key) for key in selected]
        positive = np.concatenate([f["xy"][f["xy"] > 0] for f in frames])
        vmin = float(np.percentile(positive, 5)) if positive.size else 0.0
        vmax = float(np.percentile(positive, 99.5)) if positive.size else 1.0
        representative = frames[len(frames) // 2]
        render_overview(representative, seq_out / "overview.png", vmin, vmax)
        render_animation(frames, seq_out / "sequence_100_frames.gif", args.fps, vmin, vmax)
        report = make_report(sequence, keys, selected, frames, envs)
        with open(seq_out / "report.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        manifest["choices"].append({
            "sequence": report["sequence"], "sequence_frames": len(keys),
            "visualized_range": [selected[0][2], selected[-1][2]],
            "missing_frames": len(report["missing_frame_ids"]),
        })

    with open(out / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    for env in envs.values():
        env.close()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
