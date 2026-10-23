import argparse
import torch

from . import pipeline


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description="M4Human offline temporal RT completion")
    commands = parser.add_subparsers(dest="command", required=True)
    labels = commands.add_parser("prepare-labels", help="SMPL-X -> train/val 22-joint supervision only")
    labels.add_argument("--config", default="configs/rt_completion_s2.yaml")
    labels.add_argument("--smplx-models", default="models")
    labels.add_argument("--output", required=True)
    labels.add_argument("--limit", type=positive, help="Debug only: target frames per split")
    labels.add_argument("--batch-size", type=positive, default=32)
    labels.add_argument("--resume", action="store_true")
    labels.set_defaults(func=pipeline.prepare_labels)

    teacher = commands.add_parser("prepare-teacher", help="Cache train/val articulated weak targets (not completed RT)")
    teacher.add_argument("--config", default="configs/rt_completion_s2_v2.yaml")
    teacher.add_argument("--labels", required=True)
    teacher.add_argument("--output", required=True)
    teacher.add_argument("--splits", nargs="+", choices=("train", "val"), default=["train", "val"])
    teacher.add_argument("--batch-size", type=positive, default=2)
    teacher.add_argument("--workers", type=int, default=0)
    teacher.add_argument("--map-size-gb", type=positive, default=32, help="Initial LMDB virtual map; grows on demand")
    teacher.add_argument("--limit", type=positive, help="Debug only: incomplete cache, rejected by training")
    teacher.add_argument("--resume", action="store_true", help="Skip already committed frames in a matching cache")
    teacher.set_defaults(func=pipeline.prepare_teacher)

    train = commands.add_parser("train", help="Train completion on train, select checkpoint on val")
    train.add_argument("--config", default="configs/rt_completion_s2.yaml")
    train.add_argument("--labels", required=True)
    train.add_argument("--teacher-cache", help="Complete, provenance-matched train/val weak-target cache")
    train.add_argument("--output", required=True)
    train.add_argument("--resume", help="Path to progress.pt (inside epoch) or last.pt (after epoch)")
    train.add_argument("--batch-size", type=positive)
    train.add_argument("--epochs", type=positive)
    train.add_argument("--limit-train", type=positive)
    train.add_argument("--limit-val", type=positive)
    train.add_argument("--max-steps", type=positive, help="Debug only: batches per split per epoch")
    train.add_argument("--workers", type=int, default=0)
    train.set_defaults(func=pipeline.train)

    export = commands.add_parser("export", help="RT-only inference; save sparse additions per frame")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    export.add_argument("--limit", type=positive, help="Debug only: intentionally INCOMPLETE cache")
    export.add_argument("--batch-size", type=positive, default=2)
    export.add_argument("--workers", type=int, default=0)
    export.add_argument("--probability-threshold", type=float)
    export.add_argument("--max-added-voxels", type=int)
    export.add_argument("--resume", action="store_true")
    export.set_defaults(func=pipeline.export)

    evaluate = commands.add_parser("evaluate", help="Fixed grouped val masks, held-out risk/coverage; no test GT")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--labels", required=True)
    evaluate.add_argument("--teacher-cache", help="Optional complete validation weak-target cache")
    evaluate.add_argument("--output", required=True, help="New JSON report; existing files are not overwritten")
    evaluate.add_argument("--groups", nargs="+", choices=pipeline.MASK_GROUPS)
    evaluate.add_argument("--limit", type=positive, help="Debug only: validation target frames")
    evaluate.add_argument("--max-steps", type=positive, help="Debug only: validation batches")
    evaluate.add_argument("--batch-size", type=positive, default=2)
    evaluate.add_argument("--workers", type=int, default=0)
    evaluate.set_defaults(func=pipeline.evaluate)

    make_config = commands.add_parser("make-m4human-config", help="Write a NEW config selecting a completed cache")
    make_config.add_argument("--base", default="configs/main_table/rtmesh_s2_train.yaml")
    make_config.add_argument("--cache", required=True)
    make_config.add_argument("--output", required=True)
    make_config.set_defaults(func=pipeline.write_m4human_config)

    for command in (labels, teacher, train, export, evaluate):
        command.add_argument("--device", default="auto", help="auto | cpu | cuda:0")
        command.add_argument("--threads", type=positive, default=4)
    args = parser.parse_args()
    if getattr(args, "workers", 0) < 0:
        parser.error("--workers must be non-negative")
    if hasattr(args, "threads"):
        torch.set_num_threads(args.threads)
    args.func(args)


if __name__ == "__main__":
    main()
