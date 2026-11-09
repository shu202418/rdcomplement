# S1 V3 RT completion

This configuration keeps inference RT-only while using SMPL-X during training
to supervise 22 joints and 64 topology-consistent surface anchors.

Implemented changes:

- 15-part rigid alignment. A third anatomical joint defines each local frame,
  so motion around the bone axis is represented instead of using root
  translation or minimal-axis rotation alone.
- Six mask families: random returns, weak returns, single moving part,
  continuous moving part, spatial blocks, and radar-centred range shadows.
- A surface head predicts 64 fixed-topology SMPL-X anchors. Their aggregate
  heatmap joins the 15 capsule maps in all three geometry-fusion scales.
- S1 and deterministic fractional sampling. A fraction is an exact seeded
  subset, not the first part of the index file. Full causal history remains
  available for every selected target.

The downstream M4Human task loss is deliberately not used in this version.

## One-batch functional smoke test

The checked-in code has already passed this path. Use new output directories
when repeating it:

```bash
cd /data/M4Human-main
conda activate m4human

python -m rt_completion prepare-labels --config configs/rt_completion_s1_v3_surface.yaml --smplx-models models --output MR-Mesh/rt_completion_labels_s1_v3_smoke2 --limit 8 --batch-size 8 --device cuda:0

python -u -m rt_completion train --config configs/rt_completion_s1_v3_surface.yaml --labels MR-Mesh/rt_completion_labels_s1_v3_smoke2 --output experiments/rt_completion_s1_v3_smoke2 --epochs 1 --batch-size 1 --limit-train 8 --limit-val 8 --max-steps 1 --device cuda:0 --workers 0
```

## Ten-percent S1 pilot

The label and training fractions must match. The label cache is intentionally
sampled and therefore is not suitable for an un-fractioned/full-data run. Its
top-level manifest state remains `partial` relative to all S1 frames; check
`partitions.*.requested_complete`, which is `true` when the requested fraction
is ready.

```bash
cd /data/M4Human-main
conda activate m4human

python -u -m rt_completion prepare-labels --config configs/rt_completion_s1_v3_surface.yaml --smplx-models models --output MR-Mesh/rt_completion_labels_s1_v3_10pct --train-fraction 0.1 --val-fraction 0.1 --sampling-seed 42 --batch-size 32 --device cuda:0

python -u -m rt_completion train --config configs/rt_completion_s1_v3_surface.yaml --labels MR-Mesh/rt_completion_labels_s1_v3_10pct --output experiments/rt_completion_s1_v3_10pct --train-fraction 0.1 --val-fraction 0.1 --epochs 3 --device cuda:0 --workers 4
```

For a faster pipeline check, replace both fractions with `0.01` and use one
epoch. Do not pass a full teacher cache to a fractional run; temporal pseudo-GT
is built online from the selected sequences.

S1 contains 495,506 train, 33,811 validation, and 131,806 test frames after the
project's invalid-frame filter. With seed 42, `0.1` selects 49,551 train and
3,382 validation targets.

## Full S1 training

```bash
cd /data/M4Human-main
conda activate m4human

python -u -m rt_completion prepare-labels --config configs/rt_completion_s1_v3_surface.yaml --smplx-models models --output MR-Mesh/rt_completion_labels_s1_v3 --batch-size 32 --device cuda:0

python -u -m rt_completion train --config configs/rt_completion_s1_v3_surface.yaml --labels MR-Mesh/rt_completion_labels_s1_v3 --output experiments/rt_completion_s1_v3 --device cuda:0 --workers 4
```

`best.pt`, `last.pt`, `progress.pt`, and `metrics.jsonl` follow the existing
checkpoint contract. Best-model selection uses validation masked RT error,
joint MPJPE, and surface-anchor error. Test labels are never generated.
