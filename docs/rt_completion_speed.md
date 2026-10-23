# V2 训练加速：固定伪 GT 缓存与轻量训练统计

本次只优化计算路径，不改变网络结构、混合 mask、损失权重、四组固定验证、最佳模型选择或非有限梯度保护。可以从现有 V2 `progress.pt` 继续训练，不需要重新准备 SMPL-X 标签。

## 按顺序运行

在现有 `m4human` 环境、项目根目录执行。以下命令不会自动由代码助手启动；全量预计算可能持续数小时。

```bash
cd /data/M4Human-main
conda activate m4human

# 1. 一次性生成 train/val 固定时序弱监督；可中断后用同一命令续跑。
python -m rt_completion prepare-teacher \
  --config configs/rt_completion_s2_v2.yaml \
  --labels MR-Mesh/rt_completion_labels_s2 \
  --output MR-Mesh/rt_teacher_s2_v2 \
  --splits train val \
  --batch-size 2 --workers 4 --device cuda:0 --resume && \
python -m rt_completion train \
  --config configs/rt_completion_s2_v2.yaml \
  --labels MR-Mesh/rt_completion_labels_s2 \
  --teacher-cache MR-Mesh/rt_teacher_s2_v2 \
  --output experiments/rt_completion_s2_v2_safe \
  --resume experiments/rt_completion_s2_v2_safe/progress.pt \
  --workers 4 --device cuda:0
```

`&&` 保证预计算正常退出后才启动训练。完整性校验还会阻止把调试/未完成缓存用于训练。若训练目录不同，请同时修改 `--output` 和 `--resume`。不要改变原断点的 batch size、mask、loss 或 debug 限制。

确认两个划分都完成：

```bash
python -m json.tool MR-Mesh/rt_teacher_s2_v2/manifest.json
```

应看到 `state: complete`、train/val 各自 `complete: true`。若只准备其中一个划分，请继续准备另一个；推荐始终使用默认的 train/val 两划分命令。

## 缓存不是补全 RT

```text
MR-Mesh/rt_teacher_s2_v2/
├── manifest.json
├── teacher.lmdb
└── teacher.lmdb-lock
```

这里只存训练/验证用的时序伪 GT：有效体素索引、FP32 原单位强度、FP32 可靠性权重，以及完整非零历史支持计数。支持计数包括有效 mask 之外的位置，稀疏解码可以无损重建原 teacher 的全部四个字段。不保存稠密全尺寸张量，不量化监督强度。

缓存严格绑定原 RT、数据划分、关节标签文件身份及签名、全部 loss 配置、胶囊半径和几何代码指纹。不依赖模型权重或随机 mask；只读 train/val 的 GT，不读取 test GT。配置/几何代码改变后若指纹不匹配，需另建缓存；不能使用旧缓存强行继续。

真正的补全 RT 仍需训练后运行 `export`，写入另一个目录的 `additions.lmdb`。导出和下游不读取 `teacher.lmdb`。

## 其他已落地优化

- 即使不指定 `--teacher-cache`，同一验证 batch 的四组 mask 和 AMP→FP32 重试也共享一次在线 teacher，不再重复构造。
- `train.detailed_metrics_every: 0` 默认只累计轻量训练标量，不每步运行 top-k、新增筛选和完整风险/校准统计。改为正整数 N 可每 N 批抽样诊断。所有验证仍保留完整四组诊断，不改变 best.pt 选择。
- 标量在设备端以 FP64 累积，日志汇总/进度保存时统一搬到 CPU。恢复旧半轮进度后，已有的详细诊断被标为 `sampled_diagnostics`，不会误报为完整轮统计。
- 轻量训练的 `masked_mae_log` 是按样本累计的批均值；旧完整诊断的同名顶层指标是按遮挡体素加权的均值。不要直接对比这两种训练曲线。验证仍使用原来的体素加权组指标和组间宏平均，可继续比较。
- 每次部位对齐将强度与支持图合并为一次 FP32 `grid_sample`；伪 GT 复用已计算的源人体区域，减少重复几何计算。保留关节坐标梯度和所有阈值。

本次没有启用更大 batch、跨尺度近似网格、删减分支或改变验证频率。GPU 的并行归约允许小幅浮点差异，不承诺旧实现与新实现长时间训练逐比特相同。

## 检查及评估

启动后应打印 `Temporal teacher: .../rt_teacher_s2_v2`、`train detailed_metrics_every=0`、`validation diagnostics=full`，并显示恢复的 epoch/step。

```bash
python -m rt_completion evaluate \
  --checkpoint experiments/rt_completion_s2_v2_safe/best.pt \
  --labels MR-Mesh/rt_completion_labels_s2 \
  --teacher-cache MR-Mesh/rt_teacher_s2_v2 \
  --output experiments/rt_completion_s2_v2_safe/evaluation_cached.json \
  --batch-size 2 --workers 4 --device cuda:0
```

评估报告不会覆盖已有文件，需要时换一个新文件名。缓存第一次预计算有额外成本；实际提速应按优化后完整 epoch 的 `seconds` 和相同规模的 samples/s 评估，不能把缓存构造时间漏算。

CPU 回归：

```bash
python -m unittest discover -s tests -p 'test_rt_*.py'
```
