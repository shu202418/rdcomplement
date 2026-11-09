# M4Human 流程优化：RT-Mesh-Kinematic

## 1. 文献结论与代码映射

`10.28汇报.md` 中的工作给出了三类可以迁移到 M4Human 的先验：

| 文献方向 | 可迁移结论 | 本仓库实现 |
|---|---|---|
| Human as Points | 观测几何应参与人体先验校正，不能只回归参数 | 保留 RT RoI 特征，并让 RT 能量中心、候选平移和骨架共同参与最终回归 |
| LiveHPS / LiveHPS++ | 显式轨迹和速度比把多帧简单堆叠为通道更能抑制跳变 | `RTEnergyTrajectory` + GRU + 逐帧速度监督 |
| OptimalCap | 父子骨链和多候选运动学求解可同时提升结构合理性与效率 | 22 关节 skeleton-link 优化、4 个平移候选和轻量打分头 |

M4Human 原始 RT-Mesh 仍然保留，新增模型名为 `RT-Mesh-Kinematic`，因此可以直接做严格消融。

## 2. 优化后流程

```text
4-frame RT
  ├─ 原 RT-Mesh：全局定位 → 3D RoI → 融合特征
  └─ RT 能量质心序列 → 时序编码 → 速度
                           │
融合特征 → coarse joints → parent-child links → refined joints
                           │
base / RT center / velocity extrapolation / skeleton root
                           ↓
                    candidate weighting
                           ↓
          pose + root orientation + translation refinement
                           ↓
                         SMPL-X
```

新增监督项：

- `joints`：修正后 22 关节与 MoCap 关节的 L1 损失；
- `skeleton`：父子骨链向量损失；
- `velocity`：连续帧全局平移差分损失。

残差姿态头和根朝向头采用零初始化；平移候选门控初始化接近关闭。因此新模型从原始 RT-Mesh 行为附近开始训练，而不是用随机新头覆盖已有预测。

## 3. 运行与消融

```bash
torchrun --nproc_per_node=4 main1_multigpu_clean.py \
  --config configs/rtmesh_kinematic_s2.yaml
```

建议固定 seed 和数据划分，依次比较：

1. `RT-Mesh`；
2. `RT-Mesh-Kinematic`，仅 `joints`；
3. 再加入 `skeleton`；
4. 完整加入 `velocity`；
5. 在最佳版本上叠加已有离线 RT 补全缓存。

重点报告 S2/S3 的 MVE、MJE、TE；同时按快速非原地动作单独统计，因为文献与 M4Human 原论文都指出这些动作最能体现轨迹和速度先验的价值。

## 4. 实现注意事项

- 时序平移标签只在新模型启用时读取，原始基线无额外 LMDB 开销。
- RT 能量质心是弱观测，不能替代 RoI 特征，因此通过候选门控融合而非硬覆盖预测。
- 当前速度监督使用 12 Hz 相邻帧位移；若要输出物理速度（m/s），训练标签和推理输出需同时乘以 12。
