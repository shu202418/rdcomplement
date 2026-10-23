# RT 补全 V2：部位遮挡、可信时序监督与可弃权融合

新版配置为 `configs/rt_completion_s2_v2.yaml`，选择 `architecture: reliability_v2`。
默认旧配置仍使用原有基线，不改变已有权重、缓存或实验目录。本轮没有加入 Flow Matching、Diffusion 或 mmDiff 的四个上下文模块，也没有全量精度提升的结论。

## 1. 本轮实际实现

### 混合 mask 与固定分组评测

训练时按配置权重，每个样本随机选择以下一种遮挡；目标只来自遮挡前真实存在的当前帧回波。

| 组名 | 遮挡方式 |
| --- | --- |
| `random` | 当前帧随机删除已有回波 |
| `single_part` | 当前帧删除一个骨段胶囊附近已有回波 |
| `continuous_part` | 连续若干帧删除同一部位的已有回波，胶囊逐帧跟随关节运动 |
| `spatial_block` | 在当前已有回波处选空间块，对所有帧删除同一空间区域 |

部位 mask 根据雷达坐标系下的 SMPL-X 关节生成，使用米制点到骨段距离、15 个骨段和各自半径比例。训练有半径与关节扰动。部位必须包含当前帧实测回波才可被选择；若没有任何可用部位，记录 `eligible=false`，**不偷偷改成随机遮挡**。胶囊只是空间近似，不能保证其内每个散射点都来自对应身体部位。

验证时分别运行四组，关闭几何扰动；由“验证 seed＋样本 indicator＋组名”的 SHA256 确定每个样本的随机种子，因此不随 epoch、batch 划分变化。固定不代表固定一个部位给所有样本，而是同一样本、同一组每次得到相同遮挡。

### 部位对齐时序伪 GT

教师分支使用未人工遮挡的历史实测 RT 与训练/验证关节标签，将15个部位近似对齐到当前帧。每个目标体素选择归一化距离最近的胶囊；骨轴之间使用最小旋转，骨轴方向按长度比缩放，径向不缩放。两个端点不能确定绕骨轴的扭转，所以这是**最小扭转几何近似**，不是恢复了完整人体旋转或雷达成像物理。

无效骨段、越界采样、异常长度比会被排除。单独变换观测支持图，并用插值支持归一化幅度，避免半体素插值把稀疏强度直接压低。只有不同的有效历史帧计入支持数，窗口填充的重复帧不计入。

有效 mask 的条件为：原始当前 RT 为0、位于当前人体候选区域、历史支持数至少为配置值（默认2）。没有有效目标就不施加时序监督；未知零不是“真实无反射”的标签。人工遮挡掉的当前实测回波，始终使用原始实测强度，而不被伪 GT 覆盖。

置信度为明确的启发式权重。设历史 log 强度为 \(\ell_i\)，有效支持数为 \(K\)，其稳健中心为中位数（偶数时取中间两值平均），分歧为 \(d=K^{-1}\sum_i|\ell_i-\operatorname{median}(\ell)|\)。骨长比几何权重为 \(q_i=\exp(-|\log(L_i/L_t)|/s_a)\)，则：

\[
C=\frac{K}{T-1}\;\frac{\sum_i q_i}{K}\;\exp(-d/s_c).
\]

默认 \(s_a=0.5,s_c=0.35\)。伪强度在 log 空间中按几何权重与 Huber 型残差权重聚合，再转换回原始强度。目标、有效 mask 和可信度全部停止梯度；时序损失为置信度加权稳健误差及正回波分数项，按有效目标数量归一化，不把置信度的缩小抵消掉。

这里教师的几何可信度主要由骨长比产生；没有声称它是完整的定位误差估计。部位对齐也不代表强度守恒，多径可能长期一致。因此 `confidence` 不是经过物理或概率校准的真值，不能保证天然未知位置真的存在回波。

### 局部可靠性融合与严格当前帧回退

共享 Dense 3D CNN 编码四帧，逐帧预测22关节的空间分布均值、对角方差，并运行 ConvGRU。前向分支只使用 RT 预测的关节与不确定性进行多尺度部位特征对齐，不读取 GT。

各尺度的局部融合对“历史帧1、历史帧2、历史帧3、null”共同 softmax。分数使用当前/历史特征及差异、局部支持、几何质量和时间距离；null 对应零历史增量。**移除 null 后不再次归一化历史权重。** 所有历史无效时，历史总权重严格为0。

严格回退同时覆盖三个入口：对齐历史特征、ConvGRU 时序增量、时序胶囊。解码器的时序状态由已对齐历史重新计算，每个历史帧的递归更新都受对应注意力权重控制；时序几何来自可信融合特征，不直接读取未经可靠性筛选的辅助 GRU 关节序列。不存在绕过 null 的 GRU 或时序几何旁路。各尺度门控是局部决策，不代表一个局部 null 值就能说明整个解码感受野都未使用历史；`force_no_history=True` 可检查全局当前帧回退。

历史质量分支另用人工留出的实测强度构造独立软标签：局部历史强度与当前留出强度越接近，质量标签越高。该标签停止梯度，避免靠把可信度全部压到0来降低任务损失。这仍是区域条件均值的质量代理，不是 ADC/SNR 噪声概率。

## 2. 标签边界、损失与输出

SMPL-X 关节仅用于：人工 mask、教师伪 GT、关节监督和验证目标；不作为模型前向或离线导出输入。新增逐帧关节 Smooth L1 与方差 NLL，排除重复填充历史，并在真实留出观测上统计强度不确定性覆盖率。

导出仍采用“原始 RT＋被接受新增体素”的方案，原始非零值不覆盖。反射分数、强度、预测方差与人体候选区域共同筛选新增，最后施加每帧新增预算。反射 sigmoid 分数不是已校准的反射概率；原始噪声也不会被本方案删除。

## 3. 依次运行

已有完整 `MR-Mesh/rt_completion_labels_s2` 可直接复用，因为数据与关节定义未改变。V2 结构已改变，**不要用旧 `experiments/rt_completion_s2/best.pt` 或 `last.pt` 进行 V2 resume**。以下为全量训练、验证报告、离线导出；前一步成功才继续下一步：

```bash
cd /data/M4Human-main &&
conda activate m4human &&
python -m rt_completion train \
  --config configs/rt_completion_s2_v2.yaml \
  --labels MR-Mesh/rt_completion_labels_s2 \
  --output experiments/rt_completion_s2_v2 \
  --device cuda:0 --workers 4 &&
python -m rt_completion evaluate \
  --checkpoint experiments/rt_completion_s2_v2/best.pt \
  --labels MR-Mesh/rt_completion_labels_s2 \
  --output experiments/rt_completion_s2_v2/val_report.json \
  --device cuda:0 --batch-size 2 --workers 4 &&
python -m rt_completion export \
  --checkpoint experiments/rt_completion_s2_v2/best.pt \
  --output MR-Mesh/rt_completed_s2_v2 \
  --splits train val test \
  --device cuda:0 --batch-size 2 --workers 4
```

`evaluate` 只读取 val 标签，不读取 test GT；输出四组留出恢复 MAE、68%/95%区间覆盖率、风险—覆盖率与接受统计。它检查的是人工留出实测回波，**不是天然未知体素的正确率**。报告路径必须是新文件。验证最佳模型按非空组等权汇总的恢复误差与关节误差选取，不使用测试标签。

四组验证需要额外前向，部位对齐也增加计算，不能默认与基线等速。最终应在相同新增预算、相同下游配置下比较原始 RT、旧补全、V2 补全，并报告下游姿态指标及导出耗时；单元测试通过不代表全量精度已经提高。

## 4. 本次验证记录（2026-10-17）

- 37 项 CPU 回归测试通过，覆盖旧接口、部位旋转、移动 mask、固定分组、可信度停止梯度、逐历史弃权、训练/续训/验证报告/离线导出。
- 现有 `experiments/rt_completion_s2/best.pt`（第27轮）严格加载成功，未修改旧权重。
- RTX 3090 上使用真实数据 `[2,4,121,111,31]` 完成 BF16 前向、一次反向和临时优化器更新，所有参数梯度有限；测试峰值已分配显存约 1033 MiB。该数值仅对应本次冒烟测试，不是全量训练显存承诺。
- 真实数据四类 mask 均产生有效留出目标；导出合并保留全部原始非零值；强制当前帧模式在修改历史后核心输出完全一致。
- V2 输出头使用小权重和保守偏置初始化，避免稀疏特征尖峰使初始强度与方差 NLL 爆炸；该初始化不是物理或概率校准。

没有启动正式全量 V2 训练，也没有生成可用于科研比较的 V2 完整缓存。上述结果仅证明运行与数值安全检查通过，不表示恢复精度或下游姿态已经提升。

## 5. 非有限梯度修复与训练恢复（2026-10-19）

负的 Gaussian NLL 不一定异常；本次错误是反向梯度出现非有限值。BF16 不再启用 GradScaler，FP16 仍使用梯度缩放，并在溢出时跳过不安全的参数更新、降低 scale。不能通过单独设置 `error_if_nonfinite=False` 来修复，否则可能把异常梯度写入权重。非缩放路径仍应对真实数值异常保留严格检查。

训练目录新增 `numerics_events.jsonl`：追加记录1起始的 epoch/step、AMP 模式、scale、异常原因、损失分项、异常参数名和输入张量的有限值范围。非有限指标以 JSON `null` 表示，不生成非标准的 `NaN`/`Infinity`。

在 `debug_failures/` 最多保存3个失败小批次快照，每个张量载荷不超过128 MiB。快照包含允许保留的 RT、遮挡输入、mask、22关节、历史有效标记和样本 indicator，保存到 CPU；不会保存完整 SMPL-X 网格或额外大张量。文件使用独立编号，不覆盖旧事件，可以用 `torch.load(path, map_location="cpu", weights_only=True)` 读取。这些文件只是数值排查资料，不是补全缓存，也不参与离线导出。

训练期间的周期性恢复文件为 `progress.pt`；整轮训练与验证结束后的 `last.pt`、验证集最佳的 `best.pt` 保留原有用途。`progress.pt` 用于减少长 epoch 中断的损失，不能替代经过验证选出的 `best.pt` 作为正式导出模型。

默认在第1步、每1000步、训练阶段结束（验证前）保存 `progress.pt`。它包含当前 epoch、下一批位置、优化器、mask/全局随机数状态、已处理样本的累计指标及跳过/恢复次数；续训直接取后续批次，不读取已处理批次来空跑。验证失败时也可从训练阶段末尾恢复，只重做验证。CPU 对照验证了连续训练与中断恢复的参数和指标完全一致；CUDA 算子的非确定性仍可能带来数值差异。

AMP 异常默认使用相同遮挡输入重试一次 FP32，重试前释放原计算图以避免双份图占显存。仍失败则清空梯度、记录并跳过；连续3个失败批次或每轮超过20个失败批次会保存安全进度并停止，避免持续异常被静默忽略。没有任何成功更新的短轮次不会产生 `best.pt`。可以通过 `train.checkpoint_every_steps`、`retry_fp32`、`max_nonfinite_batches`、`max_consecutive_nonfinite`、`max_failure_snapshots` 配置；这些恢复保护项允许与旧检查点不同，但模型、监督和优化器超参数仍严格校验。

第一次失败运行若没有保存任何 checkpoint，就不能恢复已经执行过的约201450步。建议保留旧目录作为排查记录，使用新目录重新开始：

```bash
cd /data/M4Human-main &&
conda activate m4human &&
python -m rt_completion train \
  --config configs/rt_completion_s2_v2.yaml \
  --labels MR-Mesh/rt_completion_labels_s2 \
  --output experiments/rt_completion_s2_v2_safe \
  --device cuda:0 --workers 4
```

以后如果中断且该目录存在 `progress.pt`，在同一实验目录续训：

```bash
cd /data/M4Human-main &&
conda activate m4human &&
python -m rt_completion train \
  --config configs/rt_completion_s2_v2.yaml \
  --labels MR-Mesh/rt_completion_labels_s2 \
  --output experiments/rt_completion_s2_v2_safe \
  --resume experiments/rt_completion_s2_v2_safe/progress.pt \
  --device cuda:0 --workers 4
```

若续训 `last.pt`，将 `--resume` 指向同目录下的 `last.pt`。不要跨配置、标签版本或调试样本限制混合续训；代码更新不会自动应用到已经运行的 Python 训练进程。上述修复与测试只验证数值安全和恢复流程，不代表全量补全质量或下游姿态指标已经提升。

2026-10-19 真实尺寸检查：RTX 3090、输入 `[2,4,121,111,31]`，BF16 scaler 禁用且旧超大 scale 被忽略；注入无限梯度后 FP32 重试成功，Adam 仅更新一次且全部参数/状态有限；模拟 FP16 的 `8.3e34` scale 溢出时正确跳过并回退至约 `4.15e34`。未启动新的正式全量训练。
