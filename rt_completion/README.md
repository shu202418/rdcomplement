# M4Human：离线时序 RT 补全

新版三个机制已独立实现：**部位运动混合 mask、带可靠性权重的部位对齐伪 GT、带 null 的多尺度历史融合**。
使用 `configs/rt_completion_s2_v2.yaml`，具体流程、限制和运行命令见
[V2 说明](../docs/rt_completion_v2.md)。原配置与下面的历史基线路径保持不变；
V2 必须使用新的训练/导出目录，不能把旧基线检查点当作 V2 续训。

已有 V2 实验的加速与断点续训：见[固定时序伪 GT 缓存及运行指令](../docs/rt_completion_speed.md)。
这项工程优化兼容已有 V2 `progress.pt`，不改变模型/损失/固定四组验证；`teacher.lmdb` 是训练监督，不是离线补全 RT。

这套代码将“补全模型训练”和“RT-Mesh 训练”分开。补全模型固定后，每帧只导出一次；RT-Mesh 读取原 RT 和新增体素，拼成补全 RT，再按原接口训练。

实现是可运行的研究基线，不是参考论文的复现，也未验证姿态精度提升。当前环境没有 `spconv`，因此编码器采用共享的轻量 **Dense 3D CNN**，不是 Sparse CNN。未安装新依赖。

## 数据流与边界

```text
准备监督（只处理 train / val）
SMPL-X 参数 + 原项目标定 → 雷达坐标下 22 关节 → joints.lmdb

补全网络训练
4 帧原 RT → 人工遮挡 → 共享 3D CNN → 3D ConvGRU
                                      ↓
                                预测 22 关节
                                      ↓
                                15 部位胶囊热图
                                      ↓
                        多尺度门控融合 + 3D U-Net 式解码
                                      ↓
                        反射概率 / log 强度 / log 方差

离线导出（不读 GT，不调用 SMPL-X）
4 帧原 RT → 固定补全网络 → 置信度/身体区域/数量筛选 → additions.lmdb

下游训练
原 RT + additions.lmdb → [4, 121, 111, 31] 补全 RT → 原 RT-Mesh
```

- RT 轴顺序是 `X,Y,Z`；模型输入为 `[B,T,X,Y,Z]`。空间范围沿用项目：X `[-3,3]`、Y `[0.5,6]`、Z `[0,3]` 米。
- 共享编码器下采样 3 次，最低分辨率用 ConvGRU 融合；当前帧的多尺度特征作为解码跳接。关节由空间 soft-argmax 预测。
- 胶囊表示躯干、颈、头、左右肩、上臂、前臂、大腿、小腿、足。它是软几何约束，不是人体所有表面都应该反射的标签。
- **GT 关节不作为网络输入**。SMPL-X 用于训练/验证的关节监督和时序伪标签；网络内的胶囊始终由预测关节生成。
- 原 RT 的所有非零值均保留，包括弱信号；零值是未知位置，不能断言为缺测。导出只允许向原来为零的位置加入候选信号。
- 历史窗口只取同一序列、同一数据划分里的连续过去帧。遇到缺帧/划分边界时停止向前取帧，重复最早可用帧；重复帧不算额外时序证据。

## 一、运行完整离线流程

从 `/data/M4Human-main` 运行，使用现有 `m4human` 环境；以下是完整任务命令，**不是已经执行的全量训练**。

```bash
conda activate m4human

# 1. 只准备 train/val 的 SMPL-X 关节监督。
python -m rt_completion prepare-labels \
  --config configs/rt_completion_s2.yaml \
  --smplx-models models \
  --output MR-Mesh/rt_completion_labels_s2 \
  --device cuda:0

# 2. 训练补全模型，以 val 遮挡恢复误差 + 关节误差选择 best.pt。
python -m rt_completion train \
  --config configs/rt_completion_s2.yaml \
  --labels MR-Mesh/rt_completion_labels_s2 \
  --output experiments/rt_completion_s2 \
  --device cuda:0 --workers 4

# 3. 固定模型，导出所有划分；这里不读取关节标签或 SMPL-X 参数。
python -m rt_completion export \
  --checkpoint experiments/rt_completion_s2/best.pt \
  --output MR-Mesh/rt_completed_s2 \
  --splits train val test \
  --batch-size 2 --workers 4 --device cuda:0

# 4. 生成独立的下游配置，不修改原 config.yaml。
python -m rt_completion make-m4human-config \
  --base configs/main_table/rtmesh_s2_train.yaml \
  --cache MR-Mesh/rt_completed_s2 \
  --output configs/rtmesh_completed_s2.yaml

# 5. 启动原 RT-Mesh 训练入口。
torchrun --standalone --nproc_per_node=1 \
  main1_multigpu_clean.py --config configs/rtmesh_completed_s2.yaml
```

默认配置为 `p1/s2`、补全 batch size 2、30 epochs。`--device auto` 在没有 CUDA 时会用 CPU；全量任务建议显式指定 GPU。补全训练本身是单进程，不需要 `torchrun`；下游仍沿用项目原来的分布式入口。

已有下游配置只需要增加：

```yaml
paths:
  cached_root: ./MR-Mesh/
  rt_completion_cache: ./MR-Mesh/rt_completed_s2
```

其他原有 `paths.smplx` 等字段保持原样；模型必须是 `rawImage_XYZ` 模态，数据 scale/split 要与缓存匹配。不设置该字段时，原始 RT 路径不变。RPC、图片和 GT 没有被补全或替换。

## 二、补全 RT 保存在哪里

```text
MR-Mesh/rt_completed_s2/
├── manifest.json      # 源数据/划分/权重哈希/阈值/完成状态
├── additions.lmdb     # 每个 [subject, action, frame] 一个记录
└── additions.lmdb-lock
```

每条记录包含：

| 字段 | 含义 |
| --- | --- |
| `indices` | 新增体素的展平索引，int32；也就是生成区域 mask 的稀疏表示 |
| `values` | 新增强度，float32、原始 RT 强度单位 |
| `probability` | 新增位置的反射分数（历史字段名，未校准概率），float16 |
| `sigma_log` | 新增位置的预测标准差，float16、log1p 强度空间 |
| `joints` | 当前帧预测的 22 关节，float32、米 |

网络预测强度使用 `log1p(raw / intensity_scale)` 空间；候选强度为 `probability * denormalize(predicted_intensity)`。通过反射分数、标准差、预测人体区域和最小强度筛选后，最多保存配置中的 `max_added_voxels` 个体素。默认上限 2048；阈值未调优，应只用验证集选择。

**保存的是可以恢复完整补全 RT 的稀疏增量，不是重复保存全部原始体素。** `dataset/m4human_dataset.py` 读取每个历史帧的增量并合并，随后仍输出原来的 4 帧张量。人工改坏的索引、覆盖原信号、缺失键、错误划分、未完成缓存会直接报错，不会悄悄回退到原 RT。

独立读取单帧：

```python
from rt_completion.pipeline import load_config
from rt_completion.data import RTSequenceDataset
from rt_completion.storage import CompletionCache

cfg = load_config("configs/rt_completion_s2.yaml")
dataset = RTSequenceDataset(cfg["data"], "test")  # 不读取 GT
item = dataset[0]
key = item["indicator"].tolist()
reader = CompletionCache(
    "MR-Mesh/rt_completed_s2", cfg["data"]["cache_path"],
    "p1", "s2", "test", cfg["data"]["shape"],
)
completed_rt = reader.apply(item["rt"][-1].numpy(), key)  # [121,111,31]
record = reader.get(key)  # 查看生成位置、概率和不确定性
```

仅增量数组在最坏的 2048 体素/帧情况下约为 24 KiB/帧，661k 帧约 15 GiB，加上 LMDB 页和记录开销；实际空间取决于筛选后的数量。LMDB map 自动增长，不会额外复制原 `radar_comp.lmdb`。若一帧无候选，也会写入空记录，表示该帧已经处理，而非缺失。

## 三、监督含义与研究限制

1. 人工遮挡：当前帧随机丢弃已观测体素，并在全部 4 帧抹掉同一立方区域；用未遮挡的原始观测做恢复监督。非遮挡观测用较小权重。
2. 时序伪标签：用 GT 根关节平移对齐历史 RT，在历史/当前 GT 胶囊区域内筛选，需要至少两个**不同**历史帧支持，只在当前零值位置提供低权重正监督。不把其他零值当负标签。当前实现**没有逐肢体旋转形变对齐**，手脚快速运动仍可能产生错误对应。
3. 22 关节损失：监督补全网络自身的粗关节分支；不是将下游 RT-Mesh 姿态损失反传进补全网络。
4. 不确定性：在实际观测恢复处通过异方差 Gaussian NLL 学习；这是模型估计，尤其在原本无观测区域**尚未经过可信度校准**，不应直接解释成真实误差概率。
5. 物理/任务效果：没有完整实测 RT GT；本方法不能证明生成信号具有真实物理反射特性，也不保证提升姿态精度。需评估原 RT、补全 RT、单帧/时序、无几何、无伪标签等消融，并监控新增体素数量、运动残影和人体区域外新增率。
6. 输入公平性：下游一次读取 4 个“各自由 4 帧补全”的 RT，可能间接使用最多 **7 帧原始历史**；与原始 4 帧基线比较时必须报告这一点，或增加等历史长度基线。
7. 协议：不同 `s1/s2/s3` 应分别训练补全权重和生成缓存；严格禁止将某划分上训练的补全权重混用于另一个划分的测试。`s1` 同序列划分本身可能相关，建议先做跨主体 `s2`。
8. 下游原入口的训练/评估规则未改写。原项目目前存在按 test MVE 保存 best 的行为；严格实验应另行改为 val 选模型后一次性报告 test，而不是将本次接入当作已修复该问题。

## 四、断点、快速验证与代码位置

- 标签准备和导出续跑：在原命令后加 `--resume`。已提交帧会跳过；源数据、协议、权重或阈值改变会拒绝混写。
- 训练续跑：`--resume experiments/rt_completion_s2/last.pt`，`--output` 保持原目录；仅允许增加 epochs，不允许悄悄更换网络、监督或抽样限制。
- 更换导出阈值、补全权重或划分：使用新输出目录。原 LMDB 和现有训练配置不覆盖。
- `--limit`、`--limit-train`、`--limit-val`、`--max-steps` 是调试参数。有限帧导出会标记 `partial`，不能伪装成完整缓存送入下游。

CPU 回归测试：

```bash
python -m unittest discover -s tests -p test_rt_completion.py -v
```

已完成的验证：7 项 CPU 回归测试；真实数据 train/val 各 4 帧的标签准备、原尺寸 RT 一次训练和验证、train/val/test 各 2 帧导出；`workers=2` 的真实数据导出。抽查关节标签与原数据集输出的最大绝对差约 `4.8e-7` 米。仅验证了运行链路，尚未进行完整训练、GPU/AMP 验证或姿态精度对比。

数据加载使用 `spawn` 创建独立 worker，避免把导出写句柄/文件锁带入子进程；若运行环境禁止多进程通信，使用 `--workers 0`。worker 等待超过 60 秒会报错，而非无限等待。

少量真实数据冒烟测试（所有输出在独立临时目录；重复运行请用新目录或正确续跑）：

```bash
RT_SMOKE_DIR=$(mktemp -d /tmp/m4human-rt-smoke.XXXXXX)
python -m rt_completion prepare-labels \
  --output "$RT_SMOKE_DIR/labels" --limit 4 --batch-size 2 --device cpu
python -m rt_completion train \
  --labels "$RT_SMOKE_DIR/labels" --output "$RT_SMOKE_DIR/run" \
  --epochs 1 --limit-train 4 --limit-val 4 --max-steps 1 --batch-size 1 --device cpu
python -m rt_completion export \
  --checkpoint "$RT_SMOKE_DIR/run/best.pt" --output "$RT_SMOKE_DIR/completed" \
  --limit 2 --batch-size 1 --device cpu
```

冒烟权重只验证链路，不能用于科研精度结论；保守筛选下新增体素为 0 也可能是正常结果，不能为“看起来补全了”而在 test 上调低阈值。

主要文件：`model.py` 网络，`geometry.py` 胶囊与平移对齐，`losses.py` 遮挡和损失，`data.py` 因果窗口与数据读取，`storage.py` 增量缓存，`pipeline.py` 四阶段命令，`../configs/rt_completion_s2.yaml` 参数，`../tests/test_rt_completion.py` 回归测试。
