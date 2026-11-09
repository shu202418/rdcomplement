# M4Human RT 补全：服务器环境、数据下载与迁移

整理日期：2026-10-24，北京时间。本文将“rd 补全”按本项目现有 **RT（Radar Tensor）补全**流程解释。

本说明基于当前 `/data/M4Human-main` 代码、`m4human` 环境的实际版本和文件大小核对；不是原论文环境的照抄。没有在本次整理中安装软件、复制大数据、停止训练或修改模型代码。目标服务器尚未实际部署验证。

## 1. 先看结论

- 只训练 RT 补全网络：需要原 RT、划分文件和已生成的22关节标签；不需要 RGB、深度或雷达点云。
- 固定时序 teacher 缓存是可选加速项，不是另一套原始数据集，也不是补全结果。
- 从零生成关节标签：另需 SMPL-X 参数、标定及模型文件。
- 仅导出补全 RT：需要原 RT、划分和已验证的补全权重，不读 GT、SMPL-X 或 teacher。
- 后续通过原主入口训练 RT-Mesh：依赖和数据更多，不能直接套用“纯补全最小环境”。

## 2. 当前实际环境

当前环境名 `m4human`，Python 位于 `/home/rtx3090/miniconda3/envs/m4human/bin/python`。以下包版本经实际环境读取，`pip check` 通过。

| 组件 | 当前版本 | 用途 |
| --- | --- | --- |
| Python | 3.10.18 | 推荐按此版本创建新环境 |
| PyTorch | 2.7.1+cu118 | 补全训练及推理 |
| PyTorch CUDA build | 11.8 | `torch.version.cuda`；不等于驱动版本或系统 nvcc |
| cuDNN | 9.1.0 | 当前 PyTorch 运行库 |
| NumPy | 2.2.6 | RT 数组和缓存 |
| PyYAML | 6.0.3 | 配置读取 |
| lmdb | 1.7.3 | 数据读写 |
| msgpack | 1.1.1 | 记录解码 |
| SMPL-X Python 包 | 0.1.28 | 仅准备标签时需要；模型文件另下 |
| SciPy | 1.15.3 | 标签模型兼容及下游依赖 |
| torchvision / torchaudio | 0.22.1+cu118 / 2.7.1+cu118 | 当前全环境已安装；纯补全不要求 |
| pip / setuptools | 25.2 / 80.9.0 | 安装工具 |

完整 Conda/Python 包清单见 [当前环境审计快照](environment/m4human-packages-20261024.json)。该 JSON 是核对记录，不是自动安装文件。

原 `environment.yml` 的环境名是 `mmwave`，带有另一台机器的绝对 `prefix`、许多非补全依赖、CUDA 扩展，以及需要额外 wheel 索引的包；不建议在新服务器直接用它创建纯补全环境。下面的 requirements 按用途拆分，没有改动旧文件。

当前补全编码器是 Dense 3D CNN，没有使用 spconv。纯补全不需要编译 PointNet2、Chamfer、pytorch3d 或 OpenGL。

## 3. 新服务器安装：推荐最小环境

本路线面向 Linux x86_64＋NVIDIA CUDA 服务器；代码还使用 `fcntl` 文件锁。Windows或ARM服务器不能直接照抄这里的wheel和工具链步骤。

以下在新服务器执行。先复制当前项目代码；仅克隆作者官方仓库不能保证含有本地新增的 `rt_completion/`、teacher 缓存与数值安全修复。

### 3.1 创建环境并安装 PyTorch

```bash
conda create -n m4human python=3.10.18 pip=25.2 -c conda-forge
conda activate m4human

cd /data/M4Human-main

# 使用 CUDA 11.8 官方 wheel，不必为纯补全另外编译 CUDA 扩展。
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements/rt-completion.txt
python -m pip check
```

这是与当前环境匹配的安装路线，不是要求所有服务器必须安装 CUDA 11.8 系统 toolkit。服务器需要正常的 NVIDIA 驱动；GPU 型号必须被所选 PyTorch CUDA 构建支持。新架构 GPU 若不支持 cu118，应重新选择官方兼容构建并做回归验证，不要强行使用旧 wheel，也不要承诺跨硬件逐比特复现。

PyTorch 安装组合核对来源：[官方历史版本页面，v2.7.1](https://pytorch.org/get-started/previous-versions/#v271)。

### 3.2 需要重新生成 SMPL-X 标签时再安装

```bash
python -m pip install -r requirements/rt-completion-labels.txt
```

如果直接迁移已经完成的 `rt_completion_labels_s2`，运行补全训练、teacher 预计算、验证和导出都不需要安装 SMPL-X 或加载其模型文件。

### 3.3 安装后检查

```bash
nvidia-smi
python -c "import torch; print('torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('visible GPUs:', torch.cuda.device_count())"
python -c "import numpy, yaml, lmdb, msgpack; from rt_completion import pipeline; print('RT completion imports OK')"
python -m rt_completion --help
```

只有 CUDA 可用时再执行：

```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); x=torch.ones(16, device='cuda'); print('GPU tensor:', x.sum().item())"
```

最小环境可先运行以下纯补全 CPU 回归测试（不需要真实数据或SMPL-X模型）：

```bash
python -m unittest tests.test_rt_geometry_v2 tests.test_rt_model_v2 \
  tests.test_rt_mask_loss_v2 tests.test_rt_speed_pipeline \
  tests.test_rt_numerics tests.test_rt_teacher_cache
```

完整 `unittest discover -s tests -p 'test_rt_*.py'` 还包括原下游Dataset集成测试，需先安装第9节的额外Python依赖；不要把缺少绘图/SMPL-X包误认为补全入口不能运行。

## 4. 运行 RT 补全需要哪个数据集？

使用 **M4Human 的预处理 LMDB 雷达数据**，本地目录是 `MR-Mesh/rf3dpose_all`。不必为现有补全代码下载约2 TB的完整原始多模态数据。

下载入口：[M4Human 官方仓库的 Download Datasets 部分](https://github.com/FanJunqiao/M4Human#clickrun-step-2-download-datasets)，选择 **Full processed dataset (radar modality)**。

作者提供的 processed 下载链接为：[SharePoint processed dataset](https://entuedu-my.sharepoint.com/:f:/g/personal/fanj0019_e_ntu_edu_sg/IgB2wdU_5tJ3S403VafC_q-mAcTVON3VY48SpwUsw4i9Egk?e=6zCncR)。本次访问跳转到 Microsoft 登录，不能保证匿名 wget 可用；需要通过作者提供的权限或联系作者获取，不绕过访问控制。

本地新增的 `rt_completion_labels_s2`、`rt_teacher_s2_v2` 和训练权重不是作者原始下载包里的标准文件，需从当前机器迁移，或在服务器重新生成。

### 4.1 各阶段必需文件

| 阶段 | 必需输入 | 不需要 |
| --- | --- | --- |
| 准备22关节标签 `prepare-labels` | RT源文件身份、划分、`params.lmdb`、`calib.lmdb`、male/female模型 | RGB、深度、点云、网络权重 |
| 预计算 teacher `prepare-teacher` | 原 RT＋划分＋22关节标签 | SMPL-X模型、params/calib、网络权重 |
| 训练补全 `train` | 原 RT＋划分＋22关节标签；续训另需断点 | RGB、深度、点云、SMPL-X模型、params/calib |
| 验证补全 `evaluate` | 原 RT＋划分＋val关节标签＋补全权重 | SMPL-X模型、params/calib |
| 导出补全 `export` | 原 RT＋划分＋完整轮验证后的 `best.pt/last.pt` | 所有关节标签、teacher、SMPL-X模型、params/calib |

teacher 缓存在训练/验证时可选；省略 `--teacher-cache` 会在线计算，不是完全取消时序监督。`prepare-labels` 虽不读取 RT 强度，也会对 `radar_comp.lmdb` 做源身份检查，因此该文件仍须存在。

### 4.2 最小迁移目录：复用已生成标签

```text
/data/M4Human-main/
├── rt_completion/                     # 必须带当前本地代码
├── dataset/                           # 至少含数据/LMDB公共工具；建议复制完整代码目录
├── configs/rt_completion_s2_v2.yaml
├── requirements/
├── MR-Mesh/
│   ├── rf3dpose_all/
│   │   ├── radar_comp.lmdb             # 原始 RT
│   │   └── indeces.pkl.gz              # 原项目拼写 indeces，不是 indices
│   ├── rt_completion_labels_s2/
│   │   ├── joints.lmdb                 # train/val 22关节标签
│   │   └── manifest.json
│   └── rt_teacher_s2_v2/               # 可选，建议迁移以加速
│       ├── teacher.lmdb
│       └── manifest.json
└── experiments/rt_completion_s2_v2_safe/ # 续训/使用已有模型时迁移
    ├── progress.pt
    ├── last.pt
    ├── best.pt
    ├── config.json
    └── metrics.jsonl
```

### 4.3 文件大小与空间预算

以下为当前本地文件的实际大小，单位 GiB＝2³⁰ bytes，不是压缩下载包大小。

| 文件 | 约占空间 | 用途 |
| --- | ---: | --- |
| `radar_comp.lmdb` | 35.003 GiB | 补全所有阶段的原 RT |
| `indeces.pkl.gz` | 0.008 GiB | 原划分及帧索引 |
| `rt_completion_labels_s2/joints.lmdb` | 0.264 GiB | 已完成的22关节监督 |
| `rt_teacher_s2_v2/teacher.lmdb` | 9.390 GiB | 已完成的固定时序伪 GT |
| `params.lmdb` | 2.557 GiB | 从零准备标签/下游训练 |
| `calib.lmdb` | 0.474 GiB | 从零准备标签/下游训练 |
| `radar_pc.lmdb` | 6.140 GiB | 纯补全不需要；原下游入口仍需要 |
| `indicator.lmdb` | 0.031 GiB | 原下游入口需要 |

仅复用标签训练补全，最小数据约 **35.28 GiB**；加现有 teacher 约 **44.66 GiB**。还需为环境、下载解压临时文件和新导出结果留空间。建议完整补全流程预留至少100 GiB可用磁盘；若迁移全项目数据和多个导出结果，应按实际容量进一步增加。

当前协议为 `p1/s2`：train 493,925帧、val 33,987帧、test 133,211帧。输入4帧，单帧体素网格 `121×111×31`，XYZ轴顺序。

## 5. 只在需要从零生成标签时增加这些文件

```text
MR-Mesh/rf3dpose_all/
├── radar_comp.lmdb
├── indeces.pkl.gz
├── params.lmdb
└── calib.lmdb

models/smplx/
├── SMPLX_MALE.npz
├── SMPLX_FEMALE.npz
├── SMPLX_MALE.pkl
└── SMPLX_FEMALE.pkl
```

**当前代码准备标签时两种扩展名都需要保留**：预检/签名读取 male/female 的 PKL；实际 `smplx.create(...)` 未指定 `ext`，当前包默认加载 NPZ。每个PKL约520 MiB、NPZ约104 MiB。不是只安装 `smplx` Python包就有模型。

模型应通过 [SMPL-X 官方网站](https://smpl-x.is.tue.mpg.de/) 按要求注册、同意许可后获取。不要把受许可限制的模型上传到公开仓库。下游训练还需 neutral NPZ，见第9节。

## 6. 迁移已有缓存与断点：最容易出错的地方

### 6.1 优先保留相同实际绝对路径和文件时间戳

现有缓存/检查点严格绑定：

- `/data/M4Human-main/MR-Mesh/rf3dpose_all` 的实际绝对路径。
- `radar_comp.lmdb` 文件大小和纳秒修改时间，划分文件 SHA256。
- 标签签名，teacher 还绑定 labels 的实际路径及 `joints.lmdb` 大小/纳秒时间。
- teacher 的 loss 参数、胶囊半径，以及 `geometry.py` 文件 SHA256。
- 导出权重内嵌的数据路径；`export/evaluate` 不是另给 `--config` 就能重定位。

因此，**不能只改 YAML 路径，就保证旧标签、teacher和权重能继续使用**。优先在服务器使用相同实际目录；`Path.resolve()` 会展开软链接，因此仅创建指向另一实际目录的软链接可能不够。若无权使用 `/data/M4Human-main`，需要管理员提供合适挂载，或另做经过校验的迁移适配。

使用保留mtime的复制方式，例如 `rsync -a`，且目标文件系统要支持相应纳秒精度。重新下载/解压的文件即使内容相同，mtime也可能不同，旧身份检查仍可能失败。不要为了绕过错误随意改manifest或删除校验。

新路径下从零训练可以重新准备标签和teacher，使用新实验目录；但不能因此把旧checkpoint直接当作可恢复断点。

### 6.2 本地向新服务器复制的示例

以下命令在**当前本地机器**执行，先把 `your_user@your_server` 改为真实账号和主机。只适用于明确用于本项目的新目标目录；不会自动由本说明执行。

```bash
RT_SERVER_HOST='your_user@your_server'

ssh "$RT_SERVER_HOST" 'mkdir -p /data/M4Human-main/MR-Mesh/rf3dpose_all /data/M4Human-main/MR-Mesh/rt_completion_labels_s2 /data/M4Human-main/MR-Mesh/rt_teacher_s2_v2 /data/M4Human-main/experiments/rt_completion_s2_v2_safe'

# 复制代码，不重复传大数据/所有历史实验/模型文件。
rsync -aP --exclude='MR-Mesh/' --exclude='models/' --exclude='experiments/' \
  --exclude='.git/' --exclude='__pycache__/' --exclude='*.log' \
  --exclude='.aws/' --exclude='.codex/' --exclude='.agents/' --exclude='.ssh/' \
  --exclude='.env' --exclude='.env.*' \
  /data/M4Human-main/ "$RT_SERVER_HOST:/data/M4Human-main/"

rsync -aP \
  /data/M4Human-main/MR-Mesh/rf3dpose_all/radar_comp.lmdb \
  /data/M4Human-main/MR-Mesh/rf3dpose_all/indeces.pkl.gz \
  "$RT_SERVER_HOST:/data/M4Human-main/MR-Mesh/rf3dpose_all/"

rsync -aP --exclude='*.lmdb-lock' --exclude='.writer.lock' \
  /data/M4Human-main/MR-Mesh/rt_completion_labels_s2/ \
  "$RT_SERVER_HOST:/data/M4Human-main/MR-Mesh/rt_completion_labels_s2/"

rsync -aP --exclude='*.lmdb-lock' --exclude='.writer.lock' \
  /data/M4Human-main/MR-Mesh/rt_teacher_s2_v2/ \
  "$RT_SERVER_HOST:/data/M4Human-main/MR-Mesh/rt_teacher_s2_v2/"

# 在源训练已保存稳定断点后复制；不要一边跨机续训一边混写同一实验历史。
rsync -aP --exclude='.writer.lock' --exclude='*.tmp' \
  /data/M4Human-main/experiments/rt_completion_s2_v2_safe/ \
  "$RT_SERVER_HOST:/data/M4Human-main/experiments/rt_completion_s2_v2_safe/"
```

迁移前检查项目中的额外私人文件。上述代码复制排除了常见凭据和代理会话目录；这些不属于项目依赖，不应上传到服务器。LMDB锁文件不是数据，不要求迁移。原 RT、标签和teacher数据文件应保持只读，不使用 LMDB 重打包/压缩后替换来迁移旧身份。

### 6.3 在服务器检查数据和缓存签名

```bash
cd /data/M4Human-main
conda activate m4human

python - <<'PY'
from rt_completion.pipeline import load_config
from rt_completion.data import RTSequenceDataset
from rt_completion.teacher_cache import TeacherCache

cfg = load_config('configs/rt_completion_s2_v2.yaml')
labels = 'MR-Mesh/rt_completion_labels_s2'
for split in ('train', 'val'):
    ds = RTSequenceDataset(cfg['data'], split, labels)
    print(split, len(ds), tuple(ds[0]['rt'].shape))
    cache = TeacherCache('MR-Mesh/rt_teacher_s2_v2', cfg, labels, split)
    print('teacher:', tuple(cache.get(ds.indicators[0])['intensity'].shape))
print('Data and teacher provenance checks passed')
PY
```

未迁移可选teacher时，删除上述检查块中 `TeacherCache` 相关三行即可；训练不传 `--teacher-cache`。

当前源RT身份可供比对：大小 `37583953920` bytes、mtime_ns `1789211583176046815`、indices SHA256 `410ef39ff23a90fda1db7ee66010046988050b1c79dd54b58901fd675f014f97`。

## 7. 在服务器运行补全代码

统一先执行：

```bash
cd /data/M4Human-main
conda activate m4human
export CUDA_VISIBLE_DEVICES=0
```

当前检查点保存了1张可见GPU的随机状态。多卡服务器先只暴露1张GPU，可避免续训时设备数量校验不匹配；需要使用服务器别的物理卡时，把 `0` 换成该卡编号，程序内仍用 `cuda:0`。补全入口是单进程，不使用torchrun。

### A. 从当前断点继续训练：已经迁移标签和teacher

```bash
python -m rt_completion train \
  --config configs/rt_completion_s2_v2.yaml \
  --labels MR-Mesh/rt_completion_labels_s2 \
  --teacher-cache MR-Mesh/rt_teacher_s2_v2 \
  --output experiments/rt_completion_s2_v2_safe \
  --resume experiments/rt_completion_s2_v2_safe/progress.pt \
  --workers 4 --device cuda:0
```

保持原 batch=2、窗口和训练设置，`--output`仍是断点原目录。不能直接把batch改大后恢复。换GPU型号或软件构建可能产生浮点差异，不保证逐bit复现。

### B. 从零训练：标签已有，teacher未迁移

先运行可选的teacher预计算，再在新实验目录训练：

```bash
python -m rt_completion prepare-teacher \
  --config configs/rt_completion_s2_v2.yaml \
  --labels MR-Mesh/rt_completion_labels_s2 \
  --output MR-Mesh/rt_teacher_s2_v2 \
  --splits train val --batch-size 2 --workers 4 --device cuda:0 --resume && \
python -m rt_completion train \
  --config configs/rt_completion_s2_v2.yaml \
  --labels MR-Mesh/rt_completion_labels_s2 \
  --teacher-cache MR-Mesh/rt_teacher_s2_v2 \
  --output experiments/rt_completion_s2_v2_server \
  --workers 4 --device cuda:0
```

若标签也没有，先准备第5节文件并执行：

```bash
python -m rt_completion prepare-labels \
  --config configs/rt_completion_s2_v2.yaml \
  --smplx-models models \
  --output MR-Mesh/rt_completion_labels_s2 \
  --device cuda:0 --resume
```

这些缓存命令会检查已有目录身份。已有缓存不匹配时，不要直接覆盖；使用新的独立目录，或者先处理迁移路径问题。

### C. 只用训练好的模型导出补全RT

```bash
python -m rt_completion export \
  --checkpoint experiments/rt_completion_s2_v2_safe/best.pt \
  --output MR-Mesh/rt_completed_s2_v2 \
  --splits train val test \
  --batch-size 2 --workers 4 --device cuda:0 --resume
```

导出只需要原 RT、划分和权重；`progress.pt`若是未完成轮的断点不能导出。输出 `additions.lmdb` 存稀疏新增体素，必须与原 RT 合并才能恢复补全RT，不能脱离原数据独立使用。

已有 `MR-Mesh/rt_completed_s2` 是旧baseline结果，不是本次V2结果；不要与V2权重混淆。

## 8. teacher与补全结果不是一回事

| 文件 | 含义 | 后续是否需要 |
| --- | --- | --- |
| `joints.lmdb` | SMPL-X转换的22关节监督 | 训练/验证需要 |
| `teacher.lmdb` | 历史实测RT经GT部位对齐的弱监督 | 训练/验证可选加速项 |
| `best.pt` | 验证集选出的补全网络 | 导出需要 |
| `additions.lmdb` | 网络推理保存的新增RT体素 | 使用补全RT的下游需要 |

## 9. 若还要训练下游原 RT-Mesh

### 9.1 环境更多

当前 `main1_multigpu_clean.py` 无条件导入P4Transformer和RETR，所以即便配置为RT-Mesh，也会要求 `pointnet2_ops._ext` 可导入。纯 `python -m rt_completion ...` 不受此依赖影响。

先安装配套 torchvision 和额外依赖：

```bash
python -m pip install torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements/rtmesh-extra.txt
```

PointNet2需按 [官方源码](https://github.com/erikwijmans/Pointnet2_PyTorch) 在目标环境编译。当前本机是 `pointnet2-ops 3.0.0`，来自该仓库 `pointnet2_ops_lib`，不是保证能从普通PyPI直接安装的标准wheel。

在目标服务器确认兼容的CUDA 11.8 nvcc、C++工具链及正确CUDA_HOME后，可按以下源码安装方式测试：

```bash
nvcc --version
g++ --version
python -m pip install ninja

# CUDA_HOME应指向服务器实际的兼容toolkit，不能照抄一个不存在的路径。
# export CUDA_HOME=/usr/local/cuda-11.8
# 先完成工具链核对，再编译；需要GPU可见以正确选择目标架构。
python -m pip install --no-build-isolation \
  'https://github.com/erikwijmans/Pointnet2_PyTorch/archive/refs/heads/master.zip#subdirectory=pointnet2_ops_lib'

python -c "from pointnet2_ops import pointnet2_utils; print('PointNet2 extension import OK')"
```

这是目标服务器需验证的源码构建步骤，并非本次已在服务器测试成功的承诺。master可变化，严格复现应使用已验证源码版本；当前安装归档SHA256保存在审计快照中。不要直接迁移原3090编译的 `.so` 到另一PyTorch版本或GPU架构，也不要硬编码3090的架构8.6用于其他GPU。PyTorch wheel自带运行库不代表已安装nvcc。

无图形库的服务器若 `cv2` 报libGL缺失，应由管理员补系统库，或在新环境中选择同版本 `opencv-python-headless` 代替 `opencv-python`，不要同时安装两者。

### 9.2 数据也更多

原下游Dataset即使使用RT也无条件加载以下五个LMDB：

```text
MR-Mesh/rf3dpose_all/
├── radar_comp.lmdb
├── radar_pc.lmdb
├── params.lmdb
├── calib.lmdb
├── indicator.lmdb
└── indeces.pkl.gz
```

另需 `models/smplx/SMPLX_{MALE,FEMALE,NEUTRAL}.npz`，并在项目根目录运行。使用补全RT时再加完整的 `rt_completed_s2_v2/{manifest.json,additions.lmdb}`。默认不使用RGB时不需要图像和深度；关节标签cache及teacher则不是下游输入。

只运行补全网络时，不要因为本节额外下载点云/图像或编译PointNet2。

## 10. 代码依据与进一步说明

- [最小RT数据读取](../rt_completion/data.py)：RT与22关节标签。
- [标签、训练与导出命令](../rt_completion/pipeline.py)。
- [源数据身份校验](../rt_completion/storage.py)、[teacher校验](../rt_completion/teacher_cache.py)。
- [V2模型说明](rt_completion_v2.md)、[缓存加速与续训说明](rt_completion_speed.md)。
- [M4Human官方仓库](https://github.com/FanJunqiao/M4Human)、[官方数据页面](https://fanjunqiao.github.io/M4Human-site/)。

本文件提供环境和数据准备说明，不表示已经完成服务器迁移、全量训练、或跨服务器结果复现。
