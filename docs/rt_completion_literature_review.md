# RT 补全网络改进：补充文献与可迁移设计

检索整理日期：2026-10-13。本文区分论文实际方法与针对本项目提出的改造建议；后者尚未实现、尚未验证，不代表已有论文证明了其在本项目上的效果或首创性。

本项目输入为四帧因果 XYZ 单通道强度 RT，单帧尺寸为 121×111×31。现有监督包括人工遮挡恢复、历史回波伪目标和 SMPL-X 派生的 22 关节标签。导出保存新增体素，并保留原始非零观测。当前输入没有 Doppler、复数相位或 ADC；也没有完整、无缺失的 RT 强度真值。以下迁移分析均以这些约束为前提。

## 1. 监督：表征、人体先验与测量恢复分别学习

### 1. Wave2Body: Rethinking mmWave Human Pose Estimation as Radar-to-Body Token Translation

**状态：2026 年预印本。** [作者论文](https://arxiv.org/html/2607.18875v1)、[官方代码](https://github.com/Galaxywalk/Wave2Body)。

论文分别学习自监督雷达 tokenizer 和由姿态数据训练的部位组合 VQ-VAE，再冻结两者学习雷达到人体 token 的映射；在 M4Human 和 mmBody 上评估。可借鉴到本项目：先预训练 RT 表征与人体结构教师，再让 RT 分支预测部位特征，形成关节坐标之外的辅助目标。人体 token 仍可能合理但错误；论文输出姿态而非完整 RT，不能将其作为雷达强度真值。

### 2. Complete-to-Partial 4D Distillation for Self-Supervised Point Cloud Sequence Representation Learning（C2P）

**状态：CVPR 2023。** [会议论文](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Complete-to-Partial_4D_Distillation_for_Self-Supervised_Point_Cloud_Sequence_Representation_Learning_CVPR_2023_paper.pdf)。

论文让教师看原始点云序列，学生看模拟裁剪后的部分序列，蒸馏窗口时空知识。可设计原始四帧 RT 教师与人工遮挡 RT 学生，在部位对齐的特征上监督。论文的 complete 指原始采集序列，并不表示物理完整；教师也不能提供天然缺失位置的完整强度标签。

### 3. Cross-Modal Masked Autoencoders for 4D Radar: Learning From Lidar to Boost Radar Object Detection Performance（RaLi-AE）

**状态：IEEE Access 2026。** [出版社页面](https://ieeexplore.ieee.org/document/11574675/)。此次主要依据出版社摘要和元信息核实，未完整核读全文。

论文以稀疏雷达点云重建稠密 LiDAR 点云进行自监督预训练，再用于 radar-only 检测。可借鉴：训练期重建 SMPL-X 派生的部位或表面距离等几何目标，同时保留独立的 RT 测量恢复目标。LiDAR 或人体网格几何不能定义真实雷达反射强度；其检测收益也不等于人体 RT 补全收益。

**本方向候选设计：** 使用三种不同目标：实测回波留出恢复、教师时空表征蒸馏、人体结构重建。分别消融，避免把所有目标混称为完整 RT 伪 GT。人体教师和雷达教师的训练均不得使用测试划分。

## 2. Flow / 生成：将先验、观测一致性与计算开销分开处理

### 4. FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems

**状态：ICCV 2025。** [会议原文](https://openaccess.thecvf.com/content/ICCV2025/html/Kim_FlowDPS__Flow-Driven_Posterior_Sampling_for_Inverse_Problems_ICCV_2025_paper.html)、[作者论文](https://arxiv.org/html/2503.08136v1)。

论文基于已有 flow 先验，将生成更新分为干净估计和噪声部分，加入似然梯度与随机性解决图像逆问题。可研究在 RT 生成步骤中加入实测位置的一致性，并逐步调整几何和历史证据指导。其无需额外训练建立在已有有效生成先验上；本项目仍需学习 RT 先验，天然零值也不能全部纳入真实零值观测约束。

### 5. Unsupervised Radar Point Cloud Enhancement via Arbitrary LiDAR Guided Diffusion Prior（代码名 RadarINV）

**状态：2025 年预印本。** [作者论文](https://arxiv.org/html/2505.09887v1)。查到 ICLR 2026 投稿版本，但未核实正式录用，不标为正式会议论文。

论文把雷达角度恢复写成逆问题，用任意 LiDAR 数据训练的扩散先验与雷达前向成像约束结合，减少成对雷达—LiDAR数据需求。可借鉴人体几何先验与 RT 测量约束分工。它需要合理前向算子及噪声模型，LiDAR 几何先验不等于 RT 强度先验；不能把现有 XYZ 缓存直接视为完整物理测量链。

### 6. Towards Dense and Accurate Radar Perception Via Efficient Cross-Modal Diffusion Model（Radar-Diffusion）

**状态：IEEE RA-L 2024。** [作者论文](https://arxiv.org/html/2403.08460v1)、[官方代码](https://github.com/ZJU-FAST-Lab/Radar-Diffusion)。

论文利用雷达 range–azimuth 热图条件生成 LiDAR BEV，再通过 consistency distillation 获得单步生成。可借鉴离线补全中的多步教师到少步学生蒸馏，用于测量质量—导出时间权衡。它生成二维几何表示，不证明 RT 强度恢复；图像感知损失也不应直接作为 RT 的物理质量标准。该工作与其他同名 Radar-diffusion 论文需区分。

**本方向候选设计：** 先在有目标的位置比较确定性解码与条件残差 flow；保留实测、历史支持和独立反射选择分支。只有质量提高后再蒸馏。若没有完整 RT 先验，先从人工留出实测和可信时序目标学习，不能因更换生成方法而假定完整终点已存在。

## 3. Mask：时空、跨尺度与噪声退化共同设计

### 7. MAEPose: Self-Supervised Spatiotemporal Learning for Human Pose Estimation on mmWave Video

**状态：2026 年预印本，核读 v2。** [作者论文](https://arxiv.org/html/2605.00242v2)。

论文对 RD/RA 谱图视频做高比例时空 patch 遮挡预训练，再接多帧二维关节热图头。可借鉴先 RT 自监督、后 SMPL-X 几何监督的分阶段训练，以及概率热图辅助目标。输入是谱图视频，输出是二维姿态；不能直接照搬其 90% mask 比例到本来稀疏的四帧 XYZ RT。

### 8. Point-M2AE: Multi-scale Masked Autoencoders for Hierarchical Point Cloud Pre-training

**状态：NeurIPS 2022。** [会议论文](https://papers.neurips.cc/paper_files/paper/2022/hash/ad1d7a4df30a9c0c46b387815a774a84-Abstract-Conference.html)。

论文使用层级编码解码和跨尺度一致的可见区域。可迁移到 RT 多尺度特征：同一观测遮挡在强度、非零指示和所有由该输入计算的 skip 中保持一致。现有代码从同一 corrupted RT 编码，已具备基本一致性；未来加入独立教师或多分支时需防止未遮输入泄露目标。固定 XYZ 网格坐标本身不是标签泄漏。

### 9. PointCMP: Contrastive Mask Prediction for Self-Supervised Learning on Point Cloud Videos

**状态：CVPR 2023。** [会议论文](https://openaccess.thecvf.com/content/CVPR2023/papers/Shen_PointCMP_Contrastive_Mask_Prediction_for_Self-Supervised_Learning_on_Point_Cloud_CVPR_2023_paper.pdf)。

论文采用局部与全局分支，并通过 mutual similarity 构造特征层困难遮挡与对比预测。可让 RT 网络同时学习局部部位恢复与动作上下文，加入避免简单邻帧复制的特征监督。它不是简单的 raw 空间/时间双 mask；特征恢复与真实强度恢复仍需分别评估。

### 10. mmSimPrior: Learning Simulation Priors for Data-Efficient Real-World Generalizable Radar-Based Human Motion Reconstruction

**状态：2026 年预印本。** [作者论文](https://arxiv.org/html/2607.22973v1)。

论文区分信号、运动和映射先验，并使用逐渐加重的多径假回波、空间扩散和响应统计扰动帮助仿真到真实迁移。可借鉴将 RT mask 扩展为缺失、衰减、模糊与空间相关污染的退化课程。其仿真及热图处理链与当前数据不同；移植的是训练机制，不能把扰动参数解释成当前传感器已知物理噪声分布。

**本方向候选设计：** 混合部位单帧缺失、随肢体运动的连续缺失、雷达扇区擦除、回波衰减和错位历史。目标只在原本实测或可信伪目标处成立。噪声注入与回波删除需要分别评测；四帧都缺少某部位时，允许高不确定性和拒绝新增。

## 4. 时序与可信度：让权重有监督、让不确定性可检验

### 11. UNSPAT: Uncertainty-Guided SpatioTemporal Transformer for 3D Human Pose and Shape Estimation on Videos

**状态：WACV 2024。** [会议论文](https://openaccess.thecvf.com/content/WACV2024/html/Lee_UNSPAT_Uncertainty-Guided_SpatioTemporal_Transformer_for_3D_Human_Pose_and_Shape_WACV_2024_paper.html)。

论文先对齐视频特征，再做逐位置时序 attention；通过跨视频特征替换等合成污染训练不确定性估计并重加权融合。可在 RT 部位特征中注入错位历史、污染或假回波，为可靠性分支提供可控标签。源任务是 RGB 人体网格估计；合成污染识别不能直接称真实雷达噪声概率，需验证其对恢复误差的排序能力。

### 12. RadProPoser: Probabilistic Radar Tensor Human Pose Estimation That Knows Its Limits

**状态：IJCNN 2026，作者 v2 明确标注录用；2025 首版标题不同。** [作者论文 v2](https://arxiv.org/html/2508.03578v2)、[发表信息](https://arxiv.org/abs/2508.03578)。

论文从复值雷达张量预测关节分布，结合谱 attention 和 Gaussian/Laplace 不确定性，并进行 isotonic 校准。可为当前关节头增加分布，使低可信关节降低胶囊及部位对齐权重；对 RT 强度不确定性另做留出回波校准。当前数据缺少相位，不能照搬谱分支；关节置信度也不是体素强度置信度。

### 13. mmDiffusion: mmWave Diffusion for Sequential 3D Human Dense Point Cloud Generation

**状态：3DV 2025。** [作者机构论文与全文入口](https://discovery.ucl.ac.uk/id/eprint/10226807/)。机构库内 PDF 文件名含 CVPR2024，但发表信息明确为 3DV 2025，引用应依发表信息。

论文将序列 mmWave 编码与双向 GRU 条件输入扩散生成器，生成时序一致的稠密人体点云。可作为四帧时序条件生成基线的参考，并增加运动对齐后的时序稳定性指标。它生成身体几何而非雷达强度；双向处理协议需与本项目因果窗口区分，不能将未来信息带入因果对照。

**本方向候选设计：** 部位对齐后计算逐体素可靠度，加入无证据选项；用留出恢复误差训练/校准可靠度。当前零值处不能以“与历史非零差异大”为理由降低历史权重。强度低、跨帧不稳定或人体外观测均不能单独证明噪声。

## 5. 胶囊与人体几何：稳定端点、局部查询与反射属性分工

### 14. Diffusion Model is a Good Pose Estimator from 3D RF-Vision（mmDiff）

**状态：ECCV 2024。** [作者论文](https://arxiv.org/html/2403.16198v2)。

论文用关节级全局上下文、围绕预测关节的局部雷达查询、骨长与历史粗姿态指导姿态扩散。可让 15 个部位 token 查询对应 RT 特征，并随关节修正更新胶囊。论文生成的是姿态；局部无回波时不能把最近的远处观测强行视为支持，也不能据此获得缺失强度标签。

### 15. mmPose-FK: A Forward Kinematics Approach to Dynamic Skeletal Pose Estimation Using mmWave Radars

**状态：IEEE Sensors Journal 2024。** [作者全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11415245/)、[作者实验室发表信息](https://radar.ece.arizona.edu/publications/mmpose-fk-forward-kinematics-pose-estimation-2024/)。

论文从雷达体素序列估计骨长与旋转，通过可微正向运动学生成骨架。可把当前坐标预测头与“骨长/旋转/根位置→FK→胶囊”比较，降低骨段长度和方向跨帧跳变。其输入含 Doppler/SNR，当前无法复刻；FK 约束的是骨架，不提供雷达强度真值。

### 16. DART: Implicit Doppler Tomography for Radar Novel View Synthesis

**状态：CVPR 2024。** [会议论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_DART_Implicit_Doppler_Tomography_for_Radar_Novel_View_Synthesis_CVPR_2024_paper.pdf)、[官方代码](https://github.com/WiseLabCMU/dart)。

论文通过雷达物理渲染学习反射和透射属性，合成新视角 range–Doppler 观测。它支持把几何位置与雷达响应属性分别建模的研究方向。当前没有 Doppler 和相应测量条件，动态人体也不同于原论文静态场景；因此适合较远期物理建模，不宜直接当作现成补全模块。

**本方向候选设计：** 稳定骨架或人体 token 负责部位结构；预测关节分布产生胶囊期望与分歧；RT 局部查询和历史支持负责反射证据。胶囊只作软条件，不要求所有内部位置都有回波，也不能将人体外响应全作负标签。

## 6. 两篇补充参照

### 17. mmPoint: Dense Human Point Cloud Generation from mmWave

**状态：BMVC 2023。** [作者全文](https://www.cs.ox.ac.uk/files/14371/BMVC-2023-mmPoint.pdf)、[官方代码](https://github.com/NUAAXQ/mmPoint)。

论文以雷达特征驱动人体模板点云变形和细化，用几何重建与均匀性损失生成稠密人体点云。可作为模板先验与胶囊先验的对照。它输出几何而非原始雷达响应；均匀人体点集不等于真实回波分布，不宜直接把均匀性损失施加在 RT 强度上。

### 18. Facilitating Radar-Based Gesture Recognition With Self-Supervised Learning（RadarAE）

**状态：IEEE SECON 2022。** [作者全文](https://dapowan.github.io/files/RadarAE.pdf)、[作者主页](https://dapowan.github.io/)。

论文对时间—距离的雷达表示做 masked autoencoder 预训练，采用非对称 patch。可支持依据 RT 时间长度与实际 XYZ 物理间距设计 mask，而非照搬图像方块。其任务为手势识别与 STMM 表征学习，不是人体三维强度恢复。

## 7. 阅读顺序与候选实验

| 优先级 | 文献 | 解决当前问题 | 首个可验证实验 |
|---|---|---|---|
| 1 | C2P、Wave2Body | 除了关节 L1/L2，怎样提供结构和时空监督 | 同一恢复模型增加部位特征蒸馏，检验留出误差和独立下游姿态 |
| 1 | UNSPAT、RadProPoser | 时序权重与不确定性怎样获得训练依据 | 注入受控污染训练可靠度，并验证误差—覆盖率关系 |
| 1 | mmPose-FK、mmDiff | 胶囊怎样稳定且利用局部回波 | 固定骨架头对比运动学/局部查询头，测量关节、胶囊和补全变化 |
| 2 | MAEPose、Point-M2AE、mmSimPrior | 掩码、预训练和噪声课程怎样设计 | 固定多类 mask 评测，比较随机、部位、持续缺失和污染训练 |
| 2 | FlowDPS、RadarINV | 生成先验怎样受真实观测约束 | 有证据集合上的条件 flow，与相同条件和预算的确定性模型对照 |
| 3 | Radar-Diffusion | 多步生成怎样降低离线开销 | 教师质量确认后比较蒸馏步数与导出时间 |
| 3 | DART | 是否进一步研究物理反射与可观测性 | 明确传感器与处理链后，先验证前向模型再使用物理似然 |

建议报告的共同指标：固定人工留出测量的 log/原始强度误差；按部位、强度、距离、缺失持续时间分组；已知合成污染下的错误新增；关节与强度各自的校准；独立下游 RT-Mesh 的 MPJPE/MVE；离线导出耗时和新增体素预算。

天然未知体素缺少完整真值时，不报告未经证实的全体积补全精度；人体外新增只能作为几何诊断，不能直接等同真实雷达假回波。所有对照须控制因果窗口、数据划分、模型选择、额外人体先验数据和有效原始时间范围。

这些论文说明 masked 学习、人体先验、生成后验、时序加权与不确定性校准分别已有充分先例。候选研究问题可以落在它们如何共同处理本项目的“动态人体、XYZ 强度 RT、无完整 GT、局部可信补全与观测保留”，但上述文献整理不构成首创性证明。
