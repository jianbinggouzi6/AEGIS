# AEGIS — 工业设备声学异常检测

基于 DCASE Task 2 基准的工业设备声音异常检测研究框架。  
所有模型代码位于 `aegis/`，两个对照组仓库作为 git submodule **保持原样不动**。

---

## 仓库结构

```
AEGIS/
├── aegis/                              # AEGIS 核心实现
│   ├── models.py                       # Conv-AE + 频率注意力 + 分类头
│   ├── engine.py                       # 训练器、逐 section 校准、评分
│   ├── train.py                        # 训练入口
│   ├── data.py                         # 基线 loader 适配器
│   ├── metrics.py                      # AUC / pAUC / F1
│   ├── report.py                       # 跨数据集对比表 + 消融表生成
│   └── requirements.txt
├── configs/                            # 6 个消融实验 YAML 配置
│   ├── exp1_convae.yaml                # 基座 Conv-AE（数据集 A）
│   ├── exp2_convae_freqattn.yaml       # +频率注意力（数据集 A）
│   ├── exp3_full_aegis.yaml            # 完整 AEGIS（数据集 A）
│   ├── exp1_convae_b.yaml              # 基座 Conv-AE（数据集 B）
│   ├── exp2_convae_freqattn_b.yaml     # +频率注意力（数据集 B）
│   └── exp3_full_aegis_b.yaml          # 完整 AEGIS（数据集 B）
├── run_all.sh                          # 一键顺序运行全部 6 个实验
├── dcase2023_task2_baseline_ae/        # 官方 Dense-AE（submodule，不改）
├── STgram-MFN/                         # STgram-MFN 强基线（submodule，不改）
├── CODEX.md                            # 实验设计说明
└── README.md
```

---

## 环境安装

```bash
# 克隆仓库（含两个对照组 submodule）
git clone --recurse-submodules git@github.com:<你的用户名>/AEGIS.git
cd AEGIS

# 安装依赖
pip install -r aegis/requirements.txt
```

---

## 数据集准备

将开发集数据放到基线仓库的 `data/` 目录下，目录结构如下：

```
dcase2023_task2_baseline_ae/data/
├── dcase2020t2/dev_data/raw/<机器类型>/train/   ← 数据集 A 训练音频
│                                        test/
└── dcase2024t2/dev_data/raw/<机器类型>/train/   ← 数据集 B 训练音频
                                         test/
```

loader 首次运行时会自动将 mel 特征缓存为 `.pickle` 文件，之后复用。

**数据集 A** — DCASE2020 Task 2  
机器类型：`fan  pump  slider  valve  ToyCar  ToyConveyor`

**数据集 B** — DCASE2024 Task 2  
机器类型：`bearing  fan  gearbox  slider  valve  ToyCar  ToyTrain`

> **注意：** DCASE2024T2 开发集每台机器只有 1 个 section，
> 因此数据集 B 的实验中 `classifier_fusion` 会自动关闭并打印警告。

---

## 模型开关说明

YAML 配置文件中 `model:` 部分的三个布尔开关对应消融表的三个维度：

| 开关 | YAML 键 | 作用 |
|------|---------|------|
| `conv_ae` | `model.conv_ae` | 2D 卷积自编码器主干，始终为 `true` |
| `freq_attention` | `model.freq_attention` | 频率轴注意力（在第一个编码块后插入） |
| `classifier_fusion` | `model.classifier_fusion` | Section-ID 分类头 + 异常分融合 |

分类相关参数（`classifier:` 节）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_classes` | `0`（自动） | section 数量，`0` 表示从数据自动检测 |
| `loss_type` | `"ce"` | `"ce"` 交叉熵 / `"arcface"` ArcFace 角度 margin |
| `lambda_cls` | `0.2` | λ，联合 loss 中分类损失的权重 |

融合参数（`fusion:` 节）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `weight` | `0.3` | w，融合公式：`异常分 = z_重构 + w × z_分类` |

分阶段训练开关（`training:` 节）：

```yaml
training:
  staged:     true   # 先训 AE，再微调分类头
  ae_epochs:  30     # AE 单独训练轮数
  clf_epochs: 20     # 分类头微调轮数（编码器/解码器冻结）
```

---

## 消融实验设计

六个配置文件实现**叠加式消融**——后一个在前一个基础上只多开一个开关，
其余超参（epoch、batch size、lr、λ、w、随机种子）完全一致，保证可比性：

| 配置文件 | 数据集 | Conv-AE | 频率注意力 | 分类融合 |
|----------|--------|:-------:|:---------:|:-------:|
| `exp1_convae.yaml` | A | ✓ | ✗ | ✗ |
| `exp2_convae_freqattn.yaml` | A | ✓ | ✓ | ✗ |
| `exp3_full_aegis.yaml` | A | ✓ | ✓ | ✓ |
| `exp1_convae_b.yaml` | B | ✓ | ✗ | ✗ |
| `exp2_convae_freqattn_b.yaml` | B | ✓ | ✓ | ✗ |
| `exp3_full_aegis_b.yaml` | B | ✓ | ✓ | ✓* |

*数据集 B 每机器仅 1 个 section，分类融合自动关闭。

---

## 运行实验

### 单个实验

```bash
python -m aegis.train --config configs/exp3_full_aegis.yaml
```

命令行覆盖参数：

```bash
python -m aegis.train --config configs/exp3_full_aegis.yaml \
    --device cuda \
    --epochs 100 \
    --batch-size 256
```

只跑部分机器类型（调试用）：

```bash
python -m aegis.train --config configs/exp1_convae.yaml \
    --machine-types fan pump
```

### 一键运行全部 6 个实验

```bash
bash run_all.sh               # CPU
bash run_all.sh --device cuda  # GPU
```

---

## 输出目录结构

每次运行结果写入 `outputs/<数据集>/<实验名>/`：

```
outputs/DCASE2020T2/exp3_full_aegis/
├── config.yaml                        ← 完整解析后的配置（可复现）
├── result.csv                         ← 各机器 AUC/pAUC/F1 + 算术均值
└── <机器类型>/
    ├── model.pt                       ← 模型权重
    ├── history.json                   ← 每轮训练/验证损失
    ├── anomaly_score_section_<id>.csv ← 逐文件异常分
    ├── section_metrics.csv            ← 逐 section AUC/pAUC/F1
    └── summary.csv                    ← 该机器类型汇总
```

`result.csv` 列：`dataset, machine_type, experiment, auc, pauc, precision, recall, f1`  
最后一行为各机器**算术均值**，可直接填入论文对比表。

---

## 生成对比表与消融表

整理好两个对照组的跑分（格式见 `aegis/reference_results.example.csv`）后执行：

```bash
python -m aegis.report \
    --output-dir outputs \
    --report-dir outputs/reports \
    --reference-csv path/to/dense_ae_results.csv \
    --reference-csv path/to/stgram_mfn_results.csv
```

参考 CSV 需包含列：`dataset, method, machine_type, auc, pauc, f1`  
结果写入 `outputs/reports/comparison.csv` 和 `ablation.csv`。

---

## 从 GitHub 完整复现

```bash
git clone --recurse-submodules git@github.com:<你的用户名>/AEGIS.git
cd AEGIS
pip install -r aegis/requirements.txt
# 按第三节说明放置数据集
bash run_all.sh --device cuda
```

---

## 致谢

本仓库在以下开源项目基础上构建（代码保持原样，不做任何修改）：

- **DCASE2023 Task 2 Baseline AE**（NTT Media Intelligence Laboratories）：`dcase2023_task2_baseline_ae/`
- **STgram-MFN**（Liu et al., 2022）：`STgram-MFN/`
