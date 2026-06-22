# AEGIS

AEGIS 是一个面向工业设备声音异常检测的分阶段实验项目。仓库保留两个对照组源码不变，
新增实现全部位于 `aegis/`：

- 阶段 1：2D log-Mel 输入的 Conv-AE，使用平均重构误差作为异常分数；
- 阶段 2：在 Conv-AE 编码器中加入频率轴注意力；
- 阶段 3：加入自监督变换分类头，融合重构分数与分类分数。

## 仓库结构

```text
AEGIS/
├── aegis/                         # AEGIS 实现、配置、测试与报表工具
├── dcase2023_task2_baseline_ae/   # 官方 Dense-AE（Git submodule，未修改）
├── STgram-MFN/                    # STgram-MFN（Git submodule，未修改）
└── CODEX.md                       # 实验设计说明
```

克隆时记得同时拉取两个对照组：

```bash
git clone --recurse-submodules <your-repository-url>
```

已有普通 clone 可执行：

```bash
git submodule update --init --recursive
```

## 环境与数据

```bash
pip install -r aegis/requirements.txt
```

按照官方 Dense-AE README 的目录约定，将开发集放到：

```text
dcase2023_task2_baseline_ae/data/dcase2020t2/dev_data/raw/
dcase2023_task2_baseline_ae/data/dcase2024t2/dev_data/raw/
```

特征、训练参数、数据集机器列表和融合权重集中配置在
`aegis/config.yaml`。官方 DCASE2024 开发集映射包含 7 类机器；如果实验必须固定为 6 类，
请使用 `--machine-types` 显式传入选定子集，避免代码静默丢弃某一类。

### 消融实验配置

独立配置位于 `aegis/configs/`：

| 配置 | 累加内容 | 频率注意力 | 自监督分类头与融合 |
| --- | --- | :---: | :---: |
| `01_stage1_conv_ae.yaml` | Conv-AE | × | × |
| `02_stage2_frequency_attention.yaml` | 阶段 1 + 频率注意力 | ✓ | × |
| `03_stage3_full_aegis.yaml` | 阶段 2 + 自监督分类与分数融合 | ✓ | ✓ |

每个文件继承 `aegis/config.yaml` 的公共训练参数，只覆盖该实验需要改变的组件，输出目录
也按实验名隔离。

## 分阶段运行

先用一个机器类型做真实数据 smoke test：

```bash
python -m aegis.run --dataset DCASE2020T2 --stage 1 \
  --machine-types fan --epochs 1 --max-train-batches 2 --max-test-files 10
```

确认数据链路后运行完整实验：

```bash
python -m aegis.run --config aegis/configs/01_stage1_conv_ae.yaml --dataset DCASE2020T2
python -m aegis.run --config aegis/configs/02_stage2_frequency_attention.yaml --dataset DCASE2020T2
python -m aegis.run --config aegis/configs/03_stage3_full_aegis.yaml --dataset DCASE2020T2
```

将最后一个参数换成 `--dataset DCASE2024T2` 即可跑另一数据集。

输出位于 `aegis/outputs/<dataset>/<experiment_name>/`，包括 checkpoint、逐文件异常分数、
section 指标和机器类型均值。

## 测试与最终报表

不依赖真实数据的三阶段 smoke tests：

```bash
python -m unittest discover -s aegis/tests -p "test_*.py" -v
```

将两个未修改对照组的跑分整理成 `aegis/reference_results.example.csv` 所示格式后，生成
跨数据集对比表与消融表：

```bash
python -m aegis.report --reference-csv path/to/baseline_results.csv
```

结果写入 `aegis/outputs/reports/comparison.csv` 和 `ablation.csv`。
