# 上海大学数据挖掘课程实验 · Group 15

文档编号：DM15-SUM-001；修订：0.1；状态：待维护者评审；源码观察基线：`a65747b9ce505fe4e64d847b4d47e68fd686a3ca`。

四组独立实验代码：词向量预处理、情感分类、医学信息抽取及轻量医疗文档 RAG。各实验使用自己的工作目录、数据和环境，不是一个统一安装包。

> 当前说明基于提交 `a65747b9ce505fe4e64d847b4d47e68fd686a3ca` 的代码静态核对。没有重新训练模型或复跑评测；已提交结果属于历史产物。医学文本、图谱和回答仅用于课程实验，不用于诊疗决策。

## 实验导航

| 实验 | 实际目录与入口 | 输入与主要产物 |
| --- | --- | --- |
| 01 文本预处理与 Word2Vec | [word2vec.py](Group%2015%20exp01-data-preprocessing/word2vec.py) | 本地 `train_part_1.csv`；CBOW/Skip-gram 模型、架构图、t-SNE 图 |
| 02 情感分类 | [bert](Group%2015%20exp02-sentiment-classificationn/bert)、[qwen-sentential-classifier](Group%2015%20exp02-sentiment-classificationn/qwen-sentential-classifier)、[textcnn](Group%2015%20exp02-sentiment-classificationn/textcnn) | 本地训练/验证/测试 CSV；分类权重、训练日志与评估指标 |
| 03 医学文档信息抽取 | [exp3.ipynb](Group%2015%20exp03-medical-information-document-extraction/exp3.ipynb)、[exp33.ipynb](Group%2015%20exp03-medical-information-document-extraction/exp33.ipynb) | 前者使用四条模拟文档；后者读取 Open-Patients，失败可回退模拟数据；实体表、图谱及可视化 |
| 04 医疗文档 RAG | [使用说明](Group%2015%20exp04-easy-rag-system/README.md) | HTML/语料 JSON、问题集；FAISS 索引、Streamlit 问答、检索与生成评估 |

目录名 `classificationn`、`sentential` 是现有拼写，命令应保持一致。

## 开始前

1. 选择一个实验，先核对下方数据路径和 [配置基线](docs/CONFIGURATION.md)。原始训练 CSV、完整语料、问题集和模型权重没有随本仓库完整提供。
2. 每个实验创建独立 Python 虚拟环境，记录 Python、依赖、模型 revision 和硬件。仓库没有统一锁定环境，不承诺任意版本组合可运行。
3. 首次运行可能下载模型、NLTK/spaCy 资源或数据集。先核对来源许可、网络设置和磁盘空间。不要提交密钥、真实患者信息或不明来源的 pickle/模型文件。
4. 在实验自己的目录运行命令；路径含空格时使用普通英文双引号。

### 实验 01

```sh
cd "Group 15 exp01-data-preprocessing"
python -m venv .venv
# 激活环境后安装代码实际导入的依赖；版本组合尚未验证
python -m pip install pandas numpy gensim matplotlib scikit-learn networkx
python word2vec.py
```

先提供 `train_part_1.csv`。代码使用 `pd.read_csv` 默认表头，至少三列，按位置读取标签、标题、正文；无表头 CSV 会把第一行当作表头。请在工作副本统一格式，勿直接与实验 02 的无表头约定混用。词频阈值为 5，t-SNE 的 perplexity 为 10，小样本词表不足时可能失败。

### 实验 02

每种模型进入各自子目录运行 `python main.py`，不要从仓库根目录混用同名 `config`/`main` 模块。

| 实现 | 当前实际数据路径（相对其子目录） | 说明 |
| --- | --- | --- |
| BERT | `train_part_1.csv`、`dev.csv`、`test.csv` | 主入口硬编码，未采用配置中的 `dataset/` 路径；训练抽样比例在 `main.py` 中为 0.2 |
| Qwen | `Config.train_path/dev_path/test_path`，默认同上 | `requirements.txt` 是现有依赖入口；实际抽样比例为 0.01（1%） |
| TextCNN | `dataset/train_part_1.csv`、`dataset/dev.csv`、`dataset/test.csv` | `main.py` 内含训练及测试评估；独立 `test.py` 重建词表，需核对训练/评估映射一致性 |

三种实现均按无表头的标签、标题、正文三列读取，情感标签原始值为 1/2，映射为 0/1。三种训练入口都合并标题与正文，但后续分词和词表逻辑不同。模型比较应固定数据划分、样本和评价口径。BERT 需核对 torch、transformers、pandas、numpy、scikit-learn、matplotlib、tqdm；TextCNN 还使用 nltk。Qwen 的依赖文件虽锁定部分版本，但未覆盖 matplotlib、seaborn、tqdm 等直接导入，组合兼容性尚未验证，不应当作完整可复现环境。依赖和入口风险见 [需求与设计](docs/REQUIREMENTS_DESIGN.md)。

### 实验 03

在对应目录用 Jupyter 打开 notebook，先检查安装、下载、输入路径和数据库单元，再按顺序执行。`exp3.ipynb` 的 `sample_pdfs/` 实际保存文本模拟文件；`exp33.ipynb` 使用规则实体抽取与实体共现关系，不应写成统一的 BioBERT 流程。

不要直接对已有 Neo4j 数据库执行导入：`create_knowledge_graph` 默认 `clear_existing=True`，会清空目标库。使用隔离的实验库并先备份。共现关系不证明因果或治疗效果。

## 工程文档

本项目按课程规模参照 GJB 438C—2021 文档结构及 GJB 5000B—2021 过程证据思路裁剪，不构成标准符合性、成熟度等级或认证结论。

- [文档适用性与阅读顺序](docs/README.md)
- [需求与设计说明](docs/REQUIREMENTS_DESIGN.md)
- [验证计划与证据状态](docs/VERIFICATION.md)
- [配置、基线与变更](docs/CONFIGURATION.md)
- [追踪矩阵与待解决事项](docs/TRACEABILITY.md)

## 使用范围与许可

原有实验 04 文档声明“本项目仅用于课程/实验与学习用途”。本次文档整理不授予新许可证，也不改变原代码、数据集和模型的权利归属。复用或再分发前分别核对授权。
