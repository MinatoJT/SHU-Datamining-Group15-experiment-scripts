# 需求与设计说明

文档编号：DM15-SRS-SDD-001；修订：0.1；状态：待维护者评审；源码观察基线：`a65747b9ce505fe4e64d847b4d47e68fd686a3ca`。

文档版本 0.1；源码基线见 [编制说明](README.md)。本文件所有 REQ 均从当前实现归纳，等待课程任务书或维护者确认；没有新增性能承诺。

## 1. 功能需求与接口

| ID | 归纳需求及边界 | 输入 | 输出 |
| --- | --- | --- | --- |
| REQ-01 | 清洗英文评论并训练 CBOW、Skip-gram；展示相似词和 t-SNE | 至少三列、默认含表头 CSV；标题+正文 | 两个 `.model`、两个架构 PNG、一个 t-SNE PNG、控制台相似度 |
| REQ-02 | 对评论进行二分类并按实现提供验证/测试指标 | 无表头 CSV，标签 1/2；本地或下载的预训练模型 | 最优权重、日志；TextCNN 的 accuracy、weighted F1、AUC、混淆矩阵；Qwen 测试 loss/accuracy |
| REQ-03A | 对四条模拟医学文档做 NER 与关键词归类，生成文档—实体图 | notebook 内模拟标题/摘要 | gene/drug/symptom/effect 分类；CONTAINS 边、JSON/CSV/PNG/文本报告 |
| REQ-03B | 对临床文本规则抽取并构建实体共现图 | `ncbi/Open-Patients` 的 description/text；下载失败时五条回退样本 | disease/symptom/treatment/demographic；实体表、共现图、网络统计 |
| REQ-04A | 将文档块编码并检索 Top-K，交给生成模型回答 | `data/processed_data.json` 数组，至少提供 title/abstract 内容 | FAISS 索引、ID 元数据、Streamlit 答案及证据 |
| REQ-04B | 分别评估检索和端到端问答 | 外部问题 JSON；匹配的索引、元数据、嵌入模型 | 检索 JSONL；生成答案、参考答案、余弦相似度 JSONL；可选 EM/F1/ROUGE-L |

质量约束 Q-01：实验能够追溯到代码、输入、参数与环境版本。Q-02：声明与实际运行一致，区分历史结果、计划和新证据。Q-03：医疗示例仅教学；不要上传未授权患者数据。上述约束为本次建议管理要求，尚无签署验收记录。

## 2. 实现结构

- 实验 01：`load_and_preprocess_data` → `preprocess_text` → `train_word2vec_model` → 相似度/可视化/保存。`min_count=5`、维数 100、窗口 5、10 epochs、4 workers；未完整固定随机状态，不能承诺逐位复现。
- 实验 02：BERT/Qwen 拆分 `load_data.py`、`dataset.py`、`model.py`、`config.py`、`main.py`；TextCNN 在 `main.py` 集中定义预处理、词表、模型、训练及评估。三种流水线预处理和抽样不同，直接比较指标需注明差异。BERT 的配置路径和 num_workers 不完全控制主入口，实际数据路径硬编码、DataLoader 使用 8 workers。Qwen `sample_ratio=0.01`，注释中的 10% 不代表实际值。
- 实验 03：`exp3.ipynb` 使用 spaCy 预处理，尝试 `alvaroalon2/biobert_diseases_ner`，失败后用 `dslim/bert-base-NER`；类别还依赖关键词归类。`exp33.ipynb` 不沿用该 NER 模型，使用规则词典，先处理前 1000 条，再处理前至多 10000 条，以共现次数阈值 3 构图。不是随机抽样，也不是因果关系抽取。
- 实验 04：`preprocess.py` 或 `preprocess_benchmark_json.py` → `data_utils.py` → `models.py` → `milvus_utils.py` → `rag_core.py` → `app.py`。活动后端是 FAISS `IndexFlatL2`，虽然文件、函数名和 UI 仍写 Milvus Lite。`config.py` 的 IVF/nlist/nprobe 等旧 Milvus 设置不控制活动 FAISS 索引。

## 3. RAG 数据契约

HTML 预处理生成 `id/title/abstract/source_file/chunk_index`；正文块放入 `abstract`。建库前按 `MAX_ARTICLES_TO_INDEX=2000` 截断数组，向量文本为标题与摘要组合。检索 ID 是截断后数组的整数位置，不能直接视为原数据集文档 ID。

索引文件 `milvus_lite_data.faiss` 与 `milvus_lite_data.meta.pkl` 成对使用；后者含 `ids` 和 `id_to_doc_map`。向量未启用归一化，返回 L2 距离，不能写成余弦检索分数。端到端 JSONL 的 `similarity_cosine` 是生成答案与参考答案的嵌入余弦相似度，与检索距离不同，也不是医学正确率。

## 4. 已知限制与风险

- 原始 CSV、处理后 RAG 数据、外部问题集未完整入库；训练权重与模型 revision 未形成闭环基线。Qwen 依赖清单锁定了部分包版本，但缺少部分直接导入，兼容性未验证。
- 实验 01 默认表头与实验 02 无表头约定不同；空语料、小词表、坏标签需要先检查。
- Qwen `test.py` 中 `plot_reconstructed_history` 使用硬编码的历史数值作图，不能当作本次训练证据。TextCNN 独立测试脚本与训练词表构建不同，可能使评估不可比。
- 实验 03 的模型/数据下载失败会改变处理路径；必须记录是否回退。`exp33.ipynb` 节点 ID 使用 `hash(name) % 10000`，存在跨进程不稳定和碰撞风险。
- RAG 按现有索引的数量判断是否跳过建库，没有数据内容或模型指纹校验。同数量换语料、换模型、下调规模时可能复用旧索引。重新实验应使用隔离工作副本，重建索引并记录基线。
- `.pkl` 和模型反序列化有安全风险，仅使用可信来源文件；本次未加载现有二进制产物。
- Neo4j 导入可清库，notebook 不应未经检查直接全部执行。历史医学报告中的应用展望不构成临床验证。
