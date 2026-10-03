# Group 15 · 轻量医疗文档 RAG 实验

文档编号：DM15-SUM-004；修订：0.1；状态：待维护者评审；源码观察基线：`a65747b9ce505fe4e64d847b4d47e68fd686a3ca`。

使用本地文档、SentenceTransformer、FAISS 和 Qwen 构建 Streamlit 问答演示，并提供检索与端到端评价脚本。源码观察基线为 `a65747b9ce505fe4e64d847b4d47e68fd686a3ca`；本说明经静态核对，未重新运行。

> 活动检索后端为 FAISS `IndexFlatL2`。`milvus_utils.py`、配置名称和页面仍保留 Milvus Lite 字样；这些字样不代表当前活动后端。医疗回答只用于课程实验，不能用于诊疗。

## 1. 文件与流程

| 文件 | 用途 |
| --- | --- |
| `preprocess.py` | 当前目录 `data/` 内 `.html` 正文提取、字符分块，写入 `data/processed_data.json` |
| `preprocess_benchmark_json.py` | 外部语料 JSON 分块；需先改脚本末尾 Windows `IN_JSON` 路径 |
| `config.py` | UI 的数据/模型/生成设置；含旧 Milvus 参数 |
| `models.py`、`milvus_utils.py`、`rag_core.py` | 模型加载、FAISS 索引与检索、答案生成 |
| `app.py` | Streamlit 页面与启动建库 |
| `eval_questions.py` | 检索评价，默认输出 `Evaluation/run_results.jsonl` |
| `eval_qa_e2e.py` | 检索+生成+答案相似度，默认输出 `Evaluation/qa_e2e_results_Max_all.jsonl` |
| `score_qa_results.py` | 从结果文件计算 EM、词级 F1，装有 rouge-score 时计算 ROUGE-L |
| `results_to_pic.py` | 读取 Max_all JSONL 绘制分布，输出 `Evaluation/qa_e2e_score_hist.png` |

## 2. 安装准备

先进入本实验目录。建议从 Python 3.10/3.11 的独立环境开始；仓库未锁定可复现版本组合，跨平台和 GPU 可用性需自行验证。

Windows PowerShell：

```powershell
cd "Group 15 exp04-easy-rag-system"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install faiss-cpu numpy beautifulsoup4 lxml matplotlib
# 可选评分依赖
python -m pip install rouge-score
```

macOS/Linux 的环境激活命令为 `source .venv/bin/activate`，其余安装命令相同。若脚本激活受系统策略阻止，使用虚拟环境中的 Python 可执行文件运行命令；不要为了复现而全局降低安全策略。

现有 `requirements.txt` 包含 streamlit、pymilvus、sentence-transformers、transformers、torch、accelerate，但缺少活动后端 `faiss-cpu` 和预处理/绘图直接依赖。以上为文档补充安装项，未改依赖文件；成功运行后应记录实际版本。

`app.py` 会直接设置 `HF_ENDPOINT=https://hf-mirror.com` 与 `HF_HOME=./hf_cache`；评测脚本使用 setdefault。启动前检查所用镜像和模型来源，首次运行需要可访问的模型或完整本地缓存，不应假设外部变量能覆盖 UI 中的赋值。

## 3. 数据准备与启动

运行前先使用隔离工作副本保存历史结果。仓库没有完整提供 `data/processed_data.json` 或外部题集。对新增实验不要盲用已提交的索引和 pickle：仅使用可信文件，并核对数据及模型一致性。

HTML 路线：将有权使用的 `.html` 文件放到本目录 `data/`，执行：

```sh
python preprocess.py
python -m streamlit run app.py
```

检查预处理日志和 JSON：应有非空记录，正文块在 `abstract`，默认字符块大小 512、重叠 50。0 chunks 不能视为成功建库。JSON 语料路线需先在工作副本中配置 `preprocess_benchmark_json.py` 的 `IN_JSON`，再执行该脚本。

不要使用 `python app.py` 启动页面。页面会加载模型、数据和索引，查询时展示证据与生成答案。当前 `data_utils.py` 无法读到数据时会跳过建库；如果历史索引已恢复文档映射，页面仍可能提供查询。因此页面可用不代表新数据已被索引，必须检查日志和证据来源。

## 4. 索引和配置

UI 默认嵌入模型 `all-MiniLM-L6-v2`（384 维）、生成模型 `Qwen/Qwen2.5-0.5B`、Top-K=3、最多索引前 2000 条记录、最多生成 512 tokens。

`MILVUS_LITE_DATA_PATH=./milvus_lite_data.db` 经代码替换后得到 `.faiss` 和 `.meta.pkl` 两文件。旧 `.db` 不是活动 FAISS 文件。`INDEX_TYPE`、`INDEX_PARAMS`、`SEARCH_PARAMS` 等旧 Milvus 配置不改变实际 `IndexFlatL2` 检索。

建库跳过逻辑主要比较数量，没有检查语料内容或模型版本。改变数据、嵌入模型或索引上限后，先在隔离工作副本将旧 `.faiss/.meta.pkl` 成对移到备份位置，重新启动并核对日志；不要覆盖唯一历史证据。不要加载不明来源的 pickle 文件。

## 5. 评价流程

两个评价脚本的 `QUESTIONS_JSON` 均硬编码为开发者 Windows 路径，没有 CLI 参数。先在工作副本中设置有效题集路径，核对 FAISS、元数据、模型、Top-K 与数据版本。检索评价的 gold ID 要与代码生成的整数位置 ID 对齐，不能直接混用原始文档 ID。

```sh
python eval_questions.py
python eval_qa_e2e.py
# 先将 RESULT_JSONL 指向本次端到端输出，再运行
python score_qa_results.py
# 先核对 JSONL_PATH 和 OUT_PATH，避免覆盖历史图
python results_to_pic.py
```

`eval_questions.py` 是检索评价，不生成答案。`eval_qa_e2e.py` 当前限制前 100 道问题，使用独立常量而非自动读取 UI 配置，生成上限 256 tokens；它追加写入并跳过已存在的 question。每次换参数应改用新的结果文件，避免混合历史结果。

端到端输出包含 `question/reference_answer/generated_answer/similarity_cosine/retrieved_doc_ids/retrieved_preview`。余弦相似度衡量答案嵌入相近程度，不代表事实或医学正确率。评分默认读 `qa_e2e_results.jsonl`，而端到端默认写 `qa_e2e_results_Max_all.jsonl`，运行前必须对齐。分布图限制横轴 0–1，可能不显示负余弦分数；完整分析应检查原始 JSONL。

## 6. 证据与限制

已提交 JSONL、索引、数据库和图片都是历史产物，不表示本次环境已复现。完整工程说明见 [验证计划](../docs/VERIFICATION.md)、[配置基线](../docs/CONFIGURATION.md) 和 [问题追踪](../docs/TRACEABILITY.md)。

## 使用范围

本项目仅用于课程/实验与学习用途。本次文档整理不新增许可证；数据与模型需分别遵守来源许可。
