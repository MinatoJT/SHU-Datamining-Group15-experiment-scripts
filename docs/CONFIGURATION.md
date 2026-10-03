# 配置、基线与变更管理

文档编号：DM15-SVD-SDP-001；修订：0.1；状态：待维护者评审；源码观察基线：`a65747b9ce505fe4e64d847b4d47e68fd686a3ca`。

版本 0.1，待评审。本文描述当前源码事实与建议记录方式，不宣称已经形成正式受控基线。

## 1. 配置项与缺口

| 配置项 | 当前来源/位置 | 管理要求与缺口 |
| --- | --- | --- |
| 源码/文档 | Git 提交 `a65747b9ce505fe4e64d847b4d47e68fd686a3ca` 为观察基线 | 发布本次文档时另外记录文档提交 SHA；不覆盖历史 |
| Python 与依赖 | Qwen 的 `requirements.txt` 锁定部分包版本，RAG 清单未锁版本；其他实验按导入与 notebook 安装单元识别 | 独立环境成功运行后保存 Python 版本、完整依赖清单、OS、CPU/GPU/CUDA；Qwen 清单还缺少 matplotlib、seaborn、tqdm 等直接导入，目前无验证完成的组合 |
| 情感数据 | 本地 CSV | 保存来源许可、列/表头约定、行数、标签分布、划分方法与哈希；本仓库未完整提供 |
| 医学数据 | 模拟样本、Open-Patients、外部 HTML/GraphRAG-Benchmark 语料/问题 | 分开标识，记录数据 revision；下载失败的模拟回退不能伪装成真实数据结果 |
| 模型 | BERT、Qwen、SentenceTransformer、NER | 记录完整 ID、revision、缓存位置及实际回退；模型名称本身不足以锁定版本 |
| RAG 索引 | `.faiss` 与 `.meta.pkl`；基线还留有旧 `.db` | 成对保存并核对来源/哈希/维数/规模；旧 `.db` 不证明使用 Milvus 后端 |
| 结果 | notebook 输出、CSV/JSON/JSONL/PNG、训练权重 | 每次实验使用独立工作副本及输出目录；保留历史产物，避免覆盖或混跑 |

## 2. 关键参数事实

| 实验 | 参数与实际值 | 注意事项 |
| --- | --- | --- |
| 01 | 100 维，window 5，min_count 5，epochs 10；t-SNE perplexity 10 | 固定 t-SNE seed 42 不等于完整训练可复现 |
| BERT | `bert-base-chinese`，max length 128，batch 150，lr 5e-5，2 epochs；main 抽样 20% | 配置中的路径/worker 字段与实际入口有差异 |
| Qwen | `Qwen/Qwen2.5-0.5B`，length 128，batch 10，lr 2e-5，5 epochs，sample_ratio 0.01 | 注释有过时的 BERT/10% 说法；以活动语句为准 |
| TextCNN | embedding 128，filters 128，窗口 [3,5]，batch 64，lr 1e-3，2 epochs | 词表构建默认取前 5% 文本；应保存映射后再比较评估 |
| 03B | 前至多 10000 条、共现阈值 3 | 数据回退时数量会显著不同；节点 hash 不是稳定标识 |
| RAG UI | MiniLM 384 维，Qwen 0.5B，Top-K 3，上限 2000，生成上限 512 | FAISS IndexFlatL2；模型或内容变化不能靠数量判断索引新鲜度 |
| RAG E2E | 独立脚本常量，Top-K 3，LIMIT_QUESTIONS 100，生成上限 256 | 不会自动继承 config.py 的 UI 参数；默认追加结果并按已有 question 跳过 |

## 3. 路径和网络

BERT/Qwen 的模型加载设置、RAG `app.py` 及评测脚本包含 Hugging Face 镜像设置。RAG UI 直接赋值 `HF_ENDPOINT`/`HF_HOME`，外部环境变量不一定能覆盖；运行前检查实际活动配置及来源。不要把凭据写入仓库。

`preprocess_benchmark_json.py` 的 `IN_JSON`，两个 RAG 评测脚本的 `QUESTIONS_JSON` 都是开发者 Windows 绝对路径；它们没有命令行参数解析。需在实验工作副本中修改为本机有效路径并记录差异。评分脚本默认读 `qa_e2e_results.jsonl`，端到端脚本默认写 `qa_e2e_results_Max_all.jsonl`，必须主动对齐。

## 4. 建议变更流程

1. 保存当前源码 SHA、输入/模型指纹和结果；提出变更原因，关联 REQ/T/问题 ID。
2. 在独立分支或工作副本修改；数据/模型/索引变化同时评估接口、规模、计算资源和结果可比性。
3. 按验证计划执行受影响项目并保留原始证据；未执行项继续标注，不用“文档完成”代替验证。
4. 维护者复核后记录提交和结论；失败则保留失败记录并回退到已保存的代码、数据和索引组合。

本次仅变更 Markdown 文档，没有删除历史 `.idea`、缓存、数据库或结果，也没有修改依赖。后续清理和代码修复应单独评审。
