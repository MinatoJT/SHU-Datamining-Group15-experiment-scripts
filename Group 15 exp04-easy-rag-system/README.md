Group 14 - Easy Medical RAG System (FAISS)

本项目实现了一个轻量级医疗文档 RAG（Retrieval-Augmented Generation）问答系统。离线阶段对医疗文档进行解析、清洗与分块，使用嵌入模型生成向量并建立本地 FAISS 向量索引；在线阶段对用户问题进行向量检索（Top-K），将检索到的证据片段与问题一起输入大语言模型生成答案。系统提供 Streamlit Web 界面用于交互展示与调试，并提供端到端评测脚本输出 qa_e2e_results.jsonl 便于分析。

注意：本仓库使用 FAISS 作为本地向量检索后端，不依赖 Milvus / Docker。

⸻

运行环境要求
	•	Python 3.10+（推荐 3.10/3.11）
	•	Windows / macOS / Linux 均可（本项目在 Windows PowerShell 环境下开发）
	•	可选：GPU（无 GPU 也可运行，但生成模型会慢）

⸻

项目结构（示例）

仓库根目录下建议保持如下结构（你也可以根据实际文件名调整）：

Group 14/
app.py
config.py
models.py
data_utils.py
rag_core.py
preprocess.py
（可选）eval_questions.py / eval_qa_e2e.py
data/
processed_data.json
Evaluation/
qa_e2e_results.jsonl

⸻

安装与运行步骤（Windows PowerShell）

1）进入项目根目录（示例路径按你电脑实际情况修改）
cd E:\datamining\exp04-easy-rag-system

2）创建并激活虚拟环境
python -m venv .venv
..venv\Scripts\Activate.ps1

3）升级 pip
python -m pip install –upgrade pip

4）安装依赖（优先使用 requirements.txt）
pip install -r “Group 14/requirements.txt”

如果仓库里没有 requirements.txt，至少安装这些核心依赖
pip install streamlit sentence-transformers transformers torch faiss-cpu numpy matplotlib

提示：如果你想安装 GPU 版 PyTorch，请到 PyTorch 官网按对应 CUDA 版本安装。

⸻

可选：HuggingFace 下载加速（镜像与缓存）

如果你需要使用 hf-mirror（或其他镜像）并将模型缓存到本地，可以在 PowerShell 设置环境变量：

$env:HF_ENDPOINT=“https://hf-mirror.com”
$env:HF_HOME=”./hf_cache”

也可以直接使用项目 app.py 中设置的 os.environ（两者二选一即可）。

⸻

数据预处理（生成 processed_data.json）

1）准备原始数据
将原始医疗文档（例如 HTML 文件）放入：
Group 14\data\

2）运行预处理脚本
python “Group 14/preprocess.py”

成功后会生成：
Group 14\data\processed_data.json

如果输出显示 0 chunks，说明预处理没有提取到正文文本或数据格式不匹配，需要检查输入文件路径、HTML 解析规则或字段抽取逻辑。

⸻

启动 Web 问答系统（Streamlit）

在仓库根目录运行：
streamlit run “Group 14/app.py”

浏览器会自动打开（默认地址通常是 http://localhost:8501 ）。在页面输入问题并点击按钮，即可看到：
	•	Top-K 检索到的证据片段（可折叠展示）
	•	基于证据生成的答案

注意：不要用 python app.py 直接运行，否则会出现 missing ScriptRunContext 等提示；必须用 streamlit run 启动。

⸻

关键参数说明（config.py）

常用调参项在 Group 14\config.py 中：
	•	MAX_ARTICLES_TO_INDEX：最大索引文本块数量（控制建库规模，用于做 1000 vs 2000 对比实验）
	•	TOP_K：检索返回证据数量
	•	EMBEDDING_MODEL_NAME：嵌入模型（默认 all-MiniLM-L6-v2，对应 384 维）
	•	GENERATION_MODEL_NAME：生成模型（默认 Qwen/Qwen2.5-0.5B）
	•	MAX_NEW_TOKENS_GEN、TEMPERATURE、TOP_P、REPETITION_PENALTY：生成控制参数

建议做对比实验时将 TEMPERATURE 设置为 0（或很小），减少随机性导致的评测波动。

⸻

端到端评测（可选）

如果仓库中包含评测脚本（例如 eval_questions.py 或 eval_qa_e2e.py），可运行批量评测并生成结果文件。

运行示例：
python “Group 14/eval_questions.py”

输出通常会保存到：
Group 14\Evaluation\qa_e2e_results.jsonl

结果文件中每条记录一般包含：
question、reference_answer、generated_answer、similarity_cosine、retrieved_doc_ids 等字段，可用于统计与可视化。

⸻

绘制评分分布图（可选）

确保你已经生成了 Group 14\Evaluation\qa_e2e_results.jsonl，然后运行下面的脚本生成直方图（会保存为 qa_e2e_score_hist.png）。如果你不想新建文件，可以把这段脚本临时保存为 plot_hist.py 再运行 python plot_hist.py：

import json
import numpy as np
import matplotlib.pyplot as plt

jsonl_path = r”./Group 14/Evaluation/qa_e2e_results.jsonl”
out_path   = r”./qa_e2e_score_hist.png”

scores = []
with open(jsonl_path, “r”, encoding=“utf-8”) as f:
for line in f:
obj = json.loads(line)
v = obj.get(“similarity_cosine”, None)
if v is not None:
scores.append(float(v))

scores = np.array(scores, dtype=float)

plt.figure(figsize=(10, 7))
bins = np.linspace(0, 1, 31)
plt.hist(scores, bins=bins)

for t, c in [(0.5, “orange”), (0.7, “black”), (0.85, “red”)]:
plt.axvline(t, linewidth=3, color=c)
plt.text(t, plt.ylim()[1] * 0.95, f”{t}”, rotation=90, va=“top”, ha=“right”, color=c)

plt.title(“Distribution of similarity_cosine (QA E2E)”)
plt.xlabel(“similarity_cosine”)
plt.ylabel(“count”)
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig(out_path, dpi=200)
print(“Saved:”, out_path)

⸻

常见问题

1）Streamlit 报 missing ScriptRunContext
原因通常是使用 python app.py 直接运行。请改用：
streamlit run “Group 14/app.py”

2）HuggingFace 429 Too Many Requests
表示镜像/网络对你的 IP 限流。解决办法：
	•	换网络（例如手机热点）
	•	使用自己的 HuggingFace 账号 Token（HF_TOKEN）
	•	等一段时间再试或确保模型缓存已下载完成

3）Git push 报 Recv failure: Connection was reset
这是网络连接问题（常见于校园网/公司网限制 GitHub）。建议：
	•	换网络（手机热点）
	•	检查并清理 git 代理配置（git config –global –get https.proxy）
	•	必要时使用 GitHub Desktop 或 VPN

⸻

License

本项目仅用于课程/实验与学习用途。
