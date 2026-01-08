import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# 你的结果文件
JSONL_PATH = Path("./Evaluation/qa_e2e_results_Max_all.jsonl")

# 输出图片路径
OUT_PATH = Path("./Evaluation/qa_e2e_score_hist.png")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

scores = []
with JSONL_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        s = obj.get("similarity_cosine", obj.get("cosine", None))
        if s is None:
            continue
        try:
            s = float(s)
        except Exception:
            continue
        if not math.isnan(s):
            scores.append(s)

scores = np.array(scores, dtype=float)
print(f"N = {len(scores)}")
if len(scores) == 0:
    raise SystemExit("没有读到任何分数，请检查 JSONL 路径或字段名。")

plt.figure()
plt.hist(scores, bins=30, range=(0, 1))
plt.xlabel("similarity_cosine")
plt.ylabel("count")
plt.title("Distribution of similarity_cosine (QA E2E)")

# 画几条阈值线（可按需删改）
# 画阈值线并改颜色
thresholds = [
    (0.5, "orange"),
    (0.7, "black"),
    (0.85, "red"),
]

for t, c in thresholds:
    plt.axvline(t, linewidth=2, color=c)
    plt.text(
        t, plt.ylim()[1] * 0.95, f"{t}",
        rotation=90, va="top", ha="right", color=c
    )


plt.tight_layout()

# 保存（dpi 越大越清晰）
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved -> {OUT_PATH.resolve()}")
