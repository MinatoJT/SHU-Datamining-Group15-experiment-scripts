import json
import os
import pickle
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# ====== 你的路径（按需改）======
QUESTIONS_JSON = r"E:\GraphRAG-Benchmark-main\Datasets\Questions\medical_questions.json"

# 下面两个文件是你 Streamlit 建索引后生成的（FAISS 方案）
FAISS_INDEX_PATH = r"./milvus_lite_data.faiss"
FAISS_META_PATH  = r"./milvus_lite_data.meta.pkl"

TOP_K = 3
OUT_JSONL = r"./Evaluation/run_results.jsonl"

# embedding 模型要和你 app.py 一致
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def load_questions(path: str) -> List[Dict[str, Any]]:
    data = json.load(open(path, "r", encoding="utf-8"))
    # 兼容：顶层是 list 或 dict 包一层
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        # 常见：{"questions":[...]} 或 {"data":[...]}
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []


def find_first_key(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d:
            return k
    return None


def ensure_out_dir(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def main():
    # ---- 加载 FAISS ----
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("faiss-cpu 未安装或不可用，请先 pip install faiss-cpu") from e

    if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_META_PATH)):
        raise FileNotFoundError(
            "找不到 FAISS 索引文件。请先运行 streamlit，让它完成 'FAISS indexing done'。\n"
            f"期望存在：{FAISS_INDEX_PATH} 和 {FAISS_META_PATH}"
        )

    index = faiss.read_index(FAISS_INDEX_PATH)
    meta = pickle.load(open(FAISS_META_PATH, "rb"))

    # ids: FAISS 向量序号 -> doc_id（你系统里是 0..N-1）
    ids: List[int] = meta.get("ids", [])
    # id_to_doc_map: doc_id -> 文档内容（用于展示/可选用于评测）
    id_to_doc_map: Dict[int, Dict[str, Any]] = meta.get("id_to_doc_map", {})

    # ---- 加载 embedding 模型 ----
    embed = SentenceTransformer(EMBED_MODEL_NAME)

    # ---- 读取 questions ----
    qs = load_questions(QUESTIONS_JSON)
    if not qs:
        raise RuntimeError("questions 文件读出来为空。请检查 QUESTIONS_JSON 路径或文件结构。")

    # 自动猜字段
    sample = qs[0]
    q_key = find_first_key(sample, ["question", "query", "prompt", "q", "input"])
    gold_key = find_first_key(sample, ["gold_doc_id", "gold_doc_ids", "doc_id", "doc_ids", "supporting_doc_ids", "evidence_ids"])
    ans_key = find_first_key(sample, ["answer", "answers", "ground_truth", "groundtruth", "label"])

    print("Detected keys:",
          {"question_key": q_key, "gold_doc_key": gold_key, "answer_key": ans_key})

    if q_key is None:
        raise RuntimeError(f"无法在 questions 里找到问题字段。样本 keys={list(sample.keys())}")

    ensure_out_dir(OUT_JSONL)

    total = 0
    hit_at_k = 0
    has_gold = gold_key is not None

    with open(OUT_JSONL, "w", encoding="utf-8") as w:
        for item in qs:
            q = item.get(q_key, "")
            if not isinstance(q, str) or not q.strip():
                continue

            total += 1
            q_vec = embed.encode([q])[0].astype(np.float32).reshape(1, -1)

            D, I = index.search(q_vec, TOP_K)
            # I 是向量序号，需要映射成 doc_id
            retrieved_doc_ids = []
            distances = []
            for idx, dist in zip(I[0].tolist(), D[0].tolist()):
                if idx < 0 or idx >= len(ids):
                    continue
                retrieved_doc_ids.append(ids[idx])
                distances.append(float(dist))

            # ---- 尝试做“命中”评测（仅当 questions 有 gold_doc_id 类字段时）----
            hit = None
            gold = None
            if has_gold:
                gold = item.get(gold_key)
                gold_set = set()

                if isinstance(gold, int):
                    gold_set.add(gold)
                elif isinstance(gold, list):
                    for g in gold:
                        if isinstance(g, int):
                            gold_set.add(g)

                if gold_set:
                    hit = any(r in gold_set for r in retrieved_doc_ids)
                    if hit:
                        hit_at_k += 1

            out = {
                "question": q,
                "retrieved_doc_ids": retrieved_doc_ids,
                "distances": distances,
                "hit_at_k": hit,
                "gold": gold,
            }

            # 可选：把检索到的 title 也写进去，方便你肉眼看检索对不对
            preview = []
            for rid in retrieved_doc_ids:
                doc = id_to_doc_map.get(rid)
                if doc:
                    preview.append({"id": rid, "title": doc.get("title", "")[:120]})
            out["retrieved_preview"] = preview

            # 可选：如果 questions 自带 answer，也记录一下
            if ans_key and ans_key in item:
                out["reference_answer"] = item.get(ans_key)

            w.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Saved: {OUT_JSONL}")
    print(f"Total questions processed: {total}")
    if has_gold:
        if total > 0:
            print(f"Hit@{TOP_K}: {hit_at_k}/{total} = {hit_at_k/total:.3f}")
        else:
            print("No valid questions found.")
    else:
        print("No gold doc id field detected; skipped Hit@K metric.")


if __name__ == "__main__":
    main()
