import os
import json
import time
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ====== 可选：和你的 app.py 一致（用 hf-mirror + 本地缓存）======
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "./hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ====== 路径配置（按你的实际位置改）======
QUESTIONS_JSON = r"E:\GraphRAG-Benchmark-main\Datasets\Questions\medical_questions.json"

# 这是你 FAISS 方案生成的索引文件（默认在项目根目录）
FAISS_INDEX_PATH = r"./milvus_lite_data.faiss"
FAISS_META_PATH  = r"./milvus_lite_data.meta.pkl"

# OUT_JSONL = r"./Evaluation/qa_e2e_results.jsonl"  #MAX=100 TOP-K=3
OUT_JSONL = r"./Evaluation/qa_e2e_results_Max_all.jsonl"  #MAX=50 TOP-K=3

# ====== 模型配置（保持和 config.py 一致）======
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME   = "Qwen/Qwen2.5-0.5B"

TOP_K = 3

# ====== 评测规模：建议先小规模跑通，再扩大 ======
LIMIT_QUESTIONS = 100      # 先跑前 50 条；想全跑改成 2062
START_OFFSET = 0          # 从第几条开始（断点续跑可用）

# ====== 生成参数（与你 config.py 类似）======
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# ====== Prompt/上下文控制（避免 prompt 太长）======
MAX_CONTEXT_CHARS_PER_DOC = 1200   # 每个检索文档最多取多少字符
MAX_PROMPT_TOKENS = 2048           # prompt 截断到多少 token（防显存/内存爆）


def load_questions(path: str) -> List[Dict[str, Any]]:
    data = json.load(open(path, "r", encoding="utf-8"))
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
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


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: (dim,)
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


def build_prompt(query: str, context_docs: List[Dict[str, Any]]) -> str:
    parts = []
    for d in context_docs:
        c = d.get("content", "")
        if not isinstance(c, str):
            c = str(c)
        c = c[:MAX_CONTEXT_CHARS_PER_DOC]
        parts.append(c)

    context = "\n\n---\n\n".join(parts)

    prompt = f"""Based ONLY on the following context documents, answer the user's question.
If the answer is not found in the context, state that clearly. Do not make up information.

Context Documents:
{context}

User Question: {query}

Answer:
"""
    return prompt


def main():
    # ---- 1) 加载 FAISS ----
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("faiss-cpu 未安装，请先 pip install faiss-cpu") from e

    if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_META_PATH)):
        raise FileNotFoundError(
            "找不到 FAISS 索引文件。请先运行 streamlit 让它完成索引。\n"
            f"需要：{FAISS_INDEX_PATH} 和 {FAISS_META_PATH}"
        )

    index = faiss.read_index(FAISS_INDEX_PATH)
    meta = pickle.load(open(FAISS_META_PATH, "rb"))
    ids: List[int] = meta.get("ids", [])
    id_to_doc_map: Dict[int, Dict[str, Any]] = meta.get("id_to_doc_map", {})

    # ---- 2) 加载 embedding 模型（用于检索 + 打分）----
    embed = SentenceTransformer(EMBED_MODEL_NAME)

    # ---- 3) 加载生成模型 ----
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_NAME,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 4) 读取 questions ----
    qs = load_questions(QUESTIONS_JSON)
    if not qs:
        raise RuntimeError("questions 读出来为空，检查 QUESTIONS_JSON 路径或文件结构。")

    sample = qs[0]
    q_key = find_first_key(sample, ["question", "query", "prompt", "q", "input"])
    a_key = find_first_key(sample, ["answer", "answers", "ground_truth", "label"])

    if not q_key:
        raise RuntimeError(f"找不到问题字段。样本 keys={list(sample.keys())}")
    if not a_key:
        raise RuntimeError(f"找不到答案字段。样本 keys={list(sample.keys())}")

    print("Detected keys:", {"question_key": q_key, "answer_key": a_key})

    # ---- 5) 断点续跑：已存在结果就跳过 ----
    ensure_out_dir(OUT_JSONL)
    done = 0
    done_questions = set()
    if os.path.exists(OUT_JSONL):
        with open(OUT_JSONL, "r", encoding="utf-8") as r:
            for line in r:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    qq = obj.get("question")
                    if isinstance(qq, str):
                        done_questions.add(qq)
                        done += 1
                except Exception:
                    pass
        if done:
            print(f"Resume: found {done} existing results, will skip duplicates.")

    # ---- 6) 主循环：检索→生成→打分 ----
    total = 0
    sims = []
    t0 = time.time()

    with open(OUT_JSONL, "a", encoding="utf-8") as w:
        for item in qs[START_OFFSET:]:
            q = item.get(q_key, "")
            ref = item.get(a_key, "")

            if not (isinstance(q, str) and q.strip()):
                continue
            if not (isinstance(ref, str) and ref.strip()):
                continue

            if q in done_questions:
                continue

            total += 1
            if LIMIT_QUESTIONS and total > LIMIT_QUESTIONS:
                break

            # --- 检索 TopK ---
            q_vec = embed.encode([q])[0].astype(np.float32).reshape(1, -1)
            D, I = index.search(q_vec, TOP_K)

            retrieved_ids = []
            for idx in I[0].tolist():
                if 0 <= idx < len(ids):
                    retrieved_ids.append(ids[idx])

            context_docs = []
            for rid in retrieved_ids:
                doc = id_to_doc_map.get(rid)
                if doc:
                    context_docs.append(doc)

            # --- 生成 ---
            prompt = build_prompt(q, context_docs)

            # inputs = tokenizer(
            #     prompt,
            #     returnif = False
            # )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_PROMPT_TOKENS
            ).to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    repetition_penalty=REPETITION_PENALTY,
                    pad_token_id=tokenizer.eos_token_id
                )

            gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            # --- 打分：用 embedding cosine（0~1 越大越像）---
            e = embed.encode([gen, ref])
            sim = cosine_sim(np.array(e[0]), np.array(e[1]))
            sims.append(sim)

            # 预览信息（方便你肉眼查）
            preview = []
            for rid in retrieved_ids:
                doc = id_to_doc_map.get(rid)
                if doc:
                    preview.append({"id": rid, "title": (doc.get("title", "") or "")[:120]})

            rec = {
                "question": q,
                "reference_answer": ref,
                "generated_answer": gen,
                "similarity_cosine": sim,
                "retrieved_doc_ids": retrieved_ids,
                "retrieved_preview": preview,
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            w.flush()

            if total % 10 == 0:
                avg = sum(sims) / len(sims)
                print(f"[{total}] avg_cos={avg:.3f} last_cos={sim:.3f}")

    dt = time.time() - t0
    if sims:
        print(f"Done. N={len(sims)}  avg_cos={sum(sims)/len(sims):.3f}")
    else:
        print("Done. No results produced (check input).")
    print(f"Saved -> {OUT_JSONL}")


if __name__ == "__main__":
    main()
