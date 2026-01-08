import json, math, re
from collections import Counter

RESULT_JSONL = r"./Evaluation/qa_e2e_results.jsonl"  # 你实际文件名改这里

def normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def exact_match(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))

def f1(pred: str, gold: str) -> float:
    p = normalize(pred).split()
    g = normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec = same / len(p)
    rec = same / len(g)
    return 2 * prec * rec / (prec + rec)

# ROUGE-L（可选，装了 rouge-score 才算）
try:
    from rouge_score import rouge_scorer
    _rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
except Exception:
    _rouge = None

def rougeL(pred: str, gold: str) -> float:
    if _rouge is None:
        return float("nan")
    return float(_rouge.score(gold, pred)["rougeL"].fmeasure)

def nanmean(xs):
    xs2 = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return sum(xs2)/len(xs2) if xs2 else float("nan")

def main():
    cos_list, em_list, f1_list, rL_list = [], [], [], []
    worst = []  # 存最差样本方便看

    with open(RESULT_JSONL, "r", encoding="utf-8") as r:
        for line in r:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            ref = obj.get("reference_answer", "")
            gen = obj.get("generated_answer", "")

            # 兼容你之前字段名 similarity_cosine
            cos = obj.get("cosine", obj.get("similarity_cosine", float("nan")))
            cos_list.append(float(cos) if cos is not None else float("nan"))

            emv = exact_match(gen, ref)
            f1v = f1(gen, ref)
            rLv = rougeL(gen, ref)

            em_list.append(emv)
            f1_list.append(f1v)
            rL_list.append(rLv)

            worst.append((float(cos) if cos is not None else -1.0, obj.get("question",""), ref, gen))

    worst.sort(key=lambda x: x[0])  # 按 cosine 从低到高

    print(f"N = {len(cos_list)}")
    print(f"avg_cosine   = {nanmean(cos_list):.3f}")
    print(f"EM%          = {100*nanmean(em_list):.2f}%")
    print(f"avg_F1       = {nanmean(f1_list):.3f}")
    if _rouge is None:
        print("avg_ROUGE-L  = (skip)  # pip install rouge-score 之后可用")
    else:
        print(f"avg_ROUGE-L  = {nanmean(rL_list):.3f}")

    print("\nWorst 10 by cosine:")
    for i, (cos, q, ref, gen) in enumerate(worst[:10], 1):
        print(f"\n[{i}] cos={cos:.3f}")
        print("Q  :", q)
        print("REF:", ref)
        print("GEN:", gen)

if __name__ == "__main__":
    main()
