# import json
#
# fp = r"E:\GraphRAG-Benchmark-main\Datasets\Corpus\medical.json"
#
# with open(fp, "r", encoding="utf-8") as f:
#     data = json.load(f)
#
# print("TOP TYPE:", type(data))
#
# def show_doc(d):
#     if isinstance(d, dict):
#         print("DOC TYPE: dict")
#         print("DOC KEYS:", list(d.keys())[:50])
#         # 打印几个典型字段的类型
#         for k in ["title","name","abstract","text","content","document","passage","body","context","article"]:
#             if k in d:
#                 v = d[k]
#                 print(f"  field {k!r} type =", type(v), "preview =", (str(v)[:120] if v is not None else None))
#     else:
#         print("DOC TYPE:", type(d))
#         print("DOC PREVIEW:", str(d)[:200])
#
# if isinstance(data, list):
#     print("LIST LEN:", len(data))
#     if data:
#         show_doc(data[0])
#
# elif isinstance(data, dict):
#     print("DICT KEYS (top):", list(data.keys())[:30])
#     # 找到一个最可能装 docs 的容器
#     list_fields = [(k, v) for k, v in data.items() if isinstance(v, list)]
#     if list_fields:
#         k, v = list_fields[0]
#         print(f"FOUND LIST FIELD: {k!r}, len={len(v)}")
#         if v:
#             show_doc(v[0])
#     else:
#         # 常见：dict[id -> doc]
#         dict_values = list(data.values())
#         print("DICT VALUES COUNT:", len(dict_values))
#         if dict_values:
#             show_doc(dict_values[0])
#


import os, json

TEXT_KEYS = ["abstract", "content", "text", "document", "passage", "body", "context", "article"]
TITLE_KEYS = ["title", "name", "doc_title"]

def to_text(v):
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        parts = []
        for x in v:
            s = to_text(x)
            if s:
                parts.append(s)
        return "\n".join(parts).strip()
    if isinstance(v, dict):
        # dict 兜底：拼接里面所有字符串/列表字符串
        parts = []
        for vv in v.values():
            s = to_text(vv)
            if s:
                parts.append(s)
        return "\n".join(parts).strip()
    return ""

def pick_title(d, default=""):
    for k in TITLE_KEYS:
        if k in d:
            s = to_text(d.get(k))
            if s:
                return s
    return default

def pick_text(d):
    # 优先找典型字段
    for k in TEXT_KEYS:
        if k in d:
            s = to_text(d.get(k))
            if s:
                return s
    # 兜底：把所有字段文本拼起来
    parts = []
    for v in d.values():
        s = to_text(v)
        if s:
            parts.append(s)
    return "\n".join(parts).strip()

def chunk_text(text, chunk_size=512, overlap=50):
    text = (text or "").strip()
    if not text:
        return []
    step = max(1, chunk_size - overlap)
    out = []
    i, n = 0, len(text)
    while i < n:
        j = min(n, i + chunk_size)
        c = text[i:j].strip()
        if c:
            out.append(c)
        if j == n:
            break
        i += step
    return out

def iter_doc_dicts(obj):
    """
    递归遍历：
    - 遇到 dict：如果含有 TEXT_KEYS 且内容够长，视为一个“文档候选”
    - 同时继续向下递归，把深层的也找出来
    """
    if isinstance(obj, dict):
        # 判断它像不像一条文档
        looks_like_doc = any(k in obj for k in TEXT_KEYS) or any(k in obj for k in TITLE_KEYS)
        if looks_like_doc:
            txt = pick_text(obj)
            if len(txt) >= 30:  # 太短的通常是噪声
                yield obj

        for v in obj.values():
            yield from iter_doc_dicts(v)

    elif isinstance(obj, list):
        for x in obj:
            yield from iter_doc_dicts(x)

def main(in_json, out_json, chunk_size=512, overlap=50, limit=None, preview=3):
    raw = json.load(open(in_json, "r", encoding="utf-8"))

    corpus_name = raw.get("corpus_name", os.path.basename(in_json))
    ctx = raw.get("context", raw)  # 关键：真正语料在 context

    docs = list(iter_doc_dicts(ctx))

    out = []
    shown = 0
    for idx, doc in enumerate(docs):
        if limit and idx >= limit:
            break
        if not isinstance(doc, dict):
            continue

        title = pick_title(doc, default=f"{corpus_name}_doc_{idx}")
        text = pick_text(doc)
        if not text:
            continue

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for ci, ch in enumerate(chunks):
            out.append({
                "title": title,
                "abstract": ch,
                "source": os.path.basename(in_json),
                "chunk_id": ci
            })

        if shown < preview:
            print("---- SAMPLE ----")
            print("title:", title[:80])
            print("text_preview:", text[:200].replace("\n", " "))
            print("chunks:", len(chunks))
            shown += 1

    # 兜底：如果没找到任何 dict 文档，但 context 本身是长字符串
    if not out:
        ctx_text = to_text(ctx)
        if len(ctx_text) >= 30:
            for ci, ch in enumerate(chunk_text(ctx_text, chunk_size=chunk_size, overlap=overlap)):
                out.append({
                    "title": str(corpus_name),
                    "abstract": ch,
                    "source": os.path.basename(in_json),
                    "chunk_id": ci
                })
            print("Fallback: used context as plain text.")

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump(out, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"OK: {len(out)} chunks -> {out_json}")

if __name__ == "__main__":
    IN_JSON = r"E:\GraphRAG-Benchmark-main\Datasets\Corpus\medical.json"
    OUT_JSON = r"./data/processed_data.json"
    main(IN_JSON, OUT_JSON, chunk_size=512, overlap=50, limit=None)
