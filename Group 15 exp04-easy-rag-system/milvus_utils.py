# import streamlit as st
# # Use MilvusClient for Lite version
# from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
# import time
# import os
#
# # Import config variables including the global map
# from config import (
#     MILVUS_LITE_DATA_PATH, COLLECTION_NAME, EMBEDDING_DIM,
#     MAX_ARTICLES_TO_INDEX, INDEX_METRIC_TYPE, INDEX_TYPE, INDEX_PARAMS,
#     SEARCH_PARAMS, TOP_K, id_to_doc_map
# )
#
# @st.cache_resource
# def get_milvus_client():
#     """Initializes and returns a MilvusClient instance for Milvus Lite."""
#     try:
#         st.write(f"Initializing Milvus Lite client with data path: {MILVUS_LITE_DATA_PATH}")
#         # Ensure the directory for the data file exists
#         os.makedirs(os.path.dirname(MILVUS_LITE_DATA_PATH), exist_ok=True)
#         # The client connects to the local file specified
#         client = MilvusClient(uri=MILVUS_LITE_DATA_PATH)
#         st.success("Milvus Lite client initialized!")
#         return client
#     except Exception as e:
#         st.error(f"Failed to initialize Milvus Lite client: {e}")
#         return None
#
# @st.cache_resource
# def setup_milvus_collection(_client):
#     """Ensures the specified collection exists and is set up correctly in Milvus Lite."""
#     if not _client:
#         st.error("Milvus client not available.")
#         return False
#     try:
#         collection_name = COLLECTION_NAME
#         dim = EMBEDDING_DIM
#
#         has_collection = collection_name in _client.list_collections()
#
#         if not has_collection:
#             st.write(f"Collection '{collection_name}' not found. Creating...")
#             # Define fields using new API style if needed (older style might still work)
#             fields = [
#                 FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
#                 FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
#                 # You can add other scalar fields directly here for storage
#                 FieldSchema(name="content_preview", dtype=DataType.VARCHAR, max_length=500), # Example
#             ]
#             schema = CollectionSchema(fields, f"PubMed Lite RAG (dim={dim})")
#
#             _client.create_collection(
#                 collection_name=collection_name,
#                 schema=schema # Pass schema directly or define dimension/primary field name
#                 # Or simpler:
#                 # dimension=dim,
#                 # primary_field_name="id",
#                 # vector_field_name="embedding",
#                 # metric_type=INDEX_METRIC_TYPE
#             )
#             st.write(f"Collection '{collection_name}' created.")
#
#             # Create an index
#             st.write(f"Creating index ({INDEX_TYPE})...")
#             index_params = _client.prepare_index_params()
#             index_params.add_index(
#                 field_name="embedding",
#                 index_type=INDEX_TYPE,
#                 metric_type=INDEX_METRIC_TYPE,
#                 params=INDEX_PARAMS
#             )
#             _client.create_index(collection_name, index_params)
#             st.success(f"Index created for collection '{collection_name}'.")
#         else:
#             st.write(f"Found existing collection: '{collection_name}'.")
#             # Optional: Check schema compatibility if needed
#
#         # Determine current entity count (fallback between num_entities and stats)
#         try:
#             if hasattr(_client, 'num_entities'):
#                 current_count = _client.num_entities(collection_name)
#             else:
#                 stats = _client.get_collection_stats(collection_name)
#                 current_count = int(stats.get("row_count", stats.get("rowCount", 0)))
#             st.write(f"Collection '{collection_name}' ready. Current entity count: {current_count}")
#         except Exception:
#             st.write(f"Collection '{collection_name}' ready.")
#
#         return True # Indicate collection is ready
#
#     except Exception as e:
#         st.error(f"Error setting up Milvus collection '{COLLECTION_NAME}': {e}")
#         return False
#
#
# def index_data_if_needed(client, data, embedding_model):
#     """Checks if data needs indexing and performs it using MilvusClient."""
#     global id_to_doc_map # Modify the global map
#
#     if not client:
#         st.error("Milvus client not available for indexing.")
#         return False
#
#     collection_name = COLLECTION_NAME
#     # Retrieve current entity count with fallback
#     try:
#         if hasattr(client, 'num_entities'):
#             current_count = client.num_entities(collection_name)
#         else:
#             stats = client.get_collection_stats(collection_name)
#             current_count = int(stats.get("row_count", stats.get("rowCount", 0)))
#     except Exception:
#         st.write(f"Could not retrieve entity count, attempting to (re)setup collection.")
#         if not setup_milvus_collection(client):
#             return False
#         current_count = 0  # Assume empty after setup
#
#     st.write(f"Entities currently in Milvus collection '{collection_name}': {current_count}")
#
#     data_to_index = data[:MAX_ARTICLES_TO_INDEX] # Limit data for demo
#     needed_count = 0
#     docs_for_embedding = []
#     data_to_insert = [] # List of dictionaries for MilvusClient insert
#     temp_id_map = {} # Build a temporary map first
#
#     # Prepare data
#     with st.spinner("Preparing data for indexing..."):
#         for i, doc in enumerate(data_to_index):
#              title = doc.get('title', '') or ""
#              abstract = doc.get('abstract', '') or ""
#              content = f"Title: {title}\nAbstract: {abstract}".strip()
#              if not content:
#                  continue
#
#              doc_id = i # Use list index as ID
#              needed_count += 1
#              temp_id_map[doc_id] = {
#                  'title': title, 'abstract': abstract, 'content': content
#              }
#              docs_for_embedding.append(content)
#              # Prepare data in dict format for MilvusClient
#              data_to_insert.append({
#                  "id": doc_id,
#                  "embedding": None, # Placeholder, will be filled after encoding
#                  "content_preview": content[:500] # Store preview if field exists
#              })
#
#
#     if current_count < needed_count and docs_for_embedding:
#         st.warning(f"Indexing required ({current_count}/{needed_count} documents found). This may take a while...")
#
#         st.write(f"Embedding {len(docs_for_embedding)} documents...")
#         with st.spinner("Generating embeddings..."):
#             start_embed = time.time()
#             embeddings = embedding_model.encode(docs_for_embedding, show_progress_bar=True)
#             end_embed = time.time()
#             st.write(f"Embedding took {end_embed - start_embed:.2f} seconds.")
#
#         # Fill in the embeddings
#         for i, emb in enumerate(embeddings):
#             data_to_insert[i]["embedding"] = emb
#
#         st.write("Inserting data into Milvus Lite...")
#         with st.spinner("Inserting..."):
#             try:
#                 start_insert = time.time()
#                 # MilvusClient uses insert() with list of dicts
#                 res = client.insert(collection_name=collection_name, data=data_to_insert)
#                 # Milvus Lite might automatically flush or sync, explicit flush isn't usually needed/available
#                 end_insert = time.time()
#                 # 使用 len(data_to_insert) 作为成功插入的数量，因为 res 可能没有 primary_keys 属性
#                 inserted_count = len(data_to_insert)
#                 st.success(f"Successfully attempted to index {inserted_count} documents. Insert took {end_insert - start_insert:.2f} seconds.")
#                 # Update the global map ONLY after successful insertion attempt
#                 id_to_doc_map.update(temp_id_map)
#                 return True
#             except Exception as e:
#                 st.error(f"Error inserting data into Milvus Lite: {e}")
#                 return False
#     elif current_count >= needed_count:
#         st.write("Data count suggests indexing is complete.")
#         # Populate the global map if it's empty but indexing isn't needed
#         if not id_to_doc_map:
#             id_to_doc_map.update(temp_id_map)
#         return True
#     else: # No docs_for_embedding found
#          st.error("No valid text content found in the data to index.")
#          return False
#
#
# def search_similar_documents(client, query, embedding_model):
#     """Searches Milvus Lite for documents similar to the query using MilvusClient."""
#     if not client or not embedding_model:
#         st.error("Milvus client or embedding model not available for search.")
#         return [], []
#
#     collection_name = COLLECTION_NAME
#     try:
#         query_embedding = embedding_model.encode([query])[0]
#
#         # 重写search调用，使用更兼容的方式
#         search_params = {
#             "collection_name": collection_name,
#             "data": [query_embedding],
#             "anns_field": "embedding",
#             "limit": TOP_K,
#             "output_fields": ["id"]
#         }
#
#         # 尝试不同的方式传递搜索参数
#         if hasattr(client, 'search_with_params'):
#             # 如果存在专门的方法
#             res = client.search_with_params(**search_params, search_params=SEARCH_PARAMS)
#         else:
#             # 标准方法，直接设置参数（当前版本会导致参数冲突）
#             try:
#                 # 尝试1：不传递param参数
#                 res = client.search(**search_params)
#             except Exception as e1:
#                 st.warning(f"搜索尝试1失败: {e1}，将尝试备用方法...")
#                 try:
#                     # 尝试2：通过搜索参数关键字传递
#                     res = client.search(**search_params, **SEARCH_PARAMS)
#                 except Exception as e2:
#                     st.warning(f"搜索尝试2失败: {e2}，将尝试最后一种方法...")
#                     # 尝试3：结合参数
#                     final_params = search_params.copy()
#                     final_params["nprobe"] = SEARCH_PARAMS.get("nprobe", 16)
#                     res = client.search(**final_params)
#
#         # Process results (structure might differ slightly)
#         # client.search returns a list of lists of hits (one list per query vector)
#         if not res or not res[0]:
#             return [], []
#
#         hit_ids = [hit['id'] for hit in res[0]]
#         distances = [hit['distance'] for hit in res[0]]
#         return hit_ids, distances
#     except Exception as e:
#         st.error(f"Error during Milvus Lite search: {e}")
#         return [], []

import os
import time
import pickle
import numpy as np
import streamlit as st

from config import (
    MILVUS_LITE_DATA_PATH,  # 这里我们拿来当本地索引存储路径用
    MAX_ARTICLES_TO_INDEX, TOP_K, id_to_doc_map
)

# 把 Milvus 的 db 路径改成 faiss 的本地文件（不想改 config.py 也行，就复用变量）
FAISS_INDEX_PATH = MILVUS_LITE_DATA_PATH.replace(".db", ".faiss")
FAISS_META_PATH = MILVUS_LITE_DATA_PATH.replace(".db", ".meta.pkl")

@st.cache_resource
def get_milvus_client():
    """
    为了不改 app.py，这里返回一个“伪 client”，实际存 FAISS index 和 id 列表
    """
    return {"index": None, "ids": []}

@st.cache_resource
def setup_milvus_collection(_client):
    """
    加载或初始化 FAISS
    """
    try:
        # 延迟导入，避免没装 faiss 时直接炸
        import faiss  # type: ignore

        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_META_PATH):
            _client["index"] = faiss.read_index(FAISS_INDEX_PATH)
            with open(FAISS_META_PATH, "rb") as f:
                meta = pickle.load(f)
            _client["ids"] = meta.get("ids", [])
            # 同时恢复 id_to_doc_map（用于 UI 展示）
            id_to_doc_map.update(meta.get("id_to_doc_map", {}))
            st.success(f"Loaded FAISS index. vectors={len(_client['ids'])}")
        else:
            _client["index"] = None
            _client["ids"] = []
            st.write("FAISS index not found. Will build on first indexing.")
        return True
    except Exception as e:
        st.error(f"Error setting up FAISS: {e}")
        return False

def _save_faiss(_client):
    import faiss  # type: ignore
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH) or ".", exist_ok=True)
    faiss.write_index(_client["index"], FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "wb") as f:
        pickle.dump({"ids": _client["ids"], "id_to_doc_map": id_to_doc_map}, f)

def index_data_if_needed(client, data, embedding_model):
    """
    用 embedding_model.encode 生成向量，然后插入 FAISS
    """
    if not client or not embedding_model:
        st.error("FAISS client or embedding model not available.")
        return False

    data_to_index = data[:MAX_ARTICLES_TO_INDEX]
    docs_for_embedding = []
    temp_id_map = {}
    ids = []

    for i, doc in enumerate(data_to_index):
        title = (doc.get("title", "") or "").strip()
        abstract = (doc.get("abstract", "") or "").strip()
        content = f"Title: {title}\nAbstract: {abstract}".strip()
        if not content:
            continue

        doc_id = i  # 保持与你现在逻辑一致
        temp_id_map[doc_id] = {"title": title, "abstract": abstract, "content": content}
        docs_for_embedding.append(content)
        ids.append(doc_id)

    if not docs_for_embedding:
        st.error("No valid text content found in the data to index.")
        return False

    # 如果已经有索引且数量够，就跳过
    if client.get("index") is not None and len(client.get("ids", [])) >= len(ids):
        st.write("Data count suggests indexing is complete.")
        if not id_to_doc_map:
            id_to_doc_map.update(temp_id_map)
        return True

    st.warning(f"Indexing required. Building embeddings for {len(docs_for_embedding)} docs...")

    start_embed = time.time()
    embs = embedding_model.encode(docs_for_embedding, show_progress_bar=True)
    embs = np.array(embs, dtype=np.float32)
    end_embed = time.time()
    st.write(f"Embedding took {end_embed - start_embed:.2f}s")

    # 归一化（可选：让内积更像余弦）
    # embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

    # 初始化 FAISS index（L2）
    import faiss  # type: ignore
    d = embs.shape[1]
    index = faiss.IndexFlatL2(d)

    index.add(embs)

    client["index"] = index
    client["ids"] = ids

    id_to_doc_map.clear()
    id_to_doc_map.update(temp_id_map)

    _save_faiss(client)
    st.success(f"FAISS indexing done. vectors={len(ids)}")
    return True

def search_similar_documents(client, query, embedding_model):
    if not client or not embedding_model or client.get("index") is None:
        st.error("FAISS index not ready.")
        return [], []

    q = embedding_model.encode([query])
    q = np.array(q, dtype=np.float32)

    # q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

    D, I = client["index"].search(q, TOP_K)
    # I 返回的是向量序号，我们用 client["ids"] 映射回 doc_id
    hit_ids = []
    distances = []
    for idx, dist in zip(I[0].tolist(), D[0].tolist()):
        if idx < 0:
            continue
        hit_ids.append(client["ids"][idx])
        distances.append(dist)
    return hit_ids, distances
