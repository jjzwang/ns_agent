import os, json, glob
from .config import RAGConfig
from .store.chroma_store import ChromaStore
from .store.bm25 import BM25Memory
def ingest_documents(cfg: RAGConfig):
    jsonls = glob.glob("ai_platform/data/cleaned/netsuite/*.jsonl")
    if not jsonls:
        return {"error": "No JSONL files in ai_platform/data/cleaned/netsuite"}
    chroma = ChromaStore(cfg.persist_directory); chroma.connect()
    bm25 = BM25Memory()
    ids, texts, metas = [], [], []
    for path in jsonls:
        with open(path, "r") as f:
            for line in f:
                row = json.loads(line)
                _id = row.get("id") or f"{os.path.basename(path)}:{len(ids)}"
                ids.append(_id); texts.append(row["text"])
                metas.append({
                    "id": _id,
                    "system": row.get("system","netsuite"),
                    "module": row.get("module",""),
                    "doc_type": row.get("doc_type","doc"),
                    "title": row.get("title",""),
                    "section_path": row.get("section_path",[]),
                    "source_url": row.get("source_url"),
                    "updated_at": row.get("updated_at"),
                })
    chroma.add_documents(ids, texts, metas)
    bm25.build(texts, metas)
    return {"ok": True, "count": len(ids)}
