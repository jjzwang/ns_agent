from typing import List, Dict, Any, Optional
from ...core.schemas import QueryContext, RetrievedPassage, CitedAnswer
from .config import load_config, RAGConfig
from .store.chroma_store import ChromaStore
from .store.bm25 import BM25Memory
from .store.hybrid import fuse

_chroma = None
_bm25 = None
_cfg: Optional[RAGConfig] = None

def init_stores(cfg: Optional[RAGConfig] = None):
    global _cfg, _chroma, _bm25
    _cfg = cfg or load_config()
    _chroma = ChromaStore(_cfg.persist_directory); _chroma.connect()
    _bm25 = BM25Memory()

def _ensure_init():
    if _cfg is None or _chroma is None or _bm25 is None:
        init_stores(load_config())

def retrieve(q: QueryContext, filters: Dict[str, Any] | None = None) -> List[RetrievedPassage]:
    _ensure_init()
    vec_hits = _chroma.query(q.text, k=getattr(_cfg, "k_vec", 40))
    bm_corpus = [h["text"] for h in vec_hits] or [""]
    bm_metas = [h["metadata"] for h in vec_hits] or [{}]
    _bm25.build(bm_corpus, bm_metas)
    bm_hits = _bm25.topk(q.text, k=getattr(_cfg, "k_bm", 200))
    fused = fuse(vec_hits, bm_hits, _cfg.fusion_weight_vec, _cfg.fusion_weight_bm25, k=_cfg.retrieval_k)
    out: List[RetrievedPassage] = []
    for h in fused:
        meta = h.get("metadata", {})
        out.append(RetrievedPassage(
            id=str(meta.get("id") or ""),
            system=str(meta.get("system") or "netsuite"),
            module=str(meta.get("module") or ""),
            doc_type=str(meta.get("doc_type") or "doc"),
            title=str(meta.get("title") or ""),
            text=h.get("text",""),
            url=meta.get("source_url"),
            score=float(h.get("score", 0.0)),
            metadata=meta
        ))
    return out

def ask_cited(q: QueryContext) -> CitedAnswer:
    passages = retrieve(q, filters={"system": "netsuite"})
    cites = [{"id": p.id, "title": p.title, "url": p.url} for p in passages[:3]]
    body = "\n\n".join([f"**{p.title or 'Doc'}** — {p.text[:500]}..." for p in passages[:3]])
    return CitedAnswer(answer_markdown=body, citations=cites, confidence=0.6, next_actions=[])
