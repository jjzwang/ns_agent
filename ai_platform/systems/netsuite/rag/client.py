import os
from typing import List, Dict, Any, Optional
from ....core.schemas import QueryContext, RetrievedPassage, CitedAnswer
from .config import load_config, RAGConfig
from .store.chroma_store import ChromaStore
from .store.bm25 import BM25Memory
from .store.hybrid import fuse

# LLM for answer generation
try:
    from openai import OpenAI
    llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: OpenAI not installed for answer generation")

_chroma = None
_bm25 = None
_cfg: Optional[RAGConfig] = None

def init_stores(cfg: Optional[RAGConfig] = None):
    """Initialize ChromaDB and BM25 stores."""
    global _cfg, _chroma, _bm25
    _cfg = cfg or load_config()
    _chroma = ChromaStore(_cfg.persist_directory)
    _chroma.connect()
    _bm25 = BM25Memory()

def _ensure_init():
    """Ensure stores are initialized before use."""
    if _cfg is None or _chroma is None or _bm25 is None:
        init_stores(load_config())

def retrieve(q: QueryContext, filters: Dict[str, Any] | None = None) -> List[RetrievedPassage]:
    """
    Retrieve relevant passages using hybrid search (vector + BM25).
    
    Args:
        q: Query context with search text
        filters: Optional filters (e.g., {"system": "netsuite"})
    
    Returns:
        List of retrieved passages sorted by relevance
    """
    _ensure_init()
    
    # Vector search
    vec_hits = _chroma.query(q.text, k=getattr(_cfg, "k_vec", 40))
    
    # BM25 search on vector results
    bm_corpus = [h["text"] for h in vec_hits] or [""]
    bm_metas = [h["metadata"] for h in vec_hits] or [{}]
    _bm25.build(bm_corpus, bm_metas)
    bm_hits = _bm25.topk(q.text, k=getattr(_cfg, "k_bm", 200))
    
    # Fusion
    fused = fuse(
        vec_hits, 
        bm_hits, 
        _cfg.fusion_weight_vec, 
        _cfg.fusion_weight_bm25, 
        k=_cfg.retrieval_k
    )
    
    # Convert to RetrievedPassage objects
    out: List[RetrievedPassage] = []
    for h in fused:
        meta = h.get("metadata", {})
        out.append(RetrievedPassage(
            id=str(meta.get("id") or ""),
            system=str(meta.get("system") or "netsuite"),
            module=str(meta.get("module") or ""),
            doc_type=str(meta.get("doc_type") or "doc"),
            title=str(meta.get("title") or ""),
            text=h.get("text", ""),
            url=meta.get("source_url"),
            score=float(h.get("score", 0.0)),
            metadata=meta
        ))
    return out

def generate_answer_with_llm(question: str, passages: List[RetrievedPassage]) -> tuple[str, float]:
    """
    Generate an answer using LLM with retrieved context.
    
    Args:
        question: The user's question
        passages: Retrieved relevant passages
    
    Returns:
        Tuple of (answer_markdown, confidence_score)
    """
    if not LLM_AVAILABLE or not passages:
        return _fallback_answer(passages), 0.3
    
    # Format context from passages
    context = "\n\n".join([
        f"[Source {i+1}: {p.title}]\n{p.text}"
        for i, p in enumerate(passages[:5])
    ])
    
    # System prompt for NetSuite Q&A
    system_prompt = """You are a NetSuite expert assistant. Answer questions based ONLY on the provided documentation context.

        Rules:
        - Be concise but complete
        - Cite sources using [Source N] notation
        - If the context doesn't contain the answer, say so clearly
        - Include step-by-step instructions when relevant
        - Use bullet points for clarity
        - Be specific about NetSuite UI elements, statuses, and workflows"""

    user_prompt = f"""Question: {question}

        Context from NetSuite Documentation:
        {context}

        Provide a helpful answer based on the context above. Cite your sources."""

    try:
        response = llm_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,  # Lower temperature for factual accuracy
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Estimate confidence based on context quality
        confidence = min(0.9, 0.5 + (len(passages) * 0.1))
        
        return answer, confidence
        
    except Exception as e:
        print(f"LLM answer generation failed: {e}")
        return _fallback_answer(passages), 0.3

def _fallback_answer(passages: List[RetrievedPassage]) -> str:
    """Fallback answer when LLM is unavailable."""
    if not passages:
        return "I couldn't find relevant information in the NetSuite documentation."
    
    # Simple concatenation as fallback
    parts = []
    for i, p in enumerate(passages[:3], 1):
        parts.append(f"**{i}. {p.title}**\n\n{p.text[:300]}...")
    
    return "\n\n".join(parts)

def suggest_next_actions(question: str, passages: List[RetrievedPassage]) -> List[str]:
    """Suggest relevant follow-up actions based on the question."""
    actions = []
    
    q_lower = question.lower()
    
    # Rule-based action suggestions
    if "how" in q_lower or "bill" in q_lower:
        actions.append("Create a task plan to execute this workflow")
    
    if "status" in q_lower or "pending" in q_lower:
        actions.append("Check current status of open transactions")
    
    if passages and any("approval" in p.text.lower() for p in passages):
        actions.append("Review approval workflow requirements")
    
    # Always offer to search more
    if len(passages) < 3:
        actions.append("Search for more detailed documentation")
    
    return actions[:3]  # Limit to top 3 actions

def ask_cited(q: QueryContext) -> CitedAnswer:
    """
    Answer a question with citations using RAG + LLM.
    
    This is the main entry point for Q&A flow.
    
    Args:
        q: Query context with the question
    
    Returns:
        CitedAnswer with generated answer, citations, and suggested actions
    """
    # Retrieve relevant passages
    passages = retrieve(q, filters={"system": "netsuite"})
    
    if not passages:
        return CitedAnswer(
            answer_markdown="I couldn't find relevant information in the NetSuite documentation. Please try rephrasing your question or check if the documentation has been ingested.",
            citations=[],
            confidence=0.0,
            next_actions=["Ingest NetSuite documentation", "Try a different question"]
        )
    
    # Generate answer using LLM
    answer, confidence = generate_answer_with_llm(q.text, passages)
    
    # Build citations
    citations = [
        {
            "id": p.id,
            "title": p.title,
            "url": p.url,
            "doc_type": p.doc_type,
            "module": p.module,
            "score": round(p.score, 3)
        }
        for p in passages[:5]
    ]
    
    # Suggest next actions
    next_actions = suggest_next_actions(q.text, passages)
    
    return CitedAnswer(
        answer_markdown=answer,
        citations=citations,
        confidence=confidence,
        next_actions=next_actions
    )
