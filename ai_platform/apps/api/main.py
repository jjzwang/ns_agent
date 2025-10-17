from fastapi import FastAPI, HTTPException
from ...core.schemas import QueryContext, TaskPlan
from ...core.router import route
from ...core.orchestrator.planner import build_plan
from ...core.orchestrator.runtime import run_plan
from ...systems.netsuite.rag.client import retrieve as ns_retrieve, ask_cited as ns_ask, init_stores as ns_init
from ..chat.server import router as chat_router

app = FastAPI(title="NetSuite AI: RAG + Vision + Chat")

# Init inline RAG
try:
    ns_init()
except Exception as e:
    print(f"Warning: Failed to initialize RAG stores: {e}")

# Chat endpoints
app.include_router(chat_router)

@app.post("/v1/route")
def post_route(q: QueryContext):
    """Route a query to determine system and flow type."""
    return route(q.text)

@app.post("/v1/retrieve")
def post_retrieve(q: QueryContext):
    """Retrieve relevant passages for a query."""
    # Only NetSuite implemented; add other systems later
    if q.filters and q.filters.get("system") == "netsuite":
        passages = ns_retrieve(q, q.filters)
        return {"passages": [p.dict() for p in passages]}
    return {"passages": []}

@app.post("/v1/qna/cited")
def post_qna_cited(q: QueryContext):
    """Answer a question with citations (Q&A flow)."""
    if q.filters and q.filters.get("system") == "netsuite":
        return ns_ask(q).dict()
    return {
        "answer_markdown": "No system matched. Add filters like {'system':'netsuite'}", 
        "citations": [], 
        "confidence": 0.0, 
        "next_actions": []
    }

@app.post("/v1/tasks/plan")
def post_tasks_plan(q: QueryContext):
    """
    Create a task execution plan with RAG context.
    
    This endpoint:
    1. Routes the query to determine flow type
    2. Retrieves relevant documentation
    3. Uses LLM to generate a dynamic plan
    """
    # Route to determine flow
    routing = route(q.text)
    flow = routing.get("flow", "TASK_EXEC")
    
    # Set system filter if not provided
    if not q.filters:
        q.filters = {"system": "netsuite"}
    elif "system" not in q.filters:
        q.filters["system"] = "netsuite"
    
    # Retrieve relevant context
    try:
        context_passages = ns_retrieve(q, q.filters)
    except Exception as e:
        print(f"Retrieval failed: {e}")
        context_passages = []
    
    # Build plan with LLM using retrieved context
    try:
        plan = build_plan(flow, q, context_passages)
        return plan.dict()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build plan: {str(e)}"
        )

@app.post("/v1/tasks/execute")
def post_tasks_execute(plan: TaskPlan):
    """
    Execute a task plan using vision automation.
    Currently stubbed - will use Playwright + Vision.
    """
    try:
        result = run_plan(plan)
        return result.dict()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Execution failed: {str(e)}"
        )

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "rag_initialized": _is_rag_initialized(),
        "systems": ["netsuite"]
    }

def _is_rag_initialized() -> bool:
    """Check if RAG system is properly initialized."""
    try:
        from ...systems.netsuite.rag.client import _cfg, _chroma, _bm25
        return _cfg is not None and _chroma is not None and _bm25 is not None
    except:
        return False
