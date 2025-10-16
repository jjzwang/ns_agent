from fastapi import FastAPI
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
except Exception:
    pass

# Chat endpoints
app.include_router(chat_router)

@app.post("/v1/route")
def post_route(q: QueryContext):
    return route(q.text)

@app.post("/v1/retrieve")
def post_retrieve(q: QueryContext):
    # Only NetSuite implemented; add other systems later
    if q.filters and q.filters.get("system") == "netsuite":
        return {"passages": [p.dict() for p in ns_retrieve(q, q.filters)]}
    return {"passages": []}

@app.post("/v1/qna/cited")
def post_qna_cited(q: QueryContext):
    if q.filters and q.filters.get("system") == "netsuite":
        return ns_ask(q).dict()
    return {"answer_markdown":"No system matched", "citations":[], "confidence":0.0, "next_actions":[]}

@app.post("/v1/tasks/plan")
def post_tasks_plan(q: QueryContext):
    plan = build_plan("TASK_EXEC", q, ctx=[])
    return plan.dict()

@app.post("/v1/tasks/execute")
def post_tasks_execute(plan: TaskPlan):
    res = run_plan(plan)
    return res.dict()
