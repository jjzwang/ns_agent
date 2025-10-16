from ...core.schemas import QueryContext
from ...core.router import route as route_intent
from ...systems.netsuite.rag.client import retrieve as ns_retrieve, ask_cited as ns_ask, init_stores as ns_init
from .state import get_or_create_conversation, append_message
from .schemas import ChatSendRequest, ChatSendResponse
try: ns_init()
except Exception: pass
def handle_send(req: ChatSendRequest) -> ChatSendResponse:
    conv = get_or_create_conversation(req.conversation_id)
    append_message(conv, "user", req.text)
    q = QueryContext(text=req.text, filters=req.filters or {})
    decision = route_intent(q.text)
    if decision.get("flow") == "QNA":
        if (req.filters or {}).get("system") == "netsuite":
            ans = ns_ask(q)
            append_message(conv, "assistant", ans.answer_markdown, citations=ans.citations)
            return ChatSendResponse(conversation_id=conv.id, messages=conv.messages)
    if decision.get("flow") == "TASK_EXEC":
        from ...core.orchestrator.planner import build_plan
        plan = build_plan("TASK_EXEC", q, ctx=[])
        preview = "\n".join([f"- {s.kind}: {s.goal}" for s in plan.steps])
        append_message(conv, "assistant", f"Proposed plan:\n{preview}\n\nSay 'run it' to execute.")
        return ChatSendResponse(conversation_id=conv.id, messages=conv.messages)
    append_message(conv, "assistant", f"Parsed as: {decision}. Add filters like {{'system':'netsuite'}} for cited Q&A.")
    return ChatSendResponse(conversation_id=conv.id, messages=conv.messages)
