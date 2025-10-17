from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from .schemas import ChatSendRequest, ChatSendResponse
from .state import get_history
from .controllers import handle_send
router = APIRouter()
@router.post("/v1/chat/send", response_model=ChatSendResponse)
def chat_send(req: ChatSendRequest):
    return handle_send(req)
@router.get("/v1/chat/history")
def chat_history(conversation_id: str):
    conv = get_history(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv
@router.websocket("/v1/chat/stream")
async def chat_stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            req = ChatSendRequest(**data)
            resp = handle_send(req)
            await ws.send_json({"conversation_id": resp.conversation_id, "messages": [m.dict() for m in resp.messages]})
    except WebSocketDisconnect:
        return
