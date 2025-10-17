import time, uuid
from typing import Dict
from .schemas import Conversation, Message
_conversations: Dict[str, Conversation] = {}
def now() -> float: return time.time()
def get_or_create_conversation(conv_id: str | None) -> Conversation:
    if conv_id and conv_id in _conversations:
        return _conversations[conv_id]
    cid = conv_id or str(uuid.uuid4())
    conv = Conversation(id=cid, messages=[])
    _conversations[cid] = conv
    return conv
def append_message(conv: Conversation, role: str, text: str, tool_calls=None, citations=None):
    msg = Message(role=role, text=text, tool_calls=tool_calls, citations=citations, timestamp=now())
    conv.messages.append(msg)
    return msg
def get_history(conv_id: str) -> Conversation | None:
    return _conversations.get(conv_id)
