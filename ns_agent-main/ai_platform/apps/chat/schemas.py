from pydantic import BaseModel
from typing import List, Optional, Dict
class Message(BaseModel):
    role: str  # 'user' | 'assistant' | 'tool'
    text: str
    tool_calls: Optional[List[Dict]] = None
    citations: Optional[List[Dict]] = None
    timestamp: Optional[float] = None
class Conversation(BaseModel):
    id: str
    messages: List[Message] = []
class ChatSendRequest(BaseModel):
    conversation_id: Optional[str] = None
    text: str
    filters: Optional[Dict[str, str]] = None
class ChatSendResponse(BaseModel):
    conversation_id: str
    messages: List[Message]
