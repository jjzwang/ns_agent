from pydantic import BaseModel
from typing import Literal, List, Dict, Optional
from datetime import datetime

class KBChunk(BaseModel):
    id: str
    system: Literal["netsuite","onestream","salesforce","other"]
    module: str
    doc_type: Literal["doc","sop","ticket","code","kb","api"]
    title: str
    section_path: List[str]
    text: str
    version: Optional[str] = None
    entities: List[str] = []
    source_url: Optional[str] = None
    updated_at: Optional[datetime] = None

class QueryContext(BaseModel):
    text: str
    entities: List[str] = []
    environment: Dict[str, str] = {}
    filters: Dict[str, str] = {}

class RetrievedPassage(BaseModel):
    id: str
    system: str
    module: str
    doc_type: str
    title: str
    text: str
    url: Optional[str] = None
    score: float
    metadata: Dict[str, str] = {}

class CitedAnswer(BaseModel):
    answer_markdown: str
    citations: List[Dict]
    confidence: float
    next_actions: List[str] = []

class PlanStep(BaseModel):
    kind: Literal["navigate","select","click","fill","verify","wait","skill"]
    goal: str
    args: Dict[str, str] = {}

class TaskPlan(BaseModel):
    task_id: str
    goal_text: str
    steps: List[PlanStep]
    citations: List[Dict] = []
    confidence: float = 0.0

class ExecutionResult(BaseModel):
    task_id: str
    success: bool
    steps: List[Dict]
    screenshots_dir: Optional[str] = None
    logs_path: Optional[str] = None
