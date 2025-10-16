# NetSuite AI: RAG + Vision + Chat (Complete Repo)

This repository provides a working skeleton for a **NetSuite automation copilot**:
- **RAG** (Retrieval-Augmented Generation) for NetSuite knowledge (vendor docs + your SOPs)
- **Planner** to convert goals into step plans
- **Vision Executor** stubs (Playwright + Vision) to act on UI
- **Chat** API (REST + WebSocket) to interact conversationally

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional (for vision executor later)
# python -m playwright install chromium
uvicorn ai_platform.apps.api.main:app --reload
```

### Test the endpoints
```bash
# Route
curl -s localhost:8000/v1/route -X POST -H "Content-Type: application/json"   -d '{"text":"find an open PO and bill it"}' | jq

# Retrieval (uses inline RAG + sample JSONL data)
curl -s localhost:8000/v1/retrieve -X POST -H "Content-Type: application/json"   -d '{"text":"bill an open PO","filters":{"system":"netsuite"}}' | jq

# Chat (Q&A)
curl -s localhost:8000/v1/chat/send -X POST -H "Content-Type: application/json"   -d '{"text":"How do I bill an open PO?","filters":{"system":"netsuite"}}' | jq

# Plan
curl -s localhost:8000/v1/tasks/plan -X POST -H "Content-Type: application/json"   -d '{"text":"find an open PO and bill it","filters":{"system":"netsuite"}}' | jq

# Execute (stubbed; echoes each step)
curl -s localhost:8000/v1/tasks/execute -X POST -H "Content-Type: application/json"   -d '{"task_id":"task-001","goal_text":"bill an open PO","steps":[{"kind":"navigate","goal":"Open PO list"},{"kind":"click","goal":"Click Bill"}]}' | jq
```

## Where to add your real pieces
- **Inline RAG**: `ai_platform/systems/netsuite/rag/` (replace store or retrieval as you like)
- **Docs**: put heading-aware JSONL chunks in `ai_platform/data/cleaned/netsuite/`
- **Vision**: implement Playwright + Vision in `ai_platform/systems/netsuite/tools/vision_agent/`

## Config
Edit `config.yaml` to point to your directories and tweak hybrid retrieval params.
