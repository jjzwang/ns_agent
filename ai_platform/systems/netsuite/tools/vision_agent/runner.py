from typing import Dict, Any
def execute_step(step: Dict[str, Any]) -> Dict[str, Any]:
    # TODO: wire to Playwright+Vision. For now, echo.
    return {"ok": True, "echo": step}
