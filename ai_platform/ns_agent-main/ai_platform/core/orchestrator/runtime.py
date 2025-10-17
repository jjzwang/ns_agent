from ..schemas import TaskPlan, ExecutionResult
from .tools_registry import call_tool
def run_plan(plan: TaskPlan) -> ExecutionResult:
    logs = []
    screenshots_dir = None
    for i, step in enumerate(plan.steps):
        res = call_tool("netsuite.vision.execute_step", {"step": step.dict()})
        logs.append({"i": i, "step": step.dict(), "result": res})
        if res.get("error"): break
    success = all(not e.get("result", {}).get("error") for e in logs)
    return ExecutionResult(task_id=plan.task_id, success=success, steps=logs, screenshots_dir=screenshots_dir, logs_path=None)
