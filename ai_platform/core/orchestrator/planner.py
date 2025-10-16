from typing import List
from ..schemas import QueryContext, RetrievedPassage, TaskPlan, PlanStep
def build_plan(flow: str, q: QueryContext, ctx: List[RetrievedPassage]) -> TaskPlan:
    steps = [
        PlanStep(kind="navigate", goal="Open PO list or saved search", args={"label":"Purchase Orders"}),
        PlanStep(kind="select", goal="Filter to Pending Billing", args={"status":"Pending Billing"}),
        PlanStep(kind="click", goal="Open first PO row", args={"row":"0"}),
        PlanStep(kind="click", goal="Click Bill button", args={"button":"Bill"}),
        PlanStep(kind="fill", goal="Fill required blanks", args={"strategy":"required_only"}),
        PlanStep(kind="click", goal="Save", args={"button":"Save"}),
        PlanStep(kind="click", goal="Submit for Approval", args={"button":"Submit for Approval"}),
        PlanStep(kind="verify", goal="Status shows Pending Approval", args={"status":"Pending Approval"}),
    ]
    return TaskPlan(task_id="task-001", goal_text=q.text, steps=steps, citations=[], confidence=0.7)
