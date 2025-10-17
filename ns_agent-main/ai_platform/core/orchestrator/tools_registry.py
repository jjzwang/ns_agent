def call_tool(name: str, args: dict) -> dict:
    if name == "netsuite.vision.execute_step":
        from ...systems.netsuite.tools.vision_agent.runner import execute_step
        return execute_step(**args)
    return {"error": f"Unknown tool: {name}"}
