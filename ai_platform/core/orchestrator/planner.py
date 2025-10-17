import json
import os
from typing import List, Optional
from pathlib import Path
from ..schemas import QueryContext, RetrievedPassage, TaskPlan, PlanStep

# LLM client - using OpenAI as example, easily swappable
try:
    from google import genai
    client = genai(api_key=os.getenv("OPENAI_API_KEY"))
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: OpenAI not installed. Install with: pip install openai")

def load_prompt_template(flow: str) -> str:
    """Load the appropriate prompt template for the flow type."""
    prompt_file = Path(__file__).parent / "prompts" / f"{flow.lower()}.md"
    if prompt_file.exists():
        return prompt_file.read_text()
    # Fallback inline prompt
    return """You are a task planner for NetSuite automation. Given a goal and context, create a step-by-step plan.

        Available step types:
        - navigate: Go to a specific page/list (args: label, url)
        - select: Filter or select items (args: field, value, status)
        - click: Click a button/link (args: button, label, row)
        - fill: Fill form fields (args: field, value, strategy)
        - verify: Check if something is true (args: field, expected_value, status)
        - wait: Wait for condition (args: condition, timeout)
        - skill: Execute a complex reusable skill (args: skill_name, params)

        Output valid JSON only, no explanation."""

def format_context(passages: List[RetrievedPassage]) -> str:
    """Format retrieved passages into context string."""
    if not passages:
        return "No relevant documentation found."
    
    context_parts = []
    for i, p in enumerate(passages[:5], 1):  # Top 5 passages
        context_parts.append(
            f"[Doc {i}] {p.title}\n"
            f"Module: {p.module} | Type: {p.doc_type}\n"
            f"{p.text[:400]}...\n"
        )
    return "\n".join(context_parts)

def call_llm_for_plan(goal: str, context: str, prompt_template: str) -> Optional[dict]:
    """Call LLM to generate a plan. Returns parsed JSON or None."""
    if not LLM_AVAILABLE:
        print("LLM not available, using fallback plan")
        return None
    
    # Build the full prompt
    full_prompt = f"""{prompt_template}

        GOAL: {goal}

        CONTEXT FROM NETSUITE DOCUMENTATION:
        {context}

        Generate a JSON plan with this structure:
        {{
        "steps": [
            {{"kind": "navigate", "goal": "description", "args": {{"key": "value"}}}},
            ...
        ],
        "confidence": 0.0-1.0,
        "reasoning": "brief explanation"
        }}

        Respond with ONLY the JSON, no other text."""

    try:
        # Using gpt-4o-mini for cost efficiency, upgrade to gpt-4o for better quality
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a NetSuite automation planner. Output only valid JSON."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.1,  # Low temperature for consistency
            max_tokens=2000
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON if wrapped in markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        return json.loads(content)
    
    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM response as JSON: {e}")
        print(f"Response was: {content[:200]}...")
        return None
    except Exception as e:
        print(f"LLM call failed: {e}")
        return None

def validate_and_parse_steps(llm_output: dict) -> List[PlanStep]:
    """Validate and convert LLM output to PlanStep objects."""
    steps = []
    valid_kinds = {"navigate", "select", "click", "fill", "verify", "wait", "skill"}
    
    for step_data in llm_output.get("steps", []):
        kind = step_data.get("kind", "")
        if kind not in valid_kinds:
            print(f"Warning: Invalid step kind '{kind}', skipping")
            continue
        
        steps.append(PlanStep(
            kind=kind,
            goal=step_data.get("goal", ""),
            args=step_data.get("args", {})
        ))
    
    return steps

def build_plan(flow: str, q: QueryContext, ctx: List[RetrievedPassage]) -> TaskPlan:
    """
    Build a dynamic task plan using LLM with retrieved context.
    
    Args:
        flow: The flow type (e.g., "TASK_EXEC", "QNA")
        q: Query context with the user's goal
        ctx: Retrieved passages from RAG system
    
    Returns:
        TaskPlan with dynamically generated steps
    """
    # Load prompt template
    prompt_template = load_prompt_template(flow)
    
    # Format retrieved context
    context_str = format_context(ctx)
    
    # Call LLM to generate plan
    llm_output = call_llm_for_plan(q.text, context_str, prompt_template)
    

    steps = validate_and_parse_steps(llm_output)
    confidence = float(llm_output.get("confidence", 0.7))
        
    # Build citations from context
    citations = [
        {
            "id": p.id,
            "title": p.title,
            "url": p.url,
            "doc_type": p.doc_type
        }
        for p in ctx[:3]
    ]
    
    # Generate task ID
    task_id = f"task-{hash(q.text) % 100000:05d}"
    
    return TaskPlan(
        task_id=task_id,
        goal_text=q.text,
        steps=steps,
        citations=citations,
        confidence=confidence
    )
