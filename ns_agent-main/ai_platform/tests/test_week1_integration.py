"""
Integration tests for Week 1 fixes:
- LLM-based planning
- Enhanced RAG with answer generation
- Full pipeline testing

Run with: pytest test_week1_integration.py -v
"""
import os
import sys
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_platform.core.schemas import QueryContext, RetrievedPassage, PlanStep
from ai_platform.core.orchestrator.planner import (
    build_plan, 
    format_context, 
    get_fallback_plan,
    validate_and_parse_steps
)
from ai_platform.systems.netsuite.rag.client import (
    retrieve,
    ask_cited,
    generate_answer_with_llm,
    init_stores
)

# Test fixtures
@pytest.fixture(scope="module")
def sample_context():
    """Sample retrieved passages for testing."""
    return [
        RetrievedPassage(
            id="test_001",
            system="netsuite",
            module="Procure-to-Pay",
            doc_type="sop",
            title="Bill an Open Purchase Order",
            text="From a saved search 'POs Pending Billing', open a purchase order. Click Bill to create a Vendor Bill. Fill required fields like Bill #, then Save. Submit for Approval; status should become Pending Approval.",
            url="internal://sop/p2p-billing",
            score=0.95,
            metadata={}
        ),
        RetrievedPassage(
            id="test_002",
            system="netsuite",
            module="Procure-to-Pay",
            doc_type="doc",
            title="Purchase Order Statuses",
            text="A purchase order marked Pending Billing can be billed. Pressing Bill opens a Vendor Bill form with most fields pre-populated.",
            url="internal://docs/po-status",
            score=0.82,
            metadata={}
        )
    ]

@pytest.fixture(scope="module")
def sample_query():
    """Sample query context."""
    return QueryContext(
        text="find an open PO and bill it",
        filters={"system": "netsuite"}
    )

# Test Context Formatting
class TestContextFormatting:
    def test_format_context_with_passages(self, sample_context):
        """Test that context is formatted correctly."""
        formatted = format_context(sample_context)
        
        assert "Bill an Open Purchase Order" in formatted
        assert "Procure-to-Pay" in formatted
        assert "sop" in formatted
        assert len(formatted) > 100
    
    def test_format_context_empty(self):
        """Test formatting with no passages."""
        formatted = format_context([])
        assert "No relevant documentation found" in formatted

# Test Fallback Plans
class TestFallbackPlans:
    def test_fallback_plan_billing(self):
        """Test fallback plan for billing workflow."""
        steps = get_fallback_plan("bill an open PO")
        
        assert len(steps) >= 5
        assert steps[0].kind == "navigate"
        assert any(s.kind == "click" and "Bill" in s.goal for s in steps)
        assert any(s.kind == "verify" for s in steps)
    
    def test_fallback_plan_generic(self):
        """Test generic fallback for unknown tasks."""
        steps = get_fallback_plan("do something random")
        
        assert len(steps) >= 1
        assert steps[0].kind == "navigate"

# Test Step Validation
class TestStepValidation:
    def test_validate_valid_steps(self):
        """Test validation accepts valid step structures."""
        llm_output = {
            "steps": [
                {"kind": "navigate", "goal": "Open POs", "args": {"label": "POs"}},
                {"kind": "click", "goal": "Click Bill", "args": {"button": "Bill"}},
                {"kind": "verify", "goal": "Check status", "args": {"status": "Pending"}}
            ]
        }
        
        steps = validate_and_parse_steps(llm_output)
        
        assert len(steps) == 3
        assert all(isinstance(s, PlanStep) for s in steps)
        assert steps[0].kind == "navigate"
    
    def test_validate_invalid_kind(self):
        """Test validation filters out invalid step kinds."""
        llm_output = {
            "steps": [
                {"kind": "navigate", "goal": "Valid", "args": {}},
                {"kind": "invalid_action", "goal": "Invalid", "args": {}},
                {"kind": "click", "goal": "Valid", "args": {}}
            ]
        }
        
        steps = validate_and_parse_steps(llm_output)
        
        assert len(steps) == 2  # Invalid step filtered out
        assert steps[0].kind == "navigate"
        assert steps[1].kind == "click"

# Test Plan Building
class TestPlanBuilding:
    def test_build_plan_with_context(self, sample_query, sample_context):
        """Test building a plan with LLM (or fallback)."""
        plan = build_plan("TASK_EXEC", sample_query, sample_context)
        
        # Verify plan structure
        assert plan.task_id.startswith("task-")
        assert plan.goal_text == sample_query.text
        assert len(plan.steps) >= 3
        assert 0.0 <= plan.confidence <= 1.0
        
        # Check for key steps in billing workflow
        step_kinds = [s.kind for s in plan.steps]
        assert "navigate" in step_kinds
        assert "click" in step_kinds or "fill" in step_kinds
    
    def test_build_plan_without_context(self, sample_query):
        """Test plan building without context (should use fallback)."""
        plan = build_plan("TASK_EXEC", sample_query, [])
        
        assert plan.task_id is not None
        assert len(plan.steps) > 0
        assert plan.confidence <= 0.7  # Lower confidence for fallback

# Test RAG Integration (requires initialized stores)
class TestRAGIntegration:
    @pytest.fixture(autouse=True)
    def setup_rag(self):
        """Initialize RAG stores before tests."""
        try:
            init_stores()
        except Exception as e:
            pytest.skip(f"RAG stores not initialized: {e}")
    
    def test_retrieve_passages(self):
        """Test retrieval returns passages."""
        query = QueryContext(
            text="how do I bill a purchase order",
            filters={"system": "netsuite"}
        )
        
        passages = retrieve(query, query.filters)
        
        # May be empty if no data ingested, but should not error
        assert isinstance(passages, list)
        for p in passages:
            assert isinstance(p, RetrievedPassage)
            assert p.system == "netsuite"
    
    def test_ask_cited(self):
        """Test Q&A with citations."""
        query = QueryContext(
            text="What is the status for billing a PO?",
            filters={"system": "netsuite"}
        )
        
        answer = ask_cited(query)
        
        assert answer.answer_markdown is not None
        assert isinstance(answer.citations, list)
        assert 0.0 <= answer.confidence <= 1.0
        assert isinstance(answer.next_actions, list)

# Test Answer Generation
class TestAnswerGeneration:
    def test_generate_answer_with_context(self, sample_context):
        """Test LLM answer generation (or fallback)."""
        answer, confidence = generate_answer_with_llm(
            "How do I bill a purchase order?",
            sample_context
        )
        
        assert len(answer) > 50
        assert 0.0 <= confidence <= 1.0
        
        # Should mention key terms from context
        answer_lower = answer.lower()
        assert "bill" in answer_lower or "purchase order" in answer_lower
    
    def test_generate_answer_no_context(self):
        """Test answer generation without context."""
        answer, confidence = generate_answer_with_llm(
            "Random question",
            []
        )
        
        assert answer is not None
        assert confidence < 0.5  # Low confidence

# Integration Test: Full Pipeline
class TestFullPipeline:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for integration tests."""
        try:
            init_stores()
        except:
            pass
    
    def test_complete_workflow(self):
        """Test complete workflow: query -> retrieve -> plan."""
        # Step 1: Create query
        query = QueryContext(
            text="find pending POs and bill the first one",
            filters={"system": "netsuite"}
        )
        
        # Step 2: Retrieve context
        try:
            passages = retrieve(query, query.filters)
        except:
            passages = []  # If no data, use empty
        
        # Step 3: Build plan
        plan = build_plan("TASK_EXEC", query, passages)
        
        # Verify complete pipeline
        assert plan is not None
        assert plan.goal_text == query.text
        assert len(plan.steps) > 0
        
        # Check plan makes sense
        first_step = plan.steps[0]
        assert first_step.kind in ["navigate", "select"]
        
        print(f"\n✓ Generated plan with {len(plan.steps)} steps")
        print(f"✓ Confidence: {plan.confidence}")
        print(f"✓ Steps: {[s.kind for s in plan.steps]}")

# Performance Tests
class TestPerformance:
    def test_plan_generation_speed(self, sample_query, sample_context):
        """Test that plan generation completes in reasonable time."""
        import time
        
        start = time.time()
        plan = build_plan("TASK_EXEC", sample_query, sample_context)
        duration = time.time() - start
        
        # Should complete in under 10 seconds (with LLM) or instantly (fallback)
        assert duration < 10.0
        print(f"\n✓ Plan generated in {duration:.2f}s")

# Smoke Tests
class TestSmokeTests:
    """Quick smoke tests to verify nothing is broken."""
    
    def test_imports(self):
        """Test all imports work."""
        from ai_platform.core.orchestrator import planner
        from ai_platform.systems.netsuite.rag import client
        from ai_platform.core import router
        assert True
    
    def test_env_setup(self):
        """Test environment configuration."""
        # Check if API key is set (optional)
        has_key = bool(os.getenv("OPENAI_API_KEY"))
        if not has_key:
            print("\n⚠ Warning: OPENAI_API_KEY not set. Using fallback plans.")
        assert True  # Not a hard requirement

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])