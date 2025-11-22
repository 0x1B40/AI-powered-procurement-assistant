"""Main procurement agent interface."""

from typing import Dict

from langchain_core.messages import HumanMessage

from ..config.constants import QuestionCategory
from ..utils.telemetry import traceable_step
from .workflow import create_procurement_agent


# Global agent instance
_procurement_agent = None


def get_procurement_agent():
    """Get or create the procurement agent."""
    global _procurement_agent
    if _procurement_agent is None:
        _procurement_agent = create_procurement_agent()
    return _procurement_agent


@traceable_step(name="procurement_chat", tags=["chat-entrypoint"])
def chat(question: str, context: Dict | None = None) -> str:
    """Generate a MongoDB-grounded answer for a procurement question using LangChain and LangGraph."""
    try:
        agent = get_procurement_agent()

        # Prepare initial state
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "mongodb_results": [],
            "reference_context": [],
            "final_answer": "",
            "question_category": QuestionCategory.OUT_OF_SCOPE.value,
            "classification_confidence": None,
        }

        # Run the agent
        result = agent.invoke(initial_state)

        return result.get("final_answer", "I couldn't process your question. Please try again.")

    except Exception as e:
        # Fallback to error message
        return f"I encountered an error while processing your question: {str(e)}. Please check your Grok API key and MongoDB connection."


# Keep the mock function for fallback if needed
def _get_mock_response(question: str) -> str:
    """Return mock responses for demo purposes when LLM is not available."""
    question_lower = question.lower()

    if "how many" in question_lower and "purchase order" in question_lower:
        return "Query result: [{\"count\": 1250}]"
    elif "highest" in question_lower and "spend" in question_lower:
        return "Query result: [{\"_id\": \"Q4 2014\", \"total_spend\": 45250000.75}]"
    elif "top 5" in question_lower and "frequently ordered" in question_lower:
        return "Query result: [{\"item\": \"Office Supplies\", \"count\": 450}, {\"item\": \"Computer Equipment\", \"count\": 380}, {\"item\": \"Maintenance Services\", \"count\": 295}, {\"item\": \"Software Licenses\", \"count\": 275}, {\"item\": \"Training Materials\", \"count\": 210}]"
    elif "average" in question_lower and "order amount" in question_lower:
        return "Query result: [{\"average_amount\": 12500.50}]"
    else:
        return "Query result: [{\"message\": \"This is a demo response. In production, this would query the actual MongoDB collection.\"}]"
