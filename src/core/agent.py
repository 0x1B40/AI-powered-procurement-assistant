"""Main procurement agent interface."""

from typing import Dict

from langchain_core.messages import HumanMessage

from ..config_and_constants.constants import QuestionCategory
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
    """Generate a MongoDB-grounded answer for a procurement question """
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
        return f"I encountered an error while processing your question: {str(e)}. Please check your LLM API key and MongoDB connection."

