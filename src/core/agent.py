"""Main procurement agent interface."""

from typing import Dict, List, Optional, Union, Tuple

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

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
def chat(question: str, conversation_history: Optional[List[BaseMessage]] = None) -> Tuple[str, List[BaseMessage]]:
    """Generate a MongoDB-grounded answer for a procurement question

    Args:
        question: The user's question
        conversation_history: Previous messages in the conversation (optional)

    Returns:
        tuple of (response, updated_conversation_history)
    """
    try:
        agent = get_procurement_agent()

        # Initialize or use existing conversation history
        if conversation_history is None:
            conversation_history = []

        # Add the current question to history (avoid duplicates)
        current_messages = conversation_history[:]
        if not current_messages or current_messages[-1].content != question:
            current_messages.append(HumanMessage(content=question))

        # Prepare initial state with full conversation history
        initial_state = {
            "messages": current_messages,
            "mongodb_results": [],
            "reference_context": [],
            "final_answer": "",
            "question_category": QuestionCategory.OUT_OF_SCOPE.value,
            "classification_confidence": None,
            "relevant_conversation_history": [],
        }

        # Run the agent
        result = agent.invoke(initial_state)
        response = result.get("final_answer", "I couldn't process your question. Please try again.")

        # Add the response to conversation history
        updated_history = current_messages + [AIMessage(content=response)]

        return response, updated_history

    except Exception as e:
        # Fallback to error message
        error_response = f"I encountered an error while processing your question: {str(e)}. Please check your LLM API key and MongoDB connection."

        # Still update history with the error
        if conversation_history is None:
            conversation_history = []
        updated_history = conversation_history + [HumanMessage(content=question), AIMessage(content=error_response)]

        return error_response, updated_history

