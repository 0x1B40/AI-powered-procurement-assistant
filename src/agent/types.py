"""Type definitions for the procurement agent."""

from typing import Dict, Any, List, TypedDict, Optional

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """State for the LangGraph agent."""
    messages: List[BaseMessage]
    mongodb_results: List[Dict[str, Any]]
    reference_context: List[Dict[str, Any]]
    final_answer: str
    question_category: str
    classification_confidence: Optional[float]
    relevant_conversation_history: List[Dict[str, Any]]
