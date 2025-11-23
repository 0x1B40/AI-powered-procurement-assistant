"""LangGraph workflow definition for the procurement agent."""

import json
from typing import Dict, Any

from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from ..llm_and_classification.classification import categorize_question
from ..config_and_constants.constants import REFERENCE_CATEGORIES, QuestionCategory, OUT_OF_SCOPE_RESPONSE
from ..database.database import execute_mongodb_query
from ..query_and_response.query_generation import generate_mongodb_query
from ..query_and_response.response_formatting import format_response
from ..utils.telemetry import traceable_step, log_child_run
from .types import AgentState
from ..utils.vector_store import retrieve_reference_chunks


@traceable_step(name="execute_mongodb_query", run_type="tool", tags=["mongodb"])
@tool
def execute_mongodb_query_tool(query: str) -> str:
    """
    Execute a MongoDB aggregation pipeline query.

    Args:
        query: A valid MongoDB aggregation pipeline as JSON string

    Returns:
        Query results as formatted JSON string
    """
    return execute_mongodb_query(query)


@traceable_step(name="analyze_question", tags=["query-generation"])
def analyze_question(state: AgentState) -> Dict[str, Any]:
    """Analyze the question and generate MongoDB query."""
    messages = state["messages"]
    question = messages[-1].content if messages else ""
    category = state.get("question_category", QuestionCategory.OUT_OF_SCOPE.value)

    if category in REFERENCE_CATEGORIES:
        references = retrieve_reference_chunks(question)
        if references:
            log_child_run(
                name="reference_retrieval",
                inputs={"question": question},
                outputs={"reference_count": len(references)},
                tags=["reference-search"],
            )
            return {
                "mongodb_results": [],
                "reference_context": references,
                "final_answer": "",
            }

        fallback_message = (
            "I could not access the procurement reference documents. "
            "Run `python -m scripts.build_reference_store` to build the Chroma index and try again."
        )
        log_child_run(
            name="reference_retrieval",
            inputs={"question": question},
            outputs={"reference_count": 0},
            tags=["reference-search"],
            error=fallback_message,
        )
        return {
            "mongodb_results": [],
            "reference_context": [],
            "final_answer": fallback_message,
        }

    pipeline = generate_mongodb_query(question)

    if pipeline is None:
        final_error = "Failed to generate a valid MongoDB pipeline."
        return {
            "mongodb_results": [{"error": final_error}],
            "final_answer": final_error,
        }

    tool_result = execute_mongodb_query_tool(json.dumps(pipeline))

    return {
        "mongodb_results": [json.loads(tool_result)] if tool_result.startswith('[') else [{"error": tool_result}],
        "final_answer": tool_result
    }


@traceable_step(name="format_response", tags=["answer-rendering"])
def format_response_node(state: AgentState) -> Dict[str, Any]:
    """Format the final response for the user."""
    question = state["messages"][-1].content if state["messages"] else ""
    reference_context = state.get("reference_context") or []
    results = state.get("mongodb_results", [])
    category = state.get("question_category")

    final_answer = format_response(question, reference_context, results, category)
    return {"final_answer": final_answer}


@traceable_step(name="classify_question", tags=["routing"])
def classify_question_node(state: AgentState) -> Dict[str, Any]:
    """Categorize the incoming user question before running expensive steps."""
    messages = state.get("messages", [])
    question = messages[-1].content if messages else ""
    category, confidence = categorize_question(question)
    log_child_run(
        name="question_classifier",
        inputs={"question": question},
        outputs={"category": category.value, "confidence": confidence},
        tags=["classifier"],
    )
    return {
        "question_category": category.value,
        "classification_confidence": confidence,
    }


@traceable_step(name="handle_out_of_scope", tags=["routing"])
def handle_out_of_scope(state: AgentState) -> Dict[str, Any]:
    """Return a formal response whenever the prompt is out of scope."""
    return {
        "mongodb_results": [],
        "reference_context": [],
        "final_answer": OUT_OF_SCOPE_RESPONSE,
        "question_category": state.get("question_category", QuestionCategory.OUT_OF_SCOPE.value),
        "classification_confidence": state.get("classification_confidence"),
    }


# Create the LangGraph workflow
def create_procurement_agent():
    """Create the LangGraph agent for procurement queries."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("classify_question", classify_question_node)
    workflow.add_node("analyze_question", analyze_question)
    workflow.add_node("format_response", format_response_node)
    workflow.add_node("handle_out_of_scope", handle_out_of_scope)

    # Define flow
    workflow.set_entry_point("classify_question")
    workflow.add_conditional_edges(
        "classify_question",
        lambda state: state.get("question_category", QuestionCategory.OUT_OF_SCOPE.value),
        {
            QuestionCategory.QUERY_GENERATION.value: "analyze_question",
            QuestionCategory.DATABASE_INFO.value: "analyze_question",
            QuestionCategory.ACQUISITION_METHODS.value: "analyze_question",
            QuestionCategory.OUT_OF_SCOPE.value: "handle_out_of_scope",
        },
    )
    workflow.add_edge("analyze_question", "format_response")
    workflow.add_edge("format_response", END)
    workflow.add_edge("handle_out_of_scope", END)

    return workflow.compile()
