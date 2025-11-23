"""Response formatting and processing logic."""

import json
from typing import Dict, Any, List

from langchain_core.messages import SystemMessage

from ..config_and_constants.constants import REFERENCE_CATEGORIES, QuestionCategory
from ..llm_and_classification.llm import get_llm


def format_response(question: str, reference_context: List[Dict[str, Any]],
                   results: List[Dict[str, Any]], category: str) -> str:
    """Format the final response for the user."""
    llm = get_llm()

    if reference_context:
        context_prompt = f"""You are a procurement SME. Use the provided reference passages to answer the user's question.

CONTEXT:
{json.dumps(reference_context, indent=2)}

QUESTION:
{question}

Guidelines:
- Cite the document name (and page when available) when referencing facts.
- If the context does not contain the answer, state that explicitly instead of guessing.
"""
        response = llm.invoke([SystemMessage(content=context_prompt)])
        return response.content

    if category in REFERENCE_CATEGORIES:
        return ""  # Handled by reference context above

    if not results or (len(results) == 1 and "error" in results[0]):
        error_msg = results[0].get("error", "No results found") if results else "No results found"
        return f"I couldn't find the information you're looking for. {error_msg}"

    # Format the results nicely
    formatted_prompt = f"""Format these MongoDB query results into a clear, natural response.

QUESTION: {question}

RESULTS: {json.dumps(results, indent=2)}

Provide a concise, readable answer that directly addresses the question."""

    response = llm.invoke([SystemMessage(content=formatted_prompt)])
    return response.content
