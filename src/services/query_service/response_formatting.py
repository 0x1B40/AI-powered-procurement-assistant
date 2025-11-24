"""Response formatting and processing logic."""

import json
from typing import Dict, Any, List

from langchain_core.messages import SystemMessage

from ...config.constants import REFERENCE_CATEGORIES, QuestionCategory
from ..llm_service.llm import get_llm


def format_response(question: str, reference_context: List[Dict[str, Any]],
                   results: List[Dict[str, Any]], category: str) -> str:
    """Format the final response for the user."""

    # Check if this is a compound query result
    if len(results) > 1 and all(isinstance(r, dict) and "sub_query" in r for r in results):
        # This is a compound query result - combine them
        return format_compound_response(question, results)

    # Single query - use existing logic
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


def format_compound_response(question: str, sub_results: List[Dict[str, Any]]) -> str:
    """Format results from multiple sub-queries into a cohesive response."""
    # Extract successful and failed results
    successful_results = [r for r in sub_results if "error" not in r.get("result", {})]
    failed_results = [r for r in sub_results if "error" in r.get("result", {})]

    if not successful_results:
        error_details = "\n".join([f"- {r['sub_query']}: {r['result']['error']}" for r in failed_results[:3]])
        return f"I couldn't process any part of your compound query. Errors encountered:\n{error_details}"

    # Build combined response using LLM for natural language formatting
    llm = get_llm()

    # Create a summary of results for the LLM
    result_summary = []
    for result in successful_results:
        sub_query = result["sub_query"]
        data = result["result"]

        # Extract the key numeric result (this could be more sophisticated)
        if isinstance(data, list) and data:
            # Try to find count or total in the result
            result_doc = data[0] if isinstance(data[0], dict) else {}
            count = result_doc.get("count") or result_doc.get("total") or result_doc.get("q1_2014_count") or result_doc.get("q4_2013_count")
            if count is not None:
                result_summary.append(f"{sub_query}: {count}")
            else:
                result_summary.append(f"{sub_query}: {json.dumps(result_doc, indent=2)}")
        else:
            result_summary.append(f"{sub_query}: {json.dumps(data, indent=2)}")

    result_text = "\n".join(result_summary)

    prompt = f"""
    The user asked: "{question}"

    I executed multiple sub-queries and got these results:
    {result_text}

    Please provide a natural language response that:
    1. Shows each result separately as requested
    2. Calculates and shows the total if the question asks to add them up
    3. Uses clear, readable formatting
    4. Explains any errors if they occurred

    Format the response naturally, like a helpful assistant would.
    """

    response = llm.invoke([SystemMessage(content=prompt)])
    return response.content
