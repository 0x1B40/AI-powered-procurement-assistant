"""Question classification and categorization logic."""

import json
import math
import re
from typing import Optional, Tuple, Any, List

from langchain_core.messages import HumanMessage, SystemMessage

from ..config_and_constants.constants import QuestionCategory, CLASSIFIER_PROMPT
from .llm import get_llm


def _coerce_confidence(value: Any) -> Optional[float]:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(confidence):
        return None
    return max(0.0, min(1.0, confidence))


def _parse_question_category(response_text: str) -> Tuple[Optional[QuestionCategory], Optional[float]]:
    cleaned = _strip_code_fences(response_text)
    try:
        payload = json.loads(cleaned)
        raw_value = payload.get("category", "")
        confidence = _coerce_confidence(payload.get("confidence"))
    except json.JSONDecodeError:
        raw_value = cleaned
        confidence = None

    normalized = (raw_value or "").strip().lower()
    for category in QuestionCategory:
        if normalized == category.value:
            return category, confidence
    return None, confidence


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences and trim whitespace."""
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned


def categorize_question(text: str, llm: Optional[Any] = None) -> Tuple[QuestionCategory, Optional[float]]:
    """
    Categorize questions using chain of thought reasoning with few-shot examples.

    This function uses structured reasoning to classify procurement questions into
    appropriate categories for routing to the correct processing pipeline.
    """
    normalized = (text or "").strip()
    if not normalized:
        return QuestionCategory.OUT_OF_SCOPE, None

    prompt = CLASSIFIER_PROMPT.format(question=normalized)
    llm = llm or get_llm()

    try:
        # Use chain of thought reasoning with few-shot examples
        response = llm.invoke([SystemMessage(content=prompt)])
    except Exception:
        return QuestionCategory.OUT_OF_SCOPE, None

    category, confidence = _parse_question_category(response.content)
    return (category or QuestionCategory.OUT_OF_SCOPE, confidence)


def detect_compound_queries(question: str) -> List[str]:
    """Use LLM to detect and split compound queries into separate sub-queries."""
    prompt = f"""
    Analyze this query and determine if it contains multiple separate data requests that should be handled independently.

    If it's a compound query (contains multiple distinct requests), split it into separate, independent sub-queries that can be executed individually. Return a JSON array of strings.

    If it's a single query, return the original query as a single-item array.

    Rules for compound queries:
    - Each sub-query should be self-contained and answerable independently
    - Preserve the original intent and specificity of each part
    - Remove any aggregation instructions (like "then add them up", "separately", "combined") from individual queries
    - Focus on the core data request for each part
    - Only split if there are truly separate, independent requests

    Examples:

    Compound query example:
    Input: "Show me total orders in Q1 2023 and Q2 2023 separately then add them up"
    Output: ["How many orders were created in the first quarter of 2023?", "How many orders were created in the second quarter of 2023?"]

    Compound query example:
    Input: "give me The total number of orders created during the first quarter of 2014, and the last quarter of 2013 show them seperately, then add them up"
    Output: ["How many orders were created during the first quarter of 2014?", "How many orders were created during the last quarter of 2013?"]

    Single query example:
    Input: "How many purchase orders are there total?"
    Output: ["How many purchase orders are there total?"]

    Single query example:
    Input: "What is the average order value?"
    Output: ["What is the average order value?"]

    Query: {question}
    """

    llm = get_llm()
    response = llm.invoke([SystemMessage(content=prompt)])
    cleaned = _strip_code_fences(response.content)

    try:
        sub_queries = json.loads(cleaned)
        if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries) and len(sub_queries) > 0:
            return sub_queries
        else:
            # Fallback: return original question if parsing fails
            return [question]
    except json.JSONDecodeError:
        # Fallback: return original question if LLM doesn't return valid JSON
        return [question]
