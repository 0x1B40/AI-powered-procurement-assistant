"""Question classification and categorization logic."""

import json
import math
import re
from typing import Optional, Tuple, Any

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
    normalized = (text or "").strip()
    if not normalized:
        return QuestionCategory.OUT_OF_SCOPE, None

    prompt = CLASSIFIER_PROMPT.format(question=normalized)
    llm = llm or get_llm()

    try:
        response = llm.invoke([SystemMessage(content=prompt)])
    except Exception:
        return QuestionCategory.OUT_OF_SCOPE, None

    category, confidence = _parse_question_category(response.content)
    return (category or QuestionCategory.OUT_OF_SCOPE, confidence)
