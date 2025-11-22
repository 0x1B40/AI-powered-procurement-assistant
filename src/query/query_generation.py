"""Query generation, validation, and retry logic."""

import json
import re
from typing import Tuple, List, Dict, Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ..config.constants import MAX_QUERY_ATTEMPTS, RETRY_SNIPPET_LIMIT, SCHEMA_FIELDS
from ..llm.llm import get_llm


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences and trim whitespace."""
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned


def _truncate(text: str, limit: int = RETRY_SNIPPET_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)..."


def _build_retry_instruction(question: str, error_message: str, previous_output: str) -> str:
    snippet = _truncate(previous_output.strip())

    # Provide specific guidance based on common errors
    guidance = ""
    if "sort key specification must be an object" in error_message:
        guidance = "\n\nCommon fix: Change {{'$sort': 'field_name'}} to {{'$sort': {{'field_name': 1}}}} (1 for ascending, -1 for descending)."
    elif "Each pipeline stage must be a JSON object" in error_message:
        guidance = "\n\nCommon fix: Ensure each stage is a JSON object like {{'$match': {{...}}}} or {{'$group': {{...}}}}."
    elif "$sort" in error_message and "object" in error_message:
        guidance = "\n\nFor sorting: Use {{'$sort': {{'field_name': 1}}}} not {{'$sort': 'field_name'}}."
    elif "No results found" in error_message and ("department" in question.lower() or "supplier" in question.lower()):
        guidance = "\n\nCommon fix for department/supplier queries: The system will automatically try fuzzy matching for simple queries. If that fails, use exact string matching with the official name format. Common department names include 'Consumer Affairs, Department of', 'Corrections and Rehabilitation, Department of', etc. Avoid reordering words."

    return (
        "The previous output was not valid JSON or contained invalid MongoDB syntax."
        f"\nError: {error_message}"
        f"\nQuestion: {question}"
        f"\nPrevious output:\n{snippet}"
        f"{guidance}"
        "\n\nRespond with ONLY a valid JSON array of MongoDB aggregation stages. "
        "Do not include markdown, comments, or prose."
    )


def validate_pipeline_text(query_text: str) -> Tuple[bool, str | List[Dict[str, Any]]]:
    try:
        pipeline = json.loads(query_text)
    except json.JSONDecodeError as exc:
        return False, f"Error: Invalid JSON query - {str(exc)}"

    if not isinstance(pipeline, list):
        return False, "Error: Query must be a list of aggregation pipeline stages"
    if not pipeline:
        return False, "Error: Query pipeline is empty"

    for stage in pipeline:
        if not isinstance(stage, dict):
            return False, "Error: Each pipeline stage must be a JSON object"
        stage_str = json.dumps(stage)
        if "$$ROOT" in stage_str:
            return False, "Error: Pipelines must not reference $$ROOT or self-referential projections"

        # Validate $sort syntax - must be an object, not a string
        if "$sort" in stage:
            sort_spec = stage["$sort"]
            if isinstance(sort_spec, str):
                return False, f"Error: $sort specification must be an object, not a string. Use {{'$sort': {{'{sort_spec}': 1}}}} instead of {{'$sort': '{sort_spec}'}}"

    return True, pipeline


def create_mongodb_query_prompt(question: str) -> str:
    """Create a prompt for generating MongoDB queries."""
    return f"""You are a MongoDB aggregation expert working over the California procurement collection.

QUESTION:
{question}

{SCHEMA_FIELDS}

Instructions:
- Always output ONLY a JSON array of aggregation stages. No prose, no markdown.
- Use snake_case field names exactly as provided.
- Prefer $match → $group → $sort → $limit patterns for analytics.
- Convert monetary questions to sums of $total_price; averages use $avg.
- Date filters use $match on creation_date or fiscal_year strings.
- For counts, use $count or $group with $sum: 1.
- Department/supplier names use exact string matching - use the full official name format (e.g., "Consumer Affairs, Department of", not "Department of Consumer Affairs").
- NEVER use $$ROOT, $out, $merge, or stages that return the full document as a field.
- If the question asks for text explanation without data (e.g., greetings), return an empty array [].

Examples:
Top departments by spend:
[{{"$group": {{"_id": "$department_name", "total_spend": {{"$sum": "$total_price"}}}}}}, {{"$sort": {{"total_spend": -1}}}}, {{"$limit": 5}}]
Total purchase orders:
[{{"$group": {{"_id": "$purchase_order_number"}}}}, {{"$count": "purchase_order_count"}}]
Unique fiscal years:
[{{"$group": {{"_id": "$fiscal_year"}}}}, {{"$project": {{"fiscal_year": "$_id", "_id": 0}}}}, {{"$sort": {{"fiscal_year": 1}}}}]
Unique departments:
[{{"$group": {{"_id": "$department_name"}}}}, {{"$project": {{"department": "$_id", "_id": 0}}}}]
Records from Consumer Affairs department:
[{{"$match": {{"department_name": "Consumer Affairs, Department of"}}}}, {{"$count": "record_count"}}]
Average spend for IT Goods in 2014-2015:
[{{"$match": {{"acquisition_type": "IT Goods", "fiscal_year": "2014-2015"}}}}, {{"$group": {{"_id": null, "average_spend": {{"$avg": "$total_price"}}}}}}]
"""


def generate_mongodb_query(question: str) -> Optional[List[Dict[str, Any]]]:
    """Generate and validate a MongoDB query pipeline."""
    llm = get_llm()

    prompt = create_mongodb_query_prompt(question)
    system_message = SystemMessage(content=prompt)

    pipeline = None
    error_msg = ""
    previous_output = ""

    for attempt in range(MAX_QUERY_ATTEMPTS):
        messages = [system_message]
        attempt_inputs = {
            "question": question,
            "system_prompt": prompt,
            "attempt": attempt + 1,
        }

        if attempt > 0:
            retry_instruction = _build_retry_instruction(question, error_msg, previous_output)
            attempt_inputs["retry_instruction"] = retry_instruction
            if previous_output:
                attempt_inputs["previous_output"] = _truncate(previous_output)
            messages.append(HumanMessage(content=retry_instruction))

        response = llm.invoke(messages)
        query_text = _strip_code_fences(response.content)
        previous_output = query_text

        is_valid, pipeline_or_error = validate_pipeline_text(query_text)

        if is_valid:
            pipeline = pipeline_or_error
            break

        error_msg = pipeline_or_error if isinstance(pipeline_or_error, str) else "Invalid query format."

    return pipeline
