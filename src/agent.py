from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Dict, Any, List, TypedDict, Annotated, Tuple, TYPE_CHECKING
from datetime import datetime

from langchain_core.tools import tool, BaseTool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import OperationFailure

from .config import get_settings
from .telemetry import traceable_step, log_child_run

# Importing ChatOpenAI eagerly causes SSL context creation, which can fail inside
# restricted CI sandboxes. We import it lazily in _get_llm(), but still want type
# checkers to know about the symbol—hence the TYPE_CHECKING guard.
if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


class AgentState(TypedDict):
    """State for the LangGraph agent."""
    messages: Annotated[List[BaseMessage], add_messages]
    mongodb_results: List[Dict[str, Any]]
    final_answer: str


@lru_cache
def _get_collection() -> Collection:
    """Get MongoDB collection with caching."""
    settings = get_settings()
    client = MongoClient(settings.mongodb_uri)
    database = client[settings.mongodb_db]
    return database[settings.mongodb_collection]


@lru_cache
def _get_llm():
    """Get configured LLM with caching."""
    from langchain_openai import ChatOpenAI  # Local import avoids SSL context issues during module import in restricted environments
    settings = get_settings()
    kwargs = {
        "api_key": settings.llm_api_key,
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
    }
    if settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url
    return ChatOpenAI(**kwargs)


SCHEMA_FIELDS = """
Important fields (MongoDB collection `purchase_orders`):
- purchase_order_number (string) — unique order id; multiple rows per PO possible
- department_name (string) — ordering agency name
- supplier_name (string), supplier_code (string) — vendor metadata
- acquisition_type / acquisition_method / sub_acquisition_method (string) — procurement classifications
- item_name, item_description (string) — commodity information
- quantity (number), unit_price (number), total_price (number)
- creation_date, purchase_date (string in MM/DD/YYYY) and fiscal_year (string)
  * creation_date/purchase_date are stored as literal strings; use $dateFromString with format "%m/%d/%Y" before applying $gte/$lte or quarter logic.
  * fiscal_year values are labels such as "2012-2013", "2013-2014", "2014-2015" (there is no bare "2013"). When the user mentions a calendar year (e.g., 2013) either filter creation_date between 01/01/YYYY–12/31/YYYY or match the fiscal_years that include that year.
- segment_title, family_title, class_title, commodity_title, normalized_unspsc (string/number) — UNSPSC hierarchy
- calcard (string 'YES'|'NO'), location (string), lpa_number (string), requisition_number (string)

Guidelines:
- Only reference existing fields above (snake_case).
- Never project the entire document (`$$ROOT`) or include recursive structures.
- Keep pipelines minimal: use $match/$group/$sort/$limit/$project as needed.
- Always return JSON array of aggregation stages, even for simple queries.
"""

SMALLTALK_PATTERNS = re.compile(
    r"\b(hi|hello|hey|thanks?|thank you|good morning|good afternoon|good evening|how are you)\b",
    re.IGNORECASE,
)


def is_smalltalk(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    return bool(SMALLTALK_PATTERNS.search(stripped)) and len(stripped.split()) <= 6


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

    return True, pipeline


MAX_QUERY_ATTEMPTS = 3
RETRY_SNIPPET_LIMIT = 800


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
    return (
        "The previous output was not valid JSON."
        f"\nError: {error_message}"
        f"\nQuestion: {question}"
        f"\nPrevious output:\n{snippet}"
        "\n\nRespond with ONLY a valid JSON array of MongoDB aggregation stages. "
        "Do not include markdown, comments, or prose."
    )


@traceable_step(name="execute_mongodb_query", run_type="tool", tags=["mongodb"])
@tool
def execute_mongodb_query(query: str) -> str:
    """
    Execute a MongoDB aggregation pipeline query.

    Args:
        query: A valid MongoDB aggregation pipeline as JSON string

    Returns:
        Query results as formatted JSON string
    """
    try:
        # Parse the query
        pipeline = json.loads(query)
        if not isinstance(pipeline, list):
            return "Error: Query must be a list of aggregation pipeline stages"

        collection = _get_collection()

        # Execute the aggregation pipeline
        results = list(collection.aggregate(pipeline, allowDiskUse=True))

        # Format results
        if not results:
            return "No results found for this query."

        # Convert ObjectId and datetime objects to strings for JSON serialization
        def serialize_result(obj):
            if hasattr(obj, '__dict__'):
                return str(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return obj

        # Serialize results
        serialized_results = []
        for result in results:
            serialized_result = {}
            for key, value in result.items():
                try:
                    # Try to serialize, if it fails, convert to string
                    json.dumps({key: value})
                    serialized_result[key] = value
                except (TypeError, ValueError):
                    serialized_result[key] = serialize_result(value)
            serialized_results.append(serialized_result)

        return json.dumps(serialized_results, indent=2, default=serialize_result)

    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON query - {str(e)}"
    except OperationFailure as e:
        return f"Error: MongoDB operation failed - {str(e)}"
    except Exception as e:
        return f"Error: Failed to execute query - {str(e)}"


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
- NEVER use $$ROOT, $out, $merge, or stages that return the full document as a field.
- If the question asks for text explanation without data (e.g., greetings), return an empty array [].

Examples:
Top departments by spend:
[{{"$group": {{"_id": "$department_name", "total_spend": {{"$sum": "$total_price"}}}}}}, {{"$sort": {{"total_spend": -1}}}}, {{"$limit": 5}}]
Total purchase orders:
[{{"$count": "purchase_order_count"}}]
Average spend for IT Goods in 2014-2015:
[{{"$match": {{"acquisition_type": "IT Goods", "fiscal_year": "2014-2015"}}}}, {{"$group": {{"_id": null, "average_spend": {{"$avg": "$total_price"}}}}}}]
"""


@traceable_step(name="analyze_question", tags=["query-generation"])
def analyze_question(state: AgentState) -> Dict[str, Any]:
    """Analyze the question and generate MongoDB query."""
    messages = state["messages"]
    question = messages[-1].content if messages else ""

    llm = _get_llm()

    prompt = create_mongodb_query_prompt(question)
    system_message = SystemMessage(content=prompt)

    pipeline = None
    error_msg = ""
    previous_output = ""

    for attempt in range(MAX_QUERY_ATTEMPTS):
        messages: List[BaseMessage] = [system_message]
        attempt_inputs: Dict[str, Any] = {
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
        attempt_metadata = {"attempt": attempt + 1, "max_attempts": MAX_QUERY_ATTEMPTS}

        if is_valid:
            pipeline = pipeline_or_error
            log_child_run(
                name="pipeline_generation_attempt",
                inputs=attempt_inputs,
                outputs={"raw_response": query_text, "pipeline": pipeline},
                metadata=attempt_metadata,
                tags=["pipeline-attempt"],
            )
            break

        error_msg = pipeline_or_error if isinstance(pipeline_or_error, str) else "Invalid query format."
        log_child_run(
            name="pipeline_generation_attempt",
            inputs=attempt_inputs,
            outputs={"raw_response": query_text},
            metadata=attempt_metadata,
            tags=["pipeline-attempt"],
            error=error_msg,
        )

    if pipeline is None:
        final_error = error_msg or "Failed to generate a valid MongoDB pipeline."
        return {
            "mongodb_results": [{"error": final_error}],
            "final_answer": final_error,
        }

    tool_result = execute_mongodb_query(json.dumps(pipeline))

    return {
        "mongodb_results": [json.loads(tool_result)] if tool_result.startswith('[') else [{"error": tool_result}],
        "final_answer": tool_result
    }


@traceable_step(name="format_response", tags=["answer-rendering"])
def format_response(state: AgentState) -> Dict[str, Any]:
    """Format the final response for the user."""
    llm = _get_llm()

    question = state["messages"][-1].content if state["messages"] else ""
    results = state.get("mongodb_results", [])

    if not results or (len(results) == 1 and "error" in results[0]):
        error_msg = results[0].get("error", "No results found") if results else "No results found"
        return {"final_answer": f"I couldn't find the information you're looking for. {error_msg}"}

    # Format the results nicely
    formatted_prompt = f"""Format these MongoDB query results into a clear, natural response.

QUESTION: {question}

RESULTS: {json.dumps(results, indent=2)}

Provide a concise, readable answer that directly addresses the question."""

    response = llm.invoke([SystemMessage(content=formatted_prompt)])
    return {"final_answer": response.content}


# Create the LangGraph workflow
def create_procurement_agent():
    """Create the LangGraph agent for procurement queries."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze_question", analyze_question)
    workflow.add_node("format_response", format_response)

    # Define flow
    workflow.set_entry_point("analyze_question")
    workflow.add_edge("analyze_question", "format_response")
    workflow.add_edge("format_response", END)

    return workflow.compile()


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
        if is_smalltalk(question):
            return (
                "Hello! I'm the California procurement assistant. "
                "Ask me about spending, suppliers, departments, dates, or other procurement metrics."
            )
        agent = get_procurement_agent()

        # Prepare initial state
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "mongodb_results": [],
            "final_answer": ""
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

