from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Dict, Any, List, TypedDict, Annotated, Tuple
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import OperationFailure

from .config import get_settings


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
    settings = get_settings()
    return ChatOpenAI(
        api_key=settings.grok_api_key,
        model=settings.grok_model,
        temperature=settings.grok_temperature,
        base_url="https://api.x.ai/v1",
    )


SCHEMA_FIELDS = """
Important fields (MongoDB collection `purchase_orders`):
- purchase_order_number (string) — unique order id; multiple rows per PO possible
- department_name (string) — ordering agency name
- supplier_name (string), supplier_code (string) — vendor metadata
- acquisition_type / acquisition_method / sub_acquisition_method (string) — procurement classifications
- item_name, item_description (string) — commodity information
- quantity (number), unit_price (number), total_price (number)
- creation_date, purchase_date (string in MM/DD/YYYY) and fiscal_year (string)
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


def analyze_question(state: AgentState) -> Dict[str, Any]:
    """Analyze the question and generate MongoDB query."""
    messages = state["messages"]
    question = messages[-1].content if messages else ""

    llm = _get_llm()

    # Create prompt for query generation
    prompt = create_mongodb_query_prompt(question)

    # Generate MongoDB query
    response = llm.invoke([SystemMessage(content=prompt)])

    # Extract the query (should be just JSON)
    query_text = response.content.strip()

    # Clean up the response (remove markdown if present)
    query_text = re.sub(r'```(?:json)?\s*', '', query_text)
    query_text = re.sub(r'```\s*$', '', query_text)

    is_valid, pipeline_or_error = validate_pipeline_text(query_text)
    if not is_valid:
        error_msg = pipeline_or_error
        return {
            "mongodb_results": [{"error": error_msg}],
            "final_answer": error_msg,
        }

    # Execute the query
    tool_result = execute_mongodb_query(json.dumps(pipeline_or_error))

    return {
        "mongodb_results": [json.loads(tool_result)] if tool_result.startswith('[') else [{"error": tool_result}],
        "final_answer": tool_result
    }


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

