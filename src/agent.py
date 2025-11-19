from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Dict, Any, List, TypedDict, Annotated
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
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=settings.openai_temperature,
    )


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
    return f"""You are a MongoDB expert. Convert this natural language question into a MongoDB aggregation pipeline.

QUESTION: {question}

CONTEXT: You are working with a California procurement dataset. The collection contains purchase order data with these relevant fields:
- purchase_order: PO number (string)
- total_amount: Purchase amount (number)
- creation_date: When order was created (date)
- vendor_name: Supplier name (string)
- department_name: Government department (string)
- item_description: What was purchased (string)
- acquisition_type: Procurement method (string)

COMMON QUERY PATTERNS:
- Count documents: Use $count stage
- Sum amounts: Use $group with $sum
- Average values: Use $group with $avg
- Filter by date: Use $match with date comparisons
- Top N items: Use $group, $sort, $limit
- Group by categories: Use $group with _id

Return ONLY a valid MongoDB aggregation pipeline as JSON array. No explanations, no markdown, just the JSON.

Examples:
Question: "How many purchase orders were created?"
Answer: [{{"$count": "total_orders"}}]

Question: "What is the total spending?"
Answer: [{{"$group": {{"_id": null, "total_spend": {{"$sum": "$total_amount"}}}}}}]

Question: "Top 5 vendors by spending?"
Answer: [{{"$group": {{"_id": "$vendor_name", "total_spend": {{"$sum": "$total_amount"}}}}}}, {{"$sort": {{"total_spend": -1}}}}, {{"$limit": 5}}]
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

    # Execute the query
    tool_result = execute_mongodb_query(query_text)

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
        return f"I encountered an error while processing your question: {str(e)}. Please check your OpenAI API key and MongoDB connection."


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

