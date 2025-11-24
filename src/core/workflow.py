"""Redesigned LangGraph workflow definition for the procurement agent."""

import json
from typing import Dict, Any, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..llm_and_classification.classification import categorize_question, detect_compound_queries
from ..llm_and_classification.llm import get_llm
from ..config_and_constants.constants import REFERENCE_CATEGORIES, QuestionCategory, OUT_OF_SCOPE_RESPONSE
from ..database.database import execute_mongodb_query
from ..query_and_response.query_generation import generate_mongodb_query
from ..query_and_response.response_formatting import format_response
from ..utils.telemetry import traceable_step, log_child_run
from .types import AgentState
from ..utils.vector_store import retrieve_reference_chunks


# ===== NEW ARCHITECTURE NODES =====

@traceable_step(name="input_processor", tags=["input"])
def input_processor(state: AgentState) -> Dict[str, Any]:
    """Process user input and initialize state for the new architecture."""
    messages = state["messages"]
    current_question = messages[-1].content if messages else ""

    # Initialize new state fields
    initial_state = {
        "schema": None,
        "sub_queries": [],
        "results": [],
        "current_agent": "supervisor",
        "guardrails_status": "pending",
        "question_category": state.get("question_category", "unknown"),
        "classification_confidence": state.get("classification_confidence"),
        "relevant_conversation_history": state.get("relevant_conversation_history", []),
        "mongodb_results": [],  # For compatibility
        "reference_context": [],
        "final_answer": "",
    }

    log_child_run(
        name="input_processing",
        inputs={"question": current_question[:100]},
        outputs={"initialized_fields": list(initial_state.keys())},
        tags=["input"]
    )

    return initial_state


@traceable_step(name="supervisor", tags=["coordination"])
def supervisor_agent(state: AgentState) -> Dict[str, Any]:
    """Supervisor agent that routes to appropriate specialized agents based on workflow state."""
    question = state["messages"][-1].content if state["messages"] else ""
    category = state.get("question_category", "unknown")
    current_agent = state.get("current_agent", "query_understanding")
    schema = state.get("schema")
    sub_queries = state.get("sub_queries", [])
    results = state.get("results", [])
    guardrails_status = state.get("guardrails_status")

    # Determine next step based on current workflow state
    if current_agent == "supervisor" or current_agent == "input_processor":
        # Initial routing - analyze the question
        next_agent = "query_understanding"

    elif current_agent == "query_understanding":
        # After understanding, check if we need schema
        if not schema:
            next_agent = "fetch_schema"
        else:
            next_agent = "planner"

    elif current_agent == "fetch_schema":
        # After getting schema, plan the query
        next_agent = "planner"

    elif current_agent == "planner":
        # After planning, build and execute queries
        if sub_queries:
            next_agent = "aggregation_builder"
        else:
            # Simple query, go straight to building
            next_agent = "aggregation_builder"

    elif current_agent == "aggregation_builder":
        # After building queries, check if we have results to merge
        if results:
            next_agent = "merge"
        else:
            # No results, might need to retry or go to guardrails
            next_agent = "guardrails"

    elif current_agent == "merge":
        # After merging, do safety checks
        next_agent = "guardrails"

    elif current_agent == "guardrails":
        # After safety checks, generate final response
        next_agent = "respond"

    else:
        # Default fallback
        next_agent = "respond"

    log_child_run(
        name="supervisor_routing",
        inputs={
            "question": question[:100],
            "current_agent": current_agent,
            "has_schema": bool(schema),
            "sub_queries_count": len(sub_queries),
            "results_count": len(results),
            "guardrails_status": guardrails_status
        },
        outputs={"next_agent": next_agent},
        tags=["routing"]
    )

    # If we're here because an agent just completed, determine next step
    if current_agent in ["query_understanding", "fetch_schema", "planner", "aggregation_builder", "merge", "guardrails"]:
        # An agent just completed, determine what to do next
        if current_agent == "query_understanding":
            next_agent = "fetch_schema" if not schema else "planner"
        elif current_agent == "fetch_schema":
            next_agent = "planner"
        elif current_agent == "planner":
            next_agent = "aggregation_builder"
        elif current_agent == "aggregation_builder":
            next_agent = "merge" if results else "guardrails"
        elif current_agent == "merge":
            next_agent = "guardrails"
        elif current_agent == "guardrails":
            next_agent = "respond"
        else:
            next_agent = "respond"

        return {"current_agent": next_agent}

    # Default case
    return {"current_agent": "query_understanding"}


@traceable_step(name="query_understanding", tags=["analysis"])
def query_understanding_agent(state: AgentState) -> Dict[str, Any]:
    """Analyze and understand the user's query intent."""
    question = state["messages"][-1].content if state["messages"] else ""
    history = state.get("relevant_conversation_history", [])

    llm = get_llm()

    analysis_prompt = f"""
    Analyze this procurement query and provide structured understanding:

    Question: {question}

    Conversation History:
    {chr(10).join([f"Q: {h['request'][:100]} A: {h['response'][:200]}" for h in history[-2:]]) if history else "None"}

    Provide analysis in this format:
    INTENT: [brief description of user intent]
    COMPLEXITY: [simple/medium/complex]
    DOMAIN: [procurement/financial/general]
    NEEDS_SCHEMA: [yes/no]
    NEEDS_DECOMPOSITION: [yes/no]
    SAFETY_CONCERNS: [none/low/medium/high]
    """

    try:
        response = llm.invoke([SystemMessage(content=analysis_prompt)])
        analysis = response.content.strip()

        log_child_run(
            name="query_analysis",
            inputs={"question": question[:100]},
            outputs={"analysis": analysis[:200]},
            tags=["analysis"]
        )

        # Analysis complete, let supervisor determine next step
        return {"current_agent": "query_understanding"}

    except Exception as e:
        log_child_run(
            name="query_analysis_error",
            inputs={"question": question[:100]},
            outputs={"error": str(e)},
            tags=["analysis-error"]
        )
        return {}


@traceable_step(name="fetch_schema", run_type="tool", tags=["schema"])
def fetch_schema_tool(state: AgentState) -> Dict[str, Any]:
    """Fetch MongoDB collection schema information."""
    # This would integrate with database to get schema
    # For now, return a placeholder schema structure
    schema = {
        "collection": "purchase_orders",
        "fields": {
            "po_number": "string",
            "vendor_name": "string",
            "total_amount": "number",
            "order_date": "date",
            "department": "string",
            "status": "string"
        },
        "indexes": ["po_number", "vendor_name", "order_date"]
    }

    log_child_run(
        name="schema_fetch",
        inputs={},
        outputs={"schema_fields": len(schema.get("fields", {}))},
        tags=["schema"]
    )

    return {"schema": schema, "current_agent": "fetch_schema"}


@traceable_step(name="query_planner", tags=["planning"])
def query_planner_agent(state: AgentState) -> Dict[str, Any]:
    """Decompose complex queries into sub-queries."""
    question = state["messages"][-1].content if state["messages"] else ""
    schema = state.get("schema")

    llm = get_llm()

    planning_prompt = f"""
    Break down this complex procurement query into simpler sub-queries:

    Question: {question}

    Schema available: {bool(schema)}

    Generate 2-4 focused sub-queries that together answer the main question.
    Each sub-query should be executable independently.

    Format as JSON array:
    [
        {{"id": "sub_1", "query": "Find all purchase orders", "purpose": "Get base data"}},
        {{"id": "sub_2", "query": "Filter by department", "purpose": "Apply filters"}}
    ]
    """

    try:
        response = llm.invoke([SystemMessage(content=planning_prompt)])
        sub_queries_text = response.content.strip()

        # Parse JSON response
        try:
            sub_queries = json.loads(sub_queries_text)
        except json.JSONDecodeError:
            # Fallback: create basic sub-queries
            sub_queries = [
                {"id": "main", "query": question, "purpose": "Main query execution"}
            ]

        log_child_run(
            name="query_planning",
            inputs={"question": question[:100]},
            outputs={"sub_queries_count": len(sub_queries)},
            tags=["planning"]
        )

        return {"sub_queries": sub_queries, "current_agent": "planner"}

    except Exception as e:
        log_child_run(
            name="query_planning_error",
            inputs={"question": question[:100]},
            outputs={"error": str(e)},
            tags=["planning-error"]
        )
        return {"sub_queries": [{"id": "fallback", "query": question, "purpose": "Fallback single query"}], "current_agent": "planner"}


@traceable_step(name="aggregation_builder", tags=["query-building"])
def aggregation_builder_agent(state: AgentState) -> Dict[str, Any]:
    """Build MongoDB aggregation pipelines with ReAct-style reasoning."""
    sub_queries = state.get("sub_queries", [])
    schema = state.get("schema")

    if not sub_queries:
        return {"current_agent": "supervisor"}

    results = []

    for sub_query in sub_queries:
        query_text = sub_query.get("query", "")
        query_id = sub_query.get("id", "unknown")

        # Use existing query generation logic
        pipeline = generate_mongodb_query(query_text)

        if pipeline:
            try:
                # Execute the query
                result_json = execute_mongodb_query_tool(json.dumps(pipeline))
                result_data = json.loads(result_json) if result_json.startswith('[') else {"error": result_json}

                results.append({
                    "sub_query_id": query_id,
                    "query": query_text,
                    "pipeline": pipeline,
                    "result": result_data,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "sub_query_id": query_id,
                    "query": query_text,
                    "error": str(e),
                    "success": False
                })
        else:
            results.append({
                "sub_query_id": query_id,
                "query": query_text,
                "error": "Failed to generate pipeline",
                "success": False
            })

    log_child_run(
        name="aggregation_building",
        inputs={"sub_queries_count": len(sub_queries)},
        outputs={"results_count": len(results), "successful": sum(1 for r in results if r.get("success"))},
        tags=["query-building"]
    )

    return {"results": results, "current_agent": "aggregation_builder"}


@traceable_step(name="guardrails", tags=["safety"])
def guardrails_check(state: AgentState) -> Dict[str, Any]:
    """Perform safety and validation checks."""
    question = state["messages"][-1].content if state["messages"] else ""
    results = state.get("results", [])

    llm = get_llm()

    safety_prompt = f"""
    Review this procurement query and results for safety and appropriateness:

    Question: {question}
    Results count: {len(results)}

    Check for:
    1. Data exposure risks
    2. Query complexity/safety
    3. Result size appropriateness
    4. Compliance with procurement policies

    Return: "safe" or "unsafe" with brief reason.
    """

    try:
        response = llm.invoke([SystemMessage(content=safety_prompt)])
        safety_result = response.content.strip().lower()

        status = "safe" if "safe" in safety_result else "unsafe"

        log_child_run(
            name="safety_check",
            inputs={"question": question[:100], "results_count": len(results)},
            outputs={"status": status},
            tags=["safety"]
        )

        return {"guardrails_status": status, "current_agent": "guardrails"}

    except Exception as e:
        log_child_run(
            name="safety_check_error",
            inputs={"question": question[:100]},
            outputs={"error": str(e), "status": "safe"},  # Default to safe
            tags=["safety-error"]
        )
        return {"guardrails_status": "safe", "current_agent": "guardrails"}


@traceable_step(name="merge_results", tags=["post-processing"])
def merge_results_node(state: AgentState) -> Dict[str, Any]:
    """Merge and post-process results from multiple sub-queries."""
    results = state.get("results", [])
    question = state["messages"][-1].content if state["messages"] else ""

    # Simple merging logic - combine all successful results
    merged_data = []
    errors = []

    for result in results:
        if result.get("success"):
            result_data = result.get("result", [])
            if isinstance(result_data, list):
                merged_data.extend(result_data)
            else:
                merged_data.append(result_data)
        else:
            errors.append(result.get("error", "Unknown error"))

    # Convert back to legacy format for compatibility
    mongodb_results = merged_data if merged_data else [{"error": "; ".join(errors)}] if errors else []

    log_child_run(
        name="result_merging",
        inputs={"results_count": len(results)},
        outputs={"merged_count": len(merged_data), "errors": len(errors)},
        tags=["post-processing"]
    )

    return {
        "mongodb_results": mongodb_results,  # For compatibility
        "current_agent": "merge"
    }


@traceable_step(name="respond", tags=["response"])
def response_generator(state: AgentState) -> Dict[str, Any]:
    """Generate the final response to the user."""
    # Use existing response formatting logic
    return format_response_node(state)


# ===== LEGACY FUNCTIONS (kept for compatibility) =====

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


@traceable_step(name="check_conversation_history_relevance", tags=["conversation-history"])
def check_conversation_history_relevance(state: AgentState) -> Dict[str, Any]:
    """Check conversation history and collect relevant context for the current question."""
    messages = state["messages"]
    current_question = messages[-1].content if messages else ""

    # If there's only one message (the current question), no history to check
    if len(messages) <= 1:
        return {"relevant_conversation_history": []}

    # For better context understanding, collect all recent conversation pairs
    # We'll collect up to the last 4 pairs (8 messages) to keep context manageable
    conversation_pairs = []
    # Start from the most recent history, working backwards
    start_idx = max(0, len(messages) - 9)  # -1 for current question, -8 for up to 4 pairs
    for i in range(start_idx, len(messages) - 1, 2):  # Skip the last message (current question)
        if i + 1 < len(messages):
            request = messages[i].content
            response = messages[i + 1].content
            conversation_pairs.append({"request": request, "response": response})

    if not conversation_pairs:
        return {"relevant_conversation_history": []}

    # For ambiguous questions like "What about X?", include recent context by default
    # Use LLM to determine if we should include all context or filter
    llm = get_llm()

    # Check if the current question is likely a follow-up that needs context
    context_check_prompt = f"""
    Analyze if the current question likely needs conversation context to be properly understood.

    Current question: "{current_question}"

    Recent conversation (most recent first):
    {chr(10).join([f"Q: {pair['request'][:100]}... A: {pair['response'][:200]}..." for pair in conversation_pairs[-2:]])}

    Does this question appear to be a follow-up, clarification, or reference to previous context?
    Examples of questions that need context: "What about X?", "How about Y?", "And Z?", "The previous one"

    Return only "true" or "false".
    """

    try:
        context_response = llm.invoke([SystemMessage(content=context_check_prompt)])
        needs_context = context_response.content.strip().lower() == "true"
    except:
        # Default to including context if we can't determine
        needs_context = len(conversation_pairs) > 0

    if needs_context:
        # Include all recent conversation pairs as context
        relevant_history = conversation_pairs
        log_child_run(
            name="conversation_history_check",
            inputs={"current_question": current_question, "total_pairs": len(conversation_pairs)},
            outputs={"relevant_pairs_count": len(relevant_history), "needs_context": True},
            tags=["conversation-history"]
        )
    else:
        # For standalone questions, check which parts are actually relevant
        relevance_prompt = f"""
        Analyze the following conversation history and determine which previous request-response pairs are relevant to the current question.

        Current question: "{current_question}"

        Conversation history (each pair represents a previous request and its response):
        {chr(10).join([f"Pair {i+1}: Request: '{pair['request']}' Response: '{pair['response'][:200]}...'" for i, pair in enumerate(conversation_pairs)])}

        Instructions:
        - Return a JSON array of indices (0-based) of the conversation pairs that are relevant to the current question.
        - A pair is relevant if it contains information, context, or results that would help answer or provide context for the current question.
        - Consider semantic similarity, shared topics, entities, or data that could inform the current query.
        - Only include pairs that genuinely provide useful context or information.

        Return format: [0, 2, 4] (array of relevant pair indices)
        """

        try:
            response = llm.invoke([SystemMessage(content=relevance_prompt)])
            cleaned_response = response.content.strip()

            # Extract JSON array from response
            import re
            json_match = re.search(r'\[.*\]', cleaned_response)
            if json_match:
                import json
                relevant_indices = json.loads(json_match.group())
                if isinstance(relevant_indices, list) and all(isinstance(idx, int) for idx in relevant_indices):
                    relevant_history = [conversation_pairs[idx] for idx in relevant_indices if 0 <= idx < len(conversation_pairs)]
                else:
                    relevant_history = []
            else:
                relevant_history = []

        except Exception as e:
            log_child_run(
                name="conversation_history_check",
                inputs={"current_question": current_question, "conversation_pairs_count": len(conversation_pairs)},
                outputs={"error": str(e)},
                tags=["conversation-history-error"]
            )
            relevant_history = []

        log_child_run(
            name="conversation_history_check",
            inputs={"current_question": current_question, "total_pairs": len(conversation_pairs)},
            outputs={"relevant_pairs_count": len(relevant_history), "needs_context": False},
            tags=["conversation-history"]
        )

    return {"relevant_conversation_history": relevant_history}


@traceable_step(name="analyze_question", tags=["query-generation"])
def analyze_question(state: AgentState) -> Dict[str, Any]:
    """Analyze the question and generate MongoDB queries."""
    messages = state["messages"]
    question = messages[-1].content if messages else ""
    category = state.get("question_category", QuestionCategory.OUT_OF_SCOPE.value)
    relevant_history = state.get("relevant_conversation_history", [])

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

    # Enhance question with relevant conversation history context
    enhanced_question = question
    if relevant_history:
        history_context = "\n".join([
            f"Previous request: {pair['request']}\nPrevious response: {pair['response'][:300]}..."
            for pair in relevant_history
        ])
        enhanced_question = f"Context from conversation history:\n{history_context}\n\nCurrent question: {question}"

    # Check if this is a compound query
    sub_queries = detect_compound_queries(enhanced_question)

    if len(sub_queries) > 1:
        # Handle compound query - execute each sub-query separately
        all_results = []
        successful_queries = 0

        for i, sub_query in enumerate(sub_queries):
            log_child_run(
                name=f"sub_query_{i}",
                inputs={"sub_query": sub_query},
                tags=["compound-query"]
            )

            # Generate and execute each sub-query (use fewer retries for speed)
            pipeline = generate_mongodb_query(sub_query, max_attempts=2)  # Reduce retries for compound queries

            if pipeline:
                try:
                    result = execute_mongodb_query_tool(json.dumps(pipeline))
                    # Validate result is proper JSON
                    if result and result.strip():
                        try:
                            parsed_result = json.loads(result) if result.startswith('[') else {"error": result}
                            all_results.append({
                                "sub_query": sub_query,
                                "result": parsed_result
                            })
                            successful_queries += 1
                        except json.JSONDecodeError:
                            all_results.append({
                                "sub_query": sub_query,
                                "result": {"error": f"Invalid JSON response: {result[:100]}..."}
                            })
                    else:
                        all_results.append({
                            "sub_query": sub_query,
                            "result": {"error": "Empty response from database"}
                        })
                except Exception as e:
                    all_results.append({
                        "sub_query": sub_query,
                        "result": {"error": f"Execution error: {str(e)}"}
                    })
            else:
                all_results.append({
                    "sub_query": sub_query,
                    "result": {"error": "Failed to generate pipeline"}
                })

        # Log compound query summary
        log_child_run(
            name="compound_query_summary",
            inputs={"total_sub_queries": len(sub_queries), "successful": successful_queries},
            outputs={"results_count": len(all_results)},
            tags=["compound-query-summary"]
        )

        return {
            "mongodb_results": all_results,
            "final_answer": "",  # Will be formatted later
        }

    # Single query - use existing logic
    pipeline = generate_mongodb_query(enhanced_question)

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
    relevant_history = state.get("relevant_conversation_history", [])

    # Enhance question with relevant conversation history for better classification
    enhanced_question = question
    if relevant_history:
        history_context = "\n".join([
            f"Previous request: {pair['request']}\nPrevious response: {pair['response'][:300]}..."
            for pair in relevant_history
        ])
        enhanced_question = f"Context from conversation history:\n{history_context}\n\nCurrent question: {question}"

    category, confidence = categorize_question(enhanced_question)
    log_child_run(
        name="question_classifier",
        inputs={"question": question, "enhanced_question": enhanced_question, "history_pairs": len(relevant_history)},
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


# Create the redesigned LangGraph workflow
def create_procurement_agent():
    """Create the redesigned LangGraph agent for procurement queries."""
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("input_processor", input_processor)
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("query_understanding", query_understanding_agent)
    workflow.add_node("fetch_schema", fetch_schema_tool)
    workflow.add_node("planner", query_planner_agent)
    workflow.add_node("aggregation_builder", aggregation_builder_agent)
    workflow.add_node("guardrails", guardrails_check)
    workflow.add_node("merge", merge_results_node)
    workflow.add_node("respond", response_generator)

    # Legacy nodes for compatibility
    workflow.add_node("classify_question", classify_question_node)
    workflow.add_node("check_conversation_history", check_conversation_history_relevance)
    workflow.add_node("analyze_question", analyze_question)
    workflow.add_node("format_response", format_response_node)
    workflow.add_node("handle_out_of_scope", handle_out_of_scope)

    # Define the new architecture flow
    workflow.set_entry_point("check_conversation_history")

    # Initial processing
    workflow.add_edge("check_conversation_history", "classify_question")
    workflow.add_conditional_edges(
        "classify_question",
        lambda state: state.get("question_category", QuestionCategory.OUT_OF_SCOPE.value),
        {
            QuestionCategory.QUERY_GENERATION.value: "input_processor",
            QuestionCategory.DATABASE_INFO.value: "input_processor",
            QuestionCategory.ACQUISITION_METHODS.value: "input_processor",
            QuestionCategory.OUT_OF_SCOPE.value: "handle_out_of_scope",
        },
    )

    # New architecture flow
    workflow.add_edge("input_processor", "supervisor")

    # Supervisor routes to specialized agents
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state.get("current_agent", "respond"),
        {
            "query_understanding": "query_understanding",
            "fetch_schema": "fetch_schema",
            "planner": "planner",
            "aggregation_builder": "aggregation_builder",
            "guardrails": "guardrails",
            "merge": "merge",
            "respond": "respond",
        },
    )

    # All specialized agents route back to supervisor for coordination
    workflow.add_edge("query_understanding", "supervisor")
    workflow.add_edge("fetch_schema", "supervisor")
    workflow.add_edge("planner", "supervisor")
    workflow.add_edge("aggregation_builder", "supervisor")
    workflow.add_edge("guardrails", "supervisor")
    workflow.add_edge("merge", "supervisor")

    # Final response generation
    workflow.add_edge("respond", END)
    workflow.add_edge("handle_out_of_scope", END)

    # Legacy flow (fallback)
    workflow.add_edge("analyze_question", "format_response")
    workflow.add_edge("format_response", END)

    return workflow.compile()
