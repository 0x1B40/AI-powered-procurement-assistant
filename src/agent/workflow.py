"""LangGraph workflow definition for the procurement agent."""

import json
from typing import Dict, Any, List

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from ..services.llm_service.classification import categorize_question, detect_compound_queries
from ..services.llm_service.llm import get_llm
from ..config.constants import REFERENCE_CATEGORIES, QuestionCategory, OUT_OF_SCOPE_RESPONSE
from ..data.mongodb import execute_mongodb_query
from ..services.query_service.query_generation import generate_mongodb_query
from ..services.query_service.response_formatting import format_response
from ..shared.telemetry import traceable_step, log_child_run
from .types import AgentState
from ..data.vector_store import retrieve_reference_chunks


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
    """Check conversation history and collect relevant context using chain of thought reasoning with few-shot examples."""
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

    Use chain of thought reasoning to determine if this question references previous conversation:

    Current question: "{current_question}"

    Recent conversation (most recent first):
    {chr(10).join([f"Q: {pair['request'][:100]}... A: {pair['response'][:200]}..." for pair in conversation_pairs[-2:]])}

    Chain of thought process:
    1. Does the question contain pronouns like "it", "that", "this", "those"?
    2. Does it reference "previous", "last", "above", or similar temporal terms?
    3. Does it ask about specific entities mentioned in recent conversation?
    4. Is it a follow-up question like "What about X?" or "How about Y?"?

    Few-shot examples:

    Example 1:
    Question: "What about the other departments?"
    Previous: "Show me top 5 departments by spend"
    Analysis: Contains "other" referring to departments, needs context → true

    Example 2:
    Question: "How many orders were created in 2014?"
    Previous: "What fiscal years does the data cover?"
    Analysis: Standalone question with no references → false

    Example 3:
    Question: "Can you show me that broken down by month?"
    Previous: "What's the total spend for 2014?"
    Analysis: "that" refers to the spend calculation, needs context → true

    Example 4:
    Question: "What fields are available?"
    Analysis: Schema question, no conversation references → false

    Does this question appear to be a follow-up, clarification, or reference to previous context?

    Return only "true" or "false".
    """

    try:
        context_response = llm.invoke([SystemMessage(content=context_check_prompt)])
        needs_context = context_response.content.strip().lower() == "true"

        # Log the chain of thought reasoning for context determination
        log_child_run(
            name="context_relevance_check",
            inputs={"question": current_question, "method": "chain_of_thought_with_few_shot"},
            outputs={"needs_context": needs_context, "llm_reasoning": context_response.content.strip()},
            tags=["conversation-history", "chain-of-thought"]
        )
    except Exception as e:
        # Default to including context if we can't determine
        needs_context = len(conversation_pairs) > 0
        log_child_run(
            name="context_relevance_check",
            inputs={"question": current_question, "error": str(e)},
            outputs={"needs_context": needs_context, "fallback": True},
            tags=["conversation-history", "error"]
        )

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
        Analyze the conversation history and determine which previous request-response pairs are relevant to the current question.

        Use chain of thought reasoning to identify relevant context:

        Current question: "{current_question}"

        Conversation history (each pair represents a previous request and its response):
        {chr(10).join([f"Pair {i+1}: Request: '{pair['request']}' Response: '{pair['response'][:200]}...'" for i, pair in enumerate(conversation_pairs)])}

        Chain of thought process:
        1. Identify the core topic and entities in the current question
        2. Scan each conversation pair for semantic similarity
        3. Look for shared entities (departments, suppliers, fiscal years, etc.)
        4. Consider if previous results could inform or provide context for the current query
        5. Prioritize recent, highly relevant pairs over older, tangentially related ones

        Few-shot examples:

        Example 1:
        Current: "Show me spend by department for fiscal year 2014-2015"
        History includes: Pair 1: "What fiscal years are in the data?" → includes 2014-2015
        Analysis: Fiscal year context is directly relevant → [0]

        Example 2:
        Current: "How many orders from Department of Transportation?"
        History includes: Pair 1: "Top 5 departments by order count" → shows Transportation with 14,871 orders
        Analysis: Direct department information is highly relevant → [0]

        Example 3:
        Current: "What suppliers have the most orders?"
        History includes: Pair 1: "Show me top suppliers by spend" Pair 2: "What departments are there?"
        Analysis: Supplier spend data is relevant, department list is not → [0]

        Example 4:
        Current: "Calculate average order value for IT Goods"
        History includes: Pair 1: "What acquisition types exist?" → mentions IT Goods
        Analysis: Acquisition type context helps → [0]

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

            # Log the chain of thought relevance filtering
            log_child_run(
                name="relevance_filtering",
                inputs={
                    "question": current_question,
                    "total_pairs": len(conversation_pairs),
                    "method": "chain_of_thought_with_few_shot"
                },
                outputs={
                    "relevant_indices": relevant_indices if 'relevant_indices' in locals() and isinstance(relevant_indices, list) else [],
                    "relevant_pairs_count": len(relevant_history),
                    "llm_response": cleaned_response[:200] + "..." if len(cleaned_response) > 200 else cleaned_response
                },
                tags=["conversation-history", "chain-of-thought", "filtering"]
            )

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
    """Categorize the incoming user question before running expensive steps using chain of thought reasoning."""
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

    # Log the enhanced question for debugging
    log_child_run(
        name="classification_input",
        inputs={"original_question": question, "enhanced_question": enhanced_question, "conversation_pairs": len(relevant_history)},
        tags=["classifier-input"],
    )

    category, confidence = categorize_question(enhanced_question)

    # Log detailed classification reasoning for better traceability
    log_child_run(
        name="question_classifier",
        inputs={
            "question": question,
            "enhanced_question": enhanced_question,
            "history_pairs": len(relevant_history),
            "classification_method": "chain_of_thought_with_few_shot"
        },
        outputs={
            "category": category.value,
            "confidence": confidence,
            "routing_decision": "analyze_question" if category.value in ["query_generation", "database_info", "acquisition_methods"] else "handle_out_of_scope"
        },
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
    workflow.add_node("check_conversation_history", check_conversation_history_relevance)
    workflow.add_node("analyze_question", analyze_question)
    workflow.add_node("format_response", format_response_node)
    workflow.add_node("handle_out_of_scope", handle_out_of_scope)

    # Define flow
    workflow.set_entry_point("check_conversation_history")
    workflow.add_edge("check_conversation_history", "classify_question")
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
