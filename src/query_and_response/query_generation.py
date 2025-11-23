"""Query generation, validation, and retry logic."""

import json
import re
from typing import Tuple, List, Dict, Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ..config_and_constants.constants import MAX_QUERY_ATTEMPTS, RETRY_SNIPPET_LIMIT, SCHEMA_FIELDS
from ..llm_and_classification.llm import get_llm


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
- For questions about data ranges or "what years/periods does the data cover?", find min/max dates using $dateFromString and $year/$month functions.
- For quarters: Use {{"$ceil": {{"$divide": [{{$month: date}}, 3]}}}} to get quarter numbers (1-4). Do NOT use invalid format specifiers like '%q' in $dateToString.
- NEVER use $$ROOT, $out, $merge, or stages that return the full document as a field.
- If the question asks for text explanation without data (e.g., greetings), return an empty array [].

Chain-of-Thought Examples:
==================

Example 1: "What are the top 5 departments by total spend?"
Step-by-step reasoning:
1. Need to group by department and sum total_price
2. Sort by total spend descending
3. Limit to top 5
4. Project to clean format
Query: [{{"$group": {{"_id": "$department_name", "total_spend": {{"$sum": "$total_price"}}}}}}, {{"$sort": {{"total_spend": -1}}}}, {{"$limit": 5}}]

Example 2: "How many unique suppliers are there for each acquisition type?"
Step-by-step reasoning:
1. Group by acquisition_type and collect unique supplier_names using $addToSet
2. Count the size of each supplier set
3. Sort by supplier count descending
Query: [{{"$group": {{"_id": "$acquisition_type", "suppliers": {{"$addToSet": "$supplier_name"}}}}}}, {{"$project": {{"acquisition_type": "$_id", "unique_supplier_count": {{"$size": "$suppliers"}}, "_id": 0}}}}, {{"$sort": {{"unique_supplier_count": -1}}}}]

Example 3: "What is the average order value for IT Goods in fiscal year 2014-2015?"
Step-by-step reasoning:
1. Filter for IT Goods and fiscal year 2014-2015
2. Group all matching documents to calculate average total_price
Query: [{{"$match": {{"acquisition_type": "IT Goods", "fiscal_year": "2014-2015"}}}}, {{"$group": {{"_id": null, "average_order_value": {{"$avg": "$total_price"}}}}}}]

Example 4: "Which suppliers have the most purchase orders over $100,000?"
Step-by-step reasoning:
1. Filter for orders over $100,000
2. Group by supplier_name and count orders
3. Sort by order count descending
4. Limit to top suppliers
Query: [{{"$match": {{"total_price": {{"$gt": 100000}}}}}}, {{"$group": {{"_id": "$supplier_name", "order_count": {{"$sum": 1}}}}}}, {{"$sort": {{"order_count": -1}}}}, {{"$limit": 10}}]

Example 5: "What is the total spend by department for fiscal year 2013-2014, sorted by spend?"
Step-by-step reasoning:
1. Filter for fiscal year 2013-2014
2. Group by department_name and sum total_price
3. Sort by total spend descending
Query: [{{"$match": {{"fiscal_year": "2013-2014"}}}}, {{"$group": {{"_id": "$department_name", "total_spend": {{"$sum": "$total_price"}}}}}}, {{"$sort": {{"total_spend": -1}}}}]

Example 6: "How many orders were created in 2014?"
Step-by-step reasoning:
1. Filter creation_date to contain "2014" (since it's stored as string)
2. Count the matching documents
Query: [{{"$match": {{"creation_date": {{"$regex": "2014"}}}}}}, {{"$count": "orders_in_2014"}}]

Example 6.5: "How many orders were created in the first quarter of 2014?"
Step-by-step reasoning:
1. First quarter of 2014 is January-March (01-03)
2. Filter creation_date using regex for months 01, 02, 03 and year 2014
3. Group by purchase_order_number to count unique orders (since multiple rows per PO)
4. Count the unique purchase orders
Query: [{{"$match": {{"creation_date": {{"$regex": "^(0[1-3])/\\d{{2}}/2014$"}}}}}}, {{"$group": {{"_id": "$purchase_order_number"}}}}, {{"$count": "orders_q1_2014"}}]

Example 6.6: "How many orders were created in the last quarter of 2013?"
Step-by-step reasoning:
1. Last quarter of 2013 is October-December (10-12)
2. Filter creation_date using regex for months 10, 11, 12 and year 2013
3. Group by purchase_order_number to count unique orders (since multiple rows per PO)
4. Count the unique purchase orders
Query: [{{"$match": {{"creation_date": {{"$regex": "^(1[0-2])/\\d{{2}}/2013$"}}}}}}, {{"$group": {{"_id": "$purchase_order_number"}}}}, {{"$count": "orders_q4_2013"}}]

Example 7: "What are the largest 5 orders with supplier and department details?"
Step-by-step reasoning:
1. Sort all documents by total_price descending
2. Limit to top 5
3. Project only needed fields (purchase_order_number, supplier_name, department_name, total_price)
Query: [{{"$sort": {{"total_price": -1}}}}, {{"$limit": 5}}, {{"$project": {{"purchase_order_number": 1, "supplier_name": 1, "department_name": 1, "total_price": 1}}}}]

Example 8: "What percentage of orders use CalCard?"
Step-by-step reasoning:
1. Group to count total orders and CalCard orders
2. Calculate percentage using $multiply and $divide
Query: [{{"$group": {{"_id": null, "total_orders": {{"$sum": 1}}, "calcard_orders": {{"$sum": {{"$cond": [{{"$eq": ["$calcard", "YES"]}}, 1, 0]}}}}}}}}, {{"$project": {{"percentage_calcard": {{"$multiply": [{{"$divide": ["$calcard_orders", "$total_orders"]}}, 100]}}, "_id": 0}}}}]

Example 9: "Which acquisition types have the highest average order values?"
Step-by-step reasoning:
1. Group by acquisition_type and calculate average total_price
2. Sort by average order value descending
Query: [{{"$group": {{"_id": "$acquisition_type", "avg_order_value": {{"$avg": "$total_price"}}}}}}, {{"$sort": {{"avg_order_value": -1}}}}]

Example 10: "How many unique purchase orders are there total?"
Step-by-step reasoning:
1. Group by purchase_order_number to get unique orders
2. Count the groups
Query: [{{"$group": {{"_id": "$purchase_order_number"}}}}, {{"$count": "unique_purchase_orders"}}]

Example 11: "What years does the purchase order data cover?"

Step-by-step reasoning:

1. Extract years from creation_date using $dateFromString and $year

2. Find minimum and maximum years

Query: [{{"$group": {{"_id": null, "min_year": {{"$min": {{"$year": {{"$dateFromString": {{"dateString": "$creation_date", "format": "%m/%d/%Y"}}}}}}}}, "max_year": {{"$max": {{"$year": {{"$dateFromString": {{"dateString": "$creation_date", "format": "%m/%d/%Y"}}}}}}}}}}, {{"$project": {{"_id": 0, "data_range": {{"$concat": [{{"$toString": "$min_year"}}, " to ", {{"$toString": "$max_year"}}]}}}}}}]

Complex Multi-Stage Examples:
============================

Example 12: "Top 3 suppliers by total spend in each fiscal year"
Step-by-step reasoning:
1. Group by fiscal_year and supplier_name, sum total_price
2. Sort within each fiscal year group by total_spend
3. Use $push to collect top suppliers per fiscal year
4. Unwind to flatten results
Query: [{{"$group": {{"_id": {{"fiscal_year": "$fiscal_year", "supplier": "$supplier_name"}}, "total_spend": {{"$sum": "$total_price"}}}}}}, {{"$sort": {{"_id.fiscal_year": 1, "total_spend": -1}}}}, {{"$group": {{"_id": "$_id.fiscal_year", "suppliers": {{"$push": {{"supplier_name": "$_id.supplier", "total_spend": "$total_spend"}}}}}}}}, {{"$project": {{"fiscal_year": "$_id", "top_suppliers": {{"$slice": ["$suppliers", 3]}}, "_id": 0}}}}]

Example 13: "Monthly spend trends for 2014"
Step-by-step reasoning:
1. Filter for 2014 in creation_date
2. Add month field using $dateFromString and $month
3. Group by month and sum total_price
4. Sort by month
Query: [{{"$match": {{"creation_date": {{"$regex": "2014"}}}}}}, {{"$addFields": {{"month": {{"$month": {{"$dateFromString": {{"dateString": "$creation_date", "format": "%m/%d/%Y"}}}}}}}}}}, {{"$group": {{"_id": "$month", "monthly_spend": {{"$sum": "$total_price"}}}}}}, {{"$sort": {{"_id": 1}}}}, {{"$project": {{"month": "$_id", "total_spend": "$monthly_spend", "_id": 0}}}}]

Example 14: "Departments with above-average spend in IT Goods"
Step-by-step reasoning:
1. Filter for IT Goods
2. Calculate overall average spend
3. Group by department and sum spend
4. Filter departments above the calculated average
Query: [{{"$match": {{"acquisition_type": "IT Goods"}}}}, {{"$group": {{"_id": null, "overall_avg": {{"$avg": "$total_price"}}, "dept_totals": {{"$push": {{"dept": "$department_name", "spend": "$total_price"}}}}}}}}, {{"$unwind": "$dept_totals"}}, {{"$match": {{"$expr": {{"$gt": ["$dept_totals.spend", "$overall_avg"]}}}}}}, {{"$project": {{"department": "$dept_totals.dept", "spend": "$dept_totals.spend", "_id": 0}}}}]

==================
Now generate the MongoDB aggregation pipeline for the question above.
"""


def generate_mongodb_query(question: str, max_attempts: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
    """Generate and validate a MongoDB query pipeline."""
    llm = get_llm()

    prompt = create_mongodb_query_prompt(question)
    system_message = SystemMessage(content=prompt)

    pipeline = None
    error_msg = ""
    previous_output = ""

    attempts = max_attempts or MAX_QUERY_ATTEMPTS
    for attempt in range(attempts):
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
