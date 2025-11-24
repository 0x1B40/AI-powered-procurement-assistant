"""Constants and configuration for the procurement agent."""

from enum import Enum


class QuestionCategory(str, Enum):
    QUERY_GENERATION = "query_generation"
    DATABASE_INFO = "database_info"
    ACQUISITION_METHODS = "acquisition_methods"
    OUT_OF_SCOPE = "out_of_scope"


REFERENCE_CATEGORIES = {
    QuestionCategory.DATABASE_INFO.value,
    QuestionCategory.ACQUISITION_METHODS.value,
}

SCHEMA_FIELDS = """
Important fields (MongoDB collection `purchase_orders`):
- purchase_order_number (string) — unique order id; multiple rows per PO possible. When counting "orders", group by this field to get unique purchase orders.
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

CLASSIFIER_PROMPT = """You are an intent classifier for the California procurement assistant.

Your task is to categorize user questions into exactly one of these categories. Use chain of thought reasoning to analyze the question step by step.

Categories:
- "query_generation": The user wants analytics, aggregations, or specific data results from procurement records (e.g., "how many orders", "what fiscal years are present", "show me totals by department").
- "database_info": The user is asking about the dataset structure, field definitions, or technical schema (e.g., "what fields are available", "what does this column mean", "how is the data structured").
- "acquisition_methods": The user is asking about acquisition_type, acquisition_method, or procurement method definitions/usages.
- "out_of_scope": Greetings, chit-chat, or any request unrelated to procurement data, the schema, or acquisition methods.

Chain of thought process:
1. Identify the core intent: What is the user actually asking for?
2. Determine if they want data results vs. structural information vs. definitions
3. Check if the question relates to procurement data analysis

Few-shot examples:

Example 1:
Question: "How many purchase orders were created in 2013?"
Analysis: User wants a count of orders for a specific year - this requires querying and aggregating procurement data.
Category: "query_generation"

Example 2:
Question: "What fiscal years does this dataset cover?"
Analysis: User wants to know what fiscal year values exist in the data - this requires examining the actual data content.
Category: "query_generation"

Example 3:
Question: "What fields are available in the purchase orders table?"
Analysis: User is asking about the database structure and available columns - this is about schema, not data content.
Category: "database_info"

Example 4:
Question: "What does the acquisition_type field mean?"
Analysis: User wants explanation of a field definition - this is about understanding procurement terminology.
Category: "acquisition_methods"

Example 5:
Question: "Hello, how are you?"
Analysis: This is a greeting with no relation to procurement data - not relevant to the assistant's purpose.
Category: "out_of_scope"

Now analyze the following question using the same chain of thought process:

QUESTION: {question}

Return ONLY a JSON object like {{"category": "query_generation", "confidence": 0.87}}:
- "category" must be one of the allowed strings.
- "confidence" must be a number between 0 and 1 indicating your certainty.
"""

OUT_OF_SCOPE_RESPONSE = (
    "I'm focused on generating procurement MongoDB queries, describing the "
    "dataset, and explaining acquisition methods. Please ask about those topics."
)

MAX_QUERY_ATTEMPTS = 5
RETRY_SNIPPET_LIMIT = 800
