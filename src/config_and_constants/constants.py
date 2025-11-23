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

Categorize the user's message into exactly one of these values:
- "query_generation": the user wants analytics or MongoDB query results over procurement data.
- "database_info": the user is asking about the dataset schema, metadata, or data dictionary.
- "acquisition_methods": the user is asking about acquisition_type, acquisition_method, or procurement method definitions/usages.
- "out_of_scope": greetings, chit-chat, or any request unrelated to procurement data, the schema, or acquisition methods.

Return ONLY a JSON object like {{"category": "query_generation", "confidence": 0.87}}:
- "category" must be one of the allowed strings.
- "confidence" must be a number between 0 and 1 indicating your certainty.

QUESTION:
{question}
"""

OUT_OF_SCOPE_RESPONSE = (
    "I'm focused on generating procurement MongoDB queries, describing the "
    "dataset, and explaining acquisition methods. Please ask about those topics."
)

MAX_QUERY_ATTEMPTS = 3
RETRY_SNIPPET_LIMIT = 800
