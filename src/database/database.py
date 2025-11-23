"""Database connection and query execution for MongoDB."""

import json
from datetime import datetime
from functools import lru_cache
from typing import List, Dict, Any

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import OperationFailure

from ..config_and_constants.config import get_settings


@lru_cache
def _get_collection() -> Collection:
    """Get MongoDB collection with caching."""
    settings = get_settings()
    client = MongoClient(settings.mongodb_uri)
    database = client[settings.mongodb_db]
    return database[settings.mongodb_collection]


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

        # If no results and this is a simple $match + $count query on department/supplier,
        # try fuzzy matching as fallback
        if not results and len(pipeline) == 2:
            match_stage = pipeline[0]
            count_stage = pipeline[1]
            if (isinstance(match_stage, dict) and "$match" in match_stage and
                isinstance(count_stage, dict) and "$count" in count_stage):

                match_conditions = match_stage["$match"]
                # Check if it's a simple department or supplier name match
                if len(match_conditions) == 1:
                    field_name, field_value = list(match_conditions.items())[0]
                    if field_name in ["department_name", "supplier_name"] and isinstance(field_value, str):
                        # Try fuzzy regex matching
                        fuzzy_pipeline = [
                            {"$match": {field_name: {"$regex": field_value.replace(" ", ".*"), "$options": "i"}}},
                            count_stage
                        ]
                        fuzzy_results = list(collection.aggregate(fuzzy_pipeline, allowDiskUse=True))
                        if fuzzy_results:
                            results = fuzzy_results

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
