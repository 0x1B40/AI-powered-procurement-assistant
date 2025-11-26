"""MongoDB data access layer."""

import json
import logging
from typing import Any, Dict, List

from pymongo import MongoClient
from pymongo.collection import Collection

from ..config.config import get_settings

logger = logging.getLogger(__name__)


def get_collection() -> Collection:
    """Get the MongoDB collection for procurement data."""
    settings = get_settings()
    client = MongoClient(settings.mongodb_uri)
    return client[settings.mongodb_db][settings.mongodb_collection]


def execute_mongodb_query(query: str) -> str:
    """
    Execute a MongoDB aggregation pipeline query.

    Args:
        query: A valid MongoDB aggregation pipeline as JSON string

    Returns:
        Query results as formatted JSON string
    """
    try:
        collection = get_collection()
        pipeline = json.loads(query)

        logger.info("Executing MongoDB pipeline: %s", query[:200] + "..." if len(query) > 200 else query)

        results = list(collection.aggregate(pipeline, allowDiskUse=True))

        # Convert ObjectId and other non-serializable types
        def serialize_result(result: Dict[str, Any]) -> Dict[str, Any]:
            if isinstance(result, dict):
                return {k: serialize_result(v) for k, v in result.items()}
            elif isinstance(result, list):
                return [serialize_result(item) for item in result]
            elif hasattr(result, '__dict__'):
                # Handle custom objects
                return str(result)
            else:
                return result

        serialized_results = [serialize_result(result) for result in results]

        return json.dumps(serialized_results, indent=2, default=str)

    except Exception as e:
        error_msg = f"MongoDB query execution failed: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})
