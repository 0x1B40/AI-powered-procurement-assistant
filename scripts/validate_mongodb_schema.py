from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from pymongo import MongoClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import get_settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

EXPECTED_FIELDS: List[str] = [
    "creation_date",
    "purchase_date",
    "fiscal_year",
    "lpa_number",
    "purchase_order_number",
    "requisition_number",
    "acquisition_type",
    "sub_acquisition_type",
    "acquisition_method",
    "sub_acquisition_method",
    "department_name",
    "supplier_code",
    "supplier_name",
    "supplier_qualifications",
    "supplier_zip_code",
    "calcard",
    "item_name",
    "item_description",
    "quantity",
    "unit_price",
    "total_price",
    "classification_codes",
    "normalized_unspsc",
    "commodity_title",
    "class",
    "class_title",
    "family",
    "family_title",
    "segment",
    "segment_title",
    "location",
]


def get_collection():
    settings = get_settings()
    client = MongoClient(settings.mongodb_uri)
    return client[settings.mongodb_db][settings.mongodb_collection]


def summarize_field_presence(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    total_docs = len(documents) or 1
    for field in EXPECTED_FIELDS:
        present = 0
        non_null = 0
        for doc in documents:
            if field in doc:
                present += 1
                if doc[field] is not None:
                    non_null += 1
        summary.append(
            {
                "field": field,
                "present_pct": round((present / total_docs) * 100, 2),
                "non_null_pct": round((non_null / total_docs) * 100, 2),
            }
        )
    return summary


def sample_documents(collection, sample_size: int) -> List[Dict[str, Any]]:
    if sample_size <= 0:
        return []
    cursor = collection.aggregate([{"$sample": {"size": sample_size}}])
    return list(cursor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MongoDB procurement collection schema.")
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="Expected total number of documents (optional).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of random documents to sample for schema analysis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collection = get_collection()

    total_docs = collection.count_documents({})
    logger.info("MongoDB collection contains %s documents", f"{total_docs:,}")

    if args.expected_count is not None and total_docs != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count:,} documents but found {total_docs:,}."
        )

    sample_size = min(args.sample_size, total_docs)
    sample_docs = sample_documents(collection, sample_size)
    logger.info("Sampled %d documents for schema inspection", len(sample_docs))

    field_stats = summarize_field_presence(sample_docs)
    logger.info("Field coverage summary (top missing fields):")
    for stat in sorted(field_stats, key=lambda x: x["non_null_pct"])[:10]:
        logger.info(
            "%-25s present=%5.1f%% | non-null=%5.1f%%",
            stat["field"],
            stat["present_pct"],
            stat["non_null_pct"],
        )

    if sample_docs:
        example = sample_docs[0].copy()
        example.pop("_id", None)
        logger.info("Example document:\n%s", json.dumps(example, indent=2, default=str))

    logger.info("Validation complete.")


if __name__ == "__main__":
    main()


