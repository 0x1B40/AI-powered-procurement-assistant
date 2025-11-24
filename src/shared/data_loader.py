from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
from pymongo import MongoClient

from ..config.config import get_settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

MONETARY_COLUMNS = ("total_price", "unit_price")


def snake_case(name: str) -> str:
    return (
        name.strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("__", "_")
        .lower()
    )


def sanitize_currency_series(series: pd.Series) -> pd.Series:
    """Strip currency formatting and return floats."""
    cleaned = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"[^\d\.\-]", "", regex=True)
    )
    cleaned = cleaned.replace("", pd.NA)
    return pd.to_numeric(cleaned, errors="coerce")


def normalize_chunk(chunk: pd.DataFrame) -> Iterable[dict]:
    chunk = chunk.rename(columns={col: snake_case(col) for col in chunk.columns})

    for column in MONETARY_COLUMNS:
        if column in chunk.columns:
            chunk[column] = sanitize_currency_series(chunk[column])

    # Convert DataFrame to dict and manually replace NaN with None
    records = chunk.to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
    return records


def load_csv(csv_path: Path, batch_size: int = 5_000) -> None:
    settings = get_settings()
    client = MongoClient(settings.mongodb_uri)
    collection = client[settings.mongodb_db][settings.mongodb_collection]
    collection.drop()

    total_inserted = 0
    for chunk in pd.read_csv(csv_path, chunksize=batch_size):
        records = list(normalize_chunk(chunk))
        if not records:
            continue
        result = collection.insert_many(records)
        total_inserted += len(result.inserted_ids)
        logger.info("Inserted %d rows (total=%d)", len(result.inserted_ids), total_inserted)

    db_count = collection.count_documents({})
    if db_count != total_inserted:
        raise RuntimeError(
            f"MongoDB document count mismatch (expected {total_inserted}, found {db_count})."
        )

    logger.info(
        "Finished loading %d rows into %s.%s",
        total_inserted,
        settings.mongodb_db,
        settings.mongodb_collection,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load the procurement CSV into MongoDB.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to the PURCHASE ORDER DATA EXTRACT.csv file")
    parser.add_argument("--batch-size", type=int, default=5_000, help="Number of rows per MongoDB insert chunk")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv}")
    load_csv(args.csv, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

