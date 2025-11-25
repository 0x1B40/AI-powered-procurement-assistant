from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.vector_store import build_reference_vector_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or refresh the Chroma vector store for procurement reference docs."
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=None,
        help="Directory containing PDF/DOCX reference documents (defaults to Settings.reference_docs_dir).",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=None,
        help="Directory where the Chroma store should be persisted (defaults to Settings.vector_store_dir).",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help="Override the Chroma collection name (defaults to Settings.vector_collection_name).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the vector store even if one already exists at the destination.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    location = build_reference_vector_store(
        docs_dir=args.docs_dir,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        force_rebuild=args.force,
    )
    print(f"Vector store ready at: {location}")


if __name__ == "__main__":
    main()

