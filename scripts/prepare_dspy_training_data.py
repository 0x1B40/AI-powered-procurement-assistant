"""Prepare training data for DSPy from existing query test results."""

import json
import re
from pathlib import Path
from typing import Dict, Any, List

import dspy


def convert_pandas_to_mongodb_query(pandas_query: str, question: str) -> str:
    """Convert a pandas query to MongoDB aggregation pipeline based on the question intent."""

    # This is a simplified conversion - in practice, you'd want more sophisticated logic
    # For now, we'll create basic examples based on patterns

    if "len(df)" in pandas_query:
        # Count query
        return '[{"$count": "total_count"}]'

    elif "unique()" in pandas_query:
        # Unique values query
        if "department_name" in pandas_query:
            return '[{"$group": {"_id": "$department_name"}}, {"$project": {"department": "$_id", "_id": 0}}]'
        elif "fiscal_year" in pandas_query:
            return '[{"$group": {"_id": "$fiscal_year"}}, {"$project": {"fiscal_year": "$_id", "_id": 0}}]'
        elif "acquisition_type" in pandas_query:
            return '[{"$group": {"_id": "$acquisition_type"}}, {"$project": {"acquisition_type": "$_id", "_id": 0}}]'

    elif "sum()" in pandas_query and "groupby" in pandas_query:
        # Aggregation by group
        if "department_name" in pandas_query and "total_price" in pandas_query:
            return '[{"$group": {"_id": "$department_name", "total_spend": {"$sum": "$total_price"}}}, {"$sort": {"total_spend": -1}}]'
        elif "fiscal_year" in pandas_query and "total_price" in pandas_query:
            return '[{"$group": {"_id": "$fiscal_year", "total_spend": {"$sum": "$total_price"}}}, {"$sort": {"total_spend": -1}}]'

    elif "mean()" in pandas_query:
        # Average query
        if "total_price" in pandas_query:
            return '[{"$group": {"_id": null, "average_price": {"$avg": "$total_price"}}}]'

    elif "value_counts()" in pandas_query:
        # Distribution query
        if "acquisition_type" in pandas_query:
            return '[{"$group": {"_id": "$acquisition_type", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}]'

    elif "$match" in pandas_query or "df[" in pandas_query:
        # Filter query
        if "fiscal_year" in pandas_query and "2014-2015" in pandas_query:
            return '[{"$match": {"fiscal_year": "2014-2015"}}, {"$count": "count"}]'

    # Default fallback - return empty pipeline for unsupported queries
    return "[]"


def load_query_test_results(results_path: Path) -> List[Dict[str, Any]]:
    """Load and parse query test results."""
    if not results_path.exists():
        print(f"Warning: {results_path} not found")
        return []

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data.get('results', [])


def create_dspy_training_examples(results: List[Dict[str, Any]]) -> List[dspy.Example]:
    """Convert query test results to DSPy training examples."""
    examples = []

    for result in results:
        if not result.get('success', False):
            continue  # Skip failed examples

        question = result['natural_language']
        pandas_query = result['pandas_query']

        # Convert pandas query to MongoDB pipeline
        mongodb_pipeline = convert_pandas_to_mongodb_query(pandas_query, question)

        # Skip if conversion failed
        if mongodb_pipeline == "[]":
            continue

        # Create DSPy example
        example = dspy.Example(
            question=question,
            schema="",  # Will be filled in during training
            pipeline_json=mongodb_pipeline,
            expected_type=result.get('expected_type', 'unknown')
        ).with_inputs("question", "schema")

        examples.append(example)

    return examples


def save_training_data(examples: List[dspy.Example], output_path: Path):
    """Save training examples to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert examples to serializable format
    training_data = {
        "metadata": {
            "total_examples": len(examples),
            "created_at": str(Path(__file__).parent / "prepare_dspy_training_data.py")
        },
        "examples": [
            {
                "question": ex.question,
                "pipeline_json": ex.pipeline_json,
                "expected_type": getattr(ex, 'expected_type', 'unknown')
            }
            for ex in examples
        ]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(examples)} training examples to {output_path}")


def main():
    """Main function to prepare DSPy training data."""
    print("Preparing DSPy training data from query test results...")

    # Paths
    query_results_path = Path("query_test_results.json")
    training_data_path = Path("data/dspy_training_data.json")

    # Load existing test results
    results = load_query_test_results(query_results_path)
    print(f"Loaded {len(results)} test results")

    # Convert to DSPy examples
    examples = create_dspy_training_examples(results)
    print(f"Created {len(examples)} DSPy training examples")

    # Save training data
    save_training_data(examples, training_data_path)

    print("Training data preparation complete!")


if __name__ == "__main__":
    main()
