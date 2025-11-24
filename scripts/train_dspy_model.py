"""Train and optimize DSPy model for MongoDB query generation."""

import json
from pathlib import Path
from typing import List

import dspy

from src.config_and_constants.config import get_settings
from src.llm_and_classification.llm import configure_dspy
from src.query_and_response.dspy_query_generation import (
    MongoDBQueryGenerator,
    load_training_examples,
    train_query_generator
)


def load_training_data(json_path: Path) -> List[dspy.Example]:
    """Load training examples from JSON file."""
    if not json_path.exists():
        raise FileNotFoundError(f"Training data not found at {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    examples = []
    for item in data.get('examples', []):
        example = dspy.Example(
            question=item['question'],
            schema="",  # Schema will be provided during training
            pipeline_json=item['pipeline_json'],
            expected_type=item.get('expected_type', 'unknown')
        ).with_inputs("question", "schema")
        examples.append(example)

    return examples


def validate_pipeline_metric(example, prediction, trace=None):
    """Metric to evaluate MongoDB pipeline quality."""
    try:
        # Parse the generated pipeline
        pipeline = json.loads(prediction.pipeline_json)

        # Basic validation
        if not isinstance(pipeline, list):
            return 0.0

        if not pipeline:
            return 0.0  # Empty pipeline

        score = 0.5  # Base score for valid JSON structure

        # Check for proper MongoDB operators
        pipeline_str = json.dumps(pipeline)
        if '$match' in pipeline_str:
            score += 0.1
        if '$group' in pipeline_str:
            score += 0.1
        if '$sort' in pipeline_str:
            score += 0.1
        if '$project' in pipeline_str:
            score += 0.1
        if '$limit' in pipeline_str:
            score += 0.05

        # Bonus for multi-stage pipelines
        if len(pipeline) > 1:
            score += 0.1

        # Check for proper field references (snake_case)
        if any(field in pipeline_str for field in [
            'purchase_order_number', 'department_name', 'supplier_name',
            'total_price', 'fiscal_year', 'acquisition_type'
        ]):
            score += 0.1

        return min(score, 1.0)

    except (json.JSONDecodeError, KeyError, TypeError):
        return 0.0


def main():
    """Train the DSPy MongoDB query generator."""
    print("Starting DSPy model training...")

    # Configure DSPy
    configure_dspy()
    settings = get_settings()

    # Paths
    training_data_path = Path("data/dspy_training_data.json")
    model_save_path = settings.dspy_optimized_model_path

    # Load training data
    print(f"Loading training data from {training_data_path}")
    if training_data_path.name == "dspy_training_data.json":
        examples = load_training_data(training_data_path)
    else:
        # Load from original query test results format
        from src.query_and_response.dspy_query_generation import load_training_examples
        examples = load_training_examples(training_data_path)
    print(f"Loaded {len(examples)} training examples")

    if len(examples) < 5:
        print("Warning: Very few training examples. Consider generating more training data.")
        return

    # Split into train/validation sets
    train_size = int(0.8 * len(examples))
    trainset = examples[:train_size]
    valset = examples[train_size:]

    print(f"Training on {len(trainset)} examples, validating on {len(valset)} examples")

    # Create and train the model
    print("Training model...")
    try:
        trained_model = train_query_generator(
            training_data_path=training_data_path,
            save_path=model_save_path,
            num_examples=len(trainset)
        )

        print(f"Model trained and saved to {model_save_path}")

        # Evaluate on validation set
        print("Evaluating on validation set...")
        correct = 0
        total = len(valset)

        for example in valset:
            prediction = trained_model(
                question=example.question,
                schema=""  # Provide schema during inference
            )
            score = validate_pipeline_metric(example, prediction)
            if score > 0.7:  # Consider it correct if score > 0.7
                correct += 1

        accuracy = correct / total if total > 0 else 0
        print(".1%")

        # Save training metadata
        metadata_path = model_save_path.parent / "training_metadata.json"
        metadata = {
            "training_date": str(Path(__file__).parent),
            "training_examples": len(trainset),
            "validation_examples": len(valset),
            "validation_accuracy": accuracy,
            "model_path": str(model_save_path)
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print("Training complete!")

    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
