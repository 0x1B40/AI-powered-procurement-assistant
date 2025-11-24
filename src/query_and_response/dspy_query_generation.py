"""DSPy-based MongoDB query generation modules."""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import dspy
from dspy.teleprompt import BootstrapFewShot

from ..config_and_constants.config import get_settings
from ..config_and_constants.constants import SCHEMA_FIELDS
from ..llm_and_classification.llm import get_dspy_lm, configure_dspy


class MongoDBQueryIntentClassifier(dspy.Module):
    """Classify the intent of a MongoDB query question."""

    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(
            "question, schema -> intent: str, reasoning: str"
        )

    def forward(self, question: str, schema: str) -> dspy.Prediction:
        return self.classify(question=question, schema=schema)


class MongoDBFieldExtractor(dspy.Module):
    """Extract relevant fields from the question and schema."""

    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(
            "question, schema -> relevant_fields: List[str], reasoning: str"
        )

    def forward(self, question: str, schema: str) -> dspy.Prediction:
        return self.extract(question=question, schema=schema)


class MongoDBAggregationBuilder(dspy.Module):
    """Build MongoDB aggregation pipeline from intent and fields."""

    def __init__(self):
        super().__init__()
        self.build = dspy.ChainOfThought(
            "question, intent, relevant_fields, schema -> pipeline_json: str, reasoning: str"
        )

    def forward(self, question: str, intent: str, relevant_fields: List[str], schema: str) -> dspy.Prediction:
        fields_str = ", ".join(relevant_fields)
        return self.build(
            question=question,
            intent=intent,
            relevant_fields=fields_str,
            schema=schema
        )


class MongoDBPipelineValidator(dspy.Module):
    """Validate and fix MongoDB aggregation pipelines."""

    def __init__(self):
        super().__init__()
        self.validate = dspy.ChainOfThought(
            "pipeline_json, schema -> is_valid: bool, fixed_pipeline: str, errors: List[str]"
        )

    def forward(self, pipeline_json: str, schema: str) -> dspy.Prediction:
        return self.validate(pipeline_json=pipeline_json, schema=schema)


class MongoDBQueryGenerator(dspy.Module):
    """Complete MongoDB query generation pipeline using DSPy."""

    def __init__(self):
        super().__init__()
        self.intent_classifier = MongoDBQueryIntentClassifier()
        self.field_extractor = MongoDBFieldExtractor()
        self.aggregation_builder = MongoDBAggregationBuilder()
        self.validator = MongoDBPipelineValidator()

    def forward(self, question: str, schema: str = SCHEMA_FIELDS) -> dspy.Prediction:
        # Step 1: Classify intent
        intent_result = self.intent_classifier(question=question, schema=schema)

        # Step 2: Extract relevant fields
        fields_result = self.field_extractor(question=question, schema=schema)

        # Step 3: Build aggregation pipeline
        pipeline_result = self.aggregation_builder(
            question=question,
            intent=intent_result.intent,
            relevant_fields=fields_result.relevant_fields,
            schema=schema
        )

        # Step 4: Validate the pipeline
        validation_result = self.validator(
            pipeline_json=pipeline_result.pipeline_json,
            schema=schema
        )

        return dspy.Prediction(
            intent=intent_result.intent,
            reasoning=intent_result.reasoning,
            relevant_fields=fields_result.relevant_fields,
            pipeline_json=validation_result.fixed_pipeline if validation_result.is_valid else pipeline_result.pipeline_json,
            is_valid=validation_result.is_valid,
            errors=validation_result.errors if hasattr(validation_result, 'errors') else []
        )


class OptimizedMongoDBQueryGenerator(dspy.Module):
    """Optimized version of the query generator with compiled signatures."""

    def __init__(self, optimized_path: Optional[Path] = None):
        super().__init__()
        self.generator = MongoDBQueryGenerator()

        # Try to load optimized model if path exists
        if optimized_path and optimized_path.exists():
            try:
                self.generator = self.generator.load(str(optimized_path))
            except Exception:
                # Fall back to base model if loading fails
                pass

    def forward(self, question: str, schema: str = SCHEMA_FIELDS) -> dspy.Prediction:
        return self.generator(question=question, schema=schema)


def load_training_examples(json_path: Path) -> List[dspy.Example]:
    """Load training examples from JSON file (supports both original and processed formats)."""
    if not json_path.exists():
        return []

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        examples = []

        # Check if it's the processed DSPy format
        if 'examples' in data and data.get('examples', []):
            for item in data['examples']:
                example = dspy.Example(
                    question=item['question'],
                    schema=SCHEMA_FIELDS,
                    pipeline_json=item['pipeline_json'],
                    expected_type=item.get('expected_type', 'unknown')
                ).with_inputs("question", "schema")
                examples.append(example)
        # Check if it's the original query test results format
        elif 'results' in data and data.get('results', []):
            for item in data['results']:
                if item.get('success', False):
                    example = dspy.Example(
                        question=item['natural_language'],
                        schema=SCHEMA_FIELDS,
                        pipeline_json=json.dumps(item['pandas_query'])  # Simplified for now
                    ).with_inputs("question", "schema")
                    examples.append(example)

        return examples
    except Exception:
        return []


def train_query_generator(
    training_data_path: Path,
    save_path: Path,
    num_examples: int = 50
) -> MongoDBQueryGenerator:
    """Train and optimize the MongoDB query generator."""
    # Configure DSPy
    configure_dspy()

    # Load training examples
    examples = load_training_examples(training_data_path)
    if len(examples) < 10:
        raise ValueError(f"Need at least 10 training examples, got {len(examples)}")

    # Use a subset for training
    trainset = examples[:min(num_examples, len(examples))]

    # Define metric for optimization
    def query_quality_metric(example, prediction, trace=None):
        """Metric to evaluate query generation quality."""
        try:
            # Parse the generated pipeline
            pipeline = json.loads(prediction.pipeline_json)

            # Basic validation checks
            if not isinstance(pipeline, list):
                return 0.0

            if not pipeline:  # Empty pipeline
                return 0.0

            # Check for required stages in aggregation
            has_match = any('$match' in str(stage) for stage in pipeline)
            has_group = any('$group' in str(stage) for stage in pipeline)

            # Reward pipelines with proper aggregation structure
            score = 0.5  # Base score for valid JSON

            if has_match or has_group:
                score += 0.3

            # Check if pipeline follows common patterns
            if len(pipeline) > 1:
                score += 0.2

            return min(score, 1.0)

        except (json.JSONDecodeError, KeyError, TypeError):
            return 0.0

    # Create optimizer
    optimizer = BootstrapFewShot(metric=query_quality_metric)

    # Base generator
    generator = MongoDBQueryGenerator()

    # Optimize
    optimized_generator = optimizer.compile(generator, trainset=trainset)

    # Save optimized model
    save_path.parent.mkdir(parents=True, exist_ok=True)
    optimized_generator.save(str(save_path))

    return optimized_generator


def generate_mongodb_query_dspy(
    question: str,
    use_optimized: bool = True,
    max_attempts: int = 3
) -> Optional[List[Dict[str, Any]]]:
    """Generate MongoDB query using DSPy-based approach."""
    # Configure DSPy
    configure_dspy()

    # Get the generator
    settings = get_settings()
    if use_optimized and settings.dspy_optimized_model_path.exists():
        generator = OptimizedMongoDBQueryGenerator(settings.dspy_optimized_model_path)
    else:
        generator = MongoDBQueryGenerator()

    # Generate query with retries
    for attempt in range(max_attempts):
        try:
            result = generator(question=question)

            # Parse the pipeline
            pipeline = json.loads(result.pipeline_json)

            # Basic validation
            if isinstance(pipeline, list) and pipeline:
                return pipeline

        except (json.JSONDecodeError, AttributeError):
            continue

    return None
