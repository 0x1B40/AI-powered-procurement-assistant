# DSPy Integration Guide

This guide explains how to use DSPy (Declarative Self-improving Python) to enhance your MongoDB query generation system.

## Overview

DSPy replaces manual prompt engineering with automatic prompt optimization and structured program decomposition. Instead of crafting lengthy prompts with examples, DSPy learns optimal prompts from your data and can improve over time.

## Key Benefits

1. **Automatic Optimization**: DSPy optimizes prompts using your training data
2. **Structured Programs**: Break complex tasks into modular, optimizable components
3. **Multi-Provider Support**: Works with OpenAI, Grok, Anthropic, Google, and others
4. **Continuous Learning**: Models can be retrained as you collect more data
5. **Better Reliability**: Systematic approach reduces hallucinations and errors

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Update your `.env` file with DSPy settings:

```bash
# LLM Provider (choose one)
LLM_PROVIDER=grok  # or openai, anthropic, google

# Enable DSPy for query generation
USE_DSPY_FOR_QUERIES=true

# DSPy-specific paths
DSPY_MODEL_CACHE_DIR=data/dspy_cache
DSPY_TRAINING_DATA_PATH=data/dspy_training_data.json
DSPY_OPTIMIZED_MODEL_PATH=models/dspy_optimized
```

### 3. Prepare Training Data

Convert your existing query test results to DSPy format:

```bash
python scripts/prepare_dspy_training_data.py
```

This creates `data/dspy_training_data.json` from your `query_test_results.json`.

### 4. Train the Model

Train and optimize the DSPy model:

```bash
python scripts/train_dspy_model.py
```

This creates an optimized model saved to `models/dspy_optimized/`.

## Architecture

The DSPy integration consists of several modules:

### MongoDBQueryIntentClassifier
Classifies the intent of user questions (count, aggregate, filter, etc.)

### MongoDBFieldExtractor
Identifies relevant fields from the question and schema.

### MongoDBAggregationBuilder
Constructs MongoDB aggregation pipelines from intent and fields.

### MongoDBPipelineValidator
Validates and fixes generated pipelines.

### MongoDBQueryGenerator
Orchestrates the entire pipeline with proper error handling.

## Usage

### Basic Usage

Once trained, DSPy is used automatically when `USE_DSPY_FOR_QUERIES=true`:

```python
from src.query_and_response.dspy_query_generation import generate_mongodb_query_dspy

# Generate a query using DSPy
pipeline = generate_mongodb_query_dspy("How many orders were placed in 2014?")
if pipeline:
    result = execute_mongodb_query(json.dumps(pipeline))
```

### Advanced Usage

For custom DSPy programs:

```python
from src.query_and_response.dspy_query_generation import MongoDBQueryGenerator
from src.llm_and_classification.llm import configure_dspy

# Configure DSPy
configure_dspy()

# Create generator
generator = MongoDBQueryGenerator()

# Generate with full prediction details
result = generator(question="Top 5 departments by spend?")
print(f"Intent: {result.intent}")
print(f"Fields: {result.relevant_fields}")
print(f"Pipeline: {result.pipeline_json}")
print(f"Valid: {result.is_valid}")
```

## Training and Optimization

### Training Process

1. **Data Preparation**: Convert existing query logs to DSPy format
2. **Metric Definition**: Define quality metrics (pipeline validity, execution success)
3. **Optimization**: Use BootstrapFewShot to optimize prompts
4. **Validation**: Test on held-out data
5. **Deployment**: Save and load optimized models

### Custom Metrics

You can define custom evaluation metrics:

```python
def custom_metric(example, prediction, trace=None):
    """Custom metric for evaluating query quality."""
    try:
        pipeline = json.loads(prediction.pipeline_json)
        # Your evaluation logic here
        score = evaluate_pipeline_quality(pipeline, example.expected_result)
        return score
    except:
        return 0.0
```

### Retraining

To improve the model over time:

```python
# Collect new examples from user interactions
new_examples = collect_user_query_examples()

# Retrain with combined data
optimizer = BootstrapFewShot(metric=custom_metric)
improved_model = optimizer.compile(
    MongoDBQueryGenerator(),
    trainset=existing_examples + new_examples
)
improved_model.save("models/dspy_improved")
```

## LLM Provider Support

DSPy supports multiple providers through a unified interface:

### Grok (xAI)
```bash
LLM_PROVIDER=grok
PRIMARY_LLM_API_KEY=your-xai-key
PRIMARY_LLM_MODEL=grok-beta
```

### OpenAI
```bash
LLM_PROVIDER=openai
PRIMARY_LLM_API_KEY=your-openai-key
PRIMARY_LLM_MODEL=gpt-4
```

### Anthropic
```bash
LLM_PROVIDER=anthropic
PRIMARY_LLM_API_KEY=your-anthropic-key
PRIMARY_LLM_MODEL=claude-3-sonnet-20240229
```

## Troubleshooting

### Common Issues

1. **Training fails**: Ensure you have enough training examples (minimum 10)
2. **Poor performance**: Check your metric function and training data quality
3. **Provider errors**: Verify API keys and model availability
4. **Memory issues**: Reduce batch size or use smaller models

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# DSPy debug mode
dspy.settings.configure(log_level="DEBUG")
```

### Performance Monitoring

Track model performance over time:

```python
# Log predictions and user feedback
def log_prediction(question, prediction, user_rating):
    with open("data/prediction_logs.jsonl", "a") as f:
        json.dump({
            "question": question,
            "prediction": prediction,
            "rating": user_rating,
            "timestamp": datetime.now().isoformat()
        }, f)
```

## Migration from Prompt Engineering

### Before (Manual Prompts)
- Long, complex prompts with hardcoded examples
- Manual tuning based on trial and error
- No systematic improvement
- Limited to single LLM provider

### After (DSPy)
- Structured, modular programs
- Automatic optimization from data
- Continuous learning and improvement
- Multi-provider support with unified interface

## Best Practices

1. **Start Small**: Begin with a subset of your query types
2. **Quality Data**: Ensure training examples are high-quality and diverse
3. **Iterate**: Regularly retrain with new data and user feedback
4. **Monitor**: Track performance metrics and user satisfaction
5. **Fallback**: Keep the original system as a fallback option

## Next Steps

1. Experiment with different LLM providers
2. Collect more diverse training examples
3. Implement user feedback collection
4. Set up automated retraining pipelines
5. A/B test DSPy vs. traditional approaches
