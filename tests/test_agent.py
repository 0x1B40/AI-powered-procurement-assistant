import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from src.config_and_constants.constants import QuestionCategory
from src.core.workflow import analyze_question
from src.llm_and_classification.classification import categorize_question


def _build_state(question: str):
    return {
        "messages": [HumanMessage(content=question)],
        "mongodb_results": [],
        "final_answer": "",
        "question_category": QuestionCategory.QUERY_GENERATION.value,
        "classification_confidence": 1.0,
    }


def _mock_classifier_llm(category_value: str, confidence: float = 0.91):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content=json.dumps({"category": category_value, "confidence": confidence})
    )
    return mock_llm


@patch("src.database.database.execute_mongodb_query")
@patch("src.llm_and_classification.llm.get_llm")
def test_analyze_question_retries_on_invalid_json(mock_get_llm, mock_execute_query):
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        MagicMock(
            content='[{"$match": {"fiscal_year": "2013-2014"}} {"$group": {"_id": null, "total": {"$sum": "$total_price"}}}]'
        ),
        MagicMock(
            content='[{"$match": {"fiscal_year": "2013-2014"}}, {"$group": {"_id": null, "total_spend": {"$sum": "$total_price"}}}]'
        ),
    ]
    mock_get_llm.return_value = mock_llm
    mock_execute_query.return_value = '[{"total_spend": 123}]'

    result = analyze_question(_build_state("Total spend in FY 2013-2014"))

    assert mock_llm.invoke.call_count == 2
    assert mock_execute_query.called

    parsed_results = result["mongodb_results"][0]
    assert isinstance(parsed_results, list)
    assert parsed_results[0]["total_spend"] == 123
    assert result["final_answer"] == '[{"total_spend": 123}]'


@patch("src.database.database.execute_mongodb_query")
@patch("src.llm_and_classification.llm.get_llm")
def test_analyze_question_returns_error_after_max_retries(mock_get_llm, mock_execute_query):
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        MagicMock(content='{"not": "a list"}'),
        MagicMock(content="[]"),
        MagicMock(content="not json at all"),
    ]
    mock_get_llm.return_value = mock_llm

    result = analyze_question(_build_state("Give me something"))

    assert mock_llm.invoke.call_count == 3
    mock_execute_query.assert_not_called()
    error_entry = result["mongodb_results"][0]
    assert "error" in error_entry
    assert "Invalid JSON" in error_entry["error"] or "Invalid" in error_entry["error"]


def test_categorize_question_detects_query_generation():
    mock_llm = _mock_classifier_llm("query_generation", 0.77)
    category, confidence = categorize_question(
        "Show the total spend by department in FY 2014-2015", llm=mock_llm
    )
    assert category == QuestionCategory.QUERY_GENERATION
    mock_llm.invoke.assert_called_once()
    assert confidence == 0.77


def test_categorize_question_detects_acquisition_methods():
    mock_llm = _mock_classifier_llm("acquisition_methods", 0.65)
    category, confidence = categorize_question(
        "Explain the acquisition methods used in the dataset", llm=mock_llm
    )
    assert category == QuestionCategory.ACQUISITION_METHODS
    mock_llm.invoke.assert_called_once()
    assert confidence == 0.65


def test_categorize_question_detects_out_of_scope():
    mock_llm = _mock_classifier_llm("out_of_scope", 0.85)
    category, confidence = categorize_question("Hello there, how's your day?", llm=mock_llm)
    assert category == QuestionCategory.OUT_OF_SCOPE
    mock_llm.invoke.assert_called_once()
    assert confidence == 0.85


def test_categorize_question_handles_empty_input():
    mock_llm = MagicMock()
    category, confidence = categorize_question("", llm=mock_llm)
    assert category == QuestionCategory.OUT_OF_SCOPE
    mock_llm.invoke.assert_not_called()
    assert confidence is None

