from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from src.agent import analyze_question


def _build_state(question: str):
    return {
        "messages": [HumanMessage(content=question)],
        "mongodb_results": [],
        "final_answer": "",
    }


@patch("src.agent.execute_mongodb_query")
@patch("src.agent._get_llm")
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


@patch("src.agent.execute_mongodb_query")
@patch("src.agent._get_llm")
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

