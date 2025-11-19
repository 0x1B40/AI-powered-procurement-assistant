from __future__ import annotations

from functools import lru_cache
from typing import Dict

from langchain.agents import AgentExecutor
from langchain_community.agent_toolkits.mongo.toolkit import MongoDBToolkit
from langchain_community.agent_toolkits import create_mongo_agent
from langchain_openai import ChatOpenAI
from pymongo import MongoClient

from .config import get_settings


@lru_cache
def _build_agent() -> AgentExecutor:
    settings = get_settings()
    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=settings.openai_temperature,
    )

    client = MongoClient(settings.mongodb_uri)
    database = client[settings.mongodb_db]
    toolkit = MongoDBToolkit(
        mongo_client=client,
        database_name=settings.mongodb_db,
        collection_name=settings.mongodb_collection,
    )

    return create_mongo_agent(llm=llm, toolkit=toolkit, verbose=True)


def chat(question: str, context: Dict | None = None) -> str:
    """Generate a MongoDB-grounded answer for a procurement question."""
    agent = _build_agent()
    payload = context or {}
    payload["input"] = question
    result = agent.invoke(payload)
    return result["output"]

