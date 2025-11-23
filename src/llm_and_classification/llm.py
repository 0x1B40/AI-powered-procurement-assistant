"""LLM initialization and interaction utilities."""

from functools import lru_cache
from typing import TYPE_CHECKING, Optional

from ..config_and_constants.config import get_settings

# Importing ChatOpenAI eagerly causes SSL context creation, which can fail inside
# restricted CI sandboxes. We import it lazily in _get_llm(), but still want type
# checkers to know about the symbol—hence the TYPE_CHECKING guard.
if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


@lru_cache
def _get_llm():
    """Get configured LLM with caching."""
    from langchain_openai import ChatOpenAI  # Local import avoids SSL context issues during module import in restricted environments
    settings = get_settings()
    kwargs = {
        "api_key": settings.llm_api_key,
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
    }
    if settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url
    return ChatOpenAI(**kwargs)


def get_llm():
    """Get the configured LLM instance."""
    return _get_llm()
