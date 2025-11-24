"""LLM initialization and interaction utilities."""

import os
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

import dspy
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


@lru_cache
def get_dspy_lm():
    """Get configured DSPy language model with caching."""
    settings = get_settings()

    if settings.llm_provider.lower() == "openai":
        lm = dspy.OpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            temperature=settings.llm_temperature,
        )
    elif settings.llm_provider.lower() == "grok":
        # Use Grok via xAI
        lm = dspy.Grok(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            temperature=settings.llm_temperature,
        )
    elif settings.llm_provider.lower() == "anthropic":
        lm = dspy.Claude(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            temperature=settings.llm_temperature,
        )
    elif settings.llm_provider.lower() == "google":
        lm = dspy.Google(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            temperature=settings.llm_temperature,
        )
    else:
        # Default to OpenAI as fallback
        lm = dspy.OpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            temperature=settings.llm_temperature,
        )

    return lm


def configure_dspy():
    """Configure DSPy with the appropriate language model."""
    lm = get_dspy_lm()
    dspy.settings.configure(lm=lm)

    # Set up model caching if configured
    settings = get_settings()
    if settings.dspy_model_cache_dir:
        os.makedirs(settings.dspy_model_cache_dir, exist_ok=True)
        # DSPy handles caching automatically, but we ensure the directory exists
