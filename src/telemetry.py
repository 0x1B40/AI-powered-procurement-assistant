"""LangSmith tracing helpers that stay no-op when tracing is disabled."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, TypeVar

# Tag every run so LangSmith dashboards can filter for this app easily.
DEFAULT_TAGS = ["procurement-agent", "langgraph"]

try:  # pragma: no cover - optional dependency guard
    # Only import LangSmith when the SDK is installed; otherwise tracing stays idle.
    from langsmith.run_helpers import (
        get_current_run_tree,
        traceable as _traceable,
    )
except ImportError:  # pragma: no cover - LangSmith may be optional in some envs
    _traceable = None
    get_current_run_tree = None

F = TypeVar("F", bound=Callable[..., Any])


def _merge_tags(tags: Optional[list[str]]) -> list[str]:
    """Combine default tags with user-provided ones without duplicates."""
    merged = list(DEFAULT_TAGS)
    if tags:
        for tag in tags:
            if tag not in merged:
                merged.append(tag)
    return merged


def traceable_step(
    *,
    name: str,
    run_type: str = "chain",
    tags: Optional[list[str]] = None,
):
    """Return a decorator that wraps a function with LangSmith tracing metadata."""

    if _traceable is None:
        # LangSmith is not installed; fall back to a passthrough decorator.
        def decorator(func: F) -> F:
            return func

        return decorator

    return _traceable(name=name, run_type=run_type, tags=_merge_tags(tags))


def log_child_run(
    name: str,
    *,
    inputs: Optional[Dict[str, Any]] = None,
    outputs: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    run_type: str = "chain",
    tags: Optional[list[str]] = None,
    error: Optional[str] = None,
) -> None:
    """Create a child run under the current trace to record intermediate data."""

    if get_current_run_tree is None:
        return

    parent = get_current_run_tree()
    if parent is None:
        return

    child = parent.create_child(
        name=name,
        run_type=run_type,
        inputs=inputs or {},
        tags=_merge_tags(tags),
    )
    if metadata:
        child.add_metadata(metadata)

    child.end(outputs=outputs, error=error)

