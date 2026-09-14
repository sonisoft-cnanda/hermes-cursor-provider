"""Hermes public model-provider profile registration."""

from __future__ import annotations

from typing import Any

DEFAULT_BRIDGE_BASE_URL = "http://127.0.0.1:8765/v1"
FALLBACK_MODELS = (
    "auto",
    "cursor-grok-4.6-high",
    "cursor-grok-4.6-high-fast",
    "cursor-grok-4.6-medium",
    "cursor-grok-4.6-xhigh",
    "composer-2.5",
    "composer-2.5-fast",
    "composer-2",
    "composer-2-fast",
)


def register_cursor_provider() -> Any:
    """Register Cursor through Hermes's standard declarative plugin API."""
    from providers import register_provider
    from providers.base import ProviderProfile

    profile = ProviderProfile(
        name="cursor",
        aliases=("cursor-agent", "cursor-cli", "cursor-sub", "cursor-subscription"),
        display_name="Cursor",
        description="Hermes-controlled inference through a local Cursor CLI bridge",
        signup_url="https://cursor.com/dashboard/integrations",
        api_mode="chat_completions",
        env_vars=("CURSOR_BRIDGE_API_KEY",),
        base_url=DEFAULT_BRIDGE_BASE_URL,
        auth_type="api_key",
        fallback_models=FALLBACK_MODELS,
        supports_health_check=True,
    )
    register_provider(profile)
    return profile
