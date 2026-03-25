"""Dynamic model router: escalate to stronger models on failure, fall back when idle.

Reads tier config from ``~/.nanobot/model_tiers.json`` and provider configs from
``~/.nanobot/config.json``.  Creates a provider instance per tier so each can
have its own api_key / api_base.

Usage in loop.py:
    provider, model = self.model_router.select(last_response)
    response = await provider.chat_with_retry(messages=..., model=model)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_ESCALATION = {
    "max_consecutive_errors": 2,
    "max_repeated_tool_calls": 3,
    "low_quality_patterns": [
        "我不确定", "我不知道", "I'm not sure", "I don't know",
        "I cannot", "无法完成", "无法处理",
    ],
}

_DEFAULT_COOLDOWN = 3  # iterations without issues → fall back


def _load_tiers_config(config_dir: Path) -> dict | None:
    """Load model_tiers.json from the config directory."""
    path = config_dir / "model_tiers.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load model_tiers.json: {}", e)
        return None


def _make_tier_provider(
    provider_name: str, model: str, config_dir: Path,
) -> tuple[LLMProvider, str]:
    """Create a provider instance for a single tier.

    Reads api_key / api_base from config.json's providers section.
    Returns (provider_instance, model_name).
    """
    # Load main config for provider credentials
    config_path = config_dir / "config.json"
    cfg: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)

    providers_cfg = cfg.get("providers", {})
    pcfg = providers_cfg.get(provider_name, {})
    # Support camelCase keys from config
    api_key = pcfg.get("api_key") or pcfg.get("apiKey") or ""
    api_base = pcfg.get("api_base") or pcfg.get("apiBase")
    extra_headers = pcfg.get("extra_headers") or pcfg.get("extraHeaders")

    if provider_name == "custom":
        from nanobot.providers.custom_provider import CustomProvider
        return CustomProvider(
            api_key=api_key or "no-key",
            api_base=api_base or "http://localhost:8000/v1",
            default_model=model,
            extra_headers=extra_headers,
        ), model
    else:
        from nanobot.providers.litellm_provider import LiteLLMProvider
        return LiteLLMProvider(
            api_key=api_key or None,
            api_base=api_base,
            default_model=model,
            extra_headers=extra_headers,
            provider_name=provider_name,
        ), model


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class ModelRouter:
    """State machine for dynamic model tier selection."""

    def __init__(
        self,
        default_provider: LLMProvider,
        default_model: str,
        config_dir: Path | None = None,
    ):
        self._default_provider = default_provider
        self._default_model = default_model
        self._config_dir = config_dir or (Path.home() / ".nanobot")

        # Tier state
        self._tiers: list[dict] = []          # [{name, provider_name, model}, ...]
        self._tier_providers: dict[str, tuple[LLMProvider, str]] = {}
        self._tier_names: list[str] = []      # ordered low → high
        self._default_tier: str = "default"
        self._current_tier: str = "default"

        # Escalation config
        self._escalation = dict(_DEFAULT_ESCALATION)
        self._cooldown_threshold = _DEFAULT_COOLDOWN

        # Counters
        self._consecutive_errors = 0
        self._tool_call_history: list[str] = []  # recent tool call signatures
        self._cooldown_counter = 0
        self._manually_set = False  # True if user used /model command

        self._enabled = False
        self._load_config()

    def _load_config(self) -> None:
        """Load tier configuration."""
        raw = _load_tiers_config(self._config_dir)
        if not raw or not raw.get("enabled", False):
            return

        tiers = raw.get("tiers", [])
        if len(tiers) < 2:
            logger.warning("model_tiers.json needs at least 2 tiers, disabling router")
            return

        self._tiers = tiers
        self._tier_names = [t["name"] for t in tiers]
        self._default_tier = raw.get("default_tier", tiers[0]["name"])
        self._current_tier = self._default_tier

        esc = raw.get("escalation", {})
        self._escalation.update(esc)
        self._cooldown_threshold = raw.get("cooldown_iterations", _DEFAULT_COOLDOWN)

        # Pre-create provider for the default tier (reuse existing provider)
        for tier in tiers:
            name = tier["name"]
            if name == self._default_tier:
                self._tier_providers[name] = (self._default_provider, tier["model"])

        self._enabled = True
        logger.info("Model router enabled: {} tiers, default='{}'",
                     len(tiers), self._default_tier)

    def _get_provider(self, tier_name: str) -> tuple[LLMProvider, str]:
        """Get or lazily create provider for a tier."""
        if tier_name in self._tier_providers:
            return self._tier_providers[tier_name]

        tier = next((t for t in self._tiers if t["name"] == tier_name), None)
        if not tier:
            return self._default_provider, self._default_model

        provider, model = _make_tier_provider(
            tier["provider"], tier["model"], self._config_dir,
        )
        self._tier_providers[tier_name] = (provider, model)
        return provider, model

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def current_tier(self) -> str:
        return self._current_tier

    @property
    def is_highest_tier(self) -> bool:
        if not self._tier_names:
            return True
        return self._current_tier == self._tier_names[-1]

    def _next_tier(self) -> str | None:
        """Return the next higher tier name, or None if at max."""
        if not self._tier_names:
            return None
        try:
            idx = self._tier_names.index(self._current_tier)
        except ValueError:
            return None
        if idx + 1 < len(self._tier_names):
            return self._tier_names[idx + 1]
        return None

    def select(
        self, last_response: LLMResponse | None = None,
    ) -> tuple[LLMProvider, str]:
        """Select provider + model for the next LLM call.

        Call this before each chat_with_retry().
        """
        if not self._enabled:
            return self._default_provider, self._default_model

        tier_before = self._current_tier
        if last_response is not None:
            self._check_escalation(last_response)

        provider, model = self._get_provider(self._current_tier)
        if self._current_tier != tier_before:
            logger.info("Model router: {} → {} (model: {})", tier_before, self._current_tier, model)
        return provider, model

    def _check_escalation(self, response: LLMResponse) -> None:
        """Check if we need to escalate based on the last response."""
        if self._manually_set:
            return  # user explicitly chose a tier, don't auto-switch

        escalated = False

        # 1. Error check
        if response.finish_reason == "error":
            self._consecutive_errors += 1
            max_err = self._escalation.get("max_consecutive_errors", 2)
            if self._consecutive_errors >= max_err:
                escalated = self._escalate()
        else:
            self._consecutive_errors = 0

        # 2. Repeated tool calls
        if response.has_tool_calls and not escalated:
            sig = "|".join(
                f"{tc.name}:{sorted(tc.arguments.items())}"
                for tc in response.tool_calls
            )
            self._tool_call_history.append(sig)
            max_repeat = self._escalation.get("max_repeated_tool_calls", 3)
            if len(self._tool_call_history) >= max_repeat:
                recent = self._tool_call_history[-max_repeat:]
                if len(set(recent)) == 1:
                    logger.warning("Repeated tool call detected ({} times), escalating", max_repeat)
                    escalated = self._escalate()
                    self._tool_call_history.clear()

        # 3. Low quality response
        if (response.content and not response.has_tool_calls
                and not escalated and response.finish_reason != "error"):
            patterns = self._escalation.get("low_quality_patterns", [])
            content_lower = response.content.lower()
            if any(p.lower() in content_lower for p in patterns):
                logger.warning("Low quality response detected, escalating")
                escalated = self._escalate()

        # 4. Cooldown: if on a higher tier and no issues, count down
        if not escalated and self._current_tier != self._default_tier:
            if response.finish_reason != "error" and not response.has_tool_calls:
                self._cooldown_counter += 1
                if self._cooldown_counter >= self._cooldown_threshold:
                    self._deescalate()

    def _escalate(self) -> bool:
        """Move to the next higher tier. Returns True if escalated."""
        next_tier = self._next_tier()
        if next_tier is None:
            return False
        old = self._current_tier
        self._current_tier = next_tier
        self._consecutive_errors = 0
        self._cooldown_counter = 0
        provider, model = self._get_provider(next_tier)
        logger.info("Model escalated: {} → {} (model: {})", old, next_tier, model)
        return True

    def _deescalate(self) -> None:
        """Fall back to the default tier."""
        if self._current_tier == self._default_tier:
            return
        old = self._current_tier
        self._current_tier = self._default_tier
        self._cooldown_counter = 0
        self._consecutive_errors = 0
        self._manually_set = False
        provider, model = self._get_provider(self._default_tier)
        logger.info("Model de-escalated: {} → {} (model: {})", old, self._default_tier, model)

    def notify_task_complete(self) -> None:
        """Called when a task finishes (final response). Reset to default."""
        if not self._enabled:
            return
        if self._manually_set:
            return  # respect manual override
        if self._current_tier != self._default_tier:
            self._deescalate()
        self._consecutive_errors = 0
        self._tool_call_history.clear()
        self._cooldown_counter = 0

    # ----- Manual switching (/model command) -----

    def set_tier(self, tier_name: str) -> str:
        """Manually switch to a tier. Returns status message."""
        if not self._enabled:
            return "Model router is not enabled. Create ~/.nanobot/model_tiers.json to enable."

        if tier_name not in self._tier_names:
            available = ", ".join(self._tier_names)
            return f"Unknown tier '{tier_name}'. Available: {available}"

        old = self._current_tier
        self._current_tier = tier_name
        self._manually_set = True
        self._consecutive_errors = 0
        self._cooldown_counter = 0
        self._tool_call_history.clear()
        provider, model = self._get_provider(tier_name)
        logger.info("Model manually set: {} → {} (model: {})", old, tier_name, model)
        return f"Switched to '{tier_name}' (model: {model})"

    def get_status(self) -> str:
        """Return current router status for /model command."""
        if not self._enabled:
            return "Model router is not enabled. Create ~/.nanobot/model_tiers.json to enable."

        lines = [f"Current tier: {self._current_tier}"]
        if self._manually_set:
            lines.append("(manually set)")
        lines.append("")
        for tier in self._tiers:
            marker = " ← current" if tier["name"] == self._current_tier else ""
            lines.append(f"  {tier['name']}: {tier['model']}{marker}")
        return "\n".join(lines)
