"""Tests for model_router.py - dynamic model tier selection."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from nanobot.agent.model_router import (
    ModelRouter,
    _load_tiers_config,
    _DEFAULT_ESCALATION,
)
from nanobot.providers.base import LLMResponse, ToolCallRequest


class TestModelRouterInit:
    """Test ModelRouter initialization and disabled mode."""

    def test_disabled_when_no_config(self, tmp_path):
        """Router should be disabled when no model_tiers.json exists."""
        router = ModelRouter(
            default_provider=MagicMock(),
            default_model="test-model",
            config_dir=tmp_path,
        )
        assert not router.enabled
        provider, model = router.select()
        assert provider is router._default_provider
        assert model == "test-model"

    def test_disabled_when_enabled_false(self, tmp_path):
        """Router should be disabled when config has enabled=false."""
        config = {"enabled": False, "tiers": []}
        (tmp_path / "model_tiers.json").write_text(json.dumps(config))

        router = ModelRouter(
            default_provider=MagicMock(),
            default_model="test-model",
            config_dir=tmp_path,
        )
        assert not router.enabled

    def test_enabled_with_valid_config(self, tmp_path):
        """Router should be enabled with valid config and 2+ tiers."""
        config = {
            "enabled": True,
            "tiers": [
                {"name": "fast", "provider": "custom", "model": "gpt-4o"},
                {"name": "smart", "provider": "custom", "model": "gpt-4o-mini"},
            ],
            "default_tier": "fast",
        }
        (tmp_path / "model_tiers.json").write_text(json.dumps(config))

        router = ModelRouter(
            default_provider=MagicMock(),
            default_model="gpt-4o",
            config_dir=tmp_path,
        )
        assert router.enabled
        assert router.current_tier == "fast"


class TestModelRouterSelect:
    """Test provider/model selection logic."""

    def test_select_returns_default_when_disabled(self, tmp_path):
        """When disabled, should return default provider and model."""
        default_provider = MagicMock()
        router = ModelRouter(
            default_provider=default_provider,
            default_model="default-model",
            config_dir=tmp_path,
        )

        provider, model = router.select()

        assert provider is default_provider
        assert model == "default-model"

    def test_select_returns_current_tier_provider(self, tmp_path):
        """When enabled, should return provider for current tier."""
        config = {
            "enabled": True,
            "tiers": [
                {"name": "tier1", "provider": "custom", "model": "model-1"},
                {"name": "tier2", "provider": "custom", "model": "model-2"},
            ],
            "default_tier": "tier1",
        }
        (tmp_path / "model_tiers.json").write_text(json.dumps(config))

        default_provider = MagicMock()
        router = ModelRouter(
            default_provider=default_provider,
            default_model="fallback",
            config_dir=tmp_path,
        )

        provider, model = router.select()
        assert model == "model-1"

    def test_select_with_none_response(self, tmp_path):
        """select(None) should not crash - first iteration case."""
        config = {
            "enabled": True,
            "tiers": [
                {"name": "tier1", "provider": "custom", "model": "model-1"},
                {"name": "tier2", "provider": "custom", "model": "model-2"},
            ],
            "default_tier": "tier1",
        }
        (tmp_path / "model_tiers.json").write_text(json.dumps(config))

        default_provider = MagicMock()
        router = ModelRouter(
            default_provider=default_provider,
            default_model="fallback",
            config_dir=tmp_path,
        )

        # This should not raise
        provider, model = router.select(None)
        assert model == "model-1"


class TestModelRouterEscalation:
    """Test escalation logic on errors and repeated tool calls."""

    def test_escalate_on_consecutive_errors(self, tmp_path):
        """Should escalate after max_consecutive_errors."""
        config = {
            "enabled": True,
            "tiers": [
                {"name": "tier1", "provider": "custom", "model": "model-1"},
                {"name": "tier2", "provider": "custom", "model": "model-2"},
            ],
            "default_tier": "tier1",
            "escalation": {"max_consecutive_errors": 2},
        }
        (tmp_path / "model_tiers.json").write_text(json.dumps(config))

        default_provider = MagicMock()
        router = ModelRouter(
            default_provider=default_provider,
            default_model="model-1",
            config_dir=tmp_path,
        )

        # First error
        router.select(LLMResponse(content="", finish_reason="error"))
        assert router.current_tier == "tier1"

        # Second error -> escalate
        router.select(LLMResponse(content="", finish_reason="error"))
        assert router.current_tier == "tier2"

    def test_escalate_on_repeated_tool_calls(self, tmp_path):
        """Should escalate when same tool call repeats max_repeated_tool_calls times."""
        config = {
            "enabled": True,
            "tiers": [
                {"name": "tier1", "provider": "custom", "model": "model-1"},
                {"name": "tier2", "provider": "custom", "model": "model-2"},
            ],
            "default_tier": "tier1",
            "escalation": {"max_repeated_tool_calls": 3},
        }
        (tmp_path / "model_tiers.json").write_text(json.dumps(config))

        default_provider = MagicMock()
        router = ModelRouter(
            default_provider=default_provider,
            default_model="model-1",
            config_dir=tmp_path,
        )

        tool_call = ToolCallRequest(
            id="1",
            name="search",
            arguments={"query": "test"},
        )
        response = LLMResponse(
            content="",
            tool_calls=[tool_call],
            finish_reason="stop",
        )

        # Same tool call 3 times
        for _ in range(3):
            router.select(response)

        assert router.current_tier == "tier2"

    def test_no_escalate_on_manual_override(self, tmp_path):
        """Should not auto-escalate when user manually set tier."""
        config = {
            "enabled": True,
            "tiers": [
                {"name": "tier1", "provider": "custom", "model": "model-1"},
                {"name": "tier2", "provider": "custom", "model": "model-2"},
            ],
            "default_tier": "tier1",
        }
        (tmp_path / "model_tiers.json").write_text(json.dumps(config))

        default_provider = MagicMock()
        router = ModelRouter(
            default_provider=default_provider,
            default_model="model-1",
            config_dir=tmp_path,
        )

        # Manually set tier
        router.set_tier("tier2")

        # Errors should not cause escalation
        for _ in range(3):
            router.select(LLMResponse(content="", finish_reason="error"))

        assert router.current_tier == "tier2"

    def test_deescalate_after_cooldown(self, tmp_path):
        """Should fall back after cooldown iterations without issues."""
        config = {
            "enabled": True,
            "tiers": [
                {"name": "tier1", "provider": "custom", "model": "model-1"},
                {"name": "tier2", "provider": "custom", "model": "model-2"},
            ],
            "default_tier": "tier1",
            "cooldown_iterations": 2,
        }
        (tmp_path / "model_tiers.json").write_text(json.dumps(config))

        default_provider = MagicMock()
        router = ModelRouter(
            default_provider=default_provider,
            default_model="model-1",
            config_dir=tmp_path,
        )

        # Escalate first
        router._current_tier = "tier2"

        # Successful responses without tool calls
        for _ in range(2):
            router.select(LLMResponse(content="ok", finish_reason="stop"))

        assert router.current_tier == "tier1"


class TestModelRouterManualSwitch:
    """Test /model command functionality."""

    def test_set_tier_valid(self, tmp_path):
        """set_tier with valid tier name should switch."""
        config = {
            "enabled": True,
            "tiers": [
                {"name": "tier1", "provider": "custom", "model": "model-1"},
                {"name": "tier2", "provider": "custom", "model": "model-2"},
            ],
            "default_tier": "tier1",
        }
        (tmp_path / "model_tiers.json").write_text(json.dumps(config))

        default_provider = MagicMock()
        router = ModelRouter(
            default_provider=default_provider,
            default_model="model-1",
            config_dir=tmp_path,
        )

        result = router.set_tier("tier2")
        assert "tier2" in result
        assert router.current_tier == "tier2"

    def test_set_tier_invalid(self, tmp_path):
        """set_tier with invalid tier should return error."""
        config = {
            "enabled": True,
            "tiers": [
                {"name": "tier1", "provider": "custom", "model": "model-1"},
            ],
            "default_tier": "tier1",
        }
        (tmp_path / "model_tiers.json").write_text(json.dumps(config))

        router = ModelRouter(
            default_provider=MagicMock(),
            default_model="model-1",
            config_dir=tmp_path,
        )

        result = router.set_tier("nonexistent")
        assert "Unknown tier" in result

    def test_get_status(self, tmp_path):
        """get_status should return formatted tier info."""
        config = {
            "enabled": True,
            "tiers": [
                {"name": "fast", "provider": "custom", "model": "gpt-4o"},
                {"name": "smart", "provider": "custom", "model": "gpt-4o-mini"},
            ],
            "default_tier": "fast",
        }
        (tmp_path / "model_tiers.json").write_text(json.dumps(config))

        router = ModelRouter(
            default_provider=MagicMock(),
            default_model="gpt-4o",
            config_dir=tmp_path,
        )

        status = router.get_status()
        assert "Current tier: fast" in status
        assert "fast:" in status
        assert "smart:" in status

    def test_notify_task_complete_resets_to_default(self, tmp_path):
        """notify_task_complete should reset tier unless manually set."""
        config = {
            "enabled": True,
            "tiers": [
                {"name": "tier1", "provider": "custom", "model": "model-1"},
                {"name": "tier2", "provider": "custom", "model": "model-2"},
            ],
            "default_tier": "tier1",
        }
        (tmp_path / "model_tiers.json").write_text(json.dumps(config))

        router = ModelRouter(
            default_provider=MagicMock(),
            default_model="model-1",
            config_dir=tmp_path,
        )

        # Escalate
        router._current_tier = "tier2"

        # Task complete -> should reset
        router.notify_task_complete()
        assert router.current_tier == "tier1"

    def test_notify_task_complete_respects_manual(self, tmp_path):
        """notify_task_complete should NOT reset if manually set."""
        config = {
            "enabled": True,
            "tiers": [
                {"name": "tier1", "provider": "custom", "model": "model-1"},
                {"name": "tier2", "provider": "custom", "model": "model-2"},
            ],
            "default_tier": "tier1",
        }
        (tmp_path / "model_tiers.json").write_text(json.dumps(config))

        router = ModelRouter(
            default_provider=MagicMock(),
            default_model="model-1",
            config_dir=tmp_path,
        )

        router.set_tier("tier2")
        router.notify_task_complete()
        assert router.current_tier == "tier2"


class TestLoadTiersConfig:
    """Test _load_tiers_config function."""

    def test_load_valid_config(self, tmp_path):
        """Should load and parse valid config file."""
        config = {"enabled": True, "tiers": []}
        (tmp_path / "model_tiers.json").write_text(json.dumps(config))

        result = _load_tiers_config(tmp_path)
        assert result == config

    def test_load_missing_file(self, tmp_path):
        """Should return None when file doesn't exist."""
        result = _load_tiers_config(tmp_path)
        assert result is None

    def test_load_invalid_json(self, tmp_path):
        """Should return None and log warning for invalid JSON."""
        (tmp_path / "model_tiers.json").write_text("not json")

        result = _load_tiers_config(tmp_path)
        assert result is None
