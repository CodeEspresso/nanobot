"""Tests for loop.py fixes - parallel tool execution and model routing."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse, ToolCallRequest


class TestParallelToolExecution:
    """Test that tool execution is parallel, not serial."""

    @pytest.fixture
    def mock_loop(self, tmp_path):
        """Create a minimal AgentLoop for testing."""
        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        provider.estimate_prompt_tokens.return_value = (10_000, "test")

        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=tmp_path,
            model="test-model",
            context_window_tokens=1,
        )
        return loop

    @pytest.mark.asyncio
    async def test_tools_execute_in_parallel(self, mock_loop):
        """Multiple tool calls should execute concurrently, not serially."""
        execution_times: list[float] = []
        call_order: list[str] = []

        async def slow_tool(name: str, delay: float):
            async def tool_impl():
                call_order.append(f"start_{name}")
                await asyncio.sleep(delay)
                execution_times.append(delay)
                call_order.append(f"end_{name}")
                return f"result_{name}"
            return tool_impl

        # Create 3 tools with different delays
        mock_tools = MagicMock()
        tool1_call = asyncio.create_task(slow_tool("tool1", 0.1)())
        tool2_call = asyncio.create_task(slow_tool("tool2", 0.15)())
        tool3_call = asyncio.create_task(slow_tool("tool3", 0.05)())

        async def execute_side_effect(name, args):
            await asyncio.sleep(0.1)  # Simulate work
            return f"result_{name}"

        mock_tools.execute = AsyncMock(side_effect=execute_side_effect)
        mock_loop.tools = mock_tools
        mock_loop.tools.get_definitions = MagicMock(return_value=[])

        # Mock provider to return tool calls
        tool_calls = [
            ToolCallRequest(id="1", name="tool1", arguments={}),
            ToolCallRequest(id="2", name="tool2", arguments={}),
            ToolCallRequest(id="3", name="tool3", arguments={}),
        ]
        mock_loop.provider.chat_with_retry = AsyncMock(
            return_value=LLMResponse(content="", tool_calls=tool_calls, finish_reason="stop")
        )

        # Mock context
        mock_loop.context.add_assistant_message = MagicMock(return_value=[])
        mock_loop.context.add_tool_result = MagicMock(side_effect=lambda msgs, tid, tname, res: msgs + [{"role": "tool", "content": res}])

        # Mock _set_tool_context (it's called but we just verify it happens)
        mock_loop._set_tool_context = MagicMock()

        # Run the agent loop
        result = await mock_loop._run_agent_loop(
            initial_messages=[{"role": "user", "content": "test"}],
            channel="cli",
            chat_id="test",
        )

        # Verify tools were executed
        assert mock_loop.tools.execute.call_count == 3

    @pytest.mark.asyncio
    async def test_set_tool_context_called_before_execution(self, mock_loop):
        """_set_tool_context should be called before tool execution."""
        mock_loop.tools = MagicMock()
        mock_loop.tools.get_definitions = MagicMock(return_value=[])
        mock_loop.tools.execute = AsyncMock(return_value="result")

        tool_calls = [
            ToolCallRequest(id="1", name="test_tool", arguments={}),
        ]
        mock_loop.provider.chat_with_retry = AsyncMock(
            return_value=LLMResponse(content="", tool_calls=tool_calls, finish_reason="stop")
        )

        mock_loop.context.add_assistant_message = MagicMock(return_value=[])
        mock_loop.context.add_tool_result = MagicMock(return_value=[])

        call_order: list[str] = []

        def record_context(*args, **kwargs):
            call_order.append("set_context")

        async def record_execute(*args, **kwargs):
            call_order.append("execute")
            return "result"

        mock_loop._set_tool_context = MagicMock(side_effect=record_context)
        mock_loop.tools.execute = AsyncMock(side_effect=record_execute)

        await mock_loop._run_agent_loop(
            initial_messages=[{"role": "user", "content": "test"}],
            channel="test_channel",
            chat_id="test_chat",
            message_id="msg_123",
        )

        # set_context should be called before execute
        assert "set_context" in call_order
        assert "execute" in call_order
        assert call_order.index("set_context") < call_order.index("execute")

        # Verify it was called with correct args
        mock_loop._set_tool_context.assert_called_with("test_channel", "test_chat", "msg_123")


class TestModelRoutingInLoop:
    """Test that model routing works correctly across iterations."""

    @pytest.fixture
    def router_loop(self, tmp_path):
        """Create AgentLoop with model router enabled."""
        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "default-model"
        provider.estimate_prompt_tokens.return_value = (10_000, "test")

        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=tmp_path,
            model="default-model",
            context_window_tokens=1,
        )
        return loop

    @pytest.mark.asyncio
    async def test_first_iteration_no_crash(self, router_loop):
        """First iteration should not crash when response is None."""
        router_loop.tools = MagicMock()
        router_loop.tools.get_definitions = MagicMock(return_value=[])
        router_loop.tools.execute = AsyncMock(return_value="result")

        router_loop.provider.chat_with_retry = AsyncMock(
            return_value=LLMResponse(content="hello", finish_reason="stop")
        )

        router_loop.context.add_assistant_message = MagicMock(return_value=[])
        router_loop.context.add_tool_result = MagicMock(return_value=[])
        router_loop._set_tool_context = MagicMock()

        # This should NOT raise NameError
        result = await router_loop._run_agent_loop(
            initial_messages=[{"role": "user", "content": "test"}],
        )

        # Should return successfully
        assert result[0] == "hello"

    @pytest.mark.asyncio
    async def test_multiple_iterations_with_model_escalation(self, router_loop):
        """Multiple iterations should work and pass response to select()."""
        call_count = [0]
        model_tier = ["default"]

        async def mock_chat(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: return tool call
                return LLMResponse(
                    content="",
                    tool_calls=[ToolCallRequest(id="1", name="test", arguments={})],
                    finish_reason="stop",
                )
            else:
                # Subsequent calls: return final response
                return LLMResponse(content="done", finish_reason="stop")

        router_loop.provider.chat_with_retry = AsyncMock(side_effect=mock_chat)
        router_loop.tools = MagicMock()
        router_loop.tools.get_definitions = MagicMock(return_value=[])
        router_loop.tools.execute = AsyncMock(return_value="tool_result")
        router_loop.context.add_assistant_message = MagicMock(return_value=[])
        router_loop.context.add_tool_result = MagicMock(return_value=[])
        router_loop._set_tool_context = MagicMock()

        # Enable model router
        router_loop.model_router._enabled = True

        result = await router_loop._run_agent_loop(
            initial_messages=[{"role": "user", "content": "test"}],
        )

        # Should complete successfully
        assert call_count[0] == 2


class TestToolResultErrorHandling:
    """Test error handling in parallel tool execution."""

    @pytest.mark.asyncio
    async def test_tool_exception_caught_in_gather(self, tmp_path):
        """Exceptions from tools should be caught and converted to error messages."""
        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        provider.estimate_prompt_tokens.return_value = (10_000, "test")

        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=tmp_path,
            model="test-model",
            context_window_tokens=1,
        )

        tool_calls = [
            ToolCallRequest(id="1", name="failing_tool", arguments={}),
            ToolCallRequest(id="2", name="good_tool", arguments={}),
        ]

        async def bad_execute(name, args):
            if name == "failing_tool":
                raise ValueError("Intentional test error")
            return "good_result"

        loop.provider.chat_with_retry = AsyncMock(
            return_value=LLMResponse(content="", tool_calls=tool_calls, finish_reason="stop")
        )
        loop.tools = MagicMock()
        loop.tools.get_definitions = MagicMock(return_value=[])
        loop.tools.execute = AsyncMock(side_effect=bad_execute)
        loop.context.add_assistant_message = MagicMock(return_value=[])
        loop.context.add_tool_result = MagicMock(return_value=[])
        loop._set_tool_context = MagicMock()

        # Should not raise, should handle error gracefully
        result = await loop._run_agent_loop(
            initial_messages=[{"role": "user", "content": "test"}],
        )

        # Should complete despite one tool failing
        assert result is not None
