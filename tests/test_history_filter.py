"""Tests for history_filter.py - session history noise removal and compression."""

import pytest
from nanobot.agent.history_filter import (
    filter_session_history,
    _is_noise_message,
    _summarize_tool_result,
    _compress_tool_result,
)


class TestNoiseDetection:
    """Test noise message detection."""

    def test_user_noise_acknowledgments(self):
        """User messages that are just acknowledgments should be filtered."""
        for msg in [
            {"role": "user", "content": "好的"},
            {"role": "user", "content": "ok"},
            {"role": "user", "content": "谢谢"},
            {"role": "user", "content": "知道了"},
            {"role": "user", "content": "yes"},
            {"role": "user", "content": "k"},
        ]:
            assert _is_noise_message(msg), f"Expected noise: {msg['content']}"

    def test_assistant_noise_acknowledgments(self):
        """Assistant messages that are just acknowledgments should be filtered."""
        for msg in [
            {"role": "assistant", "content": "好的"},
            {"role": "assistant", "content": "明白了"},
            {"role": "assistant", "content": "收到"},
            {"role": "assistant", "content": "ok"},
        ]:
            assert _is_noise_message(msg), f"Expected noise: {msg['content']}"

    def test_transitional_progress_reports(self):
        """Short messages starting with ack + transitional phrase should be filtered."""
        msg = {"role": "assistant", "content": "好的，让我来帮你查一下"}
        assert _is_noise_message(msg)

    def test_meaningful_messages_preserved(self):
        """Messages with real content should NOT be filtered."""
        for msg in [
            {"role": "user", "content": "帮我写一个Python脚本"},
            {"role": "assistant", "content": "我来帮你写这个脚本。首先..."},
            {"role": "user", "content": "解释一下什么是闭包"},
        ]:
            assert not _is_noise_message(msg), f"Expected preserved: {msg['content']}"

    def test_empty_assistant_without_tool_calls(self):
        """Empty assistant message without tool_calls is noise."""
        assert _is_noise_message({"role": "assistant", "content": ""})
        assert _is_noise_message({"role": "assistant", "content": "   "})

    def test_empty_tool_results_not_filtered(self):
        """Tool result messages should not be filtered even if content is empty."""
        assert not _is_noise_message({"role": "tool", "content": ""})

    def test_multimodal_content_not_filtered(self):
        """Multimodal content (list) should not be filtered."""
        assert not _is_noise_message({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "..."}}]})


class TestToolResultCompression:
    """Test tool result summarization and compression."""

    def test_summarize_read_file(self):
        """read_file results should be summarized as one-liner."""
        msg = {
            "role": "tool",
            "name": "read_file",
            "content": "line1\nline2\nline3\n... many more lines ...",
        }
        result = _summarize_tool_result(msg)
        assert "[read_file OK:" in result["content"]
        assert len(result["content"]) <= 300

    def test_summarize_exec_success(self):
        """exec results should show OK status."""
        msg = {
            "role": "tool",
            "name": "exec",
            "content": "output\nmore output\n",
        }
        result = _summarize_tool_result(msg)
        assert "[exec OK:" in result["content"]

    def test_summarize_exec_error(self):
        """exec with error should show ERR status."""
        msg = {
            "role": "tool",
            "name": "exec",
            "content": "Error: something failed\nTraceback...",
        }
        result = _summarize_tool_result(msg)
        assert "[exec ERR:" in result["content"]

    def test_summarize_list_dir(self):
        """list_dir should show item count preview."""
        items = "\n".join([f"file{i}.py" for i in range(20)])
        msg = {
            "role": "tool",
            "name": "list_dir",
            "content": items,
        }
        result = _summarize_tool_result(msg)
        assert "items:" in result["content"]
        assert "..." in result["content"]

    def test_compress_truncates_long_content(self):
        """Long tool results should be compressed."""
        long_content = "\n".join([f"line{i}" for i in range(100)])
        msg = {"role": "tool", "name": "exec", "content": long_content}
        result = _compress_tool_result(msg, max_chars=200)
        assert len(result["content"]) <= 200
        assert "..." in result["content"]

    def test_compress_preserves_short_content(self):
        """Short tool results should not be compressed."""
        msg = {"role": "tool", "name": "exec", "content": "short output"}
        result = _compress_tool_result(msg, max_chars=200)
        assert result["content"] == "short output"


class TestFilterSessionHistory:
    """Test the main filter_session_history function."""

    def test_empty_history(self):
        """Empty history should return empty list."""
        assert filter_session_history([]) == []

    def test_noise_filtered_meaningful_preserved(self):
        """Noise messages should be removed, meaningful ones kept."""
        history = [
            {"role": "user", "content": "好的"},
            {"role": "assistant", "content": "好的，我来帮你"},
            {"role": "user", "content": "帮我写一个函数"},
        ]
        result = filter_session_history(history)
        assert len(result) == 1
        assert result[0]["content"] == "帮我写一个函数"

    def test_assistant_with_tool_calls_preserved(self):
        """Assistant messages with tool_calls should always be kept."""
        history = [
            {"role": "assistant", "content": "好的", "tool_calls": [{"id": "1", "name": "test"}]},
        ]
        result = filter_session_history(history)
        assert len(result) == 1

    def test_tool_results_compressed_in_old_messages(self):
        """Old tool results should be summarized, not fully kept."""
        history = [
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1", "name": "list_dir"}]},
            {"role": "tool", "tool_call_id": "tc1", "name": "list_dir", "content": "file1.py\nfile2.py\nfile3.py\n..."},
        ] * 6  # Repeat to have old tool results
        result = filter_session_history(history, max_messages=10)
        # Recent ones should be compressed but not summarized
        tool_msgs = [m for m in result if m["role"] == "tool"]
        for tm in tool_msgs:
            assert len(tm["content"]) <= 300

    def test_max_messages_boundary(self):
        """History should be trimmed at tool_call boundary."""
        history = [
            {"role": "user", "content": f"msg{i}"}
            for i in range(50)
        ]
        result = filter_session_history(history, max_messages=20)
        # Should not cut in middle of user message sequence
        assert len(result) <= 20

    def test_preserves_tool_call_chain_integrity(self):
        """Tool_call + tool_result chains should stay together."""
        history = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "name": "test"}]},
            {"role": "tool", "tool_call_id": "1", "name": "test", "content": "result"},
        ]
        result = filter_session_history(history)
        # Both should be present
        roles = [m["role"] for m in result]
        assert "assistant" in roles
        assert "tool" in roles

    def test_does_not_mutate_input(self):
        """Original history should not be modified."""
        original = [{"role": "user", "content": "test"}]
        original_copy = list(original)
        filter_session_history(original)
        assert original == original_copy
