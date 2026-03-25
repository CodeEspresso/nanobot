"""Read-time session history filter: remove noise + compress tool results.

Pure-function module — no nanobot class dependencies, operates on list[dict].
Applied between session.get_history() and context.build_messages() so it never
mutates stored history and never touches the current turn's tool calls.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Noise detection
# ---------------------------------------------------------------------------

_NOISE_PATTERNS_USER: set[str] = {
    "好的", "好", "ok", "okay", "嗯", "嗯嗯", "继续", "谢谢", "thanks", "thx",
    "收到", "了解", "明白", "知道了", "行", "可以", "对", "是的", "yes", "no",
    "不", "哦", "哈哈", "呵呵", "嗯哼", "got it", "sure", "yep", "yup", "k",
    "nice", "cool", "great", "good", "fine", "right",
}

_NOISE_PATTERNS_ASSISTANT: set[str] = {
    "好的", "好", "明白了", "收到", "了解", "知道了", "明白", "好的，",
    "ok", "okay", "got it", "understood", "sure",
}

# Short assistant messages that START with an acknowledgment AND contain a
# transitional phrase are progress reports with no lasting value.
_TRANSITIONAL_START_RE = re.compile(
    r"^(好的|好|ok|okay|alright|done|完成了?|成功了?|搞定了?)[，,．.\s!！]",
    re.IGNORECASE | re.UNICODE,
)
_TRANSITIONAL_PHRASE_RE = re.compile(
    r"(让我|我来|接下来|下一步|现在|然后|那么|继续|let me|now i|next|i.ll|moving on)",
    re.IGNORECASE | re.UNICODE,
)

# Max length for transitional detection — longer messages likely have real content
_TRANSITIONAL_MAX_LEN = 200

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize(text: str) -> str:
    """Strip punctuation and whitespace, lowercase."""
    return _PUNCT_RE.sub("", text).strip().lower()


def _is_noise_message(msg: dict) -> bool:
    """Return True if the message is low-value noise that can be dropped."""
    role = msg.get("role")
    content = msg.get("content")

    if not isinstance(content, str) or not content.strip():
        # Empty assistant without tool_calls → noise
        if role == "assistant" and not msg.get("tool_calls"):
            return True
        # Don't drop empty tool results or multimodal content
        return False

    norm = _normalize(content)
    if not norm:
        return role == "assistant" and not msg.get("tool_calls")

    if role == "user":
        return norm in _NOISE_PATTERNS_USER
    if role == "assistant" and not msg.get("tool_calls"):
        if norm in _NOISE_PATTERNS_ASSISTANT:
            return True
        # Short transitional progress reports
        if (len(content) <= _TRANSITIONAL_MAX_LEN
                and _TRANSITIONAL_START_RE.search(content)
                and _TRANSITIONAL_PHRASE_RE.search(content)):
            return True

    return False


# ---------------------------------------------------------------------------
# Tool result compression
# ---------------------------------------------------------------------------

def _extract_tool_name(msg: dict) -> str:
    """Best-effort extraction of the tool name from a tool-result message."""
    return msg.get("name", msg.get("tool_name", ""))


def _summarize_tool_result(msg: dict) -> dict:
    """Collapse a tool result to a one-line summary. For old results already
    processed by the LLM — we only need to remind it what happened."""
    tool = _extract_tool_name(msg)
    content = msg.get("content", "")
    if not isinstance(content, str):
        content = str(content)

    # Extract a meaningful one-liner based on tool type
    tl = tool.lower()
    lines = content.split("\n")
    total_lines = len(lines)
    first_line = lines[0][:120] if lines else ""

    if tl in ("read_file", "readfile"):
        summary = f"[read_file OK: {total_lines} lines] {first_line}"
    elif tl in ("exec", "shell", "execute", "run"):
        # Check for error indicators
        has_error = any(k in content.lower() for k in ("error", "traceback", "failed", "exit code"))
        status = "ERR" if has_error else "OK"
        summary = f"[exec {status}: {total_lines} lines] {first_line}"
    elif tl in ("edit_file", "editfile"):
        summary = f"[edit_file OK] {first_line}"
    elif tl in ("write_file", "writefile"):
        summary = f"[write_file OK] {first_line}"
    elif tl in ("list_dir", "listdir", "list_files", "listfiles"):
        items = [l.strip() for l in lines if l.strip()]
        preview = ", ".join(items[:5])
        summary = f"[{len(items)} items: {preview}, ...]" if len(items) > 5 else content
    elif tl in ("web_search", "websearch"):
        summary = f"[web_search: {total_lines} lines] {first_line}"
    elif tl in ("web_fetch", "webfetch"):
        summary = f"[web_fetch: {len(content)} chars] {first_line}"
    else:
        summary = f"[{tool} OK: {len(content)} chars] {first_line}"

    out = dict(msg)
    out["content"] = summary[:300]
    return out


def _compress_tool_result(msg: dict, max_chars: int) -> dict:
    """Return a (possibly compressed) copy of a tool-result message."""
    content = msg.get("content", "")
    if not isinstance(content, str) or len(content) <= max_chars:
        return msg

    tool = _extract_tool_name(msg)
    compressed = _compress_by_tool(tool, content, max_chars)
    out = dict(msg)
    out["content"] = compressed
    return out


def _compress_by_tool(tool: str, content: str, max_chars: int) -> str:
    """Compress content based on tool type."""
    tl = tool.lower()

    if tl in ("read_file", "readfile"):
        return _compress_read_file(content, max_chars)
    if tl in ("exec", "shell", "execute", "run"):
        return _compress_exec(content, max_chars)
    if tl in ("list_dir", "listdir", "list_files", "listfiles"):
        return _compress_listing(content, max_chars)
    if tl in ("web_search", "websearch", "web_fetch", "webfetch"):
        return content[:max_chars] + "\n... (truncated)"
    # edit_file / write_file are usually short — fall through to generic
    if tl in ("edit_file", "editfile", "write_file", "writefile"):
        return content  # don't compress

    return _compress_generic(content, max_chars)


def _compress_read_file(content: str, max_chars: int) -> str:
    lines = content.split("\n")
    header_lines = lines[:10]
    header = "\n".join(header_lines)
    total = len(lines)
    return f"[read_file: {total} lines]\n{header}\n... ({total - 10} more lines, truncated)"


def _compress_exec(content: str, max_chars: int) -> str:
    lines = content.split("\n")
    if len(lines) <= 10:
        return content
    head = "\n".join(lines[:5])
    tail = "\n".join(lines[-3:])
    omitted = len(lines) - 8
    return f"{head}\n... (omitted {omitted} lines)\n{tail}"


def _compress_listing(content: str, max_chars: int) -> str:
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    if len(lines) <= 10:
        return content
    preview = ", ".join(lines[:5])
    return f"[{len(lines)} items: {preview}, ...]"


def _compress_generic(content: str, max_chars: int) -> str:
    head_budget = int(max_chars * 0.6)
    tail_budget = max_chars - head_budget
    return content[:head_budget] + "\n... (truncated)\n" + content[-tail_budget:]


# ---------------------------------------------------------------------------
# Main filter
# ---------------------------------------------------------------------------

# How many recent messages to keep with full (compressed) tool results.
# Older tool results get collapsed to one-line summaries.
_RECENT_WINDOW = 10


def filter_session_history(
    history: list[dict],
    max_tool_chars: int = 2000,
    max_messages: int = 40,
) -> list[dict]:
    """Read-time filter: remove noise + compress tool results + cap total messages.

    Does NOT mutate input — returns a new list.

    Strategy:
    1. Cap history to last `max_messages` at a safe tool_call boundary
    2. Remove noise messages (short acks, transitional reports)
    3. Recent tool results (last _RECENT_WINDOW messages): compress if over max_chars
    4. Old tool results (before that): collapse to one-line summary
    5. tool_call chains stay intact (assistant with tool_calls + matching tool results)
    """
    if not history:
        return []

    # Step 0: cap history length at a safe boundary
    if max_messages and len(history) > max_messages:
        history = _trim_at_safe_boundary(history, max_messages)

    # Step 1: identify assistant messages that have tool_calls
    required_tool_ids: set[str] = set()
    assistant_with_tools: set[int] = set()

    for i, msg in enumerate(history):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            assistant_with_tools.add(i)
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id") if isinstance(tc, dict) else None
                if tc_id:
                    required_tool_ids.add(tc_id)

    # The "recent window" boundary: messages within the last N are recent
    recent_start = max(0, len(history) - _RECENT_WINDOW)

    # Step 2: build filtered list
    result: list[dict] = []
    for i, msg in enumerate(history):
        role = msg.get("role")
        is_recent = i >= recent_start

        # Always keep assistant messages with tool_calls
        if i in assistant_with_tools:
            result.append(msg)
            continue

        # Tool results: recent ones get normal compression, old ones get summarized
        if role == "tool":
            if is_recent:
                result.append(_compress_tool_result(msg, max_tool_chars))
            else:
                result.append(_summarize_tool_result(msg))
            continue

        # Filter noise for user / assistant plain-text messages
        if _is_noise_message(msg):
            continue

        result.append(msg)

    return result


def _trim_at_safe_boundary(history: list[dict], max_messages: int) -> list[dict]:
    """Trim history from the front, cutting at a safe tool_call boundary.

    Scans forward from the cut point to find a position where we won't
    leave orphaned tool results (tool msg without its assistant+tool_calls).
    """
    cut = len(history) - max_messages

    # Scan forward from cut to find a safe boundary:
    # A safe point is a 'user' message (start of a new turn)
    # or an 'assistant' message WITHOUT tool_calls
    while cut < len(history):
        msg = history[cut]
        role = msg.get("role")
        if role == "user":
            break
        if role == "assistant" and not msg.get("tool_calls"):
            break
        # Skip tool results and assistant+tool_calls to keep chains intact
        cut += 1

    return history[cut:]
