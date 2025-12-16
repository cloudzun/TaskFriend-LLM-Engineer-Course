# context.py

from . import config
from typing import List, Dict, Any, Optional, Callable
from .utils import estimate_tokens, log_gray, log_yellow

def truncate_conversation(
    conversation: List[Dict[str, str]],
    client: Any = None,
    summarizer_fn: Optional[Callable[[str, Any], str]] = None
) -> List[Dict[str, str]]:
    """
    Truncate conversation based on context window.
    Optionally summarize dropped messages using a custom function.

    Args:
        conversation: List of message dicts with 'role' and 'content'
        client: LLM client (needed for summarization)
        summarizer_fn: Optional function with signature f(text: str, client) -> str

    Returns:
        List of messages to send to model (with optional summary)
    """

    if not config.USE_CONTEXT_WINDOW:
        return conversation  # No truncation needed

    system_msg = conversation[0]
    other_msgs = conversation[1:]

    total_tokens = estimate_tokens(system_msg["content"])
    selected_msgs = []

    # Add messages from newest to oldest until limit
    for msg in reversed(other_msgs):
        msg_tokens = estimate_tokens(msg["content"])
        if total_tokens + msg_tokens <= config.CONTEXT_WINDOW:
            total_tokens += msg_tokens
            selected_msgs.append(msg)
        else:
            break

    # Reverse back to chronological order
    kept_messages = [system_msg] + list(reversed(selected_msgs))

    # Identify dropped messages (non-system, not kept)
    kept_ids = {id(m) for m in kept_messages}
    dropped_msgs = [
        msg for msg in other_msgs
        if id(msg) not in kept_ids and msg["role"] != "system"
    ]

    # Show dropped messages with gray background
    if config.SHOW_TRUNCATED and dropped_msgs:
        log_gray("🗑️  TRUNCATED MESSAGES (not sent to model):")
        log_gray("─" * 50)
        for msg in dropped_msgs:
            content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
            log_gray(f"{msg['role'].capitalize()}: {content}")
        log_gray("─" * 50)

    # Summarize dropped messages and show with yellow background
    if config.SUMMARIZE_DROPPED and dropped_msgs:
        summary_content = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in dropped_msgs
        ])

        if summarizer_fn is not None:
            try:
                summary = summarizer_fn(summary_content, client)
            except Exception as e:
                summary = f"[Custom summary failed: {e}]"
        else:
            from .utils import summarize_text
            summary = summarize_text(summary_content, client, max_tokens=60)

        log_yellow("\n🧠 MEMORY SUMMARY CREATED:")
        log_yellow("🔹" * 50)
        log_yellow(summary)
        log_yellow("🔹" * 50)

        # Inject as an assistant message
        memory_msg = {
            "role": "assistant",
            "content": f"[Summary: {summary}]"
        }

        chronological_msgs = list(reversed(selected_msgs))

        return [system_msg, memory_msg] + chronological_msgs

    return kept_messages
