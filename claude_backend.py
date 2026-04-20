"""
Claude LLM backend for PageIndex.

Monkey-patches OpenAI LLM functions in pageindex.utils so that PageIndex
uses Claude (via the Claude Agent SDK) without any source modifications.
Call activate() before importing pageindex.
"""

import asyncio
import logging
import os
import sys

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    query,
)

logger = logging.getLogger(__name__)

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "sonnet")

_VALID_EFFORTS = ("low", "medium", "high", "max")
_effort_env = os.environ.get("CLAUDE_EFFORT", "").strip().lower()
CLAUDE_EFFORT = _effort_env if _effort_env in _VALID_EFFORTS else None
if _effort_env and CLAUDE_EFFORT is None:
    logger.warning(
        "Ignoring CLAUDE_EFFORT=%r; expected one of %s or empty.",
        _effort_env,
        _VALID_EFFORTS,
    )


async def _call_sdk(prompt_text: str) -> str:
    """Send a prompt to Claude via the Agent SDK and return the text response."""
    options_kwargs = {"model": CLAUDE_MODEL, "max_turns": 1}
    if CLAUDE_EFFORT is not None:
        options_kwargs["effort"] = CLAUDE_EFFORT
    options = ClaudeAgentOptions(**options_kwargs)
    parts: list[str] = []
    async for message in query(prompt=prompt_text, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    parts.append(block.text)
    return "".join(parts).strip() or "Error"


async def _call_llm_async(prompt_text: str, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            return await _call_sdk(prompt_text)
        except Exception as e:
            logger.error("SDK error (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)
    return "Error"


def _call_llm(prompt_text: str, max_retries: int = 5) -> str:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run, _call_llm_async(prompt_text, max_retries)
            ).result()
    return asyncio.run(_call_llm_async(prompt_text, max_retries))


def _build_prompt_text(prompt, chat_history=None):
    parts = []
    if chat_history:
        for msg in chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[System instruction]: {content}")
            elif role == "assistant":
                parts.append(f"[Previous assistant response]: {content}")
            elif role == "user":
                parts.append(f"[Previous user message]: {content}")
    parts.append(prompt)
    return "\n\n".join(parts)


def claude_llm_completion(model, prompt, chat_history=None, return_finish_reason=False):
    result = _call_llm(_build_prompt_text(prompt, chat_history))
    if return_finish_reason:
        return result, "finished"
    return result


async def claude_llm_acompletion(model, prompt):
    return await _call_llm_async(prompt)


def claude_count_tokens(text, model=None):
    if not text:
        return 0
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


_activated = False


def activate():
    """Monkey-patch pageindex to use Claude instead of OpenAI."""
    global _activated
    if _activated:
        return

    import pageindex.page_index
    import pageindex.utils

    patches = {
        "llm_completion": claude_llm_completion,
        "llm_acompletion": claude_llm_acompletion,
        "count_tokens": claude_count_tokens,
    }

    for module_name in ("pageindex.utils", "pageindex.page_index"):
        mod = sys.modules[module_name]
        for attr, fn in patches.items():
            if hasattr(mod, attr):
                setattr(mod, attr, fn)

    try:
        import pageindex.page_index_md  # noqa: F401

        mod = sys.modules["pageindex.page_index_md"]
        for attr, fn in patches.items():
            if hasattr(mod, attr):
                setattr(mod, attr, fn)
    except ImportError:
        pass

    _activated = True
    logger.info(
        "Activated: model=%s effort=%s (claude-agent-sdk)",
        CLAUDE_MODEL,
        CLAUDE_EFFORT or "default",
    )
