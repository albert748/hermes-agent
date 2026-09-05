"""Shared detection for system-injected user-role messages.

The runtime injects certain rows with a *user* role even though they are not
the user speaking: background-process notifications
(``[IMPORTANT: Background process ...]``), out-of-band steering wrappers
(``[OUT-OF-BAND USER MESSAGE ...]``), delegation echoes, planning-state
notices, cron job responses, etc. Role-based filtering cannot see these, so
every consumer that treats user rows as real user words must skip them.

Consumers (all must use this one list so it cannot drift):
- context compression  (``agent/context_compressor.py``)
- conversation compression (``agent/conversation_compression.py``)
- external-memory retain / recall-prefetch (``run_agent.py``,
  ``agent/memory_manager.py``)
"""

# Keep prefixes SHORT on purpose: each entry is a startswith() anchor, so a
# short prefix also covers its longer historical variants (e.g. "[System:"
# covers "[System: Your previous response was truncated").
SYSTEM_INJECTED_PREFIXES: tuple[str, ...] = (
    "[System:",
    "[CONTEXT",
    "[PRIOR CONTEXT",
    "[IMPORTANT: Background",
    "[Your active task list",
    "[Planning state preserved",
    "[ASYNC DELEGATION",
    "[OUT-OF-BAND",
    "[/OUT-OF-BAND",
    "Cronjob Response:",
)

__all__ = ["SYSTEM_INJECTED_PREFIXES", "is_system_injected_message"]


def is_system_injected_message(content: object) -> bool:
    """True when *content* is a system-injected user-role message (or empty).

    Empty/whitespace content is treated as injected: it carries no real user
    words and should never be retained or quoted as the user speaking.
    """
    if not isinstance(content, str) or not content.strip():
        return True
    return content.lstrip().startswith(SYSTEM_INJECTED_PREFIXES)
