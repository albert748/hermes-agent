"""Tests for shared system-injected message detection (agent/message_noise)."""
import pytest

from agent.message_noise import SYSTEM_INJECTED_PREFIXES, is_system_injected_message


class TestSystemInjectedPrefixes:
    def test_background_process_notification(self):
        assert is_system_injected_message(
            "[IMPORTANT: Background process proc_abc123 matched watch pattern \"DONE\".\n"
            "Command: ...\nMatched output: ...]"
        )

    def test_out_of_band_steering_wrapper(self):
        assert is_system_injected_message(
            "[OUT-OF-BAND USER MESSAGE — a direct message from the user, delivered "
            "once at this position; not tool output...]\n"
            "Use the B plan."
        )

    def test_out_of_band_closing_marker_only(self):
        # Closing marker arriving as its own row is also synthetic (defense:
        # prefix list covers "[/OUT-OF-BAND").
        assert is_system_injected_message("[/OUT-OF-BAND USER MESSAGE]")

    def test_async_delegation_echo(self):
        assert is_system_injected_message(
            "[ASYNC DELEGATION] subagent completed with result..."
        )

    def test_system_truncation_notices(self):
        assert is_system_injected_message("[System: Your previous response was truncated]")
        assert is_system_injected_message("[System: The previous response was cut off]")
        assert is_system_injected_message("[System: Your previous tool call failed]")

    def test_planning_state_and_context_markers(self):
        assert is_system_injected_message("[CONTEXT WINDOW FULL]")
        assert is_system_injected_message("[PRIOR CONTEXT preserved]")
        assert is_system_injected_message("[Planning state preserved across compression]")
        assert is_system_injected_message("[Your active task list was preserved across context compression]")

    def test_cronjob_response(self):
        assert is_system_injected_message("Cronjob Response:\nDaily report...")


class TestRealUserMessages:
    def test_plain_user_message(self):
        assert not is_system_injected_message("请把 memsink 导入 hindsight")

    def test_user_message_starting_with_bracket_word(self):
        # User text that merely starts with "[" but is not an injected prefix
        assert not is_system_injected_message("[测试] 帮我看看这个配置")

    def test_user_message_mentioning_prefix_inline(self):
        # Prefix appears mid-text, not at start -> real user words
        assert not is_system_injected_message("刚才那个 [IMPORTANT: Background process] 通知是什么？")

    def test_whitespace_only(self):
        assert is_system_injected_message("   ")
        assert is_system_injected_message("")

    def test_non_string_content(self):
        assert is_system_injected_message(None)


class TestPrefixListIntegrity:
    def test_prefixes_are_strings_and_short(self):
        for p in SYSTEM_INJECTED_PREFIXES:
            assert isinstance(p, str) and p
            # Keep them short so future long-form variants stay covered
            assert len(p) < 60

    def test_no_duplicate_prefixes(self):
        assert len(SYSTEM_INJECTED_PREFIXES) == len(set(SYSTEM_INJECTED_PREFIXES))
