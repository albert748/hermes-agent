import pytest
import json
from unittest.mock import patch
from agent.display import get_cute_tool_message, _classify_tool_result, _STATUS_ICONS


# ═══════════════════════════════════════════════════════════════════
# _classify_tool_result 测试
# ═══════════════════════════════════════════════════════════════════

def test_terminal_error_with_message():
    data = {"output": "some output", "exit_code": 1, "error": "This is a very specific and long error message that should be truncated"}
    status, msg = _classify_tool_result("terminal", json.dumps(data))
    assert status == "error"
    assert msg.startswith("exit 1: ")
    assert "This is a very specific" in msg

def test_terminal_error_no_message():
    data = {"output": "some output", "exit_code": 1}
    status, msg = _classify_tool_result("terminal", json.dumps(data))
    assert status == "error"
    assert "exit 1" in msg

def test_terminal_success():
    data = {"output": "hello", "exit_code": 0}
    status, msg = _classify_tool_result("terminal", json.dumps(data))
    assert status == "success"
    assert msg == "OK"

def test_terminal_warning_no_matches():
    """exit_code=1 with 'No matches found' should be warning, not error."""
    data = {"output": "No matches found (not an error)", "exit_code": 1, "error": None,
            "exit_code_meaning": "No matches found (not an error)"}
    status, msg = _classify_tool_result("terminal", json.dumps(data))
    assert status == "warning"
    assert "no matches" in msg

def test_terminal_timeout_is_error():
    """exit_code=124 (timeout) should always be error."""
    data = {"output": "", "exit_code": 124, "error": None}
    status, msg = _classify_tool_result("terminal", json.dumps(data))
    assert status == "error"


def test_file_not_found_path_shortening():
    data = {"error": "File not found: /very/long/path/name.txt"}
    status, msg = _classify_tool_result("read_file", json.dumps(data))
    assert status == "error"
    assert msg == "File not found: name.txt"

def test_read_file_success_with_lines():
    data = {"content": "line1\nline2\nline3\n", "total_lines": 3}
    status, msg = _classify_tool_result("read_file", json.dumps(data))
    assert status == "success"
    assert "3 lines" in msg


def test_web_search_empty_results_warning():
    data = {"success": True, "data": {"web": []}}
    status, msg = _classify_tool_result("web_search", json.dumps(data))
    assert status == "warning"
    assert msg == "no results"

def test_web_search_with_results_success():
    data = {"success": True, "data": {"web": [{"title": "X", "url": "http://x.com", "description": "desc"}]}}
    status, msg = _classify_tool_result("web_search", json.dumps(data))
    assert status == "success"
    assert "1 result" in msg

def test_web_search_error():
    data = {"error": "0 results after 2 attempts", "success": False}
    status, msg = _classify_tool_result("web_search", json.dumps(data))
    assert status == "error"


def test_search_files_zero_matches_warning():
    data = {"total_count": 0, "matches": []}
    status, msg = _classify_tool_result("search_files", json.dumps(data))
    assert status == "warning"
    assert "no matches" in msg

def test_search_files_with_matches_success():
    data = {"total_count": 5, "matches": [{"path": "/a", "line": 1}]}
    status, msg = _classify_tool_result("search_files", json.dumps(data))
    assert status == "success"
    assert "5 matches" in msg


def test_patch_success():
    data = {"success": True, "diff": "--- a\n+++ b\n", "files_modified": 1}
    status, msg = _classify_tool_result("patch", json.dumps(data))
    assert status == "success"

def test_patch_error():
    data = {"success": False, "error": "No match found for old_string"}
    status, msg = _classify_tool_result("patch", json.dumps(data))
    assert status == "error"

def test_patch_lsp_diagnostics_not_error():
    """LSP diagnostics on a successful patch should NOT be treated as error."""
    data = {"success": True, "diff": "--- a\n+++ b\n", "lsp_diagnostics": "<diagnostics>ERROR</diagnostics>"}
    status, msg = _classify_tool_result("patch", json.dumps(data))
    assert status == "success"


def test_write_file_success():
    data = {"bytes_written": 123, "lint": {"status": "ok"}}
    status, msg = _classify_tool_result("write_file", json.dumps(data))
    assert status == "success"
    assert "123 bytes" in msg

def test_write_file_lint_warning():
    """Lint error on write should be warning, not error (preserving existing behavior)."""
    data = {"bytes_written": 12, "lint": {"status": "error", "output": "SyntaxError"}}
    status, msg = _classify_tool_result("write_file", json.dumps(data))
    assert status == "warning"
    assert "lint error" in msg


def test_memory_full_warning():
    data = {"success": False, "error": "Memory entries exceed the limit"}
    status, msg = _classify_tool_result("memory", json.dumps(data))
    assert status == "warning"
    assert "memory full" in msg

def test_memory_success():
    data = {"success": True, "target": "memory", "entry_count": 1}
    status, msg = _classify_tool_result("memory", json.dumps(data))
    assert status == "success"


def test_session_search_no_results_warning():
    data = {"success": True, "count": 0, "results": []}
    status, msg = _classify_tool_result("session_search", json.dumps(data))
    assert status == "warning"


# ═══════════════════════════════════════════════════════════════════
# get_cute_tool_message 新格式测试
# ═══════════════════════════════════════════════════════════════════

def test_new_format_has_status_icon():
    """All completion lines should end with a status icon."""
    for status, icon in _STATUS_ICONS.items():
        assert icon in ("✅", "🟡", "❌")

def test_terminal_success_shows_both_input_and_result():
    """Completion line shows BOTH input (command) and result (OK)."""
    msg = get_cute_tool_message("terminal", {"command": "ls -la /tmp"}, 0.5,
                                json.dumps({"exit_code": 0, "output": "file1\nfile2"}))
    assert "ls -la" in msg      # input: the command
    assert "→" in msg           # separator between input and result
    assert "OK" in msg          # result: what happened
    assert "✅" in msg          # status icon
    assert "0.5s" in msg        # duration

def test_terminal_error_shows_exit_code():
    msg = get_cute_tool_message("terminal", {"command": "bad_cmd"}, 0.2,
                                json.dumps({"exit_code": 127, "error": "command not found: bad_cmd"}))
    assert "exit 127" in msg
    assert "command not found" in msg
    assert "❌" in msg
    # The error message from the shell naturally contains the command name —
    # that's the result, not the input we're displaying.

def test_web_search_shows_both_query_and_result():
    """Completion line shows BOTH query (input) and result count."""
    msg = get_cute_tool_message("web_search", {"query": "long search query here"}, 1.5,
                                json.dumps({"success": True, "data": {"web": [{"title": "A"}, {"title": "B"}]}}))
    assert "long search" in msg   # input: the query
    assert "→" in msg             # separator
    assert "2 results" in msg     # result
    assert "✅" in msg

def test_web_search_zero_results_warning():
    msg = get_cute_tool_message("web_search", {"query": "nothing"}, 0.8,
                                json.dumps({"success": True, "data": {"web": []}}))
    assert "🟡" in msg
    assert "no results" in msg

def test_read_file_error_shows_filename():
    msg = get_cute_tool_message("read_file", {"path": "/long/nested/path/to/file.txt"}, 0.1,
                                json.dumps({"error": "File not found: /long/nested/path/to/file.txt"}))
    assert "File not found: file.txt" in msg
    assert "❌" in msg

def test_search_files_shows_match_count():
    msg = get_cute_tool_message("search_files", {"pattern": "grep_pattern"}, 0.3,
                                json.dumps({"total_count": 3, "matches": [{"a": 1}, {"b": 2}, {"c": 3}]}))
    assert "3 matches" in msg
    assert "✅" in msg

def test_patch_success_shows_file_count():
    msg = get_cute_tool_message("patch", {"path": "/tmp/a.py"}, 0.2,
                                json.dumps({"success": True, "diff": "--- a\n+++ b\n", "files_modified": 1}))
    assert "1 file" in msg
    assert "✅" in msg

def test_skill_view_success():
    msg = get_cute_tool_message("skill_view", {"name": "hermes-agent"}, 0.4,
                                json.dumps({"success": True, "name": "hermes-agent"}))
    assert "loaded" in msg
    assert "✅" in msg

def test_memory_full_shows_warning():
    msg = get_cute_tool_message("memory", {"action": "add", "target": "memory", "content": "test"}, 0.1,
                                json.dumps({"success": False, "error": "Memory entries exceed the limit"}))
    assert "🟡" in msg
    assert "memory full" in msg

def test_no_result_param_still_shows_ok():
    """When result is None, show OK with success icon."""
    msg = get_cute_tool_message("web_search", {"query": "test"}, 0.5)
    assert "OK" in msg
    assert "✅" in msg

def test_write_file_lint_warning_not_error_icon():
    msg = get_cute_tool_message("write_file", {"path": "/tmp/a.py"}, 0.1,
                                json.dumps({"bytes_written": 12, "lint": {"status": "error", "output": "SyntaxError"}}))
    assert "🟡" in msg
    assert "❌" not in msg
