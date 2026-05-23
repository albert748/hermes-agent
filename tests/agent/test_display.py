"""Tests for agent/display.py — build_tool_preview() and inline diff previews."""

import os
import json
import pytest
from unittest.mock import MagicMock, patch

from agent.display import (
    build_tool_preview,
    capture_local_edit_snapshot,
    extract_edit_diff,
    get_cute_tool_message,
    set_tool_preview_max_len,
    _render_inline_unified_diff,
    _summarize_rendered_diff_sections,
    render_edit_diff_with_delta,
)


@pytest.fixture(autouse=True)
def reset_tool_preview_max_len():
    set_tool_preview_max_len(0)
    yield
    set_tool_preview_max_len(0)


class TestBuildToolPreview:
    """Tests for build_tool_preview defensive handling and normal operation."""

    def test_none_args_returns_none(self):
        """PR #453: None args should not crash, should return None."""
        assert build_tool_preview("terminal", None) is None

    def test_empty_dict_returns_none(self):
        """Empty dict has no keys to preview."""
        assert build_tool_preview("terminal", {}) is None

    def test_known_tool_with_primary_arg(self):
        """Known tool with its primary arg should return a preview string."""
        result = build_tool_preview("terminal", {"command": "ls -la"})
        assert result is not None
        assert "ls -la" in result

    def test_web_search_preview(self):
        result = build_tool_preview("web_search", {"query": "hello world"})
        assert result is not None
        assert "hello world" in result

    def test_read_file_preview(self):
        result = build_tool_preview("read_file", {"path": "/tmp/test.py", "offset": 1})
        assert result is not None
        assert "/tmp/test.py" in result

    def test_unknown_tool_with_fallback_key(self):
        """Unknown tool but with a recognized fallback key should still preview."""
        result = build_tool_preview("custom_tool", {"query": "test query"})
        assert result is not None
        assert "test query" in result

    def test_unknown_tool_no_matching_key(self):
        """Unknown tool with no recognized keys should return None."""
        result = build_tool_preview("custom_tool", {"foo": "bar"})
        assert result is None

    def test_long_value_truncated(self):
        """Preview should truncate long values."""
        long_cmd = "a" * 100
        result = build_tool_preview("terminal", {"command": long_cmd}, max_len=40)
        assert result is not None
        assert len(result) <= 43  # max_len + "..."

    def test_process_tool_with_none_args(self):
        """Process tool special case should also handle None args."""
        assert build_tool_preview("process", None) is None

    def test_process_tool_normal(self):
        result = build_tool_preview("process", {"action": "poll", "session_id": "abc123"})
        assert result is not None
        assert "poll" in result

    def test_todo_tool_read(self):
        result = build_tool_preview("todo", {"merge": False})
        assert result is not None
        assert "reading" in result

    def test_todo_tool_with_todos(self):
        result = build_tool_preview("todo", {"todos": [{"id": "1", "content": "test", "status": "pending"}]})
        assert result is not None
        assert "1 task" in result

    def test_memory_tool_add(self):
        result = build_tool_preview("memory", {"action": "add", "target": "user", "content": "test note"})
        assert result is not None
        assert "user" in result

    def test_memory_replace_missing_old_text_marked(self):
        # Avoid empty quotes "" in the preview when old_text is missing/None.
        result = build_tool_preview("memory", {"action": "replace", "target": "memory"})
        assert result == '~memory: "<missing old_text>"'
        result = build_tool_preview("memory", {"action": "remove", "target": "memory", "old_text": None})
        assert result == '-memory: "<missing old_text>"'

    def test_session_search_preview(self):
        result = build_tool_preview("session_search", {"query": "find something"})
        assert result is not None
        assert "find something" in result

    def test_false_like_args_zero(self):
        """Non-dict falsy values should return None, not crash."""
        assert build_tool_preview("terminal", 0) is None
        assert build_tool_preview("terminal", "") is None
        assert build_tool_preview("terminal", []) is None


class TestCuteToolMessagePreviewLength:
    def test_terminal_completion_shows_both_input_and_result(self):
        """Completion line shows BOTH input (command) and result."""
        set_tool_preview_max_len(0)
        # Use a short command that fits within the preview length
        command = "git status"

        # Without explicit result: shows command + OK
        line = get_cute_tool_message("terminal", {"command": command}, 0.1)
        assert command in line       # input: the command
        assert "→" in line           # separator
        assert "OK" in line          # result
        assert "✅" in line          # status icon

        # With error result: shows command + exit code/message
        line2 = get_cute_tool_message("terminal", {"command": command}, 0.1,
                                       json.dumps({"exit_code": 1, "error": "something went wrong"}))
        assert command in line2      # input is still shown
        assert "→" in line2          # separator
        assert "exit 1" in line2     # result
        assert "❌" in line2

    def test_result_message_truncation(self):
        """Long error messages are truncated to ~64 chars."""
        set_tool_preview_max_len(80)
        long_error = "E" * 100

        line = get_cute_tool_message("terminal", {"command": "cmd"}, 0.1,
                                      json.dumps({"exit_code": 1, "error": long_error}))
        # The message should be truncated (64 chars + "exit 1: " prefix)
        assert len(long_error) > 64
        # Error text should appear truncated
        assert "..." in line or len(line.split("exit 1: ")[-1].split("  ")[0]) <= 70

    def test_search_files_shows_match_count_not_pattern(self):
        set_tool_preview_max_len(80)
        pattern = "function.formatToolCall.context.preview.compactPreview.maxLength.truncate"

        # Without result: shows OK
        line = get_cute_tool_message("search_files", {"pattern": pattern}, 0.1)
        assert "OK" in line
        assert "✅" in line
        assert pattern not in line

        # With result: shows match count
        line2 = get_cute_tool_message("search_files", {"pattern": pattern}, 0.1,
                                       json.dumps({"total_count": 5, "matches": [{} for _ in range(5)]}))
        assert "5 matches" in line2
        assert pattern not in line2

    def test_read_file_shows_result_not_path(self):
        set_tool_preview_max_len(80)
        path = "/tmp/hermes-test-preview-length/deeply/nested/path/test-output.txt"

        # Without result: shows OK
        line = get_cute_tool_message("read_file", {"path": path}, 0.1)
        assert "OK" in line
        assert "✅" in line
        assert path not in line

        # With error: shows filename
        line2 = get_cute_tool_message("read_file", {"path": path}, 0.1,
                                       json.dumps({"error": f"File not found: {path}"}))
        assert "File not found: test-output.txt" in line2
        assert "❌" in line2

    def test_write_file_lint_error_shows_warning_not_error(self):
        result = json.dumps({
            "bytes_written": 12,
            "lint": {"status": "error", "output": "SyntaxError: invalid syntax"},
        })

        line = get_cute_tool_message("write_file", {"path": "/tmp/a.py"}, 0.1, result=result)

        assert "🟡" in line       # warning icon, not error
        assert "❌" not in line   # should NOT show error icon
        assert "lint error" in line

    def test_patch_lsp_diagnostics_shows_success_not_error(self):
        result = json.dumps({
            "success": True,
            "diff": "--- a/tmp.py\n+++ b/tmp.py\n",
            "lsp_diagnostics": "<diagnostics>ERROR [1:1] type mismatch</diagnostics>",
        })

        line = get_cute_tool_message("patch", {"path": "/tmp/a.py"}, 0.1, result=result)

        assert "✅" in line       # success icon
        assert "❌" not in line   # should NOT show error icon


class TestEditDiffPreview:
    def test_extract_edit_diff_for_patch(self):
        diff = extract_edit_diff("patch", '{"success": true, "diff": "--- a/x\\n+++ b/x\\n"}')
        assert diff is not None
        assert "+++ b/x" in diff

    def test_render_inline_unified_diff_colors_added_and_removed_lines(self):
        rendered = _render_inline_unified_diff(
            "--- a/cli.py\n"
            "+++ b/cli.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-old line\n"
            "+new line\n"
            " context\n"
        )

        assert "a/cli.py" in rendered[0]
        assert "b/cli.py" in rendered[0]
        assert any("old line" in line for line in rendered)
        assert any("new line" in line for line in rendered)
        assert any("48;2;" in line for line in rendered)

    def test_extract_edit_diff_ignores_non_edit_tools(self):
        assert extract_edit_diff("web_search", '{"diff": "--- a\\n+++ b\\n"}') is None

    def test_extract_edit_diff_uses_local_snapshot_for_write_file(self, tmp_path):
        target = tmp_path / "note.txt"
        target.write_text("old\n", encoding="utf-8")

        snapshot = capture_local_edit_snapshot("write_file", {"path": str(target)})

        target.write_text("new\n", encoding="utf-8")

        diff = extract_edit_diff(
            "write_file",
            '{"bytes_written": 4}',
            function_args={"path": str(target)},
            snapshot=snapshot,
        )

        assert diff is not None
        assert "--- a/" in diff
        assert "+++ b/" in diff
        assert "-old" in diff
        assert "+new" in diff

    def test_render_edit_diff_with_delta_invokes_printer(self):
        printer = MagicMock()

        rendered = render_edit_diff_with_delta(
            "patch",
            '{"diff": "--- a/x\\n+++ b/x\\n@@ -1 +1 @@\\n-old\\n+new\\n"}',
            print_fn=printer,
        )

        assert rendered is True
        assert printer.call_count >= 2
        calls = [call.args[0] for call in printer.call_args_list]
        assert any("a/x" in line and "b/x" in line for line in calls)
        assert any("old" in line for line in calls)
        assert any("new" in line for line in calls)

    def test_render_edit_diff_with_delta_skips_without_diff(self):
        rendered = render_edit_diff_with_delta(
            "patch",
            '{"success": true}',
        )

        assert rendered is False

    def test_render_edit_diff_with_delta_handles_renderer_errors(self, monkeypatch):
        printer = MagicMock()

        monkeypatch.setattr("agent.display._summarize_rendered_diff_sections", MagicMock(side_effect=RuntimeError("boom")))

        rendered = render_edit_diff_with_delta(
            "patch",
            '{"diff": "--- a/x\\n+++ b/x\\n"}',
            print_fn=printer,
        )

        assert rendered is False
        assert printer.call_count == 0

    def test_summarize_rendered_diff_sections_truncates_large_diff(self):
        diff = "--- a/x.py\n+++ b/x.py\n" + "".join(f"+line{i}\n" for i in range(120))

        rendered = _summarize_rendered_diff_sections(diff, max_lines=20)

        assert len(rendered) == 21
        assert "omitted" in rendered[-1]

    def test_summarize_rendered_diff_sections_limits_file_count(self):
        diff = "".join(
            f"--- a/file{i}.py\n+++ b/file{i}.py\n+line{i}\n"
            for i in range(8)
        )

        rendered = _summarize_rendered_diff_sections(diff, max_files=3, max_lines=50)

        assert any("a/file0.py" in line for line in rendered)
        assert any("a/file1.py" in line for line in rendered)
        assert any("a/file2.py" in line for line in rendered)
        assert not any("a/file7.py" in line for line in rendered)
        assert "additional file" in rendered[-1]


class TestGenericStatusField:
    """Plugins can declare status via _status in response JSON — no display.py branch needed."""

    def test_zero_results_via_status_field(self):
        """Plugin response with _status='warning' → 🟡 warning icon."""
        result = json.dumps({
            "matches": [], "count": 0, "query": "nothing",
            "_summary": "0 results", "_status": "warning",
        })
        # Use a non-builtin tool name to prove it hits generic fallback
        line = get_cute_tool_message("obsidian_search", {"query": "nothing"}, 0.1, result=result)
        assert "🟡" in line, f"Expected warning icon via _status, got: {line}"
        assert "✅" not in line

    def test_success_via_status_field(self):
        """Plugin response with _status='success' → ✅ success icon."""
        result = json.dumps({
            "matches": [{"path": "a.md"}], "count": 1, "query": "test",
            "_summary": "1 result", "_status": "success",
        })
        line = get_cute_tool_message("obsidian_search", {"query": "test"}, 0.1, result=result)
        assert "✅" in line
        assert "1 result" in line

    def test_error_with_success_false(self):
        """Plugin response with success=False → ❌ error (error path, not _status)."""
        result = json.dumps({"error": "API unreachable", "success": False})
        line = get_cute_tool_message("obsidian_search", {"query": "test"}, 0.1, result=result)
        assert "❌" in line

    def test_plain_content_shows_size(self):
        """Non-dict result > 100 chars → shows size + ✅."""
        content = "x" * 200
        line = get_cute_tool_message("obsidian_read", {"path": "note.md"}, 0.1, result=content)
        assert "✅" in line
        assert "200 chars" in line

    def test_invalid_status_ignored(self):
        """Invalid _status value is ignored, falls back to 'success'."""
        result = json.dumps({
            "matches": [], "count": 0,
            "_summary": "0 results", "_status": "bogus",
        })
        line = get_cute_tool_message("obsidian_search", {"query": "test"}, 0.1, result=result)
        assert "✅" in line  # invalid status → success
