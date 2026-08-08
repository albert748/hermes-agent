"""Binary-detection regression tests for ShellFileOperations.read_file.

Bug: read_file samples the first 1000 *bytes* with ``head -c 1000``. When
that byte boundary splits a multi-byte UTF-8 character (CJK / emoji text),
the truncated stream decodes with errors="replace" to a single trailing
U+FFFD replacement char *at the end of the sample*. ``_is_likely_binary()``
mistook that truncation artifact for real corruption and reported a
perfectly valid UTF-8 text file as binary ("Binary file - cannot display
as text"), even though the file on disk decodes cleanly.

Fix: a trailing U+FFFD at the very end of the sample is the signature of a
byte-truncated multi-byte char (an incomplete sequence only ever decodes at
stream end). Real corruption produces U+FFFD in the middle of the sample,
which the check still catches.
"""
import subprocess

from tools.file_operations import ShellFileOperations


class ReplaceDecodeEnv:
    """Minimal terminal backend that decodes stdout with errors="replace",
    exactly like the real backends do (the U+FFFD arrives in the sample
    through this decode)."""

    def __init__(self, cwd):
        self.cwd = cwd

    def execute(self, command, cwd=None, **kwargs):
        proc = subprocess.run(
            command, shell=True, cwd=cwd or self.cwd,
            capture_output=True, text=False,
        )
        return {
            "output": proc.stdout.decode("utf-8", errors="replace"),
            "returncode": proc.returncode,
        }


# 3-byte CJK chars everywhere, so the 1000-byte sample boundary is
# extremely likely to split a character.
CJK_BODY = "## 基本信息\n- 女，2012年11月16日生\n- 肝旺脾虚，寒湿内伏\n" + "寒湿偏盛。" * 400


class TestReadFileTruncatedUtf8Boundary:

    def _ops(self, tmp_path):
        return ShellFileOperations(ReplaceDecodeEnv(str(tmp_path)), cwd=str(tmp_path))

    def test_cjk_file_with_split_utf8_boundary_reads_as_text(self, tmp_path):
        """Regression: a UTF-8 CJK .md file whose first-1000-byte boundary
        splits a multi-byte char must NOT be reported as binary."""
        path = tmp_path / "医案记录.md"
        raw = CJK_BODY.encode("utf-8")
        path.write_bytes(raw)
        # Fixture self-check: the 1000-byte boundary must actually split a
        # multi-byte char, otherwise this test proves nothing.
        sample = raw[:1000].decode("utf-8", errors="replace")
        assert "\ufffd" in sample, (
            "fixture does not split a multi-byte char at the 1000-byte "
            "boundary; adjust CJK_BODY"
        )
        # And the file on disk is valid UTF-8 (no real corruption).
        raw.decode("utf-8", errors="strict")

        result = self._ops(tmp_path).read_file(str(path))

        assert result.is_binary is False, (
            "Bug: read_file reported a valid UTF-8 CJK file as binary because "
            "head -c 1000 split a multi-byte char and the trailing U+FFFD "
            "truncation artifact was treated as file corruption"
        )
        assert result.error is None
        assert "基本信息" in result.content

    def test_likely_binary_ignores_trailing_replacement_char_from_truncation(self, tmp_path):
        """Unit: a sample whose *last* char is U+FFFD (head -c truncation
        artifact) must not be classified binary."""
        ops = self._ops(tmp_path)
        sample = "普通中文文本内容" * 80 + "\ufffd"
        assert ops._is_likely_binary("foo.md", sample) is False, (
            "Bug: trailing U+FFFD from byte truncation was classified as "
            "binary; only mid-sample replacement chars indicate corruption"
        )

    def test_likely_binary_still_detects_mid_sample_corruption(self, tmp_path):
        """Guard: real corruption (U+FFFD in the middle of the sample) is
        still classified binary — the fix must not weaken detection."""
        ops = self._ops(tmp_path)
        sample = "普通中文文本内容" * 40 + "\ufffd" + "更多普通文本" * 40
        assert ops._is_likely_binary("foo.md", sample) is True

    def test_likely_binary_still_detects_nul_bytes(self, tmp_path):
        """Guard: NUL-byte density detection (>30% non-printable) is
        unaffected by the fix."""
        ops = self._ops(tmp_path)
        sample = ("A\x00" * 100) + "tail"
        assert ops._is_likely_binary("foo.md", sample) is True

    def test_ascii_file_unaffected(self, tmp_path):
        """Guard: plain ASCII files keep reading as text."""
        path = tmp_path / "plain.txt"
        path.write_bytes(("line one\nline two\n" * 100).encode("ascii"))
        result = self._ops(tmp_path).read_file(str(path))
        assert result.is_binary is False
        assert "line one" in result.content
