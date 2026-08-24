"""api_usage 记录功能测试：每次 API 调用 usage（含 cache 分解）写入独立库。

覆盖：
- model_dump 对象（OpenAI SDK CompletionUsage 形态）
- SimpleNamespace（vars 形态）
- usage=None 不建库不写入
- 非法 usage 对象 best-effort 吞异常
"""

import os
import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.chat_completion_helpers import _record_api_usage


class _FakeUsage:
    """模拟 OpenAI SDK CompletionUsage（pydantic，有 model_dump）。"""

    def __init__(self, hit, miss, prompt, completion):
        self.prompt_cache_hit_tokens = hit
        self.prompt_cache_miss_tokens = miss
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion

    def model_dump(self):
        return {
            "prompt_cache_hit_tokens": self.prompt_cache_hit_tokens,
            "prompt_cache_miss_tokens": self.prompt_cache_miss_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def _agent(sid="sess-abc"):
    a = MagicMock()
    a.session_id = sid
    return a


def _read_rows(db):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(
            "SELECT session_id, model, prompt_tokens, cache_hit_tokens, "
            "cache_miss_tokens, completion_tokens FROM api_usage"
        ).fetchall()
    finally:
        conn.close()


def test_record_pydantic_usage(tmp_path):
    """Bug: model_dump 形态的 usage（SDK 实际返回）应完整落库。"""
    db = tmp_path / "usage.db"
    _record_api_usage(
        _agent("sess-1"), "deepseek-v4-flash",
        _FakeUsage(hit=1000, miss=500, prompt=1500, completion=200),
        db_path=str(db),
    )
    rows = _read_rows(db)
    assert len(rows) == 1, f"Bug: rows={rows}"
    assert rows[0] == ("sess-1", "deepseek-v4-flash", 1500, 1000, 500, 200), (
        f"Bug: row={rows[0]}"
    )


def test_record_namespace_usage(tmp_path):
    """Bug: SimpleNamespace 形态（流式 stub 返回）也应落库。"""
    db = tmp_path / "usage.db"
    ns = SimpleNamespace(
        prompt_tokens=10, completion_tokens=5, total_tokens=15,
        prompt_cache_hit_tokens=8, prompt_cache_miss_tokens=2,
    )
    _record_api_usage(_agent(), "m", ns, db_path=str(db))
    rows = _read_rows(db)
    assert len(rows) == 1, f"Bug: rows={rows}"
    assert rows[0][3] == 8 and rows[0][4] == 2, f"Bug: row={rows[0]}"


def test_none_usage_is_noop(tmp_path):
    """Bug: usage=None（调用失败/无 usage）不应建库。"""
    db = tmp_path / "usage.db"
    _record_api_usage(_agent(), "m", None, db_path=str(db))
    assert not os.path.exists(db), "Bug: None usage 不应创建 db"


def test_bad_usage_swallowed(tmp_path):
    """Bug: 非法 usage 对象（非 dict/非对象）应被吞掉，不抛异常。"""
    db = tmp_path / "usage.db"
    _record_api_usage(_agent(), "m", "not-a-usage-object", db_path=str(db))
    conn = sqlite3.connect(db)
    try:
        n = conn.execute("SELECT COUNT(*) FROM api_usage").fetchone()[0]
    except sqlite3.OperationalError:
        n = 0  # 建表失败被吞 → 无表
    conn.close()
    assert n == 0, f"Bug: rows={n}"


def test_multiple_calls_append(tmp_path):
    """Bug: 多次调用应追加多行（每次 API 调用一条）。"""
    db = tmp_path / "usage.db"
    for i in range(3):
        _record_api_usage(
            _agent("sess-x"), "m",
            _FakeUsage(hit=1, miss=1, prompt=2, completion=1),
            db_path=str(db),
        )
    rows = _read_rows(db)
    assert len(rows) == 3, f"Bug: rows={rows}"
