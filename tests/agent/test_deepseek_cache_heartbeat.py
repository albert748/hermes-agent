"""Tests for agent/deepseek_cache_heartbeat.py — adaptive interval logic."""

import copy
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.deepseek_cache_heartbeat import (
    DEFAULT_INTERVAL_SECONDS,
    DEFAULT_MAX_CONSECUTIVE_MISSES,
    DEFAULT_MAX_UNREASONABLE_MISSES,
    HeartbeatState,
    SessionHeartbeatManager,
    UNREASONABLE_TOLERANCE_SECONDS,
)

# ── Test helpers ───────────────────────────────────────────────────────────


class _FakeUsage:
    """Simple fake for openai.types.chat.ChatCompletionUsage."""
    def __init__(self, prompt_tokens=0, prompt_cache_hit_tokens=0,
                 completion_tokens=0, total_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.prompt_cache_hit_tokens = prompt_cache_hit_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _FakeResponse:
    """Simple fake for openai.types.chat.ChatCompletion."""
    def __init__(self, usage: _FakeUsage):
        self.usage = usage


def _make_response(cache_hit_tokens: int = 0, prompt_tokens: int = 1000) -> _FakeResponse:
    """Create a fake API response with given cache and prompt token counts."""
    return _FakeResponse(_FakeUsage(
        prompt_tokens=prompt_tokens,
        prompt_cache_hit_tokens=cache_hit_tokens,
    ))


def _make_client(response: _FakeResponse):
    """Create a mock OpenAI client that returns the given response."""
    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


# ── HeartbeatState dataclass ──────────────────────────────────────────────

class TestHeartbeatState:
    def test_default_values(self):
        state = HeartbeatState(
            session_id="s1",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
        )
        assert state.session_id == "s1"
        assert state.interval_seconds == DEFAULT_INTERVAL_SECONDS
        assert state.consecutive_misses == 0
        assert state.disabled is False
        assert state.disabled_reason == ""

    def test_miss_increments(self):
        state = HeartbeatState(
            session_id="s1",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
        )
        state.consecutive_misses += 1
        state.unreasonable_misses += 1
        assert state.consecutive_misses == 1
        assert state.unreasonable_misses == 1


# ── SessionHeartbeatManager ───────────────────────────────────────────────

class TestSessionHeartbeatManager:
    def test_disabled_by_default(self):
        mgr = SessionHeartbeatManager()
        assert mgr.enabled is False
        assert mgr._running is False

    def test_enabled_starts_thread(self):
        mgr = SessionHeartbeatManager(enabled=True)
        mgr.start()
        assert mgr._running is True
        assert mgr._thread is not None
        assert mgr._thread.is_alive()
        mgr.stop()
        assert mgr._running is False

    def test_record_api_call_new_session(self):
        mgr = SessionHeartbeatManager(enabled=True)
        api_msgs = [{"role": "system", "content": "test prompt"}]
        mgr.record_api_call(
            session_id="s1",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
            api_messages=api_msgs,
        )
        assert "s1" in mgr._sessions
        state = mgr._sessions["s1"]
        assert state.session_id == "s1"
        assert state.last_api_messages == api_msgs
        assert state.last_api_call_time > 0

    def test_record_api_call_updates_existing(self):
        mgr = SessionHeartbeatManager(enabled=True)
        mgr.record_api_call("s1", "deepseek-chat", "https://api.deepseek.com/v1", "sk-xxx", [])
        old_time = mgr._sessions["s1"].last_api_call_time

        time.sleep(0.01)
        mgr.record_api_call("s1", "deepseek-chat", "https://api.deepseek.com/v1", "sk-xxx",
                            [{"role": "system", "content": "updated"}])

        state = mgr._sessions["s1"]
        assert state.last_api_call_time > old_time
        assert state.last_api_messages == [{"role": "system", "content": "updated"}]

    def test_re_enable_after_user_returns(self):
        mgr = SessionHeartbeatManager(enabled=True)
        mgr.record_api_call("s1", "deepseek-chat", "https://api.deepseek.com/v1", "sk-xxx", [])
        state = mgr._sessions["s1"]
        state.disabled = True
        state.consecutive_misses = 5

        # User comes back and sends a message
        mgr.record_api_call("s1", "deepseek-chat", "https://api.deepseek.com/v1", "sk-xxx",
                            [{"role": "system", "content": "hello"}])
        assert state.disabled is False
        assert state.consecutive_misses == 0
        assert state.disabled_reason == ""

    def test_record_does_nothing_when_disabled(self):
        mgr = SessionHeartbeatManager(enabled=False)
        mgr.record_api_call("s1", "deepseek-chat", "https://api.deepseek.com/v1", "sk-xxx", [])
        assert len(mgr._sessions) == 0

    def test_pause_session(self):
        mgr = SessionHeartbeatManager(enabled=True)
        mgr.record_api_call("s1", "deepseek-chat", "https://api.deepseek.com/v1", "sk-xxx", [])
        assert "s1" in mgr._sessions

        mgr.pause_session("s1")
        assert "s1" not in mgr._sessions

    def test_max_idle_removes_session(self):
        mgr = SessionHeartbeatManager(enabled=True, max_idle_seconds=0.1)
        mgr.record_api_call("s1", "deepseek-chat", "https://api.deepseek.com/v1", "sk-xxx", [])
        assert "s1" in mgr._sessions

        # Wait for max_idle to expire
        time.sleep(0.15)
        mgr._tick()
        assert "s1" not in mgr._sessions

    def test_disabled_session_not_pinged(self):
        mgr = SessionHeartbeatManager(enabled=True)
        mgr.record_api_call("s1", "deepseek-chat", "https://api.deepseek.com/v1", "sk-xxx", [])
        mgr._sessions["s1"].disabled = True

        # _tick should skip disabled sessions
        mock_ping = MagicMock()
        mgr._ping = mock_ping
        mgr._tick()
        mock_ping.assert_not_called()

    def test_consecutive_miss_adapts_interval(self):
        """After a miss, interval should decrease by 15 seconds."""
        mgr = SessionHeartbeatManager(
            enabled=True,
            interval_seconds=300,
            min_interval_seconds=60,
        )
        response = _make_response(cache_hit_tokens=0)
        mock_client = _make_client(response)

        state = HeartbeatState(
            session_id="s1",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
            last_api_messages=[{"role": "system", "content": "test"}],
            last_api_call_time=time.monotonic() - 310,  # expired
            interval_seconds=300,
        )
        mgr._sessions["s1"] = state
        mgr._get_client = lambda s: mock_client

        old_api_call_time = state.last_api_call_time
        mgr._ping(state)

        assert state.consecutive_misses == 1
        assert state.interval_seconds == 285  # 300 - 15
        assert state.pings_sent == 1
        # Ping (even a miss) resets the idle clock.
        assert state.last_api_call_time > old_api_call_time

    def test_consecutive_misses_disable(self):
        """After max_consecutive_misses, session should be disabled."""
        mgr = SessionHeartbeatManager(
            enabled=True,
            max_consecutive_misses=3,
            min_interval_seconds=60,
        )
        mock_client = _make_client(_make_response(cache_hit_tokens=0))

        state = HeartbeatState(
            session_id="s1",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
            last_api_messages=[{"role": "system", "content": "test"}],
            last_api_call_time=time.monotonic() - 310,
            interval_seconds=300,
            consecutive_misses=2,  # already 2 misses
        )
        mgr._sessions["s1"] = state
        mgr._get_client = lambda s: mock_client

        mgr._ping(state)

        # 3rd miss → should disable
        assert state.disabled is True
        assert "Consecutive misses" in state.disabled_reason

    def test_cache_hit_does_not_change_interval(self):
        """Cache hit should not modify interval or counters, but should
        update last_api_call_time (the ping itself refreshes the cache TTL)
        and MUST set _persist_dirty (v1.10 regression guard)."""
        mgr = SessionHeartbeatManager(enabled=True)
        mock_client = _make_client(_make_response(cache_hit_tokens=10000, prompt_tokens=12000))

        state = HeartbeatState(
            session_id="s1",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
            last_api_messages=[{"role": "system", "content": "test"}],
            last_api_call_time=time.monotonic() - 310,
            interval_seconds=300,
        )
        old_api_call_time = state.last_api_call_time
        mgr._sessions["s1"] = state
        mgr._get_client = lambda s: mock_client
        mgr._persist_dirty = False

        mgr._ping(state)

        assert state.cache_hits == 1
        assert state.consecutive_misses == 0
        assert state.interval_seconds == 300  # unchanged
        # The ping itself refreshes the server-side cache — the client
        # should record that time so the next idle check is accurate.
        assert state.last_api_call_time > old_api_call_time
        # v1.10 regression guard: HIT must trigger persist so SIGKILL
        # restart can restore the session.
        assert mgr._persist_dirty is True, (
            "BUG: cache HIT must set _persist_dirty=True"
        )

    def test_unreasonable_miss_detection(self):
        """Miss when pinging frequently but cache still expired → unreasonable."""
        mgr = SessionHeartbeatManager(
            enabled=True,
            max_unreasonable_misses=3,
        )
        mock_client = _make_client(_make_response(cache_hit_tokens=0))

        # Interval=60s but idle=500s:
        # 60 + 60 = 120 < 500 → UNREASONABLE
        # (we were pinging every 60s for 500s, cache should NOT have expired)
        state_frequent = HeartbeatState(
            session_id="s1",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
            last_api_messages=[{"role": "system", "content": "test"}],
            last_api_call_time=time.monotonic() - 500,
            interval_seconds=60,
        )
        mgr._get_client = lambda s: mock_client
        mgr._ping(state_frequent)
        assert state_frequent.unreasonable_misses == 1
        assert state_frequent.consecutive_misses == 0  # classified as unreasonable, not regular

        # Interval=300s but idle=120s:
        # 300 + 60 = 360 > 120 → NOT unreasonable
        # (we waited 300s but idle was only 120s, normal that cache expired)
        state_normal = HeartbeatState(
            session_id="s2",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
            last_api_messages=[{"role": "system", "content": "test"}],
            last_api_call_time=time.monotonic() - 120,
            interval_seconds=300,
        )
        mgr._ping(state_normal)
        assert state_normal.unreasonable_misses == 0
        assert state_normal.consecutive_misses == 1  # classified as regular miss

    def test_unreasonable_misses_disable(self):
        """3 unreasonable misses → disable."""
        mgr = SessionHeartbeatManager(
            enabled=True,
            max_unreasonable_misses=3,
        )
        mock_client = _make_client(_make_response(cache_hit_tokens=0))

        # interval=60s, idle=500s → unreasonable (60+60 < 500)
        state = HeartbeatState(
            session_id="s2",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
            last_api_messages=[{"role": "system", "content": "test"}],
            last_api_call_time=time.monotonic() - 500,
            interval_seconds=60,
            unreasonable_misses=2,
        )
        mgr._get_client = lambda s: mock_client

        mgr._ping(state)

        assert state.unreasonable_misses == 3
        assert state.disabled is True
        assert "Unreasonable misses" in state.disabled_reason

    def test_interval_cannot_go_below_min(self):
        """Adaptive reduction should floor at min_interval_seconds."""
        mgr = SessionHeartbeatManager(
            enabled=True,
            min_interval_seconds=60,
        )
        mock_client = _make_client(_make_response(cache_hit_tokens=0))

        # interval=65s, idle=90s: 65+60=125 > 90 → reasonable miss → interval decreases
        state = HeartbeatState(
            session_id="s1",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
            last_api_messages=[{"role": "system", "content": "test"}],
            last_api_call_time=time.monotonic() - 90,
            interval_seconds=65,  # close to floor
        )
        mgr._get_client = lambda s: mock_client

        mgr._ping(state)
        assert state.interval_seconds == 60  # 65 - 15 = 50 but floored at 60

    def test_ping_reuses_stored_api_kwargs(self):
        """When last_api_kwargs is stored, ping should use them not hardcoded params."""
        mgr = SessionHeartbeatManager(enabled=True)
        mock_client = _make_client(_make_response(cache_hit_tokens=2000))

        state = HeartbeatState(
            session_id="s1",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
            last_api_messages=[{"role": "system", "content": "test"}],
            last_api_call_time=time.monotonic() - 310,
            last_api_kwargs={
                "model": "deepseek-chat",
                "messages": [{"role": "system", "content": "original"}],
                "max_tokens": 4096,
                "temperature": 0.7,
                "extra_body": {"thinking": {"type": "enabled"}},
                "tools": [{"type": "function", "function": {"name": "test"}}],
                "tool_choice": "auto",
                "stream": True,
            },
        )
        mgr._get_client = lambda s: mock_client

        mgr._ping(state)

        # Verify client was called
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]

        # Messages should be the heartbeat messages (from api_kwargs["messages"] + tail)
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0] == {"role": "system", "content": "original"}
        assert call_kwargs["messages"][1]["content"].startswith("heartbeat only")

        # All original params should be preserved unchanged
        assert call_kwargs["model"] == "deepseek-chat"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
        assert call_kwargs["stream"] is True
        assert call_kwargs["tools"] == [{"type": "function", "function": {"name": "test"}}]
        assert call_kwargs["tool_choice"] == "auto"

    def test_record_api_call_stores_api_kwargs(self):
        """record_api_call should store sanitised api_kwargs."""
        mgr = SessionHeartbeatManager(enabled=True)
        mgr.record_api_call(
            session_id="s1",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
            api_messages=[{"role": "system", "content": "test"}],
            api_kwargs={
                "model": "deepseek-chat",
                "messages": [{"role": "system", "content": "test"}],
                "max_tokens": 4096,
                "extra_body": {"thinking": {"type": "enabled"}},
                "api_key": "sk-should-be-stripped",
                "api_secret": "secret-should-be-stripped",
            },
        )
        state = mgr._sessions["s1"]
        assert state.last_api_kwargs == {
            "model": "deepseek-chat",
            "messages": [{"role": "system", "content": "test"}],
            "max_tokens": 4096,
            "extra_body": {"thinking": {"type": "enabled"}},
        }
        assert "api_key" not in state.last_api_kwargs
        assert "api_secret" not in state.last_api_kwargs

    def test_ping_falls_back_when_no_stored_kwargs(self):
        """When last_api_kwargs is empty, ping should use hardcoded fallback."""
        mgr = SessionHeartbeatManager(enabled=True)
        mock_client = _make_client(_make_response(cache_hit_tokens=2000))

        # State with NO last_api_kwargs (empty dict — see dataclass default)
        state = HeartbeatState(
            session_id="s1",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
            last_api_messages=[{"role": "system", "content": "test"}],
            last_api_call_time=time.monotonic() - 310,
            # last_api_kwargs defaults to empty dict
        )
        mgr._get_client = lambda s: mock_client

        mgr._ping(state)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        # Fallback path: hardcoded model, temperature=0, extra_body with thinking disabled
        assert call_kwargs["model"] == "deepseek-chat"
        assert call_kwargs["temperature"] == 0
        assert call_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}

    def test_ping_handles_exception_gracefully(self):
        """Ping errors should not crash the heartbeat thread."""
        mgr = SessionHeartbeatManager(enabled=True)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Connection error")

        state = HeartbeatState(
            session_id="s1",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
            last_api_messages=[{"role": "system", "content": "test"}],
            last_api_call_time=time.monotonic() - 310,
        )
        mgr._get_client = lambda s: mock_client

        # Should not raise
        mgr._ping(state)
        assert state.pings_sent == 0  # no ping counted on error

    # ── Persist/restore contract tests (v1.10 regression guard) ──────────

    def test_cache_hit_sets_persist_dirty(self):
        """Cache HIT must set _persist_dirty so the updated last_api_call_time
        is written to disk on the next _maybe_persist() tick.

        Regression: v1.10 — HIT updated last_api_call_time in memory but
        never marked persist dirty.  On SIGKILL restart, the stale
        persist file caused _restore_sessions() to skip the session.
        """
        mgr = SessionHeartbeatManager(enabled=True)
        mock_client = _make_client(_make_response(
            cache_hit_tokens=10000, prompt_tokens=12000,
        ))

        state = HeartbeatState(
            session_id="s1",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
            last_api_messages=[{"role": "system", "content": "test"}],
            last_api_call_time=time.monotonic() - 310,
            interval_seconds=300,
        )
        mgr._get_client = lambda s: mock_client

        # Before ping, persist should be clean
        mgr._persist_dirty = False
        mgr._ping(state)

        assert state.cache_hits == 1
        assert state.last_api_call_time > 0
        assert mgr._persist_dirty is True, (
            "BUG: cache HIT must set _persist_dirty=True so the updated "
            "last_api_call_time is persisted to disk.  Without this, Gateway "
            "SIGKILL restart loses the fresh timestamp and _restore_sessions() "
            "skips the session."
        )

    def test_persist_writes_fresh_timestamp_after_cache_hit(self):
        """After cache HIT + _maybe_persist(), the persist file must contain
        a last_api_wall_time close to 'now' (not a stale value from hours ago).

        This is an integration test: it creates a temp persist file,
        triggers a HIT ping, calls _maybe_persist(), and inspects the
        file to verify the wall-clock timestamp was updated.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            persist_path = Path(f.name)

        try:
            mgr = SessionHeartbeatManager(enabled=True)
            mgr._persist_path = persist_path

            # Register a session (simulating record_api_call)
            state = HeartbeatState(
                session_id="s1",
                model="deepseek-chat",
                base_url="https://api.deepseek.com/v1",
                api_key="sk-xxx",
                last_api_messages=[{"role": "system", "content": "test"}],
                last_api_call_time=time.monotonic() - 310,
            )
            mgr._sessions["s1"] = state

            # Simulate a cache HIT ping
            mock_client = _make_client(_make_response(
                cache_hit_tokens=10000, prompt_tokens=12000,
            ))
            mgr._get_client = lambda s: mock_client
            mgr._ping(state)

            assert mgr._persist_dirty is True, "HIT must set _persist_dirty"

            # Trigger persist
            mgr._maybe_persist()
            assert mgr._persist_dirty is False, "persist should clear dirty flag"

            # Read the file and verify
            data = json.loads(persist_path.read_text(encoding="utf-8"))
            sessions = data.get("sessions", [])
            assert len(sessions) == 1, f"Expected 1 session, got {len(sessions)}"
            sdata = sessions[0]
            wall_time = sdata.get("last_api_wall_time", 0)
            now = time.time()
            age = now - wall_time
            assert age < 5, (
                f"last_api_wall_time age={age:.1f}s exceeds 5s — "
                f"persist file has stale timestamp (wall={wall_time}, now={now})"
            )
        finally:
            persist_path.unlink(missing_ok=True)

    def test_restore_session_after_cache_hit_and_simulated_restart(self):
        """Full cycle: HIT → persist → new Manager (simulated restart) → restore.

        Manager A: registers session, cache HIT ping, persist to temp file.
        Manager B: reads same persist file, restores session, verifies it's
        present with a recent last_api_call_time.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            persist_path = Path(f.name)

        try:
            # ── Manager A: register + HIT + persist ──
            mgr_a = SessionHeartbeatManager(enabled=True)
            mgr_a._persist_path = persist_path

            state_a = HeartbeatState(
                session_id="s1",
                model="deepseek-chat",
                base_url="https://api.deepseek.com/v1",
                api_key="sk-xxx",
                last_api_messages=[{"role": "system", "content": "test"}],
                last_api_call_time=time.monotonic() - 310,
            )
            mgr_a._sessions["s1"] = state_a

            mock_client = _make_client(_make_response(
                cache_hit_tokens=10000, prompt_tokens=12000,
            ))
            mgr_a._get_client = lambda s: mock_client
            mgr_a._ping(state_a)
            mgr_a._maybe_persist()

            # Verify file was written
            assert persist_path.exists()
            data_before = json.loads(persist_path.read_text(encoding="utf-8"))
            assert len(data_before.get("sessions", [])) == 1

            # ── Manager B: restore (simulated restart) ──
            mgr_b = SessionHeartbeatManager(enabled=True)
            mgr_b._persist_path = persist_path
            mgr_b._restore_sessions()

            assert "s1" in mgr_b._sessions, (
                "BUG: _restore_sessions() must restore session from persist "
                "file after cache HIT.  If this fails, SIGKILL restart loses "
                "the session."
            )
            restored = mgr_b._sessions["s1"]
            assert restored.session_id == "s1"
            assert restored.last_api_call_time > 0, (
                "Restored session must have a valid last_api_call_time "
                "(>0), not 0 or negative."
            )
            assert restored.last_api_messages == [
                {"role": "system", "content": "test"},
            ]
        finally:
            persist_path.unlink(missing_ok=True)

    # ── Regression: last_user_call_time (2026-05-10) ──────────────────────

    def test_remain_reflects_real_idle_not_heartbeat_idle(self):
        """remain must decrease toward 0 as the session ages, NOT stay constant.

        Bug: heartbeat ping resets last_api_call_time, so _idle_before is always
        ≈ interval_seconds, making remain = (max_idle - interval)/60 a constant.

        Fix: compute remain from last_user_call_time (only set by record_api_call,
        never by heartbeat pings), so it truly counts down as the session ages.
        """
        mgr = SessionHeartbeatManager(enabled=True, max_idle_seconds=30)
        mock_client = _make_client(_make_response(
            cache_hit_tokens=10000, prompt_tokens=12000,
        ))

        # Simulate a session whose last real API call was almost at max_idle
        state = HeartbeatState(
            session_id="s1",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
            last_api_messages=[{"role": "system", "content": "test"}],
            last_api_call_time=time.monotonic() - 5,  # heartbeat idle: 5s
            last_user_call_time=time.monotonic() - 28,  # real idle: 28s (near 30s max)
            interval_seconds=10,
            max_idle_seconds=30,
        )
        mgr._get_client = lambda s: mock_client
        mgr._sessions["s1"] = state

        mgr._ping(state)

        # Hard assertion: last_user_call_time must NOT have been updated by _ping()
        assert state.last_user_call_time < state.last_api_call_time, (
            f"BUG: heartbeat ping reset last_user_call_time "
            f"({state.last_user_call_time:.1f}) to or past "
            f"last_api_call_time ({state.last_api_call_time:.1f}). "
            f"Heartbeat pings must NOT reset the user call clock."
        )

    def test_max_idle_timeout_after_successful_heartbeats(self):
        """Successful heartbeat pings must NOT prevent max_idle timeout.

        Bug: heartbeat ping resets last_api_call_time, so _tick() always sees
        idle ≈ interval_seconds << max_idle, and the session never expires.

        Fix: _tick() uses last_user_call_time for the timeout check, which is
        never touched by _ping().
        """
        mgr = SessionHeartbeatManager(
            enabled=True,
            max_idle_seconds=0.1,
            min_interval_seconds=0.01,
            interval_seconds=0.05,
        )
        mock_client = _make_client(_make_response(
            cache_hit_tokens=10000, prompt_tokens=12000,
        ))

        state = HeartbeatState(
            session_id="s1",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
            last_api_messages=[{"role": "system", "content": "test"}],
            last_api_call_time=time.monotonic(),
            last_user_call_time=time.monotonic(),
            interval_seconds=0.05,
            max_idle_seconds=0.1,
        )
        mgr._get_client = lambda s: mock_client
        mgr._sessions["s1"] = state

        # Fire a successful heartbeat ping (old code resets last_api_call_time
        # which also serves as the max_idle clock).
        mgr._ping(state)
        assert state.cache_hits == 1

        # Wait for max_idle (0.1s) to expire from last_user_call_time
        time.sleep(0.15)

        # Run _tick() — must remove the session because real idle > max_idle
        mgr._tick()

        assert "s1" not in mgr._sessions, (
            "BUG: session survived beyond max_idle despite being idle. "
            "Heartbeat pings must NOT reset the max_idle clock. "
            f"idle from last_user_call_time = {time.monotonic() - state.last_user_call_time:.2f}s, "
            f"max_idle = 0.1s"
        )

    def test_record_api_call_inherits_chat_key_when_missing(self):
        """When chat_key is empty but a sibling session for the same
        platform+model+base_url already has one, inherit it so eviction works.

        Regression: if TG adapter fails to pass chat_key on a subsequent
        message, the new session would coexist with the old one, causing
        duplicate heartbeat pings for the same chat.
        """
        mgr = SessionHeartbeatManager(enabled=True)

        # First session: has chat_key
        mgr.record_api_call(
            session_id="s1",
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
            api_messages=[{"role": "system", "content": "msg1"}],
            chat_key="telegram:8211517881",
            platform="telegram",
        )
        assert len(mgr._sessions) == 1
        assert mgr._sessions["s1"].chat_key == "telegram:8211517881"

        # Second session: chat_key missing (TG adapter bug)
        mgr.record_api_call(
            session_id="s2",
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
            api_messages=[{"role": "system", "content": "msg2"}],
            chat_key="",           # <-- missing!
            platform="telegram",
        )

        # Old session should be evicted, new session inherits chat_key
        assert "s1" not in mgr._sessions, (
            "BUG: old session 's1' should be evicted when new session 's2' "
            "inherits its chat_key"
        )
        assert len(mgr._sessions) == 1
        assert mgr._sessions["s2"].chat_key == "telegram:8211517881", (
            "BUG: new session 's2' should inherit chat_key from evicted 's1'"
        )

    def test_stop_session_prevents_re_registration(self):
        """stop_session() must prevent record_api_call() from re-registering.

        When the user issues /new, stop_session() removes the session from
        heartbeat.  If an in-flight API call from the pre-/new agent
        completes later, record_api_call() must NOT re-register the session.
        """
        mgr = SessionHeartbeatManager(enabled=True)
        mgr.record_api_call(
            session_id="s1",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
            api_messages=[{"role": "system", "content": "hello"}],
        )
        assert "s1" in mgr._sessions

        # User runs /new → session is stopped
        mgr.stop_session("s1")
        assert "s1" not in mgr._sessions
        assert "s1" in mgr._stopped_sessions

        # Simulate in-flight API call completing after /new
        mgr.record_api_call(
            session_id="s1",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
            api_messages=[{"role": "system", "content": "late completion"}],
        )

        # MUST NOT be re-registered
        assert "s1" not in mgr._sessions, (
            "BUG: record_api_call re-registered a session that was "
            "explicitly stopped via stop_session().  In-flight API calls "
            "from the pre-/new agent must not resurrect dead sessions."
        )

    def test_stop_all_for_chat_prevents_re_registration(self):
        """stop_all_for_chat() must prevent record_api_call() re-registration."""
        mgr = SessionHeartbeatManager(enabled=True)
        # Register two sessions with different chat_keys (so chat_key
        # eviction inside record_api_call doesn't remove s1 prematurely).
        mgr.record_api_call(
            session_id="s1",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
            api_messages=[{"role": "system", "content": "hello"}],
            chat_key="telegram:8211517881",
        )
        mgr.record_api_call(
            session_id="s2",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
            api_messages=[{"role": "system", "content": "memsinker"}],
            chat_key="weixin:o9cq802XH2jh",   # different chat
        )
        assert len(mgr._sessions) == 2

        # /new on telegram → stop all sessions for that chat
        removed = mgr.stop_all_for_chat("telegram:8211517881")
        assert removed == 1  # only s1
        assert "s1" not in mgr._sessions
        assert "s2" in mgr._sessions  # weixin session unaffected

        # In-flight API call for s1 completes after /new
        mgr.record_api_call(
            session_id="s1",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
            api_messages=[{"role": "system", "content": "late"}],
            chat_key="telegram:8211517881",
        )

        # MUST NOT be re-registered
        assert "s1" not in mgr._sessions, (
            "BUG: record_api_call re-registered s1 after stop_all_for_chat"
        )
        assert "s2" in mgr._sessions  # weixin session still alive
