"""DeepSeek prompt cache heartbeat.

DeepSeek uses automatic server-side prefix caching with a ~5-minute TTL.
During interactive sessions, idle periods longer than the TTL cause cache
misses on the next turn, re-billing all input tokens at full price.

This module provides ``SessionHeartbeatManager`` — a background daemon thread
that sends minimal keepalive requests to keep the DeepSeek cache warm for
extended sessions (default: 3 hours).

Architecture:
  - One manager per process (CLI or Gateway).
  - Each DeepSeek session is registered after a successful API call.
  - A background thread wakes every ``interval_seconds`` and pings sessions
    that have been idle longer than the interval.
  - Adaptive logic: cache misses decrease the interval; unreasonable misses
    or consecutive failures disable the heartbeat for that session.

The heartbeat request re-uses the exact messages prefix from the last real
API call, appending ``"heartbeat only — respond with 'ok'"`` so the model
produces minimal output (``max_tokens=10``, ``thinking=disabled``).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Configuration defaults ────────────────────────────────────────────────

DEFAULT_INTERVAL_SECONDS = 300       # 5 min — test first, then tune down
DEFAULT_MIN_INTERVAL_SECONDS = 60    # absolute floor
DEFAULT_MAX_CONSECUTIVE_MISSES = 10  # consecutive misses → disable
DEFAULT_MAX_UNREASONABLE_MISSES = 3  # unreasonable misses → disable
DEFAULT_MAX_IDLE_MINUTES = 180       # stop heartbeat after 3 h of silence
UNREASONABLE_TOLERANCE_SECONDS = 60  # if interval + this < idle, it's unreasonable
MAX_RESTORE_IDLE_SECONDS = 600       # don't restore sessions idle > 10 min


@dataclass
class HeartbeatState:
    """Per-session heartbeat tracking."""

    session_id: str
    model: str
    base_url: str
    api_key: str = field(repr=False)
    chat_key: str = ""  # e.g. "telegram:8211517881" — evict old sessions for same chat

    # The exact api_messages list sent in the last real API call.
    # Used as the prefix for heartbeat requests.
    last_api_messages: List[Dict[str, Any]] = field(default_factory=list)

    # The full api_kwargs dict from the last real API call (sanitised:
    # api_key is stripped).  The heartbeat reuses these kwargs to
    # preserve the exact request shape (thinking mode, temperature,
    # etc.) so the server-side cache prefix matches perfectly.
    last_api_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Timing
    last_api_call_time: float = 0.0
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS

    # Miss tracking
    consecutive_misses: int = 0
    unreasonable_misses: int = 0

    # Stats (informational)
    pings_sent: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    # Disabled flag
    disabled: bool = False
    disabled_reason: str = ""


class SessionHeartbeatManager:
    """Background heartbeat manager for DeepSeek prompt cache.

    Lifecycle:
        manager = SessionHeartbeatManager(enabled=True)
        manager.start()          # starts background thread
        ...
        manager.record_api_call(session_id, model, base_url, api_key,
                                api_messages)
        ...
        manager.stop()           # stops background thread
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS,
        max_consecutive_misses: int = DEFAULT_MAX_CONSECUTIVE_MISSES,
        max_unreasonable_misses: int = DEFAULT_MAX_UNREASONABLE_MISSES,
        max_idle_seconds: float = DEFAULT_MAX_IDLE_MINUTES * 60,
    ):
        self.enabled = enabled
        self.interval_seconds = interval_seconds
        self.min_interval_seconds = min_interval_seconds
        self.max_consecutive_misses = max_consecutive_misses
        self.max_unreasonable_misses = max_unreasonable_misses
        self.max_idle_seconds = max_idle_seconds

        # session_id → HeartbeatState
        self._sessions: Dict[str, HeartbeatState] = {}
        self._lock = threading.Lock()

        # Event to wake the background thread immediately on stop().
        # Replaces time.sleep() polling so the thread responds within
        # milliseconds instead of up to interval_seconds * 0.5.
        self._stop_event = threading.Event()

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # For CLI mode: direct reference to the current AIAgent so we can
        # call agent._create_openai_client() (avoids duplicating client setup).
        self._agent_ref: Any = None

        # Persistence: save/restore session states across Gateway restarts.
        self._persist_path: Optional[Path] = None
        self._persist_dirty: bool = False
        self._last_persist_time: float = 0.0
        self._persist_interval: float = 30.0  # persist at most every 30s

    # ── Public API ────────────────────────────────────────────────────────

    def bind_agent(self, agent: Any) -> None:
        """Bind to an AIAgent instance (CLI mode).

        In CLI mode the agent is long-lived.  Binding lets the heartbeat
        reuse the agent's OpenAI client factory instead of duplicating
        credential resolution.
        """
        self._agent_ref = agent

    def record_api_call(
        self,
        session_id: str,
        model: str,
        base_url: str,
        api_key: str,
        api_messages: List[Dict[str, Any]],
        api_kwargs: Optional[Dict[str, Any]] = None,
        chat_key: str = "",
    ) -> None:
        """Record a successful API call for a session.

        Call this after every real chat.completions.create() that completes
        successfully (regardless of cache hit/miss).  The *api_messages*
        argument must be the exact messages list that was sent.

        If the session was previously disabled, recording a new API call
        re-enables it (user is back).

        If *chat_key* is provided and differs from any existing session's
        chat_key, the old session(s) for that chat are evicted.  This
        handles /new (new session replaces old) and prevents dead sessions
        from consuming heartbeat pings.
        """
        if not self.enabled:
            return

        now = time.monotonic()
        with self._lock:
            # Evict old sessions for the same chat (e.g. /new).
            # Two-pass: first backfill chat_key on sessions created
            # before this feature existed, then evict all matches.
            if chat_key:
                # Pass 1: backfill — update chat_key on sessions that
                # don't have one yet but share the same model/base_url.
                for sid, st in list(self._sessions.items()):
                    if not st.chat_key and st.model == model and st.base_url == base_url:
                        st.chat_key = chat_key
                        logger.debug(
                            "Heartbeat: backfilled chat_key=%s on session %s",
                            chat_key, sid,
                        )

                # Pass 2: evict all sessions for this chat (except the current one).
                for sid, st in list(self._sessions.items()):
                    if st.chat_key == chat_key and sid != session_id:
                        logger.info(
                            "Heartbeat: evicting old session %s for chat %s (replaced by %s)",
                            sid, chat_key, session_id,
                        )
                        del self._sessions[sid]
                        self._persist_dirty = True

            state = self._sessions.get(session_id)
            if state is None:
                state = HeartbeatState(
                    session_id=session_id,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    interval_seconds=self.interval_seconds,
                    chat_key=chat_key,
                )
                self._sessions[session_id] = state
            elif chat_key:
                state.chat_key = chat_key

            state.last_api_messages = copy.deepcopy(api_messages)
            state.last_api_call_time = now
            if api_kwargs:
                _safe_kwargs = {k: v for k, v in api_kwargs.items()
                                if k not in ("api_key", "api_secret")}
                state.last_api_kwargs = copy.deepcopy(_safe_kwargs)
            state.model = model
            state.base_url = base_url
            state.api_key = api_key

            self._persist_dirty = True

            # Re-enable if it was disabled (user came back)
            if state.disabled:
                state.disabled = False
                state.disabled_reason = ""
                state.consecutive_misses = 0
                state.unreasonable_misses = 0
                state.interval_seconds = self.interval_seconds
                logger.info(
                    "DeepSeek heartbeat re-enabled for session %s (user returned)",
                    session_id,
                )

    def start(self, persist_path: Optional[Path] = None) -> None:
        """Start the background heartbeat thread (idempotent).

        If *persist_path* is provided, session states are saved to and
        restored from this file on start/stop, allowing heartbeat state
        to survive Gateway restarts.
        """
        if not self.enabled:
            return
        if self._running:
            return

        if persist_path is not None:
            self._persist_path = Path(persist_path)

        # Restore persisted sessions (only sessions idle < MAX_RESTORE_IDLE_SECONDS)
        self._restore_sessions()

        # If a previous thread is still shutting down, wait for it.
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="ds-cache-heartbeat",
        )
        self._thread.start()
        logger.info("DeepSeek heartbeat manager started (interval=%ds)", self.interval_seconds)

    def stop(self) -> None:
        """Stop the background heartbeat thread (idempotent).

        Signals the thread via ``_stop_event`` so it wakes from
        ``wait()`` within milliseconds rather than blocking until
        the next tick interval. Persists session state before stopping.
        """
        self._running = False
        self._stop_event.set()  # wake sleeping thread immediately
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

        # Force-persist on shutdown so the next start can restore state.
        self._persist_sessions(force=True)

    def pause_session(self, session_id: str) -> None:
        """Temporarily remove a session from heartbeat rotation.

        Called when a new conversation turn starts (the real API call will
        refresh the cache).  The session is re-added via record_api_call().
        """
        with self._lock:
            self._sessions.pop(session_id, None)

    def stop_session(self, session_id: str) -> None:
        """Permanently remove a session from heartbeat rotation and persist.

        Called when the user issues /new — the old session is dead and
        should never be pinged again.  Unlike ``pause_session``, this
        also force-persists so the removal survives Gateway restarts.

        Idempotent: safe to call for sessions that are not registered.
        """
        with self._lock:
            if session_id in self._sessions:
                logger.info(
                    "Heartbeat: stopping heartbeat for session %s (/new)",
                    session_id,
                )
                del self._sessions[session_id]
                self._persist_dirty = True
        self._persist_sessions(force=True)

    # ── Persistence ───────────────────────────────────────────────────────

    def _maybe_persist(self) -> None:
        """Persist if dirty and throttle interval elapsed."""
        if not self._persist_path or not self._persist_dirty:
            return
        now = time.monotonic()
        if now - self._last_persist_time < self._persist_interval:
            return
        self._persist_sessions()

    def _persist_sessions(self, force: bool = False) -> None:
        """Write session states to disk atomically.

        Only writes if dirty (or forced).  Strips api_key for security —
        restored sessions re-resolve credentials from env/config.
        """
        if not self._persist_path:
            return
        if not force and not self._persist_dirty:
            return

        with self._lock:
            sessions_data: List[Dict[str, Any]] = []
            for sid, state in self._sessions.items():
                if state.disabled:
                    continue  # don't persist disabled sessions
                sessions_data.append({
                    "session_id": sid,
                    "model": state.model,
                    "base_url": state.base_url,
                    "chat_key": state.chat_key,
                    "interval_seconds": state.interval_seconds,
                    "last_api_wall_time": time.time() - (time.monotonic() - state.last_api_call_time),
                    "last_api_messages": state.last_api_messages,
                    "last_api_kwargs": state.last_api_kwargs,
                    "pings_sent": state.pings_sent,
                    "cache_hits": state.cache_hits,
                    "cache_misses": state.cache_misses,
                })

        data = {"sessions": sessions_data, "updated_at": time.time()}
        try:
            tmp_path = self._persist_path.with_suffix(".tmp")
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(self._persist_path)
            self._persist_dirty = False
            self._last_persist_time = time.monotonic()
        except Exception as exc:
            logger.debug("Heartbeat persist failed: %s", exc)

    def _restore_sessions(self) -> None:
        """Restore persisted sessions on startup.

        Only restores sessions that have been idle less than
        MAX_RESTORE_IDLE_SECONDS — older sessions' caches have
        certainly expired and shouldn't be re-pinged.
        """
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("Heartbeat restore read failed: %s", exc)
            return

        now = time.monotonic()
        # Convert persisted wall-clock times back to monotonic.
        # time.monotonic() resets on process restart, so persisted
        # monotonic values are meaningless in a new process.
        mono_offset = time.time() - now
        restored = 0
        skipped_old = 0
        for sdata in data.get("sessions", []):
            sid = sdata.get("session_id")
            if not sid:
                continue
            # Use persisted wall-clock time (new format) with fallback
            # to old monotonic field for backward compatibility.
            last_api_wall = sdata.get("last_api_wall_time", 0)
            if not last_api_wall:
                last_api_wall = sdata.get("last_api_call_time", 0)
            last_api_mono = max(0, last_api_wall - mono_offset)
            idle = now - last_api_mono
            if idle > MAX_RESTORE_IDLE_SECONDS:
                skipped_old += 1
                continue

            state = HeartbeatState(
                session_id=sid,
                model=sdata.get("model", ""),
                base_url=sdata.get("base_url", ""),
                api_key="",  # re-resolved on first ping via _resolve_api_key
                interval_seconds=sdata.get("interval_seconds", self.interval_seconds),
                chat_key=sdata.get("chat_key", ""),
            )
            state.last_api_messages = sdata.get("last_api_messages", [])
            state.last_api_kwargs = sdata.get("last_api_kwargs", {})
            state.last_api_call_time = last_api_mono
            state.pings_sent = sdata.get("pings_sent", 0)
            state.cache_hits = sdata.get("cache_hits", 0)
            state.cache_misses = sdata.get("cache_misses", 0)
            self._sessions[sid] = state
            restored += 1

        # Deduplicate: TG is single-channel — keep only the most recently
        # active session per chat_key (or per model@base_url for pre-chat_key
        # persist files).  This cleans up dead sessions left by /new.
        if len(self._sessions) > 1:
            by_key: Dict[str, HeartbeatState] = {}
            evicted = 0
            for sid, state in list(self._sessions.items()):
                # Fallback dedup key for pre-chat_key persist files
                key = state.chat_key or f"{state.model}@{state.base_url}"
                existing = by_key.get(key)
                if existing is None or state.last_api_call_time > existing.last_api_call_time:
                    by_key[key] = state
                    if existing is not None:
                        evicted += 1
                else:
                    evicted += 1
            if evicted > 0:
                self._sessions = {s.session_id: s for s in by_key.values()}
                logger.info(
                    "Heartbeat: deduped %d stale session(s) during restore",
                    evicted,
                )
                self._persist_dirty = True  # clean up the persist file too

        if restored > 0:
            logger.info(
                "Heartbeat: restored %d session(s)%s from %s",
                restored,
                f" (skipped {skipped_old} idle > {MAX_RESTORE_IDLE_SECONDS}s)" if skipped_old else "",
                self._persist_path,
            )

    # ── Internal ──────────────────────────────────────────────────────────

    def _loop(self) -> None:
        """Main heartbeat loop — runs in daemon thread.

        Uses ``_stop_event.wait()`` instead of ``time.sleep()`` so
        ``stop()`` can wake the thread immediately instead of waiting
        up to ``interval_seconds * 0.5`` seconds.
        """
        while self._running:
            # Wait for the next tick, but break immediately if stopped.
            if self._stop_event.wait(self.interval_seconds * 0.5):
                break
            if not self._running:
                break
            self._tick()

    def _tick(self) -> None:
        """Scan sessions and ping those that need it."""
        now = time.monotonic()
        to_ping: List[HeartbeatState] = []

        # Persist if dirty and enough time has passed.
        self._maybe_persist()

        with self._lock:
            session_count = len(self._sessions)
            if session_count == 0:
                logger.warning("DeepSeek heartbeat _tick: 0 sessions registered — nothing to ping (is record_api_call called?)")
                return

            for sid, state in list(self._sessions.items()):
                if state.disabled:
                    continue

                idle = now - state.last_api_call_time

                # Stop heartbeat after max_idle_seconds
                if idle >= self.max_idle_seconds:
                    logger.debug(
                        "DeepSeek heartbeat: session %s idle %.0f min — removing",
                        sid, idle / 60,
                    )
                    del self._sessions[sid]
                    continue

                # Ping if idle exceeds current interval
                if idle >= state.interval_seconds:
                    to_ping.append(state)

            if to_ping:
                logger.warning(
                    "DeepSeek heartbeat _tick: %d session(s) to ping (total=%d, now=%.0f)",
                    len(to_ping), session_count, now,
                )
            elif session_count > 0:
                # Report state so we can see why no ping
                for sid, state in list(self._sessions.items())[:3]:  # max 3 to avoid spam
                    idle = now - state.last_api_call_time
                    logger.warning(
                        "DeepSeek heartbeat _tick: session %s idle=%.0fs interval=%.0fs → %s",
                        sid[:12] if sid else "?", idle, state.interval_seconds,
                        "NOT ready (idle < interval)" if idle < state.interval_seconds else "DISABLED" if state.disabled else "UNKNOWN",
                    )

        for state in to_ping:
            self._ping(state)

    def _ping(self, state: HeartbeatState) -> None:
        """Send one heartbeat ping for a session."""
        try:
            logger.warning(
                "DeepSeek heartbeat _ping: starting for session %s",
                state.session_id[:12] if state.session_id else "?",
            )
            hb_messages = copy.deepcopy(state.last_api_messages)
            if not hb_messages:
                return  # nothing to send

            hb_messages.append({
                "role": "user",
                "content": "heartbeat only — respond with 'ok' and nothing else.",
            })

            client = self._get_client(state)
            if client is None:
                return

            # Build API kwargs — prefer reusing the stored kwargs from
            # the last real API call so the request shape (thinking mode,
            # temperature, etc.) matches exactly and the server-side cache
            # prefix stays valid.
            if state.last_api_kwargs:
                api_kwargs = copy.deepcopy(state.last_api_kwargs)
                actual_messages = api_kwargs.get("messages")
                if actual_messages and isinstance(actual_messages, list):
                    hb_messages = copy.deepcopy(actual_messages)
                else:
                    hb_messages = copy.deepcopy(state.last_api_messages)
                hb_messages.append({
                    "role": "user",
                    "content": "heartbeat only — respond with 'ok' and nothing else.",
                })
                api_kwargs["messages"] = hb_messages
                response = client.chat.completions.create(**api_kwargs)
            else:
                # Fallback: hardcoded minimal call (no stored kwargs yet).
                response = client.chat.completions.create(
                    model=state.model,
                    messages=hb_messages,
                    temperature=0,
                    extra_body={"thinking": {"type": "disabled"}},
                )

            state.pings_sent += 1
            usage = getattr(response, "usage", None)

            prompt_tokens = 0
            cache_hit_tokens = 0
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                # DeepSeek reports cache hits in prompt_cache_hit_tokens
                cache_hit_tokens = (
                    getattr(usage, "prompt_cache_hit_tokens", 0)
                    or getattr(usage, "cache_read_tokens", 0)
                    or 0
                )

            if cache_hit_tokens > 0:
                # Cache hit — all good.  Update last_api_call_time because
                # the heartbeat request itself refreshes the server-side
                # cache TTL — the next ping should measure idle from NOW.
                state.cache_hits += 1
                state.last_api_call_time = time.monotonic()
                self._persist_dirty = True  # persist updated timestamp (v1.10 regression fix)
                # TODO: once stable, demote to DEBUG
                logger.warning(
                    "DeepSeek heartbeat #%d → cache HIT (%d/%d tokens, intv=%ds) sid=%s %s",
                    state.pings_sent, cache_hit_tokens, prompt_tokens, state.interval_seconds,
                    state.session_id or "?",
                    f"chat={state.chat_key}" if state.chat_key else "",
                )
            else:
                # Cache miss — diagnose and adapt.
                # The heartbeat request still refreshed the server-side state,
                # so update last_api_call_time regardless.
                state.cache_misses += 1
                now = time.monotonic()
                idle_seconds = now - state.last_api_call_time

                is_unreasonable = (
                    state.interval_seconds + UNREASONABLE_TOLERANCE_SECONDS
                    < idle_seconds
                )

                if is_unreasonable:
                    state.unreasonable_misses += 1
                    logger.warning(
                        "DeepSeek heartbeat UNREASONABLE MISS #%d/%d "
                        "(interval=%ds, idle=%ds, prompt=%d tokens)",
                        state.unreasonable_misses,
                        self.max_unreasonable_misses,
                        state.interval_seconds,
                        int(idle_seconds),
                        prompt_tokens,
                    )

                    if state.unreasonable_misses >= self.max_unreasonable_misses:
                        state.disabled = True
                        state.disabled_reason = (
                            f"Unreasonable misses: {state.unreasonable_misses} misses "
                            f"despite interval={state.interval_seconds}s << idle={int(idle_seconds)}s"
                        )
                        logger.error(
                            "DeepSeek heartbeat DISABLED for session %s: %s",
                            state.session_id, state.disabled_reason,
                        )
                else:
                    state.consecutive_misses += 1
                    old_interval = state.interval_seconds
                    state.interval_seconds = max(
                        state.interval_seconds - 15,
                        self.min_interval_seconds,
                    )
                    logger.warning(
                        "DeepSeek heartbeat MISS #%d/%d → interval %ds→%ds "
                        "(idle=%ds, prompt=%d tokens) sid=%s %s",
                        state.consecutive_misses,
                        self.max_consecutive_misses,
                        int(old_interval),
                        int(state.interval_seconds),
                        int(idle_seconds),
                        prompt_tokens,
                        state.session_id or "?",
                        f"chat={state.chat_key}" if state.chat_key else "",
                    )

                    if state.consecutive_misses >= self.max_consecutive_misses:
                        state.disabled = True
                        state.disabled_reason = (
                            f"Consecutive misses: {state.consecutive_misses} misses, "
                            f"interval reduced to {state.interval_seconds}s"
                        )
                        logger.error(
                            "DeepSeek heartbeat DISABLED for session %s: %s",
                            state.session_id, state.disabled_reason,
                        )

                # The heartbeat request itself resets the server-side idle
                # clock — measure next idle from now.
                state.last_api_call_time = now
                self._persist_sessions(force=True)  # persist immediately — ping is ~285s, no need to defer

        except Exception as exc:
            logger.debug(
                "DeepSeek heartbeat ping failed for session %s: %s",
                state.session_id, exc,
            )

    def _get_client(self, state: HeartbeatState) -> Any:
        """Create or reuse an OpenAI client for heartbeat pings.

        In CLI mode, delegate to the bound agent to reuse its client
        factory (handles credential pools, proxy config, etc.).
        """
        agent = self._agent_ref
        if agent is not None:
            try:
                return agent._create_openai_client(
                    agent._client_kwargs,
                    reason="heartbeat",
                    shared=False,
                )
            except Exception:
                pass

        # Fallback: create a standalone client (used in Gateway mode)
        try:
            from openai import OpenAI
            api_key = state.api_key or self._resolve_api_key(state.base_url)
            if not api_key:
                return None
            return OpenAI(
                base_url=state.base_url,
                api_key=api_key,
                timeout=30,
            )
        except Exception:
            return None

    @staticmethod
    def _resolve_api_key(base_url: str) -> str:
        """Resolve API key for a restored session from environment.

        Called when state.api_key is empty (session was restored from
        persist file without credentials).  Tries DeepSeek-specific
        env vars first, then looks for a matching base_url in .env.
        """
        key = os.getenv("DEEPSEEK_API_KEY", "")
        if key:
            return key
        key = os.getenv("DEEPSEEK_APIKEY", "")
        if key:
            return key
        # Fallback: read .env and match by base_url host
        try:
            env_path = os.path.expanduser("~/.hermes/.env")
            if os.path.isfile(env_path):
                with open(env_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        v = v.strip().strip("'").strip('"')
                        if "deepseek" in k.lower() and v.startswith("sk-"):
                            return v
        except Exception:
            pass
        return ""
