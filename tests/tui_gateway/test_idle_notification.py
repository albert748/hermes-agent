"""TUI idle-completion notification (parity with cli.py idle_notification).

Covers:
- poller fires the completion push once the idle timeout expires
- no fire before the timeout / when disabled
- exactly-once semantics (sent flag prevents repeats)
- _send_idle_notification message construction + success bookkeeping
"""

import threading
import time

import pytest

import tui_gateway.server as server


def _idle_cfg(**overrides) -> dict:
    cfg = {
        "enabled": True,
        "timeout": 0.2,
        "max_summary_chars": 500,
        "platforms": ["weixin", "feishu", "email"],
    }
    cfg.update(overrides)
    return cfg


def _session(**overrides) -> dict:
    sess = {
        "session_key": "test-idle-session",
        "history_lock": threading.Lock(),
        "running": False,
        "idle_notif_start": 0.0,
        "idle_notif_sent": False,
        "last_assistant_response": "",
    }
    sess.update(overrides)
    return sess


class TestPollerIdleCheck:
    def _start_poller(self, session: dict, monkeypatch, cfg: dict) -> tuple:
        imports_stop = threading.Event()
        fired = []
        monkeypatch.setattr(server, "_load_cfg", lambda: {"cli": {"idle_notification": cfg}})
        monkeypatch.setattr(server, "_KANBAN_POLL_SECONDS", 60.0)

        def _fake_send(sid, sess, _cfg):
            fired.append((sid, sess))
            sess["idle_notif_sent"] = True  # simulate success bookkeeping

        monkeypatch.setattr(server, "_send_idle_notification", _fake_send)
        monkeypatch.setattr(
            server, "_emit", lambda *a, **k: None
        )
        stop = threading.Event()
        thread = threading.Thread(
            target=server._notification_poller_loop,
            args=(stop, "sid-idle-test", session),
            daemon=True,
        )
        thread.start()
        return stop, thread, fired, imports_stop

    @staticmethod
    def _wait_for(predicate, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.02)
        return False

    def test_fires_after_timeout(self, monkeypatch):
        session = _session(
            idle_notif_start=time.time() - 1.0,
            last_assistant_response="long task done",
        )
        stop, thread, fired, _ = self._start_poller(session, monkeypatch, _idle_cfg())
        try:
            assert self._wait_for(lambda: fired), "idle notification never fired"
        finally:
            stop.set()
            thread.join(timeout=5)
        assert fired[0][0] == "sid-idle-test"
        assert fired[0][1] is session

    def test_does_not_fire_before_timeout(self, monkeypatch):
        session = _session(
            idle_notif_start=time.time(),  # armed just now
            last_assistant_response="still fresh",
        )
        stop, thread, fired, _ = self._start_poller(session, monkeypatch, _idle_cfg())
        try:
            # Give the poller several ticks; nothing should fire.
            time.sleep(0.4)
            assert not fired
        finally:
            stop.set()
            thread.join(timeout=5)

    def test_disabled_never_fires(self, monkeypatch):
        session = _session(
            idle_notif_start=time.time() - 10.0,
            last_assistant_response="stale but disabled",
        )
        stop, thread, fired, _ = self._start_poller(
            session, monkeypatch, _idle_cfg(enabled=False)
        )
        try:
            time.sleep(0.4)
            assert not fired
        finally:
            stop.set()
            thread.join(timeout=5)

    def test_fires_exactly_once(self, monkeypatch):
        session = _session(
            idle_notif_start=time.time() - 1.0,
            last_assistant_response="once only",
        )
        stop, thread, fired, _ = self._start_poller(session, monkeypatch, _idle_cfg())
        try:
            assert self._wait_for(lambda: fired)
            time.sleep(0.3)  # several more ticks
            assert len(fired) == 1
        finally:
            stop.set()
            thread.join(timeout=5)

    def test_no_start_never_fires(self, monkeypatch):
        # idle_notif_start stays 0.0 (user replied / nothing finished yet)
        session = _session(last_assistant_response="armed? no")
        stop, thread, fired, _ = self._start_poller(session, monkeypatch, _idle_cfg())
        try:
            time.sleep(0.4)
            assert not fired
        finally:
            stop.set()
            thread.join(timeout=5)


class TestSendIdleNotification:
    def test_success_marks_sent(self, monkeypatch):
        import json

        session = _session(
            idle_notif_start=time.time() - 65,
            last_assistant_response="调研完成，结果写入 00_虚室/xx.md",
        )
        monkeypatch.setattr(server, "_sessions", {"sid-send": session})

        sent = {}

        def _fake_send_message_tool(args):
            sent.setdefault("targets", []).append(args.get("target"))
            sent["message"] = args.get("message")
            return json.dumps({"success": True, "note": "sent"})

        monkeypatch.setattr(
            "tools.send_message_tool.send_message_tool", _fake_send_message_tool
        )

        server._send_idle_notification("sid-send", session, _idle_cfg())
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not session["idle_notif_sent"]:
            time.sleep(0.02)
        assert session["idle_notif_sent"] is True
        assert sent["targets"] == ["weixin", "feishu", "email"]
        assert "TUI 任务完成" in sent["message"]
        assert "调研完成" in sent["message"]

    def test_no_platforms_is_noop(self, monkeypatch):
        session = _session(idle_notif_start=time.time() - 65, last_assistant_response="x")
        monkeypatch.setattr(
            "tools.send_message_tool.send_message_tool",
            lambda args: pytest.fail("must not send with empty platforms"),
        )
        server._send_idle_notification("sid-x", session, _idle_cfg(platforms=[]))
        time.sleep(0.1)
        assert session["idle_notif_sent"] is False

    def test_failure_does_not_mark_sent(self, monkeypatch):
        import json

        session = _session(
            idle_notif_start=time.time() - 65,
            last_assistant_response="will fail",
        )
        monkeypatch.setattr(server, "_sessions", {"sid-fail": session})
        monkeypatch.setattr(
            "tools.send_message_tool.send_message_tool",
            lambda args: json.dumps({"error": "Platform 'feishu' is not configured."}),
        )
        server._send_idle_notification("sid-fail", session, _idle_cfg())
        time.sleep(0.2)
        assert session["idle_notif_sent"] is False

    def test_summary_truncated_to_max_chars(self, monkeypatch):
        import json

        long_summary = "字" * 5000
        session = _session(
            idle_notif_start=time.time() - 65,
            last_assistant_response=long_summary,
        )
        monkeypatch.setattr(server, "_sessions", {"sid-trunc": session})

        captured = {}

        def _fake_send(args):
            captured["message"] = args.get("message")
            return json.dumps({"success": True})

        monkeypatch.setattr(
            "tools.send_message_tool.send_message_tool", _fake_send
        )
        server._send_idle_notification(
            "sid-trunc", session, _idle_cfg(max_summary_chars=100)
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not captured:
            time.sleep(0.02)
        summary_part = captured["message"].split("结果摘要：\n", 1)[1]
        assert len(summary_part) <= 101  # 100 chars + ellipsis
        assert summary_part.endswith("…")

    def test_in_flight_reentry_no_duplicate(self, monkeypatch):
        """竞态回归：发送线程完成前再次触发不得重复发送。

        2026-08-16 实证：poller 高频 tick 窗口内（发送完成前标志未置位）
        并发启动多个发送线程，一次 idle 事件连发 10+ 条重复消息。
        """
        import json

        session = _session(
            idle_notif_start=time.time() - 65,
            last_assistant_response="race repro",
        )
        monkeypatch.setattr(server, "_sessions", {"sid-race": session})
        calls = []

        def _slow_send(args):
            calls.append(args.get("target"))
            time.sleep(0.3)  # 模拟网络 IO：发送完成前 poller 有多个 tick 窗口
            return json.dumps({"success": True})

        monkeypatch.setattr(
            "tools.send_message_tool.send_message_tool", _slow_send
        )
        cfg = _idle_cfg(platforms=["feishu"])

        server._send_idle_notification("sid-race", session, cfg)
        assert session["idle_notif_sent"] is True  # 启动前已置位（防重入）

        server._send_idle_notification("sid-race", session, cfg)  # 发送中重入
        time.sleep(0.5)  # 等发送线程完成
        assert len(calls) == 1  # 只发送了一组
        assert session["idle_notif_sent"] is True  # 成功后保持

    def test_all_failed_resets_flag_after_in_flight(self, monkeypatch):
        """全部平台失败 → 发送中防重入置位后重置为 False（可重试）。"""
        import json

        session = _session(
            idle_notif_start=time.time() - 65,
            last_assistant_response="will fail",
        )
        monkeypatch.setattr(server, "_sessions", {"sid-fail2": session})

        def _slow_fail(args):
            time.sleep(0.2)  # 模拟网络 IO：断言前线程仍在发送中
            return json.dumps({"error": "boom"})

        monkeypatch.setattr(
            "tools.send_message_tool.send_message_tool", _slow_fail
        )
        server._send_idle_notification(
            "sid-fail2", session, _idle_cfg(platforms=["feishu"])
        )
        assert session["idle_notif_sent"] is True  # 发送中防重入
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and session["idle_notif_sent"]:
            time.sleep(0.02)
        assert session["idle_notif_sent"] is False  # 全失败 → 重置可重试
