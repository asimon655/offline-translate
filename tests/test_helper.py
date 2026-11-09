"""The JSON-lines helper: in-process with the fakes, and as a real subprocess over a pipe."""
import io
import json
import subprocess
import sys
import threading
import time

import pytest

from offline_translate import Translator
from offline_translate import catalog as cat
from offline_translate.helper import Server


class Out(io.StringIO):
    def messages(self):
        return [json.loads(line) for line in self.getvalue().splitlines()]


def run(t, *requests, **kw):
    """Feed requests to a Server, wait for all work to finish, return {id: message} and the events."""
    out = Out()
    server = Server(t, out, **kw)
    for r in requests:
        server.handle_line(r if isinstance(r, str) else json.dumps(r))
    deadline = time.time() + 5
    while server._downloads and time.time() < deadline:
        time.sleep(0.01)
    server.close()
    msgs = out.messages()
    return {m["id"]: m for m in msgs if "id" in m and "event" not in m}, [m for m in msgs if "event" in m]


@pytest.fixture
def t(tmp_path, google, fake_engine):
    t = Translator(tmp_path)
    t.download("iw")
    return t


class TestMethods:
    def test_translate(self, t):
        res, _ = run(t, {"id": 1, "method": "translate", "params": {"text": "hello", "target": "he"}})
        r = res[1]["result"]
        assert (r["text"], r["source"], r["target"], r["via"]) == ("[en_iw]hello", "en", "iw", ["en", "iw"])
        assert isinstance(r["ms"], int)

    def test_not_installed_error_lists_missing(self, t):
        res, _ = run(t, {"id": "a", "method": "translate", "params": {"text": "x", "target": "fr"}})
        assert res["a"]["error"]["code"] == "not_installed"
        assert res["a"]["error"]["missing"] == ["fr"]

    def test_detect_installed_languages(self, t):
        res, _ = run(t, {"id": 1, "method": "detect", "params": {"text": "שלום"}},
                     {"id": 2, "method": "installed"}, {"id": 3, "method": "catalog_version"})
        assert res[1]["result"] == {"lang": "iw", "name": "Hebrew"}
        assert res[2]["result"] == {"languages": ["en", "iw"]}
        assert res[3]["result"] == {"version": "8.5.0"}

    def test_languages(self, t):
        res, _ = run(t, {"id": 1, "method": "languages"})
        langs = {l["code"]: l for l in res[1]["result"]["languages"]}
        assert langs["iw"]["installed"] and not langs["fr"]["installed"] and not langs["bs"]["supported"]
        assert langs["en"]["builtin"] and langs["fr"]["downloading"] is False
        assert res[1]["result"]["catalog_version"] == "8.5.0"

    def test_preload_unload(self, t, fake_engine):
        res, _ = run(t, {"id": 1, "method": "preload", "params": {"source": "en", "target": "iw"}},
                     {"id": 2, "method": "loaded"}, {"id": 3, "method": "unload"})
        assert res[1]["result"] == {"loaded": [["en", "iw"]]}
        assert res[2]["result"] == {"loaded": [["en", "iw"]]}
        assert res[3]["result"] == {"loaded": []}

    def test_remove(self, t):
        res, _ = run(t, {"id": 1, "method": "remove", "params": {"lang": "he"}},
                     {"id": 2, "method": "remove", "params": {"lang": "he"}})
        assert res[1]["result"] == {"removed": "iw"}
        assert res[2]["error"]["code"] == "translate_error"

    def test_update_catalog(self, t, google):
        google.publish(["iw", "fr", "de"], version=(8, 6, 0))
        res, _ = run(t, {"id": 1, "method": "update_catalog"})
        assert res[1]["result"] == {"version": "8.6.0"}

    def test_network_error(self, t, google):
        google.offline = True
        res, _ = run(t, {"id": 1, "method": "update_catalog"})
        assert res[1]["error"]["code"] == "network"

    def test_ping_and_shutdown(self, t):
        out = Out()
        server = Server(t, out)
        server.serve(iter([json.dumps({"id": 1, "method": "ping"}), json.dumps({"id": 2, "method": "shutdown"}),
                           json.dumps({"id": 3, "method": "ping"})]))
        msgs = out.messages()
        assert msgs[0]["event"] == "ready"
        assert msgs[1] == {"id": 1, "result": {"pong": True, "version": msgs[0]["version"]}}
        assert msgs[2] == {"id": 2, "result": {"bye": True}}
        assert len(msgs) == 3  # nothing handled after shutdown


class TestBadInput:
    @pytest.mark.parametrize("line, code", [
        ("not json", "bad_request"),
        ("[1, 2]", "bad_request"),
        ('{"id": 1, "method": "fly"}', "unknown_method"),
        ('{"id": 1}', "unknown_method"),
        ('{"id": 1, "method": "translate", "params": {"target": "iw"}}', "bad_request"),
        ('{"id": 1, "method": "download", "params": {}}', "bad_request"),
        ('{"id": 1, "method": "translate", "params": {"text": "x", "target": "klingon"}}', "not_installed"),
    ])
    def test_errors_never_kill_the_server(self, t, line, code):
        res, _ = run(t, line, {"id": 99, "method": "ping"})
        err = next(m for m in res.values() if "error" in m)
        assert err["error"]["code"] == code
        assert res[99]["result"]["pong"]

    def test_blank_lines_ignored(self, t):
        res, events = run(t, "", "   ", {"id": 1, "method": "ping"})
        assert list(res) == [1] and not events


class TestDownload:
    def test_download_streams_progress_then_result(self, tmp_path, google, fake_engine, monkeypatch):
        monkeypatch.setattr("offline_translate.helper.PROGRESS_INTERVAL", 0)
        t = Translator(tmp_path)
        res, events = run(t, {"id": 5, "method": "download", "params": {"lang": "French"}})
        assert res[5]["result"] == {"installed": True, "lang": "fr"}
        assert events and all(e["event"] == "progress" and e["id"] == 5 and e["lang"] == "fr" for e in events)
        assert events[-1]["done"] == events[-1]["total"] > 0
        assert "fr" in t.installed_languages()

    def test_cancel(self, tmp_path, google, monkeypatch):
        gate, started = threading.Event(), threading.Event()

        class Slow(io.BytesIO):
            def read(self, n=-1):
                started.set()
                gate.wait(5)
                return super().read(16)

        monkeypatch.setattr(cat, "open_url", lambda url, timeout=60: Slow(google.get(url)))
        t = Translator(tmp_path)
        t.languages()
        out = Out()
        server = Server(t, out)
        server.handle_line(json.dumps({"id": 1, "method": "download", "params": {"lang": "iw"}}))
        assert started.wait(5)
        server.handle_line(json.dumps({"id": 2, "method": "download", "params": {"lang": "iw"}}))
        server.handle_line(json.dumps({"id": 3, "method": "remove", "params": {"lang": "iw"}}))
        server.handle_line(json.dumps({"id": 4, "method": "cancel", "params": {"lang": "iw"}}))
        gate.set()
        while server._downloads:
            time.sleep(0.01)
        server.handle_line(json.dumps({"id": 5, "method": "cancel", "params": {"lang": "iw"}}))
        server.close()
        res = {m["id"]: m for m in out.messages() if "event" not in m}
        assert res[2]["error"]["code"] == "busy"
        assert res[3]["error"]["code"] == "bad_request"
        assert res[4]["result"] == {"cancelled": True}
        assert res[1]["error"]["code"] == "cancelled"
        assert res[5]["result"] == {"cancelled": False}
        assert "iw" not in t.installed_languages()


class TestIdleUnload:
    def test_unused_models_are_freed(self, t, fake_engine):
        now = [0.0]
        out = Out()
        server = Server(t, out, idle_unload=60, clock=lambda: now[0])
        server.handle_line(json.dumps({"id": 1, "method": "preload", "params": {"source": "en", "target": "iw"}}))
        server._models.submit(lambda: None).result()
        now[0] = 30
        server._models.submit(server._unload_idle).result()
        assert t.loaded() == [("en", "iw")]
        now[0] = 61
        server._models.submit(server._unload_idle).result()
        server.close()
        assert t.loaded() == []
        assert {"event": "unloaded", "pairs": [["en", "iw"]]} in out.messages()


# ---------- the real process ----------
def spawn(models, *args):
    return subprocess.Popen([sys.executable, "-m", "offline_translate.helper", "--models", str(models), *args],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8")


def ask(p, req):
    p.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
    p.stdin.flush()
    while True:
        msg = json.loads(p.stdout.readline())
        if msg.get("id") == req["id"] and "event" not in msg:
            return msg


def test_subprocess_over_pipe(tmp_path):
    p = spawn(tmp_path)
    try:
        assert json.loads(p.stdout.readline())["event"] == "ready"
        assert ask(p, {"id": 1, "method": "ping"})["result"]["pong"]
        assert ask(p, {"id": 2, "method": "installed"})["result"] == {"languages": ["en"]}
        err = ask(p, {"id": 3, "method": "translate", "params": {"text": "hi", "target": "iw", "source": "en"}})
        assert err["error"]["missing"] == ["iw"]
        assert ask(p, {"id": 4, "method": "shutdown"})["result"] == {"bye": True}
        assert p.wait(10) == 0
    finally:
        p.kill()


def test_subprocess_exits_on_stdin_eof(tmp_path):
    p = spawn(tmp_path)
    out, _ = p.communicate("", timeout=10)
    assert p.returncode == 0 and json.loads(out.splitlines()[0])["event"] == "ready"


@pytest.mark.models
def test_subprocess_real_translation(models_dir):
    p = spawn(models_dir, "--idle-unload", "0")
    try:
        p.stdout.readline()
        r = ask(p, {"id": 1, "method": "translate", "params": {"text": "car", "target": "he"}})["result"]
        assert r["text"] == "מכונית" and r["source"] == "en"
        r = ask(p, {"id": 2, "method": "translate", "params": {"text": "שלום, מה שלומך?", "target": "en"}})
        assert "hello" in r["result"]["text"].lower()
        # nothing but protocol lines on stdout (native TFLite logging goes to stderr)
        assert ask(p, {"id": 3, "method": "loaded"})["result"]["loaded"] == [["en", "iw"], ["iw", "en"]]
        ask(p, {"id": 4, "method": "shutdown"})
        assert p.wait(10) == 0
    finally:
        p.kill()
