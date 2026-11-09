"""JSON-lines server around Translator, for apps that run the library as a long-lived helper process.

    offline-translate-helper --models DIR [--idle-unload SECONDS] [--beam N] [--channel prod]

One JSON object per line on stdin/stdout. Requests carry an "id" that the response echoes:

    -> {"id": 7, "method": "translate", "params": {"text": "Hello", "target": "iw"}}
    <- {"id": 7, "result": {"text": "שלום", "source": "en", "target": "iw", "via": ["en", "iw"], "ms": 31}}
    <- {"id": 8, "error": {"code": "not_installed", "message": "...", "missing": ["de"]}}

Lines without an id are events: "ready" at startup, "progress" while downloading, "unloaded" when idle
models are freed. Methods: ping, translate, detect, languages, installed, catalog_version,
update_catalog, download, cancel, remove, preload, unload, loaded, shutdown.
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from . import __version__
from .languages import NAMES, normalize
from .translator import DownloadCancelled, NotInstalledError, TranslateError, Translator

PROGRESS_INTERVAL = 0.1  # seconds between progress events for one download


class BadRequest(Exception):
    pass


def error_body(exc):
    """Map an exception to the protocol's {"code", "message", ...} error object."""
    if isinstance(exc, NotInstalledError):
        return {"code": "not_installed", "message": str(exc), "missing": exc.missing}
    if isinstance(exc, DownloadCancelled):
        return {"code": "cancelled", "message": str(exc)}
    if isinstance(exc, TranslateError):
        return {"code": "translate_error", "message": str(exc)}
    if isinstance(exc, BadRequest):
        return {"code": "bad_request", "message": str(exc)}
    if isinstance(exc, urllib.error.URLError):
        return {"code": "network", "message": str(getattr(exc, "reason", exc))}
    if isinstance(exc, OSError):
        return {"code": "io_error", "message": str(exc)}
    return {"code": "internal", "message": f"{type(exc).__name__}: {exc}"}


class Server:
    def __init__(self, translator, out, idle_unload=600.0, clock=time.monotonic):
        self.t = translator
        self.out = out
        self.idle_unload = idle_unload
        self.clock = clock
        self._write_lock = threading.Lock()
        # Everything that touches loaded models runs on one thread; catalog/network calls on another,
        # so a slow catalog fetch never delays a translation.
        self._models = ThreadPoolExecutor(1, thread_name_prefix="models")
        self._io = ThreadPoolExecutor(1, thread_name_prefix="io")
        self._downloads = {}  # lang -> threading.Event (cancel flag)
        self._last_used = {}  # model pair -> clock()
        self.stopped = threading.Event()

    # ---------- output ----------
    def send(self, obj):
        line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        with self._write_lock:
            self.out.write(line + "\n")
            self.out.flush()

    def _respond(self, rid, fn, *args):
        try:
            result = fn(*args)
        except Exception as exc:  # every failure becomes an error response, never a dead helper
            self.send({"id": rid, "error": error_body(exc)})
        else:
            self.send({"id": rid, "result": result})

    # ---------- dispatch ----------
    def handle_line(self, line):
        line = line.strip()
        if not line:
            return
        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                raise ValueError("request must be a JSON object")
        except ValueError as exc:
            self.send({"id": None, "error": {"code": "bad_request", "message": f"invalid JSON: {exc}"}})
            return
        rid, method, params = req.get("id"), req.get("method"), req.get("params") or {}
        handler = getattr(self, f"m_{method}", None) if isinstance(method, str) else None
        if handler is None:
            self.send({"id": rid, "error": {"code": "unknown_method", "message": f"unknown method {method!r}"}})
            return
        pool = getattr(handler, "pool", None)
        if pool == "models":
            self._models.submit(self._respond, rid, handler, params)
        elif pool == "io":
            self._io.submit(self._respond, rid, handler, params)
        elif pool == "thread":
            handler(rid, params)
        else:
            self._respond(rid, handler, params)

    def serve(self, lines):
        ticker = threading.Thread(target=self._idle_loop, daemon=True)
        ticker.start()
        self.send({"event": "ready", "version": __version__, "pid": os.getpid()})
        for line in lines:
            self.handle_line(line)
            if self.stopped.is_set():
                break
        self.close()

    def close(self):
        self.stopped.set()
        for ev in list(self._downloads.values()):
            ev.set()
        self._models.shutdown(wait=True)
        self._io.shutdown(wait=True)

    # ---------- idle unloading ----------
    def _idle_loop(self):
        while not self.stopped.wait(min(30.0, max(self.idle_unload / 4, 0.05))):
            if self.idle_unload > 0:
                try:
                    self._models.submit(self._unload_idle)
                except RuntimeError:  # executor already shut down
                    return

    def _unload_idle(self):
        now = self.clock()
        idle = [p for p in self.t.loaded() if now - self._last_used.get(p, now) >= self.idle_unload]
        if idle:
            self.t.unload(idle)
            for p in idle:
                self._last_used.pop(p, None)
            self.send({"event": "unloaded", "pairs": [list(p) for p in idle]})

    def _touch(self, hops):
        now = self.clock()
        for pair in hops:
            self._last_used[tuple(pair)] = now

    # ---------- methods ----------
    @staticmethod
    def _param(params, name, required=True):
        value = params.get(name)
        if required and (not isinstance(value, str) or not value):
            raise BadRequest(f"missing string param '{name}'")
        return value

    def m_ping(self, params):
        return {"pong": True, "version": __version__}

    def m_translate(self, params):
        text = params.get("text")
        if not isinstance(text, str):
            raise BadRequest("missing string param 'text'")
        target = self._param(params, "target")
        source = params.get("source") or None
        start = time.perf_counter()
        r = self.t.translate(text, target=target, source=source)
        self._touch(zip(r.via, r.via[1:]))
        return {"text": r.text, "source": r.source, "target": r.target, "via": list(r.via),
                "ms": round((time.perf_counter() - start) * 1000)}
    m_translate.pool = "models"

    def m_detect(self, params):
        text = params.get("text")
        if not isinstance(text, str):
            raise BadRequest("missing string param 'text'")
        code = self.t.detect(text)
        return {"lang": code, "name": NAMES.get(code, code)}
    m_detect.pool = "models"

    def m_preload(self, params):
        hops = self.t.preload(self._param(params, "source"), self._param(params, "target"))
        self._touch(hops)
        return {"loaded": [list(p) for p in hops]}
    m_preload.pool = "models"

    def m_unload(self, params):
        pairs = params.get("pairs")
        self.t.unload([tuple(p) for p in pairs] if pairs else None)
        return {"loaded": [list(p) for p in self.t.loaded()]}
    m_unload.pool = "models"

    def m_loaded(self, params):
        return {"loaded": [list(p) for p in self.t.loaded()]}
    m_loaded.pool = "models"

    def m_remove(self, params):
        lang = normalize(self._param(params, "lang"))
        if lang in self._downloads:
            raise BadRequest(f"'{lang}' is downloading; cancel it first")
        self.t.remove(lang)
        return {"removed": lang}
    m_remove.pool = "models"  # models of a removed language must not be in use

    def m_installed(self, params):
        return {"languages": self.t.installed_languages()}
    m_installed.pool = "io"

    def m_languages(self, params):
        langs = self.t.languages(installed_only=bool(params.get("installed_only")))
        return {"languages": [dict(asdict(l), downloading=l.code in self._downloads) for l in langs],
                "catalog_version": self.t.catalog_version()}
    m_languages.pool = "io"

    def m_catalog_version(self, params):
        return {"version": self.t.catalog_version()}
    m_catalog_version.pool = "io"

    def m_update_catalog(self, params):
        return {"version": self.t.update_catalog()}
    m_update_catalog.pool = "io"

    def m_download(self, rid, params):
        try:
            lang = normalize(self._param(params, "lang"))
        except BadRequest as exc:
            self.send({"id": rid, "error": error_body(exc)})
            return
        if lang in self._downloads:
            self.send({"id": rid, "error": {"code": "busy", "message": f"'{lang}' is already downloading"}})
            return
        cancel = self._downloads[lang] = threading.Event()
        threading.Thread(target=self._download, args=(rid, lang, bool(params.get("force")), cancel),
                         name=f"download-{lang}", daemon=True).start()
    m_download.pool = "thread"

    def _download(self, rid, lang, force, cancel):
        last = [0.0]

        def progress(done, total):
            now = time.monotonic()
            if done >= total or now - last[0] >= PROGRESS_INTERVAL:
                last[0] = now
                self.send({"event": "progress", "id": rid, "lang": lang, "done": done, "total": total})

        try:
            self._respond(rid, lambda: (self.t.download(lang, progress=progress, force=force, cancel=cancel),
                                        {"installed": True, "lang": lang})[1])
        finally:
            self._downloads.pop(lang, None)

    def m_cancel(self, params):
        lang = normalize(self._param(params, "lang"))
        ev = self._downloads.get(lang)
        if ev:
            ev.set()
        return {"cancelled": ev is not None}

    def m_shutdown(self, params):
        self.stopped.set()
        return {"bye": True}


def protocol_stdout():
    """Keep fd 1 for the protocol only: anything else printing to stdout (Python or native code) goes to stderr."""
    proto = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return proto


def main(argv=None):
    ap = argparse.ArgumentParser(prog="offline-translate-helper", description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", required=True, help="folder for language packs and the catalog cache")
    ap.add_argument("--channel", default="prod")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--idle-unload", type=float, default=600, metavar="SECONDS",
                    help="free models unused for this long (0 = never; default 600)")
    args = ap.parse_args(argv)
    out = protocol_stdout()
    stdin = open(sys.stdin.fileno(), encoding="utf-8", closefd=False)
    server = Server(Translator(args.models, channel=args.channel, beam=args.beam), out,
                    idle_unload=args.idle_unload)
    try:
        server.serve(stdin)
    except KeyboardInterrupt:
        server.close()


if __name__ == "__main__":
    main()
