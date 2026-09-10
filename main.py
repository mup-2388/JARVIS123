"""
main.py -- J.A.R.V.I.S. launcher.

Responsibilities, in order:

1. Pre-flight: create ``data/`` & ``notes/``, read ``.env``, probe the GPU, warn
   about missing keys without ever refusing to boot.
2. Start the FastAPI/ASGI server (uvicorn) on a background thread with its own
   event loop, then wait until ``/healthz`` answers.
3. Open the HUD in a native window with ``pywebview`` pointing at
   ``static/index.html`` and hand it the backend URL + a ``js_api`` bridge so
   the page can call Python directly (``window.pywebview.api.command(...)``).
4. ``webview.start()`` blocks (that is the app's main loop). On close we ask
   uvicorn to shut down cleanly, so no zombie process is left holding the port.

If pywebview cannot open a window (headless box, WSL without an X server, CI)
the process degrades to "server only" mode and prints the URL, which is exactly
what you want when JARVIS runs as a background service on a second machine.

    python main.py                     # normal desktop launch
    python main.py --port 8899          # alternate port
    python main.py --no-window          # headless service mode
    python main.py --no-voice           # skip Whisper/XTTS warm-up (fast dev boot)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# sys.path[0] is already this folder when run as `python main.py`, but a
# scheduled task / .bat with a different CWD can break absolute imports of the
# sibling modules, so pin the project root first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402

config.ensure_importable()  # also exposes ROOT for `python -m main`, services, etc.
import server  # noqa: E402
from config import SETTINGS, get_logger  # noqa: E402

log = get_logger("main")

BANNER = r"""
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝╚═╝╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██╗███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
     Just A Rather Very Intelligent System
"""


# ---------------------------------------------------------------------------
# Environment / hardware probing
# ---------------------------------------------------------------------------

def preflight() -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for folder in (SETTINGS.notes_path, config.ASSETS_DIR, config.CACHE_DIR, config.STATIC_DIR):
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning("could not create %s: %s", folder, exc)

    report["env_file"] = config.ENV_FILE.is_file()
    if not report["env_file"]:
        log.warning("no .env found -- copying .env.example to .env so you can edit it")
        example = config.ROOT / ".env.example"
        if example.is_file():
            shutil.copyfile(example, config.ENV_FILE)

    import llm_providers

    ready = llm_providers.POOL.configured()
    report["llm_providers"] = ready
    report["llm_status"] = llm_providers.POOL.status()
    missing = [
        key
        for key, value in (
            ("DISCORD_TOKEN", SETTINGS.discord_token),
            ("API_SPORTS_KEY", SETTINGS.api_sports_key),
        )
        if not value
    ]
    if not ready:
        log.warning(
            "no AI provider key found -- JARVIS answers with its regex rules only. "
            "Set one of GROQ_API_KEY, CEREBRAS_API_KEY, CLOUDFLARE_API_TOKEN+ACCOUNT_ID, "
            "GEMINI_API_KEY, MISTRAL_API_KEY or OPENROUTER_API_KEY in .env"
        )
    else:
        log.info("AI providers configured: %s", ", ".join(ready))
    report["missing_keys"] = missing
    if missing:
        log.info("optional keys not set (features degrade gracefully): %s", ", ".join(missing))

    # GPU / VRAM -- the reason for the int8 + low_vram defaults.
    report["gpu"] = "none detected"
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            out = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=8,
            )
            if out.returncode == 0 and out.stdout.strip():
                report["gpu"] = out.stdout.strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError) as exc:
            log.debug("nvidia-smi probe failed: %s", exc)
    try:
        import psutil

        report["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        report["cpu_cores"] = psutil.cpu_count(logical=True)
    except Exception:  # noqa: BLE001
        pass

    report["ffmpeg"] = bool(shutil.which("ffmpeg"))
    if not report["ffmpeg"]:
        log.warning("ffmpeg missing: browser mic (webm/opus) cannot be decoded. "
                    "Install ffmpeg, or use the notes/search text box which needs no audio.")
    report["reference_wav"] = SETTINGS.reference_wav_path.is_file()
    if SETTINGS.tts_enabled and not report["reference_wav"]:
        log.warning("voice clone sample missing at %s -- falling back to the OS voice",
                    SETTINGS.reference_wav_path)
    report["notes"] = len(list(SETTINGS.notes_path.glob("*.md"))) if SETTINGS.notes_path.is_dir() else 0
    # The hands: what the desktop layer can actually do on this machine right now.
    try:
        import apps as _apps
        import files as _files
        import winops as _winops
        import screen as _screen

        report["apps_known"] = len(_apps.known_names())
        report["files_roots"] = [str(p) for p in _files.roots()]
        report["windows_session"] = bool(_winops.IS_WINDOWS)
        report["ocr_ready"] = bool(_screen.ocr().get("ok"))
        try:
            import wake as _wake

            report["ear"] = _wake.LISTENER.available()[1] or "ready"
        except Exception as exc:  # noqa: BLE001
            report["ear"] = str(exc)
        try:
            import reminders as _reminders

            report["reminders"] = len(_reminders.BOARD.list())
        except Exception as exc:  # noqa: BLE001
            report["reminders"] = 0
            log.debug("schedule not readable at boot: %s", exc)
    except Exception as exc:  # noqa: BLE001
        report["desktop"] = f"desktop layer unavailable: {exc}"
    return report


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) != 0


def pick_port(preferred: int, span: int = 12) -> int:
    for offset in range(span):
        candidate = preferred + offset
        if port_is_free(candidate):
            return candidate
        log.info("port %d is busy, trying the next one", candidate)
    return preferred


def wait_for_server(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/healthz"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as resp:  # noqa: S310 - fixed localhost url
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(0.15)
    return False


# ---------------------------------------------------------------------------
# uvicorn host thread
# ---------------------------------------------------------------------------

class ServerThread:
    """Runs uvicorn in its own thread + loop so pywebview owns the main thread."""

    def __init__(self, host: str, port: int, log_level: str = "warning") -> None:
        self.host, self.port, self.log_level = host, port, log_level
        self._thread: Optional[threading.Thread] = None
        self._server = None

    def start(self) -> None:
        import uvicorn

        uvicorn_config = uvicorn.Config(
            server.app,
            host=self.host,
            port=self.port,
            log_level=self.log_level,
            ws_ping_interval=20,
            ws_ping_timeout=30,
            access_log=SETTINGS.debug,
            loop="asyncio",
            lifespan="on",
            backlog=128,
        )
        self._server = uvicorn.Server(uvicorn_config)
        # We handle SIGINT ourselves so webview can close cleanly.
        self._server.install_signal_handlers = lambda: None  # type: ignore[assignment]
        self._thread = threading.Thread(target=self._server.run, name="jarvis-asgi", daemon=True)
        self._thread.start()
        log.info("ASGI thread started on %s:%d", self.host, self.port)

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=6)
        log.info("ASGI thread stopped")

    @property
    def alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())


# ---------------------------------------------------------------------------
# pywebview bridge
# ---------------------------------------------------------------------------

class JarvisBridge:
    """Methods the HUD reaches through ``window.pywebview.api.*``.

    These call the HTTP API on our own port instead of importing the loop: the
    window thread must never block on model work, and going through REST keeps
    one single code path for HUD, Discord and curl.

    **Exposure rule (this bit caused a hard freeze on Windows).**  pywebview
    generates the JavaScript API by walking ``dir()`` of this object and
    *recursing into every non-callable attribute* (``webview.util.get_functions``).
    Holding the native window on it - ``self.window = window`` - sends that walk
    into ``window.native`` (a WinForms control), whose ``AccessibilityObject`` →
    ``Bounds`` → ``Empty`` → ``Empty`` chain never terminates, so boot died with
    "maximum recursion depth exceeded" while the UI thread was stuck reflecting
    pythonnet objects and the window never repainted.  So: public surface is
    plain methods with JSON-friendly arguments only; state lives on
    ``_``-prefixed attributes, which pywebview skips, and never on GUI objects.
    """

    def __init__(self, port: int, window: Any = None) -> None:
        self._port = int(port)
        self._window: Any = window          # private: pywebview must not walk it

    # Nothing below may gain a public non-callable attribute: see class docstring.

    # -- plumbing ----------------------------------------------------------
    def _request(self, path: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 120.0) -> Dict[str, Any]:
        url = f"http://127.0.0.1:{self._port}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url, data=data, headers={"content-type": "application/json"}, method="POST" if data else "GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310
                raw = resp.read().decode("utf-8", "replace")
                return json.loads(raw) if raw.strip().startswith(("{", "[")) else {"ok": True, "raw": raw}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:400] if exc.fp else ""
            return {"ok": False, "error": f"HTTP {exc.code}: {body or exc.reason}"}
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            return {"ok": False, "error": f"core unreachable: {exc}"}

    # -- API exposed to JavaScript ----------------------------------------
    def command(self, text: str) -> Dict[str, Any]:
        return self._request("/api/command", {"text": str(text), "source": "window"})

    def speak(self, text: str) -> Dict[str, Any]:
        return self._request("/api/speak", {"text": str(text), "play": True})

    def stop(self) -> Dict[str, Any]:
        return self._request("/api/stop", {})

    def status(self) -> Dict[str, Any]:
        return self._request("/api/status")

    def config(self) -> Dict[str, Any]:
        return self._request("/api/config")

    def notes(self, topic: str = "") -> Dict[str, Any]:
        return self._request(f"/api/notes?topic={urllib.parse.quote(str(topic))}")

    def mirror(self, on: bool = True) -> Dict[str, Any]:
        return self._request("/api/discord/mirror", {"on": bool(on)})

    def llm(self) -> Dict[str, Any]:
        """Provider pool health: who is configured, who is cooling, until when."""
        return self._request("/api/llm")

    def llm_reset(self, provider: str = "") -> Dict[str, Any]:
        return self._request("/api/llm/reset", {"provider": str(provider or "")})

    def llm_probe(self) -> Dict[str, Any]:
        # Probing walks every key with a real request, so allow a longer timeout.
        return self._request("/api/llm/probe", {}, timeout=180.0)

    # -- window controls ---------------------------------------------------
    # -- window controls ---------------------------------------------------
    def _window_call(self, names: Tuple[str, ...], *args: Any) -> Dict[str, Any]:
        """Call the first window method this pywebview build actually has.

        pywebview renames and drops things between releases (5.x has ``destroy``
        but no ``close``, and removed ``toggle_frameless``), so each control is
        capability-checked: an unavailable one is reported plainly instead of
        raising an AttributeError through the JavaScript bridge.
        """
        if self._window is None:
            return {"ok": False, "error": "no window bound (running headless?)"}
        for name in names:
            attr = getattr(self._window, name, None)
            if attr is None:
                continue
            try:
                value = attr(*args) if callable(attr) else attr
            except Exception as exc:  # noqa: BLE001 - platform dependent
                return {"ok": False, "error": f"{name}() failed: {exc}"}
            return {"ok": True, "method": name, "result": value}
        return {"ok": False, "error": "this pywebview build supports none of: " + ", ".join(names)}

    def set_topmost(self, on: bool = True) -> Dict[str, Any]:
        if self._window is None:
            return {"ok": False, "error": "no window bound (running headless?)"}
        try:
            self._window.on_top = bool(on)
        except Exception as exc:  # noqa: BLE001 - not every GUI backend has the property
            return {"ok": False, "error": f"on_top unsupported here: {exc}"}
        return {"ok": True, "on_top": bool(getattr(self._window, "on_top", on))}

    def toggle_frameless(self) -> Dict[str, Any]:
        return self._window_call(("toggle_frameless", "set_frameless"))

    def minimize(self) -> Dict[str, Any]:
        return self._window_call(("minimize",))

    def maximize(self) -> Dict[str, Any]:
        return self._window_call(("maximize",))

    def quit(self) -> Dict[str, Any]:
        threading.Thread(target=self._shutdown, daemon=True).start()
        return {"ok": True}

    def _shutdown(self) -> None:
        """Close the window so ``webview.start()`` returns and the process exits."""
        if self._window is not None:
            self._window_call(("destroy", "close"))

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def llm_row(report: Dict[str, Any]) -> str:
    """One-line summary of the provider pool for the boot banner."""
    status = report.get("llm_status") or {}
    providers = status.get("providers") or []
    live = status.get("available") or []
    cooling = [p for p in providers if p.get("cooling")]
    if not providers or not status.get("configured"):
        return "no provider keys in .env -> heuristic planner (add GROQ_API_KEY etc.)"
    detail = ", ".join(live) if live else "none available"
    if len(live) == 1:
        winner = next((p for p in providers if p.get("key") == live[0]), None)
        if winner:
            detail += f" (currently {winner.get('fast_model_used', '')})"
    if cooling:
        detail += "  | cooling: " + ", ".join(
            f"{p['key']} {max(1, int(p.get('cool_left_s', 0)) // 60)}m" for p in cooling)
    return detail


def _roots_row(report: Dict[str, Any]) -> str:
    """Where the file powers are allowed to write - printed at boot, because it must not be a surprise."""
    roots = report.get("files_roots") or []
    if not roots:
        return "file tools have no writable root yet (set FILES_ROOT in .env)"
    policy = (SETTINGS.file_delete_policy or "recycle").lower()
    extra = " (+1 more)" if len(roots) > 1 else ""
    return f"writes confined to {roots[0]}{extra} · deletes to the {policy} · undo journalled"


def _screen_row(report: Dict[str, Any]) -> str:
    ocr = "Windows OCR ready" if report.get("ocr_ready") else "no local OCR engine (pip install winsdk)"
    vision = "vision AI on" if SETTINGS.screen_vision_enabled else "vision AI off"
    return f"{ocr} · {vision}"


def _ear_row(report: Dict[str, Any]) -> str:
    ear = " ".join(str(report.get("ear", "?")).split())
    # Keep one line per capability: the reason is useful, a paragraph is not.
    if len(ear) > 96:
        ear = ear[:95].rstrip(" -,.;") + "…"
    bar = "bar on (F12)" if SETTINGS.bar_enabled else "bar off"
    words = "/".join((SETTINGS.wake_words or "jarvis").split(","))[:48]
    return f"{ear} · wake on \"{words}\" · {bar}"


def print_context(port: int, report: Dict[str, Any], mode: str) -> None:
    rows = [
        ("core", f"http://127.0.0.1:{port}  (HUD at /, WS at /ws)"),
        ("GPU", report.get("gpu", "?")),
        ("RAM / cores", f"{report.get('ram_gb', '?')} GB / {report.get('cpu_cores', '?')} threads"),
        ("STT", f"faster-whisper {SETTINGS.whisper_model} · {SETTINGS.whisper_device}/{SETTINGS.whisper_compute_type}"),
        ("TTS", f"XTTSv2 · reference {'found' if report.get('reference_wav') else 'MISSING'} · low_vram={SETTINGS.tts_low_vram}"),
        ("ffmpeg", "present" if report.get("ffmpeg") else "MISSING (mic upload disabled)"),
        ("notes", f"{report.get('notes', 0)} file(s) in {SETTINGS.notes_path}"),
        ("LLM", llm_row(report)),
        # The hands: one line per capability, so a machine that is missing a driver says so
        # before the first command, instead of JARVIS sounding confident and doing nothing.
        ("apps", f"{report.get('apps_known', 0)} resolvable" if "desktop" not in report
                 else str(report.get("desktop"))),
        ("files", _roots_row(report)),
        ("screen", _screen_row(report)),
        ("ear", _ear_row(report)),
        ("schedule", f"{report.get('reminders', 0)} pending action(s)"),
        ("discord", "token set" if SETTINGS.discord_token else "not configured"),
        ("api-sports", "key set" if SETTINGS.api_sports_key else "no key -> sports tool reports why"),
        ("mode", mode),
    ]
    width = max(len(k) for k, _ in rows)
    print()
    for key, value in rows:
        print(f"   {key:<{width}}  {value}")
    print()


def _bar_top() -> int:
    """Y-position for the floating bar: bottom-left of the primary screen, above the taskbar."""
    try:
        import winops

        size = winops.screen_size()
        if len(size) > 1 and size[1]:
            return max(0, int(size[1]) - SETTINGS.bar_height - 96)
    except Exception:  # noqa: BLE001 - no display info is no reason to fail
        pass
    return 640


def attach_loaded_handler(window: object, handler) -> str:  # noqa: ANN001
    """Subscribe ``handler`` to the window's load event; report the mechanism used.

    pywebview's API moved around between major versions and the event object is
    *not* a decorator:

    * 4.x/5.x -> ``window.events.loaded += handler``
    * 3.x/2.x -> ``window.loaded += handler``

    ``+=`` on those objects calls ``Event.__iadd__``, which appends the callback and
    returns the same event, so the mutation lands on the window either way.
    Returns an empty string when no event object could be found; the caller then
    passes the handler to ``webview.start()`` as a one-shot start callback.
    """
    for label, holder in (("window.events.loaded", getattr(window, "events", None)), ("window.loaded", window)):
        event = getattr(holder, "loaded", None) if holder is not None else None
        if event is None:
            continue
        iadd = getattr(event, "__iadd__", None)
        if iadd is None:
            continue
        try:
            iadd(handler)
        except TypeError as exc:
            log.debug("%s rejected the load handler (%s)", label, exc)
            continue
        log.debug("HUD load event subscribed via %s", label)
        return label
    return ""


def launch(args: argparse.Namespace) -> int:
    print(BANNER)
    report = preflight()
    os.environ.setdefault("JARVIS_PORT", str(args.port))
    port = pick_port(args.port) if not args.keep_port else args.port
    host = args.host or SETTINGS.host

    if args.no_voice:
        os.environ["JARVIS_WARM"] = "0"
    if args.no_discord:
        os.environ["DISCORD_ENABLED"] = "false"

    http = ServerThread(host, port, log_level="info" if SETTINGS.debug else "warning")
    http.start()
    if not wait_for_server(port, timeout=args.wait):
        log.error("server did not answer on port %d within %.0fs", port, args.wait)
        if not http.alive:
            return 1
    print_context(port, report, "windowed" if not args.no_window else "headless (server only)")

    bridge = JarvisBridge(port)

    if args.no_window or config.is_headless():
        reason = "requested --no-window" if args.no_window else "no display available"
        log.info("window disabled (%s); serving HUD for the browser at http://%s:%d", reason, host, port)
        if args.browser:
            try:
                webbrowser.open(f"http://127.0.0.1:{port}/")
            except Exception:  # noqa: BLE001
                pass
        try:
            while http.alive:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n   shutting down (Ctrl+C)")
        finally:
            http.stop()
        return 0

    try:
        import webview
    except Exception as exc:  # noqa: BLE001
        log.error("pywebview unavailable (%s) -- opening the browser instead", exc)
        try:
            webbrowser.open(f"http://127.0.0.1:{port}/")
        except Exception:  # noqa: BLE001
            pass
        try:
            while http.alive:
                time.sleep(0.5)
        finally:
            http.stop()
        return 0

    # Spec contract: pywebview loads static/index.html directly (file://).  If
    # the asset is missing we point the window at the HTTP root instead, which
    # serves the same HUD with a same-origin WebSocket.
    index = str(config.INDEX_HTML) if config.INDEX_HTML.is_file() else f"http://127.0.0.1:{port}/"
    if args.url:
        index = f"http://127.0.0.1:{port}/"
    window = webview.create_window(
        "JARVIS",
        index,
        width=args.width,
        height=args.height,
        min_size=(900, 620),
        resizable=True,
        frameless=args.frameless,
        on_top=args.always_on_top,
        background_color="#04070d",
        text_select=True,
        js_api=bridge,
    )
    #: private on purpose: pywebview exposes every public attribute it can walk,
    #: so the native window must never be reachable from the JS surface.
    bridge._window = window

    # ---- the floating bar: a frameless, top-most, no-focus strip --------------
    # It exists so the user can talk or type to JARVIS while a game, IDE or video keeps the
    # foreground.  `hidden=True` matters: showing it on start would steal the caret.
    bar = None
    if SETTINGS.bar_enabled:
        bar_url = f"http://127.0.0.1:{port}/bar"
        try:
            bar = webview.create_window(
                "JARVIS bar", bar_url, width=max(360, SETTINGS.bar_width), height=max(64, SETTINGS.bar_height),
                x=48, y=_bar_top(), easy_drag=True,
                frameless=True, on_top=True, resizable=False, hidden=True, background_color="#05080e",
            )
        except TypeError:
            # Older pywebview releases reject easy_drag/hidden; retry with what they know.
            try:
                bar = webview.create_window("JARVIS bar", bar_url, width=max(360, SETTINGS.bar_width),
                                            height=max(64, SETTINGS.bar_height), frameless=True, on_top=True,
                                            background_color="#05080e")
            except Exception as exc:  # noqa: BLE001
                log.warning("overlay bar could not be created: %s", exc)
                bar = None
        except Exception as exc:  # noqa: BLE001
            log.warning("overlay bar could not be created: %s", exc)
            bar = None

    def bar_controller(action: str = "show") -> Dict[str, Any]:
        """show/hide/toggle the bar, and make it behave like an overlay, not a window."""
        if bar is None:
            return {"ok": False, "visible": False, "message": "overlay unavailable"}
        action = (action or "show").lower()
        visible = bool(getattr(bar_controller, "visible", False))
        try:
            if action == "hide":
                bar.hide()
                bar_controller.visible = False
            elif action == "toggle":
                (bar.hide() if visible else bar.show())
                bar_controller.visible = not visible
            else:
                bar.show()
                bar_controller.visible = True
                _style_overlay(bar)
        except Exception as exc:  # noqa: BLE001 - a stuck overlay must not kill the app
            return {"ok": False, "visible": visible, "message": str(exc)}
        return {"ok": True, "visible": bool(bar_controller.visible)}

    def _style_overlay(handle_owner: Any) -> None:
        """Apply WS_EX_TOOLWINDOW | NOACTIVATE so the bar never steals focus or alt-tabs."""
        try:
            import winops

            hwnd = None
            native = getattr(handle_owner, "native", None)
            for attr in ("Handle", "handle"):
                value = getattr(native, attr, None)
                if value:
                    hwnd = int(value)
                    break
            if hwnd:
                winops.tool_window(hwnd, on_top=True, no_activate=True)
        except Exception as exc:  # noqa: BLE001
            log.debug("overlay styling skipped: %s", exc)

    bar_controller.visible = False
    config.BAR_CONTROLLER = bar_controller


    def start_background_ear() -> None:
        """Switch the always-on listener on, once the server is answering commands."""
        try:
            import wake

            outcome = wake.LISTENER.start()
            log.info("background ear: %s", outcome.get("message", ""))
        except Exception as exc:  # noqa: BLE001 - optional, never fatal
            log.warning("background listening unavailable: %s", exc)

    threading.Timer(2.0, start_background_ear).start()

    def on_loaded() -> None:  # noqa: ANN001 - pywebview passes no args
        """Tell the file://-loaded HUD where its WebSocket lives, then decorate it."""
        payload = {
            "backend": f"http://127.0.0.1:{port}",
            "socket": f"ws://127.0.0.1:{port}/ws",
            "port": port,
            "speak_replies": SETTINGS.speak_replies,
            "frameless": bool(args.frameless),
        }
        script = (
            f"window.JARVIS_BACKEND = {json.dumps(payload['backend'])};"
            f"window.JARVIS_SOCKET = {json.dumps(payload['socket'])};"
            f"window.JARVIS_BOOT = {json.dumps(payload)};"
            "document.title='JARVIS';"
            "document.documentElement.dataset.chrome = 'webview';"
        )
        try:
            # One round-trip, not four: every evaluate_js waits on pywebview's
            # ready/loaded events, so stacking them multiplies any stall.
            window.evaluate_js(script)
        except Exception as exc:  # noqa: BLE001 - older pywebview versions
            log.debug("evaluate_js failed: %s", exc)

    #: ``window.events.loaded`` is an Event *object*, so subscribing means ``+=``;
    #: decorating with it calls the event and raises "TypeError: 'Event' object is
    #: not callable" - which is exactly what :func:`attach_loaded_handler` avoids.
    subscribed = attach_loaded_handler(window, on_loaded)

    try:
        # Starts the native window loop (blocks until the window closes).  With no
        # load event available we pass the bootstrap as the start callback instead.
        if subscribed:
            webview.start(debug=bool(args.devtools))
        else:
            log.warning("no pywebview load event; bootstrapping the HUD from webview.start()")
            webview.start(on_loaded, (), debug=bool(args.devtools))
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # noqa: BLE001
        log.error("window closed with an error (%s); serving the HUD in the browser instead", exc)
        try:
            webbrowser.open(f"http://127.0.0.1:{port}/")
            while http.alive:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    finally:
        http.stop()
    return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="main.py", description="Launch J.A.R.V.I.S. (core + 3D HUD).")
    parser.add_argument("--host", default=None, help=f"bind address (default {SETTINGS.host})")
    parser.add_argument("--port", type=int, default=SETTINGS.port, help=f"port (default {SETTINGS.port})")
    parser.add_argument("--keep-port", action="store_true", help="fail instead of auto-bumping a busy port")
    parser.add_argument("--wait", type=float, default=35.0, help="seconds to wait for the core to answer")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--frameless", action="store_true", help="borderless HUD window")
    parser.add_argument("--always-on-top", action="store_true", help="keep the HUD above other windows")
    parser.add_argument("--no-window", action="store_true", help="server only (no pywebview)")
    parser.add_argument("--no-voice", action="store_true", help="skip Whisper/XTTS warm-up")
    parser.add_argument("--no-discord", action="store_true", help="do not start the Discord bridge")
    parser.add_argument("--browser", action="store_true", help="with --no-window, open the system browser")
    parser.add_argument("--devtools", action="store_true", help="open pywebview devtools (debug builds)")
    parser.add_argument("--url", action="store_true", help="load the HUD over http:// instead of file://")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(launch(parse_args()))
