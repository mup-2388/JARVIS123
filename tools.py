"""
tools.py -- J.A.R.V.I.S. hands: OS automation, live data and local knowledge.

Every function here is *executed* (Track 1 regex hits) or *selected by the LLM*
(Track 2, see ``router.py``). Contract for all of them:

    return {"ok": bool, "message": str, ...extra structured fields}

``message`` is always safe to speak aloud and to print into the HUD terminal.
No function ever raises into the caller: failures are converted into
``ok=False`` with an actionable reason, because a voice assistant must always
answer the user, even when a tool blew up.

Windows-only capabilities (``os.startfile``, registry lookups, SendKeys volume
control, PowerShell screenshots) are feature-detected, so the same module also
runs unmodified on Linux/macOS while developing the agent logic.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import threading
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib import error as _urlerror
from urllib import parse as _urlparse
from urllib import request as _urlrequest

import config
from config import SETTINGS, get_logger

# The desktop-control layers.  Each one degrades to an ``{ok: False, message}`` dict when the
# platform cannot provide it, so importing tools never fails on a non-Windows dev box.
try:
    import apps as _apps
except Exception:  # noqa: BLE001 - a broken helper must never take the whole tool layer down
    _apps = None
try:
    import files as _files
except Exception:  # noqa: BLE001
    _files = None
try:
    import screen as _screen
except Exception:  # noqa: BLE001
    _screen = None
try:
    import winops as _winops
except Exception:  # noqa: BLE001
    _winops = None
try:
    import reminders as _reminders
except Exception:  # noqa: BLE001
    _reminders = None

log = get_logger("tools")

_OK = "ok"
_cache_lock = threading.RLock()

# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _slug(value: str) -> str:
    """``"FC 26 on Eden"`` -> ``"fc-26-on-eden"`` (stable keys for dicts/caches)."""
    value = unicodedata.normalize("NFKD", value or "")
    value = value.encode("ascii", "ignore").decode("ascii", "ignore")
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value or "app"


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (value or "").lower())


def _token_set(value: str) -> set[str]:
    stop = {"the", "a", "an", "on", "in", "for", "app", "application", "exe", "of", "my", "please"}
    return {t for t in _norm(value).split() if t and t not in stop}


def _clip(text: str, limit: int = 900) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _run(cmd: List[str], timeout: int = 20, env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    """stdout/err capture that never propagates OSError to the caller."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, **(env or {})},
            encoding="utf-8",
            errors="replace",
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError as exc:
        return 127, "", f"command not found: {exc}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s: {' '.join(cmd)}"
    except Exception as exc:  # pragma: no cover - defensive
        return 1, "", f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Windows application catalog
# ---------------------------------------------------------------------------
# Each entry: how to *definitely* start it. ``uri`` (protocol handler) beats a
# path lookup because Steam/Discord register schemes that always resolve, even
# when the .exe lives on D: or inside a per-user install folder.
APP_CATALOG: Dict[str, Dict[str, Any]] = {
    "steam": {
        "aliases": {"steam", "steam client", "big picture"},
        "uri": "steam://open/main",
        "paths": [
            r"C:\Program Files (x86)\Steam\Steam.exe",
            r"C:\Program Files\Steam\Steam.exe",
            r"D:\Steam\Steam.exe",
            r"E:\Steam\Steam.exe",
        ],
        "process": "steam.exe",
    },
    "discord": {
        "aliases": {"discord", "discord app", "canary"},
        "uri": "discord://discordapp.com/client",
        "paths": [
            r"%LOCALAPPDATA%\Discord\Discord.exe",
            r"%LOCALAPPDATA%\Discord\app\Discord.exe",
            r"%LOCALAPPDATA%\Programs\Discord\Discord.exe",
        ],
        "process": "discord.exe",
    },
    "eden": {
        "aliases": {"eden", "eden emulator", "fc 26", "fc26", "ea fc 26", "nintendo switch emulator"},
        "uri": "",
        "paths": [
            r"D:\Emulators\Eden\eden.exe",
            r"C:\Eden\eden.exe",
            r"%LOCALAPPDATA%\Programs\Eden\eden.exe",
            r"%APPDATA%\Eden\eden.exe",
        ],
        "process": "eden.exe",
    },
    "chrome": {
        "aliases": {"chrome", "google chrome", "web browser", "browser"},
        "uri": "start chrome",
        "paths": [
            r"%PROGRAMFILES%\Google\Chrome\Application\chrome.exe",
            r"%PROGRAMFILES(X86)%\Google\Chrome\Application\chrome.exe",
            r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe",
        ],
        "process": "chrome.exe",
        "url_arg": "--new-window {url}",
    },
    "edge": {
        "aliases": {"edge", "ms edge", "microsoft edge"},
        "uri": "start msedge",
        "paths": [
            r"%PROGRAMFILES(X86)%\Microsoft\Edge\Application\msedge.exe",
            r"%PROGRAMFILES%\Microsoft\Edge\Application\msedge.exe",
        ],
        "process": "msedge.exe",
        "url_arg": "--new-window {url}",
    },
    "firefox": {
        "aliases": {"firefox", "mozilla"},
        "uri": "start firefox",
        "paths": [r"%PROGRAMFILES%\Mozilla Firefox\firefox.exe", r"%PROGRAMFILES(X86)%\Mozilla Firefox\firefox.exe"],
        "process": "firefox.exe",
        "url_arg": "-new-window {url}",
    },
    "code": {
        "aliases": {"vs code", "vscode", "visual studio code", "code"},
        "uri": "vscode://file/{cwd}",
        "paths": [r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe", r"%PROGRAMFILES%\Microsoft VS Code\Code.exe"],
        "process": "code.exe",
    },
    "terminal": {
        "aliases": {"terminal", "windows terminal", "cmd", "command prompt", "powershell", "console"},
        "uri": "start wt",
        "paths": [],
        "process": "WindowsTerminal.exe",
        "fallback": "cmd.exe",
    },
    "explorer": {
        "aliases": {"explorer", "file explorer", "files", "this pc"},
        "uri": "explorer.exe shell:MyComputerFolder",
        "paths": [r"%WINDIR%\explorer.exe"],
        "process": "explorer.exe",
    },
    "settings": {
        "aliases": {"settings", "windows settings", "control panel"},
        "uri": "ms-settings:",
        "paths": [],
        "process": "SystemSettings.exe",
    },
    "taskmgr": {
        "aliases": {"task manager", "taskmgr", "activity monitor"},
        "uri": "taskmgr.exe",
        "paths": [r"%WINDIR%\System32\Taskmgr.exe"],
        "process": "Taskmgr.exe",
    },
    "spotify": {
        "aliases": {"spotify", "music"},
        "uri": "spotify:",
        "paths": [r"%APPDATA%\Spotify\Spotify.exe", r"%PROGRAMFILES%\Spotify\Spotify.exe"],
        "process": "spotify.exe",
    },
    "notepad": {
        "aliases": {"notepad", "text editor"},
        "uri": "start notepad",
        "paths": [r"%WINDIR%\System32\notepad.exe", r"%PROGRAMFILES%\Windows NT\Accessories\wordpad.exe"],
        "process": "Notepad.exe",
    },
    "paint": {"aliases": {"paint", "mspaint"}, "uri": "start mspaint", "paths": [], "process": "mspaint.exe"},
    "calculator": {
        "aliases": {"calculator", "calc"},
        "uri": "start calc",
        "paths": [r"%WINDIR%\System32\calc.exe"],
        "process": "CalculatorApp.exe",
    },
    "obs": {"aliases": {"obs", "obs studio", "streaming app"}, "uri": "", "paths": [r"%PROGRAMFILES%\obs-studio\bin\64bit\obs64.exe"], "process": "obs64.exe"},
    "youtube": {
        "aliases": {"youtube", "yt"},
        "uri": "https://www.youtube.com",
        "paths": [],
        "process": "",
        "web": "https://www.youtube.com",
    },
    "whatsapp": {"aliases": {"whatsapp", "wa"}, "uri": "https://web.whatsapp.com", "paths": [], "process": "", "web": "https://web.whatsapp.com"},
    "gmail": {"aliases": {"gmail", "mail", "inbox"}, "uri": "https://mail.google.com", "paths": [], "process": "", "web": "https://mail.google.com"},
}

_ALIAS_INDEX: Dict[str, str] = {}
for _key, _meta in APP_CATALOG.items():
    _ALIAS_INDEX[_key] = _key
    for _alias in _meta["aliases"]:
        _ALIAS_INDEX[_alias] = _key


def _expand(path: str) -> str:
    """Expand ``%VAR%`` and ``~``; Windows-friendly and harmless on POSIX."""
    if not path:
        return ""
    path = os.path.expandvars(path)
    if "$" in path:  # .env written with POSIX style vars
        path = os.path.expanduser(path)
    return str(Path(path).expanduser())


def _custom_apps() -> Dict[str, Dict[str, Any]]:
    """``CUSTOM_APPS=Eden=D:\\Emulators\\Eden\\eden.exe;Rust=...``"""
    out: Dict[str, Dict[str, Any]] = {}
    for chunk in (SETTINGS.custom_apps or "").split(";"):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        name, _, exe = chunk.partition("=")
        name, exe = name.strip(), _expand(exe.strip())
        if not name or not exe:
            continue
        key = _slug(name)
        out[key] = {
            "aliases": {name.lower(), key},
            "uri": "",
            "paths": [exe],
            "process": Path(exe).name,
            "display": name,
        }
        _ALIAS_INDEX[_norm(name)] = key
        _ALIAS_INDEX[key] = key
    return out


_registry_cache: Optional[Dict[str, Dict[str, str]]] = None


def _registry_installs() -> Dict[str, Dict[str, str]]:
    """Enumerate DisplayName -> InstallLocation/UninstallString from the registry.

    This is how JARVIS finds apps that were installed to a non-default drive
    (very common for Steam libraries and Switch emulators) without shipping a
    brittle hardcoded path list.
    """
    global _registry_cache
    with _cache_lock:
        if _registry_cache is not None:
            return _registry_cache
        found: Dict[str, Dict[str, str]] = {}
        if sys.platform.startswith("win"):
            try:
                import winreg  # type: ignore

                hives = (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER)
                subkeys = (
                    r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
                    r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
                )
                for hive in hives:
                    for sub in subkeys:
                        try:
                            root = winreg.OpenKey(hive, sub)
                        except OSError:
                            continue
                        try:
                            for i in range(winreg.QueryInfoKey(root)[0]):
                                try:
                                    name = winreg.EnumKey(root, i)
                                    key = winreg.OpenKey(root, name)
                                    values: Dict[str, str] = {}
                                    for j in range(winreg.QueryInfoKey(key)[1]):
                                        vname, vdata, _ = winreg.EnumValue(key, j)
                                        values[vname] = str(vdata)
                                    display = values.get("DisplayName")
                                    loc = values.get("InstallLocation") or ""
                                    uninst = values.get("UninstallString") or ""
                                    exe = values.get("DisplayIcon") or ""
                                    if display and display not in found:
                                        found[display] = {
                                            "install_location": loc,
                                            "uninstall": uninst,
                                            "icon": exe,
                                        }
                                except OSError:
                                    continue
                        finally:
                            winreg.CloseKey(root)
            except Exception as exc:  # pragma: no cover - non-Windows
                log.debug("registry scan unavailable: %s", exc)
        _registry_cache = found
        return found


def _exe_from_uninstall(entry: Dict[str, str]) -> str:
    """Derive the *launch* exe from a registry entry (icon or install dir)."""
    icon = entry.get("icon", "")
    if icon:
        icon = icon.split(",")[0].strip().strip('"')
        if icon.lower().endswith(".exe"):
            return _expand(icon)
    loc = _expand(entry.get("install_location", ""))
    if loc and Path(loc).is_dir():
        try:
            candidates = sorted(
                (p for p in Path(loc).glob("*.exe")),
                key=lambda p: (
                    0 if not re.search(r"unins|update|crash|helper", p.stem, re.I) else 1,
                    -p.stat().st_size,
                ),
            )
            if candidates:
                return str(candidates[0])
        except OSError:
            return ""
    return ""


def _match_catalog_app(query: str) -> Optional[str]:
    q = _norm(query)
    if not q:
        return None
    q = re.sub(r"^(please |jarvis[ ,]+|ok google[ ,]+|hey jarvis[ ,]+)", "", q).strip()
    if q in _ALIAS_INDEX:
        return _ALIAS_INDEX[q]
    tokens = _token_set(q)
    best_key, best_score = None, 0.0
    for alias, key in _ALIAS_INDEX.items():
        a_tokens = _token_set(alias)
        if not a_tokens:
            continue
        if a_tokens <= tokens:
            score = len(a_tokens) / max(len(tokens), 1)
            if score > best_score:
                best_key, best_score = key, score
        else:
            overlap = len(a_tokens & tokens) / len(a_tokens | tokens)
            if overlap > best_score:
                best_key, best_score = key, overlap
    return best_key if best_score >= 0.34 else None


def is_known_app(name: str) -> bool:
    """Also true when :mod:`apps` can resolve the name from the Start Menu or registry."""
    if _apps is not None and _apps.resolve(name) is not None:
        return True
    """True when :func:`launch_app` has a realistic chance of resolving ``name``.

    Track 1 uses this as a precision gate: it only claims an "open X" utterance
    when X is in the catalog, in ``CUSTOM_APPS``, on ``PATH`` or in the Windows
    uninstall registry. Anything else is handed to the agent instead of failing.
    """
    name = (name or "").strip()
    if not name:
        return False
    if _match_catalog_app(name):
        return True
    key = _slug(name)
    if key in _custom_apps():
        return True
    for guess in (key, f"{key}.exe", name, Path(name).name):
        if guess and shutil.which(guess):
            return True
    if config.is_windows():
        wanted = _norm(name)
        for display in _registry_installs():
            dn = _norm(display)
            if wanted and (wanted in dn or dn in wanted):
                return True
    return False


def _pid_names(process: str) -> List[int]:
    try:
        import psutil
    except Exception:
        return []
    out = []
    target = process.lower()
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            if target and (proc.info["name"] or "").lower() == target:
                out.append(int(proc.info["pid"]))
        except Exception:
            continue
    return out


def launch_app(app_name: str, url: str = "", args: str = "") -> Dict[str, Any]:
    """Launch *any* app the user can name, resolved by :mod:`apps`.

    Settings (``ms-settings:``), Microsoft Teams (MSIX package id), Control Panel applets,
    Start-menu shortcuts, registry installs on other drives and the user's own ``CUSTOM_APPS``
    entries all resolve here, and every one of them is started with ``ShellExecuteW`` - the
    call Explorer makes - because ``CreateProcess`` cannot launch a protocol URI, which is
    precisely why "open settings" used to fail.
    """
    if _apps is None:
        return {"ok": False, "message": "The app resolver is unavailable on this install."}
    outcome = _apps.launch(app_name, url=url, args=args)
    if not outcome.get("ok") and outcome.get("did_you_mean"):
        outcome["message"] = str(outcome.get("message", "")) + (
            " Say the name again exactly and I'll use it.")
    return outcome


def focus_app(name: str) -> Dict[str, Any]:
    """Bring an already-running app to the front instead of opening a second window."""
    if _apps is None:
        return {"ok": False, "message": "The app resolver is unavailable on this install."}
    return _apps.focus(name)


def list_apps(query: str = "") -> Dict[str, Any]:
    """Everything installed that JARVIS can launch (also the source of "did you mean")."""
    if _apps is None:
        return {"ok": False, "message": "The app resolver is unavailable.", "apps": []}
    return _apps.installed(query=query)


def windows_on_screen(limit: str = "") -> Dict[str, Any]:
    """The open windows, with the focused one marked - "what am I looking at", cheaply."""
    if _winops is None:
        return {"ok": False, "message": "Window listing needs the desktop layer."}
    rows = _winops.windows()
    try:
        count = max(1, min(int(limit or 12), 40))
    except (TypeError, ValueError):
        count = 12
    front = _winops.foreground_window()
    lines = [f"{str(r.get('title', ''))[:64]} [{r.get('process', '?')}]" for r in rows[:count]]
    return {"ok": True,
            "message": (f"Focused: “{front.get('title', '')}”. " if front.get("ok") else "")
                       + (f"{len(rows)} windows open: " + "; ".join(lines[:8]) if rows
                          else "No windows could be listed here."),
            "focused": str(front.get("title", "")) if front.get("ok") else "",
            "windows": [{"title": str(r.get("title", "")), "process": str(r.get("process", "")),
                         "minimized": bool(r.get("minimized"))} for r in rows[:count]],
            "count": len(rows)}


# --------------------------------------------------------------------------- files, as one tool
_FILE_ACTIONS = {"list", "read", "write", "append", "overwrite", "delete", "search", "find",
                 "move", "copy", "rename", "mkdir", "note", "undo", "disk", "open", "script",
                 "run", "recent"}


def manage_files(action: str = "list", path: str = "", content: str = "", destination: str = "",
                 query: str = "", text: str = "", confirm: str = "", run: str = "",
                 language: str = "", limit: str = "") -> Dict[str, Any]:
    """Create, read, change, search and remove files - journal-first, Recycle-Bin only.

    Safety, in this order: writes are confined to ``FILES_ROOT`` (+ ``FILES_ALLOWED``); a
    path outside them comes back as ``needs_confirmation`` with a token instead of happening;
    every overwrite copies the old bytes to ``data/file_backups`` first, and every delete
    copies to ``data/file_trash`` before the Recycle Bin, so ``undo`` always has something to
    restore.  ``script`` + ``run`` executes code JARVIS just wrote, with its output returned.
    """
    if _files is None:
        return {"ok": False, "message": "The file layer is unavailable on this install."}
    action = (action or "list").strip().lower()
    if action not in _FILE_ACTIONS:
        return {"ok": False, "message": f"I don't have a file action “{action}”. Try: "
                                       + ", ".join(sorted(_FILE_ACTIONS)) + "."}
    try:
        cap = int(limit) if str(limit or "").strip() else 20
    except (TypeError, ValueError):
        cap = 20
    if action == "list":
        return _files.list_dir(path, sort=text or "name", limit=cap)
    if action == "read":
        return _files.read(path)
    if action in ("write", "create"):
        return _files.write(path, content or text, mode="create" if action == "write" else "create",
                            confirm=confirm)
    if action == "overwrite":
        return _files.write(path, content or text, mode="overwrite", confirm=confirm)
    if action == "append":
        return _files.write(path, content or text, mode="append", confirm=confirm)
    if action == "delete":
        return _files.delete(path, confirm=confirm)
    if action in ("search", "find"):
        return _files.search(name=query or path, contains=text or content, root=path if query else "",
                             limit=cap)
    if action == "move":
        return _files.move(path, destination=destination)
    if action == "copy":
        return _files.move(path, destination=destination, copy="yes")
    if action == "rename":
        return _files.move(path, rename=destination or text)
    if action == "mkdir":
        return _files.mkdir(path or text)
    if action == "note":
        return _files.note(query or path or "note", text or content)
    if action == "undo":
        return _files.undo(cap)
    if action == "disk":
        return _files.disk_report(path)
    if action == "open":
        return _files.open_file(path, select=text)
    if action == "script":
        return _files.script(path or query or "script", language or "python", content or text, run=run or "no")
    if action == "run":
        return _files.execute_script(path, language or "python")
    return _files.recent(cap)


# --------------------------------------------------------------------------- desktop, as one tool
def control_desktop(action: str = "type", text: str = "", keys: str = "", combo: str = "",
                    x: str = "", y: str = "", button: str = "left", amount: str = "",
                    level: str = "", path: str = "") -> Dict[str, Any]:
    """Keyboard, mouse, windows, clipboard, volume, media, wallpaper, notifications.

    Real ``SendInput`` events through :mod:`winops` (no pywin32), so this drives whatever has
    focus - the same keys the user would press.  Typing long text goes via the clipboard +
    Ctrl+V because it is faster and cannot mangle case; the clipboard is restored afterwards.
    """
    if _winops is None:
        return {"ok": False, "message": "The desktop layer is unavailable on this install."}

    def num(value: str, default: int = 0) -> int:
        try:
            return int(str(value).strip() or default)
        except (TypeError, ValueError):
            return default

    action = (action or "type").strip().lower()
    if action in {"type", "write", "text"}:
        return _winops.type_text(text, press_enter="yes" if text.strip().lower().endswith("\n") else "")
    if action == "press":
        return _winops.press(keys or text, times=max(1, num(amount, 1)))
    if action in {"hotkey", "shortcut", "combo"}:
        return _winops.hotkey(combo or keys or text)
    if action in {"click", "doubleclick", "double", "rightclick", "right-click"}:
        double = "yes" if action in {"doubleclick", "double"} else ""
        return _winops.click(num(x, -1), num(y, -1),
                             button="right" if action in {"rightclick", "right-click"} else (button or "left"),
                             clicks=max(1, num(amount, 1)), double=bool(double))
    if action in {"move", "move_mouse"}:
        return _winops.move_mouse(num(x, 0), num(y, 0))
    if action in {"scroll", "scroll_down", "scroll_up"}:
        steps = num(amount, 3)
        return _winops.scroll(-steps if action != "scroll_up" else steps, num(x, -1), num(y, -1))
    if action in {"minimize", "maximize", "restore", "hide_window"}:
        front = _winops.foreground_window()
        if not front.get("ok"):
            return {"ok": False, "message": front.get("message", "No focused window.")}
        state = {"minimize": "minimize", "hide_window": "hide"}.get(action, action)
        return _winops.show_window(front.get("hwnd"), state)
    if action in {"focus", "activate", "bring_to_front", "alt_tab"}:
        if _apps is not None and text:
            return _apps.focus(text)
        return _winops.activate(title=text or keys)
    if action == "list_windows":
        return windows_on_screen(amount)
    if action in {"clipboard", "clipboard_read", "paste_board"}:
        return _winops.clipboard_read()
    if action == "clipboard_write":
        return _winops.clipboard_write(text)
    if action == "volume":
        verb = (text or "get").strip().lower()
        if verb.startswith(("up", "down", "set", "mute", "unmute", "get")):
            return _winops.volume(verb, level=num(level or amount, -1))
        return _winops.volume("set", level=num(verb, num(level or amount, 50)))
    if action in {"media", "music"}:
        return _winops.media(text or keys or "playpause")
    if action == "screenshot":
        return _winops.screenshot(path=path)
    if action == "lock":
        return _winops.lock_workstation()
    if action == "wallpaper":
        return _winops.set_wallpaper(path or text)
    if action in {"notify", "toast", "remind_later"}:
        return _winops.notify(text or "JARVIS", path or "")
    if action in {"battery", "power_status"}:
        return _winops.battery()
    if action in {"processes", "top_processes"}:
        rows = _winops.process_list()[:max(1, num(amount, 10))]
        return {"ok": True, "message": "Running now: " + ", ".join(
            f"{r['name']} ({r['memory_mb']:.0f} MB)" for r in rows[:8]), "processes": rows}
    if action in {"open_folder", "show_in_folder"}:
        return _winops.open_folder(path, select=text)
    if action == "beep":
        return _winops.beep(max(1, num(amount, 1)))
    return {"ok": False, "message": f"Unknown desktop action “{action}”."}


# --------------------------------------------------------------------------- eyes
def read_screen(action: str = "read", count: str = "3", question: str = "",
                target: str = "screen", save_to: str = "") -> Dict[str, Any]:
    """Look at the display: OCR it locally, list the top N items, or ask a vision model.

    ``read``/``list`` stay on the machine (Windows OCR, then Tesseract).  ``describe`` sends the
    capture to a provider model whose ``input_modalities`` include images - so on a Groq key
    that can see ``qwen3.x-27b`` but not a vision gpt-oss, it uses the right one and never
    400s on a text-only id.
    """
    if _screen is None:
        return {"ok": False, "message": "The screen layer is unavailable on this install."}
    action = (action or "read").strip().lower()
    try:
        want = max(1, min(int(count or 3), 10))
    except (TypeError, ValueError):
        want = 3
    if action in {"capture", "screenshot", "shot"}:
        return _screen.capture(target=target)
    if action in {"read", "ocr", "text"}:
        return _screen.ocr()
    if action in {"list", "top", "results", "listings"}:
        return _screen.read_screen(count=want, target=target)
    if action in {"describe", "what", "look", "explain"}:
        return _screen.describe(question=question, target=target)
    if action in {"window", "windows", "focused"}:
        return _screen.window_text()
    if action in {"save", "note"}:
        return _screen.save_note_from_screen(question, to=save_to)
    return {"ok": False, "message": f"Unknown screen action “{action}”: capture, read, list, "
                                   "describe, window or save."}


# --------------------------------------------------------------------------- time-based actions
def set_reminder(action: str = "add", text: str = "", when: str = "", minutes: str = "",
                 run: str = "") -> Dict[str, Any]:
    """Timers, reminders and scheduled commands, persisted across restarts.

    ``when`` takes whatever the user actually said ("in ten minutes", "at 7:30 pm", "tomorrow at
    9", "every 2 hours"); ``parse_when`` resolves it, including a clock time that has passed
    meaning tomorrow.  ``run`` marks the text as a *command* to execute at that time rather than
    a note to read out.
    """
    if _reminders is None:
        return {"ok": False, "message": "The reminder layer is unavailable on this install."}
    action = (action or "add").strip().lower()
    try:
        span = int(str(minutes or "").strip() or 10)
    except (TypeError, ValueError):
        span = 10
    when = when or (f"in {span} minutes" if span else "")
    if action in {"add", "set", "remind", "timer"}:
        return _reminders.BOARD.add(text=text or when, when=when, run=bool(run.strip()))
    if action in {"list", "show"}:
        rows = _reminders.BOARD.list()
        if not rows:
            return {"ok": True, "message": "You have nothing scheduled.", "reminders": []}
        return {"ok": True, "message": "Scheduled: " + "; ".join(
            f"{r['text'] or 'note'} {r.get('due_iso') or ''}" for r in rows[:6]), "reminders": rows}
    if action in {"cancel", "remove", "clear"}:
        return _reminders.BOARD.cancel(text or when)
    if action == "snooze":
        return _reminders.BOARD.snooze(text, span)
    return {"ok": False, "message": f"Unknown reminder action “{action}”: add, list, cancel or snooze."}


# --------------------------------------------------------------------------- background ear
def listening(status: str = "status") -> Dict[str, Any]:
    """Report or change background wake-word listening (the always-on ear)."""
    try:
        import wake

        listener = wake.LISTENER
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": f"The background listener is unavailable: {exc}"}
    action = (status or "status").strip().lower()
    if action in {"start", "on", "enable"}:
        return listener.start()
    if action in {"stop", "off", "disable"}:
        return listener.stop()
    if action in {"talk", "listen", "record"}:
        return listener.record_once(seconds=5)
    data = listener.status()
    return {"ok": bool(data.get("running")),
            "message": ("Listening for " + "/".join(data.get("wake_words") or ["jarvis"])
                        if data.get("running") else
                        f"Background listening is off: {data.get('detail') or 'not started'}"),
            "status": data}


def _spawn(exe: str, args: str, display: str, method: str, tried: List[str]) -> Dict[str, Any]:
    cmd = [exe] + ([args] if args else [])
    creation = 0x00000008 | 0x00000200 if config.is_windows() else 0  # DETACHED | NEW_PROCESS_GROUP
    try:
        proc = subprocess.Popen(  # noqa: S603 - path resolved above, no shell interpolation
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=creation,
            close_fds=True,
        )
        return {
            "ok": True,
            "message": f"{display} launched.",
            "pid": proc.pid,
            "exe": exe,
            "method": method,
            "tried": tried + [method],
        }
    except OSError as exc:
        return {"ok": False, "message": f"{display} failed to start: {exc}", "exe": exe, "tried": tried}


def close_app(app_name: str) -> Dict[str, Any]:
    """Close an app by resolved process name (Start menu, registry or catalogue)."""
    if _apps is not None:
        return _apps.close(app_name)
    return {"ok": False, "message": "The app resolver is unavailable on this install."}

def list_launchable_apps() -> Dict[str, Any]:
    """Names JARVIS can launch right now (catalog + custom + registry)."""
    names = sorted(set(APP_CATALOG) | set(_custom_apps()))
    reg = sorted(_registry_installs())[:40]
    return {
        "ok": True,
        "message": f"{len(names)} built-in shortcuts available.",
        "built_in": names,
        "registry": reg,
    }


def _open_url(url: str) -> Dict[str, Any]:
    url = url.strip()
    if not re.match(r"^https?://", url):
        url = "https://" + url.lstrip("/")
    if config.is_windows():
        code, _, err = _run(["cmd", "/c", "start", "", url], timeout=SETTINGS.launch_timeout)
    else:
        opener = shutil.which("xdg-open") or shutil.which("open")
        if not opener:
            return {"ok": False, "message": "No browser opener available on this system."}
        code, _, err = _run([opener, url], timeout=SETTINGS.launch_timeout)
    if code == 0:
        return {"ok": True, "message": f"Opened {url}.", "url": url}
    return {"ok": False, "message": f"Could not open {url}: {err or 'browser rejected the request'}", "url": url}


def open_website(target: str, query: str = "") -> Dict[str, Any]:
    """Open a site, optionally deep-linking into its search results."""
    target = (target or "").strip()
    known = {
        "youtube": "https://www.youtube.com/results?search_query=",
        "yt": "https://www.youtube.com/results?search_query=",
        "google maps": "https://www.google.com/maps/search/",
        "maps": "https://www.google.com/maps/search/",
        "github": "https://github.com/search?q=",
        "amazon": "https://www.amazon.in/s?k=",
        "flipkart": "https://www.flipkart.com/search?q=",
        "wikipedia": "https://en.wikipedia.org/wiki/Special:Search?search=",
        "gmail": "https://mail.google.com/mail/search?q=",
        "reddit": "https://www.reddit.com/search/?q=",
    }
    base = known.get(_norm(target))
    if base and query:
        return _open_url(base + _urlparse.quote_plus(query))
    return _open_url(target if "." in target or target.startswith("http") else f"https://www.{_slug(target)}.com")


def play_youtube(query: str) -> Dict[str, Any]:
    """Resolve a YouTube query through DDG, then deep-link the browser straight into it."""
    query = (query or "").strip()
    if not query:
        return _open_url("https://www.youtube.com")
    res = web_search(query, max_results=5, site="youtube.com")
    search_url = f"https://www.youtube.com/results?search_query={_urlparse.quote_plus(query)}"
    deep = _open_url(search_url)
    deep["query"] = query
    deep["search_url"] = search_url
    if res["ok"] and res["results"]:
        first = res["results"][0]
        deep["top_result"] = first
        deep["message"] = f"Playing {first.get('title', query)} on YouTube."
        if first.get("href", "").startswith("https://www.youtube.com/watch"):
            deep["direct_video"] = first["href"]
    elif not deep.get("ok"):
        deep["message"] = f"I searched YouTube for “{query}” but could not open the browser: {deep.get('message', '')}"
    return deep


# ---------------------------------------------------------------------------
# Web search (live)
# ---------------------------------------------------------------------------

def _load_ddgs() -> Tuple[Any, str]:
    """Return ``(DDGS instance, backend name)`` for either package spelling."""
    try:
        from ddgs import DDGS  # type: ignore  # renamed package (>= 8.x)

        return DDGS(), "ddgs"
    except Exception:
        pass
    try:
        from duckduckgo_search import DDGS  # type: ignore  # legacy package name

        return DDGS(), "duckduckgo_search"
    except Exception as exc:
        raise RuntimeError(f"duckduckgo-search is not installed ({exc})")


def web_search(
    query: str,
    max_results: int = 6,
    region: str = "",
    safesearch: str = "moderate",
    timelimit: str = "",
    site: str = "",
) -> Dict[str, Any]:
    """Live DuckDuckGo results -- e.g. ``web_search("cheapest Indian restaurants in Surat")``.

    Tries the JSON ``text`` backend first and retries with the HTML backend,
    which is what survives DDG rate limits on residential connections.
    """
    query = (query or "").strip()
    if not query:
        return {"ok": False, "message": "Search for what?"}
    if site and f"site:{site}" not in query:
        query = f"{query} site:{site}"
    region = region or ("in-en" if re.search(r"\b(surat|india|indian|₹|rupee)\b", query, re.I) else "wt-en")
    max_results = max(1, min(int(max_results or 6), 20))

    try:
        ddgs, backend = _load_ddgs()
    except RuntimeError as exc:
        return {"ok": False, "message": f"Search unavailable: {exc}", "query": query}

    last_err = ""
    for backend_name in ("auto", "html", "lite"):
        kwargs = {"region": region, "safesearch": safesearch, "max_results": max_results}
        if timelimit:
            kwargs["timelimit"] = timelimit
        try:
            try:
                raw = ddgs.text(query, backend=backend_name, **kwargs)
            except TypeError:
                # duckduckgo-search < 6.x called this argument ``engine``.
                raw = ddgs.text(query, engine=backend_name, **kwargs)
            results = [
                {
                    "title": _clip(item.get("title", ""), 160),
                    "url": item.get("href") or item.get("url", ""),
                    "snippet": _clip(item.get("body", ""), 400),
                }
                for item in (raw or [])
                if item
            ]
            if results:
                bullets = "\n".join(f"{i+1}. {r['title']} — {r['snippet']}" for i, r in enumerate(results[:5]))
                return {
                    "ok": True,
                    "query": query,
                    "count": len(results),
                    "results": results,
                    "backend": backend_name,
                    "message": f"Top result for “{query}”: {results[0]['title']}. "
                    f"{results[0]['snippet']}".strip(),
                    "bullets": bullets,
                }
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            log.debug("ddg backend %s failed: %s", backend_name, exc)
            continue

    return {
        "ok": False,
        "message": f"DuckDuckGo returned nothing for “{query}”" + (f" ({last_err})" if last_err else "."),
        "query": query,
        "results": [],
    }


def search_news(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Headlines via the DDG news backend (used for 'what's the latest on X')."""
    try:
        ddgs, _backend = _load_ddgs()
        try:
            raw = ddgs.news(query, max_results=max_results, backend="api")
        except TypeError:
            raw = ddgs.news(query, max_results=max_results)
        items = [
            {
                "title": _clip(item.get("title", ""), 160),
                "source": item.get("source", ""),
                "when": item.get("date", ""),
                "url": item.get("url", ""),
            }
            for item in (raw or [])
        ]
        if items:
            return {"ok": True, "count": len(items), "results": items, "query": query,
                    "message": "Latest: " + "; ".join(f"{i['title']} ({i['source']})" for i in items[:3])}
    except Exception as exc:
        return {"ok": False, "message": f"News search failed: {type(exc).__name__}: {exc}"}
    return {"ok": False, "message": f"No news for “{query}”."}


# ---------------------------------------------------------------------------
# Sports stats -- api-sports.io v3
# ---------------------------------------------------------------------------

_TEAM_CACHE: Path = SETTINGS.cache_path / "teams_cache.json"


def _json_cache_load(path: Path) -> Dict[str, Any]:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _json_cache_save(path: Path, data: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        log.debug("cache write failed: %s", exc)


def _api_sports_get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Signed REST call to api-sports.io with rate-limit + error surfacing."""
    if not SETTINGS.api_sports_key:
        raise RuntimeError("API_SPORTS_KEY is empty -- add your key from https://dashboard.api-football.com to .env")

    query = _urlparse.urlencode({k: v for k, v in (params or {}).items() if v not in (None, "")})
    url = f"{SETTINGS.api_sports_base.rstrip('/')}/{endpoint.lstrip('/')}" + (f"?{query}" if query else "")
    req = _urlrequest.Request(
        url,
        headers={
            "x-apisports-key": SETTINGS.api_sports_key,
            "accept": "application/json",
            "user-agent": "JARVIS-assistant/1.0",
        },
        method="GET",
    )
    try:
        with _urlrequest.urlopen(req, timeout=SETTINGS.api_sports_timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
    except _urlerror.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        if exc.code == 429:
            wait = exc.headers.get("Retry-After", "60")
            raise RuntimeError(f"api-sports rate limit hit, retry in {wait}s") from exc
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body or exc.reason}") from exc
    except (_urlerror.URLError, TimeoutError) as exc:
        raise RuntimeError(f"network error calling api-sports: {exc}") from exc

    if isinstance(payload, dict) and payload.get("errors"):
        errs = payload["errors"]
        if isinstance(errs, dict):
            errs = list(errs.values()) or list(errs.keys())
        raise RuntimeError("api-sports error: " + "; ".join(str(e) for e in errs)[:400])
    return payload.get("response", payload)


def resolve_team_id(name: str) -> Dict[str, Any]:
    """Fuzzy team name -> api-sports ``team_id`` (cached to ``data/teams_cache.json``)."""
    cache = _json_cache_load(_TEAM_CACHE)
    key = _slug(name)
    if key in cache and cache[key].get("id"):
        return cache[key]

    response = _api_sports_get("teams", {"search": name})
    if not response:
        raise RuntimeError(f"api-sports does not recognise “{name}”")
    best = None
    want = _norm(name)
    for item in response:
        team = item.get("team") or {}
        tname = _norm(team.get("name", ""))
        exact = tname == want or want in tname
        score = (2 if exact else 1) + (0.5 if team.get("country") else 0)
        if best is None or score > best["_score"]:
            best = {"_score": score, "id": team.get("id"), "name": team.get("name"), "country": team.get("country"),
                    "logo": team.get("logo"), "code": team.get("code")}
    if not best:
        raise RuntimeError(f"no team match for “{name}”")
    best.pop("_score", None)
    best["founded"] = (response[0].get("team") or {}).get("founded")
    cache[key] = best
    _json_cache_save(_TEAM_CACHE, cache)
    return best


def fetch_sports_stats(team: str = "", kind: str = "all", season_year: str = "") -> Dict[str, Any]:
    """Live football stats from ``v3.football.api-sports.io``.

    ``kind``:
      * ``results``  last 5 fixtures (opponent, score, date, league)
      * ``fixtures`` next 5 fixtures
      * ``standing`` table position of the season's league
      * ``scorers``  squad top scorers
      * ``all``      compact brief combining everything (default)
    """
    team = (team or SETTINGS.api_sports_default_team or "").strip()
    if not team:
        return {"ok": False, "message": "Which team? e.g. Real Madrid."}

    try:
        meta = resolve_team_id(team)
    except RuntimeError as exc:
        return {"ok": False, "message": f"Team lookup failed: {exc}", "team": team}

    team_id = meta["id"]
    kind = (kind or "all").lower()
    out: Dict[str, Any] = {"ok": True, "team": meta["name"], "team_id": team_id, "logo": meta.get("logo"), "requested": team}
    lines: List[str] = [f"{meta['name']}"]
    season = str(season_year or "").strip() or str(_dt.datetime.now().year - (1 if _dt.datetime.now().month < 8 else 0))
    league_ref: Optional[Tuple[Any, Any]] = None   # (league_id, season) reused for /standings

    try:
        if kind in {"all", "results"}:
            last = _api_sports_get("fixtures", {"last": 5, "team": team_id})
            results = []
            for fx in last:
                fixture, league, goal = fx.get("fixture") or {}, fx.get("league") or {}, fx.get("goals") or {}
                if league.get("id") and league_ref is None:
                    league_ref = (league["id"], league.get("season") or season)
                elapsed = fixture.get("status", {}).get("long") or "FT"
                results.append(
                    {
                        "date": (fixture.get("date") or "")[:10],
                        "opponent": (
                            (fx.get("teams", {}).get("away") or {}).get("name")
                            if (fx.get("teams", {}).get("home") or {}).get("id") == team_id
                            else (fx.get("teams", {}).get("home") or {}).get("name")
                        ),
                        "home": (fx.get("teams", {}).get("home") or {}).get("name"),
                        "away": (fx.get("teams", {}).get("away") or {}).get("name"),
                        "score": f"{goal.get('home', '-')} - {goal.get('away', '-')}",
                        "venue": (fixture.get("venue") or {}).get("name"),
                        "status": elapsed,
                        "league": league.get("name"),
                        "is_home": (fx.get("teams", {}).get("home") or {}).get("id") == team_id,
                        "winner": _outcome(fx, team_id),
                    }
                )
            out["recent"] = results
            if results:
                rec = results[0]
                lines.append(
                    f"Most recent result: {rec['home']} {rec['score']} {rec['away']} ({rec['date']}, {rec['status']})."
                )
                wins = sum(1 for r in results if r["winner"] == "W")
                draws = sum(1 for r in results if r["winner"] == "D")
                losses = sum(1 for r in results if r["winner"] == "L")
                lines.append(f"Form in last {len(results)}: {wins}W {draws}D {losses}L.")
                upcoming = [r for r in results if r["winner"] == "P"]
                if upcoming:
                    u = upcoming[0]
                    lines.append(f"Next fixture: {u['home']} vs {u['away']} on {u['date']} ({u['status']}).")

        if kind in {"all", "fixtures"}:
            nxt = _api_sports_get("fixtures", {"next": 5, "team": team_id})
            upcoming = []
            for fx in nxt:
                fixture = fx.get("fixture") or {}
                upcoming.append(
                    {
                        "date": (fixture.get("date") or "").replace("T", " ")[:16],
                        "home": (fx.get("teams", {}).get("home") or {}).get("name"),
                        "away": (fx.get("teams", {}).get("away") or {}).get("name"),
                        "league": (fx.get("league") or {}).get("name"),
                        "venue": (fixture.get("venue") or {}).get("name"),
                        "status": (fixture.get("status") or {}).get("long"),
                    }
                )
            out["upcoming"] = upcoming
            if upcoming and kind != "fixtures":
                first = upcoming[0]
                lines.append(f"Up next: {first['home']} vs {first['away']} on {first['date']} ({first['league']}).")

        if kind in {"all", "standing"}:
            if league_ref is None:
                # No fixture in this response (e.g. "standing" alone): one cheap
                # lookup for the competition, then reuse the free-tier quota.
                probe = _api_sports_get("fixtures", {"last": 1, "team": team_id})
                if probe:
                    probe_league = probe[0].get("league") or {}
                    league_ref = (probe_league.get("id"), probe_league.get("season") or season)
            if league_ref and league_ref[0]:
                standings = _api_sports_get("standings", {"league": league_ref[0], "season": league_ref[1] or season})
                for block in standings or []:
                    table = ((block.get("league") or {}).get("standings") or [[]])[0]
                    for row in table:
                        if row.get("team", {}).get("id") == team_id:
                            stat = row.get("all") or {}
                            out["standing"] = {
                                "rank": row.get("rank"),
                                "points": row.get("points"),
                                "played": row.get("played"),
                                "win": stat.get("win"),
                                "draw": stat.get("draw"),
                                "lose": stat.get("lose"),
                                "goals_for": stat.get("goals", {}).get("for") if isinstance(stat.get("goals"), dict) else None,
                                "goals_against": stat.get("goals", {}).get("against") if isinstance(stat.get("goals"), dict) else None,
                                "league": (block.get("league") or {}).get("name"),
                                "position": row.get("position"),
                                "description": row.get("description"),
                            }
                            s = out["standing"]
                            lines.append(
                                f"{s['rank']}th in {s['league']} with {s['points']} pts from {s['played']} "
                                f"({s['win']}W {s['draw']}D {s['lose']}L)."
                            )
                            break
                    if "standing" in out:
                        break

        if kind in {"all", "scorers"}:
            players = _api_sports_get("players", {"team": team_id, "top": 5})
            scorers = []
            for entry in players or []:
                player = entry.get("player") or {}
                stats = (entry.get("statistics") or [{}])[0]
                goals = stats.get("goals") or {}
                scorers.append(
                    {
                        "name": player.get("name"),
                        "position": player.get("position"),
                        "goals": goals.get("total"),
                        "assists": (stats.get("goals") or {}).get("assists"),
                        "minutes": stats.get("games", {}).get("minutes"),
                        "rating": stats.get("rating"),
                        "team": ((stats.get("team") or {}).get("name")),
                    }
                )
            scorers = sorted(scorers, key=lambda s: int(s["goals"] or 0), reverse=True)[:5]
            out["scorers"] = scorers
            if scorers and scorers[0]["goals"]:
                lines.append(f"Top scorer: {scorers[0]['name']} with {scorers[0]['goals']} goals.")
    except RuntimeError as exc:
        # Partial results are still useful -- report what we got and say why it is short.
        out["partial_error"] = str(exc)
        lines.append(f"(one section was unavailable: {_clip(str(exc), 120)})")

    out["message"] = " ".join(lines) if len(lines) > 1 else f"No data returned for {meta['name']} yet."
    out["summary"] = out["message"]
    return out


def _outcome(fixture: Dict[str, Any], team_id: int) -> str:
    goals, teams = fixture.get("goals") or {}, fixture.get("teams") or {}
    status = ((fixture.get("fixture") or {}).get("status") or {}).get("short", "")
    if status in {"NS", "TBD"}:
        return "P"
    if goals.get("home") is None or goals.get("away") is None:
        return "P"
    home_score, away_score = int(goals["home"]), int(goals["away"])
    is_home = (teams.get("home") or {}).get("id") == team_id
    mine, theirs = (home_score, away_score) if is_home else (away_score, home_score)
    if status in {"PST", "CANC", "ABD", "AWD", "WO"}:
        return "P"
    return "W" if mine > theirs else ("L" if mine < theirs else "D")


# ---------------------------------------------------------------------------
# Local knowledge: Markdown notes (German studies, CS degree prep, …)
# ---------------------------------------------------------------------------

_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.S)


def _split_front_matter(text: str) -> Tuple[Dict[str, str], str]:
    meta: Dict[str, str] = {}
    match = _FRONT_MATTER.match(text)
    if match:
        for line in match.group(1).splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                meta[k.strip().lower()] = v.strip().strip("\"'")
        text = text[match.end():]
    return meta, text


def _read_note_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    meta, body = _split_front_matter(text)
    headings = [h.strip("# ").strip() for h in re.findall(r"^\s*#{1,3}\s+(.+)$", body, re.M)]
    title = meta.get("title") or (headings[0] if headings else path.stem.replace("_", " ").replace("-", " ").title())
    tags = {t.strip().lower() for t in re.split(r"[,;]\s*", meta.get("tags", "")) if t.strip()}
    return {
        "path": str(path),
        "name": path.name,
        "title": title,
        "meta": meta,
        "tags": sorted(tags),
        "headings": headings[:12],
        "size": path.stat().st_size,
        "modified": _dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        "body": body,
    }


def _score_note(note: Dict[str, Any], tokens: set[str], raw: str) -> float:
    """Relevance for one note. Returns 0.0 when the topic is simply absent, so
    the recency tie-breaker below can never resurrect an unrelated file."""
    score = 0.0
    name_tokens = _token_set(note["name"].replace(".md", ""))
    title_tokens = _token_set(note["title"])
    tag_tokens = {t for tag in note["tags"] for t in _token_set(tag)}
    body_low = note["body"].lower()
    for tok in tokens:
        if tok in name_tokens:
            score += 4.0
        if tok in title_tokens:
            score += 3.0
        if tok in tag_tokens:
            score += 2.5
        if any(tok in h.lower() for h in note["headings"]):
            score += 2.0
        if f"#{tok}" in body_low:
            score += 1.0
        hits = body_low.count(tok)
        if hits:
            score += min(hits, 8) * 0.35
    if raw.strip().lower() in body_low:
        score += 2.0
    if score <= 0.0:
        return 0.0
    # Only now: small recency bonus so equally good notes break toward the newest.
    return score + max(0.0, 0.6 - (note["size"] / 200000))


def read_notes(topic: str = "", limit: int = 1, include_body: bool = True, max_chars: int = 2600) -> Dict[str, Any]:
    """Answer "what did I write about German dative?" from ``./notes``.

    Scoring blends file name (x4), title (x3), front-matter tags (x2.5),
    headings (x2) and body keyword density, so a note called
    ``german-dative.md`` outranks one that merely mentions dative twice.
    """
    notes_dir = SETTINGS.notes_path
    if not notes_dir.is_dir():
        return {
            "ok": False,
            "message": f"My notes folder is missing: {notes_dir}. Create it and drop Markdown files in.",
            "notes": [],
        }

    notes: List[Dict[str, Any]] = []
    for path in sorted(notes_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".markdown", ".txt"} or not path.is_file():
            continue
        if path.name.lower().startswith("readme"):
            continue
        try:
            notes.append(_read_note_file(path))
        except OSError as exc:
            log.debug("skip %s: %s", path, exc)

    if not notes:
        return {"ok": False, "message": f"No Markdown notes found in {notes_dir}.", "notes": []}

    topic = (topic or "").strip()
    if not topic:
        catalog = [
            {"name": n["name"], "title": n["title"], "tags": n["tags"], "modified": n["modified"], "topics": n["headings"][:4]}
            for n in sorted(notes, key=lambda n: n["modified"], reverse=True)
        ]
        return {
            "ok": True,
            "count": len(catalog),
            "notes": catalog,
            "message": f"You keep {len(catalog)} notes: " + ", ".join(c["title"] for c in catalog[:6]) + ".",
        }

    tokens = _token_set(topic)
    scored = sorted((( _score_note(n, tokens, topic), n) for n in notes), key=lambda pair: pair[0], reverse=True)
    best_score, best = scored[0]
    if best_score <= 0.5:
        return {
            "ok": False,
            "message": f"Nothing in your notes matches “{topic}”. Closest topics: "
            + ", ".join(n["title"] for _, n in scored[:3]),
            "searched": len(notes),
        }

    chosen = [n for s, n in scored[: max(1, min(int(limit or 1), 4))] if s >= max(0.5, best_score * 0.45)]
    payload = []
    for note in chosen:
        body = re.sub(r"\n{3,}", "\n\n", note["body"]).strip()
        if not include_body:
            body = ""
        payload.append(
            {
                "name": note["name"],
                "title": note["title"],
                "tags": note["tags"],
                "modified": note["modified"],
                "path": note["path"],
                "headings": note["headings"],
                "excerpt": _clip(body, max_chars),
                "chars": len(body),
            }
        )
    top = payload[0]
    return {
        "ok": True,
        "count": len(payload),
        "topic": topic,
        "notes": payload,
        "message": f"From “{top['title']}” ({top['modified']}): {_clip(top['excerpt'], 520)}",
    }


def write_note(topic: str, content: str, tags: str = "") -> Dict[str, Any]:
    """Persist a dictated note (Track 1 "note that …" and the agent's memory)."""
    topic, content = (topic or "").strip(), (content or "").strip()
    if not topic or not content:
        return {"ok": False, "message": "Give me a title and something worth remembering."}
    notes_dir = SETTINGS.notes_path
    notes_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y-%m-%d")
    path = notes_dir / f"{stamp}-{_slug(topic)}.md"
    if path.is_file() and re.search(r"\b(update|append|add to)\b", topic, re.I):
        previous = path.read_text(encoding="utf-8", errors="replace")
        path.write_text(
            previous.rstrip() + f"\n\n## Added {_dt.datetime.now():%H:%M}\n\n{content}\n",
            encoding="utf-8",
        )
        return {"ok": True, "path": str(path), "message": f"Updated “{topic}” — appended to {path.name}.",
                "chars": len(content), "appended": True}
    suffix = 1
    while path.exists():
        path = notes_dir / f"{stamp}-{_slug(topic)}-{suffix}.md"
        suffix += 1
    tag_line = ", ".join(t.strip() for t in re.split(r"[,;]\s*", tags or "") if t.strip())
    body = (
        f"---\ntitle: {topic}\ndate: {stamp}\ntags: [{tag_line or 'inbox'}]\nsource: jarvis-voice\n---\n\n"
        f"# {topic}\n\n{content}\n"
    )
    try:
        path.write_text(body, encoding="utf-8")
        return {"ok": True, "path": str(path), "message": f"Noted: {topic}. Saved to {path.name}.", "chars": len(content)}
    except OSError as exc:
        return {"ok": False, "message": f"Could not write the note: {exc}"}


# ---------------------------------------------------------------------------
# System telemetry (also streamed to the HUD over WebSocket)
# ---------------------------------------------------------------------------

_START_TS = _dt.datetime.now()


def gpu_stats() -> Dict[str, Any]:
    """NVIDIA telemetry via ``nvidia-smi`` -- present on the RTX 3050, absent elsewhere."""
    exe = shutil.which("nvidia-smi") or (
        r"C:\Windows\System32\nvidia-smi.exe" if config.is_windows() else None
    )
    if not exe:
        return {"available": False, "name": "", "util_pct": 0.0, "mem_used_mb": 0, "mem_total_mb": 0, "temp_c": None}
    code, out, _ = _run(
        [
            str(exe),
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=6,
    )
    if code != 0 or not out:
        return {"available": False, "name": "", "util_pct": 0.0, "mem_used_mb": 0, "mem_total_mb": 0, "temp_c": None}
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    try:
        return {
            "available": True,
            "name": parts[0],
            "util_pct": float(parts[1]),
            "mem_used_mb": int(float(parts[2])),
            "mem_total_mb": int(float(parts[3])),
            "temp_c": int(float(parts[4])) if len(parts) > 4 else None,
        }
    except (ValueError, IndexError):
        return {"available": True, "name": parts[0] if parts else "NVIDIA GPU", "util_pct": 0.0,
                "mem_used_mb": 0, "mem_total_mb": 0, "temp_c": None}


def system_report(detailed: bool = True) -> Dict[str, Any]:
    """CPU / RAM / disk / GPU / battery / top processes snapshot."""
    try:
        import psutil
    except Exception:
        return {
            "ok": False,
            "message": "psutil is not installed, so telemetry is unavailable.",
            "cpu": 0.0,
            "ram": 0.0,
        }

    cpu = psutil.cpu_percent(interval=None)
    per_core = psutil.cpu_percent(interval=None, percpu=True)
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = psutil.disk_usage(str(config.ROOT.drive + "\\") if config.is_windows() else "/")
    try:
        boot = _dt.datetime.fromtimestamp(psutil.boot_time())
    except Exception:
        boot = _START_TS
    net = psutil.net_io_counters()
    gpu = gpu_stats()

    payload: Dict[str, Any] = {
        "ok": True,
        "cpu": round(cpu, 1),
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_physical": psutil.cpu_count(logical=False),
        "per_core": [round(c, 1) for c in per_core],
        "load": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
        "ram": round(mem.percent, 1),
        "ram_used_gb": round(mem.used / 1e9, 2),
        "ram_total_gb": round(mem.total / 1e9, 2),
        "ram_available_gb": round(mem.available / 1e9, 2),
        "swap": round(swap.percent, 1),
        "disk": round(disk.percent, 1),
        "disk_free_gb": round(disk.free / 1e9, 1),
        "net_sent_mb": round(net.bytes_sent / 1e6, 1),
        "net_recv_mb": round(net.bytes_recv / 1e6, 1),
        "gpu": gpu,
        "boot_time": boot.strftime("%Y-%m-%d %H:%M"),
        "jarvis_uptime_s": int((_dt.datetime.now() - _START_TS).total_seconds()),
        "platform": f"{platform.system()} {platform.release()} (build {platform.version() if config.is_windows() else platform.release()})",
        "python": platform.python_version(),
        "process_count": len(psutil.pids()),
        "battery": _battery(),
        "message": "",
    }

    if detailed:
        top = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                top.append(proc.info)
            except Exception:
                continue
        top.sort(key=lambda p: (p.get("cpu_percent") or 0), reverse=True)
        payload["top_processes"] = [
            {
                "pid": p["pid"],
                "name": _clip(str(p["name"]), 40),
                "cpu": round(p.get("cpu_percent") or 0, 1),
                "mem": round(p.get("memory_percent") or 0, 1),
            }
            for p in top[:5]
        ]
        payload["message"] = (
            f"CPU {payload['cpu']}%, memory {payload['ram']}% of {payload['ram_total_gb']} gigabytes"
            + (f", GPU {gpu['util_pct']:.0f}% and {gpu['mem_used_mb']} of {gpu['mem_total_mb']} megabytes VRAM" if gpu["available"] else "")
            + f", disk {payload['disk']}% used."
        )
    return payload


def _battery() -> Dict[str, Any]:
    try:
        import psutil

        batt = psutil.sensors_battery()
        if batt:
            return {"percent": round(batt.percent, 1), "plugged": bool(batt.power_plugged), "time_min": int((batt.secsleft or 0) / 60)}
    except Exception:
        pass
    return {"percent": None, "plugged": True, "time_min": None}


def get_time() -> Dict[str, Any]:
    now = _dt.datetime.now()
    zone = (time.tzname[0] if time.tzname and time.tzname[0] else "local")
    return {
        "ok": True,
        "iso": now.isoformat(timespec="seconds"),
        "time": now.strftime("%I:%M %p").lstrip("0"),
        "date": now.strftime("%A, %d %B %Y"),
        "timezone": zone,
        "message": f"It is {now.strftime('%I:%M %p').lstrip('0')} on {now.strftime('%A, %d %B')}.",
    }


def set_volume(level: Optional[int] = None, delta: Optional[int] = None, mute: bool = False) -> Dict[str, Any]:
    """Master volume on stock Windows — SendKeys only, no COM bindings or extra pip packages.

    Windows moves the mixer exactly 2 % per ``Volume_Up``/``Volume_Down`` keypress, so to
    *set* an absolute level JARVIS unmutes, drives the slider to zero with 50 down keys and
    climbs back up with ``level // 2`` up keys: deterministic, reversible, no admin rights.
    """
    if level is not None and int(level) < 0:
        level = None            # the Track-1 "not requested" sentinel
    if mute is None:
        mute = False
    if level is None and not delta and not mute:
        return {"ok": False, "message": "Give me a level (0-100), up/down, or mute."}

    if not config.is_windows():
        return _unix_volume(level, delta, bool(mute))

    sendkeys = "Add-Type -AssemblyName System.Windows.Forms | Out-Null;$s=[System.Windows.Forms.SendKeys];"
    run = ["powershell", "-NoProfile", "-NonInteractive", "-Command"]

    if mute:
        script = sendkeys + "$s::SendKeys('{VOLUME_MUTE}')"
        success = "Toggled the master mute."
    elif level is not None:
        target = max(0, min(100, int(level)))
        up_keys = (target + 1) // 2
        script = (
            sendkeys
            + "$s::SendKeys('{VOLUME_MUTE}');Start-Sleep -Milliseconds 60;"
            + "$s::SendKeys('{VOLUME_DOWN}');Start-Sleep -Milliseconds 40;"
            + "$s::SendKeys('{VOLUME_MUTE}');Start-Sleep -Milliseconds 60;"
            + "for($i=0;$i -lt 50;$i++){$s::SendKeys('{VOLUME_DOWN}');Start-Sleep -Milliseconds 8};"
            + "for($i=0;$i -lt %d;$i++){$s::SendKeys('{VOLUME_UP}');Start-Sleep -Milliseconds 8};" % up_keys
            + "[Console]::WriteLine('master volume %d')" % target
        )
        success = f"Volume set to {target} percent."
    else:
        steps = max(-50, min(50, int(delta or 1)))
        key = "{VOLUME_UP}" if steps > 0 else "{VOLUME_DOWN}"
        # Explicit repeats (not SendKeys' "*N" syntax) so the mixer keeps up.
        script = sendkeys + "".join(f"$s::SendKeys('{key}');Start-Sleep -Milliseconds 25;" for _ in range(abs(steps) or 1))
        success = f"{'Raised' if steps > 0 else 'Lowered'} the volume by {abs(steps)} steps."

    code, out, err = _run(run + [script], timeout=30)
    if code == 0:
        return {"ok": True, "message": success, "level": level, "delta": delta, "muted": bool(mute)}
    return {"ok": False, "message": f"Volume control failed: {_clip(err or out, 160)}"}


def _unix_volume(level: Optional[int], delta: Optional[int], mute: bool) -> Dict[str, Any]:
    """Linux/macOS path so the same tool is testable off Windows."""
    if shutil.which("pactl"):
        base = ["pactl"]
        sink = "0"
    elif shutil.which("amixer"):
        base = ["amixer", "-M"]
        sink = "sset,Master"
    else:
        return {"ok": False, "message": "No mixer tool (pactl/amixer) available on this system."}
    if mute:
        args = (["set-sink-mute", sink, "toggle"] if base[0] == "pactl" else ["sset", "Master", "toggle"])
    elif level is not None:
        value = max(0, min(100, int(level)))
        args = (["set-sink-volume", sink, f"{value}%"] if base[0] == "pactl" else ["sset", "Master", f"{value}%"])
    else:
        steps = int(delta or 5)
        op = "+" if steps > 0 else "-"
        args = (
            ["set-sink-volume", sink, f"{op}{abs(steps)}%"]
            if base[0] == "pactl"
            else ["sset", "Master", f"{abs(steps)}%{op}"]
        )
    code, out, err = _run(base + args, timeout=15)
    if code == 0:
        return {"ok": True, "message": "Volume updated.", "output": _clip(out, 200)}
    return {"ok": False, "message": f"Volume control failed: {_clip(err or out, 160)}"}


def take_screenshot() -> Dict[str, Any]:
    """Grab the desktop into ``data/screenshots`` and return the path."""
    out_dir = SETTINGS.cache_path / "screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    target = out_dir / f"jarvis-{stamp}.png"
    if config.is_windows():
        script = (
            "Add-Type -AssemblyName System.Windows.Forms,System.Drawing;"
            "$b=[System.Windows.Forms.SystemInformation]::VirtualScreen;"
            "$bmp=New-Object System.Drawing.Bitmap($b.Width,$b.Height);"
            "$g=[System.Drawing.Graphics]::FromImage($bmp);"
            "$g.CopyFromScreen($b.Location,[System.Drawing.Point]::Empty,$b.Size);"
            "$bmp.Save('%s',[System.Drawing.Imaging.ImageFormat]::Png);"
            "$g.Dispose();$bmp.Dispose()" % str(target)
        )
        code, out, err = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], timeout=25)
    else:
        tool = shutil.which("gnome-screenshot") or shutil.which("spectacle") or shutil.which("scrot")
        if not tool:
            return {"ok": False, "message": "No screenshot utility found on this system."}
        code, out, err = _run([tool, "-f", str(target)] if "scrot" in tool or "gnome" in tool else [tool, "-b", "-n", str(target)], timeout=25)
    if code == 0 and target.is_file():
        return {"ok": True, "path": str(target), "message": f"Screenshot saved to {target.name}.", "kb": target.stat().st_size // 1024}
    return {"ok": False, "message": f"Screenshot failed: {(err or out or 'unknown error')[:160]}"}


def system_power(action: str = "lock") -> Dict[str, Any]:
    """lock | sleep | shutdown | restart | cancel-shutdown (each needs confirmation in .env-free usage)."""
    action = (action or "").lower().strip()
    if action in {"lock", "logoff"}:
        ok = True
        if config.is_windows():
            ok = _run(["rundll32.exe", "user32.dll,LockWorkStation"], timeout=10)[0] == 0
        else:
            ok = any(_run([t, "lock"], timeout=10)[0] == 0 for t in ("loginctl", "gnome-screensaver-command", "xdg-screensaver"))
        return {"ok": ok, "message": "Workstation locked." if ok else "I could not lock the session."}
    commands = {
        "sleep": (["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"], ["systemctl", "suspend"]),
        "shutdown": (["shutdown", "/s", "/t", "30"], ["shutdown", "-h", "now"]),
        "restart": (["shutdown", "/r", "/t", "30"], ["shutdown", "-r", "now"]),
        "cancel-shutdown": (["shutdown", "/a"], ["shutdown", "-c"]),
    }
    if action not in commands:
        return {"ok": False, "message": f"I do not know the power action “{action}”. I support lock, sleep, shutdown, restart."}
    win_cmd, posix_cmd = commands[action]
    code, out, err = _run(win_cmd if config.is_windows() else posix_cmd, timeout=15)
    verb = {"shutdown": "Shutting down in 30 seconds — say “cancel shutdown”.", "restart": "Restarting in 30 seconds.",
            "sleep": "Sleeping.", "cancel-shutdown": "Cancelled the pending shutdown."}[action]
    return {"ok": code == 0, "message": verb if code == 0 else f"{verb} failed: {err[:120]}"}


# ---------------------------------------------------------------------------
# Public tool registry -- consumed by router.py to build the LLM's JSON schemas
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Web services ("open google" means google.com, not the browser)
# ---------------------------------------------------------------------------

#: Names that are *programs* first, web pages second. Anything below wins over
#: :data:`SITE_ALIASES`, so "open chrome" still starts Chrome.
APP_PREFERRED: set = {
    "chrome", "google chrome", "browser", "web browser", "edge", "microsoft edge",
    "firefox", "steam", "discord", "spotify", "eden", "notepad", "calculator",
    "code", "vs code", "vscode", "terminal", "powershell", "explorer", "obs",
    "paint", "settings", "task manager", "taskmgr", "whatsapp",
}

#: Web services the user can name directly. The value is what to open.
SITE_ALIASES: Dict[str, str] = {
    "google": "https://www.google.com",
    "google search": "https://www.google.com",
    "youtube": "https://www.youtube.com",
    "yt": "https://www.youtube.com",
    "youtube music": "https://music.youtube.com",
    "gmail": "https://mail.google.com",
    "google drive": "https://drive.google.com",
    "drive": "https://drive.google.com",
    "google docs": "https://docs.google.com",
    "docs": "https://docs.google.com",
    "google calendar": "https://calendar.google.com",
    "google keep": "https://keep.google.com",
    "google translate": "https://translate.google.com",
    "maps": "https://maps.google.com",
    "google maps": "https://maps.google.com",
    "github": "https://github.com",
    "gitlab": "https://gitlab.com",
    "stackoverflow": "https://stackoverflow.com",
    "stack overflow": "https://stackoverflow.com",
    "reddit": "https://www.reddit.com",
    "x": "https://x.com",
    "twitter": "https://x.com",
    "instagram": "https://www.instagram.com",
    "facebook": "https://www.facebook.com",
    "linkedin": "https://www.linkedin.com",
    "wikipedia": "https://www.wikipedia.org",
    "chatgpt": "https://chatgpt.com",
    "claude": "https://claude.ai",
    "gemini": "https://gemini.google.com",
    "perplexity": "https://www.perplexity.ai",
    "netflix": "https://www.netflix.com",
    "prime video": "https://www.primevideo.com",
    "jio cinema": "https://www.jiocinema.com",
    "hotstar": "https://www.hotstar.com",
    "twitch": "https://www.twitch.tv",
    "soundcloud": "https://soundcloud.com",
    "bandcamp": "https://bandcamp.com",
    "spotify web": "https://open.spotify.com",
    "spotify": "https://open.spotify.com",
    "amazon": "https://www.amazon.in",
    "flipkart": "https://www.flipkart.com",
    "meesho": "https://www.meesho.com",
    "mdn": "https://developer.mozilla.org",
    "caniuse": "https://caniuse.com",
    "pypi": "https://pypi.org",
    "npm": "https://www.npmjs.com",
    "arxiv": "https://arxiv.org",
    "google scholar": "https://scholar.google.com",
    "kaggle": "https://www.kaggle.com",
    "leetcode": "https://leetcode.com",
    "hackerrank": "https://www.hackerrank.com",
    "imdb": "https://www.imdb.com",
    "letterboxd": "https://letterboxd.com",
    "steam community": "https://steamcommunity.com",
    "espncricinfo": "https://www.espncricinfo.com",
    "cricbuzz": "https://www.cricbuzz.com",
    "fpl": "https://fantasy.premierleague.com",
    "notion": "https://www.notion.so",
    "figma": "https://www.figma.com",
    "web whatsapp": "https://web.whatsapp.com",
    "whatsapp web": "https://web.whatsapp.com",
    "web telegram": "https://web.telegram.org",
    "outlook": "https://outlook.live.com",
    "protonmail": "https://mail.proton.me",
    "openrouter": "https://openrouter.ai",
    "groq console": "https://console.groq.com",
}

#: Where a "search <thing> on <site>" should land. ``%s`` receives the query.
#: Sites without an entry fall back to ``site:``-scoped DuckDuckGo results.
SITE_SEARCH_URLS: Dict[str, str] = {
    "google": "https://www.google.com/search?q=%s",
    "google search": "https://www.google.com/search?q=%s",
    "youtube": "https://www.youtube.com/results?search_query=%s",
    "yt": "https://www.youtube.com/results?search_query=%s",
    "youtube music": "https://music.youtube.com/search?q=%s",
    "gmail": "https://mail.google.com/mail/u/0/#search/%s",
    "maps": "https://www.google.com/maps/search/%s",
    "google maps": "https://www.google.com/maps/search/%s",
    "github": "https://github.com/search?q=%s",
    "gitlab": "https://gitlab.com/search?search=%s",
    "stackoverflow": "https://stackoverflow.com/search?q=%s",
    "stack overflow": "https://stackoverflow.com/search?q=%s",
    "reddit": "https://www.reddit.com/search/?q=%s",
    "x": "https://x.com/search?q=%s",
    "twitter": "https://x.com/search?q=%s",
    "instagram": "https://www.instagram.com/explore/search/keyword/?q=%s",
    "wikipedia": "https://en.wikipedia.org/w/index.php?search=%s",
    "netflix": "https://www.netflix.com/search?q=%s",
    "prime video": "https://www.primevideo.com/search/ref=atv_nb_sr?phrase=%s",
    "twitch": "https://www.twitch.tv/search?term=%s",
    "soundcloud": "https://soundcloud.com/search?q=%s",
    "bandcamp": "https://bandcamp.com/search?q=%s",
    "spotify web": "https://open.spotify.com/search/%s",
    "spotify": "https://open.spotify.com/search/%s",
    "amazon": "https://www.amazon.in/s?k=%s",
    "flipkart": "https://www.flipkart.com/search?q=%s",
    "meesho": "https://www.meesho.com/search?q=%s",
    "mdn": "https://developer.mozilla.org/en-US/search?q=%s",
    "pypi": "https://pypi.org/search/?q=%s",
    "npm": "https://www.npmjs.com/search?q=%s",
    "arxiv": "https://arxiv.org/abs/%s",
    "google scholar": "https://scholar.google.com/scholar?q=%s",
    "kaggle": "https://www.kaggle.com/search?q=%s",
    "leetcode": "https://leetcode.com/problemset/all/?search=%s",
    "hackerrank": "https://www.hackerrank.com/search?dim=demos&keywords=%s",
    "imdb": "https://www.imdb.com/find/?q=%s",
    "letterboxd": "https://letterboxd.com/search/%s/",
    "steam community": "https://steamcommunity.com/search/?text=%s",
    "notion": "https://www.notion.so",
    "espncricinfo": "https://www.espncricinfo.com/search?query=%s",
    "fpl": "https://fantasy.premierleague.com",
}

#: Domain-suffix -> canonical site key, so "on google.com" / "on youtube.com" work.
_SITE_DOMAINS: Dict[str, str] = {
    "google.com": "google", "youtube.com": "youtube", "youtu.be": "youtube",
    "github.com": "github", "reddit.com": "reddit", "amazon.in": "amazon",
    "amazon.com": "amazon", "flipkart.com": "flipkart", "wikipedia.org": "wikipedia",
    "stackoverflow.com": "stackoverflow", "twitch.tv": "twitch", "netflix.com": "netflix",
    "x.com": "x", "twitter.com": "twitter", "instagram.com": "instagram",
    "soundcloud.com": "soundcloud", "imdb.com": "imdb", "leetcode.com": "leetcode",
    "kaggle.com": "kaggle", "npmjs.com": "npm", "pypi.org": "pypi", "arxiv.org": "arxiv",
    "mail.google.com": "gmail", "maps.google.com": "maps", "open.spotify.com": "spotify web",
    "open.spotify.com": "spotify web", "spotify.com": "spotify web", "mdn.io": "mdn", "developer.mozilla.org": "mdn",
    "stackexchange.com": "stackoverflow", "telegram.org": "web telegram",
    "whatsapp.com": "web whatsapp", "linkedin.com": "linkedin", "notion.so": "notion",
}


def _site_key(name: str, allow_apps: bool = False) -> str:
    """Normalise a spoken site ("YouTube", "youtube.com", "https://youtube.com")."""
    raw = (name or "").strip().lower().rstrip(".,!?;")
    raw = re.sub(r"^(?:the|on|in|at|www\.)+", "", raw)
    raw = re.sub(r"^https?://", "", raw)
    raw = re.sub(r"^(?:www|music|mail|maps|docs|drive)\.", "", raw)
    raw = raw.split("/")[0].strip()
    if raw in APP_PREFERRED and not allow_apps:
        return ""           # "google chrome" is the browser, not google.com
    if raw in SITE_ALIASES or raw in SITE_SEARCH_URLS:
        return raw
    if raw in _SITE_DOMAINS:
        return _SITE_DOMAINS[raw]
    # "youtube search", "on the youtube app", "google maps" and friends.
    for key in sorted(SITE_ALIASES, key=len, reverse=True):
        if re.search(r"\b" + re.escape(key) + r"\b", raw):
            return key
    return ""


def is_known_site(name: str, allow_apps: bool = False) -> bool:
    """True when the words name a web service rather than an installed program.

    ``allow_apps`` is for ``on <site>`` slots, where "play lofi on spotify" clearly
    means Spotify's web search even though Spotify is also an installed app.
    """
    return bool(_site_key(name, allow_apps=allow_apps))


def site_url(name: str, allow_apps: bool = False) -> str:
    key = _site_key(name, allow_apps=allow_apps)
    if not key:
        return ""
    if key in SITE_ALIASES:
        return SITE_ALIASES[key]
    slug = re.sub(r"[^a-z0-9]+", "", key)
    return f"https://www.{slug}.com" if slug else ""


def site_search_url(name: str, query: str, allow_apps: bool = False) -> str:
    """Deep link straight into a site's own results page for ``query``."""
    key = _site_key(name, allow_apps=allow_apps)
    if not key or not query:
        return ""
    template = SITE_SEARCH_URLS.get(key) or ""
    if not template:
        for alias, url in SITE_SEARCH_URLS.items():
            if alias.endswith(key) or key.endswith(alias):
                template = url
                break
    if not template:
        return ""
    return template % _urlparse.quote_plus(query)


def search_on_site(site: str, query: str, open_browser: bool = True) -> Dict[str, Any]:
    """``search LM Arena on youtube`` / ``look X up on google.com``.

    Opens the site's own results page *and* runs a ``site:``-scoped DuckDuckGo
    search, so the browser shows the right thing and JARVIS can still speak an
    answer when the search engine is reachable.
    """
    query = " ".join((query or "").split())
    key = _site_key(site, allow_apps=True)
    if not query:
        return {"ok": False, "message": "Search for what? Tell me the words after \u201csearch\u201d."}
    if not key:
        # Unknown site name: treat it as a domain if it looks like one, else search the web.
        guess = site.strip()
        if re.fullmatch(r"[a-z0-9][a-z0-9-]*(\.[a-z]{2,})+(/.*)?", guess.lower()):
            url = f"https://{guess}"
            return {**_open_url(url), "message": f"Opening {guess} (I do not have a search URL for it).",
                    "query": query, "site": guess}
        found = web_search(f"{query} {site}", max_results=5)
        return {**found, "query": f"{query} {site}", "site": site,
                "message": f"I do not know {site}'s search page, so I searched the web for \u201c{query} {site}\u201d."}

    domain = (SITE_ALIASES.get(key, "").split("//")[-1] or "").split("/")[0]
    scoped = web_search(query, max_results=5, site=domain) if domain else {"ok": False, "results": []}
    url = site_search_url(key, query, allow_apps=True) or site_url(key, allow_apps=True)
    opened = _open_url(url) if (url and open_browser) else {"ok": False, "message": "browser opening disabled"}
    label = key.replace("-", " ").title()
    results = scoped.get("results") or []
    if results:
        headline = f"{label} results for \u201c{query}\u201d: {results[0].get('title', '')}"
    else:
        headline = (f"Opened {label} search for \u201c{query}\u201d in the browser."
                    if opened.get("ok") else
                    f"I could not open {label}: {opened.get('message', 'no search page on file')}")
    return {
        "ok": bool(results) or bool(opened.get("ok")),
        "message": headline,
        "site": key,
        "query": query,
        "search_url": url,
        "opened": bool(opened.get("ok")),
        "results": results[:5],
    }


def llm_status() -> Dict[str, Any]:
    """Report which AI providers are configured, live, or cooling down."""
    try:
        import llm_providers
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": f"provider layer unavailable: {exc}"}
    status = llm_providers.POOL.status()
    live = status.get("available") or []
    cooling = [p for p in status.get("providers", []) if p.get("cooling")]
    ready = status.get("configured") or []
    if not ready:
        lines = ["No AI provider key is set, so I am running on my regex rules and the offline planner."]
    else:
        lines = [f"AI providers ready: {', '.join(live) if live else 'none right now'}"
                 + (f"; cooling: {', '.join(str(p['key']) + ' (' + str(max(1, int(p.get('cool_left_s', 0)) // 60)) + 'm)' for p in cooling)}" if cooling else "")]
        if status.get("last_error"):
            lines.append(f"Last error: {status['last_error']}")
    return {"ok": True, "message": " ".join(lines), "llm": status}


# --------------------------------------------------------------------------- calendar
def _calendar_module() -> Any:
    """Import the agenda (calendar) module once (stdlib-only, no heavy deps)."""
    import agenda as _calendar

    return _calendar


def calendar_agenda(days: str = "") -> Dict[str, Any]:
    """What is on the user's calendar for the next N days (default 1 = today)."""
    try:
        cal = _calendar_module()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": f"Calendar module unavailable: {exc}"}
    try:
        n = float(days) if (days or "").strip() else 1.0
    except ValueError:
        n = 1.0
    events = cal.agenda(days=max(0.5, min(n, 90.0)))
    if not events:
        _, errors = cal.load_events()
        hint = f" ({errors[0]})" if errors and "configured" in errors[0] else ""
        return {"ok": True, "message": f"Nothing is scheduled for the next {n:g} day(s).{hint}",
                "events": [], "count": 0}
    lines = [f"{e.when}: {e.summary}" + (f" @ {e.location}" if e.location else "")
             for e in events[:12]]
    return {"ok": True, "message": " · ".join(lines),
            "events": [e.as_dict() for e in events[:12]], "count": len(events)}


def calendar_next() -> Dict[str, Any]:
    """The very next calendar event (class, meeting, shift…) whenever it is."""
    try:
        cal = _calendar_module()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": f"Calendar module unavailable: {exc}"}
    event = cal.next_event()
    if event is None:
        return {"ok": True, "message": "There is nothing coming up on your calendar.",
                "event": None}
    extra = f" at {event.location}" if event.location else ""
    return {"ok": True,
            "message": f"Next: {event.summary} — {event.when}{extra}.",
            "event": event.as_dict()}


# --------------------------------------------------------------------------- todo list
def _todo_path() -> Path:
    raw = (SETTINGS.todo_file or "notes/todo.md").strip()
    path = Path(raw)
    if not path.is_absolute():
        path = config.ROOT / path
    return path


def _todo_read() -> List[str]:
    path = _todo_path()
    if not path.is_file():
        return []
    return [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()]


def todo(action: str = "list", text: str = "") -> Dict[str, Any]:
    """The everyday to-do list: add items, tick them off, list, clear.  Persisted as Markdown."""
    action = (action or "list").lower()
    path = _todo_path()
    items = _todo_read()
    try:
        if action in {"list", "show", "status"}:
            if not items:
                return {"ok": True, "message": "Your to-do list is empty.",
                        "items": [], "count": 0}
            return {"ok": True, "message": "To-do: " + " | ".join(items[:20]),
                    "items": items, "count": len(items)}
        if action in {"add", "new", "append"}:
            item = (text or "").strip()
            if not item:
                return {"ok": False, "message": "Tell me what to add, e.g. 'add to my todo: submit the CBS assignment'."}
            for existing in items:
                if existing.lower() == item.lower():
                    return {"ok": True, "message": f"“{item}” is already on the list.", "items": items}
            items.append(item)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(items) + "\n", encoding="utf-8")
            return {"ok": True, "message": f"Added “{item}” to your to-do list. You have {len(items)} item(s).",
                    "items": items, "count": len(items)}
        if action in {"done", "remove", "delete", "complete"}:
            needle = (text or "").strip().lower()
            if not needle:
                return {"ok": False, "message": "Which item? Say 'tick off <the item>'."}
            kept = [it for it in items if it.lower() != needle]
            if len(kept) == len(items):
                return {"ok": False, "message": f"I couldn't find “{text}” on the list.", "items": items}
            path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
            return {"ok": True, "message": f"Done. “{text}” removed; {len(kept)} item(s) left.",
                    "items": kept, "count": len(kept)}
        if action == "clear":
            path.write_text("", encoding="utf-8")
            return {"ok": True, "message": "To-do list cleared.", "items": [], "count": 0}
    except OSError as exc:
        return {"ok": False, "message": f"Could not update the to-do list: {exc}"}
    return {"ok": False, "message": f"Unknown todo action “{action}” (list/add/done/clear)."}


def daily_brief() -> Dict[str, Any]:
    """The morning brief: today's date, calendar, to-do list, and pending reminders."""
    import datetime as _dt

    today_date = _dt.date.today().strftime("%A, %d %B %Y")
    parts = [f"Here is your brief for {today_date}."]

    try:
        cal = _calendar_module()
        todays = cal.agenda(days=1.0)
        if todays:
            parts.append("Calendar: " + " · ".join(
                f"{e.when}: {e.summary}" + (f" @ {e.location}" if e.location else "") for e in todays[:8]))
        else:
            parts.append("Calendar: nothing scheduled today.")
    except Exception as exc:  # noqa: BLE001
        parts.append(f"Calendar: unavailable ({exc}).")

    items = _todo_read()
    parts.append("To-do: " + (" | ".join(items[:12]) if items else "nothing on the list."))

    try:
        if _reminders is not None:
            rows = _reminders.BOARD.list()
            if rows:
                due = "; ".join(f"{str(r.get('text', ''))} at {r.get('due_iso', '?')}" for r in rows[:5])
                parts.append(f"Reminders: {due}.")
    except Exception:  # noqa: BLE001
        pass

    return {"ok": True, "message": " ".join(parts),
            "date": today_date, "todo": items, "events": todays if 'todays' in dir() else []}


# --------------------------------------------------------------------------- email
def _open_mailto(mailto: str) -> Dict[str, Any]:
    if config.is_windows():
        try:
            import winops as _w

            out = _w.shell_execute(mailto)
            if out.get("ok"):
                return {"ok": True, "message": "Opened your mail app with the draft."}
        except Exception:  # noqa: BLE001
            pass
    try:
        import webbrowser

        if webbrowser.open(mailto):
            return {"ok": True, "message": "Opened your mail app with the draft."}
    except Exception:  # noqa: BLE001
        pass
    return {"ok": False, "message": f"Your mail app could not be opened. Draft URI: {mailto}"}


def _save_draft(to: str, subject: str, body: str) -> Path:
    folder = config.ROOT / "notes" / "drafts"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = folder / f"email-{stamp}.md"
    text = (f"# Email draft ({stamp})\n\n"
            f"**To:** {to or '(not set)'}\n\n**Subject:** {subject or '(not set)'}\n\n{body or ''}\n")
    path.write_text(text, encoding="utf-8")
    return path


def draft_email(to: str = "", subject: str = "", body: str = "") -> Dict[str, Any]:
    """Write an email draft AND open the mail app with it prefilled (no credentials needed)."""
    to = (to or SETTINGS.email_default_to or "").strip()
    subject = (subject or "").strip()
    body = (body or "").strip()
    if not subject and not body:
        return {"ok": False, "message": "What should the email say? Give me a subject and a body."}
    try:
        path = _save_draft(to, subject, body)
    except OSError as exc:
        return {"ok": False, "message": f"Could not save the draft: {exc}"}
    mailto = ("mailto:" + _urlparse.quote(to or "") + "?subject=" + _urlparse.quote(subject)
              + "&body=" + _urlparse.quote(body))
    opened = _open_mailto(mailto)
    return {"ok": True, "message": f"Draft saved to {path.name} — {opened['message']}",
            "draft": str(path), "to": to, "subject": subject, "opened": bool(opened.get("ok"))}


def send_email(to: str = "", subject: str = "", body: str = "") -> Dict[str, Any]:
    """Actually SEND an email via the SMTP server configured in .env (falls back to drafting)."""
    to = (to or SETTINGS.email_default_to or "").strip()
    subject = (subject or "").strip()
    body = (body or "").strip()
    if not to or "@" not in to:
        return {"ok": False, "message": "I need a real recipient address to send to."}
    if not (SETTINGS.smtp_host and SETTINGS.smtp_user and SETTINGS.smtp_password):
        return {"ok": False,
                "message": ("No SMTP server is configured, so I can't send directly. "
                            "I'll draft it instead — set SMTP_HOST/SMTP_USER/SMTP_PASSWORD in .env "
                            "to send for real."),
                "hint": "draft-created" if (subject or body) else ""}
    try:
        import smtplib
        from email.message import EmailMessage

        msg = EmailMessage()
        msg["From"] = SETTINGS.email_from or SETTINGS.smtp_user
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body or "")
        with smtplib.SMTP(SETTINGS.smtp_host, int(SETTINGS.smtp_port), timeout=20) as server:
            server.starttls()
            server.login(SETTINGS.smtp_user, SETTINGS.smtp_password)
            server.send_message(msg)
        return {"ok": True, "message": f"Email sent to {to}.", "to": to, "subject": subject}
    except Exception as exc:  # noqa: BLE001 - auth/network must not crash the assistant
        return {"ok": False, "message": f"Email failed to send: {exc}"}


TOOL_FUNCTIONS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "launch_app": launch_app,
    "close_app": close_app,
    "open_website": open_website,
    "play_youtube": play_youtube,
    "web_search": web_search,
    "fetch_sports_stats": fetch_sports_stats,
    "read_notes": read_notes,
    "write_note": write_note,
    "system_report": system_report,
    "get_time": get_time,
    "set_volume": set_volume,
    "take_screenshot": take_screenshot,
    "system_power": system_power,
    "search_on_site": search_on_site,
    "llm_status": llm_status,
    "focus_app": focus_app,
    "list_apps": list_apps,
    "windows_on_screen": windows_on_screen,
    "manage_files": manage_files,
    "control_desktop": control_desktop,
    "read_screen": read_screen,
    "set_reminder": set_reminder,
    "listening": listening,
    "calendar_agenda": calendar_agenda,
    "calendar_next": calendar_next,
    "todo": todo,
    "daily_brief": daily_brief,
    "draft_email": draft_email,
    "send_email": send_email,
}


def execute_tool(name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Single entry point used by the agent loop, Track 1 and the Discord bridge."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {"ok": False, "message": f"Unknown tool “{name}”.", "available": sorted(TOOL_FUNCTIONS)}
    args = {k: v for k, v in (arguments or {}).items() if v not in (None, "", [])}
    try:
        result = fn(**args)
        if not isinstance(result, dict):
            result = {"ok": True, "message": str(result)}
        result.setdefault("ok", True)
        result.setdefault("message", "")
        return result
    except TypeError as exc:
        return {"ok": False, "message": f"Tool “{name}” was called with bad arguments: {exc}", "arguments": args}
    except Exception as exc:  # noqa: BLE001 - never crash the assistant
        log.exception("tool %s crashed", name)
        return {"ok": False, "message": f"Tool “{name}” failed: {type(exc).__name__}: {exc}", "arguments": args}


def available_apps() -> Iterable[str]:
    return sorted(set(APP_CATALOG) | set(_custom_apps()))


__all__ = [
    "APP_CATALOG",
    "TOOL_FUNCTIONS",
    "launch_app",
    "close_app",
    "focus_app",
    "list_apps",
    "windows_on_screen",
    "manage_files",
    "control_desktop",
    "read_screen",
    "set_reminder",
    "listening",
    "list_launchable_apps",
    "is_known_app",
    "open_website",
    "play_youtube",
    "web_search",
    "search_news",
    "fetch_sports_stats",
    "resolve_team_id",
    "read_notes",
    "write_note",
    "system_report",
    "gpu_stats",
    "get_time",
    "set_volume",
    "take_screenshot",
    "system_power",
    "execute_tool",
    "available_apps",
]
