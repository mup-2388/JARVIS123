"""Windows primitives: everything JARVIS does *to* the desktop, with no third-party deps.

Why this module exists
----------------------
The app-launcher used to shell out ``subprocess.run(["ms-settings:"])`` for protocol URIs,
which Windows rejects (``CreateProcess`` cannot launch a URI) - so "open Settings" fell
through to ``start "" settings`` and silently failed.  Every protocol handler, ``.lnk``,
``.appref-ms`` and shell folder in JARVIS now goes through :func:`shell_execute`, which is
``ShellExecuteW`` - the same call Explorer's address bar uses - so ``ms-settings:``,
``msteams:``, ``steam://``, ``shell:AppsFolder\\...`` and ``http(s)://`` all work.

Everything here is guarded two ways: :data:`AVAILABLE` is False off Windows, and each
function returns a ``{ok, message, ...}`` dict instead of raising, so one missing Win32
call degrades into a spoken "I can't do that here" rather than a traceback.  Nothing needs
pywin32; ctypes, PowerShell and ``taskkill`` are enough.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import config
from config import SETTINGS, get_logger

log = get_logger("winops")

IS_WINDOWS = sys.platform.startswith("win")
AVAILABLE = IS_WINDOWS

# ---------------------------------------------------------------------------- handles / msgs
_SW_RESTORE, _SW_MINIMIZE, _SW_MAXIMIZE, _SW_SHOW, _SW_HIDE, _SW_SHOWDEFAULT = 9, 6, 3, 5, 0, 10
_SWP_NOSIZE, _SWP_NOMOVE, _SWP_NOACTIVATE, _SWP_SHOWWINDOW, _SWP_FRAMECHANGED = 0x1, 0x2, 0x10, 0x40, 0x20
#: 0x20 is SWP_FRAMECHANGED, not SWP_NOACTIVATE - the old value asked Windows to re-frame the
#: window and let it take focus, which is exactly what a desktop bar must never do.

#: SHFileOperation fFlags (WinUser.h).  FOF_ALLOWUNDO is the bit that puts an item in the Recycle
#: Bin instead of destroying it, so it lives here as a named constant with a test on it rather
#: than as a literal in a call; FOF_NOERRORUI stops Windows raising a modal dialog mid-sentence.
FOF_SILENT, FOF_NOCONFIRMATION, FOF_NOERRORUI, FOF_ALLOWUNDO = 0x0004, 0x0010, 0x0200, 0x0040
RECYCLE_FLAGS = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI

#: MOUSEEVENTF down/up pairs.  A double click has no flag of its own - it is a second press and
#: release - so the pair is combined, never ORed with another button's bit.
MOUSE_BUTTONS = {"left": (0x0002, 0x0004), "right": (0x0008, 0x0010), "middle": (0x0020, 0x0040)}


def _double_click_flags(down: int, up: int) -> int:
    """dwFlags for the second press of a double click (never another button's bit)."""
    return down | up


def _char_event(char: str) -> Tuple[int, int]:
    """``(wVk, wScan)`` for typing one character through ``KEYEVENTF_UNICODE``.

    ``VkKeyW`` used to be consulted here, which lost the shift state - every letter arrived upper
    case - and put a sixteen-bit value into an eight-bit field.  Sending the character itself
    keeps case, punctuation and accents; the clipboard route is still tried first, because games
    and DirectInput apps ignore synthetic Unicode.
    """
    return 0, ord(char)
_GWL_EXSTYLE, _WS_EX_TOOLWINDOW, _WS_EX_NOACTIVATE, _WS_EX_TOPMOST, _WS_EX_APPWINDOW = -20, 0x80, 0x08000000, 0x8, 0x40000
_HWND_TOPMOST, _HWND_NOTOPMOST = -1, -2

#: Every Win32 entry point JARVIS calls, paired with the module that actually exports it and the
#: types that keep a 64-bit handle from arriving as a truncated ``c_int``.
#:
#: ctypes answers a name looked up on the wrong DLL with ``AttributeError: function 'X' not found``
#: *at the call site*, and this file used to ask user32 for ShellExecuteW (which lives in shell32)
#: and for BitBlt (gdi32) - so "open Settings" and "look at my screen" both died with a ctypes
#: message instead of an app.  Worse, the three prototype lines that would have fixed the
#: truncation sat in the same ``try`` as the bad lookup, so the first raised and none were ever
#: configured.  Names are therefore declared here, resolved once at import, and cross-checked by
#: TestWin32Bindings, which reads this file and the SDK ownership table back to back.
_PROTOTYPES = (
    # (module, function, restype, argtypes) - None means "leave ctypes' default alone".
    ("shell32", "ShellExecuteW", ctypes.c_void_p,
     [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_wchar_p,
      ctypes.c_int]),
    ("shell32", "SHFileOperationW", ctypes.c_int, [ctypes.c_void_p]),
    ("shell32", "SHEmptyRecycleBinW", ctypes.c_int, [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]),
    # gdi32 - the screen grab, with no dependencies
    ("gdi32", "CreateCompatibleDC", ctypes.c_void_p, [ctypes.c_void_p]),
    ("gdi32", "CreateCompatibleBitmap", ctypes.c_void_p,
     [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]),
    ("gdi32", "SelectObject", ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p]),
    ("gdi32", "BitBlt", ctypes.c_bool,
     [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
      ctypes.c_int, ctypes.c_int, ctypes.c_ulong]),
    ("gdi32", "GetDIBits", ctypes.c_int,
     [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
      ctypes.c_void_p, ctypes.c_uint]),
    ("gdi32", "DeleteDC", ctypes.c_bool, [ctypes.c_void_p]),
    ("gdi32", "DeleteObject", ctypes.c_bool, [ctypes.c_void_p]),
    # user32 - windows
    ("user32", "GetForegroundWindow", ctypes.c_void_p, None),
    ("user32", "SetForegroundWindow", ctypes.c_bool, [ctypes.c_void_p]),
    ("user32", "BringWindowToTop", ctypes.c_bool, [ctypes.c_void_p]),
    ("user32", "ShowWindow", ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int]),
    ("user32", "IsWindowVisible", ctypes.c_bool, [ctypes.c_void_p]),
    ("user32", "IsIconic", ctypes.c_bool, [ctypes.c_void_p]),
    ("user32", "GetWindowTextLengthW", ctypes.c_int, [ctypes.c_void_p]),
    # The calls below pass byref() results and raw buffers rather than plain ints and strings.
    # They are typed too, because c_void_p is documented to take a pointer-sized value and was
    # measured to accept byref(obj), create_string_buffer(n) and create_unicode_buffer(n) - so the
    # conversion is explicit instead of left to ctypes' defaults, which is the whole reason an
    # HWND ever arrived as a truncated c_int in the first place.
    ("user32", "GetWindowThreadProcessId", ctypes.c_ulong, [ctypes.c_void_p, ctypes.c_void_p]),
    ("user32", "GetWindowRect", ctypes.c_bool, [ctypes.c_void_p, ctypes.c_void_p]),
    ("user32", "GetWindowTextW", ctypes.c_int, [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]),
    ("user32", "GetCursorPos", ctypes.c_bool, [ctypes.c_void_p]),
    ("user32", "GetSystemMetrics", ctypes.c_int, [ctypes.c_int]),
    ("user32", "GetDC", ctypes.c_void_p, [ctypes.c_void_p]),
    ("user32", "ReleaseDC", ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p]),
    ("user32", "SetWindowPos", ctypes.c_bool,
     [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
      ctypes.c_uint]),
    ("user32", "GetWindowLongW", ctypes.c_long, [ctypes.c_void_p, ctypes.c_int]),
    ("user32", "SetWindowLongW", ctypes.c_long, [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]),
    # user32 - clipboard and input
    ("user32", "OpenClipboard", ctypes.c_bool, [ctypes.c_void_p]),
    ("user32", "CloseClipboard", ctypes.c_bool, []),
    ("user32", "EmptyClipboard", ctypes.c_bool, []),
    ("user32", "IsClipboardFormatAvailable", ctypes.c_bool, [ctypes.c_uint]),
    ("user32", "GetClipboardData", ctypes.c_void_p, [ctypes.c_uint]),
    ("user32", "SetClipboardData", ctypes.c_void_p, [ctypes.c_uint, ctypes.c_void_p]),
    ("user32", "SendInput", ctypes.c_uint, [ctypes.c_uint, ctypes.c_void_p, ctypes.c_int]),
    ("user32", "SetProcessDPIAware", ctypes.c_bool, []),
    ("user32", "LockWorkStation", ctypes.c_bool, []),
    ("user32", "keybd_event", None, None),
    ("user32", "GetAsyncKeyState", ctypes.c_short, [ctypes.c_int]),
    ("user32", "SetCursorPos", ctypes.c_bool, [ctypes.c_int, ctypes.c_int]),
    # kernel32
    ("kernel32", "Beep", ctypes.c_bool, [ctypes.c_uint, ctypes.c_uint]),
    ("kernel32", "GetSystemPowerStatus", ctypes.c_bool, [ctypes.c_void_p]),
    ("kernel32", "SetConsoleCtrlHandler", ctypes.c_bool, [ctypes.c_void_p, ctypes.c_bool]),
)

user32 = shell32 = gdi32 = kernel32 = None

#: What could not be resolved at import, in ``"module!Function"`` form.  Filled while the
#: prototypes are configured, read by ``win32_health()`` and printed by ``python main.py`` at
#: boot - a laptop should never have to discover that app launching is dead by trying it.
_WIN32_MISSING: List[str] = []

#: Which promise each entry point is load-bearing for, so a missing one can be explained.
_WIN32_FEATURES = {
    "ShellExecuteW": "opening apps, Settings pages, .lnk shortcuts and URLs",
    "SHFileOperationW": "the Recycle Bin (deletes fall back to my own backup folder)",
    "SendInput": "typing and key presses into other apps",
    "GetForegroundWindow": "knowing which window you are looking at",
    "SetForegroundWindow": "focusing a window",
    "SetWindowPos": "moving a window to another monitor",
    "BitBlt": "screen capture, so \"look at my screen\"",
    "GetDIBits": "screen capture, so \"look at my screen\"",
    "CreateCompatibleDC": "screen capture, so \"look at my screen\"",
    "CreateCompatibleBitmap": "screen capture, so \"look at my screen\"",
    "GetSystemPowerStatus": "the battery readout",
    "OpenClipboard": "clipboard read and write",
    "GetClipboardData": "clipboard read",
    "SetClipboardData": "pasting into another app",
    "Beep": "the confirmation beep (winsound covers this one anyway)",
}

if IS_WINDOWS:  # pragma: no cover - exercised on the target machine
    for _module in ("user32", "shell32", "gdi32", "kernel32"):
        try:
            globals()[_module] = getattr(ctypes.windll, _module)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            _WIN32_MISSING.append(_module)
            log.warning("Win32 module %s unavailable: %s", _module, exc)
    for _module, _name, _restype, _argtypes in _PROTOTYPES:
        _handle = globals().get(_module)
        if _handle is None:
            continue
        try:
            _proc = getattr(_handle, _name)
            if _restype is not None:
                _proc.restype = _restype
            if _argtypes is not None:
                _proc.argtypes = list(_argtypes)
        except Exception as exc:  # noqa: BLE001
            # Loud here, instead of "function not found" at the moment the user asks.
            _WIN32_MISSING.append(f"{_module}!{_name}")
            log.warning("Win32 entry point %s!%s unavailable: %s", _module, _name, exc)
    if kernel32 is not None:
        try:  # launched detached: a Ctrl+C or a closed console must not kill the assistant
            kernel32.SetConsoleCtrlHandler(0, 0)
        except Exception as exc:  # noqa: BLE001
            log.debug("SetConsoleCtrlHandler unavailable: %s", exc)

_lock = threading.RLock()
_cache: Dict[str, Any] = {}
CACHE_TTL = 300.0


def _result(ok: bool, message: str, **extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": bool(ok), "message": message}
    out.update(extra)
    return out


def _powershell(script: str, timeout: int = 20) -> Tuple[int, str]:
    """Run a PS one-liner, returning (exit, stdout).  Used for WinRT conveniences."""
    exe = shutil.which("powershell") or shutil.which("pwsh") or ""
    if not exe:
        return 127, "powershell not found"
    try:
        proc = subprocess.run([exe, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                               "-Command", script], capture_output=True, text=True, timeout=timeout,
                              encoding="utf-8", errors="replace")
        return proc.returncode, (proc.stdout or "").strip()
    except subprocess.TimeoutExpired:
        return 124, "powershell timed out"
    except Exception as exc:  # noqa: BLE001
        return 1, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------- launching
def shell_execute(target: str, args: str = "", working: str = "", verb: str = "open",
                  show: int = _SW_SHOWDEFAULT) -> Dict[str, Any]:
    """The one true way to open anything on Windows (URIs, .lnk, folders, apps, URLs)."""
    target = (target or "").strip()
    if not target:
        return _result(False, "Nothing was given to open.")
    if not IS_WINDOWS:
        opener = shutil.which("xdg-open") or shutil.which("open")
        if not opener:
            return _result(False, "No desktop opener is available on this platform.")
        cmd = [opener, target] + _split_args(args)
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
            return _result(True, f"Asked the desktop to open {Path(target).name or target}.", target=target)
        except Exception as exc:  # noqa: BLE001
            return _result(False, f"Could not open {target}: {exc}", target=target)
    names = {2: "file not found", 3: "path not found", 5: "access denied", 31: "no association",
             22: "destination needed", 32: "shared-doc error"}
    if shell32 is None:  # pragma: no cover - only when ctypes itself is crippled
        return _shell_fallback(target, args, "shell32.dll is not loaded")
    try:
        handle = shell32.ShellExecuteW(None, verb, target, args or None, working or None, show)  # type: ignore[union-attr]
    except AttributeError as exc:
        # "function 'ShellExecuteW' not found": a missing/wrong entry point, not a refusal, so
        # the shell's own launchers get the turn rather than the user getting a ctypes message.
        return _shell_fallback(target, args, f"ShellExecuteW could not be resolved ({exc})")
    except Exception as exc:  # noqa: BLE001
        return _shell_fallback(target, args, f"ShellExecuteW raised {exc!r}")
    ret = int(handle or 0)
    if ret > 32:  # HINSTANCE: anything above 32 is success
        return _result(True, f"Opened {target}", target=target, args=args, method="shellexecute")
    if ret in (2, 3, 5, 31):
        # "no association"/"access denied" on a .lnk or a URI is often just this entry point being
        # unavailable; explorer.exe reads the same association database the Start menu does.
        return {**_shell_fallback(target, args, f"ShellExecute said: {names.get(ret, ret)}"), "code": ret}
    return _result(False, f"Windows refused to open {target} ({names.get(ret, 'error code ' + str(ret))}).",
                   target=target, code=ret)


def _split_args(args: str) -> List[str]:
    """Turn one argument string into real argv entries.

    ``Popen([exe, "--folder C:\\My Docs"])`` quotes the whole string into a *single* argument, so
    every launch that carried more than one flag handed the app one mangled token - "open VS Code
    with this folder" asked it to open a file called ``--folder C:\My Docs``.  shlex splits it the
    way cmd would, and the surrounding quotes are then dropped because argv entries carry no
    quoting of their own.  (``shell_execute`` deliberately keeps one string: ShellExecute takes a
    command line, not an argv array.)
    """
    text = str(args or "").strip()
    if not text:
        return []
    try:
        parts = shlex.split(text, posix=False, comments=False)
    except ValueError:                     # unbalanced quote - "open it with 'foo
        parts = text.split()
    return [p.strip('"') if len(p) > 1 and p.startswith('"') and p.endswith('"') else p
            for p in parts if p]


def _shell_fallback(target: str, args: str, why: str) -> Dict[str, Any]:
    """Open something without ShellExecute, and say which route won.

    A voice assistant that answers "ShellExecute failed" is a dead end, so both of the shell's own
    launchers are tried: they read the same association database as the Start menu, and they work
    when the direct call was never available (a stripped ctypes, a blocked DLL, a policy that
    forbids ShellExecute but not explorer).  ``why`` is kept in the message because the fix for a
    policy block is different from the fix for a missing file, and the user should never have to
    guess which one they hit.
    """
    for cmd, method in ((["explorer.exe", target], "explorer"),
                        (["cmd", "/c", "start", "", target] + _split_args(args), "cmd start")):
        out = _run_detached(cmd, target)
        if out.get("ok"):
            return {**out, "method": method,
                    "message": f"Asked Windows to open {Path(target).name or target} via {method} "
                               f"({why})."}
    return _result(False, f"Could not open {target}: {why}, and neither explorer nor cmd start "
                         f"would run it.", target=target)


def _run_detached(cmd: Sequence[str], target: str = "") -> Dict[str, Any]:
    try:
        subprocess.Popen(list(cmd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL,
                         creationflags=(0x00000008 | 0x00000200) if IS_WINDOWS else 0, close_fds=True)
        return _result(True, f"Started {target or cmd[0]}", command=list(cmd))
    except Exception as exc:  # noqa: BLE001
        return _result(False, f"Could not start {target or cmd[0]}: {exc}", command=list(cmd))


def launch_exe(exe: str, args: str = "", cwd: str = "") -> Dict[str, Any]:
    """Start an .exe directly (faster and argument-safe), falling back to ShellExecute."""
    exe = (exe or "").strip().strip('"')
    if not exe:
        return _result(False, "No executable given.")
    expanded = os.path.expandvars(os.path.expanduser(exe))
    resolved = expanded if Path(expanded).is_file() else (shutil.which(expanded) or "")
    if not resolved:
        return shell_execute(expanded, args, cwd)
    argv = [resolved] + _split_args(args)
    if not IS_WINDOWS:
        try:
            subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL, cwd=cwd or None)
            return _result(True, f"Started {Path(resolved).stem}.", exe=resolved)
        except Exception as exc:  # noqa: BLE001
            return _result(False, f"{Path(resolved).name} failed to start: {exc}", exe=resolved)
    try:
        # CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS, so JARVIS dying never kills the app
        proc = subprocess.Popen(
            argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
            cwd=cwd or str(Path(resolved).parent), creationflags=0x00000008 | 0x00000200, close_fds=True)
        return _result(True, f"Started {Path(resolved).stem} (pid {proc.pid}).", exe=resolved, pid=proc.pid)
    except OSError as exc:
        # Some per-user/AppX installs refuse CreateProcess but allow ShellExecute.
        fallback = shell_execute(resolved, args, cwd)
        if fallback.get("ok"):
            return {**fallback, "message": fallback["message"] + f" (via ShellExecute: {exc})"}
        return _result(False, f"{Path(resolved).name} failed to start: {exc}", exe=resolved)


def run_quiet(cmd: Sequence[str], timeout: int = 20) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout,
                              encoding="utf-8", errors="replace")
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return 1, "", f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------- app discovery
def _cached(key: str, builder: Callable[[], Any], ttl: float = CACHE_TTL) -> Any:
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
        value = builder()
        _cache[key] = (time.time(), value)
        return value


def _persistable_path(path: Path) -> Path:
    return config.ROOT / path if not path.is_absolute() else path


def _index_file() -> Path:
    return _persistable_path(Path(SETTINGS.app_index_file))


def start_menu_entries() -> List[Dict[str, str]]:
    """Every shortcut in the user's + all-users Start Menu, as ``{name, target}``.

    Launching a ``.lnk`` through ShellExecute is exactly what the Start menu does, so
    this covers everything the user can see and type - including apps that never
    registered an exe, a protocol or a registry key.
    """
    roots = []
    for env in ("ProgramData", "APPDATA"):
        base = os.environ.get(env, "")
        if base:
            roots.append(Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    roots += [Path(r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs")]
    out: List[Dict[str, str]] = []
    seen: set = set()
    for root in roots:
        if not root.is_dir():
            continue
        try:
            for path in root.rglob("*"):
                if path.suffix.lower() not in (".lnk", ".appref-ms"):
                    continue
                name = path.stem.strip()
                if not name or name.lower() in seen:
                    continue
                if re.search(r"uninstall|remove|readme|license|changelog|documentation|support|website|donat",
                             name, re.I):
                    continue
                seen.add(name.lower())
                out.append({"name": name, "target": str(path)})
        except OSError as exc:  # pragma: no cover
            log.debug("start menu scan %s: %s", root, exc)
    return out


def uwp_apps() -> Dict[str, str]:
    """``Get-StartApps`` -> {display name: AUMID}.  The only reliable way to launch
    Store/MSIX apps (new Teams, Clock, Photos, Xbox...) without pywin32."""
    def build() -> Dict[str, str]:
        code, text = _powershell("Get-StartApps | ForEach-Object { $_.Name + \"`t\" + $_.AppID }")
        out: Dict[str, str] = {}
        if code == 0 and text:
            for line in text.splitlines():
                name, _, appid = line.partition("\t")
                name, appid = name.strip(), appid.strip()
                if name and appid and "!" in appid:
                    out[name] = appid
        if not out:  # a stale file is better than no list at all
            try:
                saved = json.loads(_index_file().read_text(encoding="utf-8"))
                out = {str(k): str(v) for k, v in (saved.get("uwp") or {}).items()}
            except Exception:  # noqa: BLE001
                out = {}
        else:
            try:
                _index_file().parent.mkdir(parents=True, exist_ok=True)
                _index_file().write_text(json.dumps({"uwp": out, "at": int(time.time())}, ensure_ascii=False),
                                         encoding="utf-8")
            except OSError as exc:  # pragma: no cover
                log.debug("app index not saved: %s", exc)
        return out

    return _cached("uwp", build, ttl=max(60.0, SETTINGS.app_index_ttl))


def app_paths() -> Dict[str, str]:
    """HKLM/HKCU ``App Paths`` -> {exe name: full path} for everything on PATH-by-name."""
    def build() -> Dict[str, str]:
        found: Dict[str, str] = {}
        if not IS_WINDOWS:
            return found
        try:
            import winreg  # type: ignore

            for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                for sub in (r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths",
                            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths"):
                    try:
                        root = winreg.OpenKey(hive, sub)
                    except OSError:
                        continue
                    try:
                        for i in range(winreg.QueryInfoKey(root)[0]):
                            try:
                                name = winreg.EnumKey(root, i)
                                key = winreg.OpenKey(root, name)
                                value, _ = winreg.QueryValueEx(key, "")
                                winreg.CloseKey(key)
                            except OSError:
                                continue
                            path = os.path.expandvars(str(value).strip().strip('"'))
                            if path.lower().endswith(".exe") and Path(path).is_file():
                                found.setdefault(name.lower().rstrip(".exe").rstrip("."), path)
                                found.setdefault(Path(name).stem.lower(), path)
                    finally:
                        winreg.CloseKey(root)
        except Exception as exc:  # noqa: BLE001
            log.debug("App Paths scan failed: %s", exc)
        return found

    return _cached("app_paths", build, ttl=max(60.0, SETTINGS.app_index_ttl))


def installed_exes() -> Dict[str, str]:
    """Uninstall-registry DisplayName -> exe (handles non-default drives)."""
    def build() -> Dict[str, str]:
        out: Dict[str, str] = {}
        if not IS_WINDOWS:
            return out
        try:
            import winreg  # type: ignore

            for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                for sub in (r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
                            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"):
                    try:
                        root = winreg.OpenKey(hive, sub)
                    except OSError:
                        continue
                    try:
                        for i in range(winreg.QueryInfoKey(root)[0]):
                            try:
                                key = winreg.OpenKey(root, winreg.EnumKey(root, i))
                                values = {winreg.EnumValue(key, j)[0]: str(winreg.EnumValue(key, j)[1])
                                          for j in range(winreg.QueryInfoKey(key)[1])}
                                winreg.CloseKey(key)
                            except OSError:
                                continue
                            display = (values.get("DisplayName") or "").strip()
                            if not display or re.search(r"update|runtime|redistributable|visual c\+\+|language "
                                                         r"pack|help|sdk|driver", display, re.I):
                                continue
                            exe = ""
                            icon = (values.get("DisplayIcon") or "").split(",")[0].strip().strip('"')
                            if icon.lower().endswith(".exe"):
                                exe = os.path.expandvars(icon)
                            location = os.path.expandvars((values.get("InstallLocation") or "").strip().strip('"'))
                            if not exe and location and Path(location).is_dir():
                                try:
                                    cands = sorted(Path(location).glob("*.exe"),
                                                   key=lambda p: (bool(re.search(r"unins|update|crash|helper|install",
                                                                                p.stem, re.I)), -p.stat().st_size))
                                    exe = str(cands[0]) if cands else ""
                                except OSError:
                                    exe = ""
                            if exe and Path(exe).is_file():
                                out.setdefault(display.lower(), exe)
                    finally:
                        winreg.CloseKey(root)
        except Exception as exc:  # noqa: BLE001
            log.debug("uninstall scan failed: %s", exc)
        return out

    return _cached("installed_exes", build, ttl=max(60.0, SETTINGS.app_index_ttl))


def windows() -> List[Dict[str, Any]]:
    """Visible top-level windows: ``[{hwnd, title, process, pid, minimized}]``."""
    def build() -> List[Dict[str, Any]]:
        if not IS_WINDOWS or user32 is None:
            return []
        items: List[Dict[str, Any]] = []
        EnumWindows = user32.EnumWindows                      # noqa: N806
        GetWindowTextLength = user32.GetWindowTextLengthW      # noqa: N806
        GetWindowText = user32.GetWindowTextW                  # noqa: N806
        IsWindowVisible = user32.IsWindowVisible               # noqa: N806
        IsMin = user32.IsIconic                                # noqa: N806
        GetWindowThreadProcessId = user32.GetWindowThreadProcessId  # noqa: N806
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)  # noqa: N806

        def callback(hwnd: Any, _lparam: Any) -> bool:
            try:
                if not IsWindowVisible(hwnd):
                    return True
                length = GetWindowTextLength(hwnd)
                if not length:
                    return True
                buf = ctypes.create_unicode_buffer(length + 1)
                GetWindowText(hwnd, buf, length + 1)
                pid = ctypes.c_ulong()
                GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                items.append({"hwnd": hwnd, "title": buf.value or "", "pid": int(pid.value),
                              "minimized": bool(IsMin(hwnd))})
            except Exception:  # noqa: BLE001
                return True
            return True

        try:
            EnumWindows(WNDENUMPROC(callback), 0)
        except Exception as exc:  # noqa: BLE001
            log.debug("EnumWindows failed: %s", exc)
        names: Dict[int, str] = {}
        for item in items:
            pid = int(item["pid"])
            if pid not in names:
                names[pid] = _process_name(pid)
            item["process"] = names[pid]
        return items

    return build()          # never cached: window state is what the user is looking at


def _process_name(pid: int) -> str:
    try:
        import psutil  # type: ignore

        return psutil.Process(pid).name()
    except Exception:  # noqa: BLE001 - psutil missing or process gone
        pass
    code, text = _powershell(f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue).ProcessName", timeout=8)
    return text.strip() if code == 0 else ""


def foreground_window() -> Dict[str, Any]:
    if not IS_WINDOWS or user32 is None:
        return _result(False, "No foreground window on this platform.")
    try:
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return _result(False, "Windows reported no foreground window.")
        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return _result(True, buf.value or "", hwnd=int(hwnd), title=buf.value or "", rect=_window_rect(hwnd))
    except Exception as exc:  # noqa: BLE001
        return _result(False, f"Could not read the foreground window: {exc}")


def _window_rect(hwnd: Any) -> Optional[List[int]]:
    try:
        from ctypes import wintypes  # local import: only valid on Windows

        r = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(r))
        return [int(r.left), int(r.top), int(r.right), int(r.bottom)]
    except Exception:  # noqa: BLE001
        return None


def activate(title: str = "", process: str = "") -> Dict[str, Any]:
    """Bring a window to the front instead of starting a second instance."""
    wanted = (title or "").lower().strip()
    proc = (process or "").lower().strip().removesuffix(".exe")
    if not wanted and not proc:
        return _result(False, "Which window should I focus?")
    for item in windows():
        title_ok = bool(wanted) and wanted in str(item.get("title", "")).lower()
        proc_ok = bool(proc) and proc in str(item.get("process", "")).lower()
        if not (title_ok or proc_ok):
            continue
        hwnd = item.get("hwnd")
        if IS_WINDOWS and user32 is not None and hwnd:
            try:
                if user32.IsIconic(hwnd):
                    user32.ShowWindow(hwnd, _SW_RESTORE)
                # Alt-key trick: SetForegroundWindow refuses for background processes.
                user32.keybd_event(0x12, 0, 0, 0)
                user32.SetForegroundWindow(hwnd)
                user32.BringWindowToTop(hwnd)
                user32.keybd_event(0x12, 0, 2, 0)
                return _result(True, f"{item.get('title') or proc} is in front.", hwnd=int(hwnd),
                               title=str(item.get("title", "")))
            except Exception as exc:  # noqa: BLE001
                return _result(False, f"Could not focus that window: {exc}")
        return _result(True, f"{item.get('title')} found.", title=str(item.get("title", "")))
    return _result(False, f"No window matching “{title or process}” is open.",
                   open=[str(w.get("title", ""))[:60] for w in windows()[:8]])


def show_window(hwnd: Any, state: str = "restore") -> Dict[str, Any]:
    if not IS_WINDOWS or user32 is None:
        return _result(False, "Window control needs Windows.")
    mapping = {"restore": _SW_RESTORE, "minimize": _SW_MINIMIZE, "maximize": _SW_MAXIMIZE,
               "hide": _SW_HIDE, "show": _SW_SHOW}
    if state not in mapping:
        return _result(False, f"Unknown window state “{state}”.")
    try:
        user32.ShowWindow(int(hwnd), mapping[state])
        return _result(True, f"Window set to {state}.")
    except Exception as exc:  # noqa: BLE001
        return _result(False, f"Could not change the window: {exc}")


def tool_window(hwnd: Any, on_top: bool = True, no_activate: bool = True) -> Dict[str, Any]:
    """Turn a window into an overlay: no taskbar entry, stays on top, never steals focus.

    Used by the background "bar" so JARVIS can appear over another app without yanking the
    caret out of whatever the user is typing.
    """
    if not IS_WINDOWS or user32 is None:
        return _result(False, "Overlay styling needs Windows.")
    try:
        ex = user32.GetWindowLongW(int(hwnd), _GWL_EXSTYLE)
        mask = _WS_EX_TOOLWINDOW | (_WS_EX_TOPMOST if on_top else 0) | (_WS_EX_NOACTIVATE if no_activate else 0)
        user32.SetWindowLongW(int(hwnd), _GWL_EXSTYLE, int(ex) | mask)
        user32.SetWindowPos(int(hwnd), _HWND_TOPMOST if on_top else _HWND_NOTOPMOST, 0, 0, 0, 0,
                            _SWP_NOMOVE | _SWP_NOSIZE | (_SWP_NOACTIVATE if no_activate else 0)
                            | _SWP_FRAMECHANGED | _SWP_SHOWWINDOW)
        return _result(True, "Overlay style applied.")
    except Exception as exc:  # noqa: BLE001
        return _result(False, f"Could not style the overlay: {exc}")


def move_window(hwnd: Any, x: int = -1, y: int = -1, width: int = -1, height: int = -1) -> Dict[str, Any]:
    if not IS_WINDOWS or user32 is None:
        return _result(False, "Window geometry needs Windows.")
    flags = _SWP_NOACTIVATE
    if x < 0 or y < 0:
        flags |= _SWP_NOMOVE
    if width <= 0 or height <= 0:
        flags |= _SWP_NOSIZE
    try:
        user32.SetWindowPos(int(hwnd), _HWND_TOPMOST, x, y, max(width, 1), max(height, 1), flags)
        return _result(True, "Window moved.")
    except Exception as exc:  # noqa: BLE001
        return _result(False, f"Could not move the window: {exc}")


# ---------------------------------------------------------------------------- input synthesis
_KEYEVENTF_UNICODE, _KEYEVENTF_KEYUP = 0x4, 0x2
_VK = {"enter": 0x0D, "return": 0x0D, "tab": 0x09, "esc": 0x1B, "escape": 0x1B, "space": 0x20,
       "backspace": 0x08, "delete": 0x2E, "del": 0x2E, "insert": 0x2D, "up": 0x26, "down": 0x28,
       "left": 0x25, "right": 0x27, "home": 0x24, "end": 0x23, "pageup": 0x21, "prior": 0x21,
       "pagedown": 0x22, "next": 0x22, "printscreen": 0x2C, "snapshot": 0x2C,
       "win": 0x5B, "windows": 0x5B, "meta": 0x5B, "alt": 0x12, "menu": 0x12, "ctrl": 0x11,
       "control": 0x11, "shift": 0x10, "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73, "f5": 0x74,
       "f6": 0x75, "f7": 0x76, "f8": 0x77, "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
       "volumeup": 0xAF, "volumedown": 0xAE, "volumemute": 0xAD, "mediaplaypause": 0xB3,
       "medianexttrack": 0xB0, "mediaprevioustrack": 0xB1, "mediastop": 0xB2, "launchmail": 0xB4,
       "launchmedia": 0xB6, "apps": 0x5D, "capslock": 0x14, "numlock": 0x90, "scrolllock": 0x91}


def _vk_for(token: str) -> int:
    token = token.strip().lower()
    if token in _VK:
        return _VK[token]
    if re.fullmatch(r"f\d{1,2}", token) and token not in _VK:
        number = int(token[1:])
        if 1 <= number <= 24:
            return 0x6F + number          # VK_F1 == 0x70
    if len(token) == 1:
        char = token.upper()
        if char.isdigit():
            return ord(char)
        if char.isalpha():
            return ord(char)
    return 0


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.c_void_p)]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.c_void_p)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT)]


class _INPUT(ctypes.Structure):
    _anonymous_ = ("union",)
    _fields_ = [("type", ctypes.c_ulong), ("union", _INPUTUNION)]


def _send(inputs: List[_INPUT]) -> Dict[str, Any]:
    if not IS_WINDOWS or user32 is None:
        return _result(True, f"(simulated {len(inputs)} input event(s) - no Windows here)", simulated=True)
    try:
        array_type = _INPUT * len(inputs)
        payload = array_type(*inputs)
        sent = user32.SendInput(len(inputs), ctypes.byref(payload), ctypes.sizeof(_INPUT))
        if int(sent) != len(inputs):
            return _result(False, f"Windows accepted {sent} of {len(inputs)} input events "
                                  "(a secure desktop such as UAC blocks synthetic input).")
        return _result(True, f"Sent {len(inputs)} input event(s).")
    except Exception as exc:  # noqa: BLE001
        return _result(False, f"SendInput failed: {exc}")


def type_text(text: str, press_enter: bool = False) -> Dict[str, Any]:
    """Type into whatever has focus.  Unicode events, so any language works; Ctrl+V is
    used for long or multi-line text because it is 100x faster and never mangles case."""
    if not text:
        return _result(False, "Nothing to type.")
    if len(text) > 60 or "\n" in text:
        clip = clipboard_write(text)
        if clip.get("ok"):
            pasted = hotkey("ctrl+v")
            if pasted.get("ok"):
                if press_enter:
                    press("enter")
                return _result(True, f"Typed {len(text)} characters into the focused window.",
                               method="clipboard")
        # clipboard route failed (locked session, RDP): fall through to per-char typing
    inputs: List[_INPUT] = []
    for char in text:
        vk, scan = _char_event(char)
        inputs.append(_INPUT(type=1, ki=_KEYBDINPUT(wVk=vk, wScan=scan, dwFlags=_KEYEVENTF_UNICODE,
                                                   time=0, dwExtraInfo=None)))
        inputs.append(_INPUT(type=1, ki=_KEYBDINPUT(wVk=vk, wScan=scan,
                                                   dwFlags=_KEYEVENTF_UNICODE | _KEYEVENTF_KEYUP,
                                                   time=0, dwExtraInfo=None)))
    if press_enter:
        inputs.append(_INPUT(type=1, ki=_KEYBDINPUT(wVk=_VK["enter"], wScan=0, dwFlags=0, time=0, dwExtraInfo=None)))
        inputs.append(_INPUT(type=1, ki=_KEYBDINPUT(wVk=_VK["enter"], wScan=0, dwFlags=_KEYEVENTF_KEYUP,
                                                   time=0, dwExtraInfo=None)))
    out = _send(inputs)
    return {**out, "characters": len(text), "method": "unicode"}


def press(key: str, times: int = 1) -> Dict[str, Any]:
    codes = [tok for tok in re.split(r"[ ,+]+", (key or "").strip()) if tok]
    if not codes:
        return _result(False, "Which key should I press?")
    vk = _vk_for(codes[0])
    if not vk:
        return _result(False, f"I don't know the key “{codes[0]}”.")
    inputs: List[_INPUT] = []
    for _ in range(max(1, min(int(times or 1), 100))):
        inputs.append(_INPUT(type=1, ki=_KEYBDINPUT(wVk=vk, wScan=0, dwFlags=0, time=0, dwExtraInfo=None)))
        inputs.append(_INPUT(type=1, ki=_KEYBDINPUT(wVk=vk, wScan=0, dwFlags=_KEYEVENTF_KEYUP,
                                                   time=0, dwExtraInfo=None)))
    out = _send(inputs)
    return {**out, "key": codes[0]}


def hotkey(combo: str) -> Dict[str, Any]:
    """``ctrl+alt+delete`` style chords (win+l, ctrl+s, alt+tab, ctrl+shift+esc...)."""
    tokens = [t for t in re.split(r"[+]+", (combo or "").strip().lower()) if t]
    if not tokens:
        return _result(False, "Which shortcut? e.g. ctrl+s")
    vks = [_vk_for(t) for t in tokens]
    if 0 in vks:
        unknown = tokens[vks.index(0)]
        return _result(False, f"I don't know the key “{unknown}”.")
    inputs: List[_INPUT] = []
    for vk in vks:
        inputs.append(_INPUT(type=1, ki=_KEYBDINPUT(wVk=vk, wScan=0, dwFlags=0, time=0, dwExtraInfo=None)))
    for vk in reversed(vks):
        inputs.append(_INPUT(type=1, ki=_KEYBDINPUT(wVk=vk, wScan=0, dwFlags=_KEYEVENTF_KEYUP,
                                                   time=0, dwExtraInfo=None)))
    out = _send(inputs)
    return {**out, "combo": "+".join(tokens)}


def click(x: int = -1, y: int = -1, button: str = "left", clicks: int = 1, double: bool = False) -> Dict[str, Any]:
    if not IS_WINDOWS or user32 is None:
        return _result(True, "(simulated click - no Windows here)", simulated=True)
    if x < 0 or y < 0:
        from ctypes import wintypes

        pos = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pos))
        x, y = int(pos.x), int(pos.y)
    user32.SetCursorPos(int(x), int(y))
    flags = MOUSE_BUTTONS
    if button.lower() not in flags:
        return _result(False, f"Unknown mouse button “{button}”.")
    down, up = flags[button.lower()]
    inputs: List[_INPUT] = []
    total = max(1, min(int(clicks or 1), 5))
    for i in range(total):
        inputs.append(_INPUT(type=0, mi=_MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=down, time=0, dwExtraInfo=None)))
        inputs.append(_INPUT(type=0, mi=_MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=up, time=0, dwExtraInfo=None)))
        if double and i == 0:
            # A double click is just a second press and release inside the system's double-click
            # time; the API accepts both bits in one event, which is what _double_click_flags
            # returns.  (This used to OR in 0x0008 - MOUSEEVENTF_RIGHTDOWN - so every "double
            # click" also poked the right mouse button and raised a context menu.)
            extra = _double_click_flags(down, up)
            inputs.append(_INPUT(type=0, mi=_MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=extra, time=0,
                                                        dwExtraInfo=None)))
    out = _send(inputs)
    return {**out, "at": [int(x), int(y)], "button": button}


def move_mouse(x: int, y: int, duration_ms: int = 0) -> Dict[str, Any]:
    if not IS_WINDOWS or user32 is None:
        return _result(True, "(simulated move)", simulated=True)
    user32.SetCursorPos(int(x), int(y))
    return _result(True, f"Pointer at {x},{y}.", at=[int(x), int(y)])


def scroll(amount: int = -3, x: int = -1, y: int = -1) -> Dict[str, Any]:
    """Negative scrolls down, positive up; one unit is one wheel notch."""
    if not IS_WINDOWS or user32 is None:
        return _result(True, "(simulated scroll)", simulated=True)
    if x >= 0 and y >= 0:
        user32.SetCursorPos(int(x), int(y))
    notches = max(-100, min(int(amount or -3), 100)) or -1
    # MOUSEEVENTF_WHEEL is 0x0800 and delta is a signed 32-bit count of WHEEL_DELTA notches.
    delta = struct.unpack("i", struct.pack("i", 120 if notches > 0 else -120))[0] & 0xFFFFFFFF
    inputs = [_INPUT(type=0, mi=_MOUSEINPUT(dx=0, dy=0, mouseData=delta, dwFlags=0x0800, time=0,
                                           dwExtraInfo=None)) for _ in range(abs(notches))]
    out = _send(inputs)
    return {**out, "notches": notches}


def cursor_position() -> List[int]:
    if not IS_WINDOWS or user32 is None:
        return [0, 0]
    from ctypes import wintypes

    pos = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pos))
    return [int(pos.x), int(pos.y)]


def screen_size() -> List[int]:
    if not IS_WINDOWS or user32 is None:
        return [0, 0]
    return [int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))]


def work_area() -> List[int]:
    """The desktop rectangle *excluding* the taskbar: ``[x, y, width, height]``.

    ``screen_size()`` reports the full primary monitor (taskbar included), so a
    window positioned with it can end up under the taskbar or, on a DPI-scaled
    laptop, entirely off-screen.  This is the geometry the floating bar should
    actually be placed inside.
    """
    if not IS_WINDOWS or user32 is None:
        return [0, 0, 0, 0]
    try:
        from ctypes import wintypes

        rect = wintypes.RECT()
        SPI_GETWORKAREA = 0x0030
        if user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(rect), 0):
            return [int(rect.left), int(rect.top), int(rect.right - rect.left),
                    int(rect.bottom - rect.top)]
    except Exception:  # noqa: BLE001 - any failure here just means "unknown geometry"
        pass
    size = screen_size()
    return [0, 0, int(size[0]), int(size[1])]


# ---------------------------------------------------------------------------- clipboard
def clipboard_read() -> Dict[str, Any]:
    if not IS_WINDOWS:
        return _result(False, "Clipboard access is implemented for Windows.")
    CF_UNICODETEXT = 13
    try:
        user32.OpenClipboard(None)
        try:
            if not user32.IsClipboardFormatAvailable(CF_UNICODETEXT):
                return _result(True, "The clipboard holds no text.", text="")
            handle = user32.GetClipboardData(CF_UNICODETEXT)
            if not handle:
                return _result(True, "The clipboard is empty.", text="")
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.GlobalLock.restype = ctypes.c_void_p
            kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
            pointer = kernel32.GlobalLock(ctypes.c_void_p(handle))
            if not pointer:
                return _result(False, "Could not read the clipboard.")
            text = ctypes.wstring_at(ctypes.c_wchar_p(pointer))
            kernel32.GlobalUnlock(ctypes.c_void_p(handle))
        finally:
            user32.CloseClipboard()
        return _result(True, f"Clipboard holds {len(text)} characters." if text else "The clipboard is empty.",
                       text=text)
    except Exception as exc:  # noqa: BLE001
        return _result(False, f"Clipboard read failed: {exc}")


def clipboard_write(text: str) -> Dict[str, Any]:
    if not IS_WINDOWS:
        return _result(False, "Clipboard access is implemented for Windows.")
    CF_UNICODETEXT, GMEM_MOVEABLE = 13, 0x0002
    try:
        data = (text + "\0").encode("utf-16-le")
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.GlobalAlloc.restype = ctypes.c_void_p
        kernel32.GlobalSize.restype = ctypes.c_size_t
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
        if not handle:
            return _result(False, "Could not allocate clipboard memory.")
        pointer = kernel32.GlobalLock(ctypes.c_void_p(handle))
        ctypes.memmove(pointer, data, len(data))
        kernel32.GlobalUnlock(ctypes.c_void_p(handle))
        user32.OpenClipboard(None)
        try:
            user32.EmptyClipboard()
            user32.SetClipboardData(CF_UNICODETEXT, ctypes.c_void_p(handle))
        finally:
            user32.CloseClipboard()
        return _result(True, f"Copied {len(text)} characters.", characters=len(text))
    except Exception as exc:  # noqa: BLE001
        return _result(False, f"Clipboard write failed: {exc}")


# ---------------------------------------------------------------------------- media / audio
_MEDIA_KEYS = {"next": 0xB0, "previous": 0xB1, "playpause": 0xB3, "stop": 0xB2,
               "volumeup": 0xAF, "volumedown": 0xAE, "mute": 0xAD}


def media(key: str) -> Dict[str, Any]:
    token = (key or "").lower().replace("_", "").replace("-", "").replace(" ", "")
    aliases = {"nexttrack": "next", "skip": "next", "forward": "next", "prev": "previous",
               "previous": "previous", "back": "previous", "play": "playpause", "pause": "playpause",
               "play_pause": "playpause"}
    token = aliases.get(token, token)
    vk = _MEDIA_KEYS.get(token)
    if not vk:
        return _result(False, f"Unknown media key “{key}”.")
    out = press({"next": "medianexttrack", "previous": "mediaprevioustrack", "playpause": "mediaplaypause",
                 "stop": "mediastop", "volumeup": "volumeup", "volumedown": "volumedown",
                 "mute": "volumemute"}[token])
    return {**out, "message": {"next": "Skipping to the next track.", "previous": "Back one track.",
                               "playpause": "Toggled play/pause.", "stop": "Stopped playback.",
                               "volumeup": "Volume up.", "volumedown": "Volume down.",
                               "mute": "Muted."}[token]}


def volume(action: str = "get", level: int = -1) -> Dict[str, Any]:
    """Master volume.  Up/down/mute use the real media keys (they always work); an
    absolute level is driven through WScript.SendKeys, which walks the same mixer the
    user does, so it lands within a couple of percent with no COM endpoint needed."""
    action = (action or "get").lower()
    if action == "mute":
        return media("mute")
    if action == "unmute":
        return media("mute")          # the mute key is a toggle: it unmutes too
    if action in ("up", "down"):
        try:
            steps = int(level) if level and int(level) > 0 else 5
        except (TypeError, ValueError):
            steps = 5
        for _ in range(max(1, min(steps, 50))):
            media("volumeup" if action == "up" else "volumedown")
        return _result(True, f"Volume {'up' if action == 'up' else 'down'}.")
    if action == "set":
        try:
            target = max(0, min(int(level), 100))
        except (TypeError, ValueError):
            return _result(False, f"'{level}' is not a volume level.")
        if not IS_WINDOWS:
            return _result(False, "Setting an exact volume needs Windows.")
        script = ("$shell = New-Object -ComObject WScript.Shell; "
                  "for ($i = 0; $i -lt 100; $i++) { $shell.SendKeys([char]175) }; "
                  f"for ($i = 0; $i -lt [math]::Round({target} / 2); $i++) {{ $shell.SendKeys([char]174) }}; "
                  "'ok'")
        code, text = _powershell(script, timeout=60)
        if code == 0:
            return _result(True, f"Volume set to about {target}%.", method="wscript-mixer", level=target)
        return _result(False, f"Could not set the volume ({(text or 'powershell failed')[:80]}).")
    return _result(True, "Say 'volume up', 'volume down', 'mute' or 'set volume to 40'.")

# ---------------------------------------------------------------------------- system bits
def lock_workstation() -> Dict[str, Any]:
    if IS_WINDOWS and user32 is not None:
        try:
            return _result(bool(user32.LockWorkStation()), "Workstation locked.")
        except Exception as exc:  # noqa: BLE001
            return _result(False, f"Lock failed: {exc}")
    return _result(False, "Locking only makes sense on Windows.")


def set_wallpaper(image: str) -> Dict[str, Any]:
    path = Path(os.path.expandvars(os.path.expanduser(image or "")))
    if not path.is_file():
        return _result(False, f"No image at {image}.")
    SPI_SETDESKWALLPAPER, SPIF_UPDATEINIFILE, SPIF_SENDWININICHANGE = 0x0014, 0x01, 0x02
    if not IS_WINDOWS or user32 is None:
        return _result(False, "Wallpapers need Windows.")
    try:
        ok = user32.SystemParametersInfoW(SPI_SETDESKWALLPAPER, 0, str(path),
                                          SPIF_UPDATEINIFILE | SPIF_SENDWININICHANGE)
        return _result(bool(ok), "Wallpaper set." if ok else "Windows refused the wallpaper.")
    except Exception as exc:  # noqa: BLE001
        return _result(False, f"Could not set the wallpaper: {exc}")


def notify(title: str, body: str = "", seconds: int = 8) -> Dict[str, Any]:
    """A real Windows toast (WinRT, no deps); falls back to a MessageBox so a reminder is
    never silently lost."""
    title = (title or "JARVIS").strip()[:120]
    body = (body or "").strip()[:600]
    if not IS_WINDOWS:
        return _result(True, f"(notification on a non-Windows box) {title}: {body}", simulated=True)
    xml = (f"<toast><visual><binding template='ToastGeneric'><text>{_xml_escape(title)}</text>"
           f"<text>{_xml_escape(body)}</text></binding></visual></toast>")
    script = f"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType = WindowsRuntime] | Out-Null
$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml("{xml.replace('"', '&quot;')}")
$tag = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$node = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("JARVIS")
$toast = New-Object Windows.UI.Notifications.ToastNotification $xml
$node.Show($toast)
"toast shown"
"""
    code, text = _powershell(script, timeout=20)
    if code == 0 and "toast shown" in text:
        return _result(True, f"Toast sent: {title}", method="winrt")
    return _result(True, f"Fell back to a dialog: {title}", method="messagebox",
                   detail=shell_execute("rundll32.exe", f"user32.dll,MessageBoxW \"{body}\" \"{title}\" 0x40"))


def _xml_escape(value: str) -> str:
    return (value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("'", "&apos;"))


def recycle(paths: List[str]) -> Dict[str, Any]:
    """Send files to the Recycle Bin - never ``os.remove`` - so the user can undo it."""
    if not paths:
        return _result(False, "Nothing to delete.")
    if not IS_WINDOWS:
        return _result(False, "Recycle Bin needs Windows.")
    FO_DELETE = 0x0003
    try:
        from ctypes import wintypes

        class SHFILEOPSTRUCTW(ctypes.Structure):
            _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT),
                        ("pFrom", wintypes.LPCWSTR), ("pTo", wintypes.LPCWSTR),
                        ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", wintypes.BOOL),
                        ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]

        # Double-NUL terminated list, as the API requires
        from_buffer = "\0".join(str(p) for p in paths) + "\0\0"
        op = SHFILEOPSTRUCTW(None, FO_DELETE, from_buffer, None,
                             RECYCLE_FLAGS, False, None, None)
        ret = int(shell32.SHFileOperationW(ctypes.byref(op)))  # type: ignore[union-attr]
        if ret == 0:
            return _result(True, f"{len(paths)} item(s) moved to the Recycle Bin.",
                           items=[Path(str(p)).name for p in paths], aborted=bool(op.fAnyOperationsAborted))
        return _result(False, f"Windows refused the delete (code {ret}).", code=ret)
    except Exception as exc:  # noqa: BLE001
        return _result(False, f"Recycle Bin call failed: {exc}")


def empty_recycle_bin() -> Dict[str, Any]:
    if not IS_WINDOWS or shell32 is None:
        return _result(False, "This needs Windows.")
    try:
        ret = int(shell32.SHEmptyRecycleBinW(None, None, 0x0001 | 0x0002 | 0x0004))
        return _result(ret in (0, -2147418113), "Recycle Bin emptied." if ret == 0 else f"Nothing to do (code {ret}).")
    except Exception as exc:  # noqa: BLE001
        return _result(False, f"Could not empty the Recycle Bin: {exc}")


def battery() -> Dict[str, Any]:
    if not IS_WINDOWS or kernel32 is None:      # GetSystemPowerStatus is a kernel32 call
        return _result(False, "Battery status needs Windows.")
    try:
        from ctypes import wintypes

        class SYSTEM_POWER_STATUS(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", wintypes.DWORD), ("BatteryFullLifeTime", wintypes.DWORD)]

        status = SYSTEM_POWER_STATUS()
        if not kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            return _result(False, "Windows has no battery (desktop PC).")
        minutes = int(status.BatteryLifeTime) // 60
        # BatteryFlag: 0x01 means charging, 0x08 means "no system battery" - the old code read
        # the no-battery bit as "charging" and reported a desktop as plugged in and vice versa.
        flag = int(status.BatteryFlag)
        if flag & 0x08:
            state = "no battery"
        elif flag & 0x01:
            state = "charging"
        elif int(status.ACLineStatus) == 0:
            state = "on battery"
        else:
            state = "plugged in"
        return _result(True, f"Battery at {status.BatteryLifePercent}% {state}"
                             + (f", {minutes} minutes left" if state == "on battery" and minutes
                                and minutes < 2400 else ""),
                       percent=int(status.BatteryLifePercent), ac=int(status.ACLineStatus),
                       minutes=minutes, state=state)
    except Exception as exc:  # noqa: BLE001
        return _result(False, f"Power status unavailable: {exc}")


def process_list(kind: str = "apps") -> List[Dict[str, Any]]:
    try:
        import psutil  # type: ignore

        out = []
        for proc in psutil.process_iter(["pid", "name", "username", "memory_info"]):
            try:
                info = proc.info
                out.append({"pid": info["pid"], "name": info["name"] or "",
                            "memory_mb": round((info["memory_info"].rss if info["memory_info"] else 0) / 1e6, 1)})
            except Exception:  # noqa: BLE001 - a process vanished mid-scan
                continue
        return sorted(out, key=lambda d: -d["memory_mb"])[:40]
    except Exception:  # noqa: BLE001
        code, text = _powershell("Get-Process | Sort-Object WS -Descending | Select-Object -First 25 Name,Id | "
                                 "ForEach-Object { $_.Name + \" \" + $_.Id }", timeout=15)
        if code != 0:
            return []
        rows = []
        for line in text.splitlines():
            name, _, pid = line.rpartition(" ")
            rows.append({"name": name.strip(), "pid": int(pid) if pid.isdigit() else 0, "memory_mb": 0})
        return rows


def kill_process(name: str, grace_seconds: int = 4) -> Dict[str, Any]:
    exe = (name or "").strip()
    if not exe:
        return _result(False, "Which process should I close?")
    if not exe.lower().endswith(".exe"):
        exe = f"{exe}.exe"
    if IS_WINDOWS:
        code, text, err = run_quiet(["taskkill", "/IM", exe, "/F"], timeout=25)
        if code == 0:
            return _result(True, f"Closed {exe}.", output=text[:200])
        if "not found" in (err + text).lower():
            return _result(False, f"{exe} is not running.")
        return _result(False, f"Windows refused to close {exe}: {(err or text)[:120]}")
    code, text, err = run_quiet(["pkill", "-f", exe], timeout=15)
    return _result(code == 0, f"Closed {exe}." if code == 0 else f"{exe} is not running.")


def open_folder(path: str, select: str = "") -> Dict[str, Any]:
    target = os.path.expandvars(os.path.expanduser(path or ""))
    if select:
        full = os.path.abspath(os.path.join(target, select)) if not Path(select).is_absolute() else select
        if IS_WINDOWS:
            return _run_detached(["explorer.exe", f"/select,{full}"], f"{target} (selected {select})")
    if not target:
        return shell_execute("explorer.exe", "shell:MyComputerFolder")
    return shell_execute(target)


# ---------------------------------------------------------------------------- misc helpers
def win32_health() -> Dict[str, Any]:
    """What the Windows layer could actually set up, in one sentence.

    Called by ``main.preflight()`` and by ``/api/desktop`` so an unresolvable entry point shows up
    at boot with the feature it breaks, instead of arriving as "ShellExecute failed: function
    'ShellExecuteW' not found" in front of a user who just wanted Settings to open.
    """
    if not IS_WINDOWS:
        return _result(True, "Not on Windows: the desktop calls below are simulated.",
                       windows=False, missing=[], disabled=[])
    if not _WIN32_MISSING:
        return _result(True, "Win32 ready: user32, shell32, gdi32 and kernel32 are bound.",
                       windows=True, missing=[], disabled=[])
    disabled = sorted({f"{_WIN32_FEATURES[name.split('!')[-1]]}" for name in _WIN32_MISSING
                       if name.split("!")[-1] in _WIN32_FEATURES})
    message = ("Win32 gaps: " + ", ".join(_WIN32_MISSING)
               + (".  Lost: " + "; ".join(disabled) if disabled else ""))
    return _result(False, message, windows=True, missing=list(_WIN32_MISSING), disabled=disabled)


def _winsound_beep(frequency: int, duration_ms: int) -> None:
    """One beep through the standard library.

    Split out so a test can verify the order of preference without the laptop actually making a
    noise in the middle of ``python -m unittest``.
    """
    import winsound  # local import: standard library, Windows only

    winsound.Beep(int(frequency), int(duration_ms))


def beep(times: int = 1, frequency: int = 880, duration_ms: int = 120) -> Dict[str, Any]:
    """Beep on the PC speaker - a courtesy, so it may never cost the user a command.

    ``winsound`` goes first because the standard library already knows this call; the raw entry
    point is the fallback.  Both are guarded: a beep that cannot happen should appear as a line in
    ``python winops.py``, not as an exception inside "open settings".  Asking user32 for ``Beep``
    was the old bug - that name is exported by kernel32, while user32 has ``MessageBeep``.
    """
    if not IS_WINDOWS:
        return _result(True, "(beep simulated)", simulated=True)
    count = max(1, min(int(times or 1), 6))
    freq, ms = int(frequency), max(10, min(int(duration_ms or 120), 5000))
    reasons: List[str] = []
    try:
        for _ in range(count):
            _winsound_beep(freq, ms)
        return _result(True, "Beeped.", method="winsound", times=count)
    except Exception as exc:  # noqa: BLE001
        reasons.append("winsound: " + repr(exc))
    if kernel32 is not None:
        try:
            for _ in range(count):
                kernel32.Beep(freq, ms)  # type: ignore[union-attr]
            return _result(True, "Beeped.", method="kernel32", times=count)
        except Exception as exc:  # noqa: BLE001
            reasons.append("kernel32!Beep: " + repr(exc))
    else:
        reasons.append("kernel32 not loaded")
    return _result(False, "No beep available (" + "; ".join(reasons) + ").")


def temp_path(suffix: str = ".png") -> Path:
    folder = Path(tempfile.gettempdir()) / "jarvis"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{time.strftime('%Y%m%d-%H%M%S')}{suffix}"


def screenshot(path: str = "", region: Optional[List[int]] = None, delay_ms: int = 0) -> Dict[str, Any]:
    """GDI screen grab with no dependencies; Pillow is used only to write PNG."""
    target = Path(os.path.expandvars(os.path.expanduser(path))) if path else temp_path(".png")
    if delay_ms:
        time.sleep(max(0, min(int(delay_ms), 8000)) / 1000.0)
    if not IS_WINDOWS or user32 is None or gdi32 is None:
        return _result(False, "Screenshots need a Windows desktop session with the GDI objects "
                              "available (user32 + gdi32).")
    try:
        from ctypes import wintypes

        user32.SetProcessDPIAware()
        width = int(region[2] - region[0]) if region else int(user32.GetSystemMetrics(0))
        height = int(region[3] - region[1]) if region else int(user32.GetSystemMetrics(1))
        screen_dc = user32.GetDC(0)
        mem_dc = gdi32.CreateCompatibleDC(screen_dc)
        bitmap = gdi32.CreateCompatibleBitmap(screen_dc, width, height)
        gdi32.SelectObject(mem_dc, bitmap)
        SRCCOPY, CAPTUREBLT = 0x00CC0020, 0x00000001
        src_x, src_y = (region[0], region[1]) if region else (0, 0)
        gdi32.BitBlt(mem_dc, 0, 0, width, height, screen_dc, src_x, src_y, SRCCOPY | CAPTUREBLT)

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [("biSize", wintypes.DWORD), ("biWidth", ctypes.c_long), ("biHeight", ctypes.c_long),
                        ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD),
                        ("biCompression", wintypes.DWORD), ("biSizeImage", wintypes.DWORD),
                        ("biXPelsPerMeter", ctypes.c_long), ("biYPelsPerMeter", ctypes.c_long),
                        ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]

        header = BITMAPINFOHEADER()
        header.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        header.biWidth, header.biHeight = width, -height
        header.biPlanes, header.biBitCount, header.biCompression = 1, 32, 0
        buffer = ctypes.create_string_buffer(width * height * 4)
        got = gdi32.GetDIBits(mem_dc, bitmap, 0, height, buffer, ctypes.byref(header), 0)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(0, screen_dc)
        gdi32.DeleteObject(bitmap)
        if not got:
            return _result(False, "The screen capture came back empty (a DRM-protected window?).")
        saved = _write_png(target, width, height, buffer.raw)
        if not saved.get("ok"):
            return saved
        return _result(True, f"Screenshot saved to {target}", path=str(target), width=width, height=height)
    except Exception as exc:  # noqa: BLE001
        return _result(False, f"Screen capture failed: {exc}")


def _write_png(path: Path, width: int, height: int, rgba_rows: bytes) -> Dict[str, Any]:
    """BGRA -> PNG.  Pillow if present, otherwise a hand-built PNG (zlib + chunks)."""
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image  # type: ignore

            image = Image.frombuffer("RGBA", (width, height), rgba_rows, "raw", "BGRA", 0, 1)
            image.convert("RGB").save(target, "PNG")
            return _result(True, "png", path=str(target))
        except ImportError:
            import zlib

            raw = bytearray()
            stride = width * 4
            for row in range(height):
                raw.append(0)
                start, end = row * stride, row * stride + stride
                line = bytearray(rgba_rows[start:end])
                for i in range(0, len(line), 4):
                    line[i], line[i + 2] = line[i + 2], line[i]        # BGRA -> RGBA
                raw += line

            def chunk(tag: bytes, payload: bytes) -> bytes:
                body = tag + payload
                return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

            png = (b"\x89PNG\r\n\x1a\n"
                   + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
                   + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
                   + chunk(b"IEND", b""))
            target.write_bytes(png)
            return _result(True, "png", path=str(target))
    except Exception as exc:  # noqa: BLE001
        return _result(False, f"Could not write {path}: {exc}")


def key_down(vk: int) -> bool:
    """Poll a virtual key (used by the hotkey watcher without a message loop)."""
    if not IS_WINDOWS or user32 is None:
        return False
    try:
        return bool(user32.GetAsyncKeyState(vk) & 0x8000)
    except Exception:  # noqa: BLE001
        return False


def describe_probe(label: str, out: Any) -> Dict[str, Any]:
    """Turn any probe's return value into one self-check row, whatever shape it came back in.

    The desktop primitives deliberately disagree about shape - ``windows()`` returns a list,
    ``battery()`` a result dict, ``uwp_apps()``/``app_paths()``/``installed_exes()`` a bare
    ``{name: target}`` mapping - and the first version read only ``message``, so three rows printed
    nothing on the first run on real hardware.  A blank row is worse than no row: it reads as
    "checked, nothing to report" when it actually means "we did not know how to say this".
    """
    if isinstance(out, dict):
        if "ok" in out or "message" in out:
            ok = bool(out.get("ok", True))
            message = str(out.get("message", "")).strip()
            if not message:
                message = "reported a failure without saying why" if not ok else "done"
            return {"name": label, "ok": ok, "message": message[:220]}
        return {"name": label, "ok": True, "message": "%d entries" % len(out)}
    if isinstance(out, (list, tuple, set)):
        return {"name": label, "ok": True, "message": "%d entries" % len(out)}
    if out is None:
        return {"name": label, "ok": False, "message": "returned nothing"}
    return {"name": label, "ok": True, "message": str(out)[:220]}


def check(light: bool = False) -> Dict[str, Any]:
    """Run every read-only desktop probe and report what each one can actually do.

    ``light`` skips the rows that take seconds on a real desktop - the Start Menu, Store, registry
    and Program Files enumerations (each one a PowerShell or registry walk) and the screen grab -
    so the unit suite stays quick on Windows while ``python winops.py`` still checks everything.
    On the laptop the skipped ones were nine seconds of the twenty a full run takes.

    ``python winops.py`` prints this table.  It launches nothing, clicks nothing and touches no
    file outside a temp PNG, so it is safe to run at any moment - and it is the thing to paste when
    "it can't open apps" happens again, because it covers the five places where a laptop and a
    sandbox disagree: the Win32 bindings themselves, the window list, the app index, the OCR
    backend and a real screen grab.
    """
    rows: List[Dict[str, Any]] = []

    def probe(label: str, call: Callable[[], Any], windows_only: bool = False,
              heavy: bool = False) -> None:
        if heavy and light:
            rows.append({"name": label, "ok": True, "message": "skipped - light check"})
            return
        if windows_only and not IS_WINDOWS:
            # Reporting a Windows-only probe as a failure on another OS teaches the wrong lesson:
            # the point of this table is to separate "not here" from "broken here".
            rows.append({"name": label, "ok": True, "message": "skipped - needs a Windows session"})
            return
        try:
            out = call()
        except Exception as exc:  # noqa: BLE001 - a probe that raises is exactly what we want to see
            rows.append({"name": label, "ok": False, "message": f"raised {type(exc).__name__}: {exc}"})
            return
        rows.append(describe_probe(label, out))

    probe("win32 bindings", win32_health)
    probe("screen size", lambda: {"ok": all(screen_size()), "message": " x ".join(map(str, screen_size()))}, windows_only=True)
    probe("window list", lambda: windows(), windows_only=True)
    probe("foreground window", foreground_window, windows_only=True)
    probe("start menu entries", lambda: start_menu_entries(), windows_only=True, heavy=True)
    probe("store apps (UWP)", lambda: uwp_apps(), windows_only=True, heavy=True)
    probe("app paths registry", lambda: app_paths(), windows_only=True, heavy=True)
    probe("installed programs", lambda: installed_exes(), windows_only=True, heavy=True)
    probe("battery", battery, windows_only=True)
    probe("clipboard", clipboard_read, windows_only=True)
    probe("screen grab", lambda: screenshot(str(temp_path(".png"))), windows_only=True,
          heavy=True)
    probe("input injection", lambda: {"ok": True, "message": "SendInput bound"}
          if "user32!SendInput" not in _WIN32_MISSING else {"ok": False, "message": "SendInput missing"})
    problems = [row for row in rows if not row["ok"]]
    return {"ok": not problems, "windows": bool(IS_WINDOWS), "rows": rows,
            "problems": [row["name"] for row in problems],
            "message": (f"{len(rows) - len(problems)}/{len(rows)} desktop probes passed"
                        + (f" - failing: {', '.join(r['name'] for r in problems)}" if problems else ""))}


if __name__ == "__main__":  # pragma: no cover - a human runs this on the target machine
    report = check(light="--fast" in sys.argv or "-f" in sys.argv)
    print(f"\nJARVIS desktop self-check  ({'Windows' if IS_WINDOWS else 'not Windows'})\n" + "-" * 66)
    for row in report["rows"]:
        print(f"  [{'ok' if row['ok'] else 'XX'}] {row['name']:<22} {row['message']}")
    print("-" * 66 + f"\n{report['message']}\n")
    if report["problems"]:
        print("Paste the block above when reporting a problem: each line names the feature that is")
        print("lost, and the .env setting or install that brings it back.")
    raise SystemExit(0 if report["ok"] else 1)


__all__ = ["IS_WINDOWS", "AVAILABLE", "shell_execute", "launch_exe", "windows", "foreground_window",
           "activate", "show_window", "tool_window", "move_window", "type_text", "press", "hotkey",
           "click", "move_mouse", "scroll", "cursor_position", "screen_size", "clipboard_read",
           "clipboard_write", "media", "volume", "lock_workstation", "set_wallpaper", "notify",
           "recycle", "empty_recycle_bin", "battery", "process_list", "kill_process", "open_folder",
           "start_menu_entries", "uwp_apps", "app_paths", "installed_exes", "screenshot", "beep",
           "temp_path", "key_down", "run_quiet", "win32_health", "check"]
