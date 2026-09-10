"""Name -> something that actually launches: JARVIS's application intelligence.

The old launcher knew 15 hardcoded apps and tried ``subprocess.run(["ms-settings:"])`` for
protocol URIs, which Windows cannot do - so "open settings" failed, "open Microsoft Teams"
failed (new Teams is an MSIX package with no exe on PATH), and every app that was not in
the little table was a coin flip.

Resolution chain (:func:`resolve`), most reliable first - and each rung is *independent*,
so a broken Start Menu scan never takes the whole thing down:

1. ``ms-settings:`` page            - "open bluetooth settings", "open windows update"
2. built-in table                   - stock Windows + popular apps, with AUMIDs/protocols
3. ``CUSTOM_APPS`` from .env        - the user's own ``name=C:\\path\\to.exe`` lines
4. Start Menu index                 - every ``.lnk``/``.appref-ms`` the user can see
5. ``Get-StartApps`` (MSIX/UWP)     - Teams, Clock, Photos, Xbox, Store...
6. registry ``App Paths``           - anything that registered a launchable exe name
7. uninstall registry               - non-default drives (D:\\Steam, per-user installs)
8. ``PATH`` lookup, then ShellExecute on the raw words (Start-menu style)

Every rung ends in :func:`winops.shell_execute` / :func:`winops.launch_exe`, so the result is
the same call Explorer makes.  :func:`suggest` turns "open crome" into a real offer
("did you mean Chrome?") instead of a shrug.
"""

from __future__ import annotations

import difflib
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config
import winops
from config import SETTINGS, get_logger

log = get_logger("apps")

# --------------------------------------------------------------------------- text helpers
_STOPWORDS = {"app", "application", "the", "my", "please", "jarvis", "open", "launch", "start",
              "run", "up", "on", "for", "a", "an", "windows", "microsoft", "pc", "desktop",
              "program", "exe", "official", "client", "of"}


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (value or "").lower()).strip()


def _tokens(value: str) -> set:
    return {t for t in _norm(value).split() if t and t not in _STOPWORDS} or set(_norm(value).split())


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", _norm(value)).strip("-")
    return re.sub(r"-{2,}", "-", cleaned)


_LEAD = re.compile(r"^(?:hey[ ,]+)?(?:jarvis|computer|ok[ ,]+google)[ ,]+", re.I)
_POLITE = re.compile(r"^(?:could you |can you |would you |please |kindly |just |now |quickly )+", re.I)
_VERB = re.compile(r"^(?:open|launch|start|boot(?: up)?|fire up|spin up|bring up|run|wake|show|"
                   r"switch to|focus|go to|close|quit|kill|exit|terminate|shut down|minimise|minimize)"
                   r"\s+(?:up\s+)?(?:the\s+|my\s+|our\s+|a\s+|an\s+)*", re.I)
_TAIL = re.compile(r"\s*(?:\b(app|application|program|software|exe|desktop app)\b|"
                   r"\b(for me|now|please|thanks|thank you|instantly)\b)[ .,!]*$", re.I)


def strip_verbs(value: str) -> str:
    """"please open up the teams app for me" -> "teams".

    Idempotent: applied until it stops shrinking, because the polite opener, the verb and
    the trailing "app" can all be present at once and one pass leaves "up the teams".
    """
    text = (value or "").strip().strip('"').strip("'")
    for _ in range(4):
        before = text
        text = _LEAD.sub("", text)
        text = _POLITE.sub("", text)
        text = _VERB.sub("", text)
        text = _TAIL.sub("", text)
        text = text.strip(" .,!?")
        if text == before:
            break
    return text or (value or "").strip()


# --------------------------------------------------------------------------- built-in table
#: kind: "uri" (ShellExecute a protocol), "aumid" (shell:AppsFolder\...), "exe" (path),
#: "lnk" (Start-menu shortcut path), "url" (open in the default browser), "shell" (cmd /c).
BUILTIN: Dict[str, Dict[str, Any]] = {
    # --- Windows settings: ms-settings: pages are the only reliable way in ---
    "settings": {"display": "Settings", "kind": "uri", "target": "ms-settings:",
                 "aliases": {"settings", "windows settings", "windows 11 settings", "the settings", "settings app"},
                 "process": "SystemSettings.exe"},
    "display settings": {"display": "Display settings", "kind": "uri", "target": "ms-settings:display",
                         "aliases": {"display", "screen resolution", "brightness settings", "night light",
                                      "display settings", "monitors"}},
    "sound settings": {"display": "Sound settings", "kind": "uri", "target": "ms-settings:sound",
                       "aliases": {"sound", "audio settings", "volume mixer", "sound settings", "speakers",
                                    "microphone settings", "input device"}},
    "bluetooth settings": {"display": "Bluetooth & devices", "kind": "uri", "target": "ms-settings:bluetooth",
                           "aliases": {"bluetooth", "bluetooth settings", "devices", "pair device",
                                        "bluetooth and devices"}},
    "wifi settings": {"display": "Network & internet", "kind": "uri", "target": "ms-settings:network-wifi",
                     "aliases": {"wifi", "wi-fi", "network", "internet settings", "wifi settings",
                                  "network settings", "adapter settings", "change adapter options"}},
    "airplane mode": {"display": "Airplane mode", "kind": "uri", "target": "ms-settings:network-airplanemode"},
    "notification settings": {"display": "Notifications", "kind": "uri", "target": "ms-settings:notifications",
                              "aliases": {"notifications", "focus assist", "do not disturb", "dnd"}},
    "power settings": {"display": "Power & battery", "kind": "uri", "target": "ms-settings:powersleep",
                       "aliases": {"power", "battery settings", "sleep settings", "power options",
                                    "power plan", "power settings"}},
    "storage settings": {"display": "Storage", "kind": "uri", "target": "ms-settings:storagesense",
                         "aliases": {"storage", "disk space", "storage sense", "cleanup recommendation"}},
    "apps settings": {"display": "Installed apps", "kind": "uri", "target": "ms-settings:appsfeatures",
                      "aliases": {"installed apps", "add or remove programs", "uninstall a program",
                                   "apps and features", "default apps"}},
    "windows update": {"display": "Windows Update", "kind": "uri", "target": "ms-settings:windowsupdate",
                       "aliases": {"update", "updates", "windows update", "check for updates",
                                    "windows updater", "software update"}},
    "privacy settings": {"display": "Privacy & security", "kind": "uri", "target": "ms-settings:privacy",
                         "aliases": {"privacy", "security settings", "microphone permission",
                                      "camera permission"}},
    "windows security": {"display": "Windows Security", "kind": "uri", "target": "windowsdefender:",
                        "aliases": {"defender", "antivirus", "virus", "security", "firewall settings",
                                     "windows security"}, "process": "MSASCuiL.exe"},
    "account settings": {"display": "Your info", "kind": "uri", "target": "ms-settings:yourinfo",
                         "aliases": {"account", "accounts", "user accounts", "profile settings", "email accounts"}},
    "date time settings": {"display": "Date & time", "kind": "uri", "target": "ms-settings:dateandtime",
                           "aliases": {"date and time", "timezone", "time zone", "clock settings"}},
    "region settings": {"display": "Language & region", "kind": "uri", "target": "ms-settings:regionlanguage",
                        "aliases": {"language", "region", "keyboard layout", "date format"}},
    "mouse settings": {"display": "Mouse settings", "kind": "uri", "target": "ms-settings:mousetouchpad",
                       "aliases": {"mouse", "touchpad", "pointer settings", "scroll speed"}},
    "keyboard settings": {"display": "Typing settings", "kind": "uri", "target": "ms-settings:typing",
                          "aliases": {"keyboard", "typing", "sticky keys", "keyboard layout settings"}},
    "personalisation": {"display": "Personalisation", "kind": "uri", "target": "ms-settings:personalization",
                        "aliases": {"personalization", "background settings", "lock screen", "colours",
                                     "colors", "themes", "taskbar settings", "start menu settings"}},
    "wallpaper settings": {"display": "Background", "kind": "uri", "target": "ms-settings:personalization-background"},
    "multitasking": {"display": "Multitasking", "kind": "uri", "target": "ms-settings:multitasking",
                      "aliases": {"snap layouts", "snap assist", "alt tab", "task view settings"}},
    "troubleshoot": {"display": "Troubleshoot", "kind": "uri", "target": "ms-settings:troubleshoot",
                     "aliases": {"troubleshooting", "repair windows", "fix sound", "network reset"}},
    "recovery settings": {"display": "Recovery", "kind": "uri", "target": "ms-settings:recovery",
                          "aliases": {"recovery", "reset this pc", "advanced startup", "safe mode"}},
    "control panel": {"display": "Control Panel", "kind": "uri", "target": "control",
                      "aliases": {"control panel", "classic control panel", "old settings", "programs and features",
                                   "device manager", "power options", "sound recorder settings"}},
    "device manager": {"display": "Device Manager", "kind": "uri", "target": "devmgmt",
                       "aliases": {"device manager", "drivers", "driver update", "display adapters"}},
    "services": {"display": "Services", "kind": "uri", "target": "services",
                 "aliases": {"services", "services.msc", "background services"}},
    "registry editor": {"display": "Registry Editor", "kind": "uri", "target": "regedit",
                        "aliases": {"registry", "regedit", "reg edit"}},
    "event viewer": {"display": "Event Viewer", "kind": "uri", "target": "eventvwr",
                     "aliases": {"event viewer", "logs", "windows logs", "system logs"}},
    "task manager": {"display": "Task Manager", "kind": "uri", "target": "taskmgr",
                     "aliases": {"task manager", "taskmgr", "performance tab", "startup apps", "kill process panel"},
                     "process": "Taskmgr.exe", "paths": [r"%WINDIR%\System32\Taskmgr.exe"]},
    "run dialog": {"display": "Run", "kind": "hotkey", "target": "win+r",
                   "aliases": {"run", "run box", "run command", "cmd run"}},
    "file explorer": {"display": "File Explorer", "kind": "exe", "target": r"%WINDIR%\explorer.exe",
                      "args": "shell:MyComputerFolder", "aliases": {"explorer", "files", "file explorer", "this pc",
                                                                     "my computer", "folder", "documents folder"},
                      "process": "explorer.exe"},
    "downloads": {"display": "Downloads", "kind": "shell", "target": "downloads:",
                  "aliases": {"downloads", "my downloads", "download folder"}},
    "documents": {"display": "Documents", "kind": "shell", "target": "documents:",
                  "aliases": {"documents", "my documents"}},
    "recycle bin": {"display": "Recycle Bin", "kind": "shell", "target": "shell:RecycleBinFolder",
                    "aliases": {"recycle bin", "trash", "bin", "deleted files"}},
    "task view": {"display": "Task view", "kind": "hotkey", "target": "win+tab",
                  "aliases": {"task view", "all windows", "window switcher", "virtual desktop"}},
    "show desktop": {"display": "Desktop", "kind": "hotkey", "target": "win+d",
                     "aliases": {"show desktop", "minimize everything", "hide all windows"}},
    "snipping tool": {"display": "Snipping Tool", "kind": "uri", "target": "ms-screenclip:",
                     "aliases": {"snip", "snipping tool", "screenshot tool", "screen clip", "crop"},
                     "aumid": "Microsoft.ScreenSketch_8wekyb3d8bbwe!App"},
    "paint": {"display": "Paint", "kind": "uri", "target": "mspaint", "aliases": {"paint", "mspaint", "draw"},
              "process": "mspaint.exe", "paths": [r"%WINDIR%\System32\mspaint.exe"]},
    "notepad": {"display": "Notepad", "kind": "uri", "target": "notepad", "aliases": {"notepad", "text editor"},
                "process": "Notepad.exe", "paths": [r"%WINDIR%\System32\notepad.exe"]},
    "calculator": {"display": "Calculator", "kind": "uri", "target": "calc", "aliases": {"calculator", "calc"},
                   "process": "CalculatorApp.exe", "aumid": "Microsoft.WindowsCalculator_8wekyb3d8bbwe!App",
                   "paths": [r"%WINDIR%\System32\calc.exe"]},
    "cmd": {"display": "Command Prompt", "kind": "uri", "target": "cmd", "aliases": {"cmd", "command prompt",
                                                                                        "terminal classic"},
            "process": "cmd.exe", "paths": [r"%WINDIR%\System32\cmd.exe"]},
    "powershell": {"display": "PowerShell", "kind": "uri", "target": "powershell",
                   "aliases": {"powershell", "ps", "terminal blue"}, "process": "powershell.exe"},
    "terminal": {"display": "Terminal", "kind": "aumid", "aumid": "Microsoft.WindowsTerminal_8wekyb3d8bbwe!App",
                 "target": "Microsoft.WindowsTerminal_8wekyb3d8bbwe!App", "uri": "wt",
                 "aliases": {"terminal", "windows terminal", "console", "new terminal"},
                 "process": "WindowsTerminal.exe"},
    "settings sound mixer": {"display": "Volume mixer", "kind": "uri", "target": "ms-settings:appsvolume"},
    "clock": {"display": "Clock", "kind": "aumid", "target": "Microsoft.WindowsClock_8wekyb3d8bbwe!App",
              "aliases": {"clock", "alarm", "timer app", "stopwatch", "world clock"}},
    "photos": {"display": "Photos", "kind": "aumid", "target": "Microsoft.Windows.Photos_8wekyb3d8bbwe!App",
               "aliases": {"photos", "gallery", "image viewer", "picture app"}, "process": "Photos.exe"},
    "camera": {"display": "Camera", "kind": "aumid", "target": "Microsoft.WindowsCamera_8wekyb3d8bbwe!App",
               "aliases": {"camera", "webcam", "selfie"}},
    "store": {"display": "Microsoft Store", "kind": "aumid", "target": "Microsoft.WindowsStore_8wekyb3d8bbwe!App",
              "aliases": {"store", "microsoft store", "app store", "marketplace"}},
    "xbox": {"display": "Xbox", "kind": "aumid", "target": "Microsoft.GamingApp_8wekyb3d8bbwe!Microsoft.Xbox.App",
             "aliases": {"xbox", "xbox app", "game bar", "game pass"}, "uri": "ms-gamingoverlay:"},
    "media player": {"display": "Media Player", "kind": "aumid",
                     "target": "Microsoft.ZuneMusic_8wekyb3d8bbwe!Microsoft.ZuneMusic",
                     "aliases": {"media player", "groove", "windows media", "wmplayer", "music player"},
                     "process": "wmplayer.exe"},
    "voice recorder": {"display": "Voice Recorder", "kind": "aumid",
                       "target": "Microsoft.WindowsSoundRecorder_8wekyb3d8bbwe!App",
                       "aliases": {"voice recorder", "record voice", "sound recorder"}},
    "sticky notes": {"display": "Sticky Notes", "kind": "aumid",
                     "target": "Microsoft.MicrosoftStickyNotes_8wekyb3d8bbwe!App",
                     "aliases": {"sticky notes", "post it", "notes app"}},
    "to do": {"display": "To Do", "kind": "url", "target": "https://to-do.office.com/tasks",
              "aliases": {"to do", "todo", "checklist", "task list"}, "aumid": "Microsoft.ToDo_8wekyb3d8bbwe!App"},
    "snipaste": {"display": "Snipaste", "kind": "exe", "target": r"%LOCALAPPDATA%\Snipaste\Snipaste.exe",
                 "aliases": {"snipaste", "screen pin"}},
    "power toys": {"display": "PowerToys", "kind": "exe",
                   "target": r"%LOCALAPPDATA%\Microsoft\PowerToys\PowerToys.exe",
                   "aliases": {"powertoys", "power toys", "fancy zones", "always on top"},
                   "process": "PowerToys.exe"},
    # --- browsers ---
    "chrome": {"display": "Chrome", "kind": "exe", "aliases": {"chrome", "google chrome", "browser", "web browser"},
               "process": "chrome.exe", "url_arg": "--new-window {url}",
               "paths": [r"%PROGRAMFILES%\Google\Chrome\Application\chrome.exe",
                         r"%PROGRAMFILES(X86)%\Google\Chrome\Application\chrome.exe",
                         r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"]},
    "edge": {"display": "Edge", "kind": "uri", "target": "start msedge", "aliases": {"edge", "ms edge",
                                                                                       "microsoft edge", "ie"},
             "process": "msedge.exe", "url_arg": "--new-window {url}", "aumid": "Microsoft.MicrosoftEdge_8wekyb3d8bbwe!MicrosoftEdge",
             "paths": [r"%PROGRAMFILES(X86)%\Microsoft\Edge\Application\msedge.exe",
                       r"%PROGRAMFILES%\Microsoft\Edge\Application\msedge.exe"]},
    "firefox": {"display": "Firefox", "kind": "uri", "target": "start firefox", "aliases": {"firefox", "mozilla"},
                "process": "firefox.exe", "url_arg": "-new-window {url}",
                "paths": [r"%PROGRAMFILES%\Mozilla Firefox\firefox.exe", r"%PROGRAMFILES(X86)%\Mozilla Firefox\firefox.exe"]},
    "brave": {"display": "Brave", "kind": "uri", "target": "start brave", "aliases": {"brave", "brave browser"},
              "process": "brave.exe", "url_arg": "--new-window {url}",
              "paths": [r"%PROGRAMFILES%\BraveSoftware\Brave-Browser\Application\brave.exe"]},
    "opera": {"display": "Opera", "kind": "uri", "target": "start opera", "aliases": {"opera", "opera gx"},
              "process": "opera.exe", "paths": [r"%LOCALAPPDATA%\Programs\Opera\opera.exe"]},
    "vivaldi": {"display": "Vivaldi", "kind": "uri", "target": "start vivaldi", "aliases": {"vivaldi"},
                "process": "vivaldi.exe"},
    # --- communication ---
    "teams": {"display": "Microsoft Teams", "kind": "aumid", "target": "MSTeams_8wekyb3d8bbwe!MicrosoftTeams",
              "aliases": {"teams", "microsoft teams", "ms teams", "new teams", "team", "meetings"},
              "uri": "msteams:", "process": "msTeams.exe",
              "paths": [r"%LOCALAPPDATA%\Microsoft\WindowsApps\msTeams.exe",
                        r"%ProgramFiles%\Microsoft\Teams\current\Teams.exe",
                        r"%LOCALAPPDATA%\Microsoft\Teams\current\Teams.exe"]},
    "outlook": {"display": "Outlook", "kind": "exe", "aliases": {"outlook", "mail", "email", "inbox"},
                "process": "OUTLOOK.EXE", "uri": "outlook:",
                "paths": [r"%PROGRAMFILES%\Microsoft Office\root\Office16\OUTLOOK.EXE",
                          r"%PROGRAMFILES(X86)%\Microsoft Office\root\Office16\OUTLOOK.EXE"],
                "aumid": "Microsoft.OutlookForWindows_8wekyb3d8bbwe!Microsoft.OutlookforWindows"},
    "slack": {"display": "Slack", "kind": "exe", "aliases": {"slack"}, "uri": "slack://", "process": "slack.exe",
              "paths": [r"%LOCALAPPDATA%\slack\Update.exe", r"%LOCALAPPDATA%\slack\app-\Slack.exe"]},
    "discord": {"display": "Discord", "kind": "uri", "target": "discord://discordapp.com/client",
                "aliases": {"discord", "canary", "discord app"}, "process": "discord.exe",
                "paths": [r"%LOCALAPPDATA%\Discord\Discord.exe", r"%LOCALAPPDATA%\Programs\Discord\Discord.exe"]},
    "telegram": {"display": "Telegram", "kind": "uri", "target": "tg://", "aliases": {"telegram", "tg desktop"},
                 "process": "Telegram.exe", "paths": [r"%APPDATA%\Telegram Desktop\Telegram.exe"]},
    "whatsapp": {"display": "WhatsApp", "kind": "url", "target": "https://web.whatsapp.com",
                 "aliases": {"whatsapp", "wa"}, "aumid": "5319275A.WhatsAppDesktop_8wekyb3d8bbwe!App",
                 "process": "WhatsApp.exe"},
    "zoom": {"display": "Zoom", "kind": "exe", "aliases": {"zoom", "meetings app"}, "process": "Zoom.exe",
             "uri": "zoommtg:", "paths": [r"%LOCALAPPDATA%\Programs\Zoom\bin\Zoom.exe",
                                          r"%PROGRAMFILES(X86)%\Zoom\bin\Zoom.exe"]},
    "skype": {"display": "Skype", "kind": "uri", "target": "skype:", "aliases": {"skype"}, "process": "skype.exe",
              "aumid": "Microsoft.SkypeApp_kzf8qxf38zg5c!App"},
    "signal": {"display": "Signal", "kind": "exe", "aliases": {"signal"}, "process": "Signal.exe",
               "paths": [r"%LOCALAPPDATA%\Programs\signal-desktop\Signal.exe"]},
    # --- office / study ---
    "word": {"display": "Word", "kind": "exe", "aliases": {"word", "ms word", "microsoft word", "docx"},
             "process": "WINWORD.EXE",
             "paths": [r"%PROGRAMFILES%\Microsoft Office\root\Office16\WINWORD.EXE",
                       r"%PROGRAMFILES(X86)%\Microsoft Office\root\Office16\WINWORD.EXE"]},
    "excel": {"display": "Excel", "kind": "exe", "aliases": {"excel", "ms excel", "spreadsheet", "xlsx"},
              "process": "EXCEL.EXE",
              "paths": [r"%PROGRAMFILES%\Microsoft Office\root\Office16\EXCEL.EXE",
                        r"%PROGRAMFILES(X86)%\Microsoft Office\root\Office16\EXCEL.EXE"]},
    "powerpoint": {"display": "PowerPoint", "kind": "exe", "aliases": {"powerpoint", "slides", "ppt", "presentation"},
                   "process": "POWERPNT.EXE",
                   "paths": [r"%PROGRAMFILES%\Microsoft Office\root\Office16\POWERPNT.EXE",
                             r"%PROGRAMFILES(X86)%\Microsoft Office\root\Office16\POWERPNT.EXE"]},
    "onenote": {"display": "OneNote", "kind": "uri", "target": "onenote:", "aliases": {"onenote", "one note"},
                "process": "ONENOTE.EXE"},
    "acrobat": {"display": "Acrobat", "kind": "exe", "aliases": {"acrobat", "pdf reader", "reader"},
                "process": "Acrobat.exe",
                "paths": [r"%PROGRAMFILES%\Adobe\Acrobat DC\Acrobat\Acrobat.exe",
                           r"%PROGRAMFILES(X86)%\Adobe\Acrobat DC\Acrobat\Acrobat.exe"]},
    "foxit": {"display": "Foxit Reader", "kind": "exe", "aliases": {"foxit", "foxit reader"},
              "process": "FoxitPDFReader.exe",
              "paths": [r"%PROGRAMFILES%\Foxit Software\Foxit PDF Reader\FoxitPDFReader.exe"]},
    "libreoffice": {"display": "LibreOffice", "kind": "exe", "aliases": {"libreoffice", "openoffice", "writer"},
                    "process": "soffice.exe", "paths": [r"%PROGRAMFILES%\LibreOffice\program\soffice.exe"]},
    "latex": {"display": "TeXstudio", "kind": "exe", "aliases": {"latex", "texstudio", "overleaf local"},
              "process": "texstudio.exe", "paths": [r"%PROGRAMFILES%\TeXstudio\texstudio.exe"]},
    "jupyter": {"display": "Jupyter", "kind": "shell", "target": "cmd /c start jupyter lab",
                "aliases": {"jupyter", "notebook", "jupyter lab"}},
    "obsidian": {"display": "Obsidian", "kind": "uri", "target": "obsidian://", "aliases": {"obsidian"},
                 "process": "Obsidian.exe", "paths": [r"%LOCALAPPDATA%\Programs\obsidian\Obsidian.exe"]},
    "notion": {"display": "Notion", "kind": "uri", "target": "notion://", "aliases": {"notion"},
               "process": "Notion.exe", "paths": [r"%LOCALAPPDATA%\Programs\Notion\Notion.exe"]},
    "anki": {"display": "Anki", "kind": "exe", "aliases": {"anki", "flashcards"}, "process": "anki.exe",
             "paths": [r"%PROGRAMFILES%\Anki\anki.exe", r"%LOCALAPPDATA%\Programs\Anki\anki.exe"]},
    "matlab": {"display": "MATLAB", "kind": "exe", "aliases": {"matlab", "octave"}, "process": "MATLAB.exe",
               "paths": [r"%PROGRAMFILES%\MATLAB\R2024a\bin\matlab.exe"]},
    # --- dev ---
    "code": {"display": "VS Code", "kind": "exe", "aliases": {"code", "vs code", "vscode", "visual studio code"},
             "process": "Code.exe", "uri": "vscode://file/{cwd}",
             "paths": [r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe",
                       r"%PROGRAMFILES%\Microsoft VS Code\Code.exe"]},
    "visual studio": {"display": "Visual Studio", "kind": "exe", "aliases": {"visual studio", "vs 2022", "studio"},
                      "process": "devenv.exe",
                      "paths": [r"%PROGRAMFILES%\Microsoft Visual Studio\2022\Community\Common7\IDE\devenv.exe",
                                r"%PROGRAMFILES%\Microsoft Visual Studio\2022\Professional\Common7\IDE\devenv.exe"]},
    "git bash": {"display": "Git Bash", "kind": "exe", "aliases": {"git bash", "bash", "git shell"},
                 "process": "bash.exe", "paths": [r"%PROGRAMFILES%\Git\git-bash.exe",
                                                   r"%PROGRAMFILES(X86)%\Git\git-bash.exe"]},
    "github desktop": {"display": "GitHub Desktop", "kind": "uri", "target": "github-windows://",
                       "aliases": {"github desktop", "gh desktop"}, "process": "GitHubDesktop.exe"},
    "docker": {"display": "Docker Desktop", "kind": "exe", "aliases": {"docker", "docker desktop"},
               "process": "Docker Desktop.exe",
               "paths": [r"%PROGRAMFILES%\Docker\Docker\Docker Desktop.exe"]},
    "wsl": {"display": "Ubuntu (WSL)", "kind": "uri", "target": "start wsl", "aliases": {"wsl", "ubuntu", "linux"}},
    "python": {"display": "Python", "kind": "uri", "target": "start idle", "aliases": {"python", "idle",
                                                                                         "python shell"}},
    "node": {"display": "Node REPL", "kind": "shell", "target": "cmd /c start cmd /k node",
             "aliases": {"node", "nodejs", "npm"}},
    "blender": {"display": "Blender", "kind": "exe", "aliases": {"blender", "3d"}, "process": "blender.exe",
                "paths": [r"%PROGRAMFILES%\Blender Foundation\Blender 4.2\blender.exe"]},
    "unity": {"display": "Unity Hub", "kind": "exe", "aliases": {"unity", "unity hub"}, "process": "Unity Hub.exe",
              "paths": [r"%PROGRAMFILES%\Unity Hub\Unity Hub.exe"]},
    # --- media / games ---
    "spotify": {"display": "Spotify", "kind": "uri", "target": "spotify:", "aliases": {"spotify", "music app"},
                "process": "Spotify.exe", "paths": [r"%APPDATA%\Spotify\Spotify.exe",
                                                     r"%PROGRAMFILES%\Spotify\Spotify.exe"]},
    "vlc": {"display": "VLC", "kind": "exe", "aliases": {"vlc", "vlc media player", "player"},
            "process": "vlc.exe", "url_arg": "{url}",
            "paths": [r"%PROGRAMFILES%\VideoLAN\VLC\vlc.exe", r"%PROGRAMFILES(X86)%\VideoLAN\VLC\vlc.exe"]},
    "youtube music": {"display": "YouTube Music", "kind": "url", "target": "https://music.youtube.com",
                      "aliases": {"youtube music", "yt music"}},
    "steam": {"display": "Steam", "kind": "uri", "target": "steam://open/main",
              "aliases": {"steam", "big picture", "steam library"}, "process": "steam.exe",
              "paths": [r"C:\Program Files (x86)\Steam\Steam.exe", r"C:\Program Files\Steam\Steam.exe",
                        r"D:\Steam\Steam.exe", r"E:\Steam\Steam.exe"]},
    "epic": {"display": "Epic Games", "kind": "uri", "target": "com.epicgames.launcher://",
             "aliases": {"epic", "epic games", "epic launcher"}, "process": "EpicGamesLauncher.exe"},
    "battle.net": {"display": "Battle.net", "kind": "uri", "target": "battlenet://",
                   "aliases": {"battle net", "battlenet", "wow launcher"}, "process": "Battle.net.exe"},
    "riot": {"display": "Riot Client", "kind": "uri", "target": "riotclient://",
             "aliases": {"riot", "riot client", "valorant", "league", "league of legends"},
             "process": "RiotClientServices.exe", "paths": [r"%LOCALAPPDATA%\Riot Games\Riot Client\RiotClientServices.exe"]},
    "ea app": {"display": "EA app", "kind": "uri", "target": "easthreads://", "aliases": {"ea app", "origin",
                                                                                            "ea"}},
    "obs": {"display": "OBS Studio", "kind": "exe", "aliases": {"obs", "streaming", "record screen"},
            "process": "obs64.exe", "paths": [r"%PROGRAMFILES%\obs-studio\bin\64bit\obs64.exe"]},
    "sharex": {"display": "ShareX", "kind": "uri", "target": "sharex://", "aliases": {"sharex", "screen record"},
               "process": "ShareX.exe"},
    "anydesk": {"display": "AnyDesk", "kind": "uri", "target": "anydesk:", "aliases": {"anydesk", "remote desktop"},
                "process": "AnyDesk.exe"},
    "teamviewer": {"display": "TeamViewer", "kind": "uri", "target": "teamviewer:", "aliases": {"teamviewer"},
                   "process": "TeamViewer.exe"},
    "nvidia": {"display": "NVIDIA App", "kind": "exe", "aliases": {"nvidia", "geforce", "nvidia control panel", "graphics settings"},
               "process": "NVIDIA App.exe",
               "paths": [r"%LOCALAPPDATA%\NVIDIA Corporation\NVIDIA App\CEF\NVIDIA App.exe",
                         r"%PROGRAMFILES%\NVIDIA Corporation\NVIDIA Control Panel\nvcplui.exe"]},
    "7zip": {"display": "7-Zip", "kind": "exe", "aliases": {"7zip", "7-zip", "archive manager"},
             "process": "7zFM.exe", "paths": [r"%PROGRAMFILES%\7-Zip\7zFM.exe"]},
    "winrar": {"display": "WinRAR", "kind": "exe", "aliases": {"winrar", "rar"}, "process": "WinRAR.exe",
               "paths": [r"%PROGRAMFILES%\WinRAR\WinRAR.exe"]},
    "ccleaner": {"display": "CCleaner", "kind": "exe", "aliases": {"ccleaner", "cleaner"},
                 "process": "ccleaner64.exe", "paths": [r"%PROGRAMFILES%\CCleaner\ccleaner64.exe"]},
    # --- web services that people call "apps" (open in the browser instead) ---
    "gmail": {"display": "Gmail", "kind": "url", "target": "https://mail.google.com", "aliases": {"gmail", "my mail"}},
    "youtube": {"display": "YouTube", "kind": "url", "target": "https://www.youtube.com", "aliases": {"youtube", "yt"}},
    "chatgpt": {"display": "ChatGPT", "kind": "url", "target": "https://chat.openai.com",
                "aliases": {"chatgpt", "gpt", "openai chat"}},
    "gemini web": {"display": "Gemini", "kind": "url", "target": "https://gemini.google.com",
                   "aliases": {"gemini web", "google gemini"}},
    "github": {"display": "GitHub", "kind": "url", "target": "https://github.com", "aliases": {"github", "gh"}},
    "google calendar": {"display": "Calendar", "kind": "url", "target": "https://calendar.google.com",
                        "aliases": {"calendar", "google calendar", "agenda"}},
    "google drive": {"display": "Drive", "kind": "url", "target": "https://drive.google.com",
                     "aliases": {"drive", "google drive"}},
    "maps": {"display": "Maps", "kind": "url", "target": "https://www.google.com/maps",
             "aliases": {"maps", "google maps", "directions"}},
}

#: ``ms-settings:`` sub-pages people ask for in everyday words.
SETTINGS_PAGES: Dict[str, str] = {
    "display": "ms-settings:display", "resolution": "ms-settings:display", "brightness": "ms-settings:display",
    "night light": "ms-settings:display", "bluetooth": "ms-settings:bluetooth",
    "devices": "ms-settings:bluetooth", "printer": "ms-settings:printers", "printers": "ms-settings:printers",
    "wifi": "ms-settings:network-wifi", "wi-fi": "ms-settings:network", "network": "ms-settings:network",
    "proxy": "ms-settings:network-proxy", "vpn": "ms-settings:network-vpn",
    "sound": "ms-settings:sound", "audio": "ms-settings:sound", "microphone": "ms-settings:sound",
    "notifications": "ms-settings:notifications", "focus": "ms-settings:notifications",
    "do not disturb": "ms-settings:notifications", "battery": "ms-settings:batterysaver",
    "power": "ms-settings:powersleep", "sleep": "ms-settings:powersleep",
    "storage": "ms-settings:storagesense", "disk": "ms-settings:storagesense",
    "apps": "ms-settings:appsfeatures", "uninstall": "ms-settings:appsfeatures",
    "startup": "ms-settings:startupapps", "default apps": "ms-settings:defaultapps",
    "update": "ms-settings:windowsupdate", "updates": "ms-settings:windowsupdate",
    "privacy": "ms-settings:privacy", "security": "ms-settings:privacy",
    "location": "ms-settings:privacy-location", "camera permission": "ms-settings:privacy-webcam",
    "account": "ms-settings:emailandaccounts", "email": "ms-settings:emailandaccounts",
    "sign in options": "ms-settings:sign-in-options", "language": "ms-settings:regionlanguage",
    "region": "ms-settings:regionlanguage", "keyboard": "ms-settings:typing", "mouse": "ms-settings:mousetouchpad",
    "touchpad": "ms-settings:mousetouchpad", "background": "ms-settings:personalization-background",
    "wallpaper": "ms-settings:personalization-background", "lock screen": "ms-settings:lockscreen",
    "colors": "ms-settings:colors", "colours": "ms-settings:colors", "themes": "ms-settings:themes",
    "taskbar": "ms-settings:taskbar", "start": "ms-settings:startupapps", "multitasking": "ms-settings:multitasking",
    "snap layouts": "ms-settings:multitasking", "date": "ms-settings:dateandtime", "time": "ms-settings:dateandtime",
    "timezone": "ms-settings:dateandtime", "troubleshoot": "ms-settings:troubleshoot",
    "recovery": "ms-settings:recovery", "reset this pc": "ms-settings:recovery", "search": "ms-settings:cortana",
    "windows ink": "ms-settings:pen", "pen": "ms-settings:pen", "autoplay": "ms-settings:autoplay",
    "remote desktop": "ms-settings:remotedesktop", "for developers": "ms-settings:developers",
    "developer mode": "ms-settings:developers", "hyper-v": "ms-settings:optionalfeatures",
    "optional features": "ms-settings:optionalfeatures", "language pack": "ms-settings:regionlanguage",
    "ease of access": "ms-settings:easeofaccess", "accessibility": "ms-settings:easeofaccess",
    "night light settings": "ms-settings:displaysleep", "game mode": "ms-settings:gaming-gamebar",
    "captures": "ms-settings:gaming-captures", "delivery optimisation": "ms-settings:deliveroptimization",
    "time and language": "ms-settings:regionlanguage", "sync your settings": "ms-settings:sync",
    "windows security": "windowsdefender:", "defender": "windowsdefender:", "firewall": "ms-settings:network",
    "about": "ms-settings:about", "system info": "ms-settings:about", "rename this pc": "ms-settings:about",
}


@dataclass
class AppRef:
    """One resolved launch target, with the evidence of how it was found."""

    key: str
    display: str
    kind: str
    target: str
    args: str = ""
    process: str = ""
    source: str = ""
    url_arg: str = ""
    aumid: str = ""
    fallback: str = ""            # protocol or exe to try when `target` is refused
    corrected_from: str = ""       # set when the spelling was fixed ("crome" -> Chrome)
    candidates: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "display": self.display, "kind": self.kind, "target": self.target,
                "process": self.process, "found_via": self.source}


def _builtins_alias_index() -> Dict[str, str]:
    index: Dict[str, str] = {}
    for key, meta in BUILTIN.items():
        for alias in set(meta.get("aliases") or set()) | {key, _slug(key)}:
            alias = _norm(str(alias))
            if alias:
                index.setdefault(alias, key)
        index.setdefault(_norm(meta.get("display", key)), key)
    return index


_ALIAS_INDEX = _builtins_alias_index()


def custom_apps() -> Dict[str, str]:
    """``CUSTOM_APPS=name=C:\\path;other=D:\\x.exe`` from .env, always honoured first."""
    raw = (os.environ.get("CUSTOM_APPS") or getattr(SETTINGS, "custom_apps", "") or "").strip()
    out: Dict[str, str] = {}
    for chunk in re.split(r"[;\n]+", raw):
        if "=" not in chunk:
            continue
        name, _, path = chunk.partition("=")
        name, path = name.strip(), path.strip().strip('"')
        if name and path:
            out[_slug(name)] = path
    return out


def start_menu_index() -> Dict[str, str]:
    """``{slug: shortcut path}`` over the real Start Menu, so *any* installed app resolves."""
    def build() -> Dict[str, str]:
        out: Dict[str, str] = {}
        for entry in winops.start_menu_entries():
            name = str(entry.get("name", ""))
            if not name:
                continue
            out.setdefault(_slug(name), str(entry.get("target", "")))
            # "Microsoft Teams (2)" and "Teams" should both find the same thing
            cleaned = _slug(re.sub(r"\(\d+\)$|\[.*?\]$|- \d.*$", "", name).strip())
            if cleaned and cleaned != _slug(name):
                out.setdefault(cleaned, str(entry.get("target", "")))
        return out

    return winops._cached("start_index", build, ttl=max(60.0, SETTINGS.app_index_ttl))


def uwp_index() -> Dict[str, str]:
    def build() -> Dict[str, str]:
        return {_slug(name): appid for name, appid in (winops.uwp_apps() or {}).items() if appid}

    return winops._cached("uwp_index", build, ttl=max(60.0, SETTINGS.app_index_ttl))


def registry_index() -> Dict[str, str]:
    def build() -> Dict[str, str]:
        names = {slug: path for slug, path in (winops.app_paths() or {}).items()}
        for display, exe in (winops.installed_exes() or {}).items():
            names.setdefault(_slug(display), exe)
            short = _slug(re.sub(r"\(.*?\)$|\b(20\d\d|64-bit|32-bit|version \d+.*?)\b", "", display).strip())
            if short and short != _slug(display):
                names.setdefault(short, exe)
        return names

    return winops._cached("registry_index", build, ttl=max(60.0, SETTINGS.app_index_ttl))


def known_names() -> List[str]:
    names = {str(meta.get("display") or key) for key, meta in BUILTIN.items()}
    names |= set(uwp_index()) | set(start_menu_index()) | set(registry_index()) | set(custom_apps())
    return sorted(n for n in names if n)


def _best(slug: str, index: Dict[str, str], query_tokens: set) -> Tuple[str, float]:
    """Pick the closest key in ``index`` for ``slug``; returns (key, score)."""
    if slug in index:
        return slug, 1.0
    query = set(re.split(r"-", slug)) - _STOPWORDS
    best_key, best_score = "", 0.0
    for key in index:
        if not key:
            continue
        if slug and (slug in key or key in slug) and min(len(slug), len(key)) >= 4:
            score = 0.86 + 0.1 * (len(set(re.split(r"-", key)) & query) / max(len(query), 1))
            if score > best_score:
                best_key, best_score = key, score
            continue
        tokens = set(re.split(r"-", key)) - _STOPWORDS
        if not tokens or not query:
            continue
        overlap = len(tokens & query) / len(tokens | query)
        if tokens <= query:
            overlap = 0.7 + 0.3 * (len(tokens) / max(len(query), 1))
        if overlap > best_score:
            best_key, best_score = key, overlap
    if best_key and best_score < 0.34:
        return "", 0.0
    return best_key, best_score


def _expand(path: str) -> str:
    return os.path.expandvars(os.path.expanduser((path or "").strip().strip('"')))


def _ref_from_builtin(key: str, query_tokens: Optional[set] = None) -> Optional[AppRef]:
    """Turn a BUILTIN row into a launchable :class:`AppRef`, preferring a path that exists.

    Order matters for honesty: a real ``.exe`` on disk beats a protocol handler, which beats
    ``ShellExecute`` on a bare program name (Windows resolves those through App Paths, which
    is exactly how Word/Chrome/Teams launch from the Run box).  Entries whose ``target`` is
    written ``start x`` are shell forms, so the leading ``start`` is dropped and Windows does
    the resolving - the old code handed ``start msedge`` to CreateProcess and failed.
    """
    meta = BUILTIN.get(key) or {}
    if not meta:
        return None
    if meta.get("kind") == "hotkey":
        # Win+D / Win+R are chords, not launchable names - no path exists for them.
        return AppRef(key, str(meta.get("display") or key).title(), "hotkey", str(meta.get("target") or ""),
                      source="builtin")
    display = str(meta.get("display") or key).title()
    process = str(meta.get("process") or "")
    aumid = str(meta.get("aumid") or "")
    target = str(meta.get("target") or "")
    uri = str(meta.get("uri") or "") or ("" if target.startswith(("http", "ms-", "shell:")) else target)
    args = str(meta.get("args") or "")
    for candidate in ([target] if target else []) + [str(p) for p in meta.get("paths", [])]:
        path = _expand(candidate)
        if path.lower().endswith(".exe") and Path(path).is_file():
            return AppRef(key, display, "exe", path, args=args, process=process or Path(path).name,
                          source="builtin", url_arg=str(meta.get("url_arg") or ""), aumid=aumid,
                          fallback=uri or aumid)
    if meta.get("kind") == "url" and target.startswith("http"):
        return AppRef(key, display, "url", target, source="builtin", fallback=uri)
    if aumid or "!" in target:
        return AppRef(key, display, "aumid", target if "!" in target else aumid, args=args, process=process,
                      source="builtin", aumid=aumid or target, fallback=uri)
    if uri:
        clean = re.sub(r"^start\s+", "", uri, flags=re.I)
        if clean != uri:
            target = clean
        return AppRef(key, display, "uri", target or clean, args=args, process=process, source="builtin",
                      url_arg=str(meta.get("url_arg") or ""), aumid=aumid, fallback=process)
    if target.startswith(("ms-settings:", "shell:", "http")):
        return AppRef(key, display, "uri", target, args=args, process=process, source="builtin",
                      aumid=aumid, fallback=uri)
    if process:
        return AppRef(key, display, "uri", process, args=args, process=process, source="builtin",
                      aumid=aumid, fallback=target)
    return AppRef(key, display, "uri", target or display, args=args, process=process, source="builtin")


def settings_page(name: str) -> str:
    """Map everyday words onto a ``ms-settings:`` page (empty when it is not one)."""
    text = _norm(strip_verbs(name))
    if not text:
        return ""
    if text in SETTINGS_PAGES:
        return SETTINGS_PAGES[text]
    for words in sorted(SETTINGS_PAGES, key=len, reverse=True):
        if words in text and (len(words) > 4 or text.endswith(words)):
            return SETTINGS_PAGES[words]
    return ""


def resolve(name: str, allow_web: bool = True) -> Optional[AppRef]:
    """Find the best launch target for a spoken app name.  ``None`` = no idea."""
    raw = (name or "").strip()
    if not raw:
        return None
    cleaned = strip_verbs(raw)
    slug = _slug(cleaned)
    tokens = _tokens(cleaned)

    # 0. an absolute path or a protocol the user spelled out
    if re.match(r"^[a-z][a-z0-9+.\-]*://", cleaned, re.I) or cleaned.lower().startswith("ms-") \
            or cleaned.lower().startswith("shell:"):
        return AppRef(slug or "uri", Path(cleaned).name or cleaned, "uri", cleaned, source="literal")
    if Path(os.path.expandvars(cleaned)).is_file():
        return AppRef(slug, Path(cleaned).stem, "exe", os.path.expandvars(cleaned), source="path")

    # 1. a settings page asked for in plain words ("open bluetooth settings")
    if re.search(r"settings?|panel|permissions?|options?$", _norm(cleaned)) or slug in SETTINGS_PAGES:
        page = settings_page(cleaned)
        if page:
            label = cleaned.title()
            return AppRef(slug, label, "uri", page, source="ms-settings")

    # 2. the user's own override wins over the built-in catalogue - it is how they teach
    #    JARVIS a program we have never heard of, and it must not be second-guessed.
    #
    custom = custom_apps()
    if slug in custom:
        return AppRef(slug, cleaned.title(), "exe", custom[slug], source="CUSTOM_APPS")

    # 3. built-in catalogue
    alias_key = _ALIAS_INDEX.get(_norm(cleaned)) or _ALIAS_INDEX.get(slug.replace("-", " "))
    if not alias_key and tokens:
        best, best_score = "", 0.0
        for alias, key in _ALIAS_INDEX.items():
            alias_tokens = _tokens(alias)
            if not alias_tokens:
                continue
            score = (len(alias_tokens & tokens) / len(alias_tokens | tokens) if alias_tokens
                     else 0.0)
            if alias_tokens <= tokens:
                score = 0.75 + 0.25 * (len(alias_tokens) / max(len(tokens), 1))
            if score > best_score:
                best, best_score = key, score
        alias_key = best if best_score >= 0.4 else ""
    if alias_key and alias_key in BUILTIN:
        ref = _ref_from_builtin(alias_key, tokens)
        if ref is not None and not (ref.kind == "url" and not allow_web):
            return ref

    # 4. Start Menu  ->  5. UWP  ->  6/7. registry  (fuzzy-matched, best score wins)
    contenders = [("startmenu", start_menu_index()), ("uwp", uwp_index()), ("registry", registry_index())]
    scored: List[Tuple[float, str, str, str]] = []
    for source, index in contenders:
        key, score = _best(slug, index, tokens)
        if key and score:
            scored.append((score, key, index[key], source))
    if scored:
        scored.sort(key=lambda row: -row[0])
        score, key, target, source = scored[0]
        display = key.replace("-", " ").title()
        if source == "uwp":
            return AppRef(key, display, "aumid", target, source="Get-StartApps",
                          candidates=[s[1] for s in scored[:4]])
        kind = "lnk" if target.lower().endswith((".lnk", ".appref-ms")) else "exe"
        return AppRef(key, display, kind, target, source=source, candidates=[s[1] for s in scored[:4]])

    # 8. last resort: PATH, then let Windows itself try (Start-menu style search)
    found = shutil.which(cleaned) or shutil.which(f"{slug}.exe") or shutil.which(slug.replace("-", "_"))
    if found:
        return AppRef(slug, Path(found).stem.title(), "exe", found, source="PATH", args="")
    # A confident typo still gets served: "open crome" should not be an error.
    near = difflib.get_close_matches(slug, list(BUILTIN), n=1, cutoff=0.86)
    if near:
        ref = _ref_from_builtin(near[0], tokens)
        if ref is not None:
            ref.corrected_from = cleaned
            return ref
    return None


def suggest(name: str, limit: int = 4) -> List[str]:
    """Closest real app names, for "I don't know X - did you mean Y?"."""
    slug = _slug(strip_verbs(name))
    pool = [_slug(n) for n in known_names()]
    near = [n for n in difflib.get_close_matches(slug, pool, n=limit * 3, cutoff=0.6)]
    tokens = set(re.split(r"-", slug))
    extra = [key for key in pool if key and tokens & set(re.split(r"-", key))]
    ordered: List[str] = []
    for key in near + extra:
        label = key.replace("-", " ").title()
        if label not in ordered:
            ordered.append(label)
        if len(ordered) >= limit:
            break
    return ordered


def _launch_once(name: str, url: str = "", args: str = "", wait_seconds: float = -1) -> Dict[str, Any]:
    """Open any installed app (or a web service) by the name the user actually said."""
    ref = resolve(name)
    if ref is None:
        hints = suggest(name)
        message = (f"I don't know how to open {name.strip() or 'that'} on this machine. "
                   + (f"Did you mean {', '.join(hints)}? " if hints else ""))
        message += ("If it is installed but I missed it, add "
                    f"CUSTOM_APPS={_slug(name)}=C:\\path\\to\\app.exe to .env.")
        return {"ok": False, "message": message, "app": _slug(name), "did_you_mean": hints,
                "known": known_names()[:24]}

    args = args or ref.args or ""
    ref.target = str(ref.target).replace("{cwd}", str(config.ROOT)).replace("{home}", str(Path.home()))
    if url and ref.url_arg:
        extra_args = ref.url_arg.replace("{url}", url).replace("{cwd}", str(config.ROOT))
        args = (args + " " + extra_args).strip()
    elif url and not ref.url_arg and ref.kind == "url":
        ref.target = url if re.match(r"^https?://", url) else ref.target

    note = f" (I read “{ref.corrected_from}” as {ref.display})" if ref.corrected_from else ""

    if ref.kind == "hotkey":
        pressed = winops.hotkey(ref.target)
        return {**pressed, "app": ref.key, "display": ref.display, "method": "hotkey", "note": note}
    if ref.kind == "shell":
        cmd = ["cmd", "/c"] + ref.target.split() if winops.IS_WINDOWS else ["sh", "-c", ref.target.replace("start ", "xdg-open ")]
        out = winops._run_detached(cmd, ref.display)
        return {**out, "app": ref.key, "display": ref.display, "method": "shell"}
    if ref.kind == "url":
        opened = winops.shell_execute(ref.target)
        return {**opened, "app": ref.key, "display": ref.display, "url": ref.target,
                "message": f"Opening {ref.display} at {ref.target}." if opened.get("ok")
                else opened.get("message", "Could not open the browser.")}
    if ref.kind == "aumid":
        target = ref.aumid or (ref.target if "!" in ref.target else "")
        if not target:
            return {"ok": False, "message": f"{ref.display} has no registered package id on this machine.",
                    "app": ref.key, "did_you_mean": suggest(name)}
        opened = winops.shell_execute("explorer.exe", f"shell:AppsFolder\\{target}")
        if not opened.get("ok") and ref.fallback:
            opened = winops.shell_execute(re.sub(r"^start\s+", "", ref.fallback, flags=re.I), args)
        if not opened.get("ok") and ref.process:
            opened = winops.launch_exe(ref.process, args)
        return {**opened, "app": ref.key, "display": ref.display, "method": "appsfolder", "aumid": target,
                "message": f"{ref.display} is starting." if opened.get("ok") else opened.get("message", "")}
    if ref.kind in ("exe", "lnk"):
        if url and ref.kind == "exe" and not ref.url_arg and re.match(r"^https?://", url):
            opened = winops.shell_execute(url)
            return {**opened, "app": ref.key, "display": ref.display, "method": "url-fallback"}
        out = winops.launch_exe(ref.target, args)
        if not out.get("ok"):
            out = winops.shell_execute(ref.target, args)
        return {**out, "app": ref.key, "display": ref.display, "method": ref.kind, "exe": ref.target,
                "message": f"{ref.display} is starting." if out.get("ok") else out.get("message", "")}
    # uri / shell-execute of a bare program name - what the Run box does
    out = winops.shell_execute(ref.target, args)
    if not out.get("ok") and ref.aumid:
        out = winops.shell_execute("explorer.exe", f"shell:AppsFolder\\{ref.aumid}")
    if not out.get("ok") and ref.fallback:
        out = winops.launch_exe(ref.fallback, args)
    return {**out, "app": ref.key, "display": ref.display, "method": "uri",
            "message": f"{ref.display} is opening." if out.get("ok") else out.get("message", "")}


def _verify(result: Dict[str, Any], ref: AppRef, wait_seconds: float) -> Dict[str, Any]:
    """Wait briefly for the app's window, so "opened" is a fact and not a hope."""
    if not result.get("ok") or not winops.IS_WINDOWS:
        return result
    seconds = SETTINGS.app_wait_seconds if wait_seconds < 0 else wait_seconds
    if seconds <= 0:
        return result
    seen = wait_for_window(ref.process or (Path(ref.target).stem if ref.target.lower().endswith(".exe") else ""),
                           ref.display, seconds)
    result["window_seen"] = bool(seen.get("visible"))
    if seen.get("visible"):
        result["message"] = f"{ref.display} is up."
    elif ref.kind in ("uri", "shell"):
        # ShellExecute returned success but no window yet: many apps are slow or single-instance
        result["message"] = (f"{ref.display} is starting - no window yet, so if nothing appears, "
                             "tell me and I'll look for it.")
    if ref.corrected_from:
        result["note"] = f"read as {ref.display}"
        result["message"] = (f"{ref.display} is opening (I read “{ref.corrected_from}” as that).")
    return result


def launch(name: str, url: str = "", args: str = "", wait_seconds: float = -1) -> Dict[str, Any]:
    """Public entry: resolve + start, then confirm a window really appeared."""
    result = _launch_once(name, url=url, args=args, wait_seconds=wait_seconds)
    ref = resolve(name)
    if ref is not None:
        result = _verify(result, ref, wait_seconds)
    return result


def focus(name: str) -> Dict[str, Any]:
    """Bring an already-open app to the front rather than starting a second copy."""
    ref = resolve(name)
    if ref is None:
        started = launch(name)
        return {**started, "message": started.get("message", "") + " (it wasn't open, so I started it)."} \
            if started.get("ok") else {"ok": False, "message": f"Nothing named {name} is open.",
                                       "did_you_mean": suggest(name)}
    out = winops.activate(title="", process=ref.process or Path(ref.target).stem)
    if not out.get("ok"):
        out = winops.activate(title=ref.display.split("(")[0].strip())
    if not out.get("ok"):
        retry = launch(name)
        if retry.get("ok"):
            return {**retry, "app": ref.key, "display": ref.display, "method": "launch-then-focus",
                    "message": f"{retry.get('message', '')} Focusing it instead of opening a new window."}
        return {"ok": False, "app": ref.key, "display": ref.display,
                "message": f"I could not bring {ref.display} to the front, and opening it failed too: "
                           f"{retry.get('message') or 'Windows refused'}",
                "did_you_mean": suggest(name)}
    return {**out, "app": ref.key, "display": ref.display, "method": "foreground"}


def close(name: str) -> Dict[str, Any]:
    """Close an app: graceful window-close first, then taskkill on the resolved exe."""
    ref = resolve(name)
    process = (ref.process if ref else "") or ""
    if not process:
        guess = _slug(strip_verbs(name)).replace("-", "")
        for item in winops.windows():
            title = str(item.get("title", "")).lower()
            if guess and guess.replace(" ", "") in title.replace(" ", ""):
                process = str(item.get("process", "")) or ""
                if process:
                    break
    if not process:
        return {"ok": False, "message": f"I don't know which program “{name}” runs as.",
                "did_you_mean": suggest(name)}
    out = winops.kill_process(process)
    display = ref.display if ref else name
    return {**out, "app": _slug(name), "display": display, "process": process,
            "message": f"{display} closed." if out.get("ok") else out.get("message", "")}


def installed(query: str = "", limit: int = 24) -> Dict[str, Any]:
    """The app list, so the model (and the user) stops guessing names."""
    names = known_names()
    if query:
        tokens = _tokens(query)
        ranked = sorted(names, key=lambda n: (-(len(_tokens(n) & tokens) / max(len(_tokens(n) | tokens), 1)), n))
        names = [n for n in ranked if _tokens(n) & tokens or _slug(query) in _slug(n)] or names
    return {"ok": True, "count": len(names), "apps": names[:limit],
            "message": (f"{len(names)} launchable apps. Some of them: {', '.join(names[:limit])}."
                        if names else "I could not read the Start Menu on this machine.")}


def wait_for_window(process: str, title: str = "", seconds: float = 6.0) -> Dict[str, Any]:
    """Poll until the app's window exists - used to tell the truth about "opened"."""
    deadline = time.time() + max(0.0, seconds)
    proc, want = _norm(process).removesuffix(".exe"), _norm(title)
    while time.time() < deadline:
        for item in winops.windows():
            if proc and proc in _norm(str(item.get("process", ""))):
                return {"ok": True, "visible": True, "title": str(item.get("title", ""))}
            if want and want in _norm(str(item.get("title", ""))):
                return {"ok": True, "visible": True, "title": str(item.get("title", ""))}
        time.sleep(0.25)
    return {"ok": False, "visible": False, "title": ""}


__all__ = ["AppRef", "BUILTIN", "SETTINGS_PAGES", "resolve", "launch", "close", "focus", "installed",
           "suggest", "known_names", "custom_apps", "settings_page", "strip_verbs", "wait_for_window",
           "start_menu_index", "uwp_index", "registry_index"]
