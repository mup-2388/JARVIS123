"""
router.py -- Track 2 of the latency design: the agentic LLM brain.

Pipeline for one utterance
-------------------------
1. Build the JARVIS system prompt (persona + today's date + hard tool rules)
   and prepend :class:`ConversationMemory` for follow-ups.
2. Call a tool-calling model through the provider pool in ``llm_providers``
   ``tools=TOOL_SCHEMAS`` (strict JSON schema: every property required,
   ``additionalProperties: false``, enums wherever the argument space is small).
3. The model may answer directly, or emit one or more tool calls. Every call is
   validated against the schema (unknown key / wrong enum / missing required
   field) and *then* executed through :func:`tools.execute_tool`.
4. Tool output is appended as a ``tool`` role message and the loop repeats, up
   to ``MAX_TOOL_ROUNDS`` (3). The final natural-language reply is what the
   server speaks and prints in the HUD terminal.
5. If the Hub is unreachable, the token is missing, or the model misbehaves, we
   degrade to :func:`heuristic_plan` -- a deterministic keyword→tool planner
   that still executes real tools and returns real data instead of an apology.

No network call is ever made with an unset token, and no tool is ever run
blind: everything funnels through the allow-list in ``TOOL_SCHEMAS``.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import tools
from config import SETTINGS, get_logger
from llm_providers import LlmError, POOL, choose_tier, looks_like_research, strip_reasoning

log = get_logger("router")

#: "open Chrome, search for X and read me the top three results" is four tool calls deep, and
#: a chain that stops at round three is the difference between an assistant and a toy.  Rounds
#: are cheap when the tools answer in milliseconds - LLM_BUDGET_SECONDS is what bounds a
#: turn, not this number.
MAX_TOOL_ROUNDS = 6
HISTORY_TURNS = max(4, SETTINGS.history_size // 2)

# ---------------------------------------------------------------------------
# Strict JSON schemas bound to tools.py
# ---------------------------------------------------------------------------

def _fn(name: str, description: str, properties: Dict[str, Any], required: Optional[List[str]] = None) -> Dict[str, Any]:
    """Strict schema helper: all declared properties are required (use "" / null to omit)."""
    props = {
        key: {**spec, "description": spec.get("description", "")} if "description" not in spec else spec
        for key, spec in properties.items()
    }
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required if required is not None else list(props),
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    _fn(
        "search_on_site",
        "Run a search INSIDE a named web service and open its results page. Use this whenever the "
        "user names a site: 'search LM Arena on YouTube', 'look that up on google.com', "
        "'search spotify for lofi'. site accepts a name (youtube, google, github, reddit, amazon, "
        "spotify, wikipedia, imdb...) or a domain (youtube.com).",
        {
            "site": {"type": "string", "description": "Site name or domain, e.g. 'youtube' or 'google.com'"},
            "query": {"type": "string", "description": "What to search for, without the site words"},
            "open_browser": {"type": "boolean", "description": "Open the results page in the browser"},
        },
    ),
    _fn(
        "focus_app",
        "Bring an already-open app to the front instead of launching a second copy: 'look at Teams', "
        "'switch to Chrome', 'go back to my terminal'.",
        {"name": {"type": "string", "description": "App or window title to focus"}},
    ),
    _fn(
        "list_apps",
        "List what is installed and launchable on this machine. Use it before claiming an app does not "
        "exist, and when the user asks 'what apps can I open'.",
        {"query": {"type": "string", "description": "Optional filter, e.g. 'emulator' or 'adobe'"}},
    ),
    _fn(
        "windows_on_screen",
        "Which windows are open and which one has focus - the cheap way to answer 'what am I looking at' "
        "or 'is that download still running in another window'.",
        {"limit": {"type": "string", "description": "How many windows to list, as digits"}},
    ),
    _fn(
        "manage_files",
        "Create, write, append, read, list, search, move, copy, rename, delete (to the Recycle Bin), undo, "
        "and run a script you just wrote. This is how the user gets real work done: 'make a file called "
        "notes.md with...', 'delete old.txt', 'find every python file with TODO in it', 'write a python "
        "script that renames my downloads and run it'. Writes are confined to the folders JARVIS owns; a "
        "path outside them returns needs_confirmation, so ask first and only repeat with confirm='yes' when "
        "the user agrees. .docx/.pptx can be read. undo restores the last delete or overwrite.",
        {
            "action": {"type": "string", "enum": ["list", "read", "write", "overwrite", "append", "delete",
                                                 "search", "move", "copy", "rename", "mkdir", "note",
                                                 "undo", "disk", "open", "script", "run", "recent"],
                      "description": "What to do with files. 'write' creates (and refuses to clobber), 'overwrite' replaces, 'delete' means Recycle Bin, 'undo' reverses the last change."},
            "path": {"type": "string", "description": "File or folder, e.g. 'notes.md' or 'Desktop/todo.txt'; '' means the JARVIS folder"},
            "content": {"type": "string", "description": "Text to write (markdown is fine)"},
            "destination": {"type": "string", "description": "Target for move/copy/rename"},
            "query": {"type": "string", "description": "Name pattern for search, e.g. '*.md'"},
            "text": {"type": "string", "description": "Phrase to find inside files, or a sort key for list"},
            "confirm": {"type": "string", "description": "'yes' only after the user agreed to an outside-root write or delete"},
            "run": {"type": "string", "description": "'yes' to execute a script immediately after writing it"},
            "language": {"type": "string", "description": "python | powershell | bat | js | sh | html"},
            "limit": {"type": "string", "description": "How many results, as digits"},
        },
    ),
    _fn(
        "control_desktop",
        "Drive the desktop itself: type into the focused window, press keys or shortcuts (ctrl+s, alt+tab, "
        "win+d), click, scroll, minimise/maximise the current window, read or set the clipboard, volume and "
        "media keys, screenshot, wallpaper, a Windows notification, lock the PC, list processes. Use it for "
        "'type hello into the box', 'press escape', 'save that file', 'mute', 'next song', 'minimise this'.",
        {
            "action": {"type": "string", "enum": ["type", "press", "hotkey", "click", "move", "scroll",
                                                  "scroll_up", "minimize", "maximize", "restore", "focus",
                                                  "list_windows", "clipboard", "clipboard_write", "volume",
                                                  "media", "screenshot", "lock", "wallpaper", "notify",
                                                  "battery", "processes", "open_folder", "beep"],
                      "description": "The desktop action. Everything acts on the window that has focus, so say what the user said rather than guessing coordinates."},
            "text": {"type": "string", "description": "What to type, or the volume/media target"},
            "keys": {"type": "string", "description": "Key for press, e.g. 'escape' or 'down'"},
            "combo": {"type": "string", "description": "Shortcut for hotkey, e.g. 'ctrl+s'"},
            "x": {"type": "string", "description": "Screen x, or '' for the current pointer"},
            "y": {"type": "string", "description": "Screen y, or '' for the current pointer"},
            "button": {"type": "string", "enum": ["left", "right", "middle"],
                      "description": "Mouse button for click; 'left' when unused."},
            "amount": {"type": "string", "description": "Repeat count or scroll notches, as digits"},
            "level": {"type": "string", "description": "Volume percent for set"},
            "path": {"type": "string", "description": "File/folder for screenshot, wallpaper, open_folder"},
        },
    ),
    _fn(
        "read_screen",
        "Use the eyes. 'read' = OCR every word currently on screen (offline). 'list' = the top N results or "
        "listings, which is what 'read out the top three' means. 'describe' = send the capture to a vision "
        "model and answer a question about it ('what is this error', 'what am I looking at'). 'window' = the "
        "focused window and the open ones. 'save' = write what you read into a file.",
        {
            "action": {"type": "string", "enum": ["read", "list", "describe", "window", "capture", "save"],
                      "description": "read = OCR text, list = top N items, describe = vision model answer, window = titles only, capture = save an image, save = write what was read to a file."},
            "count": {"type": "string", "description": "How many items for list, as digits (default 3)"},
            "question": {"type": "string", "description": "What to ask about the screen for describe"},
            "target": {"type": "string", "enum": ["screen", "window"], "description": "Whole display or the focused window"},
            "save_to": {"type": "string", "description": "Filename for action=save"},
        },
    ),
    _fn(
        "set_reminder",
        "Timers, reminders and scheduled commands - things that happen later without the user asking again. "
        "'remind me in ten minutes to stretch', 'set a timer for 25 minutes', 'at 7:30 pm check my download' "
        "(run='yes' when the text is a task, not a note), 'what have I got scheduled', 'cancel the gym "
        "reminder', 'snooze it 10 minutes'. when takes the user's own words.",
        {
            "action": {"type": "string", "enum": ["add", "list", "cancel", "snooze"],
                      "description": "add schedules one, list shows what is pending, cancel removes by text or id, snooze pushes it back by minutes."},
            "text": {"type": "string", "description": "What to remind about, or which reminder to cancel"},
            "when": {"type": "string", "description": "'in ten minutes' | 'at 7:30 pm' | 'tomorrow at 9' | 'every 2 hours'"},
            "minutes": {"type": "string", "description": "Fallback span as digits, used only when when is empty"},
            "run": {"type": "string", "description": "'yes' to execute text as a command at that time"},
        },
    ),
    _fn(
        "listening",
        "The background ear: start or stop always-on wake-word listening, ask for five seconds of mic now, "
        "or report why it is off (no sounddevice, no mic, disabled in .env).",
        {"status": {"type": "string", "enum": ["status", "start", "stop", "listen"],
                   "description": "status reports, start/stop switch the always-on ear, listen grabs five seconds from the mic right now."}},
    ),
    _fn(
        "llm_status",
        "Report which AI providers are configured, which one answers, and which are cooling down "
        "after hitting a quota. For 'which model are you using', 'AI status', 'check the providers'.",
        {},
    ),
    _fn(
        "launch_app",
        "Open ANY application, Settings page, Control Panel applet or installed program by the name the user "
        "used - Chrome, Microsoft Teams, Task Manager, Bluetooth settings, Steam, Word, Downloads, Recycle "
        "Bin. Do not wonder whether it is installed: the resolver searches the Start Menu, UWP packages, the "
        "uninstall registry and App Paths, and when it misses it answers with the closest real names to "
        "offer instead. Use for 'open/launch/start/focus <app>'.",
        {
            "app_name": {"type": "string", "description": "Application name or alias, e.g. 'Steam', 'Eden', 'chrome'."},
            "url": {"type": "string", "description": "Optional URL to open if the app is a browser. '' when unused."},
            "args": {"type": "string", "description": "Optional extra command line flags. '' when unused."},
        },
    ),
    _fn("close_app", "Quit/kill a running application by name.", {"app_name": {"type": "string", "description": "Application name."}}),
    _fn(
        "open_website",
        "Open a website in the default browser, optionally deep-linking a search into it (youtube, github, amazon, maps, wikipedia, reddit).",
        {
            "target": {"type": "string", "description": "Domain or known site nickname, e.g. 'youtube.com' or 'youtube'."},
            "query": {"type": "string", "description": "Search terms to inject into that site. '' when unused."},
        },
    ),
    _fn("play_youtube", "Search YouTube and open/playing the best matching video. Use for 'play <song/video>'.",
        {"query": {"type": "string", "description": "What to play."}}),
    _fn(
        "web_search",
        "Live DuckDuckGo search for anything factual or local: prices, restaurants, addresses, news, docs, 'cheapest Indian restaurants in Surat'.",
        {
            "query": {"type": "string", "description": "Search query in natural language."},
            "max_results": {"type": "integer", "description": "1..20 results, 6 is a good default."},
            "timelimit": {"type": "string", "enum": ["", "d", "w", "m", "y"], "description": "Recency filter: day/week/month/year, or '' for none."},
            "site": {"type": "string", "description": "Restrict to one domain, e.g. 'youtube.com'. '' for any."},
        },
    ),
    _fn(
        "fetch_sports_stats",
        "Live football data from api-sports.io: recent results, next fixtures, league standing and top scorers for a club "
        "(Real Madrid, Barcelona, Manchester City...). Use for any score, fixture, table or player-stat question.",
        {
            "team": {"type": "string", "description": "Club name as typed by the user."},
            "kind": {"type": "string", "enum": ["all", "results", "fixtures", "standing", "scorers"], "description": "Which slice to fetch; 'all' for a brief."},
        },
    ),
    _fn(
        "read_notes",
        "Read the user's own Markdown study notes (German lessons, CS degree prep...). Pass topic='' to list all notes.",
        {
            "topic": {"type": "string", "description": "Keyword or phrase to match, e.g. 'german dative'."},
            "limit": {"type": "integer", "description": "How many notes to return (1-4)."},
        },
    ),
    _fn(
        "write_note",
        "Persist something into ./notes as Markdown. Use when the user says 'note that ...' or asks to remember study progress.",
        {
            "topic": {"type": "string", "description": "Note title."},
            "content": {"type": "string", "description": "Body text to remember."},
            "tags": {"type": "string", "description": "Comma separated tags, e.g. 'german,homework'. '' for none."},
        },
    ),
    _fn(
        "system_report",
        "Local machine telemetry: CPU, per-core load, RAM, disk, GPU/VRAM (nvidia-smi), battery, uptime, busiest processes.",
        {"detailed": {"type": "boolean", "description": "Include top processes and per-core bars."}},
    ),
    _fn("get_time", "Current local time and date. Use for 'what time is it' / 'what is today'.", {}),
    _fn(
        "set_volume",
        "Control the Windows master volume. Provide exactly one intent: absolute level, relative delta, or mute.",
        {
            "level": {"type": "integer", "description": "Absolute 0-100 percent, or -1 to leave untouched."},
            "delta": {"type": "integer", "description": "Relative steps (+ louder / -1 quieter), 0 to leave untouched."},
            "mute": {"type": "boolean", "description": "Toggle mute."},
        },
    ),
    _fn("take_screenshot", "Capture the whole desktop into data/screenshots and report the file path.", {}),
    _fn(
        "system_power",
        "Session/power control. Only use when the user explicitly asks: lock, sleep, shutdown, restart, cancel-shutdown.",
        {"action": {"type": "string", "enum": ["lock", "sleep", "shutdown", "restart", "cancel-shutdown"], "description": "Power action."}},
    ),
    _fn(
        "calendar_agenda",
        "What is on the user's Google/Outlook calendar for the next N days (default 1 = today): classes, "
        "meetings, shifts, visits. Use for 'what's on my calendar', 'what do I have tomorrow', 'my agenda', "
        "'any meetings this week'.",
        {"days": {"type": "string", "description": "How many days to look ahead, as digits. '1' = today, '7' = a week."}},
    ),
    _fn(
        "calendar_next",
        "The single next upcoming calendar event (class, meeting, shift…). Use for 'what's next', 'my next class', 'when's my next meeting'.",
        {},
    ),
    _fn(
        "todo",
        "The everyday to-do list. add writes an item, list shows them, done ticks one off, clear empties it. "
        "Use for 'add submit the CBS assignment to my todo', 'what's on my todo', 'tick off gym'.",
        {
            "action": {"type": "string", "enum": ["list", "add", "done", "clear"], "description": "list / add / done(remove) / clear."},
            "text": {"type": "string", "description": "The item text for add/done."},
        },
    ),
    _fn(
        "daily_brief",
        "The morning brief: today's date, calendar, to-do list and pending reminders in one answer. "
        "Use for 'good morning', 'what does my day look like', 'daily brief', 'summarise my day'.",
        {},
    ),
    _fn(
        "draft_email",
        "Write an email draft and open the mail app (Outlook/mailto) with it prefilled - no credentials needed. "
        "Use for 'draft an email to X saying...', 'write an email to my professor'.",
        {
            "to": {"type": "string", "description": "Recipient address. '' to leave blank."},
            "subject": {"type": "string", "description": "Subject line."},
            "body": {"type": "string", "description": "Email body text."},
        },
    ),
    _fn(
        "send_email",
        "Actually SEND an email through the SMTP server in .env (SMTP_HOST/SMTP_USER/SMTP_PASSWORD). Falls back to "
        "drafting if SMTP is not configured. Use only when the user explicitly says 'send'.",
        {
            "to": {"type": "string", "description": "Recipient address."},
            "subject": {"type": "string", "description": "Subject line."},
            "body": {"type": "string", "description": "Email body text."},
        },
    ),
]

TOOL_NAMES = {s["function"]["name"] for s in TOOL_SCHEMAS}
_TOOL_INDEX = {s["function"]["name"]: s["function"]["parameters"]["properties"] for s in TOOL_SCHEMAS}
_TOOL_ENUMS = {
    name: {key: spec["enum"] for key, spec in props.items() if isinstance(spec, dict) and "enum" in spec}
    for name, props in _TOOL_INDEX.items()
}

# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    role: str            # "user" | "assistant" | "tool"
    content: str
    ts: float = field(default_factory=time.time)
    track: str = ""
    tool: str = ""

    def as_message(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class ConversationMemory:
    """Bounded rolling context; good enough for pronoun follow-ups, cheap on tokens."""

    def __init__(self, max_turns: int = HISTORY_TURNS * 2) -> None:
        self._items: Deque[Turn] = deque(maxlen=max(4, max_turns))
        self._lock = threading.Lock()

    def add(self, role: str, content: str, track: str = "", tool: str = "") -> None:
        with self._lock:
            self._items.append(Turn(role=role, content=_clip(content, 1200), track=track, tool=tool))

    def recent(self, limit: int = HISTORY_TURNS) -> List[Dict[str, str]]:
        with self._lock:
            items = list(self._items)[-limit * 2:]
        return [t.as_message() for t in items if t.content]

    def snapshot(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {"role": t.role, "content": t.content, "track": t.track, "tool": t.tool,
                 "at": datetime.fromtimestamp(t.ts).strftime("%H:%M:%S")}
                for t in list(self._items)[-limit:]
            ]

    def last_tool(self) -> str:
        with self._lock:
            for turn in reversed(self._items):
                if turn.tool:
                    return turn.tool
        return ""

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


MEMORY = ConversationMemory()


def _clip(text: str, limit: int = 400) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# Validation of whatever the model asks for
# ---------------------------------------------------------------------------

#: Arguments without which a call is meaningless. The JSON schema deliberately
#: marks *every* property required (strict mode demands it), so the semantic
#: "you cannot omit this one" list lives here instead.
ESSENTIAL_ARGS: Dict[str, Tuple[str, ...]] = {
    "launch_app": ("app_name",),
    "close_app": ("app_name",),
    "open_website": ("target",),
    "play_youtube": ("query",),
    "web_search": ("query",),
    "search_on_site": ("query",),
    "write_note": ("topic", "content"),
    "system_power": ("action",),
    "fetch_sports_stats": (),   # falls back to API_SPORTS_DEFAULT_TEAM
    "read_notes": (),           # empty topic == "list my notes"
    # The wide tools are action-dispatched, so the action is the one thing that must be there;
    # everything else is optional and validated by the tool itself with a spoken error.
    "manage_files": ("action",),
    "control_desktop": ("action",),
    "read_screen": ("action",),
    "set_reminder": ("action",),
    "focus_app": ("name",),
    "listening": (),
    "list_apps": (),
    "windows_on_screen": (),
}


def validate_call(name: str, args: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]], str]:
    """Return ``(tool, cleaned_args, error)`` -- the LLM never runs raw kwargs.

    Unknown keys are dropped, enums enforced, scalars coerced, and impossible
    combinations (volume with no target value) rejected before anything in
    :mod:`tools` is touched.
    """
    if name not in TOOL_NAMES:
        return None, None, f"tool '{name}' is not in the allow-list"
    props = _TOOL_INDEX.get(name, {})
    args = dict(args or {})
    cleaned: Dict[str, Any] = {}

    for key, spec in props.items():
        if key not in args:
            continue
        value = args[key]
        expected = spec.get("type")
        try:
            if expected == "string":
                cleaned[key] = ("" if value is None else str(value)).strip()
            elif expected == "integer":
                cleaned[key] = int(float(value))
            elif expected == "number":
                cleaned[key] = float(value)
            elif expected == "boolean":
                cleaned[key] = value in (True, "true", "True", 1, "1", "yes", "on")
            else:
                cleaned[key] = value
        except (TypeError, ValueError):
            return None, None, f"'{_clip(str(value), 60)}' is not a valid {expected} for {name}.{key}"
        if "enum" in spec and cleaned[key] not in spec["enum"]:
            return None, None, f"'{cleaned[key]}' is not allowed for {name}.{key} (choose from {spec['enum']})"

    for key in ESSENTIAL_ARGS.get(name, ()):
        if not str(cleaned.get(key, "")).strip():
            return None, None, f"{name} needs a non-empty '{key}'"

    # Semantic normalisation: level=-1 / delta=0 / "" mean "not requested".
    if name == "set_volume":
        if int(cleaned.get("level", -1)) < 0:
            cleaned.pop("level", None)
        if not cleaned.get("delta"):
            cleaned.pop("delta", None)
        if not cleaned.get("mute"):
            cleaned.pop("mute", None)
        if not cleaned:
            return None, None, "set_volume needs level (0-100), a non-zero delta, or mute=true"
    for drop in ("max_results", "limit"):
        if drop in cleaned and int(cleaned[drop] or 0) <= 0:
            cleaned.pop(drop)
    if name == "system_report":
        cleaned.setdefault("detailed", True)
    if name == "fetch_sports_stats" and not cleaned.get("team"):
        cleaned["team"] = SETTINGS.api_sports_default_team
    return name, cleaned, ""


# ---------------------------------------------------------------------------
# Heuristic planner (offline Track 2 -- no LLM, still real tools)
# ---------------------------------------------------------------------------

_KEYWORD_PLAN: List[Tuple[re.Pattern[str], str, Callable[[re.Match[str]], Dict[str, Any]]]] = [
    # "what windows are open", "which apps do I have running" - that is a question about the
    # desktop, and the generic "open X" rule below would otherwise read it as a request to launch
    # the word "right".  Both rules answer with the window list, focused window first.
    (re.compile(r"\b(?:what|which|list|show|how many)\b[^.?]{0,30}\b(?:windows|apps|programs|windows)\b"
                r"[^.?]{0,20}\b(?:open|running|up right|right now|currently|on screen)\b", re.I),
     "windows_on_screen", lambda m: {"limit": "12"}),
    (re.compile(r"^\s*(?:hey[ ,]+)?(?:jarvis[ ,]+)?what(?:'s| is| do i) (?:open|running|up)"
                r"(?: on (?:my|the) (?:screen|desktop|pc))?(?: right now)?\s*[?.]*$", re.I),
     "windows_on_screen", lambda m: {"limit": "12"}),
    # The whole tail is the app name: the resolver knows ms-settings pages, UWP packages,
    # Start-menu shortcuts and the registry, so "open my bluetooth settings" must not be
    # flattened to "settings" by a keyword list that can never keep up with Windows.  It is
    # anchored to the start of the sentence, because "what windows are open right now" is a
    # question - and a name never survives a conjunction: "open google and read the top 3"
    # means google, and the reading part is its own call, which the planner chains.
    (re.compile(r"^\s*(?:(?:hey|ok|yo)[ ,]+)?(?:jarvis[ ,]+)?(?:could you |can you |would you |please |just |simply |quickly |now )*?"
                r"(?:open|launch|start|boot(?: up)?|fire up|spin up|bring up|run|wake|show me)\s+(?:up\s+)?"
                r"(?:the\s+|my\s+|our\s+)?(?P<app>[a-z][a-z0-9 .&'+_-]{1,40}?)"
                r"(?:\s+(?:app|application|program|software|exe))?(?:\s+right\s+now|\s+now|\s+please|\s+for me|\s+thanks)?"
                r"[.!?]*$", re.I),
     "launch_app", lambda m: {"app_name": re.split(r"\s+(?:and|then|plus|after that)\s+",
                                                   (m.group("app") or "").strip())[0].strip(" .,")}),
    (re.compile(r"\b(close|quit|kill|exit|terminate)\b[^.]*?\b(chrome|edge|firefox|browser|steam|discord|spotify|notepad|calculator|code|terminal|explorer|obs|eden)\b", re.I),
     "close_app", lambda m: {"app_name": m.group(2).strip()}),
    (re.compile(r"\b(play|put on|queue up)\b(?:\s+(?:the\s+|my\s+))?(.+?)(?:\s+on\s+youtube|\s+on\s+yt|\?|$)", re.I),
     "play_youtube", lambda m: {"query": (m.group(2) or "").strip(" .,?!")}),
    # "search X on youtube" / "search youtube for X" - the named site owns it.
    (re.compile(r"\b(?:search|look up|google)\b(?:\s+for)?\s+(?P<q>.+?)\s+(?:on|in|at)\s+(?P<site>[a-z][a-z0-9 ._-]{1,24}(?:\.[a-z]{2,})?)$", re.I),
     "search_on_site", lambda m: {"site": (m.group("site") or "").strip(), "query": (m.group("q") or "").strip(" .?"), "open_browser": True}),
    (re.compile(r"\bsearch\s+(?P<site>[a-z][a-z0-9 ._-]{1,24}?)\s+for\s+(?P<q>.+)$", re.I),
     "search_on_site", lambda m: {"site": (m.group("site") or "").strip(), "query": (m.group("q") or "").strip(" .?"), "open_browser": True}),
    # "open my bluetooth settings", "open device manager", "open the downloads folder"
    (re.compile(r"\b(?:open|show|go to)\s+(?:the\s+|my\s+)?(?P<t>[a-z][a-z .&'-]{2,38}?(?:settings|panel|manager|bin|folder|center|centre))\s*$", re.I),
     "launch_app", lambda m: {"app_name": (m.group("t") or "").strip()}),
    # ---- files ------------------------------------------------------------------
    (re.compile(r"\b(?:create|make|write|start)(?: me| up)?\s+(?:a\s+|my\s+)?(?:new\s+)?(?:(?:txt|md|markdown|py|python|csv|html|json|js|text)\s+)?(?:file|document|doc|folder|directory|script)?\s*(?:called|named|titled)?\s*['\"]?(?P<name>[\w\- .():\\/]{2,120}?)['\"]?\s*(?:with|that says|containing)\s*['\"]?(?P<body>.{1,900})$", re.I),
     "manage_files", lambda m: {"action": "write", "path": (m.group("name") or "").strip(),
                                "content": (m.group("body") or "").strip().strip('"')}),
    (re.compile(r"\b(?:delete|remove|erase|trash|get rid of)\s+(?:the\s+|my\s+|a\s+|an\s+|this\s+|that\s+)?"
                r"(?:file\s+|folder\s+|document\s+|doc\s+)?(?:called\s+|named\s+|titled\s+)?[\x22\x27]?"
                r"(?P<name>[\w\- .():\\/]{2,120}?)[\x22\x27]?\s*(?:please|for me|now|thanks|dot\s+\w+)?$", re.I),
     "manage_files", lambda m: {"action": "delete", "path": (m.group("name") or "").strip(" .,")}),
    (re.compile(r"\b(?:what(?:'s| is) in|read|summar(?:ise|ize)|tell me about)\s+(?:the\s+|my\s+)?(?:file|doc(?:ument)?|notes?)\s*['\"]?(?P<name>[\w\- .():\\/]{2,120})", re.I),
     "manage_files", lambda m: {"action": "read", "path": (m.group("name") or "").strip()}),
    (re.compile(r"\b(?:list|show)\s+(?:my|the)\s+(?:files|folder|documents|downloads)\b", re.I),
     "manage_files", lambda m: {"action": "list", "path": "downloads" if "downloads" in m.group(0).lower() else ""}),
    (re.compile(r"\b(?:find|search for)\s+(?:a\s+|my\s+)?files?\s+(?:named\s+|with\s+)?['\"]?(?P<name>[\w\- .*():\\/]{2,120})", re.I),
     "manage_files", lambda m: {"action": "search", "query": (m.group("name") or "").strip()}),
    (re.compile(r"\bundo\s+(?:that|it|the last (?:file )?(?:change|delete|write))\b", re.I),
     "manage_files", lambda m: {"action": "undo", "path": "", "content": "", "destination": "", "query": "",
                                "text": "", "confirm": "", "run": "", "language": "", "limit": "1"}),
    # ---- eyes -------------------------------------------------------------------
    (re.compile(r"\b(what am i looking at|what('s| is) on (?:my|the) screen|read (?:my|the) screen|describe (?:the|my) screen|look at (?:my|the) screen)\b", re.I),
     "read_screen", lambda m: {"action": "describe" if ("describe" in m.group(0) or "looking at" in m.group(0)) else "read",
                               "count": "3", "question": "", "target": "screen", "save_to": ""}),
    (re.compile(r"\b(?:top|first)\s+(?P<n>\d{1,2}|one|two|three|four|five)\s+(?:results?|listings?|items?|links?|products?|offers?|entries)\b", re.I),
     "read_screen", lambda m: {"action": "list",
                               "count": {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}.get((m.group("n") or "3").lower(), m.group("n")),
                               "question": "", "target": "screen", "save_to": ""}),
    # ---- desktop ----------------------------------------------------------------
    (re.compile(r"\b(?:type|write)\s+['\"](?P<text>.{1,300})['\"]", re.I),
     "control_desktop", lambda m: {"action": "type", "text": (m.group("text") or "").strip()}),
    (re.compile(r"\b(?:type|write)\s+(?P<text>.{1,200}?)\s+(?:in|into)\s+(?:the|my)\s+(?:box|field|search|bar|text ?box|terminal|console|editor|address bar)\b", re.I),
     "control_desktop", lambda m: {"action": "type", "text": (m.group("text") or "").strip()}),
    (re.compile(r"\b(?:press|hit|tap)\s+(?P<k>escape|enter|tab|space|backspace|delete|f\d{1,2}|down arrow|up arrow|right arrow|left arrow)\b", re.I),
     "control_desktop", lambda m: {"action": "press", "keys": (m.group("k") or "").strip()}),
    (re.compile(r"\b(?:save (?:this|that|the file)|press ctrl ?s)\b", re.I),
     "control_desktop", lambda m: {"action": "hotkey", "combo": "ctrl+s"}),
    (re.compile(r"\b(?:minimi[sz]e|hide)\s+(?:this|the current|the active)\s*(?:window|app)?\b", re.I),
     "control_desktop", lambda m: {"action": "minimize"}),
    (re.compile(r"\bmaximi[sz]e\s+(?:this|the window)\b", re.I),
     "control_desktop", lambda m: {"action": "maximize"}),
    (re.compile(r"\b(?:switch to|go back to|focus|look at)\s+(?P<t>[a-z][a-z0-9 .'_-]{1,30})$", re.I),
     "focus_app", lambda m: {"name": (m.group("t") or "").strip()}),
    # ---- later, at a time -------------------------------------------------------
    (re.compile(r"\bremind me\b(?:\s+(?:to|that|about))?\s*(?P<what>.{0,140}?)\s*(?P<when>in\s+[\w ]{2,30}|at\s+[\w :]{2,20}|tomorrow[\w ]{0,20}|tonight[\w ]{0,20})\s*$", re.I),
     "set_reminder", lambda m: {"action": "add", "text": (m.group("what") or "").strip(" .,!?"),
                                "when": (m.group("when") or "").strip(), "minutes": "", "run": ""}),
    (re.compile(r"\bset (?:a\s+)?timer for\s+(?P<when>[\w ]{2,30})", re.I),
     "set_reminder", lambda m: {"action": "add", "text": "Timer finished", "when": "in " + (m.group("when") or "").strip(),
                                "minutes": "", "run": ""}),
    (re.compile(r"\bcancel\s+(?:the\s+|my\s+)?(?:timer|reminder|alarm)(?:\s+(?:about|for|to)\s+(?P<what>.{2,60}))?\s*$", re.I),
     "set_reminder", lambda m: {"action": "cancel", "text": (m.group("what") or "").strip(),
                                "when": "", "minutes": "", "run": ""}),
    (re.compile(r"\b(?:what(?:'s| is) (?:on|my) (?:schedule|reminders|timers)|list (?:my )?reminders)\b", re.I),
     "set_reminder", lambda m: {"action": "list", "text": "", "when": "", "minutes": "", "run": ""}),
    # Explicit research verbs only: "what is X" is the model's job, not DDG's.
    (re.compile(r"\b(search|google|look up|find out|news about|weather in)\s+(?:for\s+)?(.+)", re.I),
     "web_search", lambda m: {"query": re.sub(r"\b(please|for me|me)\b", "", m.group(2), flags=re.I).strip(" .?")}),
    (re.compile(r"\b(real madrid|barcelona|manchester city|manchester united|liverpool|arsenal|chelsea|bayern|dortmund|psg|juventus|milan|inter|mumbai city|bengaluru|kerala blasters)\b", re.I),
     "fetch_sports_stats", lambda m: {"team": m.group(1).title(), "kind": "all"}),
    (re.compile(r"\b(?:note|write)\s+(?:that|down)\b[: ,]*(.+)|\bremember\s+(?:that\s+)?(.+)", re.I | re.S),
     "write_note", lambda m: {
         "topic": _clip(re.sub(r"\s+", " ", (m.group(1) or m.group(2) or "")).split(".")[0][:48], 48) or "Voice memo",
         "content": _clip(re.sub(r"\s+", " ", (m.group(1) or m.group(2) or "")).strip(), 900),
         "tags": "voice-memo",
     }),
    (re.compile(r"\b(my )?(notes?|revision|study|flashcard)\b", re.I),
     "read_notes", lambda m: {"topic": _strip_fillers(m.string), "limit": 1}),
    # Coarse telemetry intent. ``server.py`` owns the precise Track-1 rules;
    # this copy only exists so the offline planner can answer "how much RAM".
    (re.compile(
        r"\b(?:cpu|ram|memory|gpu|vram|disk|ssd|battery|uptime)\b[^?!.\n]{0,20}"
        r"\b(?:usage|use|load|percent|temp|temperature|status|stats?|statistics|report|level|high|hot|free|available|space|capacity)\b"
        r"|\b(?:how much|free|available)\b[^?!.\n]{0,12}\b(?:ram|memory|vram|disk|ssd|space)\b"
        r"|\bhow'?s (?:my|the) (?:pc|system|machine|rig|laptop)\b"
        r"|\bhow(?:'s| is| are)?\b[^?!.\n]{0,14}\b(?:my|the)\b[^?!.\n]{0,10}\b(?:cpu|ram|memory|gpu|vram|disk|ssd|battery|pc|system|machine|rig|laptop)\b"
        r"|\b(?:system|hardware|core) (?:report|status|health)\b|\btelemetry\b|\bdiagnostics\b",
        re.I,
    ), "system_report", lambda m: {"detailed": True}),
    (re.compile(r"\b(time|date|day is it|today'?s date)\b", re.I),
     "get_time", lambda m: {}),
    (re.compile(r"\b(screenshot|screen ?shot|capture (?:the )?(?:screen|display))\b", re.I),
     "take_screenshot", lambda m: {}),
    (re.compile(r"\b(volume|mute|louder|quieter|turn (?:it )?(?:up|down))\b", re.I),
     "set_volume", lambda m: {"delta": 6 if re.search(r"up|louder|increase", m.string, re.I) else -6}),
    # ---- calendar / todo / brief / email ----------------------------------------
    (re.compile(r"\b(?:what(?:'s| is)? (?:on|in) my (?:calendar|schedule|agenda)|my agenda|"
                r"what (?:do i have|am i doing|have i got)(?: scheduled| on)?\b)", re.I),
     "calendar_agenda", lambda m: {"days": "7" if re.search(r"week|coming", m.string, re.I) else "1"}),
    (re.compile(r"\b(?:next|upcoming) (?:class|lecture|meeting|shift|appointment|event)|what('s| is) (?:next|up next|coming up)\b", re.I),
     "calendar_next", lambda m: {}),
    (re.compile(r"\b(?:add|put|write)\b[^.]{0,60}\b(?:to (?:my )?(?:to ?do|todo|to-do|task list|checklist))\b", re.I),
     "todo", lambda m: {"action": "add", "text": re.sub(r"^.*?(?:add|put|write)\s+", "", m.string, flags=re.I).split(" to ", 1)[0].strip(" .,")}),
    (re.compile(r"\b(?:what(?:'s| is) (?:on )?(?:my )?(?:to ?do|todo|to-do|task list|checklist)|show (?:my )?(?:to ?do|todo|task list))\b", re.I),
     "todo", lambda m: {"action": "list", "text": ""}),
    (re.compile(r"\b(?:good morning|morning brief|daily brief|start my day|what does my day look like|summar(?:ise|ize) my day)\b", re.I),
     "daily_brief", lambda m: {}),
    (re.compile(r"\b(?:draft|write|compose)\s+(?:an?\s+)?(?:email|mail)\b", re.I),
     "draft_email", lambda m: {"to": "", "subject": "", "body": ""}),
    (re.compile(r"\bsend\s+(?:an?\s+)?(?:email|mail)\b", re.I),
     "send_email", lambda m: {"to": "", "subject": "", "body": ""}),
]


def _strip_fillers(text: str) -> str:
    text = re.sub(r"^(hey\s+)?jarvis[,. ]*", "", text, flags=re.I)
    text = re.sub(r"\b(read|show|check|open|pull up|what (?:did|is|was))\b", " ", text, flags=re.I)
    text = re.sub(r"\b(my|the|about|on|from|again)\b", " ", text, flags=re.I)
    text = re.sub(r"\bnotes?\b", " ", text, flags=re.I)
    return _clip(re.sub(r"\s+", " ", text).strip(" ?."), 80)


#: Higher = more specific. "score of Real Madrid" must not fall through to a
#: generic web search when the sports tool clearly owns the intent.
_SPECIFICITY = {
    "manage_files": 94, "set_reminder": 92, "read_screen": 91,
    "write_note": 90, "read_notes": 88, "fetch_sports_stats": 86, "system_power": 84,
    "control_desktop": 83, "focus_app": 79, "listening": 76,
    "take_screenshot": 80, "set_volume": 78, "close_app": 74, "launch_app": 72,
    "play_youtube": 70, "search_on_site": 66, "open_website": 60, "system_report": 58, "get_time": 55,
    "daily_brief": 96, "calendar_agenda": 95, "calendar_next": 93, "todo": 92, "draft_email": 90, "send_email": 89,
    "web_search": 20,
}


def heuristic_plan(text: str, allow_search: bool = True) -> List[Dict[str, Any]]:
    """Deterministic intent → tool calls. Used offline and as a repair path."""
    calls: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for pattern, name, build in _KEYWORD_PLAN:
        match = pattern.search(text or "")
        if not match or name in seen:
            continue
        try:
            args = build(match)
        except Exception:  # noqa: BLE001
            continue
        if not args and name not in {"get_time", "take_screenshot", "daily_brief", "calendar_next"}:
            continue
        seen.add(name)
        calls.append({"tool": name, "arguments": args})
        if len(calls) >= 4:
            break
    calls.sort(key=lambda c: -_SPECIFICITY.get(c["tool"], 0))
    explicit_search = re.search(r"\b(search|google|look up|find out)\b", text or "", re.I)
    multi_intent = re.search(r"\s+(?:and|also|plus|then|after\s+that)\s+", text or "", re.I)
    if calls and not explicit_search and not multi_intent:
        # "what's the score of Real Madrid" -> sports only, no redundant crawl.
        # "...and how much ram" keeps both halves, so multi-part asks still work.
        calls = [c for c in calls if not (c["tool"] == "web_search" and _SPECIFICITY.get(c["tool"], 0) < 55)]
    if any(c["tool"] == "write_note" for c in calls):
        calls = [c for c in calls if c["tool"] != "read_notes"]  # dictation, not lookup
    if any(c["tool"] == "manage_files" and (c.get("arguments") or {}).get("action") == "write" for c in calls):
        # "write a file called X with Y" is a file job, not a notes job - exactly one of them runs.
        calls = [c for c in calls if c["tool"] != "write_note"]
    file_actions = {(c.get("arguments") or {}).get("action") for c in calls if c["tool"] == "manage_files"}
    if file_actions & {"read", "search", "list"}:
        # "read the file notes.md" names a file, so don't also search the notes folder for it.
        calls = [c for c in calls if c["tool"] != "read_notes"]
    if len(calls) > 4:
        calls = calls[:4]
    if not calls and (text or "").strip() and allow_search:
        # Only reached when the question genuinely needs the outside world; the agent -
        # not this function - owns "what is X" style questions.
        calls.append({"tool": "web_search", "arguments": {"query": _clip(re.sub(r"^(hey[ ,]+)?jarvis[ ,]+", "", text, flags=re.I), 160), "max_results": 5, "timelimit": "", "site": ""}})
    return calls


# ---------------------------------------------------------------------------
# Provider pool / tool-call parsing
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are J.A.R.V.I.S, a native Windows assistant with direct control of this machine.

Rules, in priority order:
1. Act, do not speculate. When the user asks for anything on this PC - open/close/focus apps, Settings pages, volume, screenshots, creating or deleting files, typing or clicking, reading the screen, reminders - call the matching tool instead of answering from memory, and never say "done" before a tool returned ok. If a tool fails, say in one line what failed.
1b. You have hands now: manage_files (create/overwrite/append/read/search/delete-to-Recycle-Bin/undo/run a script you wrote), control_desktop (SendInput typing, hotkeys, clicks, scroll, window control, clipboard, volume, media, wallpaper, notifications, lock), read_screen (offline OCR, "top three listings", or a vision model for "what am I looking at"), set_reminder (timers and scheduled commands), list_apps, focus_app, windows_on_screen. Prefer acting over explaining how the user could do it themselves.
1c. For a multi-step request ("open Chrome, search X, read me the top three") keep calling tools for as many rounds as it takes and give the spoken summary at the end.
1d. manage_files returns needs_confirmation for anything outside the folders you own: ask one short question, and only repeat with confirm="yes" after the user agrees. Deletes inside your own folders are always recoverable, so never refuse one.
2. One tool call at a time is fine, and several in sequence are better for multi-part requests ("open Steam and tell me my RAM").
3. Pass empty strings for unused optional arguments. Never invent file paths, team ids, prices or scores - read them from the tool response.
4. Only call system_power when the user explicitly asks to lock/sleep/shutdown/restart.
5. After the tool result, answer in 1-3 short spoken sentences: concrete numbers first, no markdown, no preamble like "here is what I found".
6. If no tool fits and you do not truly know the answer, say so in one line and suggest the tool you would need.
7. Study/revision questions read the user's own notes (read_notes). Live or verifiable facts
   (scores, prices, news, "latest", anything after your cutoff) go to web_search - or
   search_on_site when the user names a site ("search X on youtube", "look X up on google").
8. Everything else - explanations, ideas, writing, translation, maths, opinions about their
   code, "how do I", "what does X mean" - you answer YOURSELF from what you know. Do not call
   web_search to be polite, and never claim you cannot answer when you actually can.

Environment: Windows 11 desktop, RTX 3050 4 GB, local Whisper for input, XTTS-v2 for output.
Today: {today} ({weekday}).
"""


#: Some models wrap the tool call in a fenced block or pad it with prose, so the
#: "answer with JSON" contract needs a tolerant extractor.
_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def _salvage_json(blob: str) -> Dict[str, Any]:
    """Best-effort extraction of the first JSON object in a chatty completion."""
    blob = (blob or "").strip()
    if blob.startswith("```"):
        blob = re.sub(r"^```[a-zA-Z]*\s*", "", blob)
        blob = re.sub(r"\s*```$", "", blob)
    for candidate in (blob, (_JSON_BLOCK.search(blob).group(0) if _JSON_BLOCK.search(blob) else "")):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return {}


def parse_tool_call_from_text(content: str) -> Optional[Dict[str, Any]]:
    """Models that ignore the tools API still obey 'answer with JSON'."""
    blob = _salvage_json(content or "")
    if not blob:
        return None
    name = blob.get("tool") or blob.get("name") or blob.get("function") or blob.get("action")
    if not isinstance(name, str):
        return None
    args: Any = {}
    for key in ("arguments", "args", "parameters"):
        if key in blob:
            args = blob[key]
            break
    if isinstance(args, str):
        args = _salvage_json(args)
    if not isinstance(args, dict):
        args = {}
    if not args:
        skip = {"tool", "name", "function", "action", "thought", "reasoning", "arguments", "args", "parameters"}
        args = {k: v for k, v in blob.items() if k not in skip and isinstance(v, (str, int, float, bool))}
    return {"name": name.strip(), "arguments": args, "thought": _clip(str(blob.get("thought", "")), 200)}


# ---------------------------------------------------------------------------
# The router
# ---------------------------------------------------------------------------

@dataclass
class RouteResult:
    ok: bool = True
    track: str = "agent"                 # "instant" | "agent" | "agent-heuristic"
    answer: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    latency_ms: int = 0
    model: str = ""
    error: str = ""
    speak: bool = True
    pending: Optional[Dict[str, Any]] = None   # data the HUD renders as cards

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "track": self.track,
            "answer": self.answer,
            "tool_calls": self.tool_calls,
            "sources": self.sources,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "error": self.error,
            "speak": self.speak,
            "pending": self.pending,
        }


class Router:
    def __init__(self, memory: Optional[ConversationMemory] = None) -> None:
        #: :data:`llm_providers.POOL` rotates Groq / Cerebras / Cloudflare / Gemini /
        #: Mistral / OpenRouter / GitHub Models and cools down whoever is out of quota.
        self.brain = POOL
        self.memory = memory or MEMORY
        self._busy = threading.Lock()
        self.last_model = ""

    # -- public ------------------------------------------------------------
    def route(self, text: str, prefer_agent: bool = False) -> RouteResult:
        """Track 2 entry point. Track 1 (regex) lives in ``server.py``.

        The agent gets the first shot at everything that survived Track 1. The heuristic
        planner is a *repair* path, not a shortcut to DuckDuckGo: it may only fall through
        to a web search when the utterance is genuinely fact-shaped (see
        :func:`llm_providers.looks_like_research`) and ``LLM_FALLBACK_SEARCH`` is on -
        otherwise JARVIS says what is wrong instead of searching the word "hello"."""
        started = time.perf_counter()
        text = " ".join((text or "").split())
        if not text:
            return RouteResult(ok=False, answer="I did not catch that.", error="empty utterance")
        self.memory.add("user", text, track="agent")

        result = self.run_agent(text)
        if not result.ok:
            log.info("falling back to heuristic planner (%s)", _clip(result.error, 120))
            allow_search = bool(SETTINGS.llm_fallback_search) and looks_like_research(text)
            plan = heuristic_plan(text, allow_search=allow_search)
            if plan:
                result = self.run_tools(plan, text, track="agent-heuristic")
            else:
                # Name the reason in the HUD pill: "no provider key" and "all
                # providers cooling" look identical to a user otherwise.
                why = "no-provider-key" if "no LLM provider key" in (result.error or "") else "provider-unreachable"
                result = RouteResult(ok=False, track="agent-heuristic", model=why,
                                     answer=self._no_brain_answer(result.error),
                                     error=result.error, speak=True)
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        if result.answer:
            self.memory.add("assistant", result.answer, track=result.track,
                            tool=result.tool_calls[0]["tool"] if result.tool_calls else "")
        return result

    # -- agentic loop --------------------------------------------------------
    def run_agent(self, text: str) -> RouteResult:
        if not POOL.configured():
            return RouteResult(ok=False, answer="", model="",
                               error="no LLM provider key configured - add one to .env (see LLM_PROVIDER_ORDER)")
        tier = SETTINGS.llm_tier_mode if SETTINGS.llm_tier_mode in ("fast", "smart") else choose_tier(text)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(
                today=datetime.now().strftime("%d %B %Y"), weekday=datetime.now().strftime("%A"))},
        ]
        messages.extend(self.memory.recent())
        messages.append({"role": "user", "content": text})

        tool_log: List[Dict[str, Any]] = []
        sources: List[str] = []
        answer = ""
        try:
            for round_no in range(1, MAX_TOOL_ROUNDS + 1):
                completion = POOL.complete(messages, tools=TOOL_SCHEMAS, tier=tier)
                self.last_model = completion["provider"] + ":" + completion["model"]
                if completion.get("model_switched"):
                    # The provider's advertised model was refused and the pool recovered -
                    # worth a line in the log, because it means the .env id needs pinning.
                    log.info("%s answered with %s instead of the configured %s model",
                             completion["provider"], completion["model"], tier)
                content = completion["content"]
                native_calls = completion["tool_calls"]
                calls = list(native_calls) or ([c for c in [parse_tool_call_from_text(content)] if c])

                if not calls:
                    answer = _clean_answer(content)
                    if not answer and round_no == 1:
                        # A hub that answers with an empty message is a real failure:
                        # say so rather than claiming "Done." to the user.
                        log.warning("%s returned an empty completion", completion["provider"])
                        return RouteResult(ok=False, answer="", error="model returned no tool call and no text",
                                           model=self.last_model)
                    return RouteResult(ok=True, track="agent", answer=answer or "Done.", tool_calls=tool_log,
                                       sources=sources, model=self.last_model,
                                       pending=_pending_payload(tool_log))

                # Some hubs reject an assistant `tool_calls` message that they did
                # not actually emit, so only echo the field for native calls and
                # feed inline-JSON results back as a plain user turn.
                if native_calls:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content or "",
                            "tool_calls": [
                                {"id": c.get("id") or f"call_{i}", "type": "function",
                                 "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                                for i, c in enumerate(calls)
                            ],
                        }
                    )
                else:
                    messages.append({"role": "assistant", "content": content or ""})

                for idx, call in enumerate(calls):
                    name = call["name"]
                    tool, args, err = validate_call(name, call["arguments"])
                    if err:
                        payload = {"ok": False, "message": f"invalid tool call: {err}"}
                        tool_log.append({"tool": name, "arguments": call["arguments"], "error": err, "ok": False})
                    else:
                        outcome = tools.execute_tool(str(tool), args)
                        payload = {"ok": bool(outcome.get("ok", False)),
                                   "message": str(outcome.get("message", "")), **_card_data(outcome)}
                        tool_log.append({"tool": tool, "arguments": args, "ok": payload["ok"],
                                         "message": _clip(payload["message"], 400), "data": payload})
                        sources.extend(_collect_sources(outcome))
                    blob = json.dumps(payload, ensure_ascii=False, default=str)[:4000]
                    if native_calls:
                        messages.append({"role": "tool", "tool_call_id": call.get("id") or f"call_{idx}",
                                         "name": str(tool or name), "content": blob})
                    else:
                        messages.append({"role": "user", "content": f"TOOL_RESULT {tool or name}: {blob[:2600]}"})

                # Ask for the spoken summary once tools have run.
                final = POOL.complete(
                    messages + [{"role": "user", "content": "Summarise the tool results above for the user in 1-3 short spoken sentences. No JSON, no tool call."}],
                    tier="fast",
                    max_tokens=220,
                )
                self.last_model = final["provider"] + ":" + final["model"]
                answer = _clean_answer(final["content"])
                if answer:
                    return RouteResult(ok=True, track="agent", answer=answer, tool_calls=tool_log,
                                       sources=sources, model=self.last_model,
                                       pending=_pending_payload(tool_log))
                # No summary this round -> loop again (model may want another tool).
            return RouteResult(ok=True, track="agent", answer=answer or _spoken_from_tools(tool_log),
                               tool_calls=tool_log, sources=sources, model=self.last_model,
                               pending=_pending_payload(tool_log))
        except LlmError as exc:  # every provider failed or is cooling -> planner
            log.warning("agent track unavailable: %s", _clip(str(exc), 200))
            return RouteResult(ok=False, answer="", error=_clip(str(exc), 240), model=self.last_model)
        except Exception as exc:  # noqa: BLE001 - unexpected bug: still degrade, never crash
            log.warning("agent track errored: %s", _clip(str(exc), 200))
            log.warning("agent track unavailable: %s", _clip(str(exc), 200))
            return RouteResult(ok=False, answer="", error=_clip(str(exc), 240), model=self.last_model)

    def _no_brain_answer(self, reason: str) -> str:
        """What to say when no provider answered and a search would be nonsense."""
        if "no LLM provider key" in (reason or ""):
            return ("My AI brain has no key yet. Add one to .env - Groq is the quickest "
                    "(console.groq.com/keys), then Cerebras, Cloudflare or Gemini - and restart me. "
                    "Meanwhile I can still open apps, read your notes, report system stats, "
                    "set the volume and take screenshots.")
        return ("Every AI provider is busy or unreachable right now, so I did not guess. "
                + ("Last error: " + _clip(reason, 140) + ". " if reason else "")
                + "Ask me for the local stuff - notes, apps, system stats - or try again in a minute.")

    # -- direct execution (Track 1 reuses this) ------------------------------
    def run_tools(self, calls: List[Dict[str, Any]], utterance: str = "", track: str = "instant") -> RouteResult:
        tool_log, sources, answers = [], [], []
        pending: Optional[Dict[str, Any]] = None
        for call in calls or []:
            name = call.get("tool") or call.get("name") or ""
            tool, args, err = validate_call(name, call.get("arguments") or {})
            if err:
                answers.append(f"I could not run {name}: {err}")
                tool_log.append({"tool": name, "arguments": call.get("arguments") or {}, "ok": False, "error": err})
                continue
            outcome = tools.execute_tool(str(tool), args)
            answers.append(str(outcome.get("message") or ("Done." if outcome.get("ok") else "That failed.")))
            sources.extend(_collect_sources(outcome))
            entry = {
                "tool": tool,
                "arguments": args,
                "ok": bool(outcome.get("ok", False)),
                "message": _clip(str(outcome.get("message", "")), 400),
                "data": {"ok": bool(outcome.get("ok", False)), "message": str(outcome.get("message", "")), **_card_data(outcome)},
            }
            pending = _pending_payload([entry]) or pending
            tool_log.append(entry)
        self.memory.add("assistant", " ".join(answers), track=track, tool=tool_log[0]["tool"] if tool_log else "")
        return RouteResult(
            ok=all(t.get("ok", False) for t in tool_log) if tool_log else False,
            track=track,
            answer=" ".join(a for a in answers if a).strip() or "Nothing to do.",
            tool_calls=tool_log,
            sources=sources,
            model="local-regex-planner" if track == "instant" else "heuristic-planner",
            pending=pending,
        )

    # -- diagnostics ---------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "brain": POOL.status(),
            "tools": sorted(TOOL_NAMES),
            "memory_turns": len(self.memory.snapshot()),
            "max_tool_rounds": MAX_TOOL_ROUNDS,
            "fallback_search_allowed": SETTINGS.llm_fallback_search,
        }


CARD_FIELDS = (
    "results", "notes", "recent", "upcoming", "standing", "scorers",
    "query", "topic", "team", "path", "top_result", "bullets", "count",
)


def _card_data(outcome: Dict[str, Any]) -> Dict[str, Any]:
    """Trim a tool payload to the fields the HUD renders as cards / the LLM sees."""
    if not isinstance(outcome, dict):
        return {}
    data = {k: outcome[k] for k in CARD_FIELDS if k in outcome}
    if "results" in data and isinstance(data["results"], list):
        data["results"] = data["results"][:6]
    if "notes" in data and isinstance(data["notes"], list):
        data["notes"] = data["notes"][:4]
    return data


def _collect_sources(outcome: Dict[str, Any]) -> List[str]:
    found: List[str] = []
    for item in outcome.get("results") or []:
        if isinstance(item, dict):
            found.append(item.get("url") or item.get("title") or "")
    for note in outcome.get("notes") or []:
        if isinstance(note, dict):
            found.append(note.get("path") or note.get("name") or "")
    return [f for f in found if f][:6]


def _pending_payload(tool_log: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Structured data the HUD renders as cards (search results, notes, fixtures)."""
    for entry in reversed(tool_log or []):
        data = entry.get("data") or {}
        tool = entry.get("tool")
        if tool == "web_search" and data.get("results"):
            return {"type": "search", "query": data.get("query", ""), "items": data["results"][:6]}
        if tool == "read_notes" and data.get("notes"):
            return {"type": "notes", "topic": data.get("topic", ""), "items": data["notes"][:4]}
        if tool == "fetch_sports_stats" and (data.get("recent") or data.get("upcoming")):
            return {"type": "sports", "team": data.get("team", ""),
                    "recent": (data.get("recent") or [])[:5], "upcoming": (data.get("upcoming") or [])[:5],
                    "standing": data.get("standing"), "scorers": (data.get("scorers") or [])[:5]}
        if tool == "system_report" and data:
            return {"type": "system", "items": data}
    return None


def _spoken_from_tools(tool_log: List[Dict[str, Any]]) -> str:
    parts = [str(entry.get("message", "")).strip() for entry in tool_log or [] if entry.get("message")]
    return _clean_answer(" ".join(parts)) or "Executed."


def _clean_answer(text: str) -> str:
    text = strip_reasoning(text or "")
    text = text.strip()
    text = re.sub(r"^(```[a-z]*\s*|\s*```$)", "", text).strip()
    text = re.sub(r"^(assistant|jarvis)\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


ROUTER = Router()


def route(text: str) -> RouteResult:
    return ROUTER.route(text)


def execute_instant_calls(calls: List[Dict[str, Any]], utterance: str = "") -> RouteResult:
    return ROUTER.run_tools(calls, utterance, track="instant")


__all__ = [
    "ROUTER",
    "Router",
    "RouteResult",
    "TOOL_SCHEMAS",
    "TOOL_NAMES",
    "MEMORY",
    "ConversationMemory",
    "validate_call",
    "heuristic_plan",
    "route",
    "execute_instant_calls",
]
