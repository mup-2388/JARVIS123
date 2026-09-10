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
from llm_providers import LlmError, POOL, choose_tier, looks_like_research

log = get_logger("router")

MAX_TOOL_ROUNDS = 3
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
        "llm_status",
        "Report which AI providers are configured, which one answers, and which are cooling down "
        "after hitting a quota. For 'which model are you using', 'AI status', 'check the providers'.",
        {},
    ),
    _fn(
        "launch_app",
        "Start a Windows application (Steam, Discord, the Eden emulator for FC 26, Chrome/Edge/Firefox, VS Code, Terminal, Spotify, Calculator...). "
        "Use for 'open/launch/start <app>'. Returns ok=false with a hint when the app is unknown.",
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
    (re.compile(r"\b(open|launch|start|boot up|fire up)\b[^.]*?\b(chrome|edge|firefox|browser|steam|discord|spotify|notepad|calculator|code|vs ?code|terminal|powershell|explorer|obs|paint|settings|task ?manager|eden|fc ?26)\b", re.I),
     "launch_app", lambda m: {"app_name": m.group(2).strip()}),
    (re.compile(r"\b(close|quit|kill|exit|terminate)\b[^.]*?\b(chrome|edge|firefox|browser|steam|discord|spotify|notepad|calculator|code|terminal|explorer|obs|eden)\b", re.I),
     "close_app", lambda m: {"app_name": m.group(2).strip()}),
    (re.compile(r"\b(play|put on|queue up)\b(?:\s+(?:the\s+|my\s+))?(.+?)(?:\s+on\s+youtube|\s+on\s+yt|\?|$)", re.I),
     "play_youtube", lambda m: {"query": (m.group(2) or "").strip(" .,?!")}),
    # "search X on youtube" / "search youtube for X" - the named site owns it.
    (re.compile(r"\b(?:search|look up|google)\b(?:\s+for)?\s+(?P<q>.+?)\s+(?:on|in|at)\s+(?P<site>[a-z][a-z0-9 ._-]{1,24}(?:\.[a-z]{2,})?)$", re.I),
     "search_on_site", lambda m: {"site": (m.group("site") or "").strip(), "query": (m.group("q") or "").strip(" .?"), "open_browser": True}),
    (re.compile(r"\bsearch\s+(?P<site>[a-z][a-z0-9 ._-]{1,24}?)\s+for\s+(?P<q>.+)$", re.I),
     "search_on_site", lambda m: {"site": (m.group("site") or "").strip(), "query": (m.group("q") or "").strip(" .?"), "open_browser": True}),
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
    "write_note": 90, "read_notes": 88, "fetch_sports_stats": 86, "system_power": 84,
    "take_screenshot": 80, "set_volume": 78, "close_app": 74, "launch_app": 72,
    "play_youtube": 70, "search_on_site": 66, "open_website": 60, "system_report": 58, "get_time": 55,
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
        if not args and name not in {"get_time", "take_screenshot"}:
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
    if len(calls) > 3:
        calls = calls[:3]
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
1. Act, do not speculate. When the user asks for anything on this PC (open/close apps, volume, screenshot, telemetry) or for live facts (prices, restaurants, news, football scores, their own study notes), call the matching tool instead of answering from memory.
2. One tool call at a time is fine, but you may call several tools when the request has several parts ("open Steam and tell me my RAM").
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
    text = (text or "").strip()
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
