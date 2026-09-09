"""
server.py -- FastAPI + WebSocket core, and Track 1 of the two-track design.

Latency model
-------------
    utterance ─► INSTANT_RULES (compiled regex, <1 ms)
                    │  hit  ─► tools.execute_tool()  ─► reply in ~5-80 ms
                    │                                    (never touches the LLM)
                    └─ miss ─► router.ROUTER.route()  ─► HF tool-calling agent
                                                        (real tool execution, 0.6-4 s)

Why two tracks: "open Steam" through an LLM costs a network round trip and a
sampling pass for no benefit. The regex layer owns the deterministic 90 % of a
desktop assistant's traffic; the agent owns everything ambiguous, multi-tool or
research-y.

Protocol (WebSocket, /ws)
-------------------------
    client → server  {"type":"command","text":"...","speak":true}
                     {"type":"speak","text":"..."}
                     {"type":"stop"}                       -> kill TTS queue
                     {"type":"mic","format":"webm|wav|pcm","data":"<base64>"}
                     <binary frame>                        -> raw s16le mono 16 kHz
                     {"type":"ping"}
    server → client  {"type":"hello"|"telemetry"|"log"|"state"|"reply"|"card"|"transcript"|"audio"|"ack"|"error"}

The same JSON payloads are available over REST (``POST /api/command``) which is
what :mod:`discord_bridge` and curl use, so behaviour is identical everywhere.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import math
import os
import re
import struct
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import config
import router
import tools
from audio_engine import ENGINE as voice
from audio_engine import TARGET_SR
from config import SETTINGS, get_logger
from fastapi import Body, FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

log = get_logger("server")

VERSION = "1.0.0"
BOOT_TS = time.time()
TTS_DIR = SETTINGS.cache_path / "tts"
TTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Track 1 -- instant regex rules
# ---------------------------------------------------------------------------

Wake = r"(?:hey[ ,]+|ok[ ,]+|oh[ ,]+)?(?:jarvis|jervis|computer)[ ,:!]+"
Strip = rf"^\s*{Wake}?|{Wake}\s*"
_STRIP_RE = re.compile(Strip, re.I)

# Group 1 == "app / site" phrase for open-style commands.
_OPEN_RE = re.compile(
    r"^(?:could you |please |can you )?(?:open|launch|start|boot(?: up)?|fire up|spin up|bring up|run)\s+"
    r"(?:up\s+)?(?:the\s+|my\s+)?(?P<target>[a-z0-9 ._+-]{1,60}?)"
    r"(?:\s+(?:app|application|exe|program|up))?$",
    re.I,
)
_CLOSE_RE = re.compile(
    r"^(?:please )?(?:close|quit|kill|exit|terminate|shut)\s+(?:the\s+|my\s+)?(?P<target>[a-z0-9 ._+-]{1,60}?)(?:\s+(?:app|application|down|now))?$",
    re.I,
)
_PLAY_RE = re.compile(r"^(?:please )?(?:play|put on|queue up|stream)\s+(?P<what>.+?)(?:\s+(?:on|in)\s+(?:youtube|yt|the youtube))?$", re.I)
_SEARCH_RE = re.compile(
    r"^(?:please )?(?:search(?: for)?|google|look up|find(?: out)?|what(?:'s| is| are)|how much is|price of|who is|who was)\s+(?P<q>.{2,180}?)[?.!]*$",
    re.I,
)
_NOTE_READ_RE = re.compile(r"^(?:read|show|open|check|pull up|what do)\b.*?\bnotes?\b(?:.*?\babout|for|on)?\s*(?P<topic>[a-z0-9 äöüß ._+-]{0,60})$", re.I)
_NOTE_WRITE_RE = re.compile(r"^(?:jarvis[ ,]*)?(?:note|write|jot|take a note|remember)\s*(?:that|down|:)?\s*(?P<body>.{3,600})$", re.I)
_TEAM_WORDS = (
    r"real madrid|fc barcelona|barcelona|manchester city|manchester united|liverpool|arsenal|chelsea|tottenham|"
    r"newcastle|bayern munich|bayern|dortmund|bvb|psg|juventus|inter milan|ac milan|napoli|roma|atletico|"
    r"sevilla|porto|sporting|celtic|rangers|mumbai city|bengaluru fc|kerala blasters|east bengal|fc goa"
)
_SPORTS_RE = re.compile(
    rf"\b(?:(?:how did|how are|score of|result for|next match(?: of)?|fixtures? for|standing(?:s)? for|table for|top scorer(?:s)? of)\s+(?P<t1>{_TEAM_WORDS})"
    rf"|(?P<t2>{_TEAM_WORDS})\s+(?:score|result|fixture|fixtures|standing|standings|form|stats|stats?|match))\b",
    re.I,
)
_TIME_RE = re.compile(
    r"^(?:please )?(?:what(?:'s| is) the |tell me the |current |what |what's )?"
    r"(?:time|date|day)(?:\s+(?:is\s+it|today|right\s+now|now))?[?.!]*$",
    re.I,
)
_TELEMETRY_RE = re.compile(
    r"\b(?:cpu|ram|memory|gpu|vram|disk|ssd|battery|uptime)\b[^?!.\n]{0,20}"
    r"\b(?:usage|use|load|percent|temp|temperature|status|stats?|statistics|report|level|high|hot|space|capacity|free)\b"
    r"|\b(?:how much|free|available)\b[^?!.\n]{0,12}\b(?:ram|memory|vram|disk|ssd|space)\b"
    r"|\bhow'?s (?:my|the) (?:pc|system|machine|rig|laptop)\b"
    r"|\bhow(?:'s| is| are)?\b[^?!.\n]{0,14}\b(?:my|the)\b[^?!.\n]{0,10}\b(?:cpu|ram|memory|gpu|vram|disk|ssd|battery|pc|system|machine|rig|laptop)\b"
    r"|\b(?:system|hardware|core) (?:report|status|health)\b|\btelemetry\b|\bdiagnostics\b",
    re.I,
)
_VOLUME_ABS_RE = re.compile(r"^(?:set |change )?(?:the )?volume (?:to |at |= )?(?P<level>\d{1,3})\s*(?:percent|%)?\b", re.I)
_VOLUME_REL_RE = re.compile(
    r"\b(?:volume|it)\s+(?:up|down)\b|\b(?:louder|quieter)\b|\bturn (?:it |the volume |the sound )(?:up|down)\b|\bmute\b|\bunmute\b",
    re.I,
)
_VOLUME_RE = re.compile(_VOLUME_ABS_RE.pattern + r"|" + _VOLUME_REL_RE.pattern, re.I)
_SCREENSHOT_RE = re.compile(r"\b(screenshot|screen ?shot|capture (?:the )?(?:screen|display)|snap (?:the )?screen)\b", re.I)
_POWER_RE = re.compile(r"\b(lock (?:the )?(?:pc|screen|workstation)|sleep (?:the )?(?:pc|machine|now)|shut ?down|power ?off|restart|reboot|cancel (?:the )?shutdown)\b", re.I)
_APPS_LIST_RE = re.compile(r"\bwhat (?:apps|programs)(?: can you)?|list (?:launchable )?apps|what can you open\b", re.I)
_GREETING_RE = re.compile(r"^(?:hi|hello|hey|yo|good (?:morning|afternoon|evening)|are you (?:there|awake|alive)|status|report in)\b", re.I)
_STOP_RE = re.compile(
    r"^(?:jarvis[ ,]*)?(?:stop|cancel|quiet|shut up|never ?mind|enough|pause|hold on)"
    r"(?:[ ,]+(?:that|it|please|jarvis))*[.!?]*$",
    re.I,
)
_WAKEUP_RE = re.compile(r"\b(wake up|warm up|spin up the (?:model|gpu|voice)|go to sleep|unload models)\b", re.I)
_CLEAR_RE = re.compile(r"\b(clear|forget) (?:the )?(?:history|context|terminal)\b", re.I)

CONFIRM_WORDS = {"yes", "confirm", "do it", "go ahead", "affirmative", "sure", "yep", "yeah"}
DENY_WORDS = {"no", "cancel", "never mind", "abort", "negative"}


def split_clauses(text: str) -> List[str]:
    """Split a compound utterance on conjunctions so one breath = several actions."""
    parts = [p.strip(" ,.") for p in re.split(r"\s+(?:and|then|also|plus|after that)\s+", text or "", flags=re.I)]
    return [p for p in parts if p]


def strip_wake(text: str) -> str:
    return _STRIP_RE.sub(" ", text or "").strip(" ,.!?").strip()


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

class Terminal:
    """Ring buffer of HUD terminal lines + fan-out to every WebSocket."""

    def __init__(self, capacity: int = 300) -> None:
        self.lines: List[Dict[str, Any]] = []
        self.capacity = capacity
        self.hub: Optional["Hub"] = None

    def push(self, level: str, text: str, extra: Optional[Dict[str, Any]] = None) -> None:
        entry = {"at": datetime.now().strftime("%H:%M:%S"), "level": level, "text": text, **(extra or {})}
        self.lines.append(entry)
        if len(self.lines) > self.capacity:
            del self.lines[: len(self.lines) - self.capacity]
        if self.hub:
            self.hub.emit({"type": "log", **entry})


TERMINAL = Terminal()


class Hub:
    """Thread-safe WebSocket fan-out (tools run inside ``asyncio.to_thread``)."""

    def __init__(self) -> None:
        self._clients: "set[WebSocket]" = set()
        self._queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue(maxsize=512)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._pump: Optional[asyncio.Task] = None
        self.sent = 0

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        if self._pump is None or self._pump.done():
            self._pump = loop.create_task(self._drain())

    async def add(self, ws: WebSocket) -> None:
        self._clients.add(ws)
        log.info("HUD client connected (%d live)", len(self._clients))

    async def remove(self, ws: WebSocket) -> None:
        self._clients.discard(ws)
        log.info("HUD client left (%d live)", len(self._clients))

    @property
    def count(self) -> int:
        return len(self._clients)

    def emit(self, payload: Dict[str, Any]) -> None:
        """Fire-and-forget from any thread (used by tools, TTS and the router)."""
        if not self._clients:
            return
        try:
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._enqueue, payload)
            else:
                self._enqueue(payload)
        except RuntimeError:
            pass

    def _enqueue(self, payload: Dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(payload)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(payload)
            except asyncio.QueueEmpty:
                pass

    async def send(self, payload: Dict[str, Any]) -> None:
        dead: "set[WebSocket]" = set()
        text = json.dumps(payload, ensure_ascii=False, default=str)
        for ws in list(self._clients):
            try:
                await ws.send_text(text)
            except Exception:  # noqa: BLE001 - closed socket / backpressure
                dead.add(ws)
        self._clients.difference_update(dead)
        self.sent += 1

    async def _drain(self) -> None:
        while True:
            payload = await self._queue.get()
            await self.send(payload)


HUB = Hub()
TERMINAL.hub = HUB
BRIDGE = None            # discord_bridge.DiscordBridge once the lifespan starts it


def mirror_to_discord(text: str) -> bool:
    """Push an assistant line to the bound Discord channel when mirroring is on."""
    if not (_STATE.get("mirror") and BRIDGE is not None and text):
        return False
    try:
        return bool(BRIDGE.send(text))
    except Exception as exc:  # noqa: BLE001
        log.debug("discord mirror skipped: %s", exc)
        return False

_STATE: Dict[str, Any] = {
    "mode": "idle",                    # idle | listening | thinking | speaking
    "pending_confirm": None,           # {"calls":[...], "label": "..."}
    "last_error": "",
    "commands": 0,
    "instant_hits": 0,
    "agent_hits": 0,
    "history": [],                     # last 40 payloads for the HUD cards
    "muted": False,
    "discord": {"enabled": False, "connected": False, "detail": "not started"},
    "mirror": False,                     # relay local replies into the Discord channel
    "voice": voice.status(),
}


def set_mode(mode: str, detail: str = "") -> None:
    _STATE["mode"] = mode
    HUB.emit({"type": "state", "mode": mode, "detail": detail})


def push_card(payload: Dict[str, Any]) -> None:
    _STATE["history"].append({**payload, "at": datetime.now().strftime("%H:%M:%S")})
    if len(_STATE["history"]) > 40:
        del _STATE["history"][:-40]
    HUB.emit({"type": "card", **payload})


# ---------------------------------------------------------------------------
# Telemetry pump
# ---------------------------------------------------------------------------

def _telemetry_snapshot() -> Dict[str, Any]:
    report = tools.system_report(detailed=True)
    return {
        "at": datetime.now().strftime("%H:%M:%S"),
        "cpu": report.get("cpu", 0.0),
        "per_core": report.get("per_core", []),
        "cpu_count": report.get("cpu_count", 0),
        "load": report.get("load", []),
        "ram": report.get("ram", 0.0),
        "ram_used_gb": report.get("ram_used_gb", 0),
        "ram_total_gb": report.get("ram_total_gb", 0),
        "swap": report.get("swap", 0.0),
        "disk": report.get("disk", 0.0),
        "disk_free_gb": report.get("disk_free_gb", 0),
        "gpu": report.get("gpu", {}),
        "net_sent_mb": report.get("net_sent_mb", 0),
        "net_recv_mb": report.get("net_recv_mb", 0),
        "battery": report.get("battery", {}),
        "uptime_s": int(time.time() - BOOT_TS),
        "boot_time": report.get("boot_time", ""),
        "process_count": report.get("process_count", 0),
        "top_processes": report.get("top_processes", []),
        "mode": _STATE["mode"],
        "clients": HUB.count,
        "stats": {
            "commands": _STATE["commands"],
            "instant": _STATE["instant_hits"],
            "agent": _STATE["agent_hits"],
            "hub_sent": HUB.sent,
            "latency_ms": _STATE.get("last_latency_ms", 0),
        },
    }


async def _telemetry_loop() -> None:
    period = max(0.3, SETTINGS.telemetry_ms / 1000)
    tools.system_report()          # prime psutil's per-core counters
    while True:
        try:
            payload = await asyncio.to_thread(_telemetry_snapshot)
            await HUB.send({"type": "telemetry", "data": payload})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.debug("telemetry tick failed: %s", exc)
        await asyncio.sleep(period)


# ---------------------------------------------------------------------------
# Command pipeline
# ---------------------------------------------------------------------------

Rule = Tuple[str, re.Pattern[str], Callable[[re.Match[str], str], Optional[Dict[str, Any]]]]


def _calls(*pairs: Tuple[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {"calls": [{"tool": name, "arguments": args} for name, args in pairs]}


def _say(text: str, speak: Optional[bool] = None) -> Dict[str, Any]:
    return {"answer": text, "direct": True, "speak": True if speak is None else speak}


def _rule_greeting(m: re.Match[str], text: str) -> Optional[Dict[str, Any]]:
    hour = datetime.now().hour
    part = "morning" if 5 <= hour < 12 else ("afternoon" if 12 <= hour < 17 else ("evening" if 17 <= hour < 22 else "night"))
    stt = voice.stt.status["state"]
    tts = voice.tts.status["state"]
    return _say(
        f"Good {part}, sir. All systems nominal — {HUB.count} HUD link(s), "
        f"speech-to-text {stt}, voice {tts}. Standing by."
    )


def _rule_telemetry(m: re.Match[str], text: str) -> Optional[Dict[str, Any]]:
    return _calls(("system_report", {"detailed": True}))


def _looks_like_domain(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9][a-z0-9-]*(\.[a-z]{2,})+(/[^\s]*)?", value.strip().lower()))


def _rule_open(m: re.Match[str], text: str) -> Optional[Dict[str, Any]]:
    """Track 1 "open X [and Y]" -- splits into several real tool calls.

    ``open chrome and play lofi`` and ``open steam and check my cpu`` are two
    intents in one breath; the agent would burn 3 s on them, so we split them
    here on a whitespace-"and" boundary and dispatch each half instantly.
    """
    target = (m.group("target") or "").strip()
    if not target:
        return None
    parts = split_clauses(target)
    primary = parts[0] if parts else target
    low = primary.lower()
    if not (_looks_like_domain(low) or low in {"youtube", "yt"} or tools.is_known_app(primary)):
        # Track 1 never guesses: unknown apps go to the agent.
        return None

    calls: List[Tuple[str, Dict[str, Any]]] = []
    if _looks_like_domain(low):
        calls.append(("open_website", {"target": primary, "query": ""}))
    elif low in {"youtube", "yt"}:
        calls.append(("open_website", {"target": "youtube", "query": ""}))
    else:
        url_hint = re.search(r"\b(?:at|to|on)\s+(https?://\S+|\S+\.\w{2,}(?:/\S*)?)\b", text, re.I)
        calls.append(("launch_app", {"app_name": primary,
                                    "url": url_hint.group(1) if url_hint else "",
                                    "args": ""}))
    unparsed: List[str] = []
    for rest in parts[1:]:
        rest_low = rest.lower().strip(" ,.")
        play = re.match(r"^(?:play|put on|queue up|stream)\s+(.+)$", rest_low, re.I)
        if play:
            calls.append(("play_youtube", {"query": play.group(1).strip(" ?.")}))
        elif re.search(r"\b(cpu|ram|gpu|vram|memory|disk|ssd|battery|temp|temperature|status|load|usage|telemetry|diagnostics)\b", rest_low):
            calls.append(("system_report", {"detailed": True}))
        elif re.search(r"\b(search|google|look up|find out|find)\b", rest_low):
            query = re.sub(r"^.*?\b(?:search|google|look up|find out|find)\b\s+(?:for\s+)?", "", rest_low, flags=re.I).strip(" ?.")
            if query:
                calls.append(("web_search", {"query": query, "max_results": 5, "timelimit": "", "site": ""}))
            else:
                unparsed.append(rest_low)
        elif re.search(r"\b(screenshot|screen ?shot)\b", rest_low):
            calls.append(("take_screenshot", {}))
        elif re.search(r"^(?:note|write|jot|remember)\b\s*(?:that|down)\b", rest_low):
            # "...and note that X" is dictation, not a lookup. Keep the user's
            # original casing in the body: "FC 26" must not become "fc 26".
            body = re.sub(r"^(?:note|write|jot|remember)\s+(?:that|down)\s*[:,]*", "", rest, flags=re.I).strip(" ?.")
            if body:
                calls.append(("write_note", {"topic": " ".join(body.split()[:5]).capitalize()[:60],
                                             "content": body[:900], "tags": "voice-memo"}))
            else:
                unparsed.append(rest_low)
        elif re.search(r"\bnotes?\b", rest_low):
            topic = re.sub(r".*\bnotes?(?: about| on| for)?\b", "", rest_low, flags=re.I).strip(" ?.") or rest_low
            calls.append(("read_notes", {"topic": topic[:60], "limit": 1}))
        elif _looks_like_domain(rest_low) or re.match(r"^(?:open|go to)\b", rest_low):
            site = re.sub(r"^(?:open|go to)\s+", "", rest_low).strip()
            calls.append(("open_website", {"target": site, "query": ""}))
        elif re.search(r"\b(time|clock|date)\b", rest_low):
            calls.append(("get_time", {}))
        elif rest_low:
            # Never invent an action for a clause we cannot parse: report it.
            unparsed.append(rest_low)
    if unparsed:
        quoted = '", "'.join(unparsed[:2])
        plan_note = (f'I ran the "{primary}" part, but I could not map "{quoted}" — '
                     + ("ask me those separately." if len(unparsed) > 1 else "ask me that on its own."))
    else:
        plan_note = ""
    if len(calls) == 1 and low in {"youtube", "yt"} and parts[1:]:
        calls = [("play_youtube", {"query": parts[1].strip(" ?.")})]
    plan = _calls(*calls)
    plan["answer_prefix"] = f"Opening {primary}." if len(calls) > 1 else ""
    if plan_note:
        plan["unparsed"] = unparsed
        plan["answer_prefix"] = (plan["answer_prefix"] + " " + plan_note).strip()
    return plan


def _rule_close(m: re.Match[str], text: str) -> Optional[Dict[str, Any]]:
    target = (m.group("target") or "").strip()
    if not target:
        return None
    if not tools.is_known_app(target) and not re.fullmatch(r"(everything|all apps|all)", target, re.I):
        return None
    if re.fullmatch(r"(everything|all apps|all)", target, re.I):
        return {
            "answer": "Closing the apps I manage.",
            "calls": [{"tool": "close_app", "arguments": {"app_name": name}} for name in ("chrome", "discord", "spotify", "steam")],
        }
    return _calls(("close_app", {"app_name": target}))


def _rule_play(m: re.Match[str], text: str) -> Optional[Dict[str, Any]]:
    what = (m.group("what") or "").strip(" ?.")
    if not what:
        return _calls(("open_website", {"target": "youtube", "query": ""}))
    return _calls(("play_youtube", {"query": what}))


def _rule_search(m: re.Match[str], text: str) -> Optional[Dict[str, Any]]:
    query = (m.group("q") or "").strip(" ?.,")
    if len(query) < 3:
        return None
    recent = re.search(r"\b(latest|today|breaking|news)\b", text, re.I)
    return _calls(("web_search", {"query": query, "max_results": 6, "timelimit": "d" if recent else "", "site": ""}))


def _rule_note_read(m: re.Match[str], text: str) -> Optional[Dict[str, Any]]:
    """``read my notes about german dative and how is my cpu`` -> two calls.

    The topic group can swallow a trailing clause, so compound asks are split
    here the same way :func:`_rule_open` does it.
    """
    topic = (m.group("topic") or "").strip(" ?.")
    if not topic:
        topic = " ".join(
            w for w in re.findall(r"[a-zäöüß]{3,}", text.lower())
            if w not in {"read", "show", "open", "check", "pull", "what", "your", "about", "notes"}
        )
    clauses = split_clauses(topic)
    head, tails = (clauses[0], clauses[1:]) if clauses else (topic, [])
    calls: List[Tuple[str, Dict[str, Any]]] = [("read_notes", {"topic": head[:60], "limit": 2})]
    unparsed: List[str] = []
    for tail in tails:
        low = tail.lower().strip(" ,.")
        if re.search(r"^(?:note|write|jot|remember)\s+(?:that|down)\b", low):
            body = re.sub(r"^(?:note|write|jot|remember)\s+(?:that|down)\s*[:,]*", "", tail, flags=re.I).strip(" ?.")
            if body:
                calls.append(("write_note", {"topic": " ".join(body.split()[:5]).capitalize()[:60],
                                             "content": body[:900], "tags": "voice-memo"}))
            else:
                unparsed.append(low)
        elif re.search(r"\b(cpu|ram|gpu|memory|disk|vram|battery|status|load|usage|telemetry|temp|temperature)\b", low):
            calls.append(("system_report", {"detailed": True}))
        elif re.search(r"\b(time|clock|date)\b", low):
            calls.append(("get_time", {}))
        elif re.search(r"\b(screenshot|screen ?shot)\b", low):
            calls.append(("take_screenshot", {}))
        elif re.search(r"\b(search|google|look up|find out|find)\b", low):
            query = re.sub(r"^.*?\b(?:search|google|look up|find out|find)\b\s+(?:for\s+)?", "", tail, flags=re.I).strip(" ?.")
            if query:
                calls.append(("web_search", {"query": query, "max_results": 5, "timelimit": "", "site": ""}))
            else:
                unparsed.append(low)
        elif re.search(r"\bnotes?\b", low):
            extra = re.sub(r".*\bnotes?\b(?: about| on| for)?", "", low, flags=re.I).strip(" ?.")
            calls.append(("read_notes", {"topic": (extra or low)[:60], "limit": 1}))
        else:
            # Never invent an action for a clause we cannot parse.
            unparsed.append(low)
    plan = _calls(*calls)
    if unparsed:
        plan["answer_prefix"] = (f'I looked up "{head}" as asked, but I could not map '
                                 f'"{chr(34).join(unparsed[:2])}" — ask me that on its own.')
        plan["unparsed"] = unparsed
    return plan


def _rule_note_write(m: re.Match[str], text: str) -> Optional[Dict[str, Any]]:
    body = (m.group("body") or "").strip(" ?.")
    if len(body) < 4:
        return None
    title = " ".join(body.split()[:5]).capitalize()
    return _calls(("write_note", {"topic": title[:60], "content": body, "tags": "voice-memo"}))


def _rule_sports(m: re.Match[str], text: str) -> Optional[Dict[str, Any]]:
    team = (m.group("t1") or m.group("t2") or "").strip().title()
    low = text.lower()
    kind = "all"
    if "standing" in low or "table" in low:
        kind = "standing"
    elif "next" in low or "fixture" in low:
        kind = "fixtures"
    elif "scorer" in low or "top" in low:
        kind = "scorers"
    elif "result" in low or "how did" in low or "score" in low:
        kind = "results"
    return _calls(("fetch_sports_stats", {"team": team or SETTINGS.api_sports_default_team, "kind": kind}))


def _rule_volume(m: re.Match[str], text: str) -> Optional[Dict[str, Any]]:
    absolute = _VOLUME_ABS_RE.match(text)
    if absolute:
        level = max(0, min(100, int(absolute.group("level"))))
        return _calls(("set_volume", {"level": level, "delta": 0, "mute": False}))
    low = text.lower()
    if re.search(r"\bunmute\b", low):
        return _calls(("set_volume", {"level": -1, "delta": 0, "mute": True}))
    if re.search(r"\bmute\b", low):
        _STATE["muted"] = True
        return _calls(("set_volume", {"level": -1, "delta": 0, "mute": True}))
    up = bool(re.search(r"\b(up|louder|increase|raise|max)\b", low))
    step = 12 if re.search(r"\b(?:way|a lot|lots|double)\b", low) else SETTINGS.volume_step
    _STATE["muted"] = False
    return _calls(("set_volume", {"level": -1, "delta": step if up else -step, "mute": False}))


def _rule_power(m: re.Match[str], text: str) -> Optional[Dict[str, Any]]:
    low = text.lower()
    if "cancel" in low:
        return _calls(("system_power", {"action": "cancel-shutdown"}))
    action = "lock" if "lock" in low else ("sleep" if "sleep" in low else ("shutdown" if re.search(r"shut ?down|power ?off", low) else "restart"))
    args = {"action": action}
    if action in {"shutdown", "restart"}:
        return {"calls": [dict(tool="system_power", **{"arguments": args})], "confirm": f"About to {action} this machine. Say “confirm” or “cancel”."}
    return _calls(("system_power", args))


def _rule_wake(m: re.Match[str], text: str) -> Optional[Dict[str, Any]]:
    low = text.lower()
    if "sleep" in low or "unload" in low:
        return _say("Going quiet. Models stay cached until you wake me.")
    threading_warm = voice.warmup()
    return _say(f"Warming up: speech-to-text {threading_warm['stt']}, voice {threading_warm['tts']}.")


INSTANT_RULES: List[Rule] = [
    ("greeting", _GREETING_RE, _rule_greeting),
    ("clear", _CLEAR_RE, lambda m, t: {"answer": "Context cleared.", "direct": True, "speak": True, "clear_history": True}),
    ("stop", _STOP_RE, lambda m, t: {"answer": "Stopped.", "direct": True, "speak": False, "stop_tts": True}),
    ("time", _TIME_RE, lambda m, t: _calls(("get_time", {}))),
    ("open", _OPEN_RE, _rule_open),
    ("close", _CLOSE_RE, _rule_close),
    ("play", _PLAY_RE, _rule_play),
    ("apps-list", _APPS_LIST_RE, lambda m, t: _calls(("list_launchable_apps", {}))),
    ("screenshot", _SCREENSHOT_RE, lambda m, t: _calls(("take_screenshot", {}))),
    ("volume", _VOLUME_RE, _rule_volume),
    ("sports", _SPORTS_RE, _rule_sports),
    ("note-write", _NOTE_WRITE_RE, _rule_note_write),
    ("note-read", _NOTE_READ_RE, _rule_note_read),
    ("telemetry", _TELEMETRY_RE, _rule_telemetry),
    ("power", _POWER_RE, _rule_power),
    ("wake", _WAKEUP_RE, _rule_wake),
    ("search", _SEARCH_RE, _rule_search),
]

#: Tools that are safe to run without the router's schema check (Track 1 fast path).
TRACK1_EXTRA_TOOLS = {"list_launchable_apps": tools.list_launchable_apps}


def match_instant(text: str) -> Optional[Dict[str, Any]]:
    """Track 1: return a plan dict when a regex owns the utterance, else ``None``."""
    if not SETTINGS.instant_track_enabled:
        return None
    cleaned = strip_wake(text)
    if not cleaned:
        return None
    for name, pattern, handler in INSTANT_RULES:
        match = pattern.search(cleaned) or pattern.match(cleaned)
        if not match:
            continue
        try:
            plan = handler(match, cleaned)
        except (re.error, IndexError, AttributeError, TypeError) as exc:  # noqa: BLE001
            log.debug("instant rule %s misfired: %s", name, exc)
            continue
        if plan:
            plan.setdefault("rule", name)
            return plan
    return None


async def execute_plan(plan: Dict[str, Any], utterance: str) -> router.RouteResult:
    """Run a Track 1 plan. Extra tools outside the LLM allow-list are honoured."""
    calls = plan.get("calls") or []
    if not calls:
        return router.RouteResult(ok=True, track="instant", answer=plan.get("answer", ""), model="local-regex")
    prepared, extra_answers = [], []
    for call in calls:
        tool = call.get("tool", "")
        if tool in TRACK1_EXTRA_TOOLS:
            outcome = TRACK1_EXTRA_TOOLS[tool]()
            extra_answers.append(str(outcome.get("message", "")))
            TERMINAL.push("tool", f"{tool}() -> {_trim(str(outcome.get('message', '')), 160)}")
            continue
        prepared.append(call)
    result = router.ROUTER.run_tools(prepared, utterance, track="instant")
    prefix = str(plan.get("answer_prefix") or "").strip()
    if extra_answers:
        result.answer = " ".join([a for a in extra_answers if a] + [result.answer]).strip()
    if prefix and prefix not in result.answer:
        result.answer = f"{prefix} {result.answer}".strip()
    return result


async def handle_command(
    text: str,
    source: str = "hud",
    speak: bool = SETTINGS.speak_replies,
    force_agent: bool = False,
) -> Dict[str, Any]:
    """Single funnel for HUD / REST / Discord. Returns a JSON-ready payload."""
    started = time.perf_counter()
    text = " ".join((text or "").split())
    if not text:
        return {"ok": False, "error": "empty command", "answer": ""}

    # `_STATE["muted"]` gates only JARVIS's own speech (set by the "stop" rule and
    # by "mute"); a fresh utterance means the user is talking to us again, so
    # speech comes back on. Windows' own mixer state is untouched either way.
    if _STATE.get("muted") and not _STOP_RE.search(strip_wake(text)):
        _STATE["muted"] = False
    _STATE["commands"] += 1
    _STATE.setdefault("last_latency_ms", 0)
    TERMINAL.push("in", f"[{source}] {text}")
    set_mode("thinking", text[:60])

    # --- pending destructive confirmation (shutdown / restart) --------------
    pending = _STATE.get("pending_confirm")
    if pending:
        lowered = text.lower().strip(" .!?")
        _STATE["pending_confirm"] = None
        if any(lowered.startswith(w) or lowered == w for w in DENY_WORDS):
            answer = "Cancelled. Nothing was touched."
            TERMINAL.push("warn", answer)
            return await _finish(text, answer, "instant", started, speak, source, [])
        if lowered in CONFIRM_WORDS or lowered.startswith("confirm"):
            result = await execute_plan({"calls": pending.get("calls") or []}, text)
            return await _finish(text, result.answer, "instant", started, speak, source, result.tool_calls, pending=result.pending)
        TERMINAL.push("warn", "Confirmation expired — I re-routed that as a new command.")

    # --- Track 1 ------------------------------------------------------------
    plan = None if force_agent else match_instant(text)
    if plan and plan.get("confirm"):
        _STATE["pending_confirm"] = {"calls": plan.get("calls", []), "label": plan["confirm"]}
        _STATE["instant_hits"] += 1
        answer = plan["confirm"]
        TERMINAL.push("warn", answer)
        return await _finish(text, answer, "instant-confirm", started, True, source, [])
    if plan and (plan.get("calls") or plan.get("direct")):
        _STATE["instant_hits"] += 1
        if plan.get("stop_tts"):
            voice.tts._queue.clear()
            set_mode("idle")
            return {"ok": True, "track": "instant", "answer": "Stopped.", "latency_ms": 0, "tool_calls": [], "speak": False}
        if plan.get("clear_history"):
            router.MEMORY.clear()
            TERMINAL.lines.clear()
            HUB.emit({"type": "cleared"})
            return {"ok": True, "track": "instant", "answer": "Context cleared.", "latency_ms": 0,
                    "tool_calls": [], "speak": bool(plan.get("speak", False)), "source": source}
        if plan.get("direct"):
            answer = plan.get("answer", "")
            return await _finish(text, answer, "instant", started, speak, source, [])
        result = await execute_plan(plan, text)
        return await _finish(text, result.answer, "instant", started, speak, source, result.tool_calls, pending=result.pending)

    # --- Track 2 (agentic) ---------------------------------------------------
    _STATE["agent_hits"] += 1
    try:
        result = await asyncio.to_thread(router.ROUTER.route, text)
    except Exception as exc:  # noqa: BLE001
        log.exception("router exploded")
        _STATE["last_error"] = str(exc)
        result = router.RouteResult(ok=False, answer=f"The agent crashed: {exc}", error=str(exc))
    return await _finish(
        text,
        result.answer or (result.error or "No answer."),
        result.track,
        started,
        speak,
        source,
        result.tool_calls,
        pending=result.pending,
        error=result.error,
        sources=result.sources,
    )


async def _finish(
    text: str,
    answer: str,
    track: str,
    started: float,
    speak: bool,
    source: str,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    pending: Optional[Dict[str, Any]] = None,
    error: str = "",
    sources: Optional[List[str]] = None,
) -> Dict[str, Any]:
    latency = int((time.perf_counter() - started) * 1000)
    _STATE["last_latency_ms"] = latency
    answer = (answer or "").strip() or "Done."
    TERMINAL.push("out", f"{answer}  ({track}, {latency} ms)")
    if pending:
        push_card(pending)

    payload: Dict[str, Any] = {
        "ok": not bool(error),
        "track": track,
        "input": text,
        "answer": answer,
        "latency_ms": latency,
        "tool_calls": tool_calls or [],
        "sources": list(sources or [])[:6],
        "card": pending,
        "speak": bool(speak),
        "source": source,
        "at": datetime.now().strftime("%H:%M:%S"),
    }
    if error:
        payload["error"] = error
        _STATE["last_error"] = error

    if speak and not _STATE["muted"] and answer:
        set_mode("speaking", answer[:70])
        try:
            out = TTS_DIR / f"say-{int(time.time() * 1000)}.wav"
            result = await asyncio.to_thread(voice.tts.synth, answer, out)
            payload["tts"] = {"ok": bool(result.get("ok")), "engine": result.get("engine", ""), "url": None}
            if result.get("ok"):
                rel = f"/audio/{out.name}"
                payload["tts"]["url"] = rel
                payload["tts"]["seconds"] = round(result.get("seconds") or 0, 2)
                HUB.emit({"type": "audio", "url": rel, "engine": result.get("engine", "")})
                _prune_tts_files()
            elif result.get("note"):
                payload["tts"]["note"] = result["note"]
        except Exception as exc:  # noqa: BLE001
            payload["tts"] = {"ok": False, "error": str(exc)}
            log.warning("tts failed: %s", exc)
    set_mode("idle")
    HUB.emit({"type": "reply", **payload})
    if not str(source).startswith("discord"):
        payload["mirrored"] = mirror_to_discord(answer)
    return payload


def _prune_tts_files(keep: int = 12) -> None:
    try:
        files = sorted(TTS_DIR.glob("say-*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in files[keep:]:
            stale.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Microphone (binary frames -> VAD segmentation -> STT)
# ---------------------------------------------------------------------------

class StreamBuffer:
    """Energy-gated accumulator for continuous WebSocket mic streaming.

    Frames are raw s16le mono @16 kHz. A segment is cut when the voice stops,
    or when ``max_seconds`` elapses, then handed to Whisper.
    """

    def __init__(self, max_seconds: float = 12.0, min_seconds: float = 0.5, silence_hang: float = 0.55) -> None:
        self.buf = bytearray()
        self.max_bytes = int(max_seconds * TARGET_SR * 2)
        self.min_bytes = int(min_seconds * TARGET_SR * 2)
        self.hang_bytes = int(silence_hang * TARGET_SR * 2)
        self.silent = 0
        self.active = False

    def feed(self, chunk: bytes) -> Optional[bytes]:
        """Return a complete segment (raw pcm) when one is ready, else ``None``."""
        self.buf.extend(chunk)
        peak = _peak_rms(chunk)
        if peak > 0.012:
            self.active = True
            self.silent = 0
        elif self.active:
            self.silent += len(chunk)
        complete = (self.active and self.silent >= self.hang_bytes and len(self.buf) >= self.min_bytes) or len(self.buf) >= self.max_bytes
        if not complete:
            return None
        segment = bytes(self.buf)
        self.buf.clear()
        self.active = False
        self.silent = 0
        return segment


def _peak_rms(pcm: bytes) -> float:
    n = len(pcm) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", pcm[: n * 2])
    step = max(1, n // 512)
    total, count = 0.0, 0
    for i in range(0, n, step):
        val = samples[i] / 32768.0
        total += val * val
        count += 1
    return math.sqrt(total / count) if count else 0.0


# ---------------------------------------------------------------------------
# FastAPI wiring
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        HUB.attach_loop(asyncio.get_running_loop())
        tasks = [asyncio.create_task(_telemetry_loop(), name="jarvis-telemetry")]
        bridge = None
        if SETTINGS.discord_enabled and SETTINGS.discord_token:
            try:
                import discord_bridge

                bridge = discord_bridge.start_bridge(HUB)
                global BRIDGE
                BRIDGE = bridge
                _STATE["discord"] = {"enabled": True, "connected": False, "detail": "dialling"}
                app.state.discord = bridge
            except Exception as exc:  # noqa: BLE001
                _STATE["discord"] = {"enabled": False, "connected": False, "detail": f"start failed: {exc}"}
                log.warning("discord bridge not started: %s", exc)
        else:
            _STATE["discord"] = {"enabled": bool(SETTINGS.discord_enabled), "connected": False,
                                 "detail": "no DISCORD_TOKEN in .env" if SETTINGS.discord_enabled else "disabled in .env"}
        if os.environ.get("JARVIS_WARM", "1") != "0" and SETTINGS.whisper_model:
            voice.warmup()
        TERMINAL.push("sys", f"JARVIS core online · v{VERSION} · Track 1 rules: {len(INSTANT_RULES)} · tools: {len(router.TOOL_NAMES)}")
        log.info(
            "JARVIS %s core ready · HUD at / · WebSocket at /ws · REST /api/* "
            "(the launcher prints the bound port)", VERSION,
        )
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            if bridge is not None:
                try:
                    bridge.stop()
                except Exception:  # noqa: BLE001
                    pass
            await asyncio.gather(*tasks, return_exceptions=True)

    app = FastAPI(title="J.A.R.V.I.S. core", version=VERSION, lifespan=lifespan)

    if SETTINGS.cors_origins.strip() in {"*", ""}:
        app.add_middleware(
            CORSMiddleware,
            allow_origin_regex=".*",
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[o.strip() for o in SETTINGS.cors_origins.split(",") if o.strip()],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(str(config.INDEX_HTML), media_type="text/html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        svg = (
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
            "<circle cx='16' cy='16' r='13' fill='none' stroke='#22d3ee' stroke-width='2'/>"
            "<circle cx='16' cy='16' r='5' fill='#22d3ee' opacity='0.85'/></svg>"
        )
        return Response(content=svg, media_type="image/svg+xml")

    app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR), html=False), name="static")
    app.mount("/audio", StaticFiles(directory=str(TTS_DIR), html=False), name="audio")

    # ---------------- system / status ---------------------------------------
    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        return {"ok": True, "version": VERSION, "uptime_s": int(time.time() - BOOT_TS), "clients": HUB.count}

    @app.get("/api/config")
    async def get_config() -> Dict[str, Any]:
        return {
            "settings": SETTINGS.redacted(),
            "voice": voice.status(),
            "router": router.ROUTER.status(),
            "instant_rules": [name for name, _, _ in INSTANT_RULES],
            "tools": sorted(router.TOOL_NAMES),
            "apps": sorted(tools.available_apps()),
            "version": VERSION,
        }

    @app.get("/api/status")
    async def get_status() -> Dict[str, Any]:
        return {
            "mode": _STATE["mode"],
            "uptime_s": int(time.time() - BOOT_TS),
            "clients": HUB.count,
            "stats": {k: v for k, v in _STATE.items() if k in {"commands", "instant_hits", "agent_hits", "last_latency_ms"}},
            "discord": {**_STATE["discord"], **(BRIDGE.status() if BRIDGE else {})},
            "mirror": _STATE["mirror"],
            "voice": voice.status(),
            "brain": router.ROUTER.brain.status(),
            "pending_confirm": bool(_STATE["pending_confirm"]),
            "last_error": _STATE["last_error"],
        }

    @app.get("/api/telemetry")
    async def get_telemetry() -> Dict[str, Any]:
        return await asyncio.to_thread(_telemetry_snapshot)

    @app.get("/api/terminal")
    async def get_terminal() -> Dict[str, Any]:
        return {"lines": TERMINAL.lines[-200:]}

    @app.get("/api/history")
    async def get_history() -> Dict[str, Any]:
        return {"turns": router.MEMORY.snapshot(), "cards": _STATE["history"][-10:]}

    @app.post("/api/history/clear")
    async def clear_history() -> Dict[str, Any]:
        router.MEMORY.clear()
        TERMINAL.lines.clear()
        HUB.emit({"type": "cleared"})
        return {"ok": True}

    # ---------------- commands ---------------------------------------------
    @app.post("/api/command")
    async def post_command(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:  # noqa: B008
        text = str(payload.get("text") or payload.get("command") or "")
        speak = bool(payload.get("speak", SETTINGS.speak_replies))
        force = bool(payload.get("agent", payload.get("force_agent", False)))
        return await handle_command(text, source=str(payload.get("source", "rest")), speak=speak, force_agent=force)

    @app.get("/api/command")
    async def get_command(q: str = "", speak: bool = True) -> Dict[str, Any]:
        return await handle_command(q, source="rest-get", speak=speak)

    @app.post("/api/speak")
    async def post_speak(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:  # noqa: B008
        text = str(payload.get("text", ""))
        result = await asyncio.to_thread(voice.tts.speak, text, bool(payload.get("play", SETTINGS.tts_playback)))
        TERMINAL.push("tts", _trim(text, 120))
        return {"ok": bool(result.get("ok")), "engine": result.get("engine", ""), "error": result.get("error", "")}

    @app.post("/api/stop")
    async def post_stop() -> Dict[str, Any]:
        voice.tts._queue.clear()
        _STATE["muted"] = True
        set_mode("idle", "stopped")
        return {"ok": True}

    # ---------------- audio in ---------------------------------------------
    @app.post("/api/listen")
    async def post_listen(request: Request) -> Dict[str, Any]:
        """Raw mic bytes (content-type carries the codec) → STT → command pipeline."""
        body = await request.body()
        ctype = (request.headers.get("content-type") or "").lower()
        ext = "wav" if "wav" in ctype else ("pcm" if "pcm" in ctype else "webm")
        return await _process_audio(body, ext, source="rest-mic")

    @app.post("/api/transcribe")
    async def post_transcribe(file: UploadFile = File(...), speak: bool = Form(False), command: bool = Form(True)) -> Dict[str, Any]:  # noqa: B008
        payload = await file.read()
        ext = (Path(file.filename or "audio.webm").suffix or ".webm").lstrip(".") or "webm"
        if ext not in {"wav", "webm", "ogg", "opus", "m4a", "mp3", "aac", "pcm"}:
            ext = "webm"
        transcript = await asyncio.to_thread(voice.listen, payload, ext)
        result: Dict[str, Any] = {"ok": bool(transcript.text), "transcript": transcript.as_dict()}
        if not transcript.text:
            result["error"] = transcript.engine if transcript.engine.startswith(("decode-error", "error")) else "no speech detected"
            return result
        if command:
            result["reply"] = await handle_command(transcript.text, source="mic", speak=bool(speak))
        return result

    async def _process_audio(body: bytes, ext: str, source: str) -> Dict[str, Any]:
        if ext == "pcm":
            transcript = await asyncio.to_thread(voice.stt.transcribe_pcm, body, TARGET_SR)
        else:
            transcript = await asyncio.to_thread(voice.listen, body, ext)
        if not transcript.text:
            set_mode("idle")
            return {"ok": False, "error": "no speech detected", "transcript": transcript.as_dict()}
        HUB.emit({"type": "transcript", **transcript.as_dict()})
        reply = await handle_command(transcript.text, source=source, speak=SETTINGS.speak_replies)
        return {"ok": True, "transcript": transcript.as_dict(), "reply": reply}

    @app.get("/api/discord/status")
    async def discord_status() -> Dict[str, Any]:
        live = BRIDGE.status() if BRIDGE else {}
        return {**_STATE["discord"], **live, "mirror": _STATE["mirror"],
                "recent": BRIDGE.recent(20) if BRIDGE else []}

    @app.post("/api/discord/mirror")
    async def discord_mirror(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:  # noqa: B008
        _STATE["mirror"] = bool(payload.get("on", True))
        state = "mirroring my replies to" if _STATE["mirror"] else "no longer mirroring replies to"
        TERMINAL.push("sys", f"Discord {state} channel {_STATE['discord'].get('channel_id') or 'any'}.")
        return {"ok": True, "mirror": _STATE["mirror"]}

    @app.post("/api/discord/announce")
    async def discord_announce(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:  # noqa: B008
        if BRIDGE is None:
            return {"ok": False, "error": "bridge not running (set DISCORD_TOKEN in .env)"}
        return {"ok": BRIDGE.send(str(payload.get("text", "")), bool(payload.get("everyone", False)))}

    # ---------------- tools (used by the agent & curl) ----------------------
    @app.get("/api/notes")
    async def api_notes(topic: str = "", limit: int = 2) -> Dict[str, Any]:
        return await asyncio.to_thread(tools.read_notes, topic, limit)

    @app.get("/api/search")
    async def api_search(q: str, max_results: int = 6) -> Dict[str, Any]:
        if not q:
            return {"ok": False, "message": "Missing query parameter q"}
        return await asyncio.to_thread(tools.web_search, q, max_results)

    @app.get("/api/sports")
    async def api_sports(team: str = "", kind: str = "all") -> Dict[str, Any]:
        return await asyncio.to_thread(tools.fetch_sports_stats, team, kind)

    @app.get("/api/apps")
    async def api_apps() -> Dict[str, Any]:
        return await asyncio.to_thread(tools.list_launchable_apps)

    # ---------------- WebSocket --------------------------------------------
    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        await HUB.add(ws)
        buffers: Dict[str, StreamBuffer] = {}
        try:
            await ws.send_json(
                {
                    "type": "hello",
                    "version": VERSION,
                    "server_time": datetime.now().isoformat(timespec="seconds"),
                    "config": {
                        "mode": _STATE["mode"],
                        "speak_replies": SETTINGS.speak_replies,
                        "instant_rules": [name for name, _, _ in INSTANT_RULES],
                        "voice": voice.status(),
                        "brain": router.ROUTER.brain.status(),
                        "apps": sorted(tools.available_apps()),
                        "log": TERMINAL.lines[-40:],
                    },
                }
            )
            await HUB.send({"type": "telemetry", "data": await asyncio.to_thread(_telemetry_snapshot)})
            while True:
                message = await ws.receive()
                if "text" in message and message["text"] is not None:
                    await _handle_ws_json(ws, message["text"], buffers)
                elif "bytes" in message and message["bytes"] is not None:
                    await _handle_ws_bytes(ws, message["bytes"], buffers)
        except WebSocketDisconnect:
            pass
        except RuntimeError as exc:  # send-after-close, or client vanished mid-write
            log.debug("websocket write failed: %s", exc)
        finally:
            await HUB.remove(ws)

    async def _handle_ws_json(ws: WebSocket, raw: str, buffers: Dict[str, StreamBuffer]) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            await ws.send_json({"type": "error", "message": "expected a JSON object"})
            return
        kind = str(payload.get("type", "")).lower()

        if kind in {"command", "chat", "text"}:
            asyncio.create_task(
                handle_command(
                    str(payload.get("text", "")),
                    source=str(payload.get("source", "hud")),
                    speak=bool(payload.get("speak", SETTINGS.speak_replies)),
                    force_agent=bool(payload.get("agent", False)),
                )
            )
            await ws.send_json({"type": "ack", "state": "thinking", "echo": str(payload.get("text", ""))[:200]})
        elif kind == "speak":
            asyncio.create_task(
                asyncio.to_thread(voice.tts.speak, str(payload.get("text", "")), bool(payload.get("play", True)))
            )
            TERMINAL.push("tts", _trim(str(payload.get("text", "")), 120))
        elif kind in {"mic", "audio"}:
            data = str(payload.get("data", ""))
            if "," in data[:64]:
                data = data.split(",", 1)[1]
            try:
                blob = base64.b64decode(data)
            except (binascii.Error, ValueError) as exc:
                await ws.send_json({"type": "error", "message": f"bad base64 audio: {exc}"})
                return
            set_mode("listening", "buffered clip")
            reply = await _process_audio(blob, str(payload.get("format", "webm")), "hud-mic")
            await ws.send_json({"type": "transcript-complete", **reply})
        elif kind == "stream-start":
            buffers["mic"] = StreamBuffer()
            set_mode("listening", "stream open")
            await ws.send_json({"type": "stream", "state": "open", "sample_rate": TARGET_SR})
        elif kind == "stream-stop":
            buffer = buffers.pop("mic", None)
            if buffer and len(buffer.buf) >= buffer.min_bytes:
                await _transcribe_buffer(ws, bytes(buffer.buf))
            else:
                set_mode("idle")
                await ws.send_json({"type": "stream", "state": "closed", "segments": 0})
        elif kind == "stop":
            voice.tts._queue.clear()
            set_mode("idle", "stopped by HUD")
            await ws.send_json({"type": "state", "mode": "idle"})
        elif kind in {"hello", "mic-state", "mode", "ready"}:
            # Client-side announcements (HUD boot, mic button presses). Nothing
            # to answer; they exist so the terminal can show intent ordering.
            TERMINAL.push("sys", f"hud: {kind}" + (f" · {payload.get('state')}" if payload.get("state") else ""))
        elif kind == "ping":
            await ws.send_json({"type": "pong", "t": time.time(), "server_uptime_s": int(time.time() - BOOT_TS)})
        elif kind == "config":
            await ws.send_json({"type": "config", "settings": SETTINGS.redacted(), "state": _STATE["mode"]})
        else:
            await ws.send_json({"type": "error", "message": f"unknown ws type '{kind}'"})

    async def _handle_ws_bytes(ws: WebSocket, chunk: bytes, buffers: Dict[str, StreamBuffer]) -> None:
        buffer = buffers.get("mic")
        if buffer is None:
            buffer = buffers["mic"] = StreamBuffer()
            set_mode("listening", "auto-opened stream")
        segment = buffer.feed(chunk)
        level = _peak_rms(chunk)
        await ws.send_json({"type": "level", "rms": round(level, 4), "buffered": len(buffer.buf)})
        if segment:
            await _transcribe_buffer(ws, segment)

    async def _transcribe_buffer(ws: WebSocket, pcm: bytes) -> None:
        set_mode("thinking", f"{len(pcm) / (TARGET_SR * 2):.1f}s of audio")
        transcript = await asyncio.to_thread(voice.stt.transcribe_pcm, pcm, TARGET_SR)
        if not transcript.text.strip():
            set_mode("idle")
            await ws.send_json({"type": "stream", "state": "silence-dropped", "duration_s": round(len(pcm) / (TARGET_SR * 2), 2)})
            return
        HUB.emit({"type": "transcript", **transcript.as_dict()})
        await ws.send_json({"type": "segment", "text": transcript.text, "confidence": transcript.confidence})
        await handle_command(transcript.text, source="mic-stream", speak=SETTINGS.speak_replies)

    return app


def _trim(text: str, limit: int = 160) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def serve(host: Optional[str] = None, port: Optional[int] = None, log_level: str = "info") -> None:
    """Standalone dev server: ``python server.py --port 8760``."""
    import uvicorn

    config.ensure_importable()
    uvicorn.run(
        "server:app",
        host=host or SETTINGS.host,
        port=int(port or SETTINGS.port),
        log_level=log_level,
        ws_ping_interval=20,
        ws_ping_timeout=30,
        access_log=SETTINGS.debug,
    )


app = create_app()


__all__ = ["app", "create_app", "handle_command", "match_instant", "HUB", "TERMINAL", "INSTANT_RULES", "serve"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the JARVIS core (HUD + WebSocket API).")
    parser.add_argument("--host", default=SETTINGS.host)
    parser.add_argument("--port", type=int, default=SETTINGS.port)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    serve(args.host, args.port, args.log_level)
