"""Timers, reminders and scheduled actions - the part of JARVIS that keeps working after you
stop talking.

``remind me in ten minutes to stretch``, ``set a timer for 25 minutes``, ``every morning at 7
tell me my battery`` all land here.  A schedule is plain JSON at ``data/reminders.json`` (so it
survives a restart, which is the whole point of a reminder) and one asyncio task sleeps to the
next due item.  At the due time JARVIS speaks it, raises a real Windows toast through
:func:`winops.notify`, and writes a line to the HUD terminal; if the reminder text is an
instruction rather than a note ("check the download"), it runs it through the router first so
the answer is spoken too.

Interval parsing is deliberately forgiving - spoken numbers arrive as words, and Whisper will
give you "1/2 hour" as easily as "thirty minutes".
"""

from __future__ import annotations

import json
import re
import threading
import uuid
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import config
from config import SETTINGS, get_logger

log = get_logger("reminders")

#: "10 min", "in an hour and a half", "in 45 secs", "tomorrow at 7", "at 18:30"
_UNITS: Dict[str, int] = {
    "sec": 1, "secs": 1, "second": 1, "seconds": 1, "s": 1,
    "min": 60, "mins": 60, "minute": 60, "minutes": 60, "m": 60,
    "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600, "h": 3600,
    "day": 86400, "days": 86400, "week": 604800, "weeks": 604800,
}
_WORD_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                 "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                 "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
                 "half": 0.5, "quarter": 0.25, "couple": 2, "few": 3}


def _result(ok: bool, message: str, **extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": bool(ok), "message": message}
    out.update(extra)
    return out


def parse_when(text: str, now: Optional[datetime] = None) -> Tuple[Optional[float], str, str]:
    """Return ``(unix_time_or_None, human_label, error)`` for a spoken when-clause.

    Handles relative spans ("in ten minutes", "in an hour and a half") and clock times
    ("at 7", "at 18:30", "tomorrow at 7:15", "tonight at 9"); a clock time that has already
    passed means tomorrow, because that is what a person means.
    """
    raw = (text or "").strip().lower().rstrip(".!")
    if not raw:
        return None, "", "No time was given - say something like “in ten minutes” or “at 7”."
    base = now or datetime.now()
    #: Spans are measured from the reference moment, and the reference moment is "now" unless the
    #: caller supplied one (a replayed transcript, a test).  Reading a second clock here made every
    #: scheduled span drift by the gap between the two.
    now_ts = base.timestamp()
    # Spoken English writes fractions as phrases; normalise them before the number scan.
    raw = re.sub(r"\bhalf an?\s+(hour|hr)\b", "0.5 hour", raw)
    raw = re.sub(r"\bquarter an?\s+(hour|hr)\b", "0.25 hour", raw)
    raw = re.sub(r"\b(?:an|one)?\s+hour and a half\b", "1.5 hour", raw)
    raw = re.sub(r"\ba (?:couple|few) (minutes?|hours?|seconds?)\b", r"2 \1", raw)

    # relative:  [in] <number|word> [unit] (and|plus <number|word> [unit])*
    span = 0.0
    consumed = False
    words = "|".join(sorted(_WORD_NUMBERS, key=len, reverse=True))
    #: ``\b`` at both ends and a *mandatory* unit.  Without them the lone letters "a" and "an"
    #: inside "every d-a-y at 8-a-m" counted as quantities, and a daily reminder fired one minute
    #: later, then every minute, forever.
    pattern = re.compile(r"\b(?P<qty>\d+(?:\.\d+)?(?:/\d+)?|" + words + r")"
                         r"\s*(?P<unit>seconds?|secs?|mins?|minutes?|hrs?|hours?|days?|weeks?|h|m|s)\b")
    for match in pattern.finditer(raw):
        unit = (match.group("unit") or "").rstrip("s")
        key = {"min": 60, "minute": 60, "m": 60, "sec": 1, "second": 1, "s": 1, "hr": 3600,
               "hour": 3600, "h": 3600, "day": 86400, "week": 604800}.get(unit, 0)
        if not key:
            continue
        span += _quantity(match.group("qty")) * key
        consumed = True
    if not consumed:
        #: "in 20" means twenty minutes - a bare number after "in" is always minutes in speech.
        bare = re.search(r"\bin\s+(?P<qty>\d{1,4}|" + words + r")\s*$", raw)
        if bare:
            span = _quantity(bare.group("qty")) * 60.0
            consumed = True
    if consumed and span > 0:
        every_label = re.sub(r"^in ", "", _human(span))
        if re.search(r"\bevery\b|\bdaily\b|\b.each\b", raw):
            return now_ts + span, f"{every_label} from now, then every {every_label}", ""
        return now_ts + span, (f"in {every_label}" if span < 86400 else every_label), ""

    # clock:  [tomorrow|tonight|today] at [7|7:15|18:30] [am|pm]
    clock = re.search(r"(?:at\s+|by\s+|@\s*)(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>am|pm)?",
                      raw)
    if clock:
        hour = int(clock.group("hour"))
        minute = int(clock.group("minute") or 0)
        ampm = clock.group("ampm") or ""
        if hour > 24 or minute > 59:
            return None, "", f"“{clock.group(0)}” is not a time I can schedule."
        if ampm == "pm" and hour < 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
        target = base.replace(hour=hour % 24, minute=minute, second=0, microsecond=0)
        if "tomorrow" in raw:
            target += timedelta(days=1)
        elif "tonight" in raw and hour < 12:
            target += timedelta(hours=12)
        elif target.timestamp() <= now_ts:
            target += timedelta(days=1)
        repeat = 86400.0 if re.search(r"\bevery\b|\bdaily\b|\beach day\b", raw) else 0.0
        label = target.strftime("%H:%M on %d %b") + (" daily" if repeat else "")
        return target.timestamp(), label, ""

    if re.search(r"\btonight\b", raw):
        target = base.replace(hour=21, minute=0, second=0, microsecond=0)
        if target.timestamp() <= now_ts:
            target += timedelta(days=1)
        return target.timestamp(), target.strftime("at %H:%M tonight"), ""
    if re.search(r"\btomorrow morning\b", raw):
        target = (base + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)
        return target.timestamp(), "at 08:00 tomorrow", ""
    if re.search(r"\btomorrow\b", raw):
        target = (base + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        return target.timestamp(), "at 09:00 tomorrow", ""
    if re.search(r"\b(?:in a (?:bit|while|sec)|shortly|soon)\b", raw):
        return now_ts + 300, "in 5 min", ""
    return None, "", (f"I couldn't read a time in “{raw[:60]}”. Try “in 10 minutes”, "
                      "“at 7:30 pm” or “tomorrow at 9”.")


def _quantity(text: str) -> float:
    """``"7"``, ``"three"``, ``"1.5"``, ``"1/2"`` -> a float.  Unknown words are 0, never a guess."""
    raw = (text or "").strip()
    if "/" in raw:
        head, _, tail = raw.partition("/")
        try:
            return (float(head or 0) or 0.5) / float(tail or 2)
        except (TypeError, ValueError):
            return 0.5
    try:
        return float(raw)
    except ValueError:
        return float(_WORD_NUMBERS.get(raw, 0) or 0)


def _human(seconds: float) -> str:
    seconds = int(max(1, seconds))
    if seconds < 60:
        return f"in {seconds} sec"
    if seconds < 3600:
        minutes, secs = divmod(seconds, 60)
        return f"in {minutes} min" + (f" {secs} s" if secs else "")
    hours, rest = divmod(seconds, 3600)
    if hours < 24:
        return f"in {hours} h" + (f" {rest // 60} min" if rest >= 60 else "")
    days, rem = divmod(hours, 24)
    return f"in {days} day" + ("s" if days != 1 else "") + (f" {rem} h" if rem else "")


@dataclass
class Reminder:
    rid: str
    text: str
    due: float
    every: float = 0.0
    created: float = field(default_factory=time.time)
    fired: int = 0
    run: bool = False          # treat the text as a command, not just a note
    next_label: str = ""

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["due_iso"] = datetime.fromtimestamp(self.due).strftime("%Y-%m-%d %H:%M:%S")
        data["in_seconds"] = max(0, int(self.due - time.time()))
        return data


class ReminderBoard:
    """JSON-backed schedule with one sleeping worker; thread-safe because main + server touch it."""

    def __init__(self, path: Optional[Path] = None,
                 fire: Optional[Callable[[Reminder], None]] = None) -> None:
        self.path = config.ROOT / (path or SETTINGS.reminders_file)
        self.fire = fire
        self._items: Dict[str, Reminder] = {}
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.load()

    # -- persistence -------------------------------------------------------
    def load(self) -> int:
        try:
            rows = json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else []
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("reminders file unreadable (%s), starting clean", exc)
            rows = []
        with self._lock:
            self._items = {}
            for row in rows if isinstance(rows, list) else []:
                try:
                    item = Reminder(**{k: v for k, v in row.items()
                                       if k in Reminder.__dataclass_fields__})
                except TypeError:
                    continue
                self._items[item.rid] = item
        return len(self._items)

    def save(self) -> None:
        with self._lock:
            rows = [r.as_dict() for r in sorted(self._items.values(), key=lambda r: r.due)]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([{k: v for k, v in row.items()
                                               if k not in {"due_iso", "in_seconds"}} for row in rows],
                                             ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError as exc:  # pragma: no cover
            log.debug("reminders not saved: %s", exc)

    # -- api ---------------------------------------------------------------
    def add(self, text: str, when: str, run: bool = False) -> Dict[str, Any]:
        due, label, error = parse_when(when)
        if due is None:
            return _result(False, error)
        if due - time.time() > 60 * 60 * 24 * 400:
            return _result(False, "That is more than a year away - I keep reminders for 400 days.")
        #: A millisecond timestamp used to be the id, so "remind me to stretch and stand up"
     #: added both items in the same breath and the second silently replaced the first.
        item = Reminder(rid="r" + uuid.uuid4().hex[:10], text=(text or "").strip(),
                        due=due, run=bool(run), next_label=label)
        every_match = re.search(r"every\s+(\d+(?:\.\d+)?|a|an)\s*(minutes?|hours?|days?|weeks?)",
                                (when or "").lower())
        if every_match:
            unit = _UNITS.get(every_match.group(2).rstrip("s"), 60)
            try:
                count = float(every_match.group(1)) if every_match.group(1)[0].isdigit() else 1.0
            except ValueError:
                count = 1.0
            item.every = max(60.0, count * unit)
        with self._lock:
            self._items[item.rid] = item
        self.save()
        self._wake.set()
        verb = "I'll run" if item.run else "I'll remind you about"
        object_ = f" “{(item.text or 'this').strip()}”"
        return _result(True, f"{verb}{object_} {label}"
                            + (f", then every {int(item.every // 60)} min." if item.every else "."),
                       id=item.rid, due=item.due, label=label, every=item.every)

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = sorted(self._items.values(), key=lambda r: r.due)
        return [r.as_dict() for r in rows]

    def cancel(self, which: str = "") -> Dict[str, Any]:
        token = (which or "").strip().lower()
        with self._lock:
            targets = [r for r in self._items.values()
                       if not token or r.rid == token or token in r.text.lower()
                       or (token in {"last", "the last one"} and r is max(self._items.values(), key=lambda x: x.created))]
            if not targets:
                return _result(False, "You have no reminders " + (f"matching “{token}”." if token else "set."))
            removed = [t.text or t.rid for t in targets]
            for item in targets:
                self._items.pop(item.rid, None)
        self.save()
        return _result(True, f"Cancelled {len(removed)} reminder(s): {', '.join(removed[:4])}.",
                       cancelled=removed)

    def snooze(self, which: str = "", minutes: int = 10) -> Dict[str, Any]:
        token = (which or "").strip().lower()
        with self._lock:
            for item in sorted(self._items.values(), key=lambda r: r.due):
                if not token or item.rid == token or token in item.text.lower():
                    item.due = max(time.time() + 5, item.due + max(1, int(minutes)) * 60)
                    item.next_label = _human(item.due - time.time())
                    self.save()
                    return _result(True, f"“{item.text}” now {item.next_label}.", id=item.rid)
        return _result(False, "Nothing to snooze.")

    def due_now(self, limit: int = 4) -> List[Reminder]:
        now = time.time()
        with self._lock:
            ready = [r for r in self._items.values() if r.due <= now]
        ready.sort(key=lambda r: r.due)
        return ready[:limit]

    def _fire(self, item: Reminder) -> None:
        if item.every > 0:
            with self._lock:
                item.due = time.time() + item.every
                item.fired += 1
                item.next_label = _human(item.due - time.time())
            self.save()
        else:
            with self._lock:
                self._items.pop(item.rid, None)
            self.save()
        if self.fire:
            try:
                self.fire(item)
            except Exception as exc:  # noqa: BLE001 - a broken handler must not kill the timer
                log.warning("reminder handler failed: %s", exc)

    # -- worker ------------------------------------------------------------
    def start(self) -> Dict[str, Any]:
        if not SETTINGS.reminders_enabled:
            return _result(False, "REMINDERS_ENABLED is false in .env.")
        with self._lock:
            if self._thread and self._thread.is_alive():
                return _result(True, f"{len(self._items)} reminder(s) already scheduled.", running=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="jarvis-reminders", daemon=True)
        self._thread.start()
        pending = self.list()
        detail = (f"{len(pending)} scheduled, next {re.sub(r'^in ', '', _human(pending[0]['in_seconds']))}."
                  if pending else "nothing scheduled yet.")
        return _result(True, f"Reminders armed: {detail}", running=True, count=len(pending))

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            ready = self.due_now()
            for item in ready:
                self._fire(item)
            with self._lock:
                upcoming = min((r.due for r in self._items.values()), default=None)
            if upcoming is None:
                self._wake.wait(5.0)
                self._wake.clear()
                continue
            nap = max(0.2, min(upcoming - time.time(), 5.0))
            if self._wake.wait(nap):
                self._wake.clear()


BOARD = ReminderBoard()


def fire_and_speak(item: Reminder) -> None:
    """Default handler: toast + spoken line (+ run the text if it is a command)."""
    label = "Reminder" if not item.run else "Scheduled task"
    try:
        import winops

        winops.notify(label, item.text or "Time's up.")
    except Exception:  # noqa: BLE001
        pass
    spoken = f"{label}: {item.text or 'time is up.'}"
    try:
        import wake

        # Never answer yourself: the ear is closed for roughly the length of this sentence.
        wake.LISTENER.duck(max(2.0, len(spoken) / 3.6 + 3.0))
    except Exception as exc:  # noqa: BLE001 - the ear is optional, the toast is not
        log.debug("reminder could not duck the ear: %s", exc)
    try:
        import audio_engine

        audio_engine.get_engine().speak(spoken)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not speak reminder: %s", exc)
    try:
        import server

        if item.run and item.text:
            server.queue_spoken_command(item.text, source="reminder")
    except Exception as exc:  # noqa: BLE001
        log.debug("reminder could not run its command: %s", exc)


__all__ = ["Reminder", "ReminderBoard", "BOARD", "parse_when", "fire_and_speak"]
