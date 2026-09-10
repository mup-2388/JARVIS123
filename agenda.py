"""Calendar: JARVIS's day-planner, fed by the calendars the user already uses.

No OAuth and no vendor SDK: Google Calendar and Outlook both let you export a
read-only ``.ics`` feed (Google: "Settings → … → Secret address in iCal format";
Outlook: "Share → Publish this calendar").  Paste those URLs into
``CALENDAR_ICS_URLS`` in ``.env`` (colon/semicolon/comma separated) and JARVIS can
answer "what's on my calendar", "what's my next class", "do I have anything
tomorrow", and fold it into the morning brief.

The same parser also reads local ``.ics`` files (``CALENDAR_ICS_FILES``), which is
how an exported Outlook/CBS schedule lands without a live feed.

Everything is stdlib ``urllib`` + a small RFC 5545 ``VEVENT`` parser, so it works
offline on the cached copy when the network (or the calendar server) is down.
"""

from __future__ import annotations

import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import SETTINGS, get_logger

log = get_logger("calendar")

_UA = {"user-agent": "JARVIS-assistant/1.0 calendar"}

#: Line-folding: an RFC 5545 line that starts with space/tab is a continuation.
_FOLD = re.compile(r"\r?\n[ \t]")
#: ``DTSTART;TZID=Europe/Berlin:20240101T090000`` -> capture the raw value.
_PROP = re.compile(r"^(SUMMARY|DTSTART|DTEND|DTSTAMP|LOCATION|DESCRIPTION|UID)"
                   r"(?:;[^:]*)?:(.*)$", re.I | re.M)
_UID = re.compile(r"^UID(?:;[^:]*)?:(.*)$", re.I | re.M)


@dataclass
class Event:
    """One calendar entry, already normalised to the local timezone."""

    summary: str
    start: datetime
    end: datetime
    location: str = ""
    description: str = ""
    all_day: bool = False
    source: str = ""

    @property
    def when(self) -> str:
        if self.all_day:
            return self.start.strftime("%A %d %B") + " (all day)"
        return (f"{self.start.strftime('%A %d %B, %H:%M')}"
                f" – {self.end.strftime('%H:%M')}")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "start": self.start.isoformat(timespec="minutes"),
            "end": self.end.isoformat(timespec="minutes"),
            "location": self.location,
            "description": self.description,
            "all_day": self.all_day,
            "when": self.when,
            "source": self.source,
        }


def _unfold(text: str) -> str:
    return _FOLD.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))


def _parse_dt(value: str, tzid: str = "") -> Optional[datetime]:
    """Parse an iCal DATE / DATE-TIME, treating TZID/floating times as local wall time."""
    value = value.strip()
    if not value:
        return None
    if len(value) == 8 and value.isdigit():          # YYYYMMDD -> all-day
        try:
            return datetime.strptime(value, "%Y%m%d")
        except ValueError:
            return None
    # Strip a leading TZID=...: if it survived the property regex.
    if ":" in value and value.split(":", 1)[0].upper().startswith("TZID"):
        value = value.split(":", 1)[1]
    try:
        if value.endswith("Z"):                       # UTC -> local
            dt = datetime.strptime(value, "%Y%m%dT%H%M%SZ")
            return dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
        return datetime.strptime(value[:15], "%Y%m%dT%H%M%S")
    except ValueError:
        try:
            return datetime.strptime(value[:8], "%Y%m%d")
        except ValueError:
            return None


def _parse_vevent(block: str, source: str) -> Optional[Event]:
    if "DTSTART" not in block.upper():
        return None
    props: Dict[str, str] = {}
    for match in _PROP.finditer(block):
        props[match.group(1).upper()] = match.group(2).strip()

    start_raw = props.get("DTSTART", "")
    end_raw = props.get("DTEND", "")
    all_day = len(start_raw.strip()) == 8 and start_raw.strip().isdigit()
    start = _parse_dt(start_raw)
    end = _parse_dt(end_raw) if end_raw else None
    if start is None:
        return None
    if end is None:
        end = start + timedelta(days=1) if all_day else start + timedelta(hours=1)

    summary = props.get("SUMMARY", "").replace("\\,", ",").replace("\\n", "\n").strip() or "(no title)"
    location = props.get("LOCATION", "").replace("\\,", ",").strip()
    description = props.get("DESCRIPTION", "").replace("\\,", ",").replace("\\n", "\n").strip()
    return Event(summary=summary, start=start, end=end, location=location,
                 description=description, all_day=all_day, source=source)


def parse_ics(text: str, source: str = "") -> List[Event]:
    """Parse a whole ``.ics`` document into :class:`Event` objects."""
    blocks = re.split(r"BEGIN:VEVENT", _unfold(text), flags=re.I)
    events: List[Event] = []
    for block in blocks[1:]:                         # everything before the first VEVENT is headers
        end = re.split(r"END:VEVENT", block, flags=re.I, maxsplit=1)
        if len(end) != 2:
            continue
        event = _parse_vevent(end[0], source)
        if event is not None:
            events.append(event)
    return events


# --------------------------------------------------------------------------- sources
def _cache_path(key: str) -> Path:
    folder = Path(SETTINGS.calendar_cache_dir or "data/calendar")
    folder.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", key)[:60] or "feed"
    return folder / f"{safe}.ics"


def _sources() -> List[Tuple[str, str]]:
    """Configured feeds as ``(kind, value)`` where kind is ``url`` or ``file``."""
    sources: List[Tuple[str, str]] = []
    for raw in re.split(r"[:;,]", SETTINGS.calendar_ics_urls or ""):
        url = raw.strip()
        if url:
            sources.append(("url", url))
    for raw in re.split(r"[:;,]", SETTINGS.calendar_ics_files or ""):
        path = raw.strip()
        if path:
            sources.append(("file", path))
    return sources


def _read_url(url: str, key: str) -> Optional[str]:
    cache = _cache_path(key)
    max_age = max(1, int(SETTINGS.calendar_cache_minutes)) * 60
    if cache.is_file() and time.time() - cache.stat().st_mtime < max_age:
        return cache.read_text(encoding="utf-8", errors="replace")
    try:
        request = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(request, timeout=max(6, int(SETTINGS.calendar_timeout))) as response:
            text = response.read().decode("utf-8", errors="replace")
        cache.write_text(text, encoding="utf-8")
        return text
    except (urllib.error.URLError, OSError, ValueError) as exc:
        log.warning("calendar feed %s unreachable: %s", url, exc)
        if cache.is_file():                          # serve the stale copy rather than nothing
            return cache.read_text(encoding="utf-8", errors="replace")
        return None


def load_events() -> Tuple[List[Event], List[str]]:
    """Every event from every configured source, plus a list of feed errors."""
    events: List[Event] = []
    errors: List[str] = []
    if not _sources():
        return events, ["no calendar feeds configured - set CALENDAR_ICS_URLS in .env"]
    for kind, value in _sources():
        source_name = Path(value).name if kind == "file" else value
        try:
            if kind == "url":
                text = _read_url(value, value)
                if text is None:
                    errors.append(f"{source_name}: unreachable")
                    continue
            else:
                path = Path(value).expanduser()
                if not path.is_file():
                    errors.append(f"{source_name}: file not found")
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            events.extend(parse_ics(text, source_name))
        except Exception as exc:  # noqa: BLE001 - one bad feed must not sink the day
            errors.append(f"{source_name}: {exc}")
            log.warning("calendar source failed: %s", exc)
    return events, errors


def _window(days: float) -> Tuple[datetime, datetime]:
    now = datetime.now()
    return now, now + timedelta(days=max(0.0, days))


def agenda(days: float = 1.0) -> List[Event]:
    """Events between now and ``days`` from now, soonest first."""
    start, end = _window(days)
    events, _ = load_events()
    hits = [e for e in events if start <= e.start < end]
    return sorted(hits, key=lambda e: e.start)


def next_event() -> Optional[Event]:
    """The very next upcoming event, regardless of how far away it is."""
    events, _ = load_events()
    upcoming = [e for e in events if e.end >= datetime.now()]
    return min(upcoming, key=lambda e: e.start) if upcoming else None


def today() -> List[Event]:
    return agenda(1.0)


__all__ = ["Event", "agenda", "next_event", "today", "load_events", "parse_ics"]
