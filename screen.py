"""Eyes: capture the screen, read the text on it, and describe what the user is looking at.

Three tiers, best-available first, so this works on a bare Windows box and gets better with
each optional extra:

1. **Windows OCR** (``winsdk``/``pywinrt`` - free, offline, on every Win10/11 install that has
   the OCR pack, which is nearly all of them) on a GDI grab made by :mod:`winops`.  No network,
   no GPU budget spent, ~200 ms.
2. **Tesseract** if ``tesseract.exe`` happens to be on PATH or ``TESSERACT_CMD`` points at it.
3. **A vision model** - the screenshot is base64'd into the OpenAI-style
   ``image_url`` content part and sent to whichever provider's key can see a model whose
   ``input_modalities`` include ``image`` (Groq's ``qwen3.6-27b``/llama-4-scout class, Gemini
   Flash, OpenRouter vision, GitHub ``gpt-4o``).  That's the "what am I looking at" answer.

"Read out the top three listings" is the same pipeline plus :func:`top_results`, which turns a
wall of recognised text into ranked items: it drops browser chrome, keeps lines that look like
titles (length + shape heuristics), attaches nearby price/link/number hints, and returns up to
N entries the model can then speak.  Deliberately dumb-but-robust, because OCR text is noisy and
a fragile parser is exactly what makes these features feel broken.
"""

from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import winops
from config import SETTINGS, get_logger

log = get_logger("screen")

_LINE_NOISE = re.compile(r"^(?:https?://|www\.)|\b(sign in|log ?in|menu|back|next|previous|ads by|"
                         r"ad \|sponsored|sponsor|share|reply|more options|new tab|ctrl|alt|f\d{1,2}|"
                         r"people also ask|related searches|results related to|show more|page \d)\b", re.I)
#: Browser/OS chrome that OCR always picks up and that is never a result.
_CHROME_LINE = re.compile(r"^(?:(?:google chrome|chrome|microsoft edge|edge|firefox|brave|opera|vivaldi)"
                          r"\b.{0,60}(?:\.com/|/search\?|\.google\.|new tab)|"
                          r"(?:settings|bookmarks|history|extensions|downloads panel|tabs|address bar)"
                          r"[\w \t]*$|.{0,30}\b(?:of\s+\d+|out of \d+)\b.{0,20}$)", re.I)
_TITLEISH = re.compile(r"^[\W]*([A-Z0-9\u00c0-\u024f][^\n]{6,140})$")


def _result(ok: bool, message: str, **extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": bool(ok), "message": message}
    out.update(extra)
    return out


# --------------------------------------------------------------------------- capture
def capture(target: str = "screen", path: str = "", delay_ms: int = 0) -> Dict[str, Any]:
    """Grab the whole screen, the focused window, or a region, and save a PNG."""
    region: Optional[List[int]] = None
    if (target or "screen").lower() in {"window", "foreground", "app", "active"}:
        front = winops.foreground_window()
        rect = (front.get("rect") or []) if front.get("ok") else []
        if len(rect) == 4 and rect[2] - rect[0] > 40 and rect[3] - rect[1] > 40:
            region = [max(0, int(v)) for v in rect]
        elif not front.get("ok"):
            return _result(False, f"I could not see the focused window: {front.get('message', '')}")
    out = winops.screenshot(path=path, region=region, delay_ms=delay_ms)
    if not out.get("ok"):
        return out
    out["target"] = "window" if region else "screen"
    out["title"] = str((winops.foreground_window().get("title") or ""))[:120]
    return out


def _to_data_url(path: Path) -> str:
    try:
        blob = path.read_bytes()
    except OSError:
        return ""
    return "data:image/png;base64," + base64.b64encode(blob).decode("ascii")


# --------------------------------------------------------------------------- OCR engines
def _ocr_windows(image: Path) -> Tuple[Optional[str], str]:
    """Windows.Media.Ocr through the WinRT Python projection: offline, free, fast.

    Two projections exist (Microsoft's retired ``winsdk`` and the current ``pywinrt``
    packages, whose modules are imported as ``winrt.*``); whichever is installed wins.  The
    OCR engine itself is part of Windows, so nothing is downloaded and no API key is used.
    """
    import asyncio
    import importlib

    prefix = ""
    for candidate in ("winsdk", "winrt", "pywinrt"):
        try:
            importlib.import_module(candidate + ".windows.media.ocr")
            prefix = candidate
            break
        except Exception:  # noqa: BLE001 - try the next projection
            continue
    if not prefix:
        raise ImportError("no WinRT projection - pip install winsdk or pywinrt[windows-media-ocr]")

    ocr_module = importlib.import_module(prefix + ".windows.media.ocr")
    imaging = importlib.import_module(prefix + ".windows.graphics.imaging")
    storage = importlib.import_module(prefix + ".windows.storage")

    async def run() -> str:
        file = await storage.StorageFile.get_file_from_path_async(str(image))
        stream = await file.open_async(storage.FileAccessMode.READ)
        decoder = await imaging.BitmapDecoder.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()
        engine = ocr_module.OcrEngine.try_create_from_user_profile_languages()
        if engine is None:
            raise RuntimeError("no OCR language pack installed (Settings > Time & language > Language)")
        result = await engine.recognize_async(bitmap)
        return "\n".join(line.text for line in result.lines)

    try:
        text = asyncio.run(run())
    except RuntimeError as exc:                       # already inside a loop (server thread)
        if "asyncio.run() cannot be called" not in str(exc):
            raise
        import threading

        box: Dict[str, Any] = {}

        def worker() -> None:
            try:
                box["text"] = asyncio.run(run())
            except BaseException as inner:            # noqa: BLE001 - reported to the caller
                box["error"] = inner

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=25)
        if "error" in box:
            raise RuntimeError(str(box["error"]))
        text = box.get("text", "")
    return (text or ""), "windows-ocr"


def _ocr_tesseract(image: Path, lang: str = "eng") -> Tuple[Optional[str], str]:
    exe = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract") or ""
    if not exe:
        for guess in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                      r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
            if Path(os.path.expandvars(guess)).is_file():
                exe = os.path.expandvars(guess)
                break
    if not exe:
        return None, "tesseract not installed"
    out_file = image.with_suffix(".txt")
    try:
        proc = subprocess.run([exe, str(image), str(out_file.with_suffix("")), "-l", lang, "--psm", "6"],
                              capture_output=True, text=True, timeout=45, encoding="utf-8", errors="replace")
    except (subprocess.TimeoutExpired, OSError) as exc:
        return None, f"tesseract failed: {exc}"
    if proc.returncode != 0 or not out_file.is_file():
        return None, f"tesseract said: {(proc.stderr or '').strip()[:120] or 'no output'}"
    text = out_file.read_text(encoding="utf-8", errors="replace")
    try:
        out_file.unlink(missing_ok=True)
    except OSError:
        pass
    return text, "tesseract"


def ocr(path: str = "", text_only: bool = False) -> Dict[str, Any]:
    """Recognise the words in a screenshot (or take one and recognise that)."""
    image: Optional[Path] = None
    if path:
        candidate = Path(os.path.expandvars(os.path.expanduser(path)))
        if not candidate.is_file():
            return _result(False, f"No image at {path}.")
        image = candidate
    else:
        grabbed = capture()
        if not grabbed.get("ok"):
            return grabbed
        image = Path(grabbed["path"])
    engines: List[Tuple[str, Callable[[Path], Tuple[Optional[str], str]]]] = []
    engines.append(("windows", _ocr_windows))
    engines.append(("tesseract", _ocr_tesseract))
    used, note = "", ""
    text: Optional[str] = None
    for name, engine in engines:
        try:
            text, note = engine(image)
        except Exception as exc:  # noqa: BLE001 - one broken engine must not stop the other
            text, note = None, f"{name} error: {type(exc).__name__}: {exc}"
        if text and text.strip():
            used = name
            break
    if not text or not text.strip():
        return _result(False, f"No local OCR could read the screen ({note or 'no OCR engine found'}). "
                              "Install the Windows OCR language pack or Tesseract, or ask me to "
                              "“look at the screen” and I'll send the image to the AI instead.",
                       path=str(image), ocr_available=False, detail=note)
    clean = re.sub(r"[ \t]{2,}", " ", text).strip()
    if text_only:
        return _result(True, clean, path=str(image), engine=used, characters=len(clean))
    return _result(True, f"What's on the screen ({used} OCR, {len(clean)} characters): {clean[:3000]}",
                   text=clean, path=str(image), engine=used, characters=len(clean),
                   lines=[l.strip() for l in clean.splitlines() if l.strip()][:200])


# --------------------------------------------------------------------------- "read me the top 3"
def top_results(text: str, count: int = 3, context: int = 2) -> List[Dict[str, Any]]:
    """Rank lines of OCR text into "the top N things", browser-chrome noise removed.

    Not a parser: OCR gives a jagged column of words, so the score is deliberately coarse -
    does it look like a title, does it have a number or price, is it the right length, is it
    followed by a URL-ish line.  Good enough to read out "top three listings" reliably, and it
    never crashes on garbage the way a CSS-style parser would.
    """
    want = max(1, min(int(count or 3), 12))
    lines = [re.sub(r"\s+", " ", raw).strip() for raw in (text or "").splitlines()]
    lines = [l for l in lines if l]
    scored: List[Tuple[float, int, Dict[str, Any]]] = []
    for index, line in enumerate(lines):
        if len(line) < 8 or len(line) > 180:
            continue
        if _LINE_NOISE.search(line) or _CHROME_LINE.search(line):
            continue
        words = len(line.split())
        if words < 2 or words > 26:
            continue
        score = 1.0
        match = _TITLEISH.match(line)
        if match:
            score += 1.2
            line = match.group(1).strip()
        if re.search(r"[$₹€£]\s?\d|₹\s?\d", line):
            score += 1.6                                        # a price: it IS a listing
        if re.search(r"\b\d+(?:\.\d+)?\s?(?:/5|out of 5|stars?|ratings?|reviews?|%)\b", line, re.I):
            score += 1.3
        if re.search(r"https?://|\b\w+\.(com|in|org|net|io|dev|co|au|uk)\b", line, re.I):
            score += 0.9
        if re.search(r"\b(review|price|rating|best|top|vs|versus|comparison|guide|tutorial)\b", line, re.I):
            score += 0.7
        if line.endswith((".", "!", "?", ":", "»", "…")) and not re.search(r"[.!?] \S", line):
            score -= 0.5                                          # a sentence fragment, not a title
        # Search engines and shops put the source after an em/en dash and the URL on the
        # next line, so that shape is worth more than any single keyword.
        if re.search(r"\s[-–—|·]\s*[A-Z][\w .&'-]{2,26}$", line):
            score += 1.15
        neighbours = lines[max(0, index - context):index + context + 1]
        score += 0.25 * sum(1 for n in neighbours if re.search(r"https?://|\.(com|org|net)\b", n, re.I))
        if re.search(r"https?://", line) and len(line.split()) <= 5:
            score -= 0.9                                          # a bare link, not a listing
        if re.match(r"^\d[\d.,]*\s*(?:out of|/)\s*\d", line) or re.search(r"^\(?\d[\d.,]*\)?\s*(?:stars?|ratings?|reviews?)", line, re.I):
            score -= 1.25                                         # a rating blob under a title
        scored.append((score, index, {"title": line[:180], "line": index + 1,
                                      "before": lines[max(0, index - 1):index],
                                      "after": lines[index + 1:index + 3]}))
    scored.sort(key=lambda row: (-row[0], row[1]))
    picked: List[Dict[str, Any]] = []
    taken: set = set()
    for score, index, item in scored:
        key = re.sub(r"[^a-z0-9]", "", item["title"].lower())[:60]
        if not key or key in taken:
            continue
        taken.add(key)
        row = dict(item)
        row["score"] = round(score, 2)
        picked.append(row)
        if len(picked) >= want:
            break
    return picked


def read_screen(count: int = 3, target: str = "screen") -> Dict[str, Any]:
    """The one-call version of "read out the top three listings on my screen"."""
    grabbed = capture(target=target)
    if not grabbed.get("ok"):
        return grabbed
    image = Path(grabbed["path"])
    found = ocr(str(image))
    if not found.get("ok"):
        return found
    items = top_results(str(found.get("text", "")), count=count)
    if not items:
        return _result(True, f"I read {found.get('characters', 0)} characters but could not find "
                             "anything that looks like a listing. Ask me to describe the screen "
                             "instead and I'll send the image to the AI.",
                       path=str(image), text=str(found.get("text", ""))[:1500], items=[])
    spoken = "; ".join(f"{i + 1}. {item['title']}" for i, item in enumerate(items))
    return _result(True, f"Top {len(items)} on screen: {spoken}", items=items, path=str(image),
                   engine=found.get("engine", ""), text=str(found.get("text", ""))[:2000])


# --------------------------------------------------------------------------- AI eyes
def describe(question: str = "", target: str = "screen", region: Optional[List[int]] = None) -> Dict[str, Any]:
    """Send the screen to a vision-capable model and let it answer about what it sees."""
    if not SETTINGS.screen_vision_enabled:
        return _result(False, "SCREEN_VISION is off in .env, so I will not send screenshots to an "
                              "AI provider. Turn it on (or ask me to use OCR instead).")
    grabbed = capture(target=target) if not region else winops.screenshot(region=region)
    if not grabbed.get("ok"):
        return grabbed
    image = Path(grabbed["path"])
    data_url = _to_data_url(image)
    if not data_url:
        return _result(False, f"I captured {image.name} but could not encode it for the AI.")
    ask = (question or "").strip() or ("Describe what is on this screen: name the app or page, "
                                       "the main content, and anything that needs attention.")
    try:
        import llm_providers
    except Exception as exc:  # noqa: BLE001
        return _result(False, f"Provider layer unavailable: {exc}", path=str(image))
    if not llm_providers.POOL.configured():
        return _result(False, "No AI provider key is set, so I can't look at the screen with a "
                              "vision model. Add GROQ_API_KEY or GEMINI_API_KEY to .env.",
                       path=str(image))
    try:
        outcome = llm_providers.POOL.vision(
            ask, images=[data_url],
            max_tokens=min(700, max(180, SETTINGS.llm_max_tokens)))
    except llm_providers.LlmError as exc:
        return _result(False, f"No provider could look at the screen: {exc}", path=str(image),
                       hint="Install the Windows OCR language pack for a local fallback.")
    answer = (outcome.get("content") or "").strip()
    if not answer:
        return _result(False, "The vision model returned nothing.", path=str(image),
                       provider=outcome.get("provider"), model=outcome.get("model"))
    return _result(True, answer, path=str(image), provider=outcome.get("provider"),
                   model=outcome.get("model"), window=grabbed.get("title", ""),
                   latency_ms=outcome.get("latency_ms", 0))


def window_text(limit: int = 24) -> Dict[str, Any]:
    """What the user is looking at, cheaply: the focused window's title and process."""
    front = winops.foreground_window()
    if not front.get("ok"):
        return _result(False, front.get("message", "No foreground window."))
    rows = winops.windows()[:limit]
    rows = rows[:limit]
    lines = []
    for row in rows:
        marker = "->" if row.get("title") == front.get("title") else "  "
        lines.append(f"{marker} {str(row.get('title', ''))[:70]} [{row.get('process', '?')}]")
    return _result(True, f"You are looking at “{front.get('title', '')}”. Open windows: "
                        + " | ".join(l.strip("→ ").strip() for l in lines[:6]),
                   title=str(front.get("title", "")), hwnd=front.get("hwnd"), windows=rows[:limit],
                   rect=front.get("rect"))


def save_note_from_screen(question: str = "", to: str = "") -> Dict[str, Any]:
    """Capture → read → file.  'screenshot and save what it says to notes' in one tool call."""
    read = ocr() if not question else describe(question)
    if not read.get("ok"):
        return read
    body = read.get("text") or read.get("message", "")
    if not body:
        return _result(False, "The screen produced no text to save.")
    stamp = time.strftime("%Y-%m-%d %H:%M")
    name = to.strip() if to.strip() else f"screen-{time.strftime('%Y%m%d-%H%M%S')}.md"
    try:
        import files

        saved = files.write(name, f"# Screen capture {stamp}\n\n{body}\n", mode="overwrite")
    except Exception as exc:  # noqa: BLE001
        return _result(False, f"Could not save the capture: {exc}")
    return {**saved, "message": f"Saved what I could read to {Path(saved.get('path', name)).name}. "
                               f"({len(body)} characters)", "characters": len(body)}


__all__ = ["capture", "ocr", "top_results", "read_screen", "describe", "window_text",
           "save_note_from_screen"]
