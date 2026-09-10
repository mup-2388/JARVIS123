"""Background hearing: JARVIS listens while you are in another app, and only wakes on its name.

How it avoids being a nightmare
------------------------------
* The microphone is never transcribed continuously.  A tiny C-free energy gate (:func:`gate`)
  watches 30 ms frames; only after speech energy starts does it hand a segment (pre-roll + the
  trailing silence gap) to Whisper.  That's one small ASR call per utterance instead of one per
  frame, which is what keeps a 4 GB card usable.
* A segment only becomes a command if the wake word is in it (:func:`matches_wake`) - or if the
  follow-up window is open (you just spoke to JARVIS, so the next sentence is yours) or the
  push-to-talk key is held.
* While JARVIS is talking, the listener is deaf (`duck()`), so it never answers itself.
* Everything is one thread, everything is optional: no ``sounddevice``, no mic, no Windows -> the
  listener says exactly what is missing in :func:`status` and the HUD shows it, instead of
  silently doing nothing.

The floating bar is the other half: a frameless, top-most, no-focus window that slides in on the
wake word or the hotkey, so you can type when a game/IDE has the focus and stealing it would hurt.
"""

from __future__ import annotations

import collections
import difflib
import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import config
import winops
from config import SETTINGS, get_logger

log = get_logger("wake")

SAMPLE_RATE = 16_000
FRAME_MS = 30
PRE_ROLL = 0.35            # seconds of audio kept before the energy trigger
SILENCE_END = 0.75         # stop a segment after this much trailing quiet
MAX_SEGMENT = 12.0         # never hand Whisper more than this
MIN_SEGMENT = 0.35         # shorter than this is a cough, not a word
ENERGY_OPEN = 340.0        # int16 RMS that counts as speech (room-tuneable via .env)
ENERGY_HYSTERESIS = 0.55   # must fall to this fraction of the open level to close

_KEY_VK = {"f12": 0x7B, "f11": 0x7A, "f10": 0x79, "f9": 0x78, "scrolllock": 0x91,
           "capslock": 0x14, "pause": 0x13, "menu": 0x12, "rcontrol": 0xA3, "lcontrol": 0xA2,
           "rwin": 0x5C, "printscreen": 0x2C, "insert": 0x2D, "f8": 0x77, "f7": 0x76}

#: Words that Whisper commonly produces for "Jarvis" on a laptop mic.
_WAKE_VARIANTS = ("jarvis", "jervis", "jarvi", "yarves", "harvest", "jarvi's", "jervis's",
                  "java's", "jarbis", "charles", "jervais", "jarvy")
#: Variants that are also ordinary English words.  Whisper really does print these for a
#: laptop mic, but "harvest my wheat in the farm sim" must not wake anything, so they only
#: count as the wake word when they open the sentence (optionally after "hey"/"ok").
_AMBIGUOUS_VARIANTS = {"harvest", "charles", "java's", "marshal", "servis"}


def _result(ok: bool, message: str, **extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": bool(ok), "message": message}
    out.update(extra)
    return out


def wake_words() -> Tuple[str, ...]:
    raw = (SETTINGS.wake_words or "jarvis").strip()
    words = tuple(dict.fromkeys(w.strip().lower() for w in re.split(r"[,; ]+", raw) if w.strip()))
    return words or ("jarvis",)


def matches_wake(text: str, fuzzy: Optional[bool] = None) -> bool:
    """True when the utterance is addressed to JARVIS.

    Whisper mishears a name more than it mishears a sentence, so this is deliberately
    forgiving: exact word, then any token within the provider's own variant list, then a
    difflib ratio of 0.84+ against a wake word for a three-character typo.  ``charles`` is in
    the variant list because that is what a laptop mic really produces; the price of a false
    positive is one extra look at the transcript, and the price of a false negative is the user
    shouting at a laptop.
    """
    words = wake_words()
    tokens = re.findall(r"[a-z']+", (text or "").lower())
    if not tokens:
        return False
    for token in tokens:
        if token in words:
            return True
    allow_fuzzy = SETTINGS.wake_fuzzy if fuzzy is None else bool(fuzzy)
    if not allow_fuzzy:
        return False
    for index, token in enumerate(tokens):
        if len(token) < 4:
            continue
        leading = index == 0 or (index == 1 and tokens[0] in {"hey", "ok", "oh", "yo", "hi"})
        if token in _AMBIGUOUS_VARIANTS:
            if leading:
                return True
            continue
        if token in _WAKE_VARIANTS or any(v in token for v in words):
            return True
        for word in tuple(words) + _WAKE_VARIANTS:
            if len(token) >= len(word) - 2 and difflib.SequenceMatcher(None, token, word).ratio() >= 0.84:
                return True
    return False


def strip_wake(text: str) -> str:
    """Remove the address, so the router sees the request and not the greeting."""
    cleaned = (text or "").strip()
    words = sorted(wake_words(), key=len, reverse=True)
    pattern = r"^\s*(?:hey|ok|oh|yo|hi|hello|alright|okay)?[\s,.-]*(?:" + "|".join(map(re.escape, words)) + \
        r"|charles|jervis|jarvi|harvest)[\s,.-]+"
    stripped = re.sub(pattern, "", cleaned, count=1, flags=re.I)
    if not stripped:
        stripped = re.sub(r"^\s*(?:hey|ok|yo)[\s,.-]+", "", cleaned, count=1, flags=re.I)
    return stripped.strip() or cleaned.strip()


def rms(frame: bytes) -> float:
    """Int16 RMS of one frame, without importing numpy (it is optional here)."""
    if len(frame) < 2:
        return 0.0
    count = len(frame) // 2
    values = memoryview(frame).cast("h")[:count]
    total = 0
    for value in values:
        total += value * value
    return (total / max(1, count)) ** 0.5


def gate(rms_curve: Sequence, open_level: float = ENERGY_OPEN, close_fraction: float = ENERGY_HYSTERESIS
         ) -> List[Tuple[int, int]]:
    """Label the speech spans in a stream of frame energies: [(start_frame, end_frame), ...].

    Hysteresis (open at N, close at 0.55 N) is the whole trick - a single threshold chops
    consonants into twelve segments and the transcript comes back as confetti.
    """
    spans: List[Tuple[int, int]] = []
    start = -1
    low = max(1.0, open_level * close_fraction)
    quiet = 0
    for index, level in enumerate(rms_curve):
        if start < 0:
            if level >= open_level:
                start = index
                quiet = 0
        else:
            if level < low:
                quiet += 1
            else:
                quiet = 0
            if quiet * FRAME_MS / 1000.0 >= SILENCE_END:
                spans.append((max(0, start - int(PRE_ROLL * 1000 / FRAME_MS)), index - quiet))
                start = -1
                quiet = 0
    if start >= 0:
        spans.append((max(0, start - int(PRE_ROLL * 1000 / FRAME_MS)), len(rms_curve) - 1))
    return [(a, b) for a, b in spans if (b - a) * FRAME_MS / 1000.0 >= MIN_SEGMENT]


@dataclass
class WakeEvent:
    """One utterance that JARVIS decided to act on."""

    text: str
    woke: bool = True
    via: str = "wake-word"            # wake-word | hotkey | follow-up | push-to-talk | bar
    at: float = field(default_factory=time.time)
    duration_s: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "woke": self.woke, "via": self.via, "at": self.at,
                "seconds": round(self.duration_s, 2)}


class WakeListener:
    """Energy-gated, wake-word-gated background ear.  Start it once, from main.py."""

    def __init__(self, on_command: Optional[Callable[[WakeEvent], None]] = None,
                 on_state: Optional[Callable[[Dict[str, Any]], None]] = None,
                 on_speaking: Optional[Callable[[bool], None]] = None) -> None:
        self.on_command = on_command
        self.on_state = on_state
        self.on_speaking = on_speaking
        self._thread: Optional[threading.Thread] = None
        self._hotkey_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._followup_until = 0.0
        self._ducked_until = 0.0
        self._segments: "queue.Queue[bytes]" = queue.Queue(maxsize=8)
        self.state: Dict[str, Any] = {"running": False, "armed": False, "last_wake": 0.0,
                                      "heard": 0, "woke": 0, "reason": "", "engine": "",
                                      "hotkey": "", "ptt": False}
        self.recent: Deque[Dict[str, Any]] = collections.deque(maxlen=24)

    # -- lifecycle ---------------------------------------------------------
    def available(self) -> Tuple[bool, str]:
        """Can the ear be switched on at all - and if not, exactly what to install."""
        try:
            __import__("sounddevice")
        except Exception as exc:  # noqa: BLE001 - half-installed package is still "unavailable"
            return False, ("background listening needs `pip install sounddevice` (numpy comes with "
                           f"it) inside .venv - the HUD mic button works without it ({exc})")
        if os.environ.get("JARVIS_ALLOW_MIC", "1").strip().lower() not in {"1", "true", "yes", "on"}:
            return False, "JARVIS_ALLOW_MIC=0 is set, so the microphone is disabled"
        return True, ""

    def start(self) -> Dict[str, Any]:
        if not SETTINGS.wake_enabled:
            return _result(False, "WAKE_WORD_ENABLED is false in .env, so I will only respond in "
                                  "the HUD or the bar.", available=False)
        with self._lock:
            if self._thread and self._thread.is_alive():
                return _result(True, "Already listening.", running=True)
        ok, why = self.available()
        if not ok:
            self.state["reason"] = why
            return _result(False, why, available=False)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="jarvis-wake", daemon=True)
        self._thread.start()
        self._hotkey_thread = threading.Thread(target=self._hotkeys, name="jarvis-hotkeys", daemon=True)
        self._hotkey_thread.start()
        self.state.update(running=True, reason="")
        self._publish()
        return _result(True, "Listening for " + "/".join(wake_words()) + ".", running=True,
                       wake_words=list(wake_words()), hotkey=SETTINGS.wake_command_key,
                       push_to_talk=SETTINGS.wake_push_to_talk)

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        with self._lock:
            thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=3)
        self.state["running"] = False
        self._publish()
        return _result(True, "Stopped listening.", running=False)

    # -- external hints ----------------------------------------------------
    def duck(self, seconds: float = 3.0) -> None:
        """Go deaf while JARVIS speaks (and for a beat after), so it cannot wake itself."""
        self._ducked_until = max(self._ducked_until, time.time() + max(0.0, seconds))

    def allow_followup(self, seconds: Optional[float] = None) -> None:
        """Open a window where the next sentence is accepted with no wake word."""
        window = SETTINGS.wake_followup_seconds if seconds is None else seconds
        if window and window > 0:
            self._followup_until = time.time() + window

    def accepts_without_wake(self) -> bool:
        return time.time() < self._followup_until

    def _publish(self) -> None:
        if self.on_state:
            try:
                self.on_state(dict(self.state))
            except Exception as exc:  # noqa: BLE001 - the HUD is not allowed to break hearing
                log.debug("state callback failed: %s", exc)

    # -- the ear -----------------------------------------------------------
    def _run(self) -> None:  # pragma: no cover - needs a microphone
        try:
            # sounddevice brings numpy with it; the callback receives numpy int16 frames.
            import sounddevice as sd  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self.state.update(running=False, reason=f"sounddevice/numpy missing: {exc}")
            self._publish()
            return

        frames = int(SAMPLE_RATE * FRAME_MS / 1000)
        ring: Deque[bytes] = collections.deque(maxlen=int((MAX_SEGMENT * 1000 / FRAME_MS)))
        pre_roll = int(PRE_ROLL * 1000 / FRAME_MS)
        speaking = False
        silence_frames = 0
        heard: List[bytes] = []

        def callback(indata: Any, frame_count: int, time_info: Any, status: Any) -> None:
            nonlocal speaking, silence_frames
            if status:
                log.debug("mic status: %s", status)
            chunk = bytes(indata[:frame_count].tobytes())
            ring.append(chunk)
            level = rms(chunk)
            deaf = time.time() < self._ducked_until
            if not speaking:
                if deaf or level < self._open_level():
                    return
                speaking = True
                silence_frames = 0
                heard.clear()
                for item in list(ring)[-pre_roll:]:
                    heard.append(item)
                self.state["armed"] = True
                self._publish()
                return
            heard.append(chunk)
            if level < self._open_level() * ENERGY_HYSTERESIS:
                silence_frames += 1
            else:
                silence_frames = 0
            if silence_frames * FRAME_MS / 1000.0 >= SILENCE_END:
                speaking = False
                self.state["armed"] = False
                self._publish()
                blob = b"".join(heard)
                if len(blob) / 2 / SAMPLE_RATE >= MIN_SEGMENT:
                    try:
                        self._segments.put_nowait(blob)
                    except queue.Full:
                        log.debug("dropping a segment: the transcriber is behind")
                heard.clear()

        self.state["engine"] = "sounddevice"
        self._publish()
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                                blocksize=frames, callback=callback):
                while not self._stop.is_set():
                    try:
                        blob = self._segments.get(timeout=0.25)
                    except queue.Empty:
                        continue
                    self._transcribe(blob)
        except OSError as exc:
            self.state.update(running=False, reason=f"microphone unavailable: {exc}")
            self._publish()
        except Exception as exc:  # noqa: BLE001
            self.state.update(running=False, reason=f"listener stopped: {type(exc).__name__}: {exc}")
            self._publish()

    def _open_level(self) -> float:
        try:
            return max(60.0, float(os.environ.get("WAKE_ENERGY", "") or ENERGY_OPEN))
        except ValueError:
            return ENERGY_OPEN

    def _transcribe(self, pcm: bytes) -> None:
        try:
            import audio_engine

            transcript = audio_engine.get_engine().listen_pcm(pcm, SAMPLE_RATE)
        except Exception as exc:  # noqa: BLE001
            log.debug("background transcribe failed: %s", exc)
            self.state["reason"] = f"STT unavailable: {exc}"
            self._publish()
            return
        text = (getattr(transcript, "text", "") or "").strip()
        seconds = len(pcm) / 2 / SAMPLE_RATE
        if not text:
            # "Nothing heard" and "the STT engine is broken" are different problems; only
            # report the latter, or the HUD will keep saying "ready" while the ear is deaf.
            engine = str(getattr(transcript, "engine", "") or "")
            if engine in {"unavailable", "missing-file"} or engine.startswith(("decode-error", "error")):
                self.state["reason"] = f"STT: {engine}"
                self._publish()
            return
        with self._lock:
            self.state["heard"] = int(self.state.get("heard", 0)) + 1
            self.recent.append({"text": text, "at": time.time(), "seconds": round(seconds, 2)})
        follow = self.accepts_without_wake()
        woke = matches_wake(text)
        if not (woke or follow):
            log.debug("ignored (no wake word): %s", text[:80])
            return
        clean = strip_wake(text) if woke else text
        if not clean.strip():
            self._emit(WakeEvent(text="I'm listening", woke=True, via="wake-word", duration_s=seconds))
            return
        self.state["last_wake"] = time.time()
        self.state["woke"] = int(self.state.get("woke", 0)) + 1
        self._publish()
        self._emit(WakeEvent(text=clean, woke=woke, via="wake-word" if woke else "follow-up",
                             duration_s=seconds))

    def _emit(self, event: WakeEvent) -> None:
        if self.on_command:
            try:
                self.on_command(event)
            except Exception as exc:  # noqa: BLE001 - a bad handler must not kill the ear
                log.warning("wake command handler failed: %s", exc)

    # -- the hands ---------------------------------------------------------
    def _hotkeys(self) -> None:  # pragma: no cover - needs Windows
        """Poll the toggle / push-to-talk keys.  No message loop, no keyboard hook, no deps."""
        toggle = _KEY_VK.get((SETTINGS.wake_command_key or "").lower(), 0x7B)
        ptt = _KEY_VK.get((SETTINGS.wake_push_to_talk or "").lower(), 0xA3)
        last_toggle = False
        held_since = 0.0
        self.state.update(hotkey=f"{SETTINGS.wake_command_key} toggles the bar, "
                                f"hold {SETTINGS.wake_push_to_talk} to talk")
        while not self._stop.is_set():
            down = winops.key_down(toggle)
            if down and not last_toggle:
                self._toggle_bar()
            last_toggle = down
            ptt_down = winops.key_down(ptt)
            if ptt_down and held_since == 0.0:
                held_since = time.time()
                self.state["ptt"] = True
                self._publish()
            elif not ptt_down and held_since:
                held = time.time() - held_since
                held_since = 0.0
                self.state["ptt"] = False
                self._publish()
                if 0.25 < held < 30:
                    self.record_once(seconds=held)
            time.sleep(0.04)

    def _toggle_bar(self) -> None:
        controller = getattr(config, "BAR_CONTROLLER", None)
        action = "hide" if getattr(controller, "visible", False) else "show"
        if controller:
            try:
                controller(action)
            except Exception as exc:  # noqa: BLE001
                log.debug("bar toggle failed: %s", exc)
        self.state["last_toggle"] = time.time()
        self._publish()

    def record_once(self, seconds: float = 4.0) -> Dict[str, Any]:
        """Push-to-talk / HUD button path: grab N seconds from the mic and treat it as a command."""
        try:
            import audio_engine

            engine = audio_engine.get_engine()
        except Exception as exc:  # noqa: BLE001
            return _result(False, f"No speech-to-text engine here: {exc}")
        try:
            transcript = engine.stt.record_from_mic(seconds=max(0.5, min(float(seconds), 25.0)))
        except Exception as exc:  # noqa: BLE001
            return _result(False, f"Could not record: {exc}", hint="pip install sounddevice numpy")
        text = (getattr(transcript, "text", "") or "").strip()
        if not text:
            return _result(False, "I heard nothing - the mic may be muted or the wrong input device.",
                           heard="")
        self._emit(WakeEvent(text=strip_wake(text), woke=True, via="push-to-talk",
                             duration_s=float(seconds)))
        return _result(True, f"Heard: {text}", heard=text)

    def status(self) -> Dict[str, Any]:
        ok, why = self.available()
        out = dict(self.state)
        out.update(mic_available=ok, detail=out.get("reason") or why,
                   wake_words=list(wake_words()), followup_open=self.accepts_without_wake(),
                   ducked=time.time() < self._ducked_until,
                   recent=[dict(r) for r in list(self.recent)[-6:]])
        return out


#: The single instance main.py starts and the server reports.
LISTENER = WakeListener()

__all__ = ["WakeListener", "WakeEvent", "LISTENER", "matches_wake", "strip_wake", "wake_words",
           "gate", "rms"]
