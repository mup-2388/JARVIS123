"""
audio_engine.py -- J.A.R.V.I.S. voice hardware layer.

Two independent subsystems, both lazily loaded so the HUD boots in <1 s and
only pays the model-load cost the first time sound actually flows:

STT  ``faster-whisper`` (CTranslate2 runtime)
     * ``device="cuda"`` + ``compute_type="int8"`` by default -> the ``base``
       model occupies ~600 MB VRAM, leaving the RTX 3050's 4 GB free for the
       XTTS-v2 vocoder without an OOM.  ``tiny`` is available for <150 ms
       latency.
     * If CUDA init fails (driver hiccup, another process hogging the GPU,
       running on the iGPU laptop) we walk a fallback ladder
       cuda/int8 -> cpu/int8 -> cpu/float32 instead of crashing.

TTS  Coqui ``XTTS-v2`` zero-shot voice cloning from ``assets/jarvis_sample.wav``
     * ``low_vram=True`` streams the speaker encoder per call, which is the
       difference between fitting and not fitting on 4 GB.
     * If the Coqui stack is unavailable (fresh clone, no checkpoint yet) we
       fall back to the built-in Windows SAPI5 synthesiser so JARVIS *always*
       has a voice.

Everything is thread-safe: the FastAPI event loop, the Discord thread and the
HUD's WebSocket handler may all ask for audio at once.
"""

from __future__ import annotations

import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import wave
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import config
from config import SETTINGS, get_logger

log = get_logger("audio")

TARGET_SR = 16000          # what Whisper wants
TTS_SR = SETTINGS.tts_sample_rate
SILENCE_RMS_THRESHOLD = 0.012
MAX_CHUNK_SECONDS = 22      # guard against an accidental 10-minute recording


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class Transcript:
    """Normalised STT output the router and HUD both consume."""
    text: str = ""
    language: str = ""
    duration: float = 0.0
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    segments: List[Dict[str, Any]] = field(default_factory=list)
    engine: str = ""
    latency_ms: int = 0

    @property
    def confidence(self) -> float:
        """0..1 heuristic blending logprob and the no-speech token probability."""
        if not self.text.strip():
            return 0.0
        lp = max(-1.35, min(0.0, self.avg_logprob))
        base = 0.5 + 0.5 * ((lp + 1.35) / 1.35)
        return round(max(0.0, min(1.0, base - self.no_speech_prob * 0.45)), 3)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "duration": round(self.duration, 2),
            "confidence": self.confidence,
            "engine": self.engine,
            "latency_ms": self.latency_ms,
            "segments": self.segments,
        }


# ---------------------------------------------------------------------------
# ffmpeg bridge (browser mic -> 16 kHz mono wav)
# ---------------------------------------------------------------------------

def _ffmpeg_bin() -> Optional[str]:
    return shutil.which("ffmpeg")


def decode_to_wav(payload: bytes, src_ext: str = "webm") -> Tuple[Optional[Path], str]:
    """Convert arbitrary browser/encoder audio to the 16 kHz mono WAV Whisper wants.

    Returns ``(path, error)``; exactly one of them is set.
    """
    if not payload:
        return None, "empty audio payload"
    out = Path(tempfile.gettempdir()) / f"jarvis_in_{os.getpid()}_{threading.get_ident()}.wav"
    if src_ext in {"wav"} and payload[:4] == b"RIFF":
        # Already PCM-ish; sniff-convert anyway so weird float formats normalise.
        pass
    ff = _ffmpeg_bin()
    if not ff:
        if src_ext == "wav":
            out.write_bytes(payload)
            return out, ""
        return None, "ffmpeg is not installed: browser audio (webm/opus) cannot be decoded. Install ffmpeg or send 16 kHz WAV."
    tmp_in = out.with_suffix("." + (src_ext or "bin"))
    tmp_in.write_bytes(payload)
    cmd = [
        ff, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(tmp_in),
        "-ac", "1", "-ar", str(TARGET_SR), "-f", "wav", str(out),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None, "ffmpeg timed out decoding the microphone clip"
    finally:
        tmp_in.unlink(missing_ok=True)
    if proc.returncode != 0 or not out.is_file():
        return None, f"ffmpeg failed: {proc.stderr.strip()[:220] or 'no output produced'}"
    return out, ""


def pcm16_to_wav(pcm: bytes, sample_rate: int = TARGET_SR) -> Path:
    """Wrap raw little-endian s16 mono PCM (WebSocket mic stream) in a WAV file."""
    out = Path(tempfile.gettempdir()) / f"jarvis_pcm_{os.getpid()}_{threading.get_ident()}.wav"
    with wave.open(str(out), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return out


def rms_of_wav(path: Path, window: float = 0.05) -> float:
    """Peak RMS of a WAV file -- cheap VAD used to reject "you never spoke"."""
    try:
        with wave.open(str(path), "rb") as handle:
            width, rate = handle.getsampwidth(), handle.getframerate()
            if width != 2:
                return 1.0
            frames = handle.readframes(int(rate * window))
        n = len(frames) // 2
        if n == 0:
            return 0.0
        # array-based max/min scans every sample: a strided loop can alias with
        # periodic waveforms (50 Hz hum, a test tone) and read real audio as
        # silence, which would silently drop the user's utterance.
        samples = array("h")
        samples.frombytes(frames[: n * 2])
        if sys.byteorder != "little":
            samples.byteswap()
        peak = max(max(samples), -min(samples)) / 32768.0
        if peak >= SILENCE_RMS_THRESHOLD:
            return peak
        step = max(1, n // 512)
        total, count = 0.0, 0
        for i in range(0, n, step):
            val = samples[i] / 32768.0
            total += val * val
            count += 1
        return math.sqrt(total / count) if count else 0.0
    except Exception:
        return 1.0  # unreadable -> let Whisper decide


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------

class SpeechToText:
    """faster-whisper wrapper tuned for an RTX 3050 4 GB."""

    _FALLBACKS: Tuple[Tuple[str, str], ...] = (
        ("cuda", "int8"),
        ("cuda", "float16"),
        ("cpu", "int8"),
        ("cpu", "float32"),
    )

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._state = "unloaded"          # unloaded | loading | ready | failed
        self._error = ""
        self.device = SETTINGS.whisper_device
        self.compute_type = SETTINGS.whisper_compute_type
        self.model_name = SETTINGS.whisper_model
        self.warm = False

    # -- loading ---------------------------------------------------------
    def _candidates(self) -> List[Tuple[str, str]]:
        want = (self.device.lower(), self.compute_type.lower())
        ladder = [want] + [c for c in self._FALLBACKS if c != want]
        if not SETTINGS.whisper_allow_cpu_fallback:
            ladder = [c for c in ladder if c[0] != "cpu"] or [want]
        return ladder

    def load(self, force: bool = False) -> bool:
        with self._lock:
            if self._model is not None and not force:
                return True
            if self._state == "loading":
                return False
            self._state, self._error = "loading", ""
            try:
                from faster_whisper import WhisperModel  # type: ignore
            except Exception as exc:
                self._state = "failed"
                self._error = f"faster-whisper not importable ({exc}). pip install faster-whisper"
                log.warning(self._error)
                return False

            last = ""
            for device, compute in self._candidates():
                try:
                    log.info("loading whisper '%s' on %s/%s", self.model_name, device, compute)
                    self._model = WhisperModel(
                        self.model_name,
                        device=device,
                        compute_type=compute,
                        cpu_threads=max(1, SETTINGS.whisper_cpu_threads),
                        num_workers=max(1, SETTINGS.whisper_num_workers),
                    )
                    self.device, self.compute_type = device, compute
                    self._state = "ready"
                    log.info("whisper ready on %s/%s", device, compute)
                    return True
                except Exception as exc:  # OOM / missing CUDA / bad model name
                    last = f"{type(exc).__name__}: {exc}"
                    log.warning("whisper %s/%s rejected: %s", device, compute, _clip(last))
                    if device == "cpu" and compute == "float32" and self.model_name in {"small", "medium", "large-v2", "large-v3"}:
                        # Last-resort downshift so a laptop with no CUDA still hears us.
                        self.model_name = "base"
                    continue
            self._state = "failed"
            self._error = f"could not load Whisper ({last})"
            return False

    # -- transcription -----------------------------------------------------
    @property
    def status(self) -> Dict[str, Any]:
        return {
            "state": self._state,
            "model": self.model_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "error": self._error,
        }

    def transcribe_wav(self, path: Path, language: Optional[str] = None) -> Transcript:
        import time as _time

        started = _time.perf_counter()
        if not path.is_file():
            return Transcript(text="", engine="missing-file")
        if rms_of_wav(path) < SILENCE_RMS_THRESHOLD:
            return Transcript(text="", engine="vad-gate", duration=_measure(path),
                              latency_ms=int((_time.perf_counter() - started) * 1000))
        if not self._model and not self.load():
            return Transcript(text="", engine="unavailable",
                              latency_ms=int((_time.perf_counter() - started) * 1000))

        kwargs: Dict[str, Any] = {
            "beam_size": max(1, SETTINGS.whisper_beam_size),
            "vad_filter": SETTINGS.whisper_vad_filter,
            "vad_parameters": {"min_silence_duration_ms": 300, "speech_pad_ms": 200},
            "task": "transcribe",
            "word_timestamps": False,
        }
        if language or SETTINGS.whisper_language:
            kwargs["language"] = (language or SETTINGS.whisper_language).strip() or None

        try:
            segments, info = self._model.transcribe(str(path), **kwargs)
        except TypeError:  # older faster-whisper without vad_parameters
            kwargs.pop("vad_parameters", None)
            segments, info = self._model.transcribe(str(path), **kwargs)
        except Exception as exc:
            log.error("transcribe failed: %s", exc)
            self._model = None  # force a reload (device may have dropped)
            return Transcript(text="", engine=f"error:{type(exc).__name__}")

        out: List[Dict[str, Any]] = []
        chunks: List[str] = []
        avg_lp, nsp = 0.0, 0.0
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            chunks.append(text)
            out.append({"start": round(float(seg.start), 2), "end": round(float(seg.end), 2), "text": text})
            avg_lp += float(getattr(seg, "avg_logprob", 0.0) or 0.0)
            nsp += float(getattr(seg, "no_speech_prob", 0.0) or 0.0)

        count = max(1, len(out))
        duration = _measure(path)
        transcript = Transcript(
            text=" ".join(chunks).strip(),
            language=getattr(info, "language", "") or "",
            duration=duration,
            avg_logprob=avg_lp / count,
            no_speech_prob=nsp / count,
            segments=out,
            engine=f"faster-whisper/{self.model_name}/{self.device}/{self.compute_type}",
            latency_ms=int((_time.perf_counter() - started) * 1000),
        )
        log.info("heard: %s (%d ms, conf %.2f)", _clip(transcript.text, 120) or "<silence>",
                 transcript.latency_ms, transcript.confidence)
        return transcript

    def transcribe_bytes(self, payload: bytes, src_ext: str = "webm", language: Optional[str] = None) -> Transcript:
        path, err = decode_to_wav(payload, src_ext)
        if path is None:
            return Transcript(text="", engine=f"decode-error:{err}")
        try:
            return self.transcribe_wav(path, language=language)
        finally:
            path.unlink(missing_ok=True)

    def transcribe_pcm(self, pcm: bytes, sample_rate: int = TARGET_SR) -> Transcript:
        path = pcm16_to_wav(pcm, sample_rate)
        try:
            return self.transcribe_wav(path)
        finally:
            path.unlink(missing_ok=True)

    # -- optional local microphone (only if `sounddevice` happens to exist) --
    def record_from_mic(self, seconds: float = 5.0) -> Transcript:
        try:
            import sounddevice as sd  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Local microphone capture needs `pip install sounddevice numpy`; "
                "otherwise press the HUD mic button and the browser streams audio to me."
            ) from exc
        frames = sd.rec(int(seconds * TARGET_SR), samplerate=TARGET_SR, channels=1, dtype="int16")
        sd.wait()
        return self.transcribe_pcm(frames.tobytes(), TARGET_SR)


def _clip(text: str, limit: int = 160) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _measure(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / float(handle.getframerate() or 1)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

class TextToSpeech:
    """Coqui XTTS-v2 voice cloning with a Windows SAPI5 safety net."""

    XTTS_MODEL_NAMES = (
        "tts/tts_models--multilingual--multi-dataset--xtts_v2.v1",
        "tts_models/multilingual/multi-dataset/xtts_v2",
    )

    def __init__(self) -> None:
        self._tts = None
        self._lock = threading.Lock()
        self._state = "unloaded"
        self._error = ""
        self.backend = ""
        self.reference = SETTINGS.reference_wav_path
        self.language = SETTINGS.tts_language or "en"
        self._queue: "List[Tuple[str, Optional[Callable[[Dict[str, Any]], None]]]]" = []
        self._worker: Optional[threading.Thread] = None
        self._wake = threading.Event()
        self._stop = threading.Event()

    # -- reference clip ----------------------------------------------------
    def check_reference(self) -> Tuple[bool, str]:
        ref = self.reference
        if not ref.is_file():
            return False, (
                f"voice reference missing: {ref}. Record 3-10 s of clean speech, "
                "export as 22.05 kHz mono WAV and save it there."
            )
        if ref.stat().st_size < 8000:
            return False, f"voice reference {ref.name} looks truncated (<8 kB)."
        return True, str(ref)

    # -- loading -----------------------------------------------------------
    def load(self, force: bool = False) -> bool:
        with self._lock:
            if self._tts is not None and not force:
                return True
            if not SETTINGS.tts_enabled:
                self._state, self._error = "disabled", "TTS_ENABLED=false in .env"
                return False
            ok, detail = self.check_reference()
            if not ok:
                first_time = self._state != "unavailable"
                self._state, self._error = "unavailable", detail
                if first_time:      # once per state change, not once per sentence
                    log.warning("XTTS offline: %s", detail)
                return False
            if not self._coqui_available():
                first_time = self._state != "unavailable"
                self._state, self._error = "unavailable", "Coqui TTS package not importable -- using SAPI/native voice"
                if first_time:
                    log.warning(self._error)
                return False

            self._state = "loading"
            last = ""
            for name in self.XTTS_MODEL_NAMES:
                try:
                    self._tts = self._instantiate(name)
                    self.backend = f"XTTSv2/{name.split('/')[-1]}"
                    self._state = "ready"
                    log.info("XTTS-v2 ready on %s (low_vram=%s)", SETTINGS.tts_device, SETTINGS.tts_low_vram)
                    return True
                except Exception as exc:
                    last = f"{type(exc).__name__}: {exc}"
                    log.warning("TTS model %s failed: %s", name, _clip(last, 200))
            self._state, self._error = "failed", last
            return False

    @staticmethod
    def _coqui_available() -> bool:
        """Cheap importlib probe: no heavy import until we actually synthesise."""
        import importlib.util

        try:
            return importlib.util.find_spec("TTS") is not None
        except (ImportError, ValueError):
            return False

    def _instantiate(self, model_name: str) -> Any:
        """Prefer a fully local checkpoint dir, else resolve through the Coqui registry."""
        from TTS.api import TTS as CoquiTTS  # type: ignore

        local = Path(SETTINGS.tts_model_dir).expanduser() if SETTINGS.tts_model_dir else None
        if local:
            config_json = local / "config.json"
            checkpoint = next(iter(sorted(local.glob("model*.pth")) + sorted(local.glob("*.ckpt"))), None)
            if config_json.is_file() and checkpoint:
                log.info("loading XTTS from %s", local)
                return CoquiTTS.load_tts_model(
                    model_name=model_name,
                    config_path=str(config_json),
                    checkpoint_path=str(checkpoint),
                    progress_bar=False,
                    gpu=SETTINGS.tts_device == "cuda",
                )

        try:
            tts = CoquiTTS(model_name=model_name, progress_bar=False, low_vram=SETTINGS.tts_low_vram)
        except TypeError:
            tts = CoquiTTS(model_name=model_name, progress_bar=False)
        try:
            tts.to("cuda:0" if SETTINGS.tts_device == "cuda" else "cpu")
        except (AttributeError, RuntimeError) as exc:
            log.warning("moving TTS to %s failed (%s); staying on CPU", SETTINGS.tts_device, exc)
        return tts

    # -- synthesis ---------------------------------------------------------
    @property
    def status(self) -> Dict[str, Any]:
        ok_ref, ref_detail = self.check_reference()
        return {
            "state": self._state,
            "backend": self.backend or "pending",
            "language": self.language,
            "reference": ref_detail,
            "reference_found": ok_ref,
            "error": self._error,
            "device": SETTINGS.tts_device,
            "low_vram": SETTINGS.tts_low_vram,
        }

    def synth(self, text: str, out_path: Optional[Path] = None, language: Optional[str] = None) -> Dict[str, Any]:
        """Return ``{ok, path, engine, bytes, error}``. Never raises."""
        text = " ".join((text or "").split())
        if not text:
            return {"ok": False, "error": "nothing to say"}
        out_path = out_path or Path(tempfile.gettempdir()) / f"jarvis_say_{os.getpid()}_{threading.get_ident()}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink(missing_ok=True)

        script = self._prepare_script(text)
        if self._tts is None and not self.load():
            return self._fallback_synth(script, out_path)

        try:
            kwargs: Dict[str, Any] = {
                "text": script,
                "speaker_wav": str(self.reference),
                "language": (language or self.language)[:2],
                "file_path": str(out_path),
            }
            try:
                self._tts.tts_to_file(speed=SETTINGS.tts_speed, **kwargs)
            except TypeError:  # `speed` was added in TTS 0.21
                self._tts.tts_to_file(**kwargs)
            if not out_path.is_file():
                raise RuntimeError("coqui produced no audio file")
            # Offload GPU blocks so Whisper keeps its VRAM headroom.
            self._release_vram()
            return {
                "ok": True,
                "path": str(out_path),
                "engine": self.backend,
                "bytes": out_path.stat().st_size,
                "seconds": _measure(out_path),
            }
        except Exception as exc:
            log.error("XTTS synthesis failed: %s", exc)
            self._tts = None
            self._error = f"synthesis error: {exc}"
            return self._fallback_synth(script, out_path)

    @staticmethod
    def _prepare_script(text: str) -> str:
        """Make raw markdown/emoji speakable -- TTS models mangle punctuation soup."""
        import re

        script = re.sub(r"```.*?```", " code block omitted ", text, flags=re.S)
        script = re.sub(r"https?://\S+", " link ", script)
        script = re.sub(r"[*_#`>\[\]]", " ", script)
        script = script.replace("|", " ")
        script = re.sub(r"\b([Tt])B\b", " terabytes", script)
        script = re.sub(r"\b([Mm])B\b", " megabytes", script)
        script = re.sub(r"(\d)\.(\d)", r"\1 point \2", script)
        script = re.sub(r"(\d+)\s?%", r"\1 percent", script)
        script = re.sub(r"\s+", " ", script).strip()
        # XTTS chunks internally, but keep the sentence under its attention span.
        if len(script) > 420:
            sentences, buf = [], ""
            for part in re.split(r"(?<=[.!?])\s+", script):
                if len(buf) + len(part) > 380:
                    sentences.append(buf.strip())
                    buf = ""
                buf += part + " "
            if buf.strip():
                sentences.append(buf.strip())
            script = ". ".join(s.rstrip(".") for s in sentences if s) + "."
        return script[:2200]

    def _release_vram(self) -> None:
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # -- OS-native fallback voice -----------------------------------------
    def _fallback_synth(self, text: str, out_path: Path) -> Dict[str, Any]:
        if config.is_windows() and self._windows_sapi(text, out_path):
            return {"ok": out_path.is_file(), "path": str(out_path), "engine": "windows-sapi5",
                    "bytes": out_path.stat().st_size if out_path.is_file() else 0,
                    "note": self._error or "XTTS unavailable, used the built-in Windows voice"}
        # POSIX dev box: still write a WAV so the HUD's audio element works.
        self._write_beep_wav(out_path, text)
        return {"ok": out_path.is_file(), "path": str(out_path), "engine": "tone-fallback",
                "bytes": out_path.stat().st_size if out_path.is_file() else 0}

    def _windows_sapi(self, text: str, out_path: Path) -> bool:
        """Render with the built-in Windows SAPI5 voice (offline, ~0 MB VRAM).

        ``System.Speech`` writes a real 22 kHz WAV, so the HUD's <audio> element
        and ``play_wav`` keep working exactly as they do for XTTS output.
        """
        escaped = " ".join((text or "").split()).replace("`", "").replace('"', '\"').replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech;"
            "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            "$pref = $synth.GetInstalledVoices() | Where-Object { $_.Enabled } |"
            "  Sort-Object { $_.VoiceInfo.Culture.Name -notlike 'en*' } | Select-Object -First 1;"
            "if ($pref) { $synth.SelectVoice($pref.VoiceInfo.Name) };"
            "$synth.Rate = 1; $synth.Volume = 100;"
            f"$synth.SetOutputToWaveFile('{out_path}');"
            f"$synth.Speak('{escaped}');"
            "$synth.Dispose()"
        )
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],
                capture_output=True, text=True, timeout=90,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.warning("SAPI fallback failed: %s", exc)
            return False
        return proc.returncode == 0 and out_path.is_file() and out_path.stat().st_size > 1024

    @staticmethod
    def _write_beep_wav(path: Path, text: str, sr: int = 22050) -> None:
        """3 short tones at the cadence of the sentence: keeps HUD playback testable."""
        syllables = max(2, min(12, len(text.split())))
        frames = bytearray()
        for idx in range(syllables):
            freq = 320 + (idx % 4) * 60
            tone = int(sr * 0.09)
            for i in range(tone):
                env = math.sin(math.pi * i / tone)
                frames += struct.pack("<h", int(9000 * env * math.sin(2 * math.pi * freq * i / sr)))
            frames += b"\x00\x00" * int(sr * 0.03)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sr)
            handle.writeframes(bytes(frames))

    # -- playback & queue ---------------------------------------------------
    def speak(self, text: str, play: bool = True, language: Optional[str] = None) -> Dict[str, Any]:
        result = self.synth(text, language=language)
        if result.get("ok") and play and SETTINGS.tts_playback:
            result["played"] = play_wav(Path(result["path"]))
        elif not result.get("ok"):
            log.warning("TTS failed: %s", result.get("error"))
        return result

    def enqueue(self, text: str, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._drain, name="jarvis-tts", daemon=True)
            self._worker.start()
        self._queue.append((text, callback))
        self._wake.set()

    def _drain(self) -> None:
        while not self._stop.is_set():
            if not self._queue:
                self._wake.wait(0.25)
                self._wake.clear()
                continue
            text, cb = self._queue.pop(0)
            try:
                res = self.speak(text)
            except Exception as exc:  # noqa: BLE001
                res = {"ok": False, "error": str(exc)}
            if cb:
                try:
                    cb(res)
                except Exception:  # noqa: BLE001
                    log.debug("tts callback failed", exc_info=True)

    def warmup(self) -> None:
        """Pre-touch both models so the first real command is not slow."""
        threading.Thread(target=self.load, name="jarvis-tts-load", daemon=True).start()


def play_wav(path: Path) -> bool:
    """Blocking playback that picks the best available sink. Runs in a worker thread."""
    if not path.is_file():
        return False
    try:
        import winsound  # type: ignore

        winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        return True
    except Exception:
        pass
    for player in ("ffplay", "paplay", "aplay", "afplay"):
        exe = shutil.which(player)
        if exe:
            cmd = [exe, "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)] if player == "ffplay" else [exe, str(path)]
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
                return True
            except OSError:
                continue
    return False


# ---------------------------------------------------------------------------
# Facade used by server.py / main.py / discord_bridge.py
# ---------------------------------------------------------------------------

class VoiceEngine:
    """One object that owns STT + TTS lifecycle, warm-up and status reporting."""

    def __init__(self) -> None:
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self._warmed = False

    def listen(self, payload: bytes, src_ext: str = "webm") -> Transcript:
        return self.stt.transcribe_bytes(payload, src_ext)

    def listen_file(self, path: Path) -> Transcript:
        return self.stt.transcribe_wav(path)

    def listen_pcm(self, pcm: bytes, sample_rate: int = TARGET_SR) -> Transcript:
        return self.stt.transcribe_pcm(pcm, sample_rate)

    def speak(self, text: str, play: bool = True) -> Dict[str, Any]:
        return self.tts.speak(text, play=play)

    def speak_async(self, text: str, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        self.tts.enqueue(text, callback)

    def warmup(self, blocking_stt: bool = False) -> Dict[str, str]:
        if self._warmed:
            return {"stt": self.stt.status["state"], "tts": self.tts.status["state"]}
        self._warmed = True
        if blocking_stt:
            self.stt.load()
        else:
            threading.Thread(target=self.stt.load, name="jarvis-stt-load", daemon=True).start()
        self.tts.warmup()
        return {"stt": "loading" if not blocking_stt else self.stt.status["state"], "tts": self.tts.status["state"]}

    def status(self) -> Dict[str, Any]:
        return {
            "stt": self.stt.status,
            "tts": self.tts.status,
            "ffmpeg": bool(_ffmpeg_bin()),
            "max_chunk_seconds": MAX_CHUNK_SECONDS,
        }


ENGINE = VoiceEngine()


def get_engine() -> VoiceEngine:
    return ENGINE


__all__ = ["ENGINE", "VoiceEngine", "SpeechToText", "Transcript", "play_wav", "decode_to_wav", "get_engine"]
