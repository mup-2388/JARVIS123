"""
config.py -- single source of truth for every tunable in PROJECT J.A.R.V.I.S.

Design notes
------------
* Zero third-party dependencies: ``.env`` parsing is implemented here with the
  stdlib so the assistant boots even before ``pip install -r requirements.txt``
  has finished.
* Every module imports this file with an *absolute* import (``import config``)
  and never touches ``os.environ`` directly, so overrides stay consistent
  between the FastAPI process, the Discord thread and the voice workers.
* Paths are absolute (derived from ``ROOT``) because the HUD is loaded both
  from ``file://`` (pywebview) and from ``http://host:port`` (uvicorn).
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable

# --------------------------------------------------------------------------- paths
ROOT: Path = Path(__file__).resolve().parent
STATIC_DIR: Path = ROOT / "static"
INDEX_HTML: Path = STATIC_DIR / "index.html"
NOTES_DIR_DEFAULT: Path = ROOT / "notes"   # created on first launch if absent
ASSETS_DIR: Path = ROOT / "assets"
CACHE_DIR: Path = ROOT / "data"
ENV_FILE: Path = ROOT / ".env"

for _d in (STATIC_DIR, ASSETS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- .env
def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    # Drop trailing inline comments for unquoted values.
    if " #" in value:
        value = value.split(" #", 1)[0]
    return value.strip()


def parse_env_file(path: Path) -> Dict[str, str]:
    """Parse a dotenv-style file. Tolerates comments, ``export`` and BOM."""
    out: Dict[str, str] = {}
    if not path.is_file():
        return out
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        out[key.upper()] = _strip_quotes(value.strip())
    return out


_ENV_CACHE: Dict[str, str] = parse_env_file(ENV_FILE)


def dotenv(key: str, default: str = "") -> str:
    """``os.environ`` first, then ``.env``, then ``default``."""
    live = os.environ.get(key)
    if live is not None and live != "":
        return live
    cached = _ENV_CACHE.get(key.upper())
    if cached is not None and cached != "":
        return cached
    return default


def _coerce_int(raw: str, default: int) -> int:
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def _coerce_float(raw: str, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _coerce_bool(raw: str, default: bool) -> bool:
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve(raw: str) -> Path:
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (ROOT / p)


# --------------------------------------------------------------------------- settings
@dataclass(frozen=True)
class Settings:
    # LLM / agent track
    hf_token: str = field(default_factory=lambda: dotenv("HF_TOKEN"))
    hf_model: str = field(default_factory=lambda: dotenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"))
    hf_provider: str = field(default_factory=lambda: dotenv("HF_PROVIDER", "auto"))
    hf_timeout: int = field(default_factory=lambda: _coerce_int(dotenv("HF_TIMEOUT_SECONDS"), 45))
    hf_max_retries: int = field(default_factory=lambda: _coerce_int(dotenv("HF_MAX_RETRIES"), 2))
    hf_max_tokens: int = field(default_factory=lambda: _coerce_int(dotenv("HF_MAX_TOKENS"), 700))
    hf_temperature: float = field(default_factory=lambda: _coerce_float(dotenv("HF_TEMPERATURE"), 0.2))
    allow_offline_agent: bool = field(default_factory=lambda: _coerce_bool(dotenv("ALLOW_OFFLINE_AGENT"), False))

    instant_track_enabled: bool = field(default_factory=lambda: _coerce_bool(dotenv("INSTANT_TRACK_ENABLED"), True))
    agent_min_chars: int = field(default_factory=lambda: _coerce_int(dotenv("AGENT_MIN_CHARS"), 3))

    # STT
    whisper_model: str = field(default_factory=lambda: dotenv("WHISPER_MODEL", "base"))
    whisper_device: str = field(default_factory=lambda: dotenv("WHISPER_DEVICE", "cuda"))
    whisper_compute_type: str = field(default_factory=lambda: dotenv("WHISPER_COMPUTE_TYPE", "int8"))
    whisper_num_workers: int = field(default_factory=lambda: _coerce_int(dotenv("WHISPER_NUM_WORKERS"), 2))
    whisper_cpu_threads: int = field(default_factory=lambda: _coerce_int(dotenv("WHISPER_CPU_THREADS"), 4))
    whisper_beam_size: int = field(default_factory=lambda: _coerce_int(dotenv("WHISPER_BEAM_SIZE"), 1))
    whisper_language: str = field(default_factory=lambda: dotenv("WHISPER_LANGUAGE"))
    whisper_vad_filter: bool = field(default_factory=lambda: _coerce_bool(dotenv("WHISPER_VAD_FILTER"), True))
    whisper_allow_cpu_fallback: bool = field(default_factory=lambda: _coerce_bool(dotenv("WHISPER_ALLOW_CPU_FALLBACK"), True))

    # TTS
    tts_enabled: bool = field(default_factory=lambda: _coerce_bool(dotenv("TTS_ENABLED"), True))
    tts_language: str = field(default_factory=lambda: dotenv("TTS_LANGUAGE", "en"))
    tts_speed: float = field(default_factory=lambda: _coerce_float(dotenv("TTS_SPEED"), 1.0))
    tts_device: str = field(default_factory=lambda: dotenv("TTS_DEVICE", "cuda"))
    tts_low_vram: bool = field(default_factory=lambda: _coerce_bool(dotenv("TTS_LOW_VRAM"), True))
    tts_reference_wav: str = field(default_factory=lambda: dotenv("TTS_REFERENCE_WAV", "assets/jarvis_sample.wav"))
    tts_model_dir: str = field(default_factory=lambda: dotenv("TTS_MODEL_DIR"))
    tts_sample_rate: int = field(default_factory=lambda: _coerce_int(dotenv("TTS_SAMPLE_RATE"), 24000))
    tts_playback: bool = field(default_factory=lambda: _coerce_bool(dotenv("TTS_PLAYBACK"), True))

    # Discord
    discord_token: str = field(default_factory=lambda: dotenv("DISCORD_TOKEN"))
    discord_channel_id: int = field(default_factory=lambda: _coerce_int(dotenv("DISCORD_CHANNEL_ID"), 0))
    discord_prefix: str = field(default_factory=lambda: dotenv("DISCORD_COMMAND_PREFIX", "/jarvis"))
    discord_use_threads: bool = field(default_factory=lambda: _coerce_bool(dotenv("DISCORD_USE_THREADS"), True))
    discord_enabled: bool = field(default_factory=lambda: _coerce_bool(dotenv("DISCORD_ENABLED"), True))

    # Sports
    api_sports_key: str = field(default_factory=lambda: dotenv("API_SPORTS_KEY"))
    api_sports_base: str = field(default_factory=lambda: dotenv("API_SPORTS_BASE", "https://v3.football.api-sports.io"))
    api_sports_timeout: int = field(default_factory=lambda: _coerce_int(dotenv("API_SPORTS_TIMEOUT"), 12))
    api_sports_default_team: str = field(default_factory=lambda: dotenv("API_SPORTS_DEFAULT_TEAM", "Real Madrid"))

    # OS tools
    notes_dir: str = field(default_factory=lambda: dotenv("NOTES_DIR", "notes"))
    launch_timeout: int = field(default_factory=lambda: _coerce_int(dotenv("LAUNCH_TIMEOUT"), 6))
    custom_apps: str = field(default_factory=lambda: dotenv("CUSTOM_APPS"))
    volume_step: int = field(default_factory=lambda: _coerce_int(dotenv("VOLUME_STEP"), 8))

    # Server
    host: str = field(default_factory=lambda: dotenv("JARVIS_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _coerce_int(dotenv("JARVIS_PORT"), 8760))
    telemetry_ms: int = field(default_factory=lambda: _coerce_int(dotenv("JARVIS_WS_TELEMETRY_MS"), 800))
    cors_origins: str = field(default_factory=lambda: dotenv("JARVIS_CORS_ALLOW_ORIGINS", "*"))
    history_size: int = field(default_factory=lambda: _coerce_int(dotenv("JARVIS_HISTORY_SIZE"), 12))
    speak_replies: bool = field(default_factory=lambda: _coerce_bool(dotenv("SPEAK_REPLIES"), True))
    debug: bool = field(default_factory=lambda: _coerce_bool(dotenv("JARVIS_DEBUG"), True))

    # Derived absolute paths -------------------------------------------------
    @property
    def notes_path(self) -> Path:
        return _resolve(self.notes_dir)

    @property
    def reference_wav_path(self) -> Path:
        return _resolve(self.tts_reference_wav)

    @property
    def cache_path(self) -> Path:
        return CACHE_DIR

    @property
    def llm_ready(self) -> bool:
        return bool(self.hf_token)

    def redacted(self) -> Dict[str, Any]:
        """Safe view for the HUD / Discord ``/status`` style replies."""
        return {
            "hf_model": self.hf_model,
            "hf_token_set": bool(self.hf_token),
            "whisper": f"{self.whisper_model}/{self.whisper_device}/{self.whisper_compute_type}",
            "tts_enabled": self.tts_enabled,
            "tts_reference": str(self.reference_wav_path),
            "tts_reference_found": self.reference_wav_path.is_file(),
            "discord_enabled": self.discord_enabled,
            "discord_token_set": bool(self.discord_token),
            "discord_channel_id": self.discord_channel_id or None,
            "api_sports_key_set": bool(self.api_sports_key),
            "notes_dir": str(self.notes_path),
            "server": f"{self.host}:{self.port}",
            "speak_replies": self.speak_replies,
            "debug": self.debug,
        }

    def as_env_dict(self) -> Dict[str, str]:
        merged = dict(_ENV_CACHE)
        merged.update({k: v for k, v in os.environ.items() if k.isupper() and "." not in k})
        return merged


SETTINGS = Settings()


# --------------------------------------------------------------------------- logging
_LOG_READY = False


def get_logger(name: str) -> logging.Logger:
    """Consistent, colour-free logs across every JARVIS subsystem."""
    global _LOG_READY
    if not _LOG_READY:
        level = logging.DEBUG if SETTINGS.debug else logging.INFO
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)-5s] %(name)-14s %(message)s", datefmt="%H:%M:%S")
        )
        root = logging.getLogger("jarvis")
        root.setLevel(level)
        if not root.handlers:
            root.addHandler(handler)
        root.propagate = False
        _LOG_READY = True
    return logging.getLogger(f"jarvis.{name}")


def is_windows() -> bool:
    return sys.platform.startswith("win")


def is_headless() -> bool:
    """True when there is no display for pywebview (CI / containers / WSL)."""
    if is_windows():
        return False
    if os.environ.get("CI") or os.environ.get("JARVIS_HEADLESS"):
        return True
    return not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")


def ensure_importable() -> None:
    """Put the project root on ``sys.path`` so absolute imports always resolve.

    Needed because the HUD can be launched from an arbitrary working directory
    (``python main.py`` from Desktop, a scheduled task, a .bat with `cd /D`).
    """
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def iter_env_keys(prefixes: Iterable[str]) -> list[str]:
    keys = set()
    for key in SETTINGS.as_env_dict():
        if any(key.startswith(p) for p in prefixes):
            keys.add(key)
    return sorted(keys)


__all__ = [
    "SETTINGS",
    "Settings",
    "dotenv",
    "get_logger",
    "ensure_importable",
    "is_windows",
    "is_headless",
    "parse_env_file",
    "ROOT",
    "STATIC_DIR",
    "INDEX_HTML",
    "ASSETS_DIR",
    "CACHE_DIR",
    "NOTES_DIR_DEFAULT",
]
