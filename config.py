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
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

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
    # LLM providers (Track 2 brain).  Keys live in .env; every provider below
    # has a real free tier, and llm_providers.py rotates between them so one
    # vendor's quota can never silence JARVIS.
    groq_api_key: str = field(default_factory=lambda: dotenv("GROQ_API_KEY"))
    cerebras_api_key: str = field(default_factory=lambda: dotenv("CEREBRAS_API_KEY"))
    cloudflare_api_token: str = field(default_factory=lambda: dotenv("CLOUDFLARE_API_TOKEN"))
    cloudflare_account_id: str = field(default_factory=lambda: dotenv("CLOUDFLARE_ACCOUNT_ID"))
    gemini_api_key: str = field(default_factory=lambda: dotenv("GEMINI_API_KEY"))
    mistral_api_key: str = field(default_factory=lambda: dotenv("MISTRAL_API_KEY"))
    openrouter_api_key: str = field(default_factory=lambda: dotenv("OPENROUTER_API_KEY"))
    github_token: str = field(default_factory=lambda: dotenv("GITHUB_TOKEN"))

    llm_provider: str = field(default_factory=lambda: dotenv("LLM_PROVIDER", "auto"))
    llm_order: str = field(default_factory=lambda: dotenv(
        "LLM_PROVIDER_ORDER", "groq,cerebras,cloudflare,gemini,mistral,openrouter,github"))
    llm_timeout: int = field(default_factory=lambda: _coerce_int(dotenv("LLM_TIMEOUT_SECONDS"), 30))
    llm_max_tokens: int = field(default_factory=lambda: _coerce_int(dotenv("LLM_MAX_TOKENS"), 700))
    llm_temperature: float = field(default_factory=lambda: _coerce_float(dotenv("LLM_TEMPERATURE"), 0.2))
    #: How long a provider that just returned 429 / an auth error is dropped from
    #: rotation.  Shorter windows (RPM throttles, Retry-After hints) are honoured
    #: automatically; this is the ceiling for "come back tomorrow" quotas.
    llm_cooldown_hours: int = field(default_factory=lambda: _coerce_int(dotenv("LLM_COOLDOWN_HOURS"), 24))
    llm_tier_mode: str = field(default_factory=lambda: dotenv("LLM_TIER_MODE", "auto"))   # auto|fast|smart
    llm_state_file: str = field(default_factory=lambda: dotenv("LLM_STATE_FILE", "data/llm_state.json"))
    #: When no provider can answer, may the offline planner use web_search for
    #: fact-shaped questions?  True keeps JARVIS useful; false makes it say "no AI".
    llm_fallback_search: bool = field(default_factory=lambda: _coerce_bool(dotenv("LLM_FALLBACK_SEARCH"), True))
    llm_probe_timeout: int = field(default_factory=lambda: _coerce_int(dotenv("LLM_PROBE_TIMEOUT_SECONDS"), 12))
    # Ask each provider's /models endpoint which ids the key can see, then use those.
    # Free tiers rename and retire models constantly; this is what stops a retired id
    # such as "llama-3.3-70b-versatile" costing the user every answer.
    llm_auto_discover: bool = field(default_factory=lambda: _coerce_bool(dotenv("LLM_AUTO_DISCOVER"), True))
    # -- desktop control (apps, files, screen, background listening) -------------
    #: How long the Start-Menu / Get-StartApps / registry app indexes stay cached.  Long
    #: enough that JARVIS never blocks on PowerShell mid-sentence, short enough that an app
    #: installed today is findable in a few minutes.
    app_index_ttl: int = field(default_factory=lambda: _coerce_int(dotenv("APP_INDEX_TTL"), 900))
    app_index_file: str = field(default_factory=lambda: dotenv("APP_INDEX_FILE", "data/app_index.json"))
    #: "open X" waits this long for the window to appear, so JARVIS can say "Settings is
    #: opening" instead of "done" when nothing actually happened.
    app_wait_seconds: int = field(default_factory=lambda: _coerce_int(dotenv("APP_WAIT_SECONDS"), 4))
    #: Everything file-related is confined to these roots (colon/semicolon separated).
    #: "" means %USERPROFILE%\\Documents\\JARVIS, which JARVIS creates on first use.
    files_root: str = field(default_factory=lambda: dotenv("FILES_ROOT", ""))
    #: Extra folders the user grants for reads/writes, e.g. "C:\\Users\\me\\Desktop;D:\\notes".
    files_allowed: str = field(default_factory=lambda: dotenv("FILES_ALLOWED", ""))
    #: Delete = recycle bin always; "never" refuses permanent deletes outright.
    file_delete_policy: str = field(default_factory=lambda: dotenv("FILE_DELETE_POLICY", "recycle"))
    #: Screen reading: local OCR first (free, offline), then an optional vision model.
    screen_ocr_enabled: bool = field(default_factory=lambda: _coerce_bool(dotenv("SCREEN_OCR"), True))
    screen_vision_enabled: bool = field(default_factory=lambda: _coerce_bool(dotenv("SCREEN_VISION"), True))
    #: Background always-on listening: wake words, push-to-talk key and the floating bar.
    wake_enabled: bool = field(default_factory=lambda: _coerce_bool(dotenv("WAKE_WORD_ENABLED"), True))
    wake_words: str = field(default_factory=lambda: dotenv("WAKE_WORDS", "jarvis,jervis,jarvi,yarves"))
    wake_fuzzy: bool = field(default_factory=lambda: _coerce_bool(dotenv("WAKE_FUZZY"), True))
    wake_followup_seconds: int = field(default_factory=lambda: _coerce_int(dotenv("WAKE_FOLLOWUP_SECONDS"), 12))
    wake_command_key: str = field(default_factory=lambda: dotenv("WAKE_HOTKEY", "f12"))
    wake_push_to_talk: str = field(default_factory=lambda: dotenv("WAKE_PTT_KEY", "rcontrol"))
    bar_enabled: bool = field(default_factory=lambda: _coerce_bool(dotenv("OVERLAY_BAR"), True))
    bar_width: int = field(default_factory=lambda: _coerce_int(dotenv("OVERLAY_WIDTH"), 720))
    bar_height: int = field(default_factory=lambda: _coerce_int(dotenv("OVERLAY_HEIGHT"), 96))
    #: Voice reminders / timers ("remind me in 10 minutes to stretch").
    reminders_enabled: bool = field(default_factory=lambda: _coerce_bool(dotenv("REMINDERS_ENABLED"), True))
    reminders_file: str = field(default_factory=lambda: dotenv("REMINDERS_FILE", "data/reminders.json"))
    #: Hard wall-clock ceiling for one "ask the AI" turn across *all* providers.
    #: Better a heuristic answer in 25 s than a frozen mic for 3 minutes.
    llm_budget_seconds: int = field(default_factory=lambda: _coerce_int(dotenv("LLM_BUDGET_SECONDS"), 25))
    #: After two providers fail on the network (no egress / DNS / captive portal),
    #: rest the whole pool this long instead of re-probing every turn.
    llm_offline_cooldown: int = field(default_factory=lambda: _coerce_int(dotenv("LLM_OFFLINE_COOLDOWN_SECONDS"), 180))

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
    def llm_state_path(self) -> Path:
        return _resolve(self.llm_state_file)

    @property
    def llm_keys_set(self) -> List[str]:
        """Provider names this machine has credentials for (in rotation order)."""
        pairs = (
            ("groq", self.groq_api_key, ""),
            ("cerebras", self.cerebras_api_key, ""),
            ("cloudflare", self.cloudflare_api_token, self.cloudflare_account_id),
            ("gemini", self.gemini_api_key, ""),
            ("mistral", self.mistral_api_key, ""),
            ("openrouter", self.openrouter_api_key, ""),
            ("github", self.github_token, ""),
        )
        wanted = [n.strip().lower() for n in re.split(r"[,\s]+", self.llm_order) if n.strip()] or [p[0] for p in pairs]
        out = []
        for name in wanted:
            for key, token, extra in pairs:
                if key == name and token and (not extra or extra):
                    out.append(key)
        return out

    def redacted(self) -> Dict[str, Any]:
        """Safe view for the HUD / Discord ``/status`` style replies."""
        return {
            "llm_provider": self.llm_provider,
            "llm_keys_set": self.llm_keys_set,
            "llm_order": self.llm_order,
            "llm_cooldown_hours": self.llm_cooldown_hours,
            "llm_tier_mode": self.llm_tier_mode,
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
