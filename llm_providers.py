"""
llm_providers.py -- the AI brain: many free OpenAI-compatible providers, with
quota-aware failover and cost-aware model selection.

Why this exists
---------------
The original build talked to Hugging Face Inference only.  That made JARVIS
either great or mute: one router hiccup and every utterance fell through to the
heuristic planner, which turned unmatched questions into DuckDuckGo searches.
This module replaces that with a pool:

* **Groq**, **Cerebras**, **Cloudflare Workers AI**, **Google (Gemini)**,
  **Mistral**, **OpenRouter** and **GitHub Models** are wired up.  All of them
  have a real free tier and (except GitHub/Cloudflare native) speak the OpenAI
  ``/chat/completions`` schema, so one request builder covers seven vendors.
* **Quota-aware circuit breaking.**  A 429 removes that provider from rotation
  for the length of *its own* window: the ``Retry-After`` header when present,
  a per-minute window when the error is an RPM/TPM throttle, otherwise until the
  provider's documented daily reset (UTC or US/Pacific), capped by
  ``LLM_COOLDOWN_HOURS`` (24 h by default).  Cooldowns are persisted to
  ``data/llm_state.json`` so a restart does not re-probe a dead key.
* **Right-sized models.**  ``choose_tier()`` scores the utterance: greetings,
  one-line facts and tool dispatch go to the cheap/fast model (Llama 3.1 8B,
  Gemini Flash-Lite...), while "compare", "plan", "summarise my notes",
  multi-part or long requests go to the heavy model (Llama 3.3 70B,
  GPT-OSS-120B, Gemini Flash).  Free tiers are metered per token and per day, so
  this is what keeps JARVIS inside a free quota all day.

* **Your own endpoint too.**  ``CUSTOM_LLM_BASE_URL`` opens an eighth slot for any
  service that speaks the same schema (NVIDIA NIM, SambaNova, Fireworks, Azure) or a
  local Ollama / LM Studio server, key optional, native tool calls optional.

Nothing here imports a third-party package: plain ``urllib`` with a short
timeout, because a voice assistant may not block on a hung socket.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import dataclasses
from dataclasses import dataclass, field


def dataclass_replace(instance: Any, **changes: Any) -> Any:
    """``dataclasses.replace`` with a name that reads better at the call site."""
    return dataclasses.replace(instance, **changes)
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import config
from config import SETTINGS, get_logger

log = get_logger("llm")

#: Everything the router needs to know about a failed round trip.
MINUTE_WINDOW_SECONDS = 65          # RPM / TPM throttles
MODEL_NOT_FOUND_SECONDS = 6 * 3600
DISCOVER_TTL_SECONDS = 6 * 3600        # how long a /models answer is trusted
# Models that exist on these endpoints but cannot carry a conversation.
_MODEL_BANNED = re.compile(
    r"embed|embedding|whisper|tts|audio|image|vision|-vl|clip|dall|sora|guard|safety"
    r"|moderation|rerank|transcri|realtime|live|computer[-_ ]?use|translate|search-|rank"
    r"|playai|dolphin|mytho|assistant-|prompt-|-zh|il|text-bison", re.I)
_MODEL_CHAT = re.compile(
    r"instruct|chat|instant|versatile|model|llama|gemini|gpt|qwen|mistral|mixtral|deepseek"
    r"|phi|gemma|glm|kimi|magistral|devstral|scout|maverick|oss|command|nova|minimax", re.I)
_MODEL_FAST_NAME = re.compile(
    r"instant|flash[-_ ]?lite|\blite\b|\bmini\b|small|nano|tiny|turbo|(?<!\d)[2-9]b(?!\d)", re.I)
_MODEL_SMART_NAME = re.compile(
    r"(?<!\d)(2[7-9]|[3-9]\d|1\d\d)b(?!\d)|large|plus|flash|scout|maverick|\bpro\b|thinking|reasoner", re.I)
# A model with a smaller window than this cannot hold the system prompt + tool schemas +
# conversation memory, so it is never a useful pick for JARVIS even if the key can see it.
MIN_USEFUL_CONTEXT = 8000  # a retired model id will not come back within the hour
TRANSIENT_BASE_SECONDS = 120        # 5xx / timeouts, doubled per consecutive failure
TRANSIENT_CAP_SECONDS = 30 * 60


class LlmError(RuntimeError):
    """No provider could answer. ``attempts`` explains who tried what."""

    def __init__(self, message: str, attempts: Optional[List[Dict[str, Any]]] = None) -> None:
        super().__init__(message)
        self.attempts = attempts or []


class NoProviderConfigured(LlmError):
    """Nothing in ``.env`` has a key -- the caller should say so plainly."""


# ---------------------------------------------------------------------------
# Provider catalogue
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderSpec:
    """One vendor: how to reach it, which two models to use, how it resets."""

    key: str
    label: str
    base_url: str
    fast_model: str
    smart_model: str
    key_env: str
    reset: str = "daily-utc"          # daily-utc | daily-pacific | none
    style: str = "openai"             # openai | cloudflare (native /ai/run)
    account_env: str = ""             # set when the URL needs an account id
    free_quota: str = ""
    key_url: str = ""
    supports_tools: bool = True
    key_optional: bool = False          # local runtimes (Ollama, LM Studio) send no auth
    # Ids worth trying if the two above are refused or not visible to this key: providers
    # retire model names without telling anyone, and entitlements differ per account.
    fallback_models: Tuple[str, ...] = ()
    #: Ids that accept image input, for "look at my screen".  Discovery overrides these.
    vision_models: Tuple[str, ...] = ()
    extra_headers: Dict[str, str] = field(default_factory=dict)

    # -- credentials -------------------------------------------------------
    def api_key(self) -> str:
        """The configured key, accepting the usual alias spellings per vendor."""
        for name in (self.key_env, *_KEY_ALIASES.get(self.key, ())):
            value = config.dotenv(name)
            if value:
                return value
        return ""

    def account_id(self) -> str:
        return config.dotenv(self.account_env) if self.account_env else ""

    @property
    def ready(self) -> bool:
        if not self.api_key() and not self.key_optional:
            return False
        if self.account_env and not self.account_id():
            return False
        return True

    # -- urls --------------------------------------------------------------
    @property
    def root(self) -> str:
        base = self.base_url
        if "{account_id}" in base:
            base = base.replace("{account_id}", urllib.parse.quote(self.account_id(), safe=""))
        return base.rstrip("/")

    def endpoint(self, model: str) -> str:
        if self.style == "cloudflare":
            return f"{self.root}/ai/run/{urllib.parse.quote(model, safe='@/-')}"
        return f"{self.root}/chat/completions"

    def describe(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "models": {"fast": self.fast_model, "smart": self.smart_model},
            "reset": self.reset,
            "style": self.style,
            "quota": self.free_quota,
            "key_url": self.key_url,
            "key_env": self.key_env,
            "account_env": self.account_env,
            "supports_tools": self.supports_tools,
            "needs_account": bool(self.account_env),
            "key_set": bool(self.api_key()),
            "account_set": bool(self.account_id()) if self.account_env else True,
        }


#: Extra env names a vendor's key may live under.  Deliberately excludes
#: ``GH_TOKEN``/``GITHUB_TOKEN``: those belong to the gh CLI / CI runner, are scoped
#: to repo permissions, and spending them on LLM quota as a side effect would be
#: both surprising and abusive.  GitHub Models wants its own PAT (``models:read``).
_KEY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "gemini": ("GOOGLE_API_KEY",),
    "cloudflare": ("CLOUDFLARE_API_KEY",),
    "github": ("GITHUB_MODELS_TOKEN",),
    "openrouter": ("OPENROUTER_KEY",),
}

def _models(key: str, fast: str, smart: str) -> Tuple[str, str]:
    """``GROQ_MODEL_FAST`` / ``GROQ_MODEL_SMART`` override the catalogue defaults.

    Free-tier model lists change constantly (Cerebras retired two models in
    February 2026), so the .env override is the escape hatch that keeps a
    working install working without a code change.
    """
    return (config.dotenv(f"{key.upper()}_MODEL_FAST", fast) or fast,
            config.dotenv(f"{key.upper()}_MODEL_SMART", smart) or smart)


PROVIDERS: Dict[str, ProviderSpec] = {
    spec.key: dataclass_replace(spec, fast_model=_models(spec.key, spec.fast_model, spec.smart_model)[0],
                                smart_model=_models(spec.key, spec.fast_model, spec.smart_model)[1])
    for spec in (
        ProviderSpec(
            key="groq",
            label="Groq",
            base_url="https://api.groq.com/openai/v1",
            # A live /models list from a free key (2026-09) contained no Llama id at all:
            # the chat models are gpt-oss-20b/120b and qwen3.x-27b, plus audio, guard and
            # 4K-context models that are useless here. Llama stays as a fallback rung for
            # older accounts, and discovery overrides all of it per key anyway.
            fast_model="openai/gpt-oss-20b",
            smart_model="openai/gpt-oss-120b",
            vision_models=("qwen/qwen3.6-27b", "meta-llama/llama-4-scout-17b-16e-instruct"),
            fallback_models=("llama-3.1-8b-instant", "llama-3.3-70b-versatile",
                             "qwen/qwen3.6-27b", "groq/compound-mini"),
            key_env="GROQ_API_KEY",
            reset="daily-utc",
            free_quota="~30 req/min, 6K tokens/min; gpt-oss-20b/120b + qwen3.6/3.8-27b are the free chat models now",
            key_url="https://console.groq.com/keys",
        ),
        ProviderSpec(
            key="cerebras",
            label="Cerebras",
            base_url="https://api.cerebras.ai/v1",
            fast_model="llama3.1-8b",
            smart_model="gpt-oss-120b",
            key_env="CEREBRAS_API_KEY",
            reset="daily-utc",
            free_quota="~1M tokens/day shared, 30 req/min, 60K tokens/min (8K context on free)",
            key_url="https://cloud.cerebras.ai",
        ),
        ProviderSpec(
            key="cloudflare",
            label="Cloudflare Workers AI",
            base_url="https://api.cloudflare.com/client/v4/accounts/{account_id}",
            fast_model="@cf/meta/llama-3.1-8b-instruct",
            smart_model="@cf/openai/gpt-oss-120b",
            key_env="CLOUDFLARE_API_TOKEN",
            account_env="CLOUDFLARE_ACCOUNT_ID",
            reset="daily-utc",
            # Cloudflare's own /ai/run shape (answer at result.response) rather than
            # the newer /ai/v1 OpenAI shim: same body, but it is the endpoint the
            # dashboard docs and every Cloudflare token are known to accept.
            style="cloudflare",
            free_quota="10,000 Neurons/day (~200-1,000 short requests)",
            key_url="https://dash.cloudflare.com/?to=/:account/ai/workers-ai",
        ),
        ProviderSpec(
            key="gemini",
            label="Google Gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            fast_model="gemini-2.5-flash-lite",
            smart_model="gemini-2.5-flash",
            vision_models=("gemini-2.5-flash", "gemini-2.5-flash-lite"),
            key_env="GEMINI_API_KEY",
            reset="daily-pacific",
            free_quota="~1,000 req/day on Flash-Lite, 250 on Flash; 5-15 req/min",
            key_url="https://aistudio.google.com/apikey",
        ),
        ProviderSpec(
            key="mistral",
            label="Mistral",
            base_url="https://api.mistral.ai/v1",
            fast_model="mistral-small-latest",
            smart_model="magistral-small-latest",
            vision_models=("mistral-small-latest", "pixtral-128b-2409"),
            key_env="MISTRAL_API_KEY",
            reset="none",
            free_quota="free 'Experiment' tier, ~1B tokens/month, prompts may be logged for training",
            key_url="https://console.mistral.ai/api-keys",
        ),
        ProviderSpec(
            key="openrouter",
            label="OpenRouter",
            base_url="https://openrouter.ai/api/v1",
            fast_model="meta-llama/llama-3.1-8b-instruct:free",
            smart_model="openai/gpt-oss-120b:free",
            vision_models=("meta-llama/llama-4-scout-17b-16e-instruct:free",),
            key_env="OPENROUTER_API_KEY",
            reset="daily-utc",
            free_quota="50 req/day on ':free' models (1,000/day after a $10 top-up)",
            key_url="https://openrouter.ai/settings/keys",
            extra_headers={"HTTP-Referer": "http://127.0.0.1", "X-Title": "JARVIS"},
        ),
        ProviderSpec(
            key="github",
            label="GitHub Models",
            base_url="https://models.inference.ai.azure.com",
            fast_model="Meta-Llama-3.1-8B-Instruct",
            smart_model="gpt-4o",
            vision_models=("gpt-4o", "gpt-4o-mini"),
            key_env="GITHUB_MODELS_TOKEN",
            reset="none",
            free_quota="150-1,000 req/day per model; needs a PAT with the 'models:read' scope (not your gh/CI token)",
            key_url="https://github.com/settings/tokens",
        ),
    )
}


def ensure_custom() -> None:
    """Register the user-defined ``custom`` provider from ``CUSTOM_LLM_*`` (see .env.example).

    Free tiers appear and vanish monthly, so instead of hard-coding the next vendor the
    whole catalogue has one open slot: anything that speaks the OpenAI
    ``/chat/completions`` contract can be wired up from ``.env``. That covers NVIDIA NIM,
    SambaNova, Fireworks, an Azure deployment - and a local Ollama / LM Studio server for
    an offline demo, which needs no key at all.
    """
    base = (config.dotenv("CUSTOM_LLM_BASE_URL") or "").strip().rstrip("/")
    if not base:
        PROVIDERS.pop("custom", None)
        return
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]        # people paste the full URL
    fast = (config.dotenv("CUSTOM_LLM_MODEL_FAST") or "llama-3.1-8b-instruct").strip()
    smart = (config.dotenv("CUSTOM_LLM_MODEL_SMART") or fast).strip()
    tools_raw = (config.dotenv("CUSTOM_LLM_TOOLS") or "true").strip().lower()
    PROVIDERS["custom"] = ProviderSpec(
        key="custom",
        label=(config.dotenv("CUSTOM_LLM_NAME") or "Custom endpoint").strip(),
        base_url=base,
        fast_model=fast,
        smart_model=smart,
        key_env="CUSTOM_LLM_API_KEY",
        reset=(config.dotenv("CUSTOM_LLM_RESET") or "none").strip(),
        style="openai",
        free_quota=config.dotenv("CUSTOM_LLM_QUOTA", "your endpoint, your limits"),
        supports_tools=tools_raw not in ("0", "false", "no", "off"),
        key_optional=True,
    )


def order() -> List[str]:
    """Provider rotation, filtered by ``LLM_PROVIDER_ORDER`` and forced by ``LLM_PROVIDER``."""
    default = "groq,cerebras,cloudflare,gemini,mistral,openrouter,github"
    ensure_custom()
    raw = SETTINGS.llm_order or default
    names = [n.strip().lower() for n in re.split(r"[,\s]+", raw) if n.strip()]
    if "custom" in PROVIDERS and "custom" not in names and raw == default:
        # An endpoint the user configured by hand outranks the shared free tiers; if they
        # spelled out LLM_PROVIDER_ORDER themselves, that order is left exactly as written.
        names.insert(0, "custom")
    forced = (SETTINGS.llm_provider or "").strip().lower()
    if forced and forced not in ("auto", "any"):
        if forced not in PROVIDERS:
            log.warning("LLM_PROVIDER=%s is unknown; falling back to the default order", forced)
        else:
            names = [forced] + [n for n in names if n != forced]
    unknown = [n for n in names if n not in PROVIDERS]
    for name in unknown:
        log.warning("ignoring unknown LLM provider %r", name)
    return [n for n in names if n in PROVIDERS]


# ---------------------------------------------------------------------------
# Task sizing: cheap model for the small stuff, big model when it pays off
# ---------------------------------------------------------------------------

_HEAVY_HINTS = re.compile(
    r"\b(compare|versus|vs\.?|why|explain|reason|plan|design|architect|refactor|debug|fix|optimi[sz]"
    r"|summari[sz]e|summarise|analyse|analy[sz]e|review|proof.?read|translate|write|draft|compose"
    r"|essay|email|report|step[ -]by[ -]step|in.?detail|thorough|careful|thought|trade.?offs?"
    r"|pros and cons|math|calculate|equation|derive|code|function|script|regex|sql|schema"
    r"|multi[ -]?part|everything|all of|long|full)\b",
    re.I,
)
_LIGHT_HINTS = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|yes|no|yo|sup|good (morning|evening|afternoon))\b"
    r"|\b(what time|what's the time|current time|volume|mute|screenshot|lock (the )?pc|shut ?down|restart)\b",
    re.I,
)
_MULTI_CLAUSE = re.compile(r"\s+(?:and|then|also|plus|but|after that|before that)\s+", re.I)


def choose_tier(text: str, has_tool_context: bool = False) -> str:
    """``"fast"`` or ``"smart"`` -- which model this utterance deserves.

    Deliberately biased toward *fast*: free tiers are metered per token, so the
    heavy model is only worth it when the task is reasoning-shaped.  Long
    prompts and multi-part asks tip it over.
    """
    clean = " ".join((text or "").split())
    if not clean:
        return "fast"
    words = len(clean.split())
    if _LIGHT_HINTS.search(clean) and words < 12:
        return "fast"
    score = 0
    if _HEAVY_HINTS.search(clean):
        score += 2
    if words >= 22:
        score += 2
    elif words >= 13:
        score += 1
    if len(clean) > 320:
        score += 2
    if len(_MULTI_CLAUSE.findall(clean)) >= 2:
        score += 1
    if clean.rstrip().endswith("?") and words >= 9:
        score += 1
    if has_tool_context:
        score += 1                      # composing an answer over tool output is the fiddly bit
    if _MULTI_CLAUSE.search(clean) and "notes" in clean.lower():
        score += 1                      # "summarise my notes and ..." is not a one-liner
    return "smart" if score >= 3 else "fast"


_RESEARCH_HINTS = re.compile(
    r"\b(search|google|look up|find out|news|weather|price|score|fixture|result|record|standing"
    r"|who won|who is|current|latest|today'?s|stock|exchange rate|version)\b",
    re.I,
)


def looks_like_research(text: str) -> bool:
    """Whether a *no-LLM-available* fallback should reach for the network.

    Used only when every provider is unreachable: with no model to think with,
    JARVIS answers facts from search, but it must not turn "hello" or "explain
    myself better" into a DuckDuckGo query.
    """
    clean = " ".join((text or "").split())
    if len(clean) < 6:
        return False
    return bool(_RESEARCH_HINTS.search(clean))


# ---------------------------------------------------------------------------
# HTTP seam (single function so tests never touch the network)
# ---------------------------------------------------------------------------

def _get_json(url: str, headers: Dict[str, str], timeout: float) -> Tuple[int, Dict[str, str], Any]:
    """GET JSON (only ``/models`` discovery uses it) - the second and last network seam."""
    request = urllib.request.Request(url, method="GET")
    for name, value in headers.items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read().decode("utf-8", "replace")
            return response.status, {k.lower(): v for k, v in dict(response.headers).items()}, _loads(raw)
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            raw = ""
        return exc.code, {k.lower(): v for k, v in dict(exc.headers or {}).items()}, _loads(raw)
    except urllib.error.URLError as exc:
        raise ConnectionError(f"network error: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ConnectionError("request timed out") from exc


def _model_size(model: str) -> float:
    """Parameter count in billions from a model id (``llama-3.3-70b-versatile`` -> 70)."""
    found = [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*b\b", model.lower())]
    return found[-1] if found else 0.0


def _model_row(row: Any) -> Dict[str, Any]:
    """One ``/models`` entry -> ``{id, tools, json, context, text_out}``; unknown stays None.

    Providers shape this endpoint differently (OpenAI's ``data`` + ``supported_features``,
    GitHub's ``model_name``, Ollama's ``name``, Gemini's ``models/x`` ids) and several say
    nothing beyond the id, so a missing field means "not stated" and must not be read as
    "cannot" - that difference decides whether a model is usable for tool calls.
    """
    if isinstance(row, str):
        row = {"id": row}
    if not isinstance(row, dict):
        return {}
    raw = str(row.get("id") or row.get("name") or row.get("model_name") or row.get("base_model") or "")
    if not raw:
        return {}
    if raw.startswith("models/"):
        raw = raw[len("models/"):]
    feats = [str(f).lower() for f in (row.get("supported_features") or [])]
    out_mods = [str(m).lower() for m in (row.get("output_modalities") or [])]
    in_mods = [str(m).lower() for m in (row.get("input_modalities") or [])]
    ctx = row.get("context_window") or row.get("context_length") or row.get("supported_input_tokens") or 0
    try:
        ctx = int(ctx)
    except (TypeError, ValueError):
        ctx = 0
    return {"id": raw,
            "tools": (("tools" in feats or "function_calling" in feats) if feats else None),
            "json": (("json_mode" in feats or "structured_outputs" in feats) if feats else None),
            "context": ctx or None,
            "text_out": ("text" in out_mods) if out_mods else None,
            # Vision matters for "what am I looking at": an id that cannot take an image
            # would 400 on the screenshot, so it must never be chosen for that job.
            "vision": ("image" in in_mods or "image_url" in in_mods) if in_mods else None,
            "active": (row.get("active") is not False)}


def _model_rows(body: Any) -> List[Dict[str, Any]]:
    """Normalise a whole ``/models`` payload; unreadable entries are dropped."""
    rows: Any = []
    if isinstance(body, dict):
        rows = body.get("data") or body.get("models") or body.get("result") or body.get("object") or []
    elif isinstance(body, list):
        rows = body
    if not isinstance(rows, list):
        return []
    seen: List[str] = []
    out: List[Dict[str, Any]] = []
    for row in rows:
        entry = _model_row(row)
        mid = entry.get("id")
        if mid and mid not in seen:
            seen.append(mid)
            out.append(entry)
    return out


def _pick_models(rows: List[Dict[str, Any]], need_tools: bool = True) -> Tuple[str, str, List[str]]:
    """Choose ``(fast, smart, tool_capable_ids)`` from what this key can actually see.

    The ids alone are not enough: a 2026-09 Groq key lists ``allam-2-7b`` (4K window, JSON
    only), ``whisper``/``orpheus`` (audio), ``llama-prompt-guard`` (classifiers) and
    ``gpt-oss-safeguard`` next to the two models JARVIS wants. So the ranking is
    tool-capable -> long-enough context -> smallest for the fast tier, biggest for smart,
    with non-text and guard/embedding/speech models dropped outright.

    ``rows`` are normalised entries from :func:`_model_rows`; raw ``/models`` dictionaries
    are also accepted (normalised here) so callers can pass either shape.
    """
    raw_keys = {"supported_features", "context_window", "context_length", "output_modalities",
                "name", "model_name", "base_model"}
    if any(isinstance(r, dict) and (set(r) & raw_keys) for r in rows):
        rows = [m for m in (_model_row(r) for r in rows) if m]
    clean: List[Dict[str, Any]] = []
    for row in rows:
        mid = str(row.get("id") or "")
        if not mid or _MODEL_BANNED.search(mid):
            continue
        if row.get("text_out") is False or row.get("active") is False:   # audio, or listed-but-off
            continue
        clean.append(row)
    capable = [r for r in clean if r.get("tools") is True]
    pool = (capable if need_tools else []) or [r for r in clean if r.get("json") is not False] or clean
    roomy = [r for r in pool if (r.get("context") or 0) >= MIN_USEFUL_CONTEXT]
    if roomy:
        pool = roomy
    ids: List[str] = []
    for row in pool:
        mid = str(row.get("id"))
        if mid not in ids:
            ids.append(mid)
    if not ids:
        return "", "", [str(r.get("id")) for r in capable]
    sized = {i: _model_size(i) for i in ids}
    sweet = [i for i in ids if 2.0 <= sized[i] <= 14.0] or ids
    fast = min(sweet, key=lambda i: (0 if _MODEL_FAST_NAME.search(i) else 1, sized[i] or 999))
    big = [i for i in ids if sized[i] >= 20.0]
    if big:
        smart = max(big, key=lambda i: (sized[i], 1 if _MODEL_SMART_NAME.search(i) else 0))
    else:
        smart = max(ids, key=lambda i: (0 if _MODEL_FAST_NAME.search(i) else 1, sized[i]))
    return fast, (smart if smart != fast else fast), [str(r.get("id")) for r in capable]


def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> Tuple[int, Dict[str, str], Any]:
    """POST JSON, return ``(status, response_headers, decoded_body_or_raw_text)``."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    for name, value in headers.items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - https by config
            raw = response.read().decode("utf-8", "replace")
            return response.status, {k.lower(): v for k, v in dict(response.headers).items()}, _loads(raw)
    except urllib.error.HTTPError as exc:
        raw = ""
        try:
            raw = exc.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            pass
        return exc.code, {k.lower(): v for k, v in dict(exc.headers or {}).items()}, _loads(raw)
    except urllib.error.URLError as exc:
        raise ConnectionError(f"network error: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ConnectionError("request timed out") from exc


def _loads(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


# ---------------------------------------------------------------------------
# Cooldown arithmetic
# ---------------------------------------------------------------------------

_PACIFIC_OFFSET = -7    # US/Pacific without a tz database dependency (PDT; good enough for a daily reset)


def _seconds_until_reset(kind: str, now: Optional[float] = None) -> int:
    now = time.time() if now is None else now
    if kind == "none":
        return 6 * 3600
    offset = _PACIFIC_OFFSET if kind == "daily-pacific" else 0
    local = datetime.fromtimestamp(now, tz=timezone.utc) + timedelta(hours=offset)
    tomorrow = (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(60, int((tomorrow - local).total_seconds()))


def _parse_retry_after(headers: Dict[str, str]) -> Optional[int]:
    value = (headers.get("retry-after") or "").strip()
    if not value:
        return None
    if value.isdigit():
        return int(value)
    try:
        from email.utils import parsedate_to_datetime

        when = parsedate_to_datetime(value)
        if when is not None:
            delta = int(when.timestamp() - time.time())
            return max(5, delta)
    except (TypeError, ValueError):
        return None
    return None


def _reset_header_seconds(headers: Dict[str, str]) -> Optional[int]:
    """``x-ratelimit-reset-tokens: 7.66s`` style headers (Groq, OpenAI, Cerebras)."""
    for name in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        raw = (headers.get(name) or "").strip()
        if not raw:
            continue
        match = re.match(r"^(\d+(?:\.\d+)?)(ms|s|m|h)?$", raw)
        if not match:
            continue
        amount = float(match.group(1))
        unit = match.group(2) or "s"
        scale = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]
        return max(5, int(amount * scale))
    return None


def _out_of_minute_window(body: Any) -> bool:
    text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False, default=str)
    return bool(re.search(r"per[ -]minute|/min|RPM|TPM|rate.?limit.*(minute|per minute)|requests per minute", text, re.I))


def _model_not_found(body: Any) -> bool:
    text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False, default=str)
    return bool(re.search(r"model.*(not (be )?found|does not exist|unknown|unsupported|deprecat|no longer)", text, re.I))


def _quota_text(body: Any) -> str:
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err.get("code") or "")[:200]
        if isinstance(err, list) and err:
            return str(err[0])[:200]
        return str(body.get("errors") or err or body)[:200]
    return str(body)[:200]


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------

class LlmPool:
    """Rotating, quota-aware client for every configured provider."""

    def __init__(self, state_file: Optional[Path] = None) -> None:
        # Re-entrant: _penalise() holds this while it reads _health_for(), and the
        # state save happens under the same lock, so a plain Lock would deadlock the turn.
        self._lock = threading.RLock()
        self.state_file = Path(state_file) if state_file else SETTINGS.llm_state_path
        self._health: Dict[str, Dict[str, Any]] = {}
        self._load_state()
        self.calls = 0
        self.failures = 0
        self.total_ms = 0
        self.last_error = ""
        self.last_provider = ""
        self.tier_used = {"fast": 0, "smart": 0}
        self.last_tier = ""
        self._offline_until = 0.0

    # -- persistence -------------------------------------------------------
    def _load_state(self) -> None:
        try:
            raw = json.loads(self.state_file.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                providers = raw.get("providers")
                if isinstance(providers, dict):
                    self._health = {k: dict(v) for k, v in providers.items() if isinstance(v, dict)}
                self._health["_last_error"] = str(raw.get("last_error", ""))[:400]
        except (OSError, json.JSONDecodeError):
            self._health = {}

    def _save_state(self) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {"updated": datetime.now().isoformat(timespec="seconds"),
                       "last_error": self.last_error[:400],
                       "providers": self._health}
            self.state_file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            log.debug("could not persist llm state: %s", exc)

    def _health_for(self, key: str) -> Dict[str, Any]:
        with self._lock:
            return self._health.setdefault(key, {"cool_until": 0, "reason": "", "fails": 0, "ok": 0, "ms": 0, "bad_models": {}})

    # -- availability ------------------------------------------------------
    def cooling(self, key: str, now: Optional[float] = None) -> Tuple[bool, int, str]:
        health = self._health_for(key)
        until = float(health.get("cool_until", 0) or 0)
        now = time.time() if now is None else now
        left = int(until - now)
        return (left > 0, max(0, left), str(health.get("reason", "")))

    def configured(self) -> List[str]:
        return [k for k in order() if PROVIDERS[k].ready]

    def available(self) -> List[str]:
        return [k for k in self.configured() if not self.cooling(k)[0]]

    @property
    def usable(self) -> bool:
        return bool(self.available())

    def model_for(self, key: str, tier: str) -> str:
        """Which model id to send this provider right now.

        The catalogue ids in ``PROVIDERS`` are what the vendor advertised when this file
        was written; free tiers rename them constantly and a fresh key is not always
        entitled to the flagship.  So the ranking is: the configured id for this tier, the
        id ``/models`` says this key can see, the provider's other tier - skipping any id
        the key has already rejected with a 404.
        """
        spec = PROVIDERS[key]
        configured = spec.smart_model if tier == "smart" else spec.fast_model
        health = self._health_for(key)
        found = health.get("discovered") or {}
        bad = health.get("bad_models") or {}
        listed = set(found.get("ids") or [])
        other_tier = found.get("fast" if tier == "smart" else "smart") or ""
        for candidate in (configured, found.get(tier) or "", other_tier):
            if not candidate:
                continue
            if bad.get(candidate, 0) > time.time():
                continue
            if listed and candidate not in listed:
                log.debug("%s: %s is not in this key's model list", spec.label, candidate)
                continue
            return candidate
        alt = spec.fast_model if configured == spec.smart_model else spec.smart_model
        return alt or configured

    def models_to_try(self, key: str, tier: str) -> List[str]:
        """The model ladder for one provider inside a single turn."""
        spec = PROVIDERS[key]
        health = self._health_for(key)
        found = health.get("discovered") or {}
        bad = health.get("bad_models") or {}
        wanted = spec.smart_model if tier == "smart" else spec.fast_model
        other = spec.fast_model if tier == "smart" else spec.smart_model
        ladder = [self.model_for(key, tier), other,
                  found.get(tier) or "", found.get("fast" if tier == "smart" else "smart") or "",
                  found.get("fast") or "", found.get("smart") or "", wanted,
                  *spec.fallback_models]
        out: List[str] = []
        for model in ladder:
            model = (model or "").strip()
            if model and model not in out:
                out.append(model)
        # A model the key already rejected goes last instead of being dropped: if it is
        # the only one the account can see, it is still the only thing worth trying.
        out.sort(key=lambda m: 1 if bad.get(m, 0) > time.time() else 0)
        return out

    def prewarm(self) -> Dict[str, Any]:
        """Read every configured provider's model list once, at start-up.

        Cheap (one small GET each, bounded by LLM_PROBE_TIMEOUT_SECONDS) and it means the
        first utterance of the day is answered with a model this key really has, instead
        of spending a turn discovering that a free-tier id was renamed.
        """
        summary: Dict[str, Any] = {}
        for key in self.configured():
            try:
                found = self.discover(key) or {}
            except Exception as exc:  # noqa: BLE001 - boot must not fail on a diagnostic
                log.debug("model discovery skipped for %s: %s", key, exc)
                found = {}
            summary[key] = {"seen": int(found.get("count", 0) or 0),
                            "fast": self.model_for(key, "fast"),
                            "smart": self.model_for(key, "smart")}
        return summary

    def discover(self, key: str, *, force: bool = False) -> Dict[str, Any]:
        """Ask a provider which models this key can use, and remember the answer.

        Cheaper than guessing: one GET against the OpenAI-compatible ``/models`` list
        turns "the model does not exist" from a failed turn into a working model id, and
        it also tells the HUD what is really being used.  Cached for 6 hours (and in
        ``data/llm_state.json``), skipped for endpoints that do not expose it.
        """
        spec = PROVIDERS[key]
        health = self._health_for(key)
        cached = health.get("discovered") or {}
        if not force and cached.get("ids") and time.time() - float(cached.get("at", 0)) < DISCOVER_TTL_SECONDS:
            return cached
        if spec.style != "openai" or not spec.root:
            return cached          # Cloudflare's /ai/run has no compatible list endpoint
        try:
            status, _headers, body = _get_json(spec.root + "/models", self._headers(spec),
                                               SETTINGS.llm_probe_timeout)
        except Exception as exc:  # noqa: BLE001 - discovery is a bonus, never fatal
            log.debug("model discovery for %s failed: %s", spec.label, exc)
            return cached
        rows = _model_rows(body)
        ids: List[str] = []
        for row in rows:
            mid = str(row.get("id") or "")
            if mid and mid not in ids:
                ids.append(mid)
        fast, smart, capable = _pick_models(rows)
        vision = [str(r.get("id")) for r in rows if r.get("vision") and not _MODEL_BANNED.search(str(r.get("id") or ""))]
        picked = {"at": int(time.time()), "count": len(ids), "ids": ids[:200],
                  "fast": fast, "smart": smart or fast, "tools": capable[:60],
                  "vision": vision[:40],
                  "context": {str(r.get("id")): r.get("context") for r in rows
                              if r.get("context") and len(ids) <= 60},
                  "reason": "" if ids else f"HTTP {status}: no model list"}
        if ids and not fast:
            picked["reason"] = "no chat model with a usable context window on this key"
        with self._lock:
            health["discovered"] = picked
        self._save_state()
        if ids:
            log.info("%s: key sees %d models (fast=%s, smart=%s)", spec.label, len(ids), fast or "?", smart or "?")
        else:
            log.warning("%s: could not read the model list (HTTP %s)", spec.label, status)
        return picked

    # -- requests ----------------------------------------------------------
    def complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tier: str = "fast",
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        images: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Chat completion from the first healthy provider. Raises ``LlmError``.

        ``images`` (data URLs or https URLs) switches the last user turn to OpenAI-style
        multimodal content; providers whose key cannot see an image-capable model are skipped
        rather than sent a request that is sure to fail.
        """
        offline_left = float(getattr(self, "_offline_until", 0) or 0) - time.time()
        if offline_left > 0:
            raise LlmError(f"offline: LLM providers are unreachable for the next {int(offline_left)}s")
        candidates = order()
        ready = [k for k in candidates if PROVIDERS[k].ready]
        if not ready:
            raise NoProviderConfigured(
                "no LLM provider key is configured - put one of GROQ_API_KEY, CEREBRAS_API_KEY, "
                "GEMINI_API_KEY or CLOUDFLARE_API_TOKEN in .env"
            )
        tried_realms: set = set()
        attempts: List[Dict[str, Any]] = []
        first_error = ""
        deadline = time.perf_counter() + max(4.0, float(SETTINGS.llm_budget_seconds))
        network_failures = 0
        # Two passes: healthy providers first, then cooling ones that are within
        # 5 minutes of coming back (better a slow answer than a fallback search).
        for pass_no in (0, 1):
            for key in ready:
                # Budget first: seven unreachable endpoints at 30 s each would leave
                # the user talking to a frozen assistant for three and a half minutes.
                if attempts and time.perf_counter() > deadline:
                    attempts.append({"provider": key, "error": "skipped: LLM time budget exhausted", "skipped": True})
                    break
                if network_failures >= 2:
                    # Two boxes down in a row means *this network* is down (no egress,
                    # DNS, a portal page).  Rest the whole pool briefly and move on.
                    self._offline_until = time.time() + min(600, max(60, SETTINGS.llm_offline_cooldown))
                    log.warning("no outbound access to the LLM providers; resting the pool for %ds",
                                int(min(600, max(60, SETTINGS.llm_offline_cooldown))))
                    break
                cooling, left, reason = self.cooling(key)
                if pass_no == 0 and cooling and left > 300:
                    attempts.append({"provider": key, "error": f"cooling down ({reason})", "skipped": True})
                    continue
                if pass_no == 1 and not (cooling and left <= 300):
                    continue
                spec = PROVIDERS[key]
                # A provider gets a whole LADDER of model ids in this one turn: its tier
                # model, its other model, and whatever /models says the key can see. A
                # retired or not-entitled id must never cost the user the answer.
                ladder = self.models_to_try(key, tier)
                if images:
                    # Vision is a per-model property: keep only ids known to take an image.
                    found = self._health_for(key).get("discovered") or {}
                    if not found.get("vision") and SETTINGS.llm_auto_discover:
                        found = self.discover(key) or {}
                    known = list(found.get("vision") or []) + list(spec.vision_models)
                    if spec.style != "openai":
                        attempts.append({"provider": key, "error": "no image input on this API style",
                                        "skipped": True})
                        continue
                    if known:
                        ladder = [m for m in ladder if m in known] or known
                cursor = 0
                discovered_here = False
                give_up = False
                while cursor < len(ladder) and not give_up:
                    model = ladder[cursor]
                    cursor += 1
                    if not model or (key, model) in tried_realms:
                        continue
                    tried_realms.add((key, model))
                    # A model that only advertises json_mode will 400 on a tools array, so
                    # ask for native tool calls only of the ids this key can see that support them.
                    capable = (self._health_for(key).get("discovered") or {}).get("tools") or []
                    use_tools = (bool(tools) and spec.supports_tools
                                 and (not capable or model in capable))
                    #: The second request per model exists only to retry *without* a tools array.
                    #: When no tools were asked for, sending the same body twice wastes a request on
                    #: a metered free tier and makes one turn look like it tried a model twice.
                    for attempt_no in range(2 if use_tools else 1):
                        started = time.perf_counter()
                        try:
                            payload = self._payload(spec, model,
                                                     self._with_images(messages, images),
                                                     tools=tools if use_tools else None,
                                                     json_mode=json_mode, max_tokens=max_tokens, temperature=temperature)
                            url = spec.endpoint(model)
                            if spec.style == "cloudflare" or (spec.key == "cloudflare" and attempt_no == 1):
                                url, payload = self._cloudflare_native(spec, model, messages, max_tokens, temperature)
                            status, headers, body = _post_json(url, self._headers(spec), payload, SETTINGS.llm_timeout)
                        except (ConnectionError, OSError, ValueError) as exc:
                            ms = int((time.perf_counter() - started) * 1000)
                            network_failures += 1
                            self._penalise(key, "network", str(exc)[:180], SETTINGS.llm_offline_cooldown)
                            attempts.append({"provider": key, "model": model, "error": str(exc)[:180], "latency_ms": ms})
                            first_error = first_error or f"{spec.label}: {exc}"
                            give_up = True               # this network is down, not this model
                            break
                        ms = int((time.perf_counter() - started) * 1000)
                        if status == 200:
                            parsed = self._parse(body, ms, spec, model)
                            self._reward(key, ms)
                            self.calls += 1
                            self.total_ms += ms
                            self.last_provider = key
                            self.tier_used[tier if tier in self.tier_used else "fast"] += 1
                            self.last_tier = tier if tier in self.tier_used else "fast"
                            parsed["attempts"] = attempts
                            parsed["model_switched"] = model != ladder[0]
                            return parsed
                        message = _quota_text(body)
                        if status == 404 and spec.account_env and not _model_not_found(body):
                            # A 404 on the account path is not about the model: wrong
                            # CLOUDFLARE_ACCOUNT_ID, or a token without Workers AI access.
                            self._penalise(key, "account",
                                           "404 from Cloudflare - check CLOUDFLARE_ACCOUNT_ID and that the "
                                           "token has Workers AI: Read & Write")
                            give_up = True
                            break
                        if _model_not_found(body) or status == 404 or (status in (400, 408) and attempt_no == 1):
                            # Model-shaped failure: park this id, then try the next rung of
                            # the ladder - and once, ask the endpoint what it does offer.
                            self._mark_model_bad(key, model, message)
                            attempts.append({"provider": key, "model": model, "status": status,
                                             "error": message or f"HTTP {status} (model unavailable)",
                                             "latency_ms": ms, "model_problem": True})
                            first_error = first_error or f"{spec.label} HTTP {status}: {message[:160]}"
                            if SETTINGS.llm_auto_discover and not discovered_here:
                                discovered_here = True
                                if (self.discover(key) or {}).get("ids"):
                                    # Rebuild the ladder from what the key can actually see,
                                    # then walk it from the top: tried_realms still guards
                                    # the ids this turn already burned, so nothing repeats.
                                    rest = [m for m in ladder[cursor:] if (key, m) not in tried_realms]
                                    ladder = [m for m in self.models_to_try(key, tier) if m not in rest] + rest
                                    cursor = 0
                            break           # next rung of the ladder, not another round of this one
                        if status in (400, 422) and use_tools and attempt_no == 0:
                            use_tools = False               # hub rejected native tools: JSON instead
                            continue
                        self._penalise_http(key, status, headers, message)
                        attempts.append({"provider": key, "model": model, "status": status,
                                         "error": message or f"HTTP {status}", "latency_ms": ms})
                        first_error = first_error or f"{spec.label} HTTP {status}: {message[:160]}"
                        give_up = True                      # quota/auth: another model will not help
                        break
                    if give_up:
                        break
                if give_up:
                    continue                      # auth/quota/network: already penalised
                mine = [a for a in attempts if a.get("provider") == key]
                if mine and all(a.get("model_problem") for a in mine):
                    # Every model id this provider offers was refused: back off briefly
                    # instead of burning a fresh turn on the same dead end each utterance.
                    self._penalise(key, "model",
                                   "no model id on this key worked - run `python llm_providers.py` "
                                   "to see what it can use, or set "
                                   f"{key.upper()}_MODEL_FAST / {key.upper()}_MODEL_SMART in .env", 600)
        self.failures += 1
        self.last_error = first_error or "every configured provider failed"
        self._save_state()
        raise LlmError(self.last_error, attempts)

    @staticmethod
    def _with_images(messages: List[Dict[str, Any]], images: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Rebuild the last user turn as multimodal content, leaving history untouched."""
        if not images:
            return messages
        out = [dict(m) for m in messages]
        for index in range(len(out) - 1, -1, -1):
            if out[index].get("role") == "user":
                text = out[index].get("content")
                if isinstance(text, list):
                    text = " ".join(str(part.get("text", "")) for part in text if isinstance(part, dict))
                parts: List[Dict[str, Any]] = [{"type": "text", "text": str(text or "")}]
                for image in images[:4]:
                    url = str(image)
                    if url and not url.startswith(("http://", "https://", "data:")):
                        url = "data:image/png;base64," + url
                    parts.append({"type": "image_url", "image_url": {"url": url}})
                out[index]["content"] = parts
                break
        return out

    def vision(self, prompt: Union[str, List[Dict[str, Any]]], images: List[str], *,
               question: str = "", max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """:func:`complete` with an image, phrased the way the rest of JARVIS calls it.

        ``prompt`` is normally a plain string, but a list of OpenAI-style messages is
        tolerated (its user turns' text is flattened) so a caller can never trip over
        a ``'list' object has no attribute 'strip'`` crash again.
        """
        if isinstance(prompt, (list, tuple)):
            text = " ".join(
                str(item.get("content", "")) for item in prompt
                if isinstance(item, dict)
            ).strip()
        else:
            text = str(prompt or "").strip()
        text = (question or "").strip() or text or "What is on this screen?"
        return self.complete([{"role": "user", "content": text}], images=list(images),
                             tier="smart", max_tokens=max_tokens)

    def _headers(self, spec: ProviderSpec) -> Dict[str, str]:
        headers = {"content-type": "application/json", "accept": "application/json",
                   "user-agent": "JARVIS-assistant/1.0"}
        key = spec.api_key()
        if key:                     # a LAN runtime (Ollama/LM Studio) has no key at all;
            headers["authorization"] = f"Bearer {key}"   # sending "Bearer " can be a 401
        headers.update(spec.extra_headers)
        return headers

    def _payload(
        self,
        spec: ProviderSpec,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]],
        json_mode: bool,
        max_tokens: Optional[int],
        temperature: Optional[float],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": SETTINGS.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or SETTINGS.llm_max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = True
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _cloudflare_native(self, spec: ProviderSpec, model: str, messages: List[Dict[str, Any]],
                           max_tokens: Optional[int], temperature: Optional[float]) -> Tuple[str, Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens or SETTINGS.llm_max_tokens,
            "temperature": SETTINGS.llm_temperature if temperature is None else temperature,
        }
        return f"{spec.root}/ai/run/{urllib.parse.quote(model, safe='@/-')}", payload

    @staticmethod
    def _parse(body: Any, ms: int, spec: ProviderSpec, model: str) -> Dict[str, Any]:
        content, calls, finish = "", [], ""
        if isinstance(body, dict) and isinstance(body.get("result"), dict) and "response" in body["result"]:
            result = body["result"]
            content = str(result.get("response") or "")
            calls = _native_tool_calls(result.get("tool_calls"))
            finish = str(result.get("stop_reason") or "stop")
        elif isinstance(body, dict) and isinstance(body.get("choices"), list) and body["choices"]:
            choice = body["choices"][0] or {}
            message = choice.get("message") or {}
            content = str(message.get("content") or "")
            if not content and isinstance(message.get("reasoning"), str):
                content = message["reasoning"]
            calls = _native_tool_calls(message.get("tool_calls"))
            if not calls:
                for tc in message.get("tool_calls") or []:      # openrouter/mistral variants
                    fn = (tc or {}).get("function") or {}
                    if fn.get("name"):
                        calls.append({"name": fn["name"], "arguments": _as_dict(fn.get("arguments")), "id": tc.get("id", "")})
            finish = str(choice.get("finish_reason") or "")
        else:
            raise ValueError(f"unrecognised response shape: {str(body)[:180]}")
        usage = body.get("usage") if isinstance(body, dict) else None
        return {
            "content": content.strip(),
            "tool_calls": calls,
            "provider": spec.key,
            "model": model,
            "latency_ms": ms,
            "finish_reason": finish,
            "usage": usage if isinstance(usage, dict) else {},
        }

    # -- health bookkeeping ------------------------------------------------
    def _reward(self, key: str, ms: int) -> None:
        with self._lock:
            health = self._health_for(key)
            health["fails"] = 0
            health["ok"] = int(health.get("ok", 0)) + 1
            health["ms"] = int((int(health.get("ms", 0)) * (health["ok"] - 1) + ms) / max(1, health["ok"]))
            health["cool_until"] = 0
            health["reason"] = ""

    def _penalise_http(self, key: str, status: int, headers: Dict[str, str], message: str) -> None:
        cap = max(1, SETTINGS.llm_cooldown_hours) * 3600
        if status in (401, 402, 403):
            self._penalise(key, "auth", message or f"HTTP {status} (check the API key/plan)", cap)
        elif status == 429:
            hinted = _parse_retry_after(headers) or _reset_header_seconds(headers)
            if hinted is None and _out_of_minute_window(message):
                hinted = MINUTE_WINDOW_SECONDS
            seconds = min(int(hinted) if hinted else _seconds_until_reset(PROVIDERS[key].reset), cap)
            window = "minute window" if hinted and hinted <= 120 else "quota window"
            self._penalise(key, "quota", f"429 {window}: {(message or 'rate limited')[:140]}", seconds)
        elif status in (404, 410) and _model_not_found(message):
            self._mark_model_bad(key, "", message or "model unavailable")
        elif status >= 500:
            self._penalise(key, "server", f"HTTP {status}: {message[:140]}", min(TRANSIENT_BASE_SECONDS * 2 ** self._health_for(key).get("fails", 0), TRANSIENT_CAP_SECONDS))
        else:
            self._penalise(key, "error", f"HTTP {status}: {message[:140]}", 300)

    def _penalise(self, key: str, kind: str, message: str, seconds: Optional[int] = None) -> None:
        cap = max(1, SETTINGS.llm_cooldown_hours) * 3600
        if seconds is None:
            seconds = min(TRANSIENT_BASE_SECONDS * 2 ** self._health_for(key).get("fails", 0), TRANSIENT_CAP_SECONDS)
        seconds = max(30, min(int(seconds), cap))
        with self._lock:
            health = self._health_for(key)
            health["fails"] = int(health.get("fails", 0)) + 1
            health["cool_until"] = time.time() + seconds
            health["reason"] = f"{kind}: {message[:180]}"
        log.warning("%s out of rotation for %dm (%s)", PROVIDERS[key].label, max(1, seconds // 60), health_reason(self._health_for(key)))
        self._save_state()

    def _mark_model_bad(self, key: str, model: str, message: str) -> None:
        with self._lock:
            health = self._health_for(key)
            bad = dict(health.get("bad_models") or {})
            for name in (model, *[m for m in (PROVIDERS[key].fast_model, PROVIDERS[key].smart_model) if m and m in (message or "")]):
                bad[name] = time.time() + MODEL_NOT_FOUND_SECONDS
            health["bad_models"] = bad
        # Persisted, so a restart does not spend the next turn re-discovering that this
        # key cannot use that id (that repeat was the whole complaint this fixes).
        self._save_state()

    def reset(self, key: str = "") -> Dict[str, Any]:
        """Drop cooldowns and learned model lists (all, or one provider).

        ``/api/llm/reset`` calls this, which is exactly what you want after swapping a
        key in ``.env``: the old key's rejections and model list go with it.
        """
        with self._lock:
            keys = [key] if key else list(self._health)
            for name in keys:
                if name == "_last_error":
                    continue
                health = self._health_for(name)
                health.update({"cool_until": 0, "reason": "", "fails": 0, "bad_models": {}})
            self.last_error = ""
        self._save_state()
        return {"ok": True, "reset": [k for k in keys if k in PROVIDERS]}

    def status(self) -> Dict[str, Any]:
        configured = self.configured()
        return {
            "order": order(),
            "configured": configured,
            "available": [k for k in configured if not self.cooling(k)[0]],
            "active": self.last_provider or (self.available()[0] if self.available() else ""),
            "tier": SETTINGS.llm_tier_mode,
            "calls": self.calls,
            "failures": self.failures,
            "avg_ms": int(self.total_ms / self.calls) if self.calls else 0,
            "tier_used": self.last_tier,
            "fast_calls": self.tier_used.get("fast", 0),
            "smart_calls": self.tier_used.get("smart", 0),
            "last_error": self.last_error[:200],
            "cooldown_hours": SETTINGS.llm_cooldown_hours,
            "budget_seconds": SETTINGS.llm_budget_seconds,
            "offline_rest_s": max(0, int(float(getattr(self, "_offline_until", 0) or 0) - time.time())),
            "providers": [
                {**PROVIDERS[key].describe(),
                 "model": self.model_for(key, "smart"),
                 "fast_model_used": self.model_for(key, "fast"),
                 "cooling": self.cooling(key)[0],
                 "cool_left_s": self.cooling(key)[1],
                 "reason": health_reason(self._health_for(key)),
                 "ok": int(self._health_for(key).get("ok", 0) or 0),
                 "avg_ms": int(self._health_for(key).get("ms", 0) or 0),
                 # what /models discovery learned about this key, and which ids it refused
                 "models_seen": int((self._health_for(key).get("discovered") or {}).get("count", 0) or 0),
                 "discovered": {k: v for k, v in (self._health_for(key).get("discovered") or {}).items()
                                if k in ("fast", "smart", "count", "at", "reason", "vision")},
                 "model_ids": list((self._health_for(key).get("discovered") or {}).get("ids") or [])[:16],
                 "vision_models": (self._health_for(key).get("discovered") or {}).get("vision")
                 or list(PROVIDERS[key].vision_models),
                 "rejected_models": sorted(m for m, until in (self._health_for(key).get("bad_models") or {}).items()
                                           if float(until or 0) > time.time())}
                for key in order()
            ],
        }

    # -- diagnostics -------------------------------------------------------
    def probe(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Ask every configured provider for one word, so the HUD can show who answers."""
        results: Dict[str, Any] = {}
        for key in self.configured():
            spec = PROVIDERS[key]
            started = time.perf_counter()
            try:
                parsed = self._single(spec, self.model_for(key, "fast"),
                                      [{"role": "user", "content": "Reply with exactly: OK"}],
                                      max_tokens=6, temperature=0.0)
                results[key] = {"ok": True, "model": self.model_for(key, "fast"),
                                "smart_model": self.model_for(key, "smart"),
                                "reply": parsed["content"][:24],
                                "latency_ms": int((time.perf_counter() - started) * 1000)}
            except Exception as exc:  # noqa: BLE001 - diagnostics must never raise
                results[key] = {"ok": False, "error": str(exc)[:200],
                                "model": self.model_for(key, "fast"),
                                "latency_ms": int((time.perf_counter() - started) * 1000)}
            seen = (self._health_for(key).get("discovered") or {})
            if seen.get("ids"):
                results[key]["models_available"] = list(seen["ids"])[:20]
        return {"ok": any(v.get("ok") for v in results.values()), "providers": results}

    def _single(self, spec: ProviderSpec, model: str, messages: List[Dict[str, Any]],
                *, max_tokens: int, temperature: float) -> Dict[str, Any]:
        payload = self._payload(spec, model, messages, tools=None, json_mode=False,
                                max_tokens=max_tokens, temperature=temperature)
        status, headers, body = _post_json(spec.endpoint(model), self._headers(spec), payload,
                                           timeout_guard(SETTINGS.llm_probe_timeout))
        if status != 200:
            raise RuntimeError(f"{spec.label} HTTP {status}: {_quota_text(body)[:160]}")
        return self._parse(body, 0, spec, model)


def _native_tool_calls(raw: Any) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = str(fn.get("name") or fn.get("tool") or "").strip()
        if not name:
            continue
        calls.append({"name": name, "arguments": _as_dict(fn.get("arguments") or fn.get("parameters") or {}),
                      "id": str(item.get("id") or "")})
    return calls


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def health_reason(health: Dict[str, Any]) -> str:
    left = int(float(health.get("cool_until", 0) or 0) - time.time())
    if left > 0:
        return str(health.get("reason") or "cooling down")
    return str(health.get("reason") or "")


def timeout_guard(value: float) -> float:
    """A probe should fail fast; a real answer may take a while, but never forever."""
    try:
        return max(2.0, float(value))
    except (TypeError, ValueError):
        return 20.0


#: Shared pool used by router.py, server.py and main.py.
POOL = LlmPool()


def status() -> Dict[str, Any]:
    return POOL.status()


def reset(key: str = "") -> Dict[str, Any]:
    return POOL.reset(key)


def probe() -> Dict[str, Any]:
    return POOL.probe()


# ---------------------------------------------------------------------------
# ``python llm_providers.py`` -- the "why did JARVIS say no provider answered" tool
# ---------------------------------------------------------------------------

def _cli(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - manual diagnostic
    """Print what each configured key can see and use.  No third-party imports."""
    import argparse

    parser = argparse.ArgumentParser(description="JARVIS AI provider diagnostic")
    parser.add_argument("--no-list", action="store_true", help="skip the /models discovery call")
    parser.add_argument("--ask", default="", help="also send this prompt to the pool")
    args = parser.parse_args(argv)

    print("JARVIS AI providers")
    print(f"  order          : {' -> '.join(order())}")
    print(f"  tier mode      : {SETTINGS.llm_tier_mode}   auto-discover: {SETTINGS.llm_auto_discover}")
    print(f"  timeout/budget : {SETTINGS.llm_timeout}s per request, {SETTINGS.llm_budget_seconds}s per turn")
    print(f"  cooldown cap   : {SETTINGS.llm_cooldown_hours}h   state: {POOL.state_file}")
    print(f"  keys in .env   : {', '.join(POOL.configured()) or 'NONE - add one to .env'}")
    code = 1
    for key in POOL.configured():
        spec = PROVIDERS[key]
        cooling, left, reason = POOL.cooling(key)
        print(f"\n  [{key}] {spec.label}")
        print(f"    endpoint     : {spec.root}")
        print(f"    catalogue ids: fast={spec.fast_model}  smart={spec.smart_model}")
        if not args.no_list:
            found = POOL.discover(key, force=True) or {}
            ids = list(found.get("ids") or [])
            if ids:
                capable = list(found.get("tools") or [])
                skipped = [i for i in ids if i not in set(capable) | {found.get("fast"), found.get("smart")}]
                print(f"    key can see  : {len(ids)} models -> {', '.join(ids[:12])}"
                      f"{' …' if len(ids) > 12 else ''}")
                print(f"    usable for   : {len(capable)} with native tool calls"
                      f"{' (' + ', '.join(capable[:6]) + ')' if capable else ''}")
                if skipped:
                    print(f"    ignored      : {', '.join(skipped[:8])}{' …' if len(skipped) > 8 else ''}"
                          "   (audio/guard/embedding, no tool support, or too small a context)")
                print(f"    chosen       : fast={found.get('fast') or '-'}  smart={found.get('smart') or '-'}")
                code = 0
            else:
                print(f"    key can see  : nothing listed ({found.get('reason') or 'no /models endpoint?'})")
        print(f"    will send    : fast={POOL.model_for(key, 'fast')}  smart={POOL.model_for(key, 'smart')}")
        rejected = sorted(m for m, until in ((POOL._health_for(key).get("bad_models") or {}).items())  # noqa: SLF001
                         if float(until or 0) > time.time())
        if rejected:
            print(f"    refused earlier: {', '.join(rejected)}")
        if cooling:
            print(f"    COOLING        : {left // 60}m left ({reason})")
    print("\n  live probe (one word each):")
    probed = POOL.probe()
    for key, outcome in (probed.get("providers") or {}).items():
        if outcome.get("ok"):
            print(f"    {key:11s} OK    {outcome['latency_ms']}ms on {outcome.get('model')} -> {outcome.get('reply')!r}")
            code = 0
        else:
            print(f"    {key:11s} FAIL  {str(outcome.get('error'))[:150]}")
    if args.ask:
        try:
            out = POOL.complete([{"role": "user", "content": args.ask}],
                                tier=choose_tier(args.ask))
            print(f"\n  answer via {out['provider']}:{out['model']} ({out['latency_ms']}ms): {out['content'][:400]}")
            code = 0
        except LlmError as exc:
            print(f"\n  answer failed: {exc}")
            for attempt in getattr(exc, "attempts", []) or []:
                print(f"    - {attempt.get('provider')} {attempt.get('model', '')} "
                      f"{attempt.get('status', '')} {str(attempt.get('error'))[:120]}")
    print("\n  If a turn still comes back empty, put these two lines in .env - they are the ids")
    print("  this exact key was just seen to accept, so nothing has to be guessed at runtime:")
    for key in POOL.configured():
        print(f"    {key.upper()}_MODEL_FAST={POOL.model_for(key, 'fast')}")
        print(f"    {key.upper()}_MODEL_SMART={POOL.model_for(key, 'smart')}")
    print("  (Also check the keys themselves; a 401/403 is an account problem, not a model one.)")
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
