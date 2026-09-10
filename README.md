# J.A.R.V.I.S. — native Windows AI assistant

Full-stack voice assistant for Windows 10/11 on an RTX 3050 4 GB: a WebGL arc-reactor HUD,
local speech-to-text, a cloned voice for output, OS automation, live web/sports data, your own
Markdown notes as a knowledge base, an agentic LLM brain and a Discord bridge.

```
 utterance (mic / typed / Discord)
        │
        ├─► TRACK 1  regex rules in server.py            < 1 ms  → tools.py  → answer  (~5-80 ms)
        │            "open steam", "mute", "real madrid score",
        │            "read my german notes", "how is my cpu"
        │
        └─► TRACK 2  router.py → llm_providers.py tool-calling loop  0.4-4 s → tools.py → answer
                     free-tier providers (Groq → Cerebras → Cloudflare → Gemini …),
                     strict JSON schemas, up to 3 tool rounds, cheap model for chat /
                     big model for reasoning, deterministic keyword planner when all
                     providers are exhausted (it never silently answers with a search)
```

---

## 1 · Files

| File | Responsibility |
| --- | --- |
| `main.py` | Launcher: pre-flight, uvicorn in a background thread, `pywebview` HUD window, clean shutdown |
| `server.py` | FastAPI app, `/ws` WebSocket, telemetry pump, **Track 1** regex rules, mic endpoints, REST API |
| `router.py` | **Track 2**: strict tool JSON schemas, agent loop, offline keyword planner, conversation memory |
| `llm_providers.py` | The brain: 7 free-tier providers (Groq, Cerebras, Cloudflare Workers AI, Gemini, Mistral, OpenRouter, GitHub Models) plus your own endpoint, tier selection, quota-aware rotation, 24 h cooldowns that survive a restart |
| `tools.py` | OS automation + live data: `launch_app`, `close_app`, `web_search`, `search_on_site`, `launch_app`, `open_website`, `fetch_sports_stats`, `read_notes`, `write_note`, `system_report`, `set_volume`, `take_screenshot`, `system_power`, `llm_status` |
| `audio_engine.py` | `faster-whisper` STT (cuda/int8) + Coqui XTTS-v2 TTS with Windows SAPI5 fallback, VAD, playback |
| `discord_bridge.py` | Background `discord.Client` on one channel → same router → chunked replies, auto-reconnect |
| `config.py` | Stdlib `.env` loader + typed settings, logging, path resolution |
| `static/index.html` | HUD markup: Three.js canvas + glassmorphic Tailwind overlay |
| `static/styles.css` | Meters, sparklines, terminal, pills, animations, scrollbar, webview chrome |
| `static/arc_reactor.js` | WebGL render loop **and** the HUD client (WebSocket, telemetry, terminal, mic, cards) |
| `notes/*.md` | Your study notes, read by `read_notes()` (German A2 + CS prep included as worked examples) |
| `tests/test_jarvis.py` | 110 stdlib-unittest checks: schemas, regex precision, provider failover, notes scoring, VAD, REST |

## 2 · Setup (Windows)

```bat
git clone <this repo> && cd JARVIS123
py -3.11 -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env            &notepad .env
.venv\Scripts\python.exe main.py
```

`launch-jarvis.bat` does all of that (creates the venv on first run). Then open
`http://127.0.0.1:8760` — the pywebview window loads `static/index.html` directly and connects to
the same core over WebSocket.

### GPU budget (why these defaults)

| Stage | Setting | VRAM |
| --- | --- | --- |
| STT | `WHISPER_MODEL=base`, `WHISPER_DEVICE=cuda`, `WHISPER_COMPUTE_TYPE=int8` | ~0.6 GB |
| TTS | XTTS-v2, `TTS_LOW_VRAM=true` | ~1.4 GB (streamed) |
| headroom | 4 GB − 2 GB ≈ 2 GB left for Windows/DLSS/browser | |

If you hit `CUDA out of memory`, set `WHISPER_MODEL=tiny` (or `TTS_DEVICE=cpu` — XTTS on CPU
takes ~2 s per sentence but never competes for VRAM). Whisper also walks a fallback ladder
`cuda/int8 → cuda/float16 → cpu/int8 → cpu/float32` and shrinks `base` if even that fails, so a
missing driver never stops the assistant.

### Voice clone

`assets/jarvis_sample.wav` must be **3–10 s, mono, ≥ 16 kHz (22.05 kHz ideal), clean** speech.
Record your own take (or a TTS render you like), trim the silence, save it there. Without it,
`audio_engine` automatically drops to the Windows SAPI5 voice — JARVIS still talks, it just is not
cloned. `readme`-level check: `curl http://127.0.0.1:8760/api/status` shows `tts.state`.

## 3 · Secrets (.env)

| Key | Used by | Without it |
| --- | --- | --- |
| `GROQ_API_KEY` (or any one of the other provider keys below) | `llm_providers.py` → `router.py` Track 2 | Track 1 rules still work; anything else gets an honest "no key" answer instead of a guess |
| `CEREBRAS_API_KEY`, `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID`, `GEMINI_API_KEY`, `MISTRAL_API_KEY`, `OPENROUTER_API_KEY`, `GITHUB_MODELS_TOKEN` | same, as failover | that provider is skipped; the ones you did set keep working |
| `DISCORD_TOKEN`, `DISCORD_CHANNEL_ID` | `discord_bridge.py` | bridge stays idle, HUD shows why |
| `API_SPORTS_KEY` | `tools.fetch_sports_stats` | tool returns "add your key from dashboard.api-football.com" |
| `JARVIS_PORT` | launcher + HUD | defaults to `8760` (auto-bumps if busy) |

### AI providers, quotas and failover

No vendor SDK is required: `llm_providers.py` speaks HTTPS to each provider's
OpenAI-compatible (or, for Cloudflare, native `/ai/run`) endpoint with stdlib
`urllib`, so a free key is the only thing you paste into `.env`.

| Provider | Free tier (2026-09) | Fast model | Heavy model | Key |
| --- | --- | --- | --- | --- |
| **Groq** | 30 req/min, 14,400 req/day on the 8B, 1,000/day on the 70B | `llama-3.1-8b-instant` | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| **Cerebras** | ~1M tokens/day, 30 req/min, 8K context | `llama3.1-8b` | `gpt-oss-120b` | `CEREBRAS_API_KEY` |
| **Cloudflare Workers AI** | 10,000 Neurons/day | `@cf/meta/llama-3.1-8b-instruct` | `@cf/openai/gpt-oss-120b` | `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` |
| **Google Gemini** | 1,000 req/day on Flash-Lite, 15 req/min — quota resets **midnight Pacific** | `gemini-2.5-flash-lite` | `gemini-2.5-flash` | `GEMINI_API_KEY` |
| **Mistral** | "Experiment" tier, ~1B tokens/month, ~2 req/min (trains on your data) | `mistral-small-latest` | `magistral-small-latest` | `MISTRAL_API_KEY` |
| **OpenRouter** | 20 req/min, 50 req/day on `:free` (1,000/day after a $10 top-up) | `meta-llama/llama-3.1-8b-instruct:free` | `openai/gpt-oss-120b:free` | `OPENROUTER_API_KEY` |
| **GitHub Models** | 150-1,000 req/day at 10-15 req/min, includes `gpt-4o` | `Meta-Llama-3.1-8B-Instruct` | `gpt-4o` | `GITHUB_MODELS_TOKEN` (a `models:read` PAT, *not* your `gh` token) |

Set **one** key and you are done; each extra key is another lane to fall back on.

**Need a provider that is not listed?** Free tiers change monthly, so the catalogue has one
open slot: `CUSTOM_LLM_BASE_URL` accepts any endpoint that speaks the OpenAI
`/chat/completions` contract — NVIDIA NIM (`https://integrate.api.nvidia.com/v1`, ~1,000
credits/day, `nvidia/llama-3.3-70b-instruct`), SambaNova, Fireworks, an Azure deployment, or a
**local** `http://127.0.0.1:11434/v1` (Ollama) / LM Studio server for a no-internet demo (leave
`CUSTOM_LLM_API_KEY` empty). A configured custom endpoint is tried **before** the shared free
tiers; `CUSTOM_LLM_TOOLS=false` switches a runtime that cannot function-call to the
JSON-in-prompt path, and `CUSTOM_LLM_RESET` tells the cooldown maths how its quota behaves.

**How a turn is routed**

1. `choose_tier()` classifies the utterance: short chat, "open/close/volume" style
   requests and anything a tool can answer go to the **fast** model; long,
   reasoning-shaped prompts (multi-clause, "compare", "prove", "why", code + design
   words) go to the **smart** model. `LLM_TIER_MODE=fast|smart` forces one tier.
2. Providers are tried in `LLM_PROVIDER_ORDER`. The first that answers wins.
3. On `429`/quota the provider leaves rotation for the length of **its own** window:
   `Retry-After` if present, the `x-ratelimit-reset-*` header if present, the length
   of the minute for an RPM/TPM throttle, otherwise until its daily reset (UTC, or
   Pacific for Gemini), capped by `LLM_COOLDOWN_HOURS=24`. Cooldowns are written to
   `data/llm_state.json`, so restarting JARVIS does not re-burn a limited key.
4. **Model ids are self-healing.** Free tiers rename and retire models, and a brand-new key
   is not always entitled to the flagship, so "The model `x` does not exist or you do not
   have access to it" is routine. A refused id is parked for 6 h (persisted, so a restart
   does not re-learn it), the *next* model on that provider's ladder is tried **in the same
   turn**, and once per provider JARVIS reads `GET /models` to see which ids the key can
   actually see and pins those (`LLM_AUTO_DISCOVER=true`, on by default, re-read at boot).
   If every id a provider offers is refused, that provider backs off for 10 minutes and
   says so — with `python llm_providers.py` as the one command that lists what each key
   sees, what will be sent, and probes them all. Pin ids yourself with
   `<KEY>_MODEL_FAST` / `<KEY>_MODEL_SMART` and discovery stops bothering with that
   provider. A `401/402/403` still exiles the provider (bad key or plan) instead of
   cycling models.
5. Two consecutive DNS/TLS failures rest the whole pool for
   `LLM_OFFLINE_COOLDOWN_SECONDS`, and one turn never spends more than
   `LLM_BUDGET_SECONDS` walking providers — a frozen mic is worse than a heuristic answer.

**Watch it happen.** The HUD's **AI providers** panel lists every provider with
`● ready / ▲ cooling / ○ no key`, the model it will use, its latency average and how
long a cooldown has left; the header pill shows which brain answered the last
command (`groq:llama-3.1-8b-instant` vs `local-regex-planner`). Hover a provider to see
how many models your key can actually list, which ids were refused, and what JARVIS
switched to. Buttons: **reset** clears cooldowns and the learned model list, **probe**
pings each configured key with a one-word prompt (and returns its visible model ids).

```bat
curl http://127.0.0.1:8760/api/llm                                       :: who is ready / cooling and why
curl -X POST http://127.0.0.1:8760/api/llm/reset -H "Content-Type: application/json" -d "{}"
curl -X POST http://127.0.0.1:8760/api/llm/probe                          :: real round trip per provider
```

## 4 · What it can do

**Instant (Track 1)** — no LLM, no network: `open steam`, `open steam and check my cpu`
(multi-intent split), `close chrome`, `play despacito`, `what's the time`, `mute`, `set volume to 40`,
`take a screenshot`, `read my notes about german dative`, `note that the exam is friday`,
`how is my cpu`, `real madrid score`, `shutdown the pc` (asks for `confirm` first), `stop`.

**Agentic (Track 2)** — tool-calling loop, e.g. *"find the cheapest Indian restaurants in Surat and
open the best one in Chrome, then tell me my RAM usage"* → `web_search` → `open_website` →
`system_report` → one spoken summary. Unknown-app requests are deliberately **not** guessed by Track 1
(`tools.is_known_app()` gate) and go to the agent instead.

**`open` / `search` / `play` are three different verbs** — the preposition and the
target decide which tool runs, and a named website always beats the app list:

| You say | JARVIS does |
| --- | --- |
| `open google`, `open youtube`, `open chat.openai.com` | opens the **website** (`open_website`) |
| `open chrome`, `open google chrome`, `open discord`, `open spotify` | launches the **program** (`launch_app`) |
| `search LM Arena on Youtube`, `search llm routers on google.com`, `search reddit for r/nvidia` | opens that site's own results page **and** reads the headlines back (`search_on_site`) |
| `search cheapest indian restaurants in surat` | DuckDuckGo (`web_search`) |
| `play despacito`, `play lofi on youtube` | YouTube |
| `play lofi on spotify`, `play podcasts on youtube music` | that service's web player |
| `what is the velocity of an unladen swallow`, `explain the german dative case` | **the AI** — Track 1 never answers a question by searching, so a conceptual question cannot be hijacked into a DuckDuckGo dump |
| `llm status`, `who is your ai`, `check the ai providers` | reports the provider pool out loud |

**Discord** — add the bot to your server, enable *Message Content Intent*, put the channel id in
`.env`, then talk to your PC from anywhere. Toggle HUD "discord relay" to mirror local replies back
into the channel.

## 5 · API

```
GET  /                      HUD                POST /api/command      {"text","speak","agent"}
GET  /healthz               liveness           POST /api/speak        {"text","play"}
GET  /api/status            mode/voice/router  POST /api/stop         kill TTS queue
GET  /api/telemetry         one snapshot       POST /api/transcribe   {audio_base64} -> STT
GET  /api/config            resolved config    POST /api/listen       raw mic bytes
GET  /api/history           turns + cards      POST /api/discord/mirror {"on":true}
GET  /api/notes?topic=german                    GET  /api/search?q=...
GET  /api/llm                  provider pool: ready / cooling / why
POST /api/llm/reset            {"provider":"groq"} or {} for all
POST /api/llm/probe            one-word round trip per configured key
GET  /api/sports?team=Real Madrid               GET  /api/apps
WS   /ws                    bidirectional stream (see below)
```

`/ws` inbound: `{"type":"command","text":"…","speak":true}`, `{"type":"mic","format":"webm","data":"<b64>"}`,
`{"type":"stream-start"}` + raw `s16le` mono 16 kHz binary frames (server-side VAD segments and
transcribes them automatically), `{"type":"stop"}`, `{"type":"ping"}`.
Outbound: `hello`, `telemetry`, `log`, `state`, `reply`, `card`, `transcript`, `segment`, `audio`, `ack`, `error`.

## 6 · Verification

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v     # 110 cases, all offline
.venv\Scripts\python.exe -c "import router,json;print(json.dumps(router.TOOL_SCHEMAS[0],indent=2))"
.venv\Scripts\python.exe -c "import llm_providers as l;print(l.POOL.configured() or 'NO KEYS');print(l.choose_tier('open steam'), l.choose_tier('compare the dative and accusative cases, then write a study plan'))"
curl http://127.0.0.1:8760/api/telemetry
curl http://127.0.0.1:8760/api/llm
:: spoken end-to-end check: this must answer with prose, not a list of links
curl -X POST http://127.0.0.1:8760/api/command -H "Content-Type: application/json" -d "{\"text\":\"what is the velocity of an unladen swallow\",\"speak\":false}"
```

## 7 · Troubleshooting

| Symptom | Cause → fix |
| --- | --- |
| HUD says `link offline`, retries forever | core not up / wrong port → `python main.py --url` or check `JARVIS_PORT` |
| `no speech detected` on every clip | `ffmpeg` missing (`winget install Gyan.FFmpeg`) or mic privacy lock-out |
| XTTS never loads | reference wav missing/short, or the Coqui model is not downloaded yet (first run pulls ~2 GB) |
| `CUDA out of memory` | `WHISPER_MODEL=tiny`, or `TTS_DEVICE=cpu` |
| Tool says it cannot find the app | add `CUSTOM_APPS=Eden=D:\Emulators\Eden\eden.exe` to `.env` |
| Window opens, then **Not responding**, log ends with `Error while processing window.native.AccessibilityObject…: maximum recursion depth exceeded` | pywebview builds the JS API by recursively walking every public attribute of the `js_api` object, so the native `Window` must never be reachable from it. `JarvisBridge` therefore exposes *methods only* and keeps the window on `self._window` (leading underscore = skipped). If you add bridge methods, do not add public attributes |
| `TypeError: 'Event' object is not callable` right after the window is created | the load event is an object you subscribe to with `+=`, not a decorator - `attach_loaded_handler()` handles both pywebview layouts and falls back to `webview.start(func, args)` |
| Quit/menu does nothing | window controls are capability-checked (`destroy` in 5.x, `close` in older builds); the bridge reports `supports none of …` instead of throwing |
| Discord bot reads nothing | *Message Content Intent* is off in the developer portal |
| Sports returns 429 | free tier = 10 req/min; the tool reports the `Retry-After` |
| "Every AI provider is busy or unreachable right now" | every key hit its ceiling. `curl /api/llm` shows who is cooling and for how long; wait for the reset, press **reset** in the HUD panel after you fix a key, or add another provider |
| "My AI brain has no key yet" | `.env` has no `GROQ_API_KEY`/`CEREBRAS_API_KEY`/… — copy `.env.example`, add one, restart |
| Answers come back as DuckDuckGo links for a conceptual question | that only happens when no provider answered **and** the question looks like it needs live data. `LLM_FALLBACK_SEARCH=false` turns it off entirely |
| `Groq: 429` every few minutes but no failover | the next provider in `LLM_PROVIDER_ORDER` needs a key too; rotation only skips to *configured* providers |
| `Cloudflare: 404 … check CLOUDFLARE_ACCOUNT_ID` | the token is fine but the account id is wrong (it is in the dashboard URL) or lacks *Workers AI: Read & Write* |
| "Every AI provider is busy… `The model X does not exist or you do not have access to it`" | the catalogue id is retired or your key lacks it. JARVIS now retries the provider's other models in the same turn and pins an id your key really has (`/models`), so this is usually self-inflicted only if you disabled `LLM_AUTO_DISCOVER`. Run `python llm_providers.py` to see the list, then pin `GROQ_MODEL_FAST=…` / `GROQ_MODEL_SMART=…` in `.env` |
| Only ever answers with the small model, never the 70B | your key has no access to it; that is the correct fallback. `curl http://127.0.0.1:8760/api/llm` shows `model_ids` and `rejected_models` per provider |
| First answer after boot takes ~30 s | a key is set but unreachable (proxy/AV blocking TLS): JARVIS walks the pool until `LLM_BUDGET_SECONDS`, then answers with the planner. Remove the dead key from `.env` |
| Volume/power did nothing | PowerShell blocked: run `powershell -Command "Get-ExecutionPolicy"`; allow `RemoteSigned` for the user |
