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
        └─► TRACK 2  router.py → Hugging Face tool-calling loop   0.6-4 s → tools.py → answer
                     strict JSON schemas, up to 3 tool rounds,
                     deterministic keyword planner when offline
```

---

## 1 · Files

| File | Responsibility |
| --- | --- |
| `main.py` | Launcher: pre-flight, uvicorn in a background thread, `pywebview` HUD window, clean shutdown |
| `server.py` | FastAPI app, `/ws` WebSocket, telemetry pump, **Track 1** regex rules, mic endpoints, REST API |
| `router.py` | **Track 2**: Hugging Face client, strict tool JSON schemas, agent loop, offline planner, conversation memory |
| `tools.py` | OS automation + live data: `launch_app`, `close_app`, `web_search`, `fetch_sports_stats`, `read_notes`, `write_note`, `system_report`, `set_volume`, `take_screenshot`, `system_power` |
| `audio_engine.py` | `faster-whisper` STT (cuda/int8) + Coqui XTTS-v2 TTS with Windows SAPI5 fallback, VAD, playback |
| `discord_bridge.py` | Background `discord.Client` on one channel → same router → chunked replies, auto-reconnect |
| `config.py` | Stdlib `.env` loader + typed settings, logging, path resolution |
| `static/index.html` | HUD markup: Three.js canvas + glassmorphic Tailwind overlay |
| `static/styles.css` | Meters, sparklines, terminal, pills, animations, scrollbar, webview chrome |
| `static/arc_reactor.js` | WebGL render loop **and** the HUD client (WebSocket, telemetry, terminal, mic, cards) |
| `notes/*.md` | Your study notes, read by `read_notes()` (German A2 + CS prep included as worked examples) |
| `tests/test_jarvis.py` | 45 stdlib-unittest checks: schemas, regex precision, notes scoring, VAD, REST |

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
| `HF_TOKEN` | `router.py` agentic track | heuristic keyword planner still executes real tools |
| `DISCORD_TOKEN`, `DISCORD_CHANNEL_ID` | `discord_bridge.py` | bridge stays idle, HUD shows why |
| `API_SPORTS_KEY` | `tools.fetch_sports_stats` | tool returns "add your key from dashboard.api-football.com" |
| `JARVIS_PORT` | launcher + HUD | defaults to `8760` (auto-bumps if busy) |

## 4 · What it can do

**Instant (Track 1)** — no LLM, no network: `open steam`, `open steam and check my cpu`
(multi-intent split), `close chrome`, `play despacito`, `what's the time`, `mute`, `set volume to 40`,
`take a screenshot`, `read my notes about german dative`, `note that the exam is friday`,
`how is my cpu`, `real madrid score`, `shutdown the pc` (asks for `confirm` first), `stop`.

**Agentic (Track 2)** — tool-calling loop, e.g. *"find the cheapest Indian restaurants in Surat and
open the best one in Chrome, then tell me my RAM usage"* → `web_search` → `open_website` →
`system_report` → one spoken summary. Unknown-app requests are deliberately **not** guessed by Track 1
(`tools.is_known_app()` gate) and go to the agent instead.

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
GET  /api/sports?team=Real Madrid               GET  /api/apps
WS   /ws                    bidirectional stream (see below)
```

`/ws` inbound: `{"type":"command","text":"…","speak":true}`, `{"type":"mic","format":"webm","data":"<b64>"}`,
`{"type":"stream-start"}` + raw `s16le` mono 16 kHz binary frames (server-side VAD segments and
transcribes them automatically), `{"type":"stop"}`, `{"type":"ping"}`.
Outbound: `hello`, `telemetry`, `log`, `state`, `reply`, `card`, `transcript`, `segment`, `audio`, `ack`, `error`.

## 6 · Verification

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -c "import router,json;print(json.dumps(router.TOOL_SCHEMAS[0],indent=2))"
curl http://127.0.0.1:8760/api/telemetry
```

## 7 · Troubleshooting

| Symptom | Cause → fix |
| --- | --- |
| HUD says `link offline`, retries forever | core not up / wrong port → `python main.py --url` or check `JARVIS_PORT` |
| `no speech detected` on every clip | `ffmpeg` missing (`winget install Gyan.FFmpeg`) or mic privacy lock-out |
| XTTS never loads | reference wav missing/short, or the Coqui model is not downloaded yet (first run pulls ~2 GB) |
| `CUDA out of memory` | `WHISPER_MODEL=tiny`, or `TTS_DEVICE=cpu` |
| Tool says it cannot find the app | add `CUSTOM_APPS=Eden=D:\Emulators\Eden\eden.exe` to `.env` |
| Discord bot reads nothing | *Message Content Intent* is off in the developer portal |
| Sports returns 429 | free tier = 10 req/min; the tool reports the `Retry-After` |
| Volume/power did nothing | PowerShell blocked: run `powershell -Command "Get-ExecutionPolicy"`; allow `RemoteSigned` for the user |
