"""
discord_bridge.py -- the voice assistant's second mouth.

A background ``discord.Client`` runs its own asyncio loop inside a dedicated
thread so it can never block FastAPI's loop (and vice versa). Behaviour:

* listens on ``DISCORD_CHANNEL_ID`` (or any channel where it is @mentioned, when
  no channel id is configured), ignoring bots and its own messages;
* strips the optional ``DISCORD_COMMAND_PREFIX`` / mention, then hands the text
  to :func:`server.handle_command` -- the exact same two-track pipeline the HUD
  uses, so a Discord message can open Steam on the host PC;
* streams a typing indicator while the agent works, replies in <=1900 char
  chunks (markdown-safe) and can open a thread for long answers;
* reconnects forever with exponential backoff on rate limits / dropped sockets;
* ``send()`` lets JARVIS push local replies into the channel when mirroring is
  toggled on from the HUD (``POST /api/discord/mirror``).
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from typing import Any, Dict, List, Optional

from config import SETTINGS, get_logger

log = get_logger("discord")

MAX_LEN = 1900          # Discord hard limit is 2000; leave room for the ellipsis
MENTION_RE = re.compile(r"^<@!?\d+>\s*", re.I)
CODE_FENCE = "```"


class DiscordBridge:
    """Long-lived Discord gateway client bound to one text channel."""

    def __init__(self, channel_id: Optional[int] = None, on_event: Optional[Any] = None) -> None:
        self.channel_id = int(channel_id if channel_id is not None else (SETTINGS.discord_channel_id or 0))
        self.token = SETTINGS.discord_token
        self.prefix = SETTINGS.discord_prefix or "/jarvis"
        self.on_event = on_event            # server.Hub, for HUD terminal lines
        self._client = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop = threading.Event()
        self._connected = threading.Event()
        self.detail = "created"
        self._handled = 0
        self._errors = 0
        self._last_seen = 0.0
        self._buffer: List[Dict[str, Any]] = []
        self._buffer_max = 200

    # ------------------------------------------------------------------ state
    @property
    def enabled(self) -> bool:
        return bool(SETTINGS.discord_enabled and self.token)

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "connected": self.connected,
            "channel_id": self.channel_id or None,
            "prefix": self.prefix,
            "handled": self._handled,
            "errors": self._errors,
            "detail": self.detail,
            "last_activity": time.strftime("%H:%M:%S", time.localtime(self._last_seen)) if self._last_seen else None,
        }

    # ------------------------------------------------------------------ start/stop
    def start(self) -> "DiscordBridge":
        """Spawn the client thread. Safe to call once from the FastAPI lifespan."""
        if not self.token:
            self.detail = "DISCORD_TOKEN missing in .env -- bridge idle"
            log.warning(self.detail)
            return self
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_forever, name="jarvis-discord", daemon=True)
        self._thread.start()
        return self

    def _run_forever(self) -> None:
        try:
            import discord  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self.detail = f"discord.py not installed ({exc})"
            log.warning(self.detail)
            return

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        privileged = getattr(discord, "PrivilegedIntentsRequired", None)
        login_failed = getattr(discord, "LoginFailure", None)
        backoff = 2
        try:
            while not self._stop.is_set():
                try:
                    intents = discord.Intents.default()
                    try:
                        intents.message_content = True   # toggle this in the developer portal too
                    except Exception:  # noqa: BLE001
                        pass
                    self._client = discord.Client(intents=intents)
                    self._wire(self._client, discord)
                    self.detail = "dialling"
                    log.info("discord bridge connecting (channel=%s)", self.channel_id or "any")
                    self._loop.run_until_complete(self._client.start(self.token))
                    backoff = 2          # clean exit -> next reconnect is quick
                except Exception as exc:  # noqa: BLE001 - network/auth/rate limit
                    self._errors += 1
                    self._connected.clear()
                    fatal = (privileged and isinstance(exc, privileged)) or (login_failed and isinstance(exc, login_failed))
                    self.detail = f"{'fatal' if fatal else 'dropped'}: {type(exc).__name__}: {exc}"
                    if fatal:
                        log.error("%s -- enable 'Message Content Intent' / fix DISCORD_TOKEN", self.detail)
                        return
                    log.warning("%s -- retrying in %ds", self.detail, backoff)
                    time.sleep(backoff)
                    backoff = min(120, backoff * 2)
        finally:
            try:
                pending = [t for t in asyncio.all_tasks(self._loop) if not t.done()]
                for task in pending:
                    task.cancel()
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:  # noqa: BLE001
                pass
            self._loop.close()
            self._connected.clear()
            self.detail = "stopped"
            log.info("discord bridge stopped")

    # ------------------------------------------------------------------ wiring
    def _wire(self, client, discord) -> None:
        @client.event
        async def on_ready() -> None:  # noqa: ANN001
            self._connected.set()
            self.detail = f"online as {client.user}"
            log.info("discord online: %s (guilds=%d)", client.user, len(client.guilds))
            self._emit("sys", f"Discord online as {client.user}")

        @client.event
        async def on_resumed() -> None:  # noqa: ANN001
            self.detail = "resumed"
            log.info("discord session resumed")

        @client.event
        async def on_disconnect(*_args) -> None:  # noqa: ANN001
            self._connected.clear()
            self.detail = "disconnected (auto-reconnecting)"

        @client.event
        async def on_message(message) -> None:  # noqa: ANN001
            try:
                await self._handle_message(message)
            except Exception as exc:  # noqa: BLE001 - one bad message must not kill the loop
                self._errors += 1
                log.exception("discord handler failed: %s", exc)

    def _wants_me(self, message) -> bool:
        """Only act on: not a bot, right channel, and (if unrestricted) a mention,
        a DM or the configured prefix. This is what stops the bot answering every
        message in a busy shared server."""
        if message.author.bot or self._client is None:
            return False
        if getattr(getattr(message, "guild", None), "id", None) and message.channel.id == message.guild.id:
            return False
        text = (message.content or "").strip()
        if not text:
            return False
        if self.channel_id:
            return message.channel.id == self.channel_id
        me = getattr(self._client, "user", None)
        mentioned = bool(me and me.mentioned_in(message))
        dm = type(message.channel).__name__ in {"DMChannel", "GroupChannel"}
        return mentioned or dm or text.lower().startswith((self.prefix or "").lower())

    # ------------------------------------------------------------------ handling
    async def _handle_message(self, message) -> None:
        if not self._wants_me(message):
            return
        self._last_seen = time.time()
        raw = MENTION_RE.sub("", (message.content or "")).strip()
        if raw.lower().startswith(self.prefix.lower()):
            raw = raw[len(self.prefix):].strip()
        if not raw:
            await message.reply(self._help_text())
            return

        lower = raw.lower()
        if lower in {"status", "ping", "help", "?"}:
            await message.reply(self._quick_reply(lower))
            return
        if lower in {"stop", "quiet", "cancel"}:
            self._buffer.clear()
            await message.reply("Stopped. Nothing else queued from Discord.")
            return

        self._handled += 1
        self._emit("in", f"[discord:{message.author}] {raw[:120]}")
        try:
            await message.channel.trigger_typing()
        except Exception:  # noqa: BLE001
            pass

        import server  # local import: server lazily imports this module

        started = time.perf_counter()
        try:
            payload = await server.handle_command(raw, source=f"discord:#{getattr(message.channel, 'name', 'dm')}", speak=False)
        except Exception as exc:  # noqa: BLE001
            payload = {"ok": False, "answer": f"Core error: {type(exc).__name__}: {exc}", "tool_calls": []}
        elapsed = int((time.perf_counter() - started) * 1000)

        answer = str(payload.get("answer") or "No answer.")
        tools_used = [t.get("tool", "?") for t in payload.get("tool_calls") or [] if isinstance(t, dict)]
        footer = f"-# track: {payload.get('track', 'agent')} · {elapsed} ms" + (
            f" · tools: {', '.join(dict.fromkeys(tools_used))}" if tools_used else "")
        chunks = _split_blocks(f"{answer}\n{footer}")

        try:
            if len(answer) > 1200 and SETTINGS.discord_use_threads:
                thread = await message.create_thread(name=f"JARVIS · {raw[:28] or 'reply'}")
                target = thread
            else:
                target = message.channel
            for chunk in chunks:
                await target.send(chunk)
                await asyncio.sleep(0.35)
            try:
                await message.add_reaction("👍")
            except Exception:  # noqa: BLE001 - missing reactions scope is fine
                pass
            self._emit("out", f"[discord] {answer[:140]}")
        except Exception as exc:  # noqa: BLE001
            log.warning("could not deliver discord reply: %s", exc)
            try:
                await message.author.send(answer[:MAX_LEN] or "I could not post to that channel.")
            except Exception:  # noqa: BLE001
                pass

    def _help_text(self) -> str:
        return (
            "**J.A.R.V.I.S. over Discord**\n"
            f"Send anything in <#{self.channel_id}> or mention me. Prefix `{self.prefix}` also works.\n"
            "`status` · `help` · `stop` · `notes <topic>` · `open steam` · `real madrid score` · `cheapest indian restaurants in surat`"
        )

    def _quick_reply(self, lower: str) -> str:
        """Local status/help/ping replies -- no LLM, no tool, ~0 ms."""
        if lower in {"help", "?"}:
            return self._help_text()
        server = import_server()
        if server is None:
            return "Core is not loaded yet, try again in a second."
        if lower == "ping":
            return f"pong · core uptime {int(time.time() - server.BOOT_TS)}s"
        import json as _json

        st = server._STATE
        status = {
            "mode": st.get("mode"),
            "uptime_s": int(time.time() - server.BOOT_TS),
            "hits": {k: v for k, v in st.items() if k in {"commands", "instant_hits", "agent_hits", "last_latency_ms"}},
            "voice": {"stt": voice_state(), "tts": voice_state(tts=True)},
            "bridge": self.status(),
        }
        return "```json\n" + _json.dumps(status, indent=2, default=str)[: MAX_LEN - 20] + "\n```"

    # ------------------------------------------------------------------ outbound
    def send(self, text: str, mention_everyone: bool = False) -> bool:
        """Push a local JARVIS line into the bound channel (thread-safe)."""
        text = (text or "").strip()
        if not text or not self.connected or not self.channel_id or self._loop is None:
            return False
        try:
            asyncio.run_coroutine_threadsafe(self._send_async(text, mention_everyone), self._loop)
            return True
        except RuntimeError as exc:
            log.debug("discord send skipped: %s", exc)
            return False

    async def _send_async(self, text: str, mention_everyone: bool = False) -> None:
        channel = self._client.get_channel(self.channel_id) if self._client else None
        if channel is None:
            try:
                channel = await self._client.fetch_channel(self.channel_id)
            except Exception as exc:  # noqa: BLE001
                log.warning("cannot resolve discord channel %s: %s", self.channel_id, exc)
                return
        for chunk in _split_blocks(text)[:8]:
            try:
                await channel.send(chunk, allowed_mentions=discord_allowed(mention_everyone))
            except Exception as exc:  # noqa: BLE001
                log.warning("discord send failed: %s", exc)
                return
            await asyncio.sleep(0.4)

    # ------------------------------------------------------------------ misc
    def _emit(self, level: str, text: str) -> None:
        entry = {"level": level, "text": text, "at": time.strftime("%H:%M:%S")}
        self._buffer.append(entry)
        if len(self._buffer) > self._buffer_max:
            del self._buffer[:-self._buffer_max]
        hub = self.on_event
        if hub is not None:
            try:
                hub.emit({"type": "log", **entry})
            except Exception:  # noqa: BLE001
                pass

    def recent(self, limit: int = 40) -> List[Dict[str, Any]]:
        return self._buffer[-limit:]

    def stop(self) -> None:
        self._stop.set()
        loop, client = self._loop, self._client
        if client is not None and loop is not None and loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(client.close(), loop)
            except RuntimeError:
                pass
        self._connected.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def import_server():
    """Late import so this module can be imported by ``server.py``'s lifespan."""
    try:
        import server

        return server
    except Exception:  # noqa: BLE001 - status fallback when server is not loaded yet
        return None


def discord_allowed(everyone: bool):
    import discord

    if everyone:
        return discord.AllowedMentions.all()
    return discord.AllowedMentions(everyone=False, roles=False, users=True)


def voice_state(tts: bool = False) -> str:
    try:
        import audio_engine

        engine = audio_engine.ENGINE
        return engine.tts.status["state"] if tts else engine.stt.status["state"]
    except Exception:  # noqa: BLE001
        return "unknown"


def _split_blocks(text: str, limit: int = MAX_LEN) -> List[str]:
    '''Split on paragraph -> sentence -> hard slice boundaries.

    The budget is ``limit - 8`` so appending a closing code fence or an ellipsis
    can never push a chunk past Discord's 2000 character hard cap.'''
    """Chunk long answers on paragraph/line boundaries, keeping ``` fences paired."""
    budget = max(200, limit - 8)          # room for the fence / ellipsis below
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    blocks: List[str] = []
    for paragraph in text.split("\n\n"):
        for chunk in filter(None, [paragraph]):
            if len(chunk) <= budget:
                blocks.append(chunk)
                continue
            pieces: List[str] = []
            for sentence in re.split(r"(?<=[.!?])\s+", chunk):
                if len(sentence) <= budget:
                    pieces.append(sentence)
                    continue
                for i in range(0, len(sentence), budget):
                    pieces.append(sentence[i : i + budget])
            blocks.extend(pieces)

    chunks: List[str] = []
    current = ""
    for block in blocks:
        candidate = f"{current}\n\n{block}".strip() if current else block
        if len(candidate) > budget:
            if current:
                chunks.append(current)
            current = block[:budget]
        else:
            current = candidate
    if current:
        chunks.append(current)

    # Re-balance unbalanced code fences per chunk so Discord never renders junk.
    fixed: List[str] = []
    for chunk in chunks:
        if chunk.count(CODE_FENCE) % 2:
            chunk = (chunk + "\n" + CODE_FENCE)[:limit]
        fixed.append(chunk[:limit])
    return fixed


BRIDGE: Optional[DiscordBridge] = None


def start_bridge(hub: Optional[Any] = None, channel_id: Optional[int] = None) -> DiscordBridge:
    global BRIDGE
    if BRIDGE is None:
        BRIDGE = DiscordBridge(channel_id=channel_id, on_event=hub)
    elif hub is not None:
        BRIDGE.on_event = hub
    return BRIDGE.start()


def get_bridge() -> Optional[DiscordBridge]:
    return BRIDGE


__all__ = ["DiscordBridge", "start_bridge", "get_bridge", "BRIDGE"]
