"""
tests/test_jarvis.py -- self-checks for the JARVIS core.

Stdlib ``unittest`` only, so it runs on a bare Windows box without pytest:

    python -m unittest discover -s tests -v
    python -m unittest tests.test_jarvis.TestInstantTrack -v

The HTTP/REST cases use FastAPI's ``TestClient`` and skip themselves if
``httpx`` is not installed (FastAPI's TestClient needs it; the provider layer
so a normal install has it).
"""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime
from unittest import mock
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("JARVIS_WARM", "0")     # never load 1 GB models in CI
os.environ.setdefault("JARVIS_HEADLESS", "1")
os.environ.setdefault("TTS_ENABLED", "false")

import config  # noqa: E402
import discord_bridge  # noqa: E402
import router  # noqa: E402
import files  # noqa: E402
import screen  # noqa: E402
import wake  # noqa: E402
import reminders  # noqa: E402
import llm_providers  # noqa: E402
import winops  # noqa: E402
import server  # noqa: E402
import tools  # noqa: E402
from audio_engine import ENGINE as voice  # noqa: E402
from audio_engine import Transcript, pcm16_to_wav  # noqa: E402

try:  # optional HTTP layer
    from fastapi.testclient import TestClient

    HTTP_OK = True
except Exception:  # noqa: BLE001
    HTTP_OK = False


class TestConfig(unittest.TestCase):
    def test_env_file_parsing(self):
        tmp = ROOT / "data" / "_test.env"
        tmp.write_text(
            "# comment\nexport GROQ_API_KEY = 'gsk-secret' \nNOTES_DIR=notes\n\nQUOTED=\"has = sign\"\n",
            encoding="utf-8",
        )
        parsed = config.parse_env_file(tmp)
        tmp.unlink()
        self.assertEqual(parsed["GROQ_API_KEY"], "gsk-secret")
        self.assertEqual(parsed["NOTES_DIR"], "notes")
        self.assertEqual(parsed["QUOTED"], "has = sign")

    def test_redacted_never_leaks_secrets(self):
        blob = json.dumps(config.SETTINGS.redacted())
        for secret in (config.SETTINGS.groq_api_key, config.SETTINGS.cloudflare_api_token,
                       config.SETTINGS.discord_token, config.SETTINGS.api_sports_key):
            if secret:
                self.assertNotIn(secret, blob, "a raw secret reached the HUD-safe view")
        self.assertIn("llm_keys_set", blob)
        self.assertNotIn("gsk-secret", blob)
        self.assertNotIn("api_key", blob)

    def test_paths_are_absolute(self):
        self.assertTrue(config.SETTINGS.notes_path.is_absolute())
        self.assertTrue(config.SETTINGS.reference_wav_path.is_absolute())


class TestInstantTrack(unittest.TestCase):
    """Track 1 must be high precision: claim what it can execute, no more."""

    CASES = {
        "open steam": "open",
        "jarvis, launch discord": "open",
        "open youtube": "open",
        "close chrome": "close",
        "what's the time": "time",
        "take a screenshot": "screenshot",
        "volume up": "volume",
        "set volume to 40": "volume",
        "mute": "volume",
        "how did real madrid do": "sports",
        "read my notes about german dative": "note-read",
        "note that the exam is friday": "note-write",
        "how is my cpu": "telemetry",
        "cancel shutdown": "power",
        "stop": "stop",
        "hello": "greeting",
    }

    def test_patterns_route(self):
        for text, expected in self.CASES.items():
            with self.subTest(text=text):
                plan = server.match_instant(text)
                self.assertIsNotNone(plan, f"{text!r} produced no plan")
                self.assertEqual(plan.get("rule"), expected)

    def test_questions_are_deferred_to_the_model(self):
        """Track 1 must not steal "what is X" for DuckDuckGo - that is the AI's job."""
        for utterance in ("what is the capital of France", "explain the german dative case",
                          "who is the best midfielder right now", "summarise the plot of the odyssey"):
            self.assertIsNone(server.match_instant(utterance), f"{utterance!r} should reach the agent")

    def test_unknown_app_is_deferred_to_the_agent(self):
        self.assertIsNone(server.match_instant("open the next big thing"))
        self.assertIsNone(server.match_instant("explain the difference between a coroutine and a thread"))
        # ...but a factual "what is X" question is a search, which Track 1 owns.
        self.assertIsNone(server.match_instant("what is the capital of France"))

    def test_multi_command_split(self):
        plan = server.match_instant("open steam and check my cpu")
        names = [call["tool"] for call in plan["calls"]]
        self.assertIn("launch_app", names)
        self.assertIn("system_report", names)

    def test_notes_plus_telemetry_split(self):
        plan = server.match_instant("read my notes about german dative and how is my cpu")
        names = [call["tool"] for call in plan["calls"]]
        self.assertEqual(names[:2], ["read_notes", "system_report"])
        self.assertEqual(plan["calls"][0]["arguments"]["topic"], "german dative")

    def test_dictation_tail_preserves_casing(self):
        plan = server.match_instant("open eden and note that FC 26 needs firmware 18.1.0")
        write = [c for c in plan["calls"] if c["tool"] == "write_note"]
        self.assertEqual(len(write), 1, "the trailing clause must be captured as a note, not a lookup")
        self.assertIn("FC 26", write[0]["arguments"]["content"])
        self.assertIn("18.1.0", write[0]["arguments"]["content"])

    def test_unparsed_clause_is_reported_not_guessed(self):
        plan = server.match_instant("open steam and sing the national anthem loudly")
        self.assertEqual([c["tool"] for c in plan["calls"]], ["launch_app"])
        self.assertIn("could not map", plan["answer_prefix"])

    def test_wake_word_stripped(self):
        self.assertEqual(server.strip_wake("Hey Jarvis,  open steam "), "open steam")
        self.assertEqual(server.strip_wake("open steam"), "open steam")

    def test_destructive_action_requires_confirmation(self):
        plan = server.match_instant("shutdown the pc")
        self.assertTrue(plan.get("confirm"), "shutdown must ask before acting")

        import asyncio

        try:
            first = asyncio.run(server.handle_command("shutdown the pc", source="test", speak=False))
            self.assertIn("confirm", first["answer"].lower())
            self.assertTrue(server._STATE["pending_confirm"], "the action should be armed")
            second = asyncio.run(server.handle_command("cancel", source="test", speak=False))
            self.assertIn("cancelled", second["answer"].lower())
            self.assertIsNone(server._STATE["pending_confirm"])
        finally:
            server._STATE["pending_confirm"] = None


class TestToolSchemas(unittest.TestCase):
    def test_every_schema_tool_exists(self):
        for name in router.TOOL_NAMES:
            self.assertIn(name, tools.TOOL_FUNCTIONS, f"{name} has no implementation")

    def test_schemas_are_strict(self):
        for schema in router.TOOL_SCHEMAS:
            params = schema["function"]["parameters"]
            self.assertEqual(params["type"], "object")
            self.assertFalse(params["additionalProperties"], "strict mode requires additionalProperties=false")
            self.assertEqual(set(params["required"]), set(params["properties"]),
                             f"{schema['function']['name']}: strict mode needs every property required")
            for key, spec in params["properties"].items():
                self.assertIn(spec["type"], {"string", "integer", "boolean", "number"})
                self.assertTrue(spec.get("description"), f"{schema['function']['name']}.{key} lacks a description")

    def test_validate_call_normalises(self):
        tool, args, err = router.validate_call("web_search", {"query": "  Surat cafes ", "max_results": "5", "evil": 1})
        self.assertEqual(err, "")
        self.assertEqual(args, {"query": "Surat cafes", "max_results": 5})

    def test_validate_call_rejects(self):
        self.assertTrue(router.validate_call("rm -rf", {})[2])
        self.assertTrue(router.validate_call("launch_app", {"app_name": "  "})[2])
        self.assertTrue(router.validate_call("fetch_sports_stats", {"team": "x", "kind": "nope"})[2])
        self.assertTrue(router.validate_call("set_volume", {"level": -1, "delta": 0, "mute": False})[2])

    def test_execute_tool_never_raises(self):
        self.assertFalse(tools.execute_tool("nope", {})["ok"])
        self.assertTrue(tools.execute_tool("get_time", {})["ok"])
        self.assertFalse(tools.execute_tool("web_search", {"query": ""})["ok"])

    def test_heuristic_planner_covers_domains(self):
        for text, expected in {
            "open spotify": "launch_app",
            "whats the score of real madrid": "fetch_sports_stats",
            "how is my cpu load": "system_report",
            "what time is it": "get_time",
        }.items():
            with self.subTest(text=text):
                self.assertIn(expected, [c["tool"] for c in router.heuristic_plan(text)])

    def test_json_tool_call_from_prose(self):
        call = router.parse_tool_call_from_text('{"tool": "launch_app", "arguments": {"app_name": "Steam"}, "thought": "user asked"}')
        self.assertEqual(call["name"], "launch_app")
        self.assertEqual(call["arguments"], {"app_name": "Steam"})
        self.assertIsNone(router.parse_tool_call_from_text("I will just chat instead"))


class TestNotes(unittest.TestCase):
    def test_read_notes_finds_german_dative(self):
        result = tools.read_notes("german dative", limit=1)
        self.assertTrue(result["ok"])
        self.assertIn("German", result["notes"][0]["title"])

    def test_scoring_prefers_filename_match(self):
        scored = tools.read_notes("cs degree prep", limit=3)
        self.assertTrue(scored["ok"])
        self.assertIn("CS", scored["notes"][0]["title"])

    def test_topicless_lists_the_catalog(self):
        result = tools.read_notes()
        self.assertTrue(result["ok"])
        self.assertGreaterEqual(result["count"], 2)

    def test_write_then_read_roundtrip(self):
        written = tools.write_note("unit test memo", "verification payload for tests", "test")
        self.assertTrue(written["ok"])
        found = tools.read_notes("unit test memo", limit=1)
        self.assertTrue(found["ok"])
        self.assertIn("verification payload", found["notes"][0]["excerpt"])
        Path(written["path"]).unlink(missing_ok=True)

    def test_missing_topic_message_is_helpful(self):
        result = tools.read_notes("quantum tunneling in semiconductors")
        self.assertFalse(result["ok"])
        self.assertIn("Nothing in your notes matches", result["message"])


class TestWindowsTools(unittest.TestCase):
    def test_catalog_has_the_required_apps(self):
        apps = set(tools.available_apps())
        for expected in {"steam", "discord", "eden", "chrome", "edge", "firefox", "code", "terminal"}:
            self.assertIn(expected, apps)

    def test_is_known_app_precision(self):
        self.assertTrue(tools.is_known_app("Discord"))
        self.assertTrue(tools.is_known_app("steam"))
        self.assertFalse(tools.is_known_app("definitely-not-installed-xyz"))

    def test_alias_resolution(self):
        self.assertEqual(tools._match_catalog_app("launch the steam client"), "steam")
        self.assertEqual(tools._match_catalog_app("fc 26"), "eden")
        self.assertEqual(tools._match_catalog_app("visual studio code"), "code")

    def test_launch_unknown_reports_how_to_fix_it(self):
        result = tools.launch_app("definitely-not-installed-xyz")
        self.assertFalse(result["ok"])
        self.assertIn("CUSTOM_APPS", result["message"])

    def test_play_youtube_is_graceful_without_network(self):
        # No network in CI: the tool must still answer with a dict and echo the query.
        result = tools.play_youtube("lofi beats")
        self.assertIsInstance(result, dict)
        self.assertIn("lofi beats", json.dumps(result))

    def test_system_report_shape(self):
        report = tools.system_report()
        for key in ("cpu", "ram", "disk", "per_core", "gpu", "top_processes", "message"):
            self.assertIn(key, report)
        self.assertTrue(0 <= report["cpu"] <= 100)

    def test_sports_without_key_fails_loudly(self):
        if config.SETTINGS.api_sports_key:
            self.skipTest("an API_SPORTS_KEY is configured, error path not applicable")
        result = tools.fetch_sports_stats("Real Madrid")
        self.assertFalse(result["ok"])
        self.assertIn("API_SPORTS_KEY", result["message"])

    def test_fixture_outcome_math(self):
        fixture = {"goals": {"home": 3, "away": 1}, "teams": {"home": {"id": 7}}, "fixture": {"status": {"short": "FT"}}}
        self.assertEqual(tools._outcome(fixture, 7), "W")
        self.assertEqual(tools._outcome(fixture, 8), "L")
        fixture["goals"] = {"home": 2, "away": 2}
        self.assertEqual(tools._outcome(fixture, 7), "D")


class TestWindowsCommands(unittest.TestCase):
    """Assert the exact commands Track 1/2 would run on the target machine.

    ``config.is_windows`` and ``tools._run`` are patched, so nothing is executed;
    what is verified is the real PowerShell/subprocess construction.
    """

    def setUp(self):
        self.calls = []
        self._real_run = tools._run
        self._real_win = config.is_windows

        def fake_run(cmd, timeout=20, env=None):      # records argv, reports success
            self.calls.append(list(cmd))
            return 0, "", ""

        tools._run = fake_run
        config.is_windows = lambda: True
        tools.config.is_windows = lambda: True

    def tearDown(self):
        tools._run = self._real_run
        config.is_windows = self._real_win
        tools.config.is_windows = self._real_win

    def test_absolute_volume_zeroes_then_climbs(self):
        result = tools.set_volume(level=50)
        self.assertTrue(result["ok"])
        script = self.calls[-1][-1]
        self.assertIn("{VOLUME_MUTE}", script)                 # unmute first
        self.assertIn("$i -lt 50;$i++){$s::SendKeys('{VOLUME_DOWN}')", script)   # to zero
        self.assertIn("$i -lt 25;$i++){$s::SendKeys('{VOLUME_UP}')", script)     # 25 x 2 % = 50 %

    def test_relative_volume_direction(self):
        self.assertTrue(tools.set_volume(delta=-4)["ok"])
        self.assertIn("{VOLUME_DOWN}", self.calls[-1][-1])
        self.assertTrue(tools.set_volume(delta=3)["ok"])
        self.assertIn("{VOLUME_UP}", self.calls[-1][-1])

    def test_screenshot_powershell_invocation(self):
        result = tools.take_screenshot()
        self.assertIn("CopyFromScreen", self.calls[-1][-1])
        self.assertFalse(result["ok"])      # the fake run created no PNG -> reported honestly

    def test_lock_uses_rundll32(self):
        self.calls.clear()
        result = tools.system_power("lock")
        self.assertTrue(result["ok"])
        self.assertEqual(self.calls[-1][:2], ["rundll32.exe", "user32.dll,LockWorkStation"])

    def test_restart_uses_shutdown_with_delay(self):
        self.calls.clear()
        tools.system_power("restart")
        self.assertIn("/t", self.calls[-1])
        self.assertIn("30", self.calls[-1])


class TestVoiceEngine(unittest.TestCase):
    def test_transcript_confidence_band(self):
        confident = Transcript(text="open steam", avg_logprob=-0.05, no_speech_prob=0.01)
        quiet = Transcript(text="", avg_logprob=-1.2, no_speech_prob=0.9)
        self.assertGreater(confident.confidence, 0.5)
        self.assertEqual(quiet.confidence, 0.0)

    def test_pcm_roundtrip_and_vad_gate(self):
        path = pcm16_to_wav(b"\x00\x00\x10\x27" * 8000, 16000)
        try:
            with wave.open(str(path), "rb") as handle:
                self.assertEqual(handle.getframerate(), 16000)
                self.assertEqual(handle.getnchannels(), 1)
            from audio_engine import rms_of_wav

            self.assertGreater(rms_of_wav(path), 0.0)
        finally:
            path.unlink(missing_ok=True)

    def test_speech_script_sanitiser(self):
        cleaned = voice.tts._prepare_script("**CPU** 42.5% at https://example.com/x `code`\n\nlines")
        self.assertNotIn("**", cleaned)
        self.assertNotIn("`", cleaned)
        self.assertIn("percent", cleaned)
        self.assertIn("link", cleaned)

    def test_missing_reference_is_reported_not_crashing(self):
        ok, detail = voice.tts.check_reference()
        self.assertIsInstance(ok, bool)
        self.assertTrue(detail)

    def test_stt_status_reports_the_gpu_configuration(self):
        status = voice.stt.status
        self.assertIn(status["compute_type"], {"int8", "float16", "float32"})
        self.assertIn(status["state"], {"unloaded", "loading", "ready", "failed"})


class TestMemoryAndDiscord(unittest.TestCase):
    def test_memory_is_bounded(self):
        memory = router.ConversationMemory(max_turns=6)
        for i in range(40):
            memory.add("user", f"turn {i}")
        self.assertLessEqual(len(memory.snapshot()), 6)
        self.assertIn("turn 39", json.dumps(memory.snapshot()))

    def test_last_tool_tracks_the_latest_call(self):
        memory = router.ConversationMemory()
        memory.add("assistant", "done", tool="web_search")
        self.assertEqual(memory.last_tool(), "web_search")

    def test_discord_chunking_keeps_fences_balanced(self):
        text = "```json\n" + ("alpha bravo charlie delta echo " * 90) + "\n```"
        chunks = discord_bridge._split_blocks(text)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk.count("```") % 2, 0, "unbalanced code fence would render as junk")
            self.assertLessEqual(len(chunk), discord_bridge.MAX_LEN)

    def test_bridge_disabled_without_token(self):
        bridge = discord_bridge.DiscordBridge(channel_id=0)
        self.assertEqual(bridge.connected, False)
        self.assertIn("detail", bridge.status())


@unittest.skipUnless(HTTP_OK, "httpx/fastapi TestClient unavailable")
class TestHttpApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(server.app)

    def test_health_and_root(self):
        self.assertEqual(self.client.get("/healthz").json()["ok"], True)
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("arc_reactor.js", response.text)
        self.assertIn("cdn.tailwindcss.com", response.text)

    def test_static_assets(self):
        for path in ("/static/styles.css", "/static/arc_reactor.js"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertGreater(len(response.content), 500, path)

    def test_command_endpoint_instant_track(self):
        response = self.client.post("/api/command", json={"text": "what's the time", "speak": False})
        payload = response.json()
        self.assertEqual(payload["track"], "instant")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["tool_calls"][0]["tool"], "get_time")
        self.assertLess(payload["latency_ms"], 400)

    def test_command_endpoint_agent_track(self):
        response = self.client.post("/api/command", json={"text": "summarise my german notes", "speak": False, "agent": True})
        payload = response.json()
        self.assertIn("track", payload)
        self.assertTrue(payload.get("answer"))

    def test_status_endpoints(self):
        for path in ("/api/status", "/api/config", "/api/telemetry", "/api/history", "/api/apps", "/api/discord/status"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

    def test_notes_endpoint_and_card(self):
        payload = self.client.post("/api/command", json={"text": "read my notes about german dative", "speak": False}).json()
        self.assertTrue(payload.get("card"))
        self.assertEqual(payload["card"]["type"], "notes")

    def test_empty_command_rejected(self):
        self.assertFalse(self.client.post("/api/command", json={"text": "   "}).json()["ok"])

    def test_transcribe_endpoint_without_speech(self):
        # A one-second silent WAV must come back as "no speech", not a 500.
        import struct

        frames = b"".join(struct.pack("<h", 0) for _ in range(16000))
        import io

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(frames)
        import base64

        response = self.client.post(
            "/api/transcribe",
            json={"audio_base64": base64.b64encode(buffer.getvalue()).decode(), "format": "wav", "speak": False},
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["ok"])


class TestTranscribeEndpoint(unittest.TestCase):
    """The audio-in route must not require FastAPI's optional multipart extra."""

    def test_server_module_has_no_multipart_params(self):
        source = (config.ROOT / "server.py").read_text(encoding="utf-8")
        self.assertNotIn("UploadFile", source)
        self.assertNotIn("File(...)", source)
        self.assertNotIn("Form(", source)
        self.assertIn("python-multipart", source, "the reason for JSON-only uploads should be documented")

    def test_bad_base64_reports_instead_of_crashing(self):
        from fastapi.testclient import TestClient
        client = TestClient(server.create_app())
        response = client.post("/api/transcribe", json={"audio_base64": "!!not base64!!"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["ok"])
        self.assertIn("base64", body["error"])
        missing = client.post("/api/transcribe", json={}).json()
        self.assertFalse(missing["ok"])
        self.assertIn("audio_base64", missing["error"])


class TestJsApiSurface(unittest.TestCase):
    """pywebview recursively walks every public attribute of the js_api object."""

    class Endless:
        """Mimics pythonnet's Rectangle.Empty -> .Empty -> .Empty chain.

        ``Empty`` is a real property (as it is on the .NET struct) so ``dir()``
        exposes it and the recursive walk keeps descending.
        """

        @property
        def Empty(self):  # noqa: N802 - mirroring the .NET member name
            return TestJsApiSurface.Endless()

    @staticmethod
    def _walk(obj, base_name="", seen=None, found=None):
        """A copy of webview.util.get_functions' traversal rules."""
        import inspect

        seen = [] if seen is None else seen
        found = {} if found is None else found
        if id(obj) in seen:
            return found
        seen.append(id(obj))
        for name in dir(obj):
            full = f"{base_name}.{name}" if base_name else name
            if name.startswith("_"):
                continue
            attr = getattr(obj, name)
            if not getattr(attr, "_serializable", True):
                continue
            if inspect.ismethod(attr) or inspect.isfunction(attr):
                found[full] = "method"
            elif inspect.isclass(attr) or (
                isinstance(attr, object) and not callable(attr) and hasattr(attr, "__module__")
            ):
                TestJsApiSurface._walk(attr, full, seen, found)
        return found

    def setUp(self) -> None:
        import main
        self.main = main

    def test_bridge_public_surface_is_only_methods(self):
        bridge = self.main.JarvisBridge(8760, window=self.Endless())
        public = [name for name in dir(bridge) if not name.startswith("_")]
        self.assertTrue(public)
        for name in public:
            self.assertTrue(callable(getattr(bridge, name)), f"{name} must be callable or pywebview will recurse into it")
        self.assertNotIn("window", public)
        self.assertNotIn("port", public)

    def test_pywebview_walk_terminates_on_the_bridge(self):
        bridge = self.main.JarvisBridge(8760, window=self.Endless())
        found = self._walk(bridge)
        self.assertEqual(set(found.values()), {"method"})
        self.assertIn("command", found)

    def test_an_exposed_window_object_would_infinite_recurse(self):
        """Negative control: the bug this class guards against is real."""
        class BadBridge:
            def __init__(self, window):
                self.window = window      # exactly what main.py used to do

        with self.assertRaises(RecursionError):
            self._walk(BadBridge(self.Endless()))

    def test_destroy_is_preferred_over_the_missing_close(self):
        calls = []
        window = type("W", (), {"destroy": lambda self: calls.append("destroy")})()
        bridge = self.main.JarvisBridge(8760, window=window)
        bridge._shutdown()
        self.assertEqual(calls, ["destroy"], "pywebview 5.x Window has destroy(), not close()")

    def test_unsupported_control_reports_instead_of_raising(self):
        bridge = self.main.JarvisBridge(8760, window=type("W", (), {})())
        result = bridge.toggle_frameless()
        self.assertFalse(result["ok"])
        self.assertIn("supports none of", result["error"])
        self.assertFalse(bridge.minimize()["ok"])
        bridge_no_window = self.main.JarvisBridge(8760)
        self.assertIn("no window", bridge_no_window.maximize()["error"])

    def test_set_topmost_survives_a_backend_without_the_property(self):
        bridge = self.main.JarvisBridge(8760, window=object())   # plain object: no on_top setter
        result = bridge.set_topmost(True)
        self.assertIsInstance(result["ok"], bool)                 # reported, never raised


class EnvOnly:
    """Make ``config.dotenv`` read ``os.environ`` only, until the test ends.

    ``dotenv`` answers ``os.environ`` first and the ``.env`` file second, and a blank environment
    value counts as unset.  That is right for a person and fatal for a test: on the only machine
    that matters - the user's laptop, whose ``.env`` holds a real Groq key - "no key at all" and
    "this key only" cannot be expressed by deleting environment variables, so the file has to come
    out of the reader's view.  ``addCleanup`` restores it even when the test raises.
    """

    @staticmethod
    def install(test):
        real = config.dotenv
        config.dotenv = lambda key, default="": os.environ.get(key, default)
        test.addCleanup(setattr, config, "dotenv", real)


class TestLlmProviders(unittest.TestCase):
    """Pool rotation, quota cooldowns and tier selection -- all offline."""

    def setUp(self):
        EnvOnly.install(self)          # .env on this box may hold real keys

        import llm_providers as lp

        self.lp = lp
        self._real_pool = lp.POOL
        self.pool = lp.LlmPool(state_file=Path(self._tmp_state()))
        # A deterministic key set: two providers configured, the rest not.
        self._env_backup = {}
        for name, value in (("GROQ_API_KEY", "gsk-test"), ("CEREBRAS_API_KEY", "cbs-test")):
            self._remember(name)
            os.environ[name] = value
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "MISTRAL_API_KEY", "OPENROUTER_API_KEY",
                     "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_API_KEY", "CLOUDFLARE_ACCOUNT_ID",
                     "GITHUB_MODELS_TOKEN", "GITHUB_TOKEN", "GH_TOKEN", "HF_TOKEN"):
            self._remember(name)
            os.environ.pop(name, None)
        lp.POOL = self.pool               # keep the singleton off the real state file

    def _remember(self, name: str) -> None:
        self._env_backup.setdefault(name, os.environ.get(name))

    def _tmp_state(self):
        self._dir = tempfile.mkdtemp(prefix="jarvis-llm-")
        return str(Path(self._dir) / "llm_state.json")

    def tearDown(self):
        for name, value in self._env_backup.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self.lp.POOL = self._real_pool
        shutil.rmtree(self._dir, ignore_errors=True)

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _ok(content="", tool_calls=None):
        message = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return 200, {}, {"choices": [{"message": message, "finish_reason": "stop"}]}

    def _stub(self, replies):
        """Patch the HTTP seam; ``replies`` maps provider key -> (status, headers, body)."""
        calls = []

        def fake_post(url, headers, payload, timeout):
            calls.append({"url": url, "auth": headers.get("authorization", ""), "payload": payload})
            for key, spec in self.lp.PROVIDERS.items():
                if spec.root and spec.root in url:
                    reply = replies.get(key, self._ok("fallback text"))
                    return reply if isinstance(reply, tuple) and len(reply) == 3 else (200, {}, reply)
            return self._ok("other")

        self.lp._post_json = fake_post            # the single network seam
        return calls

    # -- tiering -----------------------------------------------------------
    def test_tier_choice_favours_the_cheap_model(self):
        self.assertEqual(self.lp.choose_tier("open steam"), "fast")
        self.assertEqual(self.lp.choose_tier("what is the time"), "fast")
        self.assertEqual(self.lp.choose_tier(
            "compare the dative and accusative cases, then write me a study plan for the exam"), "smart")
        self.assertEqual(self.lp.choose_tier("hello"), "fast")

    def test_research_heuristic_rejects_chatter(self):
        self.assertFalse(self.lp.looks_like_research("hello there"))
        self.assertFalse(self.lp.looks_like_research("explain recursion"))
        self.assertTrue(self.lp.looks_like_research("what is the latest transfer news"))
        self.assertTrue(self.lp.looks_like_research("who won the match today"))

    # -- rotation ----------------------------------------------------------
    def test_rotates_to_the_next_provider_on_429(self):
        calls = self._stub({
            "groq": (429, {}, {"error": {"message": "Rate limit reached for requests per day"}}),
            "cerebras": self._ok("answered by cerebras"),
        })
        out = self.pool.complete([{"role": "user", "content": "hi"}], tier="fast")
        self.assertEqual(out["provider"], "cerebras")
        self.assertEqual(out["content"], "answered by cerebras")
        self.assertEqual({c["url"].split("/")[2] for c in calls}, {"api.groq.com", "api.cerebras.ai"})

    def test_daily_quota_cooldown_survives_until_the_reset_and_persists(self):
        self._stub({"groq": (429, {}, {"error": {"message": "daily quota exceeded"}}),
                    "cerebras": self._ok("fine")})
        self.pool.complete([{"role": "user", "content": "hi"}])
        cooling, left, reason = self.pool.cooling("groq")
        self.assertTrue(cooling)
        self.assertGreater(left, 60, "a daily quota must not be retried immediately")
        self.assertLessEqual(left, 24 * 3600, "and never longer than LLM_COOLDOWN_HOURS")
        self.assertIn("quota", reason)
        self.assertTrue(Path(self.pool.state_file).is_file(), "cooldowns must survive a restart")
        reloaded = self.lp.LlmPool(state_file=self.pool.state_file)
        self.assertTrue(reloaded.cooling("groq")[0])

    def test_retry_after_header_shortens_the_cooldown(self):
        self._stub({"groq": (429, {"retry-after": "7"}, {"error": {"message": "too many requests"}}),
                    "cerebras": self._ok("fine")})
        self.pool.complete([{"role": "user", "content": "hi"}])
        _cooling, left, _reason = self.pool.cooling("groq")
        self.assertLessEqual(left, 120,
                         "Retry-After: 7 must not become a 24h exile (the pool floors it at ~30s)")

    def test_auth_failure_exiles_the_provider(self):
        self._stub({"groq": (401, {}, {"error": {"message": "invalid api key"}}),
                    "cerebras": self._ok("fine")})
        self.pool.complete([{"role": "user", "content": "hi"}])
        self.assertTrue(self.pool.cooling("groq")[0])
        self.assertIn("auth", self.pool.cooling("groq")[2])

    def test_reset_brings_providers_back(self):
        self._stub({"groq": (429, {}, {"error": {"message": "rate limit"}}), "cerebras": self._ok("x")})
        self.pool.complete([{"role": "user", "content": "hi"}])
        self.assertTrue(self.pool.cooling("groq")[0])
        self.pool.reset("groq")
        self.assertFalse(self.pool.cooling("groq")[0])

    def test_cooling_provider_is_skipped_without_a_request(self):
        calls = self._stub({"groq": (429, {}, {"error": {"message": "daily quota"}}), "cerebras": self._ok("x")})
        self.pool.complete([{"role": "user", "content": "hi"}])
        before = len(calls)
        self.pool.complete([{"role": "user", "content": "hi again"}])
        self.assertFalse(any("api.groq.com" in c["url"] for c in calls[before:]),
                         "a cooled-down provider must not be poked every turn")

    def test_missing_keys_say_so_instead_of_searching(self):
        for name in list(self._env_backup):
            os.environ.pop(name, None)
        with self.assertRaises(self.lp.NoProviderConfigured):
            self.pool.complete([{"role": "user", "content": "hi"}])

    def test_native_tool_calls_are_parsed(self):
        self._stub({"groq": self._ok("", [{"id": "c1", "function": {
            "name": "system_report", "arguments": json.dumps({"detailed": True})}}])})
        out = self.pool.complete([{"role": "user", "content": "stats"}], tools=[{"type": "function"}])
        self.assertEqual(out["tool_calls"][0]["name"], "system_report")
        self.assertEqual(out["tool_calls"][0]["arguments"], {"detailed": True})

    def test_stringified_arguments_and_usage_are_tolerated(self):
        self._stub({"groq": (200, {}, {"choices": [{"message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "x", "function": {"name": "get_time", "arguments": "{}"}}]}}],
            "usage": {"total_tokens": 12}})})
        out = self.pool.complete([{"role": "user", "content": "time"}])
        self.assertEqual(out["tool_calls"][0]["name"], "get_time")
        self.assertEqual(out["usage"]["total_tokens"], 12)

    def test_cloudflare_uses_account_url_and_bearer_token(self):
        os.environ["CLOUDFLARE_API_TOKEN"] = "cf-token"
        os.environ["CLOUDFLARE_ACCOUNT_ID"] = "acc-123"
        try:
            calls = self._stub({"groq": (429, {}, {"error": {"message": "daily quota"}}),
                                "cerebras": (429, {}, {"error": {"message": "daily quota"}}),
                                "cloudflare": self._ok("cf says hi")})
            pool = self.lp.LlmPool(state_file=Path(self._tmp_state()))
            out = pool.complete([{"role": "user", "content": "hi"}], tier="fast")
            self.assertEqual(out["content"], "cf says hi")
            self.assertIn("accounts/acc-123/ai/run/@cf/meta/", calls[-1]["url"])
            self.assertEqual(calls[-1]["auth"], "Bearer cf-token")
        finally:
            os.environ.pop("CLOUDFLARE_API_TOKEN", None)
            os.environ.pop("CLOUDFLARE_ACCOUNT_ID", None)

    def test_custom_endpoint_joins_the_rotation(self):
        os.environ["CUSTOM_LLM_BASE_URL"] = "https://integrate.api.nvidia.com/v1/chat/completions"
        os.environ["CUSTOM_LLM_API_KEY"] = "nvapi-demo"
        os.environ["CUSTOM_LLM_MODEL_FAST"] = "meta/llama-3.1-8b-instruct"
        # config.dotenv() answers os.environ first and the .env file second - and a *blank*
        # environment value counts as unset, which is how "KEY=" in .env.example behaves.  To test
        # "no SMART id at all" (the case where FAST must cover both tiers) the file is taken out of
        # the picture for the duration of the call.
        # (the class mixin already hides the .env file; this keeps the intent explicit)
        try:
            self.lp.ensure_custom()
            spec = self.lp.PROVIDERS["custom"]
            self.assertEqual(spec.base_url, "https://integrate.api.nvidia.com/v1",
                             "a pasted /chat/completions URL must be normalised")
            self.assertEqual(spec.smart_model, "meta/llama-3.1-8b-instruct",
                             "without a SMART id the FAST one covers both tiers")
            self.assertIn("custom", self.pool.configured())
            self.assertEqual(self.lp.order()[0], "custom", "a hand-picked endpoint should be tried first")
            self._stub({"custom": self._ok("nim says hi"),
                        "groq": (429, {}, {"error": {"message": "daily quota"}}),
                        "cerebras": (429, {}, {"error": {"message": "daily quota"}})})
            pool = self.lp.LlmPool(state_file=Path(self._tmp_state()))
            out = pool.complete([{"role": "user", "content": "hi"}], tier="fast")
            self.assertEqual(out["provider"], "custom")
            self.assertEqual(out["content"], "nim says hi")
        finally:
            for name in ("CUSTOM_LLM_BASE_URL", "CUSTOM_LLM_API_KEY", "CUSTOM_LLM_MODEL_FAST"):
                os.environ.pop(name, None)
            self.lp.ensure_custom()
            self.assertNotIn("custom", self.lp.PROVIDERS, "clearing the env must clear the slot")

    def test_local_endpoint_needs_no_key_and_no_native_tools(self):
        os.environ["CUSTOM_LLM_BASE_URL"] = "http://127.0.0.1:11434/v1"
        os.environ["CUSTOM_LLM_TOOLS"] = "false"
        os.environ.pop("CUSTOM_LLM_API_KEY", None)
        try:
            self.lp.ensure_custom()
            spec = self.lp.PROVIDERS["custom"]
            self.assertFalse(spec.supports_tools, "CUSTOM_LLM_TOOLS=false must disable native tools")
            self.assertIn("custom", self.pool.configured(), "a LAN server has no api key")
            sent = {}

            def fake_post(url, headers, payload, timeout):
                sent["url"] = url
                sent["auth"] = headers.get("authorization", "<none>")
                sent["tools"] = "tools" in payload
                return self._ok("hello from ollama")      # _ok() already answers (status, headers, body)

            self.lp._post_json = fake_post
            out = self.lp.LlmPool(state_file=Path(self._tmp_state())).complete(
                [{"role": "user", "content": "hi"}], tier="fast", tools=[{"type": "function"}])
            self.assertEqual(out["content"], "hello from ollama")
            self.assertEqual(sent["url"], "http://127.0.0.1:11434/v1/chat/completions")
            self.assertEqual(sent["auth"], "<none>", "no Authorization header without a key")
            self.assertFalse(sent["tools"], "support=false endpoints must not be given tool schemas")
        finally:
            for name in ("CUSTOM_LLM_BASE_URL", "CUSTOM_LLM_TOOLS"):
                os.environ.pop(name, None)
            self.lp.ensure_custom()

    def test_status_reports_the_pool_not_a_single_model(self):
        self._stub({"groq": self._ok("hi")})
        self.pool.complete([{"role": "user", "content": "hi"}])
        status = self.pool.status()
        for key in ("order", "configured", "available", "providers", "cooldown_hours", "budget_seconds"):
            self.assertIn(key, status)
        self.assertTrue(any(p["key"] == "groq" for p in status["providers"]))

    def test_network_failure_rests_the_pool(self):
        def explode(url, headers, payload, timeout):
            raise ConnectionError("unreachable")

        self.lp._post_json = explode
        with self.assertRaises(self.lp.LlmError):
            self.pool.complete([{"role": "user", "content": "hi"}], tier="fast")
        self.assertGreater(self.pool.status()["offline_rest_s"], 0,
                           "no egress should not be re-probed on every utterance")


class TestSiteAwareIntents(unittest.TestCase):
    """"open google" is a website; "search X on youtube" searches youtube."""

    def test_web_services_beat_the_browser(self):
        plan = server.match_instant("open google")
        self.assertEqual(plan["rule"], "open")
        self.assertEqual(plan["calls"][0]["tool"], "open_website")
        self.assertEqual(plan["calls"][0]["arguments"]["target"], "google")

    def test_real_programs_still_launch(self):
        for phrase in ("open chrome", "open google chrome", "open discord", "open spotify"):
            plan = server.match_instant(phrase)
            self.assertEqual(plan["calls"][0]["tool"], "launch_app", phrase)

    def test_search_on_a_named_site_searches_that_site(self):
        plan = server.match_instant("search LM Arena on Youtube")
        call = plan["calls"][0]
        self.assertEqual(call["tool"], "search_on_site")
        self.assertEqual(call["arguments"]["query"], "LM Arena")
        self.assertEqual(call["arguments"]["site"].lower(), "youtube")

    def test_domain_spelling_works_too(self):
        call = server.match_instant("search llm routers on google.com")["calls"][0]
        self.assertEqual(call["tool"], "search_on_site")
        self.assertEqual(call["arguments"]["query"], "llm routers")

    def test_search_site_for_query_word_order(self):
        call = server.match_instant("search youtube for lofi beats")["calls"][0]
        self.assertEqual(call["tool"], "search_on_site")
        self.assertEqual(call["arguments"]["query"], "lofi beats")

    def test_play_on_another_service_uses_that_service(self):
        call = server.match_instant("play lofi on spotify")["calls"][0]
        self.assertEqual(call["tool"], "search_on_site")
        self.assertEqual(call["arguments"]["site"], "spotify")

    def test_plain_search_still_goes_straight_to_ddg(self):
        call = server.match_instant("search cheapest indian restaurants in surat")["calls"][0]
        self.assertEqual(call["tool"], "web_search")

    def test_llm_status_is_instant(self):
        for phrase in ("llm status", "check the ai providers", "who is your ai"):
            plan = server.match_instant(phrase)
            self.assertEqual(plan["calls"][0]["tool"], "llm_status", phrase)

    def test_search_on_site_tool_reports_a_url(self):
        result = tools.search_on_site("youtube", "LM Arena")
        self.assertEqual(result["search_url"], "https://www.youtube.com/results?search_query=LM+Arena")
        self.assertEqual(result["site"], "youtube")
        self.assertIn("query", result)

    def test_search_on_site_without_a_query_is_not_invented(self):
        result = tools.search_on_site("youtube", "")
        self.assertFalse(result["ok"])

    def test_router_has_the_new_tools_bound(self):
        names = {s["function"]["name"] for s in router.TOOL_SCHEMAS}
        self.assertIn("search_on_site", names)
        self.assertIn("llm_status", names)
        self.assertEqual(len(router.TOOL_SCHEMAS), len(names), "schemas must stay unique")

    def test_llm_endpoints_are_served(self):
        if not HTTP_OK:
            self.skipTest("httpx missing")
        from fastapi.testclient import TestClient

        client = TestClient(server.create_app())
        payload = client.get("/api/llm").json()
        self.assertIn("order", payload)
        self.assertIn("providers", payload)
        self.assertEqual(client.post("/api/llm/reset", json={"provider": "nope"}).json()["ok"], False)





class TestAgentTrack(unittest.TestCase):
    """Track 2 must actually call a provider, run the tool, then speak a summary."""

    def setUp(self):
        EnvOnly.install(self)          # .env on this box may hold real keys

        import llm_providers as lp

        self.lp = lp
        self._pool_singleton = lp.POOL
        self._router_pool = router.POOL
        self._post = lp._post_json
        self._keys = {}
        tmp = tempfile.mkdtemp(prefix="jarvis-agent-")
        self._dir = tmp
        self.pool = lp.LlmPool(state_file=Path(tmp) / "llm.json")
        lp.POOL = self.pool
        router.POOL = self.pool          # run_agent looks the name up on its own module
        for name in ("CEREBRAS_API_KEY", "GEMINI_API_KEY", "MISTRAL_API_KEY", "OPENROUTER_API_KEY",
                     "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID", "GITHUB_MODELS_TOKEN"):
            self._keys[name] = os.environ.get(name)
            os.environ.pop(name, None)
        self._keys["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY")
        os.environ["GROQ_API_KEY"] = "gsk-test"

    def tearDown(self):
        for name, value in self._keys.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self.lp._post_json = self._post
        self.lp.POOL = self._pool_singleton
        router.POOL = self._router_pool
        shutil.rmtree(self._dir, ignore_errors=True)

    @staticmethod
    def _content(text):
        return {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}

    def test_tool_call_then_spoken_summary(self):
        seen = []

        def fake_post(url, headers, payload, timeout):
            seen.append(payload)
            if "tools" in payload and payload.get("tools"):
                return 200, {}, {"choices": [{"message": {
                    "role": "assistant", "content": "", "tool_calls": [
                        {"id": "c1", "type": "function", "function": {
                            "name": "system_report", "arguments": json.dumps({"detailed": False})}}]},
                    "finish_reason": "tool_calls"}]}
            return 200, {}, self._content("CPU is at 12 percent and memory looks fine, sir.")

        self.lp._post_json = fake_post
        result = router.ROUTER.run_agent("how is my cpu doing")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.track, "agent")
        self.assertEqual(result.tool_calls[0]["tool"], "system_report")
        self.assertEqual(result.answer, "CPU is at 12 percent and memory looks fine, sir.")
        self.assertEqual(result.model.split(":")[0], "groq")
        self.assertEqual(len(seen), 2, "the summary must be a second, tools-less call")
        self.assertNotIn("tools", seen[1])
        self.assertLessEqual(seen[1].get("max_tokens", 999), 400, "the summary should be capped and cheap")
        # the tool result has to be fed back to the model, or it cannot summarise
        self.assertIn("tool", [m.get("role") for m in seen[1]["messages"]])

    def test_empty_completion_is_reported_not_invented(self):
        self.lp._post_json = lambda url, headers, payload, timeout: (
            200, {}, self._content("   "))
        result = router.ROUTER.run_agent("tell me something")
        self.assertFalse(result.ok)
        self.assertIn("no tool call and no text", result.error)

    def test_unanswered_question_is_not_routed_to_duckduckgo(self):
        """With no key set, JARVIS admits it instead of quietly searching the web."""
        os.environ.pop("GROQ_API_KEY", None)
        self.pool.reset()
        result = router.ROUTER.route("why is the sky blue")
        self.assertFalse(result.ok)
        self.assertTrue(any(word in result.answer.lower() for word in ("no key", "no ai", "provider")),
                        result.answer)
        self.assertNotIn("search", [c["tool"] for c in result.tool_calls or []])



class TestHudMarkup(unittest.TestCase):
    """The HUD is a pile of getElementById calls: one typo id silently kills a panel."""

    STATIC = Path(__file__).resolve().parent.parent / "static"

    def _pairs(self):
        js = (self.STATIC / "arc_reactor.js").read_text(encoding="utf-8")
        html = (self.STATIC / "index.html").read_text(encoding="utf-8")
        ids = set(re.findall(r"""\$\(\s*['"]([\w-]+)['"]\s*\)""", js))
        ids |= set(re.findall(r"""getElementById\(\s*['"]([\w-]+)['"]""", js))
        return ids, set(re.findall(r'id="([\w-]+)"', html))

    def test_every_id_the_js_touches_exists(self):
        ids, have = self._pairs()
        self.assertGreater(len(ids), 20, "the regex should be finding the HUD ids")
        self.assertFalse(sorted(ids - have), f"arc_reactor.js looks up ids with no markup: {sorted(ids - have)}")

    def test_provider_panel_is_wired(self):
        ids, have = self._pairs()
        for needed in ("llm-providers", "llm-tier", "llm-summary", "btn-llm-reset", "btn-llm-probe", "pill-model"):
            self.assertIn(needed, have, f"{needed} missing from index.html")
            self.assertIn(needed, ids, f"{needed} is in the markup but the JS never touches it")

    def test_hud_has_no_localhost_hardcoding(self):
        """The preview/proxy host changes; the client must use relative/derived origins."""
        js = (self.STATIC / "arc_reactor.js").read_text(encoding="utf-8")
        self.assertNotIn("ws://127.0.0.1", js.replace("http://127.0.0.1", ""), "hard-coded WS target")
        self.assertIn("backendOrigin()", js, "origin must be derived so the HUD works from any host")

    def test_styles_cover_the_new_classes(self):
        css = (self.STATIC / "styles.css").read_text(encoding="utf-8")
        for selector in (".state-pill", ".pill-model", ".chip", ".hud-panel", ".panel-title", ".meter"):
            self.assertIn(selector, css, f"{selector} is used by the HUD but not styled")



class TestModelFallback(unittest.TestCase):
    """A model id the key refuses must be absorbed inside the same turn.

    The provider object is rebuilt with invented model names so these assertions say
    something about the failover logic and not about which id Groq happens to list this
    month (their 2026-09 free lineup has no Llama id at all - see TestRealProviderPayloads).
    """

    VISIBLE = ("small-model", "big-model", "llama-guard-4-12b", "whisper-large-v3-turbo",
               "playai-tts Array", "text-embedding-3-small")

    def setUp(self):
        EnvOnly.install(self)          # .env on this box may hold real keys
        import llm_providers as lp

        self.lp = lp
        self._dir = tempfile.mkdtemp(prefix="jarvis-modelfb-")
        self.state = Path(self._dir) / "llm.json"
        self.pool = lp.LlmPool(state_file=self.state)
        self._post, self._get = lp._post_json, lp._get_json
        self._real_pool, self._providers = lp.POOL, dict(lp.PROVIDERS)
        lp.POOL = self.pool
        lp.PROVIDERS["groq"] = dataclasses.replace(
            lp.PROVIDERS["groq"], fast_model="small-model", smart_model="big-model",
            fallback_models=(), free_quota="", reset="none")
        self._env = {}
        for name in ("GROQ_API_KEY", "CEREBRAS_API_KEY", "GEMINI_API_KEY", "MISTRAL_API_KEY",
                     "OPENROUTER_API_KEY", "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID",
                     "GITHUB_MODELS_TOKEN", "CUSTOM_LLM_BASE_URL", "CUSTOM_LLM_API_KEY"):
            self._env[name] = os.environ.get(name)
            os.environ.pop(name, None)
        os.environ["GROQ_API_KEY"] = "gsk-test"
        self.list_calls = []
        lp._get_json = self._fake_get
        # Two of these tests assert that discovery ran during the turn, and LLM_AUTO_DISCOVER=false
        # in a real .env would switch that off for them.  EnvOnly hides the file from later reads,
        # but config.SETTINGS was frozen at import - so the class owns the switch itself, on a
        # copy, instead of inheriting whatever the machine happens to have configured.
        patch = mock.patch.object(lp, "SETTINGS",
                                  dataclasses.replace(config.SETTINGS, llm_auto_discover=True))
        patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self):
        for name, value in self._env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self.lp.POOL = self._real_pool
        self.lp.PROVIDERS.clear()
        self.lp.PROVIDERS.update(self._providers)
        self.lp._post_json, self.lp._get_json = self._post, self._get
        shutil.rmtree(self._dir, ignore_errors=True)

    def _fake_get(self, url, headers, timeout):
        self.list_calls.append(url)
        return 200, {}, {"data": [{"id": i} for i in self.VISIBLE]}

    def _fake_post(self, ok_models, status=404,
                   message="The model `%s` does not exist or you do not have access to it."):
        sent = []

        def post(url, headers, payload, timeout):
            model = payload.get("model", "")
            sent.append(model)
            if model in ok_models:
                return 200, {}, {"choices": [{"finish_reason": "stop",
                                              "message": {"role": "assistant",
                                                          "content": f"answered by {model}"}}]}
            return status, {}, {"error": {"message": message % model}}

        self.lp._post_json = post
        return sent

    def test_refused_model_is_replaced_inside_the_same_turn(self):
        sent = self._fake_post({"small-model"})
        out = self.pool.complete([{"role": "user", "content": "hi"}], tier="smart")
        self.assertEqual(out["content"], "answered by small-model")
        self.assertTrue(out["model_switched"], "the caller must be able to see that a swap happened")
        self.assertEqual(sent.count("big-model"), 1, "a refused model must not be hammered twice in a turn")
        self.assertNotEqual(self.pool.model_for("groq", "smart"), "big-model")

    def test_discovery_reads_the_key_list_once_and_ignores_non_chat_models(self):
        self._fake_post({"small-model"})
        self.pool.complete([{"role": "user", "content": "hi"}], tier="smart")
        found = self.pool._health_for("groq")["discovered"]          # noqa: SLF001
        self.assertEqual(len(self.list_calls), 1, "one /models read per cache window, not one per turn")
        self.assertEqual(found["fast"], "small-model")
        self.assertEqual(found["smart"], "big-model", "the chat model, not the guard/embedding/audio ones")
        for junk in ("whisper", "guard", "embedding", "playai-tts"):
            self.assertNotIn(junk, found["fast"] + found["smart"])
            self.assertNotIn(junk, self.pool.model_for("groq", "smart"))
        self.pool.complete([{"role": "user", "content": "again"}], tier="fast")
        self.assertEqual(len(self.list_calls), 1, "the answer is cached")

    def test_parking_survives_a_restart(self):
        self._fake_post(set())
        with self.assertRaises(self.lp.LlmError):
            self.pool.complete([{"role": "user", "content": "hi"}], tier="smart")
        reloaded = self.lp.LlmPool(state_file=self.state)
        self.assertNotEqual(reloaded.model_for("groq", "smart"), "big-model",
                            "a restart must not spend the first turn on a known-dead id")
        self.assertIn("big-model", reloaded.status()["providers"][0]["rejected_models"])

    def test_all_models_refused_backs_off_with_an_actionable_message(self):
        self._fake_post(set())
        with self.assertRaises(self.lp.LlmError) as caught:
            self.pool.complete([{"role": "user", "content": "hi"}], tier="smart")
        cooling, left, reason = self.pool.cooling("groq")
        self.assertTrue(cooling and left > 60, f"expected a ~10 minute back-off, got {left}s")
        self.assertIn("llm_providers.py", reason)
        self.assertIn("MODEL_FAST", reason)
        tried = [a.get("model") for a in (caught.exception.attempts or []) if a.get("model_problem")]
        self.assertGreaterEqual(len(tried), 2, "the whole ladder is walked before giving up")
        self.assertEqual(len(tried), len(set(tried)), "no model is asked for twice in one turn")

    def test_auth_failure_does_not_cycle_models(self):
        sent = self._fake_post(set(), status=403, message="Your account does not have access to %s")
        with self.assertRaises(self.lp.LlmError):
            self.pool.complete([{"role": "user", "content": "hi"}], tier="smart")
        self.assertTrue(self.pool.cooling("groq")[0], "403 is a plan/key problem: exile the provider")
        self.assertEqual(len(sent), 1, "no point trying four models against a permission wall")

    def test_prewarm_reports_visibility_without_raising(self):
        summary = self.pool.prewarm()
        self.assertIn("groq", summary)
        self.assertEqual(summary["groq"]["seen"], len(self.VISIBLE))
        self.assertTrue(summary["groq"]["fast"] and summary["groq"]["smart"])

        def broken(url, headers, timeout):
            raise ConnectionError("no route")

        self.lp._get_json = broken
        fresh = self.lp.LlmPool(state_file=Path(self._dir) / "other.json")
        quiet = fresh.prewarm()
        self.assertEqual(quiet["groq"]["seen"], 0, "an unreadable list must be silent, not fatal")

    def test_status_shows_what_the_key_can_see(self):
        self._fake_post({"small-model"})
        self.pool.complete([{"role": "user", "content": "hi"}], tier="smart")
        provider = self.pool.status()["providers"][0]
        self.assertEqual(provider["models_seen"], len(self.VISIBLE))
        self.assertIn("llama-guard-4-12b", provider["model_ids"], "the raw list is exposed for the HUD")
        self.assertIn("big-model", provider["rejected_models"])
        self.assertEqual(provider["discovered"]["fast"], "small-model")


class TestRealProviderPayloads(unittest.TestCase):
    """The picker against a captured, real /models response -- the interesting edge cases."""

    # Trimmed to the fields that matter, from a free Groq key on 2026-09-10.
    GROQ = [
        {"id": "allam-2-7b", "supported_features": ["json_mode"], "context_window": 4096,
         "output_modalities": ["text"]},
        {"id": "qwen/qwen3.6-27b", "supported_features": ["tools", "json_mode", "reasoning"],
         "context_window": 131072, "output_modalities": ["text", "image"]},
        {"id": "whisper-large-v3-turbo", "context_window": 448,
         "output_modalities": ["transcription"]},
        {"id": "meta-llama/llama-prompt-guard-2-86m", "supported_features": ["json_mode"],
         "context_window": 512, "output_modalities": ["text"]},
        {"id": "openai/gpt-oss-120b", "supported_features": ["tools", "json_mode", "structured_outputs", "reasoning"],
         "context_window": 131072, "output_modalities": ["text"]},
        {"id": "canopylabs/orpheus-v1-english", "context_window": 4000, "output_modalities": ["speech"]},
        {"id": "qwen/qwen3.8-27b", "supported_features": ["tools", "json_mode", "reasoning"],
         "context_window": 131042, "output_modalities": ["text", "image"]},
        {"id": "openai/gpt-oss-safeguard-20b", "supported_features": ["tools", "json_mode", "structured_outputs", "reasoning"],
         "context_window": 131072, "output_modalities": ["text"]},
        {"id": "groq/compound", "supported_features": ["json_mode"], "context_window": 131072,
         "output_modalities": ["text"]},
        {"id": "groq/compound-mini", "supported_features": ["json_mode"], "context_window": 131072,
         "output_modalities": ["text"]},
        {"id": "openai/gpt-oss-20b", "supported_features": ["tools", "json_mode", "structured_outputs", "reasoning"],
         "context_window": 131072, "output_modalities": ["text"]},
    ]

    def setUp(self):
        import llm_providers as lp

        self.lp = lp
        self._pick, self._rows = lp._pick_models, lp._model_rows
        self._post, self._get = lp._post_json, lp._get_json

    def tearDown(self):
        self.lp._pick_models, self.lp._model_rows = self._pick, self._rows
        self.lp._post_json, self.lp._get_json = self._post, self._get

    def test_groq_lineup_is_picked_sensibly(self):
        fast, smart, capable = lp_pick(self.GROQ)
        self.assertEqual(fast, "openai/gpt-oss-20b", "smallest tool-capable model with a real window")
        self.assertEqual(smart, "openai/gpt-oss-120b", "the big chat model, not the 27B Qwen")
        self.assertNotIn("allam-2-7b", capable + [fast, smart], "4K context, no tools: unusable")
        for junk in ("whisper-large-v3-turbo", "canopylabs/orpheus-v1-english",
                     "meta-llama/llama-prompt-guard-2-86m", "openai/gpt-oss-safeguard-20b"):
            self.assertNotIn(junk, [fast, smart], f"{junk} must never be chosen")
        self.assertNotIn("groq/compound", capable, "it advertises no tools, so it cannot carry a tool call")

    def test_json_only_lineup_is_used_without_native_tools(self):
        rows = [r for r in self.GROQ if "tools" not in (r.get("supported_features") or [])]
        fast, smart, capable = lp_pick(rows)
        self.assertEqual(capable, [])
        self.assertIn(fast, {"groq/compound-mini", "groq/compound", "allam-2-7b"},
                      "with no tool-capable model the picker must still find a chat model")

    def test_bare_id_list_still_works(self):
        fast, smart, capable = lp_pick(self._rows({"data": [{"id": "llama3.1-8b"},
                                                            {"id": "gpt-oss-120b"}]}))
        self.assertEqual((fast, smart), ("llama3.1-8b", "gpt-oss-120b"))
        self.assertEqual(capable, [], "unknown features are not the same as 'no tools'")

    def test_string_rows_and_models_prefix_are_tolerated(self):
        rows = self._rows({"data": ["models/gemini-2.5-flash-lite", "models/gemini-2.5-flash"]})
        self.assertEqual([r["id"] for r in rows], ["gemini-2.5-flash-lite", "gemini-2.5-flash"])
        fast, smart, _capable = lp_pick(rows)
        self.assertEqual((fast, smart), ("gemini-2.5-flash-lite", "gemini-2.5-flash"))


def lp_pick(rows):
    import llm_providers
    return llm_providers._pick_models(rows)      # noqa: SLF001 - the picker under test



class TestWindowBootstrap(unittest.TestCase):
    """pywebview's load event is an object subscribed to with ``+=``, not a decorator."""

    class Event:
        def __init__(self) -> None:
            self.handlers: list = []

        def __iadd__(self, handler):
            self.handlers.append(handler)
            return self

        def fire(self) -> int:
            for handler in self.handlers:
                handler()
            return len(self.handlers)

    class ModernWindow:
        def __init__(self) -> None:
            self.events = type("E", (), {"loaded": TestWindowBootstrap.Event()})()

    class LegacyWindow:
        def __init__(self) -> None:
            self.loaded = TestWindowBootstrap.Event()

    class DeadWindow:
        pass

    def setUp(self) -> None:
        import main
        self.main = main

    def test_modern_event_object_is_used(self):
        window, called = self.ModernWindow(), []
        label = self.main.attach_loaded_handler(window, lambda: called.append(1))
        self.assertEqual(label, "window.events.loaded")
        self.assertEqual(window.events.loaded.fire(), 1)
        self.assertEqual(called, [1])

    def test_legacy_event_object_is_used(self):
        window, called = self.LegacyWindow(), []
        label = self.main.attach_loaded_handler(window, lambda: called.append(1))
        self.assertEqual(label, "window.loaded")
        self.assertEqual(window.loaded.fire(), 1, "the handler must be registered on the event")
        self.assertEqual(called, [1])

    def test_no_event_object_reports_empty_so_start_falls_back(self):
        self.assertEqual(self.main.attach_loaded_handler(self.DeadWindow(), lambda: None), "")

    def test_broken_iadd_does_not_raise(self):
        window = type("W", (), {"events": type("E", (), {"loaded": object()})()})()
        self.assertEqual(self.main.attach_loaded_handler(window, lambda: None), "")

    @unittest.skipUnless(__import__("importlib").util.find_spec("webview"), "pywebview not installed")
    def test_real_pywebview_event_object_is_subscribed(self):
        """Against the actual library: ``+=`` works, the decorator form raises."""
        from webview.event import Event

        class Emitter:
            def __init__(self) -> None:
                self.loaded = Event(None)

        class RealWindow:
            def __init__(self) -> None:
                self.events = Emitter()

        window, ran = RealWindow(), []
        self.assertEqual(self.main.attach_loaded_handler(window, lambda: ran.append(1)), "window.events.loaded")
        window.events.loaded.set()
        self.assertEqual(ran, [1])
        with self.assertRaises(TypeError):
            @window.events.loaded
            def _decorator_attempt() -> None: ...

    def test_main_py_never_uses_the_event_as_a_decorator(self):
        source = (config.ROOT / "main.py").read_text(encoding="utf-8")
        code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
        self.assertNotIn("@window.events", code, "the load event must be subscribed with +=, never used as a decorator")
        self.assertIn("attach_loaded_handler(window, on_loaded)", source)


# ------------------------------------------------------- round 7: apps, files, eyes, ears, time
class TestAppKnowledge(unittest.TestCase):
    """"It doesn't know which app is which" - the resolution ladder, no Windows required."""

    @classmethod
    def setUpClass(cls):
        import apps
        cls.apps = apps

    def test_verbs_politeness_and_suffixes_are_stripped(self):
        self.assertEqual(self.apps.strip_verbs("could you please open the bluetooth settings app now"),
                         "bluetooth settings")
        self.assertEqual(self.apps.strip_verbs("hey jarvis, launch chrome"), "chrome")
        self.assertEqual(self.apps.strip_verbs(""), "")

    def test_settings_pages_are_uris_not_executables(self):
        self.assertEqual(self.apps.settings_page("bluetooth settings"), "ms-settings:bluetooth")
        ref = self.apps.resolve("open bluetooth settings")
        self.assertIsNotNone(ref, "a Settings page must resolve on any OS - the tables are static")
        self.assertTrue(str(ref.target).startswith("ms-settings:"), ref.as_dict())

    def test_store_apps_come_with_an_aumid(self):
        ref = self.apps.resolve("microsoft teams")
        self.assertIsNotNone(ref, "new Teams is a store package; the catalogue must know its AUMID")
        self.assertTrue(ref.aumid or "AppsFolder" in str(ref.target), ref.as_dict())

    def test_spelling_is_corrected_and_reported(self):
        ref = self.apps.resolve("crome")
        self.assertIsNotNone(ref)
        self.assertIn("chrome", (ref.key or "").lower())
        self.assertEqual(ref.corrected_from, "crome")

    def test_an_unknown_app_is_an_explicit_none(self):
        self.assertIsNone(self.apps.resolve("xyzzy-not-a-program-9913"))

    def test_custom_apps_are_read_from_the_environment(self):
        os.environ["CUSTOM_APPS"] = "code editor = C:/Vscode/Code.exe"
        try:
            table = self.apps.custom_apps()
            self.assertEqual(table.get("code-editor"), "C:/Vscode/Code.exe", table)
            self.assertEqual(self.apps.resolve("code editor").target, "C:/Vscode/Code.exe")
        finally:
            del os.environ["CUSTOM_APPS"]

    def test_suggest_gives_a_did_you_mean(self):
        self.assertTrue(any("settings" in n.lower() for n in self.apps.suggest("sett")),
                        self.apps.suggest("sett"))

    def test_known_names_are_unique_and_sorted(self):
        names = self.apps.known_names()
        self.assertGreater(len(names), 40, "the catalogue alone must already be useful")
        self.assertIn("Command Prompt", names)
        self.assertIn("Microsoft Teams", names)
        self.assertEqual(names, sorted(set(names)), "no duplicates, stable order for the HUD")

    def test_launch_failure_carries_a_reason(self):
        result = self.apps.launch("xyzzy-not-a-program-9913")
        self.assertFalse(result["ok"])
        self.assertGreater(len(result["message"]), 20, result)


class _FileSandbox:
    """Confine the file layer - root, journal, backups, trash - to one temp directory."""

    def __init__(self, delete_policy="recycle"):
        self.delete_policy = delete_policy
        self.dir = None
        self.saved = {}

    def __enter__(self):
        import files
        self.files = files
        self.dir = Path(tempfile.mkdtemp(prefix="jarvis-files-"))
        root = self.dir / "Documents" / "JARVIS"
        root.mkdir(parents=True)
        self.root = root
        self.saved = {"S": files.SETTINGS, "B": files.BACKUP_DIR, "T": files.TRASH_DIR,
                      "J": files.JOURNAL}
        files.SETTINGS = type("S", (), {"files_root": str(root), "files_allowed": "",
                                        "file_delete_policy": self.delete_policy})()
        files.BACKUP_DIR = self.dir / "backups"
        files.TRASH_DIR = self.dir / "trash"
        files.JOURNAL = self.dir / "journal.jsonl"
        return root

    def __exit__(self, *exc):
        files = self.files
        files.SETTINGS = self.saved["S"]
        files.BACKUP_DIR = self.saved["B"]
        files.TRASH_DIR = self.saved["T"]
        files.JOURNAL = self.saved["J"]
        shutil.rmtree(self.dir, ignore_errors=True)
        return False


class TestFileLayer(unittest.TestCase):
    def test_create_then_overwrite_then_append_then_undo(self):
        with _FileSandbox() as root:
            made = files.write("ideas.md", "milk and eggs")
            self.assertTrue(made["ok"], made)
            self.assertTrue((root / "ideas.md").is_file())
            again = files.write("ideas.md", "bread")
            self.assertFalse(again["ok"], "create must refuse to clobber")
            self.assertIn("overwrite", again["message"].lower())
            over = files.write("ideas.md", "bread", mode="overwrite")
            self.assertTrue(over["ok"], over)
            self.assertTrue(over["backup"], "the bytes it replaced must be backed up")
            app = files.write("ideas.md", "coffee", mode="append")
            self.assertTrue(app["ok"], app)
            body = (root / "ideas.md").read_text(encoding="utf-8")
            self.assertIn("bread", body)
            self.assertIn("coffee", body)
            read = files.read("ideas.md")
            self.assertTrue(read["ok"], read)
            self.assertIn("bread", read["message"])
            self.assertIn("coffee", read["message"])
            self.assertNotIn("milk", read["message"], "overwrite really replaced the first draft")
            self.assertIn("milk", Path(over["backup"]).read_text(encoding="utf-8"),
                          "the bytes overwrite destroyed must be recoverable")
            undo = files.undo(1)
            self.assertTrue(undo["ok"], undo)
            self.assertNotIn("coffee", (root / "ideas.md").read_text(encoding="utf-8"))

    def test_unknown_mode_is_refused_with_the_list_of_modes(self):
        with _FileSandbox():
            result = files.write("odd.txt", "x", mode="smash")
            self.assertFalse(result["ok"])
            self.assertIn("append", result["message"])

    def test_delete_goes_to_the_bin_and_undoes(self):
        with _FileSandbox() as root:
            files.write("doomed.txt", "evidence")
            gone = files.delete("doomed.txt")
            self.assertTrue(gone["ok"], gone)
            self.assertFalse((root / "doomed.txt").exists())
            kept = list(Path(files.TRASH_DIR).glob("*/doomed.txt"))
            self.assertTrue(kept, "a private copy must exist so undo can put it back")
            back = files.undo(1)
            self.assertTrue(back["ok"], back)
            self.assertTrue((root / "doomed.txt").is_file())

    def test_refuse_policy_blocks_deletion(self):
        with _FileSandbox(delete_policy="refuse") as root:
            files.write("safe.txt", "keep me")
            result = files.delete("safe.txt")
            self.assertFalse(result["ok"])
            self.assertIn("FILE_DELETE_POLICY", result["message"])
            self.assertTrue((root / "safe.txt").is_file())

    def test_a_misspelled_name_is_answered_with_a_neighbor(self):
        with _FileSandbox():
            files.write("quarterly-report.md", "numbers")
            result = files.read("quarterly-reprot.md")
            self.assertFalse(result["ok"])
            self.assertIn("quarterly-report", result["message"])

    def test_office_document_text_is_extracted_without_office(self):
        import zipfile
        with _FileSandbox() as root:
            target = root / "memo.docx"
            with zipfile.ZipFile(target, "w") as book:
                book.writestr("[Content_Types].xml", "<Types/>")
                book.writestr("word/document.xml",
                              '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessing'
                              '/ml/2006/main"><w:body><w:p><w:r><w:t>Hello from Word</w:t></w:r>'
                              '</w:p></w:body></w:document>')
            read = files.read("memo.docx")
            self.assertTrue(read["ok"], read)
            self.assertIn("Hello from Word", read["message"])

    def test_search_and_list_never_raise_on_an_empty_folder(self):
        with _FileSandbox():
            self.assertTrue(files.list_dir("")["ok"])
            hits = files.search(name="nothing-with-this-name-7781")
            self.assertTrue(hits["ok"])
            self.assertEqual(hits["results"], [])

    def test_scripts_are_written_and_can_be_run(self):
        with _FileSandbox() as root:
            made = files.script("hello_jarvis.py", code="print('hi from the sandbox')")
            self.assertTrue(made["ok"], made)
            self.assertTrue((root / "scripts" / "hello_jarvis.py").is_file())
            ran = files.execute_script("scripts/hello_jarvis.py")
            self.assertTrue(ran["ok"], ran)
            self.assertIn("hi from the sandbox", ran["message"])

    def test_windows_paths_are_routable_by_the_file_rules(self):
        # A drive letter used to break these: the name character class had no ":" in it, so
        # "delete C:\Users\me\Documents\JARVIS\old.txt" matched nothing and fell through to the
        # model.  Asserted on the pattern itself, so it is checked on every platform.
        cases = (
            (r"delete C:\Users\me\Documents\JARVIS\old.txt", "delete"),
            (r"read C:\Users\me\Documents\JARVIS\notes.txt", "read"),
            (r"create a file called C:\Users\me\Documents\JARVIS\ideas.md with milk", "write"),
        )
        for text, action in cases:
            plan = server.match_instant(text)
            self.assertIsNotNone(plan, text)
            self.assertEqual(plan["calls"][0]["tool"], "manage_files", text)
            self.assertEqual(plan["calls"][0]["arguments"]["action"], action, text)
        import winops

        for name in ("C:/Users/me/Documents/JARVIS/old.txt",
                     "C:\\Users\\me\\Documents\\JARVIS\\old.txt"):
            located, error = files.resolve_path(name)
            if winops.IS_WINDOWS:
                self.assertIsNotNone(located, (name, error))
                self.assertFalse(located.inside, "a path outside the roots is never treated as inside")
            else:
                self.assertIsNone(located, "off Windows a drive-letter path is refused, not guessed")
                self.assertIn("not on Windows", error)

    def test_a_delete_outside_the_granted_roots_asks_first(self):
        with _FileSandbox() as root:
            elsewhere = root.parent.parent / "payroll.csv"      # under our temp dir, outside the root
            elsewhere.write_text("salaries", encoding="utf-8")
            try:
                plan = server.match_instant(f"delete {elsewhere}")
                self.assertIsNotNone(plan, "an absolute path must be routable")
                self.assertTrue(plan.get("confirm"), plan)
                self.assertIn("Recycle Bin", plan["confirm"])
                self.assertTrue(elsewhere.is_file(), "asking is not doing")
                # and the refusal is actionable whichever branch the platform took
                answer = files.delete(str(elsewhere))
                self.assertFalse(answer["ok"])
                self.assertIn("FILES_ALLOWED", answer["message"], answer)
            finally:
                elsewhere.unlink(missing_ok=True)

    def test_a_delete_inside_the_jarvis_folder_just_happens(self):
        with _FileSandbox() as root:
            (root / "scratch.txt").write_text("temporary", encoding="utf-8")
            plan = server.match_instant("delete scratch.txt")
            self.assertIsNone(plan.get("confirm"), plan)
            self.assertEqual(plan["calls"][0]["tool"], "manage_files")
            self.assertTrue(plan["calls"][0]["arguments"]["path"].endswith("scratch.txt"))

    def test_an_outside_write_is_never_silent(self):
        with _FileSandbox():
            outside = Path(tempfile.mkdtemp(prefix="jarvis-outside-")) / "notes-todo.txt"
            try:
                first = files.write(str(outside), "hello", mode="overwrite")
                self.assertFalse(first["ok"])
                self.assertIn("FILES_ALLOWED", first["message"], first)
                if first.get("needs_confirmation"):
                    self.assertTrue(first.get("confirm_token"), "a confirmation needs its token")
                    self.assertIn("confirm", first["message"].lower())
                    # and obeying it is allowed, which is the point of asking
                    second = files.write(str(outside), "hello", mode="overwrite", confirm="confirm")
                    self.assertTrue(second["ok"], second)
                    outside.unlink(missing_ok=True)
                self.assertFalse(outside.exists(), "refusing must not touch the disk")
            finally:
                shutil.rmtree(outside.parent, ignore_errors=True)


class TestScreenLayer(unittest.TestCase):
    def test_top_results_prefers_real_titles_over_browser_chrome(self):
        text = "\n".join([
            "Google Chrome File Edit View History Bookmarks",
            "https://www.google.com",
            "Sign in",
            "Sony WH-1000XM5 Wireless Headphones - 4.6 out of 5 stars (2,304)",
            "$292.00  FREE delivery Tue, Sep 15",
            "Bose QuietComfort Ultra Headphones, Wireless over-Ear 4.5 out of 5",
            "www.amazon.com/dp/B0CJHK2MVS",
            "Sennheiser Momentum 4 Wireless Headphones - 4.5 out of 5 stars",
            "Results 1 - 16 of over 4,000 for headphones",
        ])
        rows = screen.top_results(text, count=3)
        self.assertEqual(len(rows), 3, rows)
        titles = " ".join(r["title"] for r in rows).lower()
        self.assertIn("sony", titles)
        self.assertIn("bose", titles)
        self.assertIn("sennheiser", titles)
        self.assertNotIn("sign in", titles)
        self.assertNotIn("results 1", titles)

    def test_count_is_respected_and_duplicates_dropped(self):
        rows = screen.top_results("First listing with a decent length title here\n"
                                  "First listing with a decent length title here\n"
                                  "Second listing with a decent length title here", count=5)
        self.assertEqual(len(rows), 2)

    def test_garbage_in_and_out_without_a_crash(self):
        self.assertEqual(screen.top_results("", count=3), [])
        self.assertIsInstance(screen.top_results("\n\n1\n::\n", count=3), list)

    def test_a_numbered_results_column_is_understood(self):
        rows = screen.top_results("1.\nPython 3.12.0 download - python.org\npython.org\n"
                                  "2.\nBest Python IDEs for Windows - JetBrains\njetbrains.com", count=2)
        self.assertTrue(rows, "a numbered results column is exactly what this exists for")
        self.assertTrue(any("Python 3.12.0" in row["title"] for row in rows), rows)

    def test_read_screen_explains_a_missing_ocr_engine(self):
        result = screen.read_screen(count=3)
        self.assertIsInstance(result["ok"], bool)
        if not result["ok"]:
            self.assertGreater(len(result["message"]), 15, result)

    def test_capture_returns_a_dict_whatever_the_platform(self):
        result = screen.capture()
        self.assertIsInstance(result["ok"], bool)
        self.assertIn("message", result)


class TestWakeWordEar(unittest.TestCase):
    def setUp(self):
        self.saved = wake.SETTINGS
        wake.SETTINGS = type("S", (), {"wake_words": "jarvis,jervis", "wake_fuzzy": True,
                                        "wake_followup_seconds": 12, "wake_command_key": "f12",
                                        "wake_push_to_talk": "rcontrol", "bar_enabled": True,
                                        "wake_enabled": True, "llm_budget_seconds": 25})()

    def tearDown(self):
        wake.SETTINGS = self.saved

    def test_wake_words_are_parsed_and_deduped(self):
        self.assertEqual(wake.wake_words()[:2], ("jarvis", "jervis"))

    def test_utterances_addressed_to_jarvis_are_accepted(self):
        for text in ("Jarvis, open spotify", "hey jarvis what time is it", "jervis pause the music",
                     "JARVIS did you see that"):
            self.assertTrue(wake.matches_wake(text), text)

    def test_similar_looking_speech_is_rejected(self):
        for text in ("open the jar lid", "can you harvest the wheat", "the service is down",
                     "install java and python", "my favourite movie is jar"):
            self.assertFalse(wake.matches_wake(text), text)

    def test_strip_wake_leaves_only_the_command(self):
        self.assertEqual(wake.strip_wake("jarvis, open spotify").lower(), "open spotify")
        self.assertEqual(wake.strip_wake("open spotify"), "open spotify")

    def test_the_energy_gate_finds_one_span_not_confetti(self):
        # 0.45 s of dip mid-sentence is still one utterance; the 1.2 s tail closes the gate.
        curve = [40.0] * 30 + [900.0] * 40 + [40.0] * 15 + [500.0] * 12 + [40.0] * 40
        spans = wake.gate(curve)
        self.assertEqual(len(spans), 1, spans)
        start, end = spans[0]
        self.assertLess(start, 30, "pre-roll must reach back before the trigger")
        self.assertGreater(end, 85, "hysteresis must not chop the tail off a sentence")

    def test_silence_is_gated_out(self):
        self.assertEqual(wake.gate([20.0] * 200), [])
        self.assertEqual(wake.gate([0.0] * 100), [])

    def test_rms_measures_energy(self):
        self.assertEqual(wake.rms(b""), 0.0)
        self.assertEqual(wake.rms(b"\x00\x00" * 8), 0.0)
        self.assertGreater(wake.rms(b"\xe8\x03" * 8), 900.0)

    def test_the_listener_answers_without_a_microphone(self):
        ok, why = wake.LISTENER.available()
        self.assertIsInstance(ok, bool)
        if not ok:
            self.assertGreater(len(why), 10, "an unavailable ear must explain itself")
        self.assertIn("running", wake.LISTENER.status())


class TestWin32Bindings(unittest.TestCase):
    """Which DLL every entry point lives in, checked without a Windows box.

    ``winops`` calls Win32 through ``ctypes.windll`` handles, and ctypes answers a name looked up
    on the wrong module with ``AttributeError: function 'X' not found`` *at the call site*.  That
    is precisely how "open Settings" died on the laptop: ``ShellExecuteW`` is a shell32 export and
    was being taken from user32, ``BitBlt``/``GetDIBits``/``CreateCompatibleDC`` are gdi32 and were
    also taken from user32 (so the screen could never be read), and ``GetSystemPowerStatus`` is
    kernel32.  Nothing on Linux can call these, so the cheap proofs are used instead: the source is
    audited against a table written from the SDK headers, and the code paths that decide whether
    "open settings" works are driven with fakes.
    """

    #: Documented owner of each entry point winops uses (WinUser.h / Wingdi.h / WinBase.h /
    #: ShellAPI.h).  Adding a call to winops without adding it here fails the audit on purpose.
    OWNERS = {
        # shell32 - the association database, i.e. "open this thing the way Explorer would"
        "ShellExecuteW": "shell32", "SHFileOperationW": "shell32", "SHEmptyRecycleBinW": "shell32",
        # gdi32 - device contexts and blitting
        "CreateCompatibleDC": "gdi32", "CreateCompatibleBitmap": "gdi32", "SelectObject": "gdi32",
        "BitBlt": "gdi32", "GetDIBits": "gdi32", "DeleteDC": "gdi32", "DeleteObject": "gdi32",
        # kernel32 - memory the clipboard shares, and power
        "GlobalAlloc": "kernel32", "GlobalLock": "kernel32", "GlobalUnlock": "kernel32",
        "GlobalSize": "kernel32", "GetSystemPowerStatus": "kernel32",
        "SetConsoleCtrlHandler": "kernel32",   # the call that keeps Ctrl+C from killing the run
        # user32 - windows, clipboard, input
        "GetForegroundWindow": "user32", "SetForegroundWindow": "user32", "BringWindowToTop": "user32",
        "ShowWindow": "user32", "IsWindowVisible": "user32", "IsIconic": "user32", "SetWindowPos": "user32",
        "GetWindowRect": "user32", "GetWindowLongW": "user32", "SetWindowLongW": "user32",
        "GetWindowTextW": "user32", "GetWindowTextLengthW": "user32", "GetWindowThreadProcessId": "user32",
        "EnumWindows": "user32", "GetSystemMetrics": "user32", "GetDC": "user32", "ReleaseDC": "user32",
        "SetProcessDPIAware": "user32", "LockWorkStation": "user32", "SystemParametersInfoW": "user32",
        "OpenClipboard": "user32", "CloseClipboard": "user32", "EmptyClipboard": "user32",
        "IsClipboardFormatAvailable": "user32", "GetClipboardData": "user32", "SetClipboardData": "user32",
        "SendInput": "user32", "keybd_event": "user32", "GetAsyncKeyState": "user32",
        "SetCursorPos": "user32", "GetCursorPos": "user32",
        "Beep": "kernel32",   # WinBase.h; user32's similar name is MessageBeep
    }

    @staticmethod
    def _source() -> str:
        return Path(inspect.getsourcefile(winops)).read_text(encoding="utf-8")

    def calls(self):
        found = set()
        for module, name in re.findall(r"\b(user32|shell32|gdi32|kernel32)\.([A-Za-z_][A-Za-z0-9_]*)\b",
                                       self._source()):
            if name != "dll":            # prose such as "shell32.dll is not loaded"
                found.add((name, module))
        return found

    def test_every_win32_call_is_made_on_the_dll_that_exports_it(self):
        used = self.calls()
        self.assertGreater(len(used), 30, "the audit found almost nothing - did the regex or the "
                                          "source layout change? A blind pass here is worse than none")
        for name, module in sorted(used):
            self.assertIn(name, self.OWNERS, f"{name} is called through {module} but is not declared")
            self.assertEqual(self.OWNERS[name], module,
                             f"{name} belongs to {self.OWNERS[name]}.dll, not {module}.dll - ctypes "
                             f"will answer 'function not found' on the user's machine")

    def test_prototypes_match_the_same_table(self):
        seen = set()
        for module, name, restype, _argtypes in winops._PROTOTYPES:
            self.assertNotIn((module, name), seen, f"{module}!{name} declared twice")
            seen.add((module, name))
            self.assertEqual(self.OWNERS.get(name), module,
                             f"_PROTOTYPES configures {name} on {module}")
            if name == "ShellExecuteW":
                # It returns an HINSTANCE.  Left as ctypes' default c_int, a handle is truncated.
                self.assertIs(restype, ctypes.c_void_p)
        for needed in ("ShellExecuteW", "GetForegroundWindow", "GetClipboardData", "SendInput"):
            self.assertIn(needed, {name for _m, name, _r, _a in winops._PROTOTYPES},
                          f"{needed} must have an explicit prototype")

    def test_boot_says_what_the_windows_layer_cannot_do(self):
        """A gap has to be visible at boot, in words, not discovered by asking for Settings."""
        health = winops.win32_health()
        self.assertIn("missing", health)
        with mock.patch.object(winops, "_WIN32_MISSING", ["user32", "shell32!ShellExecuteW"]):
            with mock.patch.object(winops, "IS_WINDOWS", True):
                out = winops.win32_health()
        self.assertFalse(out["ok"])
        self.assertIn("shell32!ShellExecuteW", out["message"])
        self.assertIn("opening apps", out["message"], "the message has to name the lost feature")

    def test_the_self_check_runs_anywhere_and_names_its_probes(self):
        """``python winops.py`` is what the user pastes when something on the desktop is dead.

        It has to work on a machine with no Windows at all (it must not raise, and it must say the
        probes were skipped rather than pretending success), and it must list every subsystem the
        laptop has broken before - bindings, windows, app index, screen grab.
        """
        report = winops.check(light=True)   # the enumerations cost seconds on a real desktop
        self.assertIn("rows", report)
        names = {row["name"] for row in report["rows"]}
        for needed in ("win32 bindings", "window list", "start menu entries", "screen size"):
            self.assertIn(needed, names)
        self.assertTrue(all(row["message"] for row in report["rows"]), "every row says something")
        if not winops.IS_WINDOWS:
            self.assertTrue(report["ok"], "a Linux run is not a failure, it is a skipped probe")
            self.assertTrue(any("not on Windows" in row["message"] or "skipped" in row["message"]
                               for row in report["rows"]),
                            "skipped probes must be marked as skipped, not left out")

    def test_no_self_check_row_is_allowed_to_be_blank(self):
        """The bug the laptop reported: three rows printed nothing.

        ``uwp_apps``/``app_paths``/``installed_exes`` answer with a bare ``{name: target}`` mapping,
        which has no ``message`` key, so the formatter printed an empty string - and an empty line
        in a diagnostics table reads as "checked and fine".
        """
        self.assertEqual(winops.describe_probe("uwp", {"Teams": "x", "Edge": "y"}),
                         {"name": "uwp", "ok": True, "message": "2 entries"})
        self.assertEqual(winops.describe_probe("windows", [])["message"], "0 entries")
        self.assertEqual(winops.describe_probe("grab", None),
                         {"name": "grab", "ok": False, "message": "returned nothing"})
        plain = winops.describe_probe("thing", {"ok": False})
        self.assertIn("without saying why", plain["message"])
        # End-to-end as well, because the blank rows only appeared in the real table.  The other
        # Windows probes are faked here so this stays fast on a laptop - a diagnostics test that
        # spends nine seconds enumerating the Start Menu is a test people learn to skip.
        cheap = {"start_menu_entries": lambda: [], "uwp_apps": lambda: {"a": "1"},
                 "app_paths": lambda: {}, "installed_exes": lambda: {},
                 "screenshot": lambda *_a, **_k: {"ok": True, "message": "captured"}}
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(winops, "IS_WINDOWS", True))
            for name, fn in cheap.items():
                stack.enter_context(mock.patch.object(winops, name, fn))
            rows = {row["name"]: row for row in winops.check()["rows"]}
        self.assertEqual(rows["store apps (UWP)"]["message"], "1 entries",
                         "a probe that answers with a mapping still has to print something")
        self.assertTrue(all(row["message"].strip() for row in rows.values()),
                        "no row in the self-check may be blank")

    def test_beep_prefers_the_standard_library_and_never_raises(self):
        heard = []
        with mock.patch.object(winops, "IS_WINDOWS", True), \
                mock.patch.object(winops, "_winsound_beep", lambda f, d: heard.append((f, d))):
            out = winops.beep(times=2, frequency=700, duration_ms=90)
        self.assertTrue(out["ok"])
        self.assertEqual(out["method"], "winsound")
        self.assertEqual(heard, [(700, 90), (700, 90)])

        class Broken:
            def Beep(self, *_a):
                raise OSError("no waveform-out device")

        def loud(*_a):
            raise ImportError("no winsound on this box")

        with mock.patch.object(winops, "IS_WINDOWS", True), mock.patch.object(winops, "_winsound_beep", loud), \
                mock.patch.object(winops, "kernel32", Broken()):
            out = winops.beep()          # must answer, not raise - a beep is a courtesy
        self.assertFalse(out["ok"])
        self.assertIn("No beep available", out["message"])

    def test_launch_arguments_are_split_the_way_cmd_would(self):
        self.assertEqual(winops._split_args("--new-window \"C:\\My Docs\" x"),
                         ["--new-window", "C:\\My Docs", "x"])
        self.assertEqual(winops._split_args(""), [])
        self.assertEqual(winops._split_args("--unbalanced \"oops"), ["--unbalanced", '"oops'],
                         "a broken quote must not lose the launch, only the split")

    def test_launch_exe_passes_argv_and_not_one_quoted_blob(self):
        started = []

        class FakePopen:
            pid = 4242

            def __init__(self, argv, **kwargs):
                started.append((argv, kwargs.get("cwd")))

        with mock.patch.object(winops, "IS_WINDOWS", True), mock.patch.object(winops.subprocess, "Popen", FakePopen):
            out = winops.launch_exe(sys.executable, "--new-window \"C:\\My Docs\"")
        self.assertTrue(out["ok"], out)
        argv = started[0][0]
        self.assertEqual(argv, [sys.executable, "--new-window", "C:\\My Docs"])
        self.assertNotIn(" ".join(argv[1:]), argv, "two flags must not travel as one argument")

    def test_shellexecute_return_code_is_not_a_boolean(self):
        class Ok:
            @staticmethod
            def ShellExecuteW(*_args):
                return 42                       # any HINSTANCE above 32 means "launched"

        class Refused:
            @staticmethod
            def ShellExecuteW(*_args):
                return 5                        # SE_ERR_ACCESSDENIED

        with mock.patch.object(winops, "IS_WINDOWS", True), mock.patch.object(winops, "shell32", Ok()):
            out = winops.shell_execute("ms-settings:")
            self.assertTrue(out["ok"], out)
            self.assertEqual(out["method"], "shellexecute")
        # A refusal is only final once the shell launchers have failed too - and on a real desktop
        # explorer.exe *does* launch, so the fallback's outcome has to be supplied, not inherited
        # from whatever the machine happens to have installed.
        def nope(cmd, target=""):
            return winops._result(False, "nothing would run it")

        with mock.patch.object(winops, "IS_WINDOWS", True), mock.patch.object(winops, "shell32", Refused()), \
                mock.patch.object(winops, "_run_detached", nope):
            out = winops.shell_execute("C:\\secret\\x.txt")
            self.assertFalse(out["ok"], out)
            self.assertIn("access denied", out["message"])
            self.assertEqual(out["code"], 5)

        rescued = []

        def explorer_says_yes(cmd, target=""):
            rescued.append(list(cmd))
            return winops._result(True, f"Started {target}")

        with mock.patch.object(winops, "IS_WINDOWS", True), mock.patch.object(winops, "shell32", Refused()), \
                mock.patch.object(winops, "_run_detached", explorer_says_yes):
            out = winops.shell_execute("C:\\secret\\x.txt")
            self.assertTrue(out["ok"], "a fallback that worked is the truth, not a hidden failure")
            self.assertEqual(rescued[0], ["explorer.exe", "C:\\secret\\x.txt"])
            self.assertIn("access denied", out["message"], "and the reason stays in the sentence")

    def test_a_missing_entry_point_falls_back_to_the_shell(self):
        """The exact laptop failure: ctypes cannot resolve the name at all.

        Before this was fixed, that AttributeError became the whole answer - "ShellExecute failed
        for ms-settings:: function 'ShellExecuteW' not found" - and no app would open.  The shell
        launchers read the same association database, so they get the turn, and the reason that
        forced them stays in the sentence the user hears.
        """
        class Broken:
            @staticmethod
            def ShellExecuteW(*_args):
                raise AttributeError("function 'ShellExecuteW' not found")

        tried = []

        def fake_run(cmd, target=""):
            tried.append(list(cmd))
            return winops._result(not target.startswith("z:"), f"Started {target}")

        with mock.patch.object(winops, "IS_WINDOWS", True), mock.patch.object(winops, "shell32", Broken()), \
                mock.patch.object(winops, "_run_detached", fake_run):
            out = winops.shell_execute("ms-settings:", "")
            self.assertTrue(out["ok"], out)
            self.assertEqual(out["method"], "explorer")
            self.assertIn("ShellExecuteW could not be resolved", out["message"])
            self.assertEqual(tried[0], ["explorer.exe", "ms-settings:"])

            tried.clear()
            broken_target = "z:\\gone.txt"
            out = winops.shell_execute(broken_target)
            self.assertFalse(out["ok"], "a fallback that also failed must not be reported as success")
            self.assertIn("z:\\gone.txt", out["message"])
            self.assertEqual(len(tried), 2, "explorer first, then cmd start")
            self.assertEqual(tried[1][:4], ["cmd", "/c", "start", ""])


class TestWin32Constants(unittest.TestCase):
    """The numbers Win32 expects, asserted where checking them costs nothing.

    Every one of these was typed from a header once, and a wrong bit is invisible on the machine
    where the code is written: it is a focus-stealing frame change, a context menu opening on
    every double click, upper-case typing, or a file destroyed when it was promised to the Recycle
    Bin.  All four happened here.  Reading each constant back against the SDK value is the only
    test a Linux sandbox can offer for Win32 code - and it is enough to keep the mistake out.
    """

    def test_the_recycle_bin_bit_is_the_undo_bit(self):
        self.assertEqual(winops.FOF_ALLOWUNDO, 0x0040,
                         "0x0004 is FOF_SILENT; without the real allow-undo bit SHFileOperation "
                         "deletes the file for good while reporting that it went to the bin")
        for bit, name in ((winops.FOF_ALLOWUNDO, "undo"), (winops.FOF_NOCONFIRMATION, "the confirm box"),
                          (winops.FOF_SILENT, "the progress dialog"), (winops.FOF_NOERRORUI, "the error dialog")):
            self.assertTrue(winops.RECYCLE_FLAGS & bit, f"recycling must suppress {name}")
        self.assertEqual(winops.RECYCLE_FLAGS, 0x0254)

    def test_window_flags_do_not_steal_focus(self):
        self.assertEqual(winops._SWP_NOACTIVATE, 0x10, "0x20 is FRAMECHANGED, not NOACTIVATE")
        self.assertEqual(winops._SWP_FRAMECHANGED, 0x20)
        self.assertEqual(winops._WS_EX_NOACTIVATE, 0x08000000)
        self.assertEqual(winops._WS_EX_TOOLWINDOW, 0x80)
        self.assertEqual((winops._SW_RESTORE, winops._SW_MINIMIZE, winops._SW_MAXIMIZE), (9, 6, 3))

    def test_a_double_click_is_a_second_press_and_release(self):
        for down, up in winops.MOUSE_BUTTONS.values():
            combined = winops._double_click_flags(down, up)
            self.assertEqual(combined, down | up)
            for other_down, other_up in winops.MOUSE_BUTTONS.values():
                if (other_down, other_up) == (down, up):
                    continue
                self.assertEqual(combined & (other_down | other_up), 0,
                                 "a double click must not touch another button")

    def test_typed_characters_keep_their_case(self):
        for char in "aA$ 0é":
            vk, scan = winops._char_event(char)
            self.assertEqual(vk, 0, "Unicode injection sends no virtual key")
            self.assertEqual(scan, ord(char), "the character itself is what preserves the case")


class TestReminders(unittest.TestCase):
    def test_relative_spans_with_words_and_fractions(self):
        base = datetime(2026, 9, 10, 12, 0, 0)
        due, label, err = reminders.parse_when("in ten minutes", base)
        self.assertEqual(err, "")
        self.assertEqual(due, base.timestamp() + 600)
        self.assertIn("min", label.lower())
        self.assertEqual(reminders.parse_when("in an hour and a half", base)[0], base.timestamp() + 5400)
        self.assertEqual(reminders.parse_when("in two hours", base)[0], base.timestamp() + 7200)

    def test_clock_times_and_a_passed_time_means_tomorrow(self):
        base = datetime(2026, 9, 10, 19, 0, 0)
        due, label, err = reminders.parse_when("at 7:30 pm", base)
        self.assertEqual(err, "", label)
        self.assertEqual(due, datetime(2026, 9, 10, 19, 30).timestamp())
        self.assertEqual(reminders.parse_when("at 7:30", base)[0],
                         datetime(2026, 9, 11, 7, 30).timestamp())
        self.assertEqual(reminders.parse_when("tomorrow at 9", base)[0],
                         datetime(2026, 9, 11, 9, 0).timestamp())

    def test_repeating_asks_are_accepted(self):
        base = datetime(2026, 9, 10, 12, 0, 0)
        due, label, err = reminders.parse_when("every day at 8am", base)
        self.assertEqual(err, "", label)
        self.assertGreater(due, base.timestamp())

    def test_vague_words_have_a_documented_meaning(self):
        base = datetime(2026, 9, 10, 12, 0, 0)
        due, label, err = reminders.parse_when("in a bit", base)
        self.assertEqual(err, "")
        self.assertEqual(due, base.timestamp() + 300, "“in a bit” means five minutes, always")
        due, label, err = reminders.parse_when("sometime", base)
        self.assertIsNone(due)
        self.assertIn("10 minutes", err)

    def test_letters_inside_words_are_not_quantities(self):
        base = datetime(2026, 9, 10, 12, 0, 0)
        due, label, err = reminders.parse_when("every day at 8am", base)
        self.assertEqual(err, "", label)
        self.assertEqual(due, datetime(2026, 9, 11, 8, 0).timestamp(),
                         "“every day at 8am” is tomorrow 08:00, not one minute from now")
        self.assertIn("daily", label)
        self.assertEqual(reminders.parse_when("in 20", base)[0], base.timestamp() + 1200,
                         "a bare number after “in” is minutes")

    def test_the_board_persists_snoozes_cancels_and_fires(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "reminders.json"
            fired = []
            board = reminders.ReminderBoard(path=store, fire=fired.append)
            now = time.time()
            added = board.add("stretch", "in 5 minutes")
            self.assertTrue(added["ok"], added)
            self.assertIn("stretch", added["message"])
            board.add("stand up", "in 1 minutes")
            self.assertEqual(len(board.list()), 2)
            self.assertTrue(store.is_file())
            self.assertEqual(len(reminders.ReminderBoard(path=store).list()), 2)
            first = board.list()[0]
            self.assertLessEqual(first["due"] - now, 400)
            self.assertEqual(board.due_now(), [], "nothing is due yet")
            board.snooze("stretch", 30)
            self.assertGreater(board.list()[-1]["due"], first["due"])
            self.assertTrue(board.cancel("stand up")["ok"])
            self.assertEqual(len(board.list()), 1)
            board._items["now"] = reminders.Reminder(rid="now", text="check the download folder",
                                                     due=time.time() - 1, run=True)
            due = board.due_now()
            self.assertEqual([item.rid for item in due], ["now"])
            self.assertTrue(due[0].run, "an action item must be run, not only announced")
            board._fire(due[0])
            self.assertEqual(len(fired), 1)
            self.assertEqual(len(board.list()), 1, "one-shots are dropped after firing")

    def test_a_bad_time_is_reported_by_add(self):
        with tempfile.TemporaryDirectory() as tmp:
            board = reminders.ReminderBoard(path=Path(tmp) / "r.json")
            result = board.add("nothing", "")
            self.assertFalse(result["ok"])
            self.assertEqual(len(board.list()), 0)


class TestDesktopWiring(unittest.TestCase):
    """Wiring is where this round could break silently: schemas, registry, routes, assets."""

    def test_registry_and_schema_list_agree(self):
        names = set(router.TOOL_NAMES)
        self.assertEqual(names, set(tools.TOOL_FUNCTIONS), "every tool needs a schema and a function")
        documented = {spec["function"]["name"] for spec in router.TOOL_SCHEMAS}
        self.assertEqual(documented, names)
        for name in ("manage_files", "control_desktop", "read_screen", "set_reminder", "focus_app",
                     "list_apps", "windows_on_screen", "listening", "launch_app", "close_app"):
            self.assertIn(name, names)

    def test_the_new_tools_answer_through_execute_tool(self):
        for call in (("list_apps", {"query": ""}), ("manage_files", {"action": "list"}),
                     ("set_reminder", {"action": "list"}), ("listening", {"status": "status"}),
                     ("windows_on_screen", {}), ("read_screen", {"action": "read", "count": "2"})):
            result = tools.execute_tool(call[0], call[1])
            self.assertIn("ok", result, call)
            self.assertGreater(len(str(result.get("message", ""))), 3, call)
        self.assertFalse(tools.execute_tool("manage_files", {"action": "list", "path": "no/such/dir-91827364"})["ok"])

    def test_an_invented_action_is_rejected_before_anything_runs(self):
        tool, args, error = router.validate_call("manage_files", {"action": "format-hard-drive"})
        self.assertIsNone(tool)
        self.assertIn("not allowed", error)
        tool, args, error = router.validate_call("control_desktop", {"action": "reboot-everything"})
        self.assertIsNone(tool)
        self.assertTrue(error)

    def test_instant_rules_cover_the_new_powers(self):
        cases = {
            "create a file called ideas.md with milk and eggs": ("files-say", "manage_files"),
            "delete notes.txt": ("files-delete", "manage_files"),
            "read shopping.txt": ("files-read", "manage_files"),
            "list my files": ("files-list", "manage_files"),
            "undo that": ("files-undo", "manage_files"),
            "what am I looking at": ("screen-read", "read_screen"),
            "read out top three results": ("screen-top", "read_screen"),
            "type hello world into the search box": ("desktop-type", "control_desktop"),
            "type \"hello world\"": ("desktop-type", "control_desktop"),
            "press escape": ("desktop-press", "control_desktop"),
            "minimize": ("desktop-window", "control_desktop"),
            "remind me to stretch in ten minutes": ("remind-add", "set_reminder"),
            "cancel the timer": ("remind-cancel", "set_reminder"),
            "start listening": ("listening", "listening"),
            "list apps": ("apps-ask", "list_apps"),
            "switch to spotify": ("focus-app", "focus_app"),
        }
        for text, (rule, tool) in cases.items():
            plan = server.match_instant(text)
            self.assertIsNotNone(plan, text)
            self.assertEqual(plan.get("rule"), rule, text)
            self.assertEqual(plan["calls"][0]["tool"], tool, text)

    def test_a_chained_open_plus_read_is_not_hijacked(self):
        plan = server.match_instant("open google and read the top 3 results")
        self.assertEqual((plan or {}).get("rule"), "open", plan)

    def test_a_poem_is_not_mistaken_for_a_file(self):
        self.assertNotEqual((server.match_instant("write a poem about rain") or {}).get("rule"), "files-say")

    def test_reminders_fire_through_the_command_funnel(self):
        self.assertTrue(callable(server.queue_spoken_command))
        self.assertFalse(server.queue_spoken_command("")["ok"])

    def test_desktop_endpoints_are_served(self):
        if not HTTP_OK:
            self.skipTest("httpx not installed")
        client = TestClient(server.create_app())
        for path in ("/api/desktop", "/api/reminders", "/api/status"):
            self.assertEqual(client.get(path).status_code, 200, path)
        self.assertTrue(client.get("/api/desktop").json()["ok"])
        response = client.post("/api/bar", json={"action": "toggle"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(bool(response.json()["ok"]), bool(getattr(config, "BAR_CONTROLLER", None)))
        self.assertEqual(client.post("/api/listening", json={"action": "status"}).status_code, 200)
        self.assertEqual(client.get("/bar").status_code, 200)

    def test_the_bar_assets_really_wire_the_core(self):
        html = (config.ROOT / "static" / "bar.html").read_text(encoding="utf-8")
        script = (config.ROOT / "static" / "bar.js").read_text(encoding="utf-8")
        for needle in ("/api/command", "/api/listen", "/api/listening", "/api/bar"):
            self.assertIn(needle, script, needle)
        self.assertIn("bar.js", html)
        self.assertNotIn("TODO", html + script)
        source = (config.ROOT / "main.py").read_text(encoding="utf-8")
        for needle in ("BAR_CONTROLLER", "start_background_ear", "frameless=True", "on_top=True"):
            self.assertIn(needle, source, needle)

    def test_the_ladder_only_sends_images_to_vision_models(self):
        status = llm_providers.POOL.status()
        for row in status.get("providers", []):
            self.assertIn("vision_models", row, row.get("name"))
        source = (config.ROOT / "llm_providers.py").read_text(encoding="utf-8")
        self.assertIn("def vision(", source)
        self.assertIn("no image input on this API style", source)
        self.assertIn("input_modalities", source)

    def test_the_offline_planner_routes_the_new_tools(self):
        for text, tool in (("what am I looking at", "read_screen"),
                           ("remind me to call mom in twenty minutes", "set_reminder"),
                           ("undo that", "manage_files"),
                           ("open bluetooth settings", "launch_app"),
                           ("read out the top 5 listings", "read_screen"),
                           ("what windows are open right now", "windows_on_screen"),
                           ("delete the file called ideas.md", "manage_files")):
            calls = router.heuristic_plan(text, allow_search=False)
            self.assertTrue(any(c["tool"] == tool for c in calls), (text, calls))

if __name__ == "__main__":
    unittest.main(verbosity=2)
