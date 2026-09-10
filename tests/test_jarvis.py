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

import json
import os
import re
import shutil
import sys
import tempfile
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


class TestLlmProviders(unittest.TestCase):
    """Pool rotation, quota cooldowns and tier selection -- all offline."""

    def setUp(self):
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
    """A retired or not-entitled model id must never cost the user the answer."""

    VISIBLE = ("llama-3.1-8b-instant", "llama-guard-4-12b", "whisper-large-v3-turbo",
               "playai-tts Array", "meta-llama/llama-4-scout-17b-16e-instruct",
               "gpt-oss-120b", "qwen/qwen3-32b", "text-embedding-3-small")

    def setUp(self):
        import llm_providers as lp

        self.lp = lp
        self._dir = tempfile.mkdtemp(prefix="jarvis-model-")
        self.state = Path(self._dir) / "llm.json"
        self.pool = lp.LlmPool(state_file=self.state)
        self._post, self._get = lp._post_json, lp._get_json
        self._real_pool = lp.POOL
        lp.POOL = self.pool
        self._env = {}
        for name in ("GROQ_API_KEY", "CEREBRAS_API_KEY", "GEMINI_API_KEY", "MISTRAL_API_KEY",
                     "OPENROUTER_API_KEY", "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID",
                     "GITHUB_MODELS_TOKEN", "CUSTOM_LLM_BASE_URL", "LLM_AUTO_DISCOVER"):
            self._env[name] = os.environ.get(name)
        os.environ["GROQ_API_KEY"] = "gsk-test"
        for name in self._env:
            if name != "GROQ_API_KEY":
                os.environ.pop(name, None)
        # discovery is on by default; keep it deterministic and offline
        self.list_calls = []
        lp._get_json = self._fake_get

    def tearDown(self):
        for name, value in self._env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self.lp._post_json, self.lp._get_json = self._post, self._get
        self.lp.POOL = self._real_pool
        shutil.rmtree(self._dir, ignore_errors=True)

    def _fake_get(self, url, headers, timeout):
        self.list_calls.append(url)
        return 200, {}, {"data": [{"id": i} for i in self.VISIBLE]}

    def _fake_post(self, ok_models, status=404, message="The model `%s` does not exist or you do not have access to it."):
        sent = []

        def post(url, headers, payload, timeout):
            model = payload.get("model", "")
            sent.append(model)
            if model in ok_models:
                return 200, {}, {"choices": [{"finish_reason": "stop",
                                               "message": {"role": "assistant", "content": f"answered by {model}"}}]}
            return status, {}, {"error": {"message": message % model}}

        self.lp._post_json = post
        return sent

    def test_refused_model_is_replaced_inside_the_same_turn(self):
        # Groq's advertised smart model 404s on this account, exactly the reported failure
        sent = self._fake_post({"llama-3.1-8b-instant"})
        out = self.pool.complete([{"role": "user", "content": "hi"}], tier="smart")
        self.assertEqual(out["content"], "answered by llama-3.1-8b-instant")
        self.assertTrue(out["model_switched"], "the caller must be able to see that a swap happened")
        self.assertEqual(sent.count("llama-3.3-70b-versatile"), 1,
                         "a refused model must not be hammered twice in one turn")
        self.assertNotEqual(self.pool.model_for("groq", "smart"), "llama-3.3-70b-versatile")

    def test_discovery_reads_the_key_list_once_and_ignores_non_chat_models(self):
        self._fake_post({"llama-3.1-8b-instant"})
        self.pool.complete([{"role": "user", "content": "hi"}], tier="smart")
        found = self.pool._health_for("groq")["discovered"]      # noqa: SLF001
        self.assertEqual(len(self.list_calls), 1, "one /models read per TTL, not one per turn")
        self.assertEqual(found["fast"], "llama-3.1-8b-instant")
        self.assertEqual(found["smart"], "gpt-oss-120b", "biggest real chat model, not the 17B multimodal")
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
        self.assertNotEqual(reloaded.model_for("groq", "smart"), "llama-3.3-70b-versatile",
                            "a restart must not spend the first turn on a known-dead id")
        self.assertTrue(reloaded.status()["providers"][0]["rejected_models"])

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
        self._fake_post({"llama-3.1-8b-instant"})
        self.pool.complete([{"role": "user", "content": "hi"}], tier="smart")
        provider = self.pool.status()["providers"][0]
        self.assertEqual(provider["models_seen"], len(self.VISIBLE))
        self.assertIn("llama-guard-4-12b", provider["model_ids"], "the raw list is exposed for the HUD")
        self.assertIn("llama-3.3-70b-versatile", provider["rejected_models"])



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


if __name__ == "__main__":
    unittest.main(verbosity=2)
