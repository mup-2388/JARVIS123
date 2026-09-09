"""
tests/test_jarvis.py -- self-checks for the JARVIS core.

Stdlib ``unittest`` only, so it runs on a bare Windows box without pytest:

    python -m unittest discover -s tests -v
    python -m unittest tests.test_jarvis.TestInstantTrack -v

The HTTP/REST cases use FastAPI's ``TestClient`` and skip themselves if
``httpx`` is not installed (it is pulled in by ``huggingface-hub[inference]``,
so a normal install always has it).
"""

from __future__ import annotations

import json
import os
import sys
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
            "# comment\nexport HF_TOKEN = 'secret-token' \nNOTES_DIR=notes\n\nQUOTED=\"has = sign\"\n",
            encoding="utf-8",
        )
        parsed = config.parse_env_file(tmp)
        tmp.unlink()
        self.assertEqual(parsed["HF_TOKEN"], "secret-token")
        self.assertEqual(parsed["NOTES_DIR"], "notes")
        self.assertEqual(parsed["QUOTED"], "has = sign")

    def test_redacted_never_leaks_secrets(self):
        blob = json.dumps(config.SETTINGS.redacted())
        for secret in (config.SETTINGS.hf_token, config.SETTINGS.discord_token, config.SETTINGS.api_sports_key):
            if secret:
                self.assertNotIn(secret, blob, "a raw secret reached the HUD-safe view")
        self.assertIn("hf_token_set", blob)

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

    def test_unknown_app_is_deferred_to_the_agent(self):
        self.assertIsNone(server.match_instant("open the next big thing"))
        self.assertIsNone(server.match_instant("explain the difference between a coroutine and a thread"))
        # ...but a factual "what is X" question is a search, which Track 1 owns.
        self.assertEqual(server.match_instant("what is the capital of France")["rule"], "search")

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
        response = self.client.post(
            "/api/transcribe", files={"file": ("clip.wav", buffer.getvalue(), "audio/wav")}, data={"speak": "false"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
