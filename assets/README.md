# assets/

Place the XTTS-v2 voice reference here:

    assets/jarvis_sample.wav

Requirements for a good zero-shot clone (enforced by `audio_engine.TextToSpeech.check_reference()`):

| Property | Target |
| --- | --- |
| Length | 3–10 seconds (one or two full sentences) |
| Format | WAV (PCM 16-bit), mono |
| Sample rate | 22 050 Hz ideal; 16 kHz also works |
| Content | Calm, steady read of a technical paragraph — the tone you want JARVIS to speak in |
| Quality | No music, no reverb, no clipped peaks; -3 dBFS peak is plenty |

Quick capture on Windows PowerShell (5 s of the system voice, as a starting point you can overwrite later):

```powershell
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SetOutputToWaveFile("$PWD\assets\jarvis_raw.wav")
$s.Speak("Systems online. Voice synthesis calibrated and awaiting your command, sir.")
$s.Dispose()
ffmpeg -i assets\jarvis_raw.wav -ac 1 -ar 22050 -t 8 assets\jarvis_sample.wav
```

`TTS_REFERENCE_WAV` in `.env` points elsewhere if you prefer another location. The file is
git-ignored on purpose — it is your voice print, not source code. If it is missing the engine logs
`XTTS offline` and every reply falls back to the built-in Windows SAPI5 voice, so nothing breaks.
