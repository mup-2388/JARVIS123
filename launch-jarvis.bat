@echo off
REM ===========================================================================
REM  J.A.R.V.I.S. one-click launcher (Windows 10/11)
REM    launch-jarvis.bat              -> create venv if needed, install deps, run
REM    launch-jarvis.bat --no-window   -> run the core as a background service
REM    launch-jarvis.bat --reinstall   -> force a pip install pass
REM ===========================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "PY_CMD="
py -3.11 -c "pass" >nul 2>&1 && set "PY_CMD=py -3.11"
if not defined PY_CMD py -3 -c "pass" >nul 2>&1 && set "PY_CMD=py -3"
if not defined PY_CMD python -c "pass" >nul 2>&1 && set "PY_CMD=python"
if not defined PY_CMD (
  echo [JARVIS] No Python found. Install Python 3.11 from python.org, tick "Add to PATH", retry.
  exit /b 1
)

set "REINSTALL=0"
set "ARGS="
:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--reinstall" ( set "REINSTALL=1" ) else ( set "ARGS=!ARGS! %~1" )
shift
goto parse
:parsed

if not exist ".venv\Scripts\python.exe" (
  echo [JARVIS] Creating virtual environment...
  %PY_CMD% -m venv .venv || ( echo [JARVIS] venv creation failed & exit /b 1 )
  set "REINSTALL=1"
)
set "VPY=.venv\Scripts\python.exe"

if not exist "requirements.txt" (
  echo [JARVIS] requirements.txt is missing next to this script.
  exit /b 1
)

%VPY% -c "import fastapi,uvicorn,psutil" >nul 2>&1
if errorlevel 1 set "REINSTALL=1"
if "%REINSTALL%"=="1" (
  echo [JARVIS] Installing dependencies ^(this pulls torch/ctranslate2, give it a few minutes^)...
  "%VPY%" -m pip install --upgrade pip >nul
  "%VPY%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [JARVIS] pip failed. If TTS/faster-whisper are the problem, install CUDA wheels from
    echo          https://pytorch.org/get-started/locally/ then re-run this script.
    exit /b 1
  )
)

if not exist ".env" (
  echo [JARVIS] No .env found - copying .env.example. Add HF_TOKEN / DISCORD_TOKEN / API_SPORTS_KEY.
  copy /y ".env.example" ".env" >nul
  notepad ".env"
)

if not exist "assets\jarvis_sample.wav" (
  echo [JARVIS] assets\jarvis_sample.wav missing: XTTS voice cloning is off, the Windows voice is used.
)
where ffmpeg >nul 2>&1 || echo [JARVIS] ffmpeg not on PATH: browser microphone upload will be disabled.

title J.A.R.V.I.S. core
"%VPY%" main.py%ARGS%
endlocal
