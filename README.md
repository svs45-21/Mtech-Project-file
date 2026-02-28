# 🔥 SPARK – Intelligent Personalized AI Operating Companion

SPARK is a modular Python desktop assistant architecture designed for **Windows, macOS, and Linux** with voice, avatar, automation, and security-first controls.

## 1) Architecture Overview

SPARK follows a clean modular architecture where each concern is isolated:

- `core/`: orchestration, events, task execution, logging
- `voice_engine/`: wake word + ASR + TTS + voice profile pipeline
- `avatar_engine/`: avatar state and lip-sync hooks
- `ui/`: PyQt6 desktop shell + monitoring dashboard
- `software_control/`: app launching + shell execution wrappers
- `hardware_control/`: CPU/RAM/Battery telemetry
- `security/`: command validation, auth, audit logging
- `personality/`: adaptive style controls
- `integrations/`: weather/news/email/calendar API adapters
- `startup/`: OS autostart installers

## 2) Folder Structure

```text
spark/
├── automation/
├── avatar_engine/
├── config/
├── core/
├── hardware_control/
├── integrations/
├── personality/
├── plugins/
├── scripts/
├── security/
├── software_control/
├── startup/
├── ui/
├── voice_engine/
└── main.py
```

## 3) Core Capabilities Implemented

- Encrypted config storage (`Fernet`) with structured settings.
- Voice command plumbing with configurable wake-word checks.
- Text-to-speech engine + tuning hooks.
- Voice cloning storage pipeline (embedding stub for Coqui/XTTS integration).
- Safety command validator blocking destructive patterns.
- Audit logs for every sensitive action.
- Async task execution engine with cancellation.
- Autostart registration for Windows/macOS/Linux.

## 4) Avatar Engine (Implementation Notes)

`avatar_engine/avatar_controller.py` provides live runtime state:
- `emotion`
- `speaking`
- `listening`
- `viseme`

This state is designed as the source of truth for a real renderer.
For production rendering:
- 2D: integrate Live2D SDK renderer binding.
- 3D: bind GLB/FBX runtime with `PyOpenGL` + Qt OpenGL widget.
- Sync visemes by mapping phoneme stream from TTS pipeline.

## 5) Voice Engine (Implementation Notes)

- `speech_recognition.py`: microphone listen + online recognition fallback path.
- `tts_engine.py`: local fallback using `pyttsx3`.
- `voice_cloning.py`: secure voice profile file generation entrypoint.

Production extension options:
- ASR: Whisper (faster-whisper), Vosk offline model.
- TTS: Coqui XTTS / Piper / ElevenLabs hybrid fallback.
- Spoofing defense: speaker verification before privileged commands.

## 6) Security Layer

- `security/command_validator.py`: hard blocklist for dangerous commands.
- `security/auth.py`: admin PIN authentication hook.
- `security/audit.py`: immutable-style audit trail output file.
- Critical actions should require explicit user confirmation in UI workflow.

## 7) UI Implementation

`ui/main_window.py` includes:
- Avatar status region
- Live speech waveform placeholder
- Chat transcript panel
- CPU/RAM/Battery panel
- 1-second refresh telemetry loop

You can expand this with:
- QSS dark futuristic theme
- animated glow ring while listening
- OpenGL avatar viewport at 60 FPS timer

## 8) Installation Guide

### Prerequisites
- Python 3.11+
- PortAudio system packages (for microphone use)

### Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### First-time setup wizard
```bash
python -m spark.main --setup
```

### Launch
```bash
python -m spark.main --ui
```

## 9) Startup Integration Guide

```bash
python -m spark.main --install-autostart
```

Generated artifacts:
- Windows: Startup `.bat`
- macOS: `~/Library/LaunchAgents/com.spark.assistant.plist`
- Linux: `~/.config/systemd/user/spark.service`

After Linux install:
```bash
systemctl --user daemon-reload
systemctl --user enable --now spark.service
```

## 10) Packaging (PyInstaller)

```bash
pyinstaller --name spark --onefile -m spark.main
```

If UI assets are added, include them via `--add-data` mappings.

## 11) Avatar Upload Guide

- 2D avatar: store layered expressions under `data/avatar/2d/`
- 3D avatar: store `.glb`/`.fbx` under `data/avatar/3d/`
- Register metadata in encrypted config and link default emotion mappings.

## 12) Voice Training Guide

- Place voice sample WAV files under `data/voice_samples/`
- Call `VoiceCloningService.create_embedding(sample_path)`
- Store returned embedding ID in encrypted profile config.

## 13) Security Configuration Guide

- Set strong admin PIN via env var:
```bash
export SPARK_ADMIN_PIN="change-me"
```
- Rotate encryption key (`data/key.bin`) per environment.
- Forward logs to SIEM in enterprise deployments.

## 14) Extension / Plugin Guide

Create plugin modules in `spark/plugins/` implementing callable command handlers.
Recommended contract:
- `name: str`
- `can_handle(command: str) -> bool`
- `execute(command: str) -> str`

Register them in core assistant boot pipeline.

## 15) Future Scalability Roadmap

- Add local LLM orchestration (offline + cloud fallback).
- Add long-term memory via vector DB (Chroma/FAISS/pgvector).
- Add policy engine for role-based action scopes.
- Add signed plugin marketplace with sandbox execution.
- Add distributed task workers for heavier automations.
