from __future__ import annotations

import logging

from spark.avatar_engine.avatar_controller import AvatarController
from spark.config.config_manager import ConfigManager
from spark.core.events import EventBus
from spark.personality.engine import PersonalityEngine
from spark.security.audit import AuditLogger
from spark.security.command_validator import CommandValidator
from spark.software_control.system_controller import SoftwareController
from spark.voice_engine.speech_recognition import SpeechRecognizer
from spark.voice_engine.tts_engine import TextToSpeechEngine

logger = logging.getLogger(__name__)


class SparkAssistant:
    def __init__(self, config: ConfigManager) -> None:
        self.settings = config.load()
        self.events = EventBus()
        self.avatar = AvatarController()
        self.personality = PersonalityEngine(self.settings.personality)
        self.recognizer = SpeechRecognizer(self.settings.voice.wake_word)
        self.tts = TextToSpeechEngine()
        self.validator = CommandValidator()
        self.software = SoftwareController()
        self.audit = AuditLogger(self.settings.security.audit_log_path)

    def greet(self) -> str:
        line = "Good morning. I noticed your calendar may be full. Should I prepare your workspace?"
        spoken = self.personality.style_response(line)
        self.avatar.set_speaking(True)
        self.tts.speak(spoken)
        self.avatar.set_speaking(False)
        self.audit.log("assistant", "greet", "ok")
        return spoken

    def handle_text_command(self, command: str) -> str:
        verdict = self.validator.validate(command)
        if not verdict.allowed:
            self.audit.log("user", command, "blocked")
            return f"I can't do that safely: {verdict.reason}"

        if command.startswith("run:"):
            proc = self.software.run_command(command.removeprefix("run:").strip())
            outcome = proc.stdout.strip() or proc.stderr.strip() or "Done"
            self.audit.log("user", command, "executed")
            return outcome

        return "Command recognized, but no handler is attached yet."
