from __future__ import annotations

import pyttsx3


class TextToSpeechEngine:
    def __init__(self) -> None:
        self.engine = pyttsx3.init()

    def configure(self, speed: float, pitch: float) -> None:
        self.engine.setProperty("rate", int(180 * speed))
        self.engine.setProperty("pitch", pitch)

    def speak(self, text: str) -> None:
        self.engine.say(text)
        self.engine.runAndWait()
