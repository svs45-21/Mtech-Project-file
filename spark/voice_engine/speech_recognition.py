from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import speech_recognition as sr

logger = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    text: str
    source: str


class SpeechRecognizer:
    def __init__(self, wake_word: str = "hey spark") -> None:
        self.wake_word = wake_word.lower()
        self._recognizer = sr.Recognizer()

    def listen_once(self, timeout: int = 4) -> Optional[RecognitionResult]:
        with sr.Microphone() as source:
            audio = self._recognizer.listen(source, timeout=timeout)
        try:
            text = self._recognizer.recognize_google(audio)
            return RecognitionResult(text=text, source="online")
        except Exception as exc:
            logger.warning("Online recognition failed: %s", exc)
            return None

    def is_wake_word(self, phrase: str) -> bool:
        return self.wake_word in phrase.lower()
