from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AvatarState:
    emotion: str = "neutral"
    speaking: bool = False
    listening: bool = False
    viseme: str = "rest"


class AvatarController:
    def __init__(self) -> None:
        self.state = AvatarState()

    def set_emotion(self, emotion: str) -> None:
        self.state.emotion = emotion

    def set_speaking(self, speaking: bool) -> None:
        self.state.speaking = speaking

    def set_listening(self, listening: bool) -> None:
        self.state.listening = listening

    def set_viseme(self, viseme: str) -> None:
        self.state.viseme = viseme
