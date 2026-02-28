from __future__ import annotations

from spark.config.settings import PersonalitySettings


class PersonalityEngine:
    def __init__(self, settings: PersonalitySettings) -> None:
        self.settings = settings

    def style_response(self, base_text: str) -> str:
        if self.settings.formality > 0.7:
            prefix = "Certainly. "
        elif self.settings.humor > 0.7:
            prefix = "Got it 😄 "
        else:
            prefix = "Sure. "
        return prefix + base_text
