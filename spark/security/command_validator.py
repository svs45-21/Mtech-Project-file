from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ValidationResult:
    allowed: bool
    reason: str


class CommandValidator:
    BLOCKLIST = ["rm -rf /", "format", "dd if="]

    def validate(self, command: str) -> ValidationResult:
        lowered = command.lower()
        for blocked in self.BLOCKLIST:
            if blocked in lowered:
                return ValidationResult(False, f"Refused unsafe command pattern: {blocked}")
        return ValidationResult(True, "Command accepted")
