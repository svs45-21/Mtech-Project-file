from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field


class SecuritySettings(BaseModel):
    require_admin_auth: bool = True
    critical_action_confirmation: bool = True
    audit_log_path: Path = Path("./data/audit.log")


class VoiceSettings(BaseModel):
    wake_word: str = "Hey Spark"
    speech_rate: float = 1.0
    speech_pitch: float = 1.0
    humor_tone: float = 0.5
    confidence_tone: float = 0.7


class PersonalitySettings(BaseModel):
    humor: float = Field(default=0.4, ge=0.0, le=1.0)
    formality: float = Field(default=0.5, ge=0.0, le=1.0)
    proactiveness: float = Field(default=0.6, ge=0.0, le=1.0)
    talkativeness: float = Field(default=0.5, ge=0.0, le=1.0)
    expressiveness: float = Field(default=0.5, ge=0.0, le=1.0)


class AppSettings(BaseModel):
    app_name: str = "SPARK"
    data_dir: Path = Path("./data")
    voice: VoiceSettings = VoiceSettings()
    security: SecuritySettings = SecuritySettings()
    personality: PersonalitySettings = PersonalitySettings()
