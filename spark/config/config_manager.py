from __future__ import annotations

import json
from pathlib import Path
from cryptography.fernet import Fernet

from spark.config.settings import AppSettings


class ConfigManager:
    def __init__(self, config_path: Path, key_path: Path) -> None:
        self.config_path = config_path
        self.key_path = key_path

    def _load_or_create_key(self) -> bytes:
        if self.key_path.exists():
            return self.key_path.read_bytes()
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        key = Fernet.generate_key()
        self.key_path.write_bytes(key)
        return key

    def save(self, settings: AppSettings) -> None:
        key = self._load_or_create_key()
        cipher = Fernet(key)
        payload = settings.model_dump_json(indent=2).encode("utf-8")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_bytes(cipher.encrypt(payload))

    def load(self) -> AppSettings:
        if not self.config_path.exists():
            settings = AppSettings()
            self.save(settings)
            return settings
        key = self._load_or_create_key()
        cipher = Fernet(key)
        raw = cipher.decrypt(self.config_path.read_bytes())
        return AppSettings.model_validate(json.loads(raw.decode("utf-8")))
