from __future__ import annotations

import hashlib
from pathlib import Path


class VoiceCloningService:
    """Production hook point for XTTS/Tortoise/Coqui integration."""

    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def create_embedding(self, sample_path: Path) -> str:
        sample_hash = hashlib.sha256(sample_path.read_bytes()).hexdigest()
        embedding_file = self.storage_dir / f"{sample_hash}.voice"
        embedding_file.write_text(f"voice_embedding:{sample_hash}", encoding="utf-8")
        return sample_hash
