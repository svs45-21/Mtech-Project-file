from __future__ import annotations

from pathlib import Path

from spark.config.config_manager import ConfigManager
from spark.config.settings import AppSettings


def run_setup() -> None:
    print("=== Spark First-Time Setup Wizard ===")
    wake_word = input("Wake word [Hey Spark]: ").strip() or "Hey Spark"
    app = AppSettings()
    app.voice.wake_word = wake_word

    manager = ConfigManager(Path("data/config.enc"), Path("data/key.bin"))
    manager.save(app)
    print("Configuration encrypted and saved to data/config.enc")


if __name__ == "__main__":
    run_setup()
