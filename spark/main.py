from __future__ import annotations

import argparse
import logging
from pathlib import Path

from spark.config.config_manager import ConfigManager
from spark.core.assistant import SparkAssistant
from spark.core.logging_config import setup_logging
from spark.startup.autostart import install_autostart
from spark.ui.main_window import run_ui


def cli() -> None:
    parser = argparse.ArgumentParser(description="SPARK Intelligent Personalized AI Operating Companion")
    parser.add_argument("--ui", action="store_true", help="Launch desktop UI")
    parser.add_argument("--setup", action="store_true", help="Run setup wizard")
    parser.add_argument("--install-autostart", action="store_true", help="Install startup entry")
    parser.add_argument("--command", type=str, help="Execute single text command")
    args = parser.parse_args()

    setup_logging(Path("data/logs"), logging.INFO)
    manager = ConfigManager(Path("data/config.enc"), Path("data/key.bin"))

    if args.setup:
        from spark.scripts.setup_wizard import run_setup

        run_setup()
        return

    if args.install_autostart:
        entry = install_autostart(Path(__file__).resolve())
        print(f"Autostart installed: {entry}")
        return

    assistant = SparkAssistant(manager)
    print(assistant.greet())

    if args.command:
        print(assistant.handle_text_command(args.command))

    if args.ui:
        run_ui()


if __name__ == "__main__":
    cli()
