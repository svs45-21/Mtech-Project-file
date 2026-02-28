from __future__ import annotations

import platform
import subprocess
from pathlib import Path


class SoftwareController:
    def open_application(self, app_name: str) -> None:
        system = platform.system()
        if system == "Windows":
            subprocess.Popen(["start", app_name], shell=True)
        elif system == "Darwin":
            subprocess.Popen(["open", "-a", app_name])
        else:
            subprocess.Popen([app_name])

    def run_command(self, command: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, shell=True, check=False, text=True, capture_output=True)

    def create_folder(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
