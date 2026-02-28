from __future__ import annotations

import platform
from pathlib import Path


def install_autostart(executable_path: Path) -> Path:
    system = platform.system()
    if system == "Windows":
        startup = Path.home() / "AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup/spark.bat"
        startup.write_text(f'@echo off\nstart "" "{executable_path}"\n', encoding="utf-8")
        return startup
    if system == "Darwin":
        plist = Path.home() / "Library/LaunchAgents/com.spark.assistant.plist"
        plist.write_text(
            f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\"> 
<plist version=\"1.0\"><dict><key>Label</key><string>com.spark.assistant</string><key>ProgramArguments</key><array><string>{executable_path}</string></array><key>RunAtLoad</key><true/></dict></plist>""",
            encoding="utf-8",
        )
        return plist

    service = Path.home() / ".config/systemd/user/spark.service"
    service.parent.mkdir(parents=True, exist_ok=True)
    service.write_text(
        f"""[Unit]
Description=Spark Assistant
[Service]
ExecStart={executable_path}
Restart=always
[Install]
WantedBy=default.target
""",
        encoding="utf-8",
    )
    return service
