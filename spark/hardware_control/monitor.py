from __future__ import annotations

import psutil


class HardwareMonitor:
    def snapshot(self) -> dict[str, float]:
        battery = psutil.sensors_battery()
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.2),
            "ram_percent": psutil.virtual_memory().percent,
            "battery_percent": battery.percent if battery else -1.0,
        }
