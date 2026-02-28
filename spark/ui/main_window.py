from __future__ import annotations

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QProgressBar, QTextEdit, QVBoxLayout, QWidget

from spark.hardware_control.monitor import HardwareMonitor


class SparkMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SPARK - Intelligent Companion")
        self.resize(980, 640)

        container = QWidget()
        layout = QVBoxLayout(container)

        self.avatar = QLabel("🧠 Avatar Online")
        self.wave = QProgressBar()
        self.wave.setRange(0, 100)
        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        self.stats = QLabel("CPU: -- | RAM: -- | Battery: --")

        layout.addWidget(self.avatar)
        layout.addWidget(self.wave)
        layout.addWidget(self.chat)
        layout.addWidget(self.stats)
        self.setCentralWidget(container)

        self.monitor = HardwareMonitor()
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_stats)
        self.timer.start(1000)

    def add_message(self, message: str) -> None:
        self.chat.append(message)

    def refresh_stats(self) -> None:
        s = self.monitor.snapshot()
        self.stats.setText(
            f"CPU: {s['cpu_percent']:.0f}% | RAM: {s['ram_percent']:.0f}% | Battery: {s['battery_percent']:.0f}%"
        )


def run_ui() -> None:
    app = QApplication([])
    win = SparkMainWindow()
    win.show()
    app.exec()
