from __future__ import annotations

from pathlib import Path
from datetime import datetime


class AuditLogger:
    def __init__(self, audit_path: Path) -> None:
        self.audit_path = audit_path
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, actor: str, action: str, outcome: str) -> None:
        stamp = datetime.utcnow().isoformat()
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(f"{stamp}|{actor}|{action}|{outcome}\n")
