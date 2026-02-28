from __future__ import annotations

import asyncio


class WorkflowAutomation:
    async def send_progressive_updates(self, callback) -> None:
        for pct in [25, 50, 75, 100]:
            await asyncio.sleep(0.2)
            callback(f"Workflow progress: {pct}%")
