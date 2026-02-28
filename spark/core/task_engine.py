from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List

logger = logging.getLogger(__name__)

TaskStep = Callable[[], Awaitable[None]]


@dataclass
class PlannedTask:
    name: str
    steps: List[TaskStep]
    status: str = "pending"
    progress: float = 0.0
    metadata: Dict[str, str] = field(default_factory=dict)


class TaskExecutionEngine:
    def __init__(self) -> None:
        self._active: Dict[str, asyncio.Task[None]] = {}

    async def execute(self, task: PlannedTask) -> None:
        logger.info("Executing task: %s", task.name)
        task.status = "running"
        total = len(task.steps)
        for i, step in enumerate(task.steps, start=1):
            await step()
            task.progress = i / total
            logger.info("Task %s progress %.2f%%", task.name, task.progress * 100)
        task.status = "completed"

    def run_background(self, task: PlannedTask) -> None:
        self._active[task.name] = asyncio.create_task(self.execute(task))

    def cancel(self, task_name: str) -> bool:
        pending = self._active.get(task_name)
        if not pending:
            return False
        pending.cancel()
        return True
