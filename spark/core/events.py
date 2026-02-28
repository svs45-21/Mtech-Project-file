from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, DefaultDict, List

EventHandler = Callable[[Any], None]


class EventBus:
    def __init__(self) -> None:
        self._handlers: DefaultDict[str, List[EventHandler]] = defaultdict(list)

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        self._handlers[event_name].append(handler)

    def publish(self, event_name: str, payload: Any) -> None:
        for handler in self._handlers[event_name]:
            handler(payload)
