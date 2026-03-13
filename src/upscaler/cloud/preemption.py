"""Signal handling for graceful shutdown on vast.ai preemption.

Vast.ai sends SIGTERM to interruptible instances before stopping them.
This module provides a handler that sets a flag the engine checks
between segments, allowing clean checkpoint saves.
"""

from __future__ import annotations

import logging
import signal
import threading
from typing import Any

logger = logging.getLogger(__name__)


class PreemptionError(Exception):
    """Raised when a preemption signal is received."""


class PreemptionHandler:
    """Catches SIGTERM/SIGINT and exposes a flag for cooperative shutdown.

    Usage::

        handler = PreemptionHandler()
        handler.install()
        try:
            for segment in segments:
                if handler.is_preempted:
                    save_state()
                    break
                process(segment)
        finally:
            handler.uninstall()
    """

    def __init__(self) -> None:
        self._preempted = threading.Event()
        self._original_handlers: dict[int, Any] = {}

    def install(self) -> None:
        """Register SIGTERM and SIGINT handlers."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._original_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle_signal)
        logger.debug("Preemption handler installed")

    def uninstall(self) -> None:
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)
        self._original_handlers.clear()
        logger.debug("Preemption handler uninstalled")

    @property
    def is_preempted(self) -> bool:
        """True if a preemption signal has been received."""
        return self._preempted.is_set()

    def check_or_raise(self) -> None:
        """Raise PreemptionError if preempted."""
        if self._preempted.is_set():
            raise PreemptionError("Preemption signal received")

    def _handle_signal(self, signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.warning("Received %s — flagging for graceful shutdown", sig_name)
        self._preempted.set()
