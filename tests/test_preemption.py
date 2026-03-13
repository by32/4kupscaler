"""Tests for preemption signal handling."""

from __future__ import annotations

import signal

import pytest

from upscaler.cloud.preemption import PreemptionError, PreemptionHandler


class TestPreemptionHandler:
    def test_not_preempted_initially(self):
        handler = PreemptionHandler()
        assert not handler.is_preempted

    def test_sets_flag_on_sigterm(self):
        handler = PreemptionHandler()
        handler.install()
        try:
            # Simulate SIGTERM by calling the handler directly
            handler._handle_signal(signal.SIGTERM, None)
            assert handler.is_preempted
        finally:
            handler.uninstall()

    def test_check_or_raise(self):
        handler = PreemptionHandler()
        handler.check_or_raise()  # Should not raise

        handler._preempted.set()
        with pytest.raises(PreemptionError, match="Preemption signal"):
            handler.check_or_raise()

    def test_install_uninstall_restores_handlers(self):
        original_term = signal.getsignal(signal.SIGTERM)

        handler = PreemptionHandler()
        handler.install()

        # Handler should be changed
        current = signal.getsignal(signal.SIGTERM)
        assert current == handler._handle_signal

        handler.uninstall()

        # Should be restored
        restored = signal.getsignal(signal.SIGTERM)
        assert restored == original_term

    def test_multiple_signals_only_set_once(self):
        handler = PreemptionHandler()
        handler._handle_signal(signal.SIGTERM, None)
        handler._handle_signal(signal.SIGTERM, None)
        assert handler.is_preempted
