"""The local driver has to actually exit.

`ptinfra.intialize` registers a **non-daemon** `ElasticQueueThread`, so a
normal return from `process_gallery.py` unwinds the main thread and then
blocks forever waiting for that one. The album is already written by then, so
the run looks finished and simply never ends -- and each one leaves an
interpreter alive holding the whole photo table and its embeddings. Fifteen
albums in a session exhausted the machine's memory twice.

This guards the escape hatch, not the hang: the service keeps its queue
thread, and `main.py` does not call any of this.

    python -m pytest tests/test_process_gallery_exit.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import process_gallery  # noqa: E402


@pytest.fixture
def spy(monkeypatch):
    """Capture the exit code and prove the streams were flushed first."""
    state = {"code": None, "flushed": []}

    monkeypatch.setattr(process_gallery.os, "_exit",
                        lambda code: state.__setitem__("code", code))
    monkeypatch.setattr(process_gallery.sys.stdout, "flush",
                        lambda: state["flushed"].append("stdout"))
    monkeypatch.setattr(process_gallery.sys.stderr, "flush",
                        lambda: state["flushed"].append("stderr"))
    return state


def test_a_finished_album_exits_zero(spy):
    process_gallery._exit_now(0)

    assert spy["code"] == 0


def test_a_failed_run_exits_nonzero(spy, capsys):
    """It replaced a `raise SystemExit`, which reported the failure but did not
    end the process any sooner."""
    process_gallery._exit_now(1, "process_gallery failed: boom")

    assert spy["code"] == 1
    assert "boom" in capsys.readouterr().err


def test_both_streams_are_flushed_before_exiting():
    """`os._exit` skips interpreter shutdown -- no `atexit`, no buffer flush --
    so anything still buffered would be lost. This is the whole reason the
    helper exists rather than a bare `os._exit` at the call sites."""
    order = []

    class Recorder:
        def __init__(self, name):
            self.name = name

        def flush(self):
            order.append(self.name)

    real_exit, real_out, real_err = (process_gallery.os._exit,
                                     process_gallery.sys.stdout,
                                     process_gallery.sys.stderr)
    process_gallery.os._exit = lambda code: order.append("exit")
    process_gallery.sys.stdout = Recorder("stdout")
    process_gallery.sys.stderr = Recorder("stderr")
    try:
        process_gallery._exit_now(0)
    finally:
        process_gallery.os._exit = real_exit
        process_gallery.sys.stdout = real_out
        process_gallery.sys.stderr = real_err

    assert order == ["stdout", "stderr", "exit"]


def test_the_service_does_not_use_it():
    """The queue thread is the point in production; only the local driver may
    cut it short."""
    import inspect

    import main

    assert "_exit_now" not in inspect.getsource(main)
    assert not hasattr(main, "_exit_now")
