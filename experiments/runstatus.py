# This code is part of qcr.
#
# (C) Copyright Arnav Kapoor 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Status file for long experiment runs, read by watchdog.py.

A run calls start(name, log=...) once, phase(...) whenever it begins a step that has a known worst case,
and finish() at the end. A daemon thread rewrites the file every HEARTBEAT seconds, so a frozen process
(or one stuck in C code that holds the GIL) shows up as a stale heartbeat, a dead one as a dead pid, and a
step that ran past its limit as an overrun. Everything is written atomically to results/status/<name>.json.
"""

import atexit
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "results" / "status"
HEARTBEAT = 60

_S = {}
_LOCK = threading.Lock()


def _write():
    with _LOCK:
        if not _S:
            return
        _S["heartbeat"] = time.time()
        tmp = DIR / f".{_S['name']}.tmp"
        tmp.write_text(json.dumps(_S, indent=1))
        os.replace(tmp, DIR / f"{_S['name']}.json")


def _beat():
    while True:
        time.sleep(HEARTBEAT)
        _write()


def _excepthook(kind, value, tb):
    with _LOCK:
        _S.update(state="crashed", error="".join(traceback.format_exception(kind, value, tb))[-3000:])
    _write()
    sys.__excepthook__(kind, value, tb)


def _atexit():
    if _S.get("state") == "running":  # SystemExit / normal end without finish(): still a clean exit
        finish()


def start(name, log=None, total=None, done=0, note=""):
    """Begin a run. log: its log file (the watchdog quotes its tail); total/done: rows."""
    DIR.mkdir(parents=True, exist_ok=True)
    if log is None:  # stdout redirected to a file: that is the log
        try:
            out = os.readlink("/proc/self/fd/1")
            log = out if os.path.isfile(out) else None
        except OSError:
            pass
    _S.update(
        name=name,
        pid=os.getpid(),
        argv=sys.argv,
        log=str(log) if log else None,
        state="running",
        started=time.time(),
        total=total,
        done=done,
        note=note,
        phase=None,
        phase_started=None,
        limit=None,
    )
    _write()
    threading.Thread(target=_beat, daemon=True).start()
    sys.excepthook = _excepthook
    atexit.register(_atexit)


def phase(what, limit=None, **info):
    """A new step; limit (s) is how long it may take before the watchdog reports an overrun."""
    with _LOCK:
        _S.update(phase=what, phase_started=time.time(), limit=limit, info=info)
    _write()


def progress(done, total):
    with _LOCK:
        _S.update(done=done, total=total)
    _write()


def row_done(n=1):
    with _LOCK:
        _S["done"] = (_S.get("done") or 0) + n
        _S["last_row"] = time.time()
    _write()


def problem(msg):
    """A recorded, non-fatal problem (e.g. one job failed and the run went on); the watchdog reports it once."""
    with _LOCK:
        _S.setdefault("problems", []).append(dict(t=time.time(), msg=str(msg)[-2000:]))
    _write()


def finish():
    with _LOCK:
        _S.update(state="finished", finished=time.time(), phase=None, limit=None)
    _write()
