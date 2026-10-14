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


"""Check every run that keeps a status file (runstatus.py) and raise an alert when one needs attention.

Alerts go to results/status/ALERTS.log and to the desktop (notify-send); each distinct alert is raised
once. Run from cron every few minutes:

    */10 * * * * cd <repo> && python3 experiments/watchdog.py

`python3 experiments/watchdog.py --show` prints the state of every run instead.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "results" / "status"
ALERTS = DIR / "ALERTS.log"
SENT = DIR / ".sent.json"
STALE = 20 * 60  # heartbeat older than this while the pid is alive: frozen
GRACE = 15 * 60  # extra time past a phase limit before calling it an overrun
NO_ROW = 18 * 3600  # running but no row finished for this long


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def tail(path, n=12):
    try:
        lines = [l for l in Path(path).read_text(errors="replace").splitlines() if "Warning" not in l]
        return "\n".join(lines[-n:])
    except (OSError, TypeError):
        return ""


def oom_kills(since):
    """Kernel OOM kills since `since` (epoch s)."""
    try:
        out = subprocess.run(
            ["journalctl", "-k", "--since", f"@{int(since)}", "--no-pager", "-o", "short-unix"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return re.findall(r"^(\d+)\.\d+ .*Killed process (\d+) \((\S+)\)", out, re.M)


def fmt(s):
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m"


def check(st, now):
    """Return [(key, message)] for one status file."""
    out = []
    name, pid, state = st["name"], st["pid"], st.get("state")
    where = f"phase '{st.get('phase')}' {st.get('info') or ''}".strip()
    if state == "crashed":
        out.append(("crashed", f"{name} crashed during {where}:\n{st.get('error', '')[-1500:]}"))
    elif state == "running" and not alive(pid):
        out.append(
            ("died", f"{name} (pid {pid}) is gone without finishing, last {where}.\nlog tail:\n{tail(st.get('log'))}")
        )
    elif state == "running":
        if now - st["heartbeat"] > STALE:
            out.append(("frozen", f"{name}: no heartbeat for {fmt(now - st['heartbeat'])} (pid alive), in {where}"))
        if st.get("limit") and st.get("phase_started") and now - st["phase_started"] > st["limit"] + GRACE:
            out.append(
                (
                    f"overrun:{st['phase']}:{st.get('info')}",
                    f"{name}: {where} has run {fmt(now - st['phase_started'])}, limit {fmt(st['limit'])}",
                )
            )
        last = st.get("last_row") or st["started"]
        if now - last > NO_ROW:
            out.append((f"norow:{int(last)}", f"{name}: no row finished for {fmt(now - last)}, in {where}"))
        for t, kpid, comm in oom_kills(st["started"]):
            out.append(
                (
                    f"oom:{kpid}:{t}",
                    f"{name}: the kernel killed {comm} (pid {kpid}) for memory at "
                    f"{time.strftime('%H:%M', time.localtime(int(t)))}; a pool run may now hang on the lost job",
                )
            )
    for p in st.get("problems", []):
        out.append((f"problem:{p['t']}", f"{name}: {p['msg']}"))
    if state == "finished":
        out.append(
            (
                "finished",
                f"{name} finished ({st.get('done')} rows) at "
                f"{time.strftime('%a %H:%M', time.localtime(st['finished']))}",
            )
        )
    return [(f"{name}:{st['started']}:{k}", m) for k, m in out]


def notify(msg):
    first, _, rest = msg.partition("\n")
    env = dict(os.environ)
    env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{os.getuid()}/bus")
    try:
        subprocess.run(
            ["notify-send", "-u", "critical", "-a", "qcr watchdog", first[:200], rest[:600]],
            env=env,
            timeout=10,
            capture_output=True,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    now = time.time()
    runs = [json.loads(p.read_text()) for p in sorted(DIR.glob("*.json")) if not p.name.startswith(".")]
    if "--show" in sys.argv:
        for st in runs:
            ph = (
                f"{st.get('phase')} {st.get('info') or ''} for {fmt(now - st['phase_started'])}"
                if st.get("phase_started") and st.get("state") == "running"
                else ""
            )
            print(f"{st['name']:<24} {st['state']:<9} rows {st.get('done')}/{st.get('total')}  {ph}")
        return
    sent = set(json.loads(SENT.read_text())) if SENT.exists() else set()
    for st in runs:
        for key, msg in check(st, now):
            if key in sent:
                continue
            sent.add(key)
            with open(ALERTS, "a") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M')}] {msg}\n\n")
            notify(msg)
    SENT.write_text(json.dumps(sorted(sent)))


if __name__ == "__main__":
    main()
