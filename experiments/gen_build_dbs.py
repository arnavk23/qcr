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


"""Build the large general databases (no Clifford assumption) and save them."""

import json
import sys

from common import CACHE, RESULTS

from qcr.general import GRIDS, Database

SPECS = {  # (device grid, topology, kind): depth cap (the builder switches to keys-only when the frontier gets big)
    ("heron", "line", "phase"): 10,
    ("heron", "line", "out"): 10,
    ("heron", "line", "state"): 14,
    ("ion", "tri", "phase"): 6,
    ("ion", "tri", "out"): 6,
    ("ion", "tri", "state"): 8,
    ("ion_paper", "tri", "phase"): 9,
    ("ibm_paper", "tri", "phase"): 10,  # 12/14 ran out of memory (14 GB)
    ("nam", "tri", "phase"): 9,  # logical-level baseline comparison (VOQC, Quartz, QUESO)
}
if __name__ == "__main__":
    which = [tuple(a.split(",")) for a in sys.argv[1:]] or list(SPECS)
    info = {}
    for spec in which:
        dev, topo, kind = spec
        db = Database(GRIDS[dev], topo, kind, SPECS[spec], max_frontier=500_000)
        db.save(CACHE / f"gdb_{dev}_{topo}_{kind}.npz")
        info["/".join(spec)] = dict(
            size=db.size, levels=db.levels, depth=db.depth, seconds=db.build_seconds, pool=len(db.pool)
        )
        print(spec, info["/".join(spec)], flush=True)
        del db
    p = RESULTS / "gdb_info.json"
    old = json.load(open(p)) if p.exists() else {}
    old.update(info)
    json.dump(old, open(p, "w"), indent=1)
