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


"""Exact minimal two-qubit-gate count for every 3-qubit Clifford."""

import json

import numpy as np
from common import CACHE, RESULTS

from qcr.gates import make_pool
from qcr.tableau import twoq_bfs

codes, c2, layers = twoq_bfs(3, make_pool("ion_pm", 3))
np.savez(CACHE / "twoq3_ion_pm.npz", codes=codes, c2=c2)
out = dict(layers=layers, order=int(len(codes)), mean=float(np.dot(np.arange(len(layers)), layers) / len(codes)))
json.dump(out, open(RESULTS / "ext_twoq3.json", "w"), indent=1)
print(out)
