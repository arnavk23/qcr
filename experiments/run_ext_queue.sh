#!/bin/bash
# Extension pipeline, run after ext_cayley3.py ion_pm / nisq.
cd "$(dirname "$0")"
R=../results
python3 ext_parity.py nisq            > $R/ext_parity_nisq.log 2>&1
python3 ext_twoq.py                   > $R/ext_twoq3.log 2>&1
python3 ext_four.py spheres           > $R/ext_four_spheres.log 2>&1
python3 ext_gap.py gap3 nisq          > $R/ext_gap3_nisq.log 2>&1
python3 ext_four.py ew ion nisq       > $R/ext_four_ew.log 2>&1
python3 ext_four.py grid nisq         > $R/ext_grid_nisq.log 2>&1
python3 ext_gap.py mixing             > $R/ext_mixing.log 2>&1
echo QUEUE_DONE > $R/ext_queue.done
