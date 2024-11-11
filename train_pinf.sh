#!/bin/bash
source /afs/cs.stanford.edu/u/y1gao/anaconda3/etc/profile.d/conda.sh
conda activate /viscam/u/y1gao/pinf
cd /viscam/u/y1gao/pinf_smoke/
python run_pinf.py --config configs/scalar.txt