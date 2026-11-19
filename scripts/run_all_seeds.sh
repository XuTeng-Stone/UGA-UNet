#!/usr/bin/env bash
# Reproduce the three-seed UGA-UNet runs reported in Table 4.
#
# Usage:
#   DATA_ROOT=/path/to/data DEVICE=cuda:0 bash scripts/run_all_seeds.sh
#
# The data root must contain BUSI/, BrEaST/, and QaTa-COV19/.
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-data}
DEVICE=${DEVICE:-cuda:0}
PYTHON=${PYTHON:-python}
SEEDS=${SEEDS:-"42 123 999"}
OUT_DIR=${OUT_DIR:-runs}

run_one () {
  local dataset=$1
  local config=$2
  local seed=$3
  "$PYTHON" scripts/train.py \
    --config "$config" \
    --data-root "$DATA_ROOT" \
    --device "$DEVICE" \
    --seed "$seed" \
    --out-dir "$OUT_DIR" \
    --run-name "${dataset}_ugaunet_anchor_seed${seed}"
}

for seed in $SEEDS; do
  run_one busi     configs/busi_ugaunet.yaml     "$seed"
  run_one breast   configs/breast_ugaunet.yaml   "$seed"
  run_one qata_v1  configs/qata_v1_ugaunet.yaml  "$seed"
  run_one qata_v2  configs/qata_v2_ugaunet.yaml  "$seed"
done

echo "Done. Results are under ${OUT_DIR}/"
