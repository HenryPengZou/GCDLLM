#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Usage: bash run_reevaluate_simple.sh CHECKPOINT_PATH
CHECKPOINT_PATH=${1:?"Please provide the checkpoint file path, e.g., ./checkpoint_dir/clinc_known_cls_ratio_0.25_labeled_ratio_0.1.pt"}

python reevaluate.py \
  --data_dir data \
  --save_results_path 'outputs' \
  --experiment_name 'Reproduce_Load_Full_Model' \
  --running_method 'GCDLLMs' \
  --architecture 'Loop' \
  --checkpoint_path "${CHECKPOINT_PATH}"


