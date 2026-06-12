#!/usr/bin/env bash
set -e

set -a
source .env
set +a

python training_scripts/gcnn_training.py \
  --data-dir untracked/test_data_subset \
  --run-dir untracked/runs/gcnn_test_subset \
  --epochs 20 \
  --batch-size 16 \
  --num-neighbors 5,5 \
  --report-every 1 \
  --aa-device cpu \
  --device cpu \
  --wandb-project m2f \
  --wandb-name gcnn-test-subset