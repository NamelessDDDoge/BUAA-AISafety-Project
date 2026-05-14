#!/usr/bin/env bash
set -euo pipefail

python -u test.py \
  --method linear \
  --arch Imagenet:resnet50 \
  --ckpt weights/rn50best.pth \
  --config data/datasets.yaml \
  --batch_size 128 \
  --num_workers 4 \
  --result_folder result/linear_imgnet_rn50 \
  "$@"
