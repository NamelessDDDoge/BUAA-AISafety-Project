#!/usr/bin/env bash
set -euo pipefail

python -u test.py \
  --method linear \
  --arch CLIP:ViT-L/14 \
  --ckpt weights/fc_weights.pth \
  --config data/datasets.yaml \
  --max_sample 10000 \
  --batch_size 128 \
  --num_workers 4 \
  --result_folder result/linear_clip_vitl14 \
  "$@"
