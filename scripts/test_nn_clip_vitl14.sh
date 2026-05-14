#!/usr/bin/env bash
set -euo pipefail

python -u test.py \
  --method nn \
  --arch CLIP:ViT-L/14 \
  --config data/datasets.yaml \
  --max_sample 1000 \
  --batch_size 128 \
  --num_workers 4 \
  --nn_k 1 \
  --nn_bank_path datasets/train \
  --nn_bank_data_mode wang2020 \
  --nn_bank_max_sample 1000 \
  --nn_bank_cache result/nn_clip_vitl14_paper/clip_vitl14_progan_train_bank.pth \
  --nn_bank_chunk_size 8192 \
  --result_folder result/nn_clip_vitl14_paper/k1 \
  "$@"
