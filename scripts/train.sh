#!/usr/bin/env bash
set -euo pipefail

python train.py \
  --name clip_vitl14 \
  --checkpoints_dir ./checkpoints \
  --arch CLIP:ViT-L/14 \
  --fix_backbone \
  --optim adam \
  --init_gain 0.02 \
  --beta1 0.9 \
  --weight_decay 0.0 \
  --batch_size 256 \
  --lr 0.0001 \
  --niter 100 \
  --earlystop_epoch 5 \
  --loss_freq 400 \
  --save_epoch_freq 1 \
  --rz_interp bilinear \
  --loadSize 256 \
  --cropSize 224 \
  --blur_sig 0.0 3.0 \
  --blur_prob 0.5 \
  --jpg_prob 0.5 \
  --jpg_method cv2,pil \
  --data_label train \
  --num_threads 4 \
  --gpu_ids 0 \
  --data_mode wang2020 \
  --wang2020_data_path ./datasets \
  "$@"
