#!/bin/bash
# Retrain GME ft. with --no_prefix, 10k pairs, batch_size 4
# Run in: gme_finetune env

set -e

L0="./retriv/camera_aware_captions_L0_p5.json"
L3="./retriv/camera_aware_captions_L3_p5.json"

echo "===== GME ft. L0 (--no_prefix, 10k pairs) ====="
python retriv/finetune_gme_nuscenes.py \
  --camera_captions_path $L0 \
  --output_dir ./gme_ft_L0_noprefix \
  --batch_size 4 --lr 2e-4 --epochs 10 --use_lora --no_prefix \
  --max_train_pairs 10000 --val_samples_per_scene 4

echo "===== GME ft. L3 (--no_prefix, 10k pairs) ====="
python retriv/finetune_gme_nuscenes.py \
  --camera_captions_path $L3 \
  --output_dir ./gme_ft_L3_noprefix \
  --batch_size 4 --lr 2e-4 --epochs 10 --use_lora --no_prefix \
  --max_train_pairs 10000 --val_samples_per_scene 4

echo "Done! GME ft. L0 + L3 with --no_prefix"
