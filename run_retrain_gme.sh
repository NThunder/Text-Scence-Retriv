#!/bin/bash
# Retrain all fine-tuned models with --no_prefix
# Run in screen/tmux — total ~26 hours over ~3 sessions

set -e

L0="./retriv/camera_aware_captions_L0_p5.json"
L3="./retriv/camera_aware_captions_L3_p5.json"

# ===== Session 1: GME fine-tune (gme_finetune env, ~6 h) =====
echo "===== GME ft. L0 (--no_prefix) ====="
python retriv/finetune_gme_nuscenes.py \
  --camera_captions_path $L0 \
  --output_dir ./gme_ft_L0_noprefix \
  --batch_size 2 --lr 2e-4 --epochs 10 --use_lora --no_prefix \
  --val_samples_per_scene 4

echo "===== GME ft. L3 (--no_prefix) ====="
python retriv/finetune_gme_nuscenes.py \
  --camera_captions_path $L3 \
  --output_dir ./gme_ft_L3_noprefix \
  --batch_size 2 --lr 2e-4 --epochs 10 --use_lora --no_prefix \
  --val_samples_per_scene 4

echo "Session 1 done! GME ft. L0 + L3 with --no_prefix"
