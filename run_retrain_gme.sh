#!/bin/bash
# Retrain GME ft. on the v3 captions (per-camera dedup) with a uniform,
# scene-level training subsample. Hyperparameters match the runs that produced
# the submitted numbers, so any change is attributable to the fixes, not to a
# different setup.
# Run in: gb_gme_finetune env

set -e

C="./retriv/captions_v2"
L0="$C/camera_aware_captions_L0_p5.json"
L3="$C/camera_aware_captions_L3_p5.json"
SEED=0

echo "===== GME ft. L3 (v3: captions_v2, uniform seed $SEED) ====="
python retriv/finetune_gme_nuscenes.py \
  --camera_captions_path $L3 \
  --output_dir ./gme_ft_L3_v3 \
  --batch_size 4 --lr 2e-4 --epochs 10 --use_lora --no_prefix \
  --max_train_pairs 10000 --sample_seed $SEED --val_samples_per_scene 4

echo "===== GME ft. L0 (v3: captions_v2, uniform seed $SEED) ====="
python retriv/finetune_gme_nuscenes.py \
  --camera_captions_path $L0 \
  --output_dir ./gme_ft_L0_v3 \
  --batch_size 4 --lr 2e-4 --epochs 10 --use_lora --no_prefix \
  --max_train_pairs 10000 --sample_seed $SEED --val_samples_per_scene 4

echo "Done! GME ft. L0 + L3 -> gme_ft_{L0,L3}_v3"
