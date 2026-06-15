#!/bin/bash
# Retrain Utonia with --no_prefix
# Run in: utonia env
# ~8 hours

set -e

L0="./retriv/camera_aware_captions_L0_p5.json"

echo "===== Utonia --no_prefix ====="
python retriv/train_with_utonia.py \
  --camera_captions_path $L0 \
  --fusion_type weighted \
  --output_logs ./training_logs_utonia_noprefix \
  --disable_temporal_relevance --relevance_mode all \
  --no_prefix --max_train_samples 2200 --val_samples_per_scene 4

echo "Session 3 done! Utonia with --no_prefix"
