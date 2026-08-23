#!/bin/bash
# Retrain Utonia on the v3 captions. Queued last: it is the slowest run and the
# weakest model, so it is the first thing to drop if the schedule slips.
# Epochs cut from the 40 default to 10, matching the other models.
#
# The point encoder keeps lr/10 (train_with_utonia.py:505). That is not a
# stability hack: it is the only branch whose pretrained weights are updated
# directly, while the vision and text branches are frozen with LoRA adapters
# and the projection/fusion layers start from scratch. Full fine-tuning at
# LoRA-scale learning rates would wash out the pretrained point features.
#
# Run in: utonia env

set -e

C="./retriv/captions_v2"
L0="$C/camera_aware_captions_L0_p5.json"
SEED=0

echo "===== Utonia, v3 (10 epochs) ====="
python retriv/train_with_utonia.py \
  --camera_captions_path $L0 \
  --fusion_type weighted \
  --output_logs ./training_logs_utonia_v3 \
  --disable_temporal_relevance --relevance_mode all \
  --no_prefix --max_train_samples 2200 --sample_seed $SEED \
  --epochs 10 --val_samples_per_scene 4

echo "Done! Utonia -> training_logs_utonia_v3"
