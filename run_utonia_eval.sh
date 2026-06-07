#!/bin/bash
# Utonia validate-only with --val_samples_per_scene 4
# Loads trained weights and computes embeddings on 600 frames

set -e

OUTDIR="./results_4perscene"
mkdir -p $OUTDIR

UTONIA="./training_logs_utonia_L0_p5_all_notemporal"

python retriv/train_with_utonia.py \
  --camera_captions_path ./retriv/camera_aware_captions_L0_p5.json \
  --fusion_type weighted \
  --output_logs $UTONIA \
  --validate_only $UTONIA/models/best \
  --val_samples_per_scene 4 \
  --disable_temporal_relevance \
  --relevance_mode all \
  --batch_size 8

echo "Done! New embeddings in $UTONIA/embeddings/epoch_1/"
