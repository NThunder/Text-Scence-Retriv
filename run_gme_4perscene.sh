#!/bin/bash
# Run GME evaluation with --val_samples_per_scene 4
# Environment: gme_finetune (with qwen-vl-utils)

set -e

GPU="CUDA_VISIBLE_DEVICES=2"
OUTDIR="./results_4perscene"
mkdir -p $OUTDIR
VAL_FLAGS="--val_samples_per_scene 4 --disable_temporal_relevance --relevance_mode all"

echo "========== GME ZERO-SHOT =========="
for LEVEL in L0 L3; do
  CAPS=./retriv/camera_aware_captions_${LEVEL}_p5.json
  $GPU python retriv/validate_gme_vlm.py \
    --model_name NCSOFT/GME-VARCO-VISION-Embedding \
    --camera_captions_path $CAPS \
    --output_logs $OUTDIR/gme_zs_${LEVEL} \
    $VAL_FLAGS
done

echo "========== GME FINE-TUNED L0 =========="
for LEVEL in L0 L3; do
  CAPS=./retriv/camera_aware_captions_${LEVEL}_p5.json
  $GPU python retriv/validate_gme_vlm.py \
    --model_name ./gme_finetuned_L0_p5_v2/best_model \
    --camera_captions_path $CAPS \
    --output_logs $OUTDIR/gme_ftL0_${LEVEL} \
    $VAL_FLAGS
done

echo "========== GME FINE-TUNED L3 =========="
for LEVEL in L0 L3; do
  CAPS=./retriv/camera_aware_captions_${LEVEL}_p5.json
  $GPU python retriv/validate_gme_vlm.py \
    --model_name ./gme_finetuned_L3_p5_v2/best_model \
    --camera_captions_path $CAPS \
    --output_logs $OUTDIR/gme_ftL3_${LEVEL} \
    $VAL_FLAGS
done

echo "Done! Results in $OUTDIR/"
