#!/bin/bash
# Run GME evaluation with --val_samples_per_scene 4
# Image embeddings cached: encode once per model, reuse for L0 and L3
# Environment: gme_finetune (with qwen-vl-utils)

set -e

GPU="CUDA_VISIBLE_DEVICES=2"
OUTDIR="./results_4perscene"
mkdir -p $OUTDIR
VAL="--val_samples_per_scene 4"
EVAL="--disable_temporal_relevance --relevance_mode all --val_samples_per_scene 4"
IMG_CACHE="$OUTDIR/img_cache"

echo "========== GME ZERO-SHOT =========="
$GPU python retriv/validate_gme_vlm.py \
  --model_name NCSOFT/GME-VARCO-VISION-Embedding \
  --camera_captions_path ./retriv/camera_aware_captions_L0_p5.json \
  --output_logs $OUTDIR/gme_zs_L0 \
  $VAL --save_image_embs "$IMG_CACHE/gme_zs.pth"

$GPU python retriv/validate_gme_vlm.py \
  --model_name NCSOFT/GME-VARCO-VISION-Embedding \
  --camera_captions_path ./retriv/camera_aware_captions_L3_p5.json \
  --output_logs $OUTDIR/gme_zs_L3 \
  $VAL --load_image_embs "$IMG_CACHE/gme_zs.pth"

echo "========== GME FINE-TUNED L0 =========="
$GPU python retriv/validate_gme_vlm.py \
  --model_name ./gme_finetuned_L0_p5_v2/best_model \
  --camera_captions_path ./retriv/camera_aware_captions_L0_p5.json \
  --output_logs $OUTDIR/gme_ftL0_L0 \
  $VAL --save_image_embs "$IMG_CACHE/gme_ftL0.pth"

$GPU python retriv/validate_gme_vlm.py \
  --model_name ./gme_finetuned_L0_p5_v2/best_model \
  --camera_captions_path ./retriv/camera_aware_captions_L3_p5.json \
  --output_logs $OUTDIR/gme_ftL0_L3 \
  $VAL --load_image_embs "$IMG_CACHE/gme_ftL0.pth"

echo "========== GME FINE-TUNED L3 =========="
$GPU python retriv/validate_gme_vlm.py \
  --model_name ./gme_finetuned_L3_p5_v2/best_model \
  --camera_captions_path ./retriv/camera_aware_captions_L0_p5.json \
  --output_logs $OUTDIR/gme_ftL3_L0 \
  $VAL --save_image_embs "$IMG_CACHE/gme_ftL3.pth"

$GPU python retriv/validate_gme_vlm.py \
  --model_name ./gme_finetuned_L3_p5_v2/best_model \
  --camera_captions_path ./retriv/camera_aware_captions_L3_p5.json \
  --output_logs $OUTDIR/gme_ftL3_L3 \
  $VAL --load_image_embs "$IMG_CACHE/gme_ftL3.pth"

echo "Done! Results in $OUTDIR/"
