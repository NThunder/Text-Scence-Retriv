#!/bin/bash
# Run GME evaluation with --val_samples_per_scene 4
# Image embeddings cached: encode once per model, reuse for L0 and L3
# Environment: gme_finetune (with qwen-vl-utils)

set -e

OUTDIR="./results_4perscene"
mkdir -p $OUTDIR
mkdir -p "$OUTDIR/img_cache"
VAL="--val_samples_per_scene 4"
IMG_CACHE="$OUTDIR/img_cache"

echo "========== GME ZERO-SHOT =========="
python retriv/validate_gme_vlm.py \
  --model_name NCSOFT/GME-VARCO-VISION-Embedding \
  --camera_captions_path ./retriv/camera_aware_captions_L0_p5.json \
  --output_logs $OUTDIR/gme_zs_L0 \
  $VAL --save_image_embs "$IMG_CACHE/gme_zs.pth" --disable_temporal_relevance --relevance_mode all

python retriv/validate_gme_vlm.py \
  --model_name NCSOFT/GME-VARCO-VISION-Embedding \
  --camera_captions_path ./retriv/camera_aware_captions_L3_p5.json \
  --output_logs $OUTDIR/gme_zs_L3 \
  $VAL --load_image_embs "$IMG_CACHE/gme_zs.pth" --disable_temporal_relevance --relevance_mode all

python retriv/validate_gme_vlm.py \
  --model_name NCSOFT/GME-VARCO-VISION-Embedding \
  --camera_captions_path ./retriv/camera_aware_captions_L3_p5.json \
  --relevance_captions_path ./retriv/camera_aware_captions_L0_p5.json \
  --output_logs $OUTDIR/gme_zs_L3toL0 \
  $VAL --load_image_embs "$IMG_CACHE/gme_zs.pth" --disable_temporal_relevance --relevance_mode all

echo "========== GME FINE-TUNED L0 =========="
python retriv/validate_gme_vlm.py \
  --model_name ./gme_finetuned_L0_p5_v2/best_model \
  --camera_captions_path ./retriv/camera_aware_captions_L0_p5.json \
  --output_logs $OUTDIR/gme_ftL0_L0 \
  $VAL --save_image_embs "$IMG_CACHE/gme_ftL0.pth" --disable_temporal_relevance --relevance_mode all

python retriv/validate_gme_vlm.py \
  --model_name ./gme_finetuned_L0_p5_v2/best_model \
  --camera_captions_path ./retriv/camera_aware_captions_L3_p5.json \
  --output_logs $OUTDIR/gme_ftL0_L3 \
  $VAL --load_image_embs "$IMG_CACHE/gme_ftL0.pth" --disable_temporal_relevance --relevance_mode all

python retriv/validate_gme_vlm.py \
  --model_name ./gme_finetuned_L0_p5_v2/best_model \
  --camera_captions_path ./retriv/camera_aware_captions_L3_p5.json \
  --relevance_captions_path ./retriv/camera_aware_captions_L0_p5.json \
  --output_logs $OUTDIR/gme_ftL0_L3toL0 \
  $VAL --load_image_embs "$IMG_CACHE/gme_ftL0.pth" --disable_temporal_relevance --relevance_mode all

echo "========== GME FINE-TUNED L3 =========="
python retriv/validate_gme_vlm.py \
  --model_name ./gme_finetuned_L3_p5_v2/best_model \
  --camera_captions_path ./retriv/camera_aware_captions_L0_p5.json \
  --output_logs $OUTDIR/gme_ftL3_L0 \
  $VAL --save_image_embs "$IMG_CACHE/gme_ftL3.pth" --disable_temporal_relevance --relevance_mode all

python retriv/validate_gme_vlm.py \
  --model_name ./gme_finetuned_L3_p5_v2/best_model \
  --camera_captions_path ./retriv/camera_aware_captions_L3_p5.json \
  --output_logs $OUTDIR/gme_ftL3_L3 \
  $VAL --load_image_embs "$IMG_CACHE/gme_ftL3.pth" --disable_temporal_relevance --relevance_mode all

python retriv/validate_gme_vlm.py \
  --model_name ./gme_finetuned_L3_p5_v2/best_model \
  --camera_captions_path ./retriv/camera_aware_captions_L3_p5.json \
  --relevance_captions_path ./retriv/camera_aware_captions_L0_p5.json \
  --output_logs $OUTDIR/gme_ftL3_L3toL0 \
  $VAL --load_image_embs "$IMG_CACHE/gme_ftL3.pth" --disable_temporal_relevance --relevance_mode all

echo "Done! Results in $OUTDIR/"
