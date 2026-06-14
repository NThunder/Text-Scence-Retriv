#!/bin/bash
# GME models with --no_prefix
# Run in: gme_finetune env

set -e

GPU="CUDA_VISIBLE_DEVICES=2"
OUTDIR="./results_noprefix"
mkdir -p "$OUTDIR/img_cache"
L0="./retriv/camera_aware_captions_L0_p5.json"
VAL="--val_samples_per_scene 4 --batch_size 2"
EVAL="--disable_temporal_relevance --relevance_mode all --val_samples_per_scene 4 --batch_size 2"
NOPREFIX="--no_prefix"

for MODEL_TAG in zs ftL0 ftL3; do
  case $MODEL_TAG in
    zs)   MODEL_NAME="NCSOFT/GME-VARCO-VISION-Embedding";;
    ftL0) MODEL_NAME="./gme_finetuned_L0_p5_v2/best_model";;
    ftL3) MODEL_NAME="./gme_finetuned_L3_p5_v2/best_model";;
  esac
  IMG_CACHE="$OUTDIR/img_cache/gme_${MODEL_TAG}.pth"
  FIRST=true

  for LEVEL in L0 L3; do
    for RELEV in L0 L3; do
      [ "$LEVEL" = "L0" ] && [ "$RELEV" = "L3" ] && continue
      TAG="${MODEL_TAG}_${LEVEL}to${RELEV}"
      echo "========== GME $MODEL_TAG $LEVEL→$RELEV =========="

      RELEV_ARG=""
      [ "$LEVEL" != "$RELEV" ] && RELEV_ARG="--relevance_captions_path $L0"

      CAPS="--camera_captions_path ./retriv/camera_aware_captions_${LEVEL}_p5.json"

      if $FIRST; then
        $GPU python retriv/validate_gme_vlm.py \
          --model_name $MODEL_NAME $CAPS $RELEV_ARG \
          --output_logs "$OUTDIR/$TAG" $VAL $EVAL $NOPREFIX \
          --save_image_embs "$IMG_CACHE"
        FIRST=false
      else
        $GPU python retriv/validate_gme_vlm.py \
          --model_name $MODEL_NAME $CAPS $RELEV_ARG \
          --output_logs "$OUTDIR/$TAG" $VAL $EVAL $NOPREFIX \
          --load_image_embs "$IMG_CACHE"
      fi
    done
  done
done

echo "GME done! Results in $OUTDIR/"
