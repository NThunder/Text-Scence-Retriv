#!/bin/bash
# GME cross-level: all 5 levels (L0–L4) for all 3 GME models
# Uses MMS (old models, trained with prefix) evaluated with --no_prefix
# Uses cached image embeddings

set -e

OUTDIR="./results_noprefix"
mkdir -p "$OUTDIR/img_cache"
VAL="--val_samples_per_scene 4 --batch_size 2"
EVAL="--disable_temporal_relevance --relevance_mode all --val_samples_per_scene 4 --batch_size 2"
NOPREFIX="--no_prefix"
L0="./retriv/camera_aware_captions_L0_p5.json"

declare -A MODELS
MODELS["gme_zs"]="NCSOFT/GME-VARCO-VISION-Embedding"
MODELS["gme_ftL0"]="./gme_finetuned_L0_p5_v2/best_model"
MODELS["gme_ftL3"]="./gme_finetuned_L3_p5_v2/best_model"

for TAG in gme_zs gme_ftL0 gme_ftL3; do
  MODEL="${MODELS[$TAG]}"
  IMG_CACHE="$OUTDIR/img_cache/${TAG}.pth"
  FIRST=true

  for LEVEL in L0 L1 L2 L3 L4; do
    echo "========== $TAG ${LEVEL}→L0 =========="
    CAPS="--camera_captions_path ./retriv/camera_aware_captions_${LEVEL}_p5.json"

    if $FIRST; then
      python retriv/validate_gme_vlm.py \
        --model_name $MODEL $CAPS \
        --relevance_captions_path $L0 \
        --output_logs "$OUTDIR/${TAG}_${LEVEL}toL0" \
        $VAL $EVAL $NOPREFIX \
        --save_image_embs "$IMG_CACHE"
      FIRST=false
    else
      python retriv/validate_gme_vlm.py \
        --model_name $MODEL $CAPS \
        --relevance_captions_path $L0 \
        --output_logs "$OUTDIR/${TAG}_${LEVEL}toL0" \
        $VAL $EVAL $NOPREFIX \
        --load_image_embs "$IMG_CACHE"
    fi
  done
done

echo ""
echo "========== CROSS-LEVEL TABLE =========="
printf "%-12s %-10s %-10s %-10s %-10s\n" "Level" "GME zs" "GME ftL0" "GME ftL3" "Cos sim"
printf "%-12s %-10s %-10s %-10s %-10s\n" "-----" "-------" "--------" "--------" "-------"
for LEVEL in L0 L1 L2 L3 L4; do
  for TAG in gme_zs gme_ftL0 gme_ftL3; do
    R="$OUTDIR/${TAG}_${LEVEL}toL0/results.json"
    if [ -f "$R" ]; then
      VAL_R1=$(python3 -c "import json; print(f\"{json.load(open('$R'))['R@1']:.3f}\")")
      VAL_COS=$(python3 -c "import json; print(f\"{json.load(open('$R'))['avg_inter_query_cosine_sim']:.3f}\")")
      if [ "$TAG" = "gme_zs" ]; then
        printf "%-12s %-10s" "$LEVEL" "$VAL_R1"
      elif [ "$TAG" = "gme_ftL0" ]; then
        printf "%-10s" "$VAL_R1"
      else
        printf "%-10s %-10s\n" "$VAL_R1" "$VAL_COS"
      fi
    fi
  done
done
echo "Done!"
