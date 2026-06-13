#!/bin/bash
# Run L1, L2, L4 evaluation for all GME models (mixed mode Lx→L0)
# Uses cached image embeddings

set -e

OUTDIR="./results_4perscene"
mkdir -p $OUTDIR
VAL="--val_samples_per_scene 4 --batch_size 2"
EVAL="--disable_temporal_relevance --relevance_mode all --val_samples_per_scene 4 --batch_size 2"

declare -A MODELS
MODELS["gme_zs"]="NCSOFT/GME-VARCO-VISION-Embedding"
MODELS["gme_ftL0"]="./gme_finetuned_L0_p5_v2/best_model"
MODELS["gme_ftL3"]="./gme_finetuned_L3_p5_v2/best_model"

for TAG in gme_zs gme_ftL0 gme_ftL3; do
  MODEL="${MODELS[$TAG]}"
  for LEVEL in L1 L2 L4; do
    echo "========== $TAG $LEVEL→L0 =========="
    python retriv/validate_gme_vlm.py \
      --model_name $MODEL \
      --camera_captions_path "./retriv/camera_aware_captions_${LEVEL}_p5.json" \
      --relevance_captions_path "./retriv/camera_aware_captions_L0_p5.json" \
      --load_image_embs "$OUTDIR/img_cache/${TAG}.pth" \
      --output_logs "$OUTDIR/${TAG}_${LEVEL}toL0" $VAL $EVAL
  done
done

echo ""
echo "Done! Results in $OUTDIR/"
