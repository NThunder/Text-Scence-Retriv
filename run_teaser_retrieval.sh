#!/bin/bash
# Run GME ft. L3 for all 5 levels (L0–L4) with retrieval results saved.
# Uses cached image embeddings. Relevance always fixed at L0.
set -e

VAL="--val_samples_per_scene 4 --batch_size 2"
EVAL="--disable_temporal_relevance --relevance_mode all"
IMG_CACHE="./results_4perscene/img_cache/gme_ftL3.pth"

for LEVEL in L0 L1 L2 L3 L4; do
  echo "========== GME ft. L3 ${LEVEL}→L0 =========="
  python retriv/validate_gme_vlm.py \
    --model_name ./gme_finetuned_L3_p5_v2/best_model \
    --camera_captions_path "./retriv/camera_aware_captions_${LEVEL}_p5.json" \
    --relevance_captions_path "./retriv/camera_aware_captions_L0_p5.json" \
    --output_logs "./results_4perscene/gme_ftL3_${LEVEL}toL0" \
    $VAL $EVAL \
    --load_image_embs "$IMG_CACHE" \
    --save_retrieval_results "./results_4perscene/teaser_${LEVEL}.json"
done

echo "Done! Teaser files: ./results_4perscene/teaser_{L0,L1,L2,L3,L4}.json"
