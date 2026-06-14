#!/bin/bash
# Ablation: template effect — "The camera view contains: ..." vs comma-separated
# GME zero-shot, L0→L0, 600 frames

set -e

GPU="CUDA_VISIBLE_DEVICES=2"
BASE="./results_4perscene"
VAL="--val_samples_per_scene 4 --batch_size 2"
EVAL="--disable_temporal_relevance --relevance_mode all"
L0="./retriv/camera_aware_captions_L0_p5.json"

echo "========== WITH prefix (current default) =========="
$GPU python retriv/validate_gme_vlm.py \
  --model_name NCSOFT/GME-VARCO-VISION-Embedding \
  --camera_captions_path $L0 \
  --output_logs "$BASE/ablation_prefix" \
  $VAL $EVAL

echo ""
echo "========== WITHOUT prefix (comma-separated) =========="
$GPU python retriv/validate_gme_vlm.py \
  --model_name NCSOFT/GME-VARCO-VISION-Embedding \
  --camera_captions_path $L0 \
  --output_logs "$BASE/ablation_noprefix" \
  $VAL $EVAL --no_prefix

echo ""
echo "========== COMPARISON =========="
for TAG in prefix noprefix; do
  R="$BASE/ablation_${TAG}/results.json"
  python3 -c "import json; d=json.load(open('$R')); print(f'$TAG: R@1={d[\"R@1\"]:.4f}  R@5={d[\"R@5\"]:.4f}  MRR={d[\"MRR\"]:.4f}  cos_sim={d[\"avg_inter_query_cosine_sim\"]:.4f}')"
done
