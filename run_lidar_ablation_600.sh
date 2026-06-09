#!/bin/bash
# LiDAR threshold ablation on 600 frames (val_samples_per_scene 4)
# GME zero-shot, all 4 thresholds

set -e

GPU="CUDA_VISIBLE_DEVICES=2"
OUTDIR="./results_4perscene"
VAL="--val_samples_per_scene 4 --batch_size 2"
EVAL="--disable_temporal_relevance --relevance_mode all --val_samples_per_scene 4 --batch_size 2"

for P in 0 1 5 20; do
  echo "========== Threshold p=$P =========="
  $GPU python retriv/validate_gme_vlm.py \
    --model_name NCSOFT/GME-VARCO-VISION-Embedding \
    --camera_captions_path "./retriv/camera_aware_captions_L0_p${P}.json" \
    --output_logs "$OUTDIR/lidar_ablation_p${P}" \
    $VAL $EVAL
done

echo ""
echo "========== RESULTS =========="
for P in 0 1 5 20; do
  R="$OUTDIR/lidar_ablation_p${P}/results.json"
  if [ -f "$R" ]; then
    echo -n "p=$P: "
    python3 -c "import json; d=json.load(open('$R')); print(f'pairs={d[\"total_pairs\"]:>5}  R@1={d[\"R@1\"]:.4f}  R@5={d[\"R@5\"]:.4f}  MRR={d[\"MRR\"]:.4f}')"
  fi
done
