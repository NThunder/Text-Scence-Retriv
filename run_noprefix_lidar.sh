#!/bin/bash
# LiDAR ablation with --no_prefix on 600 frames
# Run in: gme_finetune env

set -e

OUTDIR="./results_noprefix"
mkdir -p "$OUTDIR"
VAL="--val_samples_per_scene 4 --batch_size 2"
EVAL="--disable_temporal_relevance --relevance_mode all --val_samples_per_scene 4 --batch_size 2"
NOPREFIX="--no_prefix"

for P in 0 1 5 20; do
  echo "========== p=$P =========="
  python retriv/validate_gme_vlm.py \
    --model_name NCSOFT/GME-VARCO-VISION-Embedding \
    --camera_captions_path "./retriv/camera_aware_captions_L0_p${P}.json" \
    --output_logs "$OUTDIR/lidar_p${P}" $VAL $EVAL $NOPREFIX
done

echo ""
echo "========== RESULTS =========="
for P in 0 1 5 20; do
  R="$OUTDIR/lidar_p${P}/results.json"
  python3 -c "import json; d=json.load(open('$R')); print(f'p=$P: pairs={d[\"total_pairs\"]:>5}  R@1={d[\"R@1\"]:.4f}  R@5={d[\"R@5\"]:.4f}  MRR={d[\"MRR\"]:.4f}')"
done
