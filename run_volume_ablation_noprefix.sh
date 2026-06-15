#!/bin/bash
# GME data volume ablation with --no_prefix
# Fine-tune on N pairs, evaluate on L0→L0 with --val_samples_per_scene 4

set -e

OUTDIR="./results_noprefix"
mkdir -p $OUTDIR/img_cache
L0="./retriv/camera_aware_captions_L0_p5.json"
TRAIN="--val_samples_per_scene 4 --batch_size 4 --lr 2e-4 --epochs 10 --use_lora --no_prefix"
EVAL="--val_samples_per_scene 4 --batch_size 2 --disable_temporal_relevance --relevance_mode all --no_prefix"

for N in 1000 5000 10000 50000; do
  TAG="gme_ft_${N}pairs"
  echo "========== GME FT. $N pairs =========="

  python retriv/finetune_gme_nuscenes.py \
    --camera_captions_path $L0 \
    --output_dir "$OUTDIR/$TAG" \
    $TRAIN --max_train_pairs $N

  python retriv/validate_gme_vlm.py \
    --model_name "$OUTDIR/$TAG/best_model" \
    --camera_captions_path $L0 \
    --output_logs "$OUTDIR/${TAG}_eval" \
    $EVAL --save_image_embs "$OUTDIR/img_cache/$TAG.pth"
done

echo ""
echo "========== RESULTS =========="
echo "zero-shot: R@1=0.203 (from run_noprefix_gme.sh)"
for N in 1000 5000 10000 50000; do
  R="$OUTDIR/gme_ft_${N}pairs_eval/results.json"
  python3 -c "import json; d=json.load(open('$R')); print(f'$N pairs: R@1={d[\"R@1\"]:.4f}  R@5={d[\"R@5\"]:.4f}  MRR={d[\"MRR\"]:.4f}')"
done
