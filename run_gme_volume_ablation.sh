#!/bin/bash
# GME data volume ablation
# Fine-tune on N pairs, evaluate on L0→L0 with --val_samples_per_scene 4

set -e

OUTDIR="./results_4perscene"
mkdir -p $OUTDIR
L0="./retriv/camera_aware_captions_L0_p5.json"

IMG_CACHE="$OUTDIR/img_cache"
mkdir -p "$IMG_CACHE"
VAL="--val_samples_per_scene 4 --batch_size 2"
EVAL="--val_samples_per_scene 4 --batch_size 2 --disable_temporal_relevance --relevance_mode all"

echo "========== GME FT. 1k =========="
python retriv/finetune_gme_nuscenes.py \
  --camera_captions_path $L0 \
  --output_dir "$OUTDIR/gme_ft_1k" \
  --batch_size 8 --lr 2e-4 --epochs 10 --use_lora \
  $VAL --max_train_pairs 1000

python retriv/validate_gme_vlm.py \
  --model_name "$OUTDIR/gme_ft_1k/best_model" \
  --camera_captions_path $L0 --output_logs "$OUTDIR/gme_ft_1k_eval" \
  $VAL --save_image_embs "$IMG_CACHE/gme_ft_1k.pth" $EVAL

echo "========== GME FT. 5k =========="
python retriv/finetune_gme_nuscenes.py \
  --camera_captions_path $L0 \
  --output_dir "$OUTDIR/gme_ft_5k" \
  --batch_size 8 --lr 2e-4 --epochs 10 --use_lora \
  $VAL --max_train_pairs 5000

python retriv/validate_gme_vlm.py \
  --model_name "$OUTDIR/gme_ft_5k/best_model" \
  --camera_captions_path $L0 --output_logs "$OUTDIR/gme_ft_5k_eval" \
  $VAL --save_image_embs "$IMG_CACHE/gme_ft_5k.pth" $EVAL

echo "========== GME FT. 50k =========="
python retriv/finetune_gme_nuscenes.py \
  --camera_captions_path $L0 \
  --output_dir "$OUTDIR/gme_ft_50k" \
  --batch_size 8 --lr 2e-4 --epochs 10 --use_lora \
  $VAL --max_train_pairs 50000

python retriv/validate_gme_vlm.py \
  --model_name "$OUTDIR/gme_ft_50k/best_model" \
  --camera_captions_path $L0 --output_logs "$OUTDIR/gme_ft_50k_eval" \
  $VAL --save_image_embs "$IMG_CACHE/gme_ft_50k.pth" $EVAL
