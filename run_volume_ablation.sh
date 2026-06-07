#!/bin/bash
# Data volume ablation for Full LoRA (LLM2CLIP)
# Trains at N samples, encodes, evaluates on L0→L0
# Run in llm2clip environment

set -e

OUTDIR="./results_4perscene"
mkdir -p $OUTDIR
L0="./retriv/camera_aware_captions_L0_p5.json"
VAL="--val_samples_per_scene 4"
EVAL="--disable_temporal_relevance --relevance_mode all --val_samples_per_scene 4"

for N in 500 1000 5000; do
  TAG="fullora_${N}samples"
  echo ""
  echo "=========================================="
  echo " Training with $N samples ($TAG)"
  echo "=========================================="
  
  python retriv/train_joint_lora_encoders.py \
    --camera_captions_path $L0 \
    --output_lora_text "$OUTDIR/lora_text_$TAG" \
    --output_lora_image "$OUTDIR/lora_image_$TAG" \
    --max_train_samples $N \
    --val_samples_per_scene 4 \
    --batch_size 16 --lr 5e-5 --epochs 40 --lora_r 8
  
  echo "=========================================="
  echo " Encoding with $N samples ($TAG)"
  echo "=========================================="

  python retriv/encode_camera_aware_captions_llm2clip.py \
    --lora_path "$OUTDIR/lora_text_${TAG}_best" \
    --camera_captions_path $L0 \
    --output_path "$OUTDIR/text_$TAG.pth" $VAL --max_samples -1

  python retriv/encode_nuscenes_images_evaclip_l14_336_hf.py \
    --lora_path "$OUTDIR/lora_image_${TAG}_best" \
    --output_path "$OUTDIR/img_$TAG.pth" $VAL

  echo "=========================================="
  echo " Eval with $N samples"
  echo "=========================================="

  python retriv/eval_clip_baseline_per_camera.py \
    --attributes_path $L0 \
    --text_emb_path "$OUTDIR/text_$TAG.pth" \
    --image_emb_path "$OUTDIR/img_$TAG.pth" $EVAL
done

echo ""
echo "========== DONE =========="
echo "Results for all volumes:"
echo "  N=2200 (already done): R@1=0.204"
echo "  See results above for N=500, 1000, 5000"
