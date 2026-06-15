#!/bin/bash
# Utonia retrained with --no_prefix: validate-only + eval all modes
# Run in: utonia env

set -e

OUTDIR="./results_noprefix"
mkdir -p $OUTDIR
L0="./retriv/camera_aware_captions_L0_p5.json"
L3="./retriv/camera_aware_captions_L3_p5.json"
UTONIA="./training_logs_utonia_noprefix"
EVAL="--disable_temporal_relevance --relevance_mode all --val_samples_per_scene 4"

# Step 1: validate-only to compute fused embeddings on 600 frames
echo "========== Utonia validate-only (600 frames) =========="
python retriv/train_with_utonia.py \
  --camera_captions_path $L0 \
  --fusion_type weighted \
  --output_logs $UTONIA \
  --validate_only $UTONIA/models/best \
  --val_samples_per_scene 4 \
  --disable_temporal_relevance --relevance_mode all \
  --no_prefix --batch_size 4

EMB="$UTONIA/embeddings/epoch_1"

# Step 2: Encode L3 text with retrained Utonia text encoder
echo "========== Encode L3 text =========="
python retriv/encode_camera_aware_captions_llm2clip.py \
  --lora_path $UTONIA/models/best/text_lora \
  --camera_captions_path $L3 \
  --output_path "$OUTDIR/text_utonia_L3.pth" \
  --val_samples_per_scene 4 --max_samples -1 --no_prefix

# Step 3: Eval all modes
echo "========== L0→L0 =========="
python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L0 \
  --text_emb_path $EMB/text_embeddings.pth \
  --image_emb_path $EMB/fused_embeddings.pth $EVAL

echo "========== L3→L0 =========="
python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L0 \
  --text_emb_path "$OUTDIR/text_utonia_L3.pth" \
  --image_emb_path $EMB/fused_embeddings.pth $EVAL

echo "========== L3→L3 =========="
python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L3 \
  --text_emb_path "$OUTDIR/text_utonia_L3.pth" \
  --image_emb_path $EMB/fused_embeddings.pth $EVAL

echo "Done! Results saved to $EMB"
