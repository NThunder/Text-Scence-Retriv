#!/bin/bash
# Evaluate Utonia on all three modes (L0→L0, L3→L0, L3→L3)
# Uses embeddings from validate_only, plus L3 text encoding

set -e

OUTDIR="./results_4perscene"
mkdir -p $OUTDIR

L0="./retriv/camera_aware_captions_L0_p5.json"
L3="./retriv/camera_aware_captions_L3_p5.json"
UTONIA="./training_logs_utonia_L0_p5_all_notemporal"
EMB="$UTONIA/embeddings/epoch_1"
EVAL="--disable_temporal_relevance --relevance_mode all --val_samples_per_scene 4"

# Encode L3 text with Utonia's text encoder
echo "========== Encode L3 text =========="
python retriv/encode_camera_aware_captions_llm2clip.py \
  --lora_path $UTONIA/models/best/text_lora \
  --camera_captions_path $L3 \
  --output_path $OUTDIR/text_utonia_L3.pth \
  --val_samples_per_scene 4 --max_samples -1

echo "========== Utonia L0→L0 =========="
python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L0 \
  --text_emb_path $EMB/text_embeddings.pth \
  --image_emb_path $EMB/fused_embeddings.pth $EVAL

echo "========== Utonia L3→L0 =========="
python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L0 \
  --text_emb_path $OUTDIR/text_utonia_L3.pth \
  --image_emb_path $EMB/fused_embeddings.pth $EVAL

echo "========== Utonia L3→L3 =========="
python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L3 \
  --text_emb_path $OUTDIR/text_utonia_L3.pth \
  --image_emb_path $EMB/fused_embeddings.pth $EVAL

echo "Done!"
