#!/bin/bash
# Run CLIP baseline evaluation with --val_samples_per_scene 4
# Environment: standard transformers env

set -e

OUTDIR="./results_4perscene"
mkdir -p $OUTDIR
L0="./retriv/camera_aware_captions_L0_p5.json"
L3="./retriv/camera_aware_captions_L3_p5.json"
VAL="--val_samples_per_scene 4"

echo "========== CLIP ViT-B/32 =========="
python retriv/encode_camera_aware_captions.py \
  --model_name "openai/clip-vit-base-patch32" \
  --camera_captions_path $L0 \
  --output_path $OUTDIR/text_clipB32_L0.pth $VAL

python retriv/encode_camera_aware_captions.py \
  --model_name "openai/clip-vit-base-patch32" \
  --camera_captions_path $L3 \
  --output_path $OUTDIR/text_clipB32_L3.pth $VAL

python retriv/encode_nuscenes_images_transformers.py \
  --model_name "openai/clip-vit-base-patch32" \
  --output_path $OUTDIR/img_clipB32.pth $VAL

echo "========== LLM2CLIP zero-shot =========="
python retriv/encode_camera_aware_captions_llm2clip.py \
  --camera_captions_path $L0 \
  --output_path $OUTDIR/text_llm2clip_L0.pth $VAL

python retriv/encode_camera_aware_captions_llm2clip.py \
  --camera_captions_path $L3 \
  --output_path $OUTDIR/text_llm2clip_L3.pth $VAL

python retriv/encode_nuscenes_images_evaclip_l14_336_hf.py \
  --output_path $OUTDIR/img_llm2clip.pth $VAL

echo "========== EVAL =========="
python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L0 --text_emb_path $OUTDIR/text_clipB32_L0.pth \
  --image_emb_path $OUTDIR/img_clipB32.pth \
  --disable_temporal_relevance --relevance_mode all

python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L0 --text_emb_path $OUTDIR/text_clipB32_L3.pth \
  --image_emb_path $OUTDIR/img_clipB32.pth \
  --disable_temporal_relevance --relevance_mode all

python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L3 --text_emb_path $OUTDIR/text_clipB32_L3.pth \
  --image_emb_path $OUTDIR/img_clipB32.pth \
  --disable_temporal_relevance --relevance_mode all

python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L0 --text_emb_path $OUTDIR/text_llm2clip_L0.pth \
  --image_emb_path $OUTDIR/img_llm2clip.pth \
  --disable_temporal_relevance --relevance_mode all

python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L0 --text_emb_path $OUTDIR/text_llm2clip_L3.pth \
  --image_emb_path $OUTDIR/img_llm2clip.pth \
  --disable_temporal_relevance --relevance_mode all

python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L3 --text_emb_path $OUTDIR/text_llm2clip_L3.pth \
  --image_emb_path $OUTDIR/img_llm2clip.pth \
  --disable_temporal_relevance --relevance_mode all

echo "Done! Results in $OUTDIR/"
