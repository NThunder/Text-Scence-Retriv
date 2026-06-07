#!/bin/bash
# Run remaining baselines with --val_samples_per_scene 4
# Environment: standard transformers + clip + evaclip

set -e

OUTDIR="./results_4perscene"
mkdir -p $OUTDIR
L0="./retriv/camera_aware_captions_L0_p5.json"
L3="./retriv/camera_aware_captions_L3_p5.json"
VAL="--val_samples_per_scene 4"
EVAL="--disable_temporal_relevance --relevance_mode all --val_samples_per_scene 4"

echo "========== 1. SigLIP-B/16 =========="
python retriv/encode_camera_aware_captions_siglip.py \
  --camera_captions_path $L0 --output_path $OUTDIR/text_siglip_L0.pth $VAL
python retriv/encode_camera_aware_captions_siglip.py \
  --camera_captions_path $L3 --output_path $OUTDIR/text_siglip_L3.pth $VAL
python retriv/encode_nuscenes_images_siglip.py \
  --output_path $OUTDIR/img_siglip.pth $VAL

echo "========== 2. EVA-CLIP-B/16 =========="
python retriv/encode_camera_aware_captions_evaclip.py \
  --camera_captions_path $L0 --output_path $OUTDIR/text_evaclipB16_L0.pth $VAL
python retriv/encode_camera_aware_captions_evaclip.py \
  --camera_captions_path $L3 --output_path $OUTDIR/text_evaclipB16_L3.pth $VAL
python retriv/encode_nuscenes_images_evaclip.py \
  --output_path $OUTDIR/img_evaclipB16.pth $VAL

echo "========== 3. EVA-CLIP-L/14-336 =========="
python retriv/encode_camera_aware_captions_evaclip_l14_336.py \
  --camera_captions_path $L0 --output_path $OUTDIR/text_evaclipL14_L0.pth $VAL
python retriv/encode_camera_aware_captions_evaclip_l14_336.py \
  --camera_captions_path $L3 --output_path $OUTDIR/text_evaclipL14_L3.pth $VAL
python retriv/encode_nuscenes_images_evaclip_l14_336.py \
  --output_path $OUTDIR/img_evaclipL14.pth $VAL

echo "========== 4. Image LoRA =========="
# Text already encoded (LLM2CLIP zs), only need image with LoRA
python retriv/encode_nuscenes_images_evaclip_l14_336_hf.py \
  --lora_path ./lora_image_only_p5 \
  --output_path $OUTDIR/img_lora_image.pth $VAL

echo "========== EVAL =========="

# SigLIP
python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L0 --text_emb_path $OUTDIR/text_siglip_L0.pth \
  --image_emb_path $OUTDIR/img_siglip.pth $EVAL
python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L0 --text_emb_path $OUTDIR/text_siglip_L3.pth \
  --image_emb_path $OUTDIR/img_siglip.pth $EVAL
python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L3 --text_emb_path $OUTDIR/text_siglip_L3.pth \
  --image_emb_path $OUTDIR/img_siglip.pth $EVAL

# EVA-CLIP-B/16
python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L0 --text_emb_path $OUTDIR/text_evaclipB16_L0.pth \
  --image_emb_path $OUTDIR/img_evaclipB16.pth $EVAL
python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L0 --text_emb_path $OUTDIR/text_evaclipB16_L3.pth \
  --image_emb_path $OUTDIR/img_evaclipB16.pth $EVAL
python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L3 --text_emb_path $OUTDIR/text_evaclipB16_L3.pth \
  --image_emb_path $OUTDIR/img_evaclipB16.pth $EVAL

# EVA-CLIP-L/14-336
python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L0 --text_emb_path $OUTDIR/text_evaclipL14_L0.pth \
  --image_emb_path $OUTDIR/img_evaclipL14.pth $EVAL
python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L0 --text_emb_path $OUTDIR/text_evaclipL14_L3.pth \
  --image_emb_path $OUTDIR/img_evaclipL14.pth $EVAL
python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L3 --text_emb_path $OUTDIR/text_evaclipL14_L3.pth \
  --image_emb_path $OUTDIR/img_evaclipL14.pth $EVAL

# Image LoRA — text from LLM2CLIP zs
python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L0 --text_emb_path $OUTDIR/text_llm2clip_L0.pth \
  --image_emb_path $OUTDIR/img_lora_image.pth $EVAL
python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L0 --text_emb_path $OUTDIR/text_llm2clip_L3.pth \
  --image_emb_path $OUTDIR/img_lora_image.pth $EVAL
python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L3 --text_emb_path $OUTDIR/text_llm2clip_L3.pth \
  --image_emb_path $OUTDIR/img_lora_image.pth $EVAL

echo "Done! Results in $OUTDIR/"
