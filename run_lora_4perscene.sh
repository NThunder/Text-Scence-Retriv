#!/bin/bash
# Run Full LoRA evaluation with --val_samples_per_scene 4
# Environment: llm2clip env (with llm2vec, peft)

set -e

OUTDIR="./results_4perscene"
mkdir -p $OUTDIR
L0="./retriv/camera_aware_captions_L0_p5.json"
L3="./retriv/camera_aware_captions_L3_p5.json"
VAL="--val_samples_per_scene 4"
EVALFLAGS="--disable_temporal_relevance --relevance_mode all --val_samples_per_scene 4"

echo "========== FULL LoRA L0 (10k pairs) =========="
python retriv/encode_camera_aware_captions_llm2clip.py \
  --lora_path ./lora_text_L0_p5_10k_best \
  --camera_captions_path $L0 \
  --output_path $OUTDIR/text_fulloraL0_L0.pth $VAL --max_samples -1

python retriv/encode_camera_aware_captions_llm2clip.py \
  --lora_path ./lora_text_L0_p5_10k_best \
  --camera_captions_path $L3 \
  --output_path $OUTDIR/text_fulloraL0_L3.pth $VAL --max_samples -1

python retriv/encode_nuscenes_images_evaclip_l14_336_hf.py \
  --lora_path ./lora_image_L0_p5_10k_best \
  --output_path $OUTDIR/img_fulloraL0.pth $VAL

echo "========== EVAL =========="
python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L0 \
  --text_emb_path $OUTDIR/text_fulloraL0_L0.pth \
  --image_emb_path $OUTDIR/img_fulloraL0.pth $EVALFLAGS

python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L0 \
  --text_emb_path $OUTDIR/text_fulloraL0_L3.pth \
  --image_emb_path $OUTDIR/img_fulloraL0.pth $EVALFLAGS

python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L3 \
  --text_emb_path $OUTDIR/text_fulloraL0_L3.pth \
  --image_emb_path $OUTDIR/img_fulloraL0.pth $EVALFLAGS

echo "Done! Results in $OUTDIR/"
