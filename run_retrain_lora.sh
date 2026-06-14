#!/bin/bash
# Retrain Full LoRA + Image LoRA with --no_prefix
# Run in: llm2clip env
# ~12 hours

set -e

L0="./retriv/camera_aware_captions_L0_p5.json"

echo "===== Full LoRA (image+text) --no_prefix ====="
python retriv/train_joint_lora_encoders.py \
  --camera_captions_path $L0 \
  --output_lora_text ./lora_text_noprefix \
  --output_lora_image ./lora_image_noprefix \
  --max_train_samples 2200 --no_prefix

echo "===== Image LoRA --no_prefix ====="
# For Image LoRA, need to encode L0 text without prefix first
python retriv/encode_camera_aware_captions_llm2clip.py \
  --camera_captions_path $L0 \
  --output_path ./text_embs_llm2clip_L0_noprefix.pth \
  --val_samples_per_scene 4 --max_samples -1 --no_prefix

python retriv/train_lora_image_encoder.py \
  --text_emb_path ./text_embs_llm2clip_L0_noprefix.pth \
  --output_lora_dir ./lora_image_only_noprefix

echo "Session 2 done! Full LoRA + Image LoRA with --no_prefix"
