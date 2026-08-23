#!/bin/bash
# Retrain Full LoRA + Image LoRA with --no_prefix
# Run in: llm2clip env

set -e

L0="./retriv/camera_aware_captions_L0_p5.json"

echo "===== Full LoRA (image+text) --no_prefix ====="
python retriv/train_joint_lora_encoders.py \
  --camera_captions_path $L0 \
  --output_lora_text ./lora_text_noprefix \
  --output_lora_image ./lora_image_noprefix \
  --max_train_samples 2200 --no_prefix

echo "===== Image LoRA --no_prefix ====="
# Step 1: encode L0 train text with --no_prefix
python retriv/encode_camera_aware_captions_llm2clip.py \
  --camera_captions_path $L0 \
  --output_path ./text_embs_llm2clip_train_noprefix.pth \
  --split train --max_samples 2200 --no_prefix

# Step 2: train image encoder using precomputed text embeddings
python retriv/train_lora_image_encoder.py \
  --text_emb_path ./text_embs_llm2clip_train_noprefix.pth \
  --output_lora_dir ./lora_image_only_noprefix \
  --max_train_samples 2200 \
  --camera_captions_path $L0

echo "Session 2 done! Full LoRA + Image LoRA with --no_prefix"
