#!/bin/bash
# Retrain Full LoRA + Image LoRA on the v3 captions with a uniform,
# scene-level training subsample.
# Run in: utonia or gbuhtuev_m3net env (needs llm2vec)

set -e

C="./retriv/captions_v2"
L0="$C/camera_aware_captions_L0_p5.json"
SEED=0

echo "===== Full LoRA (image+text), v3 ====="
python retriv/train_joint_lora_encoders.py \
  --camera_captions_path $L0 \
  --output_lora_text ./lora_text_v3 \
  --output_lora_image ./lora_image_v3 \
  --max_train_samples 2200 --sample_seed $SEED --no_prefix

echo "===== Image LoRA, v3 ====="
# Step 1: encode L0 train text with --no_prefix.
# --max_samples defaults to 150; without -1 the pool is silently truncated.
python retriv/encode_camera_aware_captions_llm2clip.py \
  --camera_captions_path $L0 \
  --output_path ./text_embs_llm2clip_train_v3.pth \
  --split train --max_samples 2200 --no_prefix

# Step 2: train image encoder against those embeddings.
# --camera_captions_path enables the attribute metric the checkpoint is picked on.
python retriv/train_lora_image_encoder.py \
  --text_emb_path ./text_embs_llm2clip_train_v3.pth \
  --output_lora_dir ./lora_image_only_v3 \
  --max_train_samples 2200 --sample_seed $SEED \
  --camera_captions_path $L0

echo "Done! Full LoRA + Image LoRA -> *_v3"
