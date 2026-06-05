#!/bin/bash
# Step 1: Evaluation with --val_samples_per_scene 4
# Все модели оцениваются на равномерной выборке: 4 кадра из каждой val-сцены

set -e

GPU="CUDA_VISIBLE_DEVICES=2"
OUTDIR="./results_4perscene"
mkdir -p $OUTDIR

L0="./retriv/camera_aware_captions_L0_p5.json"
L3="./retriv/camera_aware_captions_L3_p5.json"
VAL_FLAGS="--val_samples_per_scene 4 --disable_temporal_relevance --relevance_mode all"

echo "========================================="
echo " 1. GME ZERO-SHOT"
echo "========================================="

for LEVEL in L0 L3; do
  CAPS=./retriv/camera_aware_captions_${LEVEL}_p5.json
  $GPU python retriv/validate_gme_vlm.py \
    --model_name NCSOFT/GME-VARCO-VISION-Embedding \
    --camera_captions_path $CAPS \
    --output_logs $OUTDIR/gme_zs_${LEVEL} \
    $VAL_FLAGS
done

echo "========================================="
echo " 2. GME FINE-TUNED L0"
echo "========================================="

for LEVEL in L0 L3; do
  CAPS=./retriv/camera_aware_captions_${LEVEL}_p5.json
  $GPU python retriv/validate_gme_vlm.py \
    --model_name ./gme_finetuned_L0_p5_v2/best_model \
    --camera_captions_path $CAPS \
    --output_logs $OUTDIR/gme_ftL0_${LEVEL} \
    $VAL_FLAGS
done

echo "========================================="
echo " 3. GME FINE-TUNED L3"
echo "========================================="

for LEVEL in L0 L3; do
  CAPS=./retriv/camera_aware_captions_${LEVEL}_p5.json
  $GPU python retriv/validate_gme_vlm.py \
    --model_name ./gme_finetuned_L3_p5_v2/best_model \
    --camera_captions_path $CAPS \
    --output_logs $OUTDIR/gme_ftL3_${LEVEL} \
    $VAL_FLAGS
done

echo "========================================="
echo " 4. CLIP BASELINES — ENCODING"
echo "========================================="

# CLIP ViT-B/32
python retriv/encode_camera_aware_captions.py \
  --model_name "openai/clip-vit-base-patch32" \
  --camera_captions_path $L0 \
  --output_path $OUTDIR/text_clipB32_L0.pth \
  --val_samples_per_scene 4

python retriv/encode_camera_aware_captions.py \
  --model_name "openai/clip-vit-base-patch32" \
  --camera_captions_path $L3 \
  --output_path $OUTDIR/text_clipB32_L3.pth \
  --val_samples_per_scene 4

python retriv/encode_nuscenes_images_transformers.py \
  --model_name "openai/clip-vit-base-patch32" \
  --output_path $OUTDIR/img_clipB32.pth \
  --val_samples_per_scene 4

# LLM2CLIP text
python retriv/encode_camera_aware_captions_llm2clip.py \
  --camera_captions_path $L0 \
  --output_path $OUTDIR/text_llm2clip_L0.pth \
  --val_samples_per_scene 4

python retriv/encode_camera_aware_captions_llm2clip.py \
  --camera_captions_path $L3 \
  --output_path $OUTDIR/text_llm2clip_L3.pth \
  --val_samples_per_scene 4

# LLM2CLIP images (uses evaclip_l14_336_hf)
python retriv/encode_nuscenes_images_evaclip_l14_336_hf.py \
  --output_path $OUTDIR/img_llm2clip.pth \
  --val_samples_per_scene 4

echo "========================================="
echo " 5. CLIP BASELINES — EVAL"
echo "========================================="

# CLIP ViT-B/32
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

# LLM2CLIP
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

echo "========================================="
echo " 6. LoRA MODELS — ENCODING + EVAL"
echo "========================================="

# Full LoRA (image+text) trained on L0
python retriv/encode_camera_aware_captions_llm2clip.py \
  --lora_path ./lora_text_L0_p5_10k_best \
  --camera_captions_path $L0 \
  --output_path $OUTDIR/text_fulloraL0_L0.pth \
  --val_samples_per_scene 4

python retriv/encode_camera_aware_captions_llm2clip.py \
  --lora_path ./lora_text_L0_p5_10k_best \
  --camera_captions_path $L3 \
  --output_path $OUTDIR/text_fulloraL0_L3.pth \
  --val_samples_per_scene 4

python retriv/encode_nuscenes_images_evaclip_l14_336_hf.py \
  --lora_path ./lora_image_L0_p5_10k_best \
  --output_path $OUTDIR/img_fulloraL0.pth \
  --val_samples_per_scene 4

# Eval Full LoRA L0
python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L0 --text_emb_path $OUTDIR/text_fulloraL0_L0.pth \
  --image_emb_path $OUTDIR/img_fulloraL0.pth \
  --disable_temporal_relevance --relevance_mode all

python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L0 --text_emb_path $OUTDIR/text_fulloraL0_L3.pth \
  --image_emb_path $OUTDIR/img_fulloraL0.pth \
  --disable_temporal_relevance --relevance_mode all

python retriv/eval_clip_baseline_per_camera.py \
  --attributes_path $L3 --text_emb_path $OUTDIR/text_fulloraL0_L3.pth \
  --image_emb_path $OUTDIR/img_fulloraL0.pth \
  --disable_temporal_relevance --relevance_mode all

echo ""
echo "========================================="
echo "  ALL DONE!"
echo "========================================="
echo "Results saved to: $OUTDIR/"
