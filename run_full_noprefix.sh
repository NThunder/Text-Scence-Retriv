#!/bin/bash
# Full recalc with --no_prefix on 600 frames
# Run on server, all 3 environments needed: gme, baselines, llm2clip
set -e

echo "==========================================="
echo " WARNING: This script runs ALL experiments"
echo " Expected time: ~5 hours"
echo " Run in screen/tmux!"
echo "==========================================="
sleep 3

GPU="CUDA_VISIBLE_DEVICES=2"
OUTDIR="./results_noprefix"
mkdir -p "$OUTDIR/img_cache"
L0="./retriv/camera_aware_captions_L0_p5.json"
L3="./retriv/camera_aware_captions_L3_p5.json"
VAL="--val_samples_per_scene 4 --batch_size 2"
EVAL="--disable_temporal_relevance --relevance_mode all --val_samples_per_scene 4 --batch_size 2"
NOPREFIX="--no_prefix"

# ============ PART 1: GME MODELS (gme_finetune env) ============

for MODEL_TAG in zs ftL0 ftL3; do
  case $MODEL_TAG in
    zs)   MODEL_NAME="NCSOFT/GME-VARCO-VISION-Embedding";;
    ftL0) MODEL_NAME="./gme_finetuned_L0_p5_v2/best_model";;
    ftL3) MODEL_NAME="./gme_finetuned_L3_p5_v2/best_model";;
  esac
  IMG_CACHE="$OUTDIR/img_cache/gme_${MODEL_TAG}.pth"
  FIRST_RUN=true

  for LEVEL in L0 L3; do
    for RELEV in L0 L3; do
      # L0→L0, L3→L0 (mixed), L3→L3 (strict)
      # Skip L0→L3 (not used)
      [ "$LEVEL" = "L0" ] && [ "$RELEV" = "L3" ] && continue
      # L3→L3 only when both are L3
      [ "$LEVEL" = "L3" ] && [ "$RELEV" = "L0" ] && RELEV_ARG="--relevance_captions_path $L0" || RELEV_ARG=""
      TAG="${MODEL_TAG}_${LEVEL}to${RELEV}"
      echo "========== GME $MODEL_TAG $LEVEL→$RELEV =========="
      
      if $FIRST_RUN; then
        $GPU python retriv/validate_gme_vlm.py \
          --model_name $MODEL_NAME \
          --camera_captions_path "./retriv/camera_aware_captions_${LEVEL}_p5.json" \
          $RELEV_ARG \
          --output_logs "$OUTDIR/$TAG" $VAL $EVAL $NOPREFIX \
          --save_image_embs "$IMG_CACHE"
        FIRST_RUN=false
      else
        $GPU python retriv/validate_gme_vlm.py \
          --model_name $MODEL_NAME \
          --camera_captions_path "./retriv/camera_aware_captions_${LEVEL}_p5.json" \
          $RELEV_ARG \
          --output_logs "$OUTDIR/$TAG" $VAL $EVAL $NOPREFIX \
          --load_image_embs "$IMG_CACHE"
      fi
    done
  done
done

# ============ PART 2: Zero-shot baselines (baselines env) ============

# CLIP ViT-B/32
python retriv/encode_camera_aware_captions.py \
  --model_name "openai/clip-vit-base-patch32" \
  --camera_captions_path $L0 --output_path "$OUTDIR/text_clipB32_L0.pth" \
  $VAL $NOPREFIX
python retriv/encode_camera_aware_captions.py \
  --model_name "openai/clip-vit-base-patch32" \
  --camera_captions_path $L3 --output_path "$OUTDIR/text_clipB32_L3.pth" \
  $VAL $NOPREFIX
python retriv/encode_nuscenes_images_transformers.py \
  --model_name "openai/clip-vit-base-patch32" \
  --output_path "$OUTDIR/img_clipB32.pth" $VAL

# SigLIP
python retriv/encode_camera_aware_captions_siglip.py \
  --camera_captions_path $L0 --output_path "$OUTDIR/text_siglip_L0.pth" $VAL $NOPREFIX
python retriv/encode_camera_aware_captions_siglip.py \
  --camera_captions_path $L3 --output_path "$OUTDIR/text_siglip_L3.pth" $VAL $NOPREFIX
python retriv/encode_nuscenes_images_siglip.py \
  --output_path "$OUTDIR/img_siglip.pth" $VAL

# EVA-CLIP-B/16
python retriv/encode_camera_aware_captions_evaclip.py \
  --camera_captions_path $L0 --output_path "$OUTDIR/text_evaclipB16_L0.pth" $VAL $NOPREFIX
python retriv/encode_camera_aware_captions_evaclip.py \
  --camera_captions_path $L3 --output_path "$OUTDIR/text_evaclipB16_L3.pth" $VAL $NOPREFIX
python retriv/encode_nuscenes_images_evaclip.py \
  --output_path "$OUTDIR/img_evaclipB16.pth" $VAL

# EVA-CLIP-L/14-336
python retriv/encode_camera_aware_captions_evaclip_l14_336.py \
  --camera_captions_path $L0 --output_path "$OUTDIR/text_evaclipL14_L0.pth" $VAL $NOPREFIX
python retriv/encode_camera_aware_captions_evaclip_l14_336.py \
  --camera_captions_path $L3 --output_path "$OUTDIR/text_evaclipL14_L3.pth" $VAL $NOPREFIX
python retriv/encode_nuscenes_images_evaclip_l14_336.py \
  --output_path "$OUTDIR/img_evaclipL14.pth" $VAL

# EVAL for baselines
for MODEL in clipB32 siglip evaclipB16 evaclipL14; do
  for MODE in "L0 L0" "L0 L3" "L3 L3"; do
    set -- $MODE
    ATTR=$1; TEXT=$2
    echo "========== $MODEL ${ATTR}→${TEXT} =========="
    python retriv/eval_clip_baseline_per_camera.py \
      --attributes_path "./retriv/camera_aware_captions_${ATTR}_p5.json" \
      --text_emb_path "$OUTDIR/text_${MODEL}_${TEXT}.pth" \
      --image_emb_path "$OUTDIR/img_${MODEL}.pth" $EVAL
  done
done

# ============ PART 3: LLM2CLIP (llm2clip env) ============

echo "========== LLM2CLIP zero-shot =========="
python retriv/encode_camera_aware_captions_llm2clip.py \
  --camera_captions_path $L0 --output_path "$OUTDIR/text_llm2clip_L0.pth" \
  $VAL --max_samples -1 $NOPREFIX
python retriv/encode_camera_aware_captions_llm2clip.py \
  --camera_captions_path $L3 --output_path "$OUTDIR/text_llm2clip_L3.pth" \
  $VAL --max_samples -1 $NOPREFIX
python retriv/encode_nuscenes_images_evaclip_l14_336_hf.py \
  --output_path "$OUTDIR/img_llm2clip.pth" $VAL

echo "========== Full LoRA L0 =========="
python retriv/encode_camera_aware_captions_llm2clip.py \
  --lora_path ./lora_text_L0_p5_10k_best \
  --camera_captions_path $L0 --output_path "$OUTDIR/text_fullora_L0.pth" \
  $VAL --max_samples -1 $NOPREFIX
python retriv/encode_camera_aware_captions_llm2clip.py \
  --lora_path ./lora_text_L0_p5_10k_best \
  --camera_captions_path $L3 --output_path "$OUTDIR/text_fullora_L3.pth" \
  $VAL --max_samples -1 $NOPREFIX
python retriv/encode_nuscenes_images_evaclip_l14_336_hf.py \
  --lora_path ./lora_image_L0_p5_10k_best \
  --output_path "$OUTDIR/img_fullora.pth" $VAL

# EVAL LLM2CLIP
for MODEL in llm2clip fullora; do
  for MODE in "L0 L0" "L0 L3" "L3 L3"; do
    set -- $MODE
    ATTR=$1; TEXT=$2
    echo "========== $MODEL ${ATTR}→${TEXT} =========="
    python retriv/eval_clip_baseline_per_camera.py \
      --attributes_path "./retriv/camera_aware_captions_${ATTR}_p5.json" \
      --text_emb_path "$OUTDIR/text_${MODEL}_${TEXT}.pth" \
      --image_emb_path "$OUTDIR/img_${MODEL}.pth" $EVAL
  done
done

echo ""
echo "========== ALL DONE =========="
echo "Results in $OUTDIR/"
