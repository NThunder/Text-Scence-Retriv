#!/bin/bash
# LLM2CLIP + Full LoRA with --no_prefix
# Run in: llm2clip env

set -e

OUTDIR="./results_noprefix"
mkdir -p "$OUTDIR"
L0="./retriv/camera_aware_captions_L0_p5.json"
L3="./retriv/camera_aware_captions_L3_p5.json"
VAL="--val_samples_per_scene 4"
EVAL="--disable_temporal_relevance --relevance_mode all --val_samples_per_scene 4"
NOPREFIX="--no_prefix"

echo "========== LLM2CLIP zero-shot =========="
python retriv/encode_camera_aware_captions_llm2clip.py \
  --camera_captions_path $L0 --output_path "$OUTDIR/text_llm2clip_L0.pth" \
  $VAL --max_samples -1 $NOPREFIX
python retriv/encode_camera_aware_captions_llm2clip.py \
  --camera_captions_path $L3 --output_path "$OUTDIR/text_llm2clip_L3.pth" \
  $VAL --max_samples -1 $NOPREFIX
python retriv/encode_nuscenes_images_evaclip_l14_336_hf.py \
  --output_path "$OUTDIR/img_llm2clip.pth" $VAL

echo "========== Full LoRA (10k pairs) =========="
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

echo "========== EVAL =========="
for MODEL in llm2clip fullora; do
  for MODE in "L0 L0" "L0 L3" "L3 L3"; do
    set -- $MODE; ATTR=$1; TEXT=$2
    echo "$MODEL ${ATTR}→${TEXT}"
    python retriv/eval_clip_baseline_per_camera.py \
      --attributes_path "./retriv/camera_aware_captions_${ATTR}_p5.json" \
      --text_emb_path "$OUTDIR/text_${MODEL}_${TEXT}.pth" \
      --image_emb_path "$OUTDIR/img_${MODEL}.pth" $EVAL
  done
done

echo "LLM2CLIP done! Results in $OUTDIR/"
