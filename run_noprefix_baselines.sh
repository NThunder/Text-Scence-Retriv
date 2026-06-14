#!/bin/bash
# Zero-shot CLIP baselines with --no_prefix
# Run in: baselines env

set -e

OUTDIR="./results_noprefix"
mkdir -p "$OUTDIR"
L0="./retriv/camera_aware_captions_L0_p5.json"
L3="./retriv/camera_aware_captions_L3_p5.json"
VAL="--val_samples_per_scene 4"
EVAL="--disable_temporal_relevance --relevance_mode all --val_samples_per_scene 4"
NOPREFIX="--no_prefix"

echo "========== CLIP ViT-B/32 =========="
python retriv/encode_camera_aware_captions.py \
  --model_name "openai/clip-vit-base-patch32" \
  --camera_captions_path $L0 --output_path "$OUTDIR/text_clipB32_L0.pth" $VAL $NOPREFIX
python retriv/encode_camera_aware_captions.py \
  --model_name "openai/clip-vit-base-patch32" \
  --camera_captions_path $L3 --output_path "$OUTDIR/text_clipB32_L3.pth" $VAL $NOPREFIX
python retriv/encode_nuscenes_images_transformers.py \
  --model_name "openai/clip-vit-base-patch32" \
  --output_path "$OUTDIR/img_clipB32.pth" $VAL

echo "========== SigLIP-B/16 =========="
python retriv/encode_camera_aware_captions_siglip.py \
  --camera_captions_path $L0 --output_path "$OUTDIR/text_siglip_L0.pth" $VAL $NOPREFIX
python retriv/encode_camera_aware_captions_siglip.py \
  --camera_captions_path $L3 --output_path "$OUTDIR/text_siglip_L3.pth" $VAL $NOPREFIX
python retriv/encode_nuscenes_images_siglip.py \
  --output_path "$OUTDIR/img_siglip.pth" $VAL

echo "========== EVA-CLIP-B/16 =========="
python retriv/encode_camera_aware_captions_evaclip.py \
  --camera_captions_path $L0 --output_path "$OUTDIR/text_evaclipB16_L0.pth" $VAL $NOPREFIX
python retriv/encode_camera_aware_captions_evaclip.py \
  --camera_captions_path $L3 --output_path "$OUTDIR/text_evaclipB16_L3.pth" $VAL $NOPREFIX
python retriv/encode_nuscenes_images_evaclip.py \
  --output_path "$OUTDIR/img_evaclipB16.pth" $VAL

echo "========== EVA-CLIP-L/14-336 =========="
python retriv/encode_camera_aware_captions_evaclip_l14_336.py \
  --camera_captions_path $L0 --output_path "$OUTDIR/text_evaclipL14_L0.pth" $VAL $NOPREFIX
python retriv/encode_camera_aware_captions_evaclip_l14_336.py \
  --camera_captions_path $L3 --output_path "$OUTDIR/text_evaclipL14_L3.pth" $VAL $NOPREFIX
python retriv/encode_nuscenes_images_evaclip_l14_336.py \
  --output_path "$OUTDIR/img_evaclipL14.pth" $VAL

echo "========== EVAL =========="
for MODEL in clipB32 siglip evaclipB16 evaclipL14; do
  for MODE in "L0 L0" "L0 L3" "L3 L3"; do
    set -- $MODE; ATTR=$1; TEXT=$2
    echo "$MODEL ${ATTR}→${TEXT}"
    python retriv/eval_clip_baseline_per_camera.py \
      --attributes_path "./retriv/camera_aware_captions_${ATTR}_p5.json" \
      --text_emb_path "$OUTDIR/text_${MODEL}_${TEXT}.pth" \
      --image_emb_path "$OUTDIR/img_${MODEL}.pth" $EVAL
  done
done

echo "Baselines done! Results in $OUTDIR/"
