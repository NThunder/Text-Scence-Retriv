# Text-Scene Retrieval for Autonomous Driving

Multimodal retrieval pipeline for nuScenes: text-to-image and text-to-scene 
search using CLIP-based models, LLM2CLIP, Utonia (LiDAR fusion), and 
GME-VARCO-VISION-Embedding.

## Structure

| Path | Description |
|---|---|
| `build_camera_aware_captions.py` | Generate caption JSONs (L0–L4) from TOD3Cap annotations |
| `encode_*.py` | Encode text/images/LiDAR into embeddings |
| `eval_clip_baseline_per_camera.py` | Evaluate text→image retrieval |
| `validate_gme_vlm.py` | Evaluate GME model retrieval |
| `train_*.py` | Training scripts (LoRA, joint encoders, Utonia fusion) |
| `finetune_gme_nuscenes.py` | LoRA fine-tuning of GME |
| `compute_*.py` | Utilities (lidar stats, pair intersection, rel set size) |

## Quick Start

```bash
# 1. Generate captions (L0–L4 with LiDAR threshold 5)
python build_camera_aware_captions.py --level attr --min_lidar_points 5 \
  --output_path ./camera_aware_captions_L0_p5.json

# 2. Encode and evaluate CLIP ViT-B/32 baseline
python encode_camera_aware_captions.py \
  --camera_captions_path ./camera_aware_captions_L0_p5.json \
  --output_path ./text_embs_clip.pth --max_val_samples 150

python encode_nuscenes_images_transformers.py \
  --output_path ./img_embs_clip.pth --max_val_samples 150

python eval_clip_baseline_per_camera.py \
  --attributes_path ./camera_aware_captions_L0_p5.json \
  --text_emb_path ./text_embs_clip.pth \
  --image_emb_path ./img_embs_clip.pth \
  --disable_temporal_relevance --relevance_mode all

# 3. Evaluate GME zero-shot
python validate_gme_vlm.py \
  --model_name NCSOFT/GME-VARCO-VISION-Embedding \
  --camera_captions_path ./camera_aware_captions_L0_p5.json \
  --disable_temporal_relevance --relevance_mode all
```

## Caption Levels

Generated from TOD3Cap annotations with 6 field types 
(`attribute`, `motion`, `map`, `localization`, `depth`, `relation`).

| Level | Fields | Example |
|---|---|---|
| L0 | attr | "A blue car." |
| L1 | attr+motion | "A blue car is moving quickly." |
| L2 | attr+map | "A blue car in the stop lane." |
| L3 | attr+motion+map | "A blue car is moving quickly in the stop lane." |
| L4 | all fields | Full description incl. depth & relations |

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataroot` | `/path/to/nuScenes` | nuScenes dataset root |
| `--max_val_samples` | `-1` (all) | Max validation frames |
| `--val_samples_per_scene` | `-1` (all) | Evenly spaced frames per scene |
| `--filter_pairs` | `None` | JSON with allowed (token, cam) pairs |
| `--disable_temporal_relevance` | off | Disable temporal neighbors |
| `--relevance_mode` | `any` | `any` or `all` attribute matching |
| `--camera_captions_path` | — | Path to L0–L4 caption JSON |
| `--model_name` | varies | HuggingFace model name |

## Training

### LoRA fine-tuning (LLM2CLIP)
```bash
python train_joint_lora_encoders.py \
  --camera_captions_path ./camera_aware_captions_L0_p5.json \
  --output_lora_text ./lora_text_L0 \
  --output_lora_image ./lora_image_L0
```

### GME fine-tuning
```bash
python finetune_gme_nuscenes.py \
  --camera_captions_path ./camera_aware_captions_L0_p5.json \
  --output_dir ./gme_finetuned_L0 \
  --batch_size 8 --lr 2e-4 --epochs 10 --use_lora
```

## Results

See `results_tables.md` for full tables.

**Best L0→L0 (all mode, no temporal):**

| Model | R@1 | R@5 | MRR |
|---|---|---|---|
| CLIP ViT-B/32 | 0.053 | 0.380 | 0.156 |
| LLM2CLIP | 0.237 | 0.327 | 0.295 |
| Full LoRA (LLM2CLIP) | 0.399 | 0.545 | 0.463 |
| GME zero-shot | 0.371 | 0.564 | 0.465 |
| **GME ft. L0** | **0.601** | **0.791** | **0.686** |

## Utilities

| Script | Purpose |
|---|---|
| `check_lidar_filter.py` | Count pairs per LiDAR threshold |
| `compute_val_pair_intersection.py` | Intersection of pairs across thresholds |
| `compute_rel_set_size.py` | Relevance set statistics |
| `run_lidar_ablation.sh` | Lidar threshold sensitivity analysis |
| `run_eval_4perscene.sh` | Evaluation with per-scene frame sampling |

## Environment

```bash
conda create -n retrieval python=3.10 -y
conda activate retrieval
pip install torch transformers accelerate peft nuscenes-devkit tqdm matplotlib tensorboard
```

For Utonia (PointTransformerV3): additional dependencies (see Utonia repo).  
For GME: `pip install qwen-vl-utils`.
