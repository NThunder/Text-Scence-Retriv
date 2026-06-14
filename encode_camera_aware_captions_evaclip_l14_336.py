# tools/encode_camera_aware_captions_evaclip_l14_336.py
import sys
sys.path.insert(0, "/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/EVA/EVA-CLIP/rei")

import json
import torch
import argparse
from eva_clip import create_model_and_transforms, get_tokenizer
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Encode camera captions with EVA-CLIP-L/14-336")
    parser.add_argument("--dataroot", type=str, default="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval")
    parser.add_argument("--camera_captions_path", type=str, default="./camera_aware_captions_short.json")
    parser.add_argument("--output_path", type=str, default="./camera_text_embeddings_evaclip_l14_336_val150.pth")
    parser.add_argument("--max_val_samples", type=int, default=-1,
        help="Max validation samples (-1 = all)")
    parser.add_argument("--val_samples_per_scene", type=int, default=-1,
        help="Number of evenly spaced samples per validation scene (-1 = all)")
    parser.add_argument("--no_prefix", action="store_true", help="Omit 'The camera view contains: ' prefix")
    return parser.parse_args()

args = parse_args()
nuscenes_dataroot = args.dataroot
camera_captions_path = args.camera_captions_path
output_path = args.output_path

with open(camera_captions_path, "r") as f:
    camera_captions = json.load(f)

# Инициализация nuScenes и выбор val-сцен
nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_dataroot, verbose=False)
val_scenes = set(create_splits_scenes()["val"])

val_sample_tokens = []
for scene in nusc.scene:
    if scene["name"] in val_scenes:
        scene_tokens = []
        current_sample_token = scene["first_sample_token"]
        while current_sample_token != "":
            scene_tokens.append(current_sample_token)
            current_sample_token = nusc.get("sample", current_sample_token)["next"]
        if args.val_samples_per_scene > 0:
            step = max(1, len(scene_tokens) // args.val_samples_per_scene)
            scene_tokens = scene_tokens[::step][:args.val_samples_per_scene]
        val_sample_tokens.extend(scene_tokens)

if args.max_val_samples > 0:
    val_sample_tokens = val_sample_tokens[:args.max_val_samples]
print(f"Selected {len(val_sample_tokens)} validation samples.")

# Загрузка модели L/14 @ 336px
model_name = "EVA02-CLIP-L-14-336"
pretrained = "eva_clip"
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, _ = create_model_and_transforms(
    model_name,
    pretrained,
    force_custom_clip=True
)
tokenizer = get_tokenizer(model_name)
model = model.to(device).eval()

camera_text_embs = {}

for sample_token in tqdm(val_sample_tokens, desc="Encoding text (L/14-336)"):
    if sample_token not in camera_captions:
        continue
    cam_dict = camera_captions[sample_token]
    embs_dict = {}
    for cam_name, captions in cam_dict.items():
        if not captions:
            continue
        if args.no_prefix:
            scene_text = " ".join(captions)
        else:
            scene_text = "The camera view contains: " + "; ".join(captions)
        text_input = tokenizer([scene_text]).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        embs_dict[cam_name] = text_features.cpu().squeeze(0)  # [1024]
    camera_text_embs[sample_token] = embs_dict

torch.save(camera_text_embs, output_path)
print(f"Saved to {output_path}")