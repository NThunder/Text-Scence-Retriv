# tools/encode_camera_aware_captions_evaclip.py
import json
import torch

import sys
sys.path.insert(0, "/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/EVA/EVA-CLIP/rei")
from eva_clip import create_model_and_transforms, get_tokenizer
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from tqdm import tqdm

nuscenes_dataroot = "/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval"
camera_captions_path = "./camera_aware_captions_short.json"
output_path = "./camera_text_embeddings_evaclip_val150.pth"

with open(camera_captions_path, "r") as f:
    camera_captions = json.load(f)

# Инициализация nuScenes и выбор val-сцен
nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_dataroot, verbose=False)
val_scenes = set(create_splits_scenes()["val"])

# Сбор первых 150 val sample tokens
val_sample_tokens = []
for scene in nusc.scene:
    if scene["name"] in val_scenes:
        current_sample_token = scene["first_sample_token"]
        while current_sample_token != "" and len(val_sample_tokens) < 150:
            val_sample_tokens.append(current_sample_token)
            current_sample_token = nusc.get("sample", current_sample_token)["next"]
        if len(val_sample_tokens) >= 150:
            break
val_sample_tokens = val_sample_tokens[:150]
print(f"Selected {len(val_sample_tokens)} validation samples for encoding.")

# Загрузка модели
model_name = "EVA02-CLIP-B-16"
pretrained = "eva_clip"
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, _ = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
tokenizer = get_tokenizer(model_name)
model = model.to(device).eval()

camera_text_embs = {}

for sample_token in tqdm(val_sample_tokens, desc="Encoding text"):
    if sample_token not in camera_captions:
        continue
    cam_dict = camera_captions[sample_token]
    embs_dict = {}
    for cam_name, captions in cam_dict.items():
        if not captions:
            continue
        scene_text = "The camera view contains: " + "; ".join(captions)
        text_input = tokenizer([scene_text]).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        embs_dict[cam_name] = text_features.cpu().squeeze(0)
    camera_text_embs[sample_token] = embs_dict

torch.save(camera_text_embs, output_path)
print(f"Saved to {output_path}")