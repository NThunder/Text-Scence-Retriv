# tools/encode_nuscenes_images_evaclip_l14_336.py
import sys
import os
# Use relative path for EVA-CLIP rei
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "EVA_CLIP_rei"))

import argparse
import torch
from PIL import Image
from eva_clip import create_model_and_transforms
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Encode nuScenes images with EVA-CLIP")
    parser.add_argument("--dataroot", type=str, default="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval", help="Path to nuScenes dataset")
    parser.add_argument("--output_path", type=str, default="./camera_image_embeddings_evaclip_l14_336_val150.pth", help="Path to output embeddings")
    return parser.parse_args()

args = parse_args()

nuscenes_dataroot = args.dataroot
output_path = args.output_path

# Инициализация nuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_dataroot, verbose=False)
val_scenes = set(create_splits_scenes()["val"])

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
print(f"Selected {len(val_sample_tokens)} validation samples.")

# Загрузка модели
model_name = "EVA02-CLIP-L-14-336"
pretrained = "eva_clip"
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = create_model_and_transforms(
    model_name,
    pretrained,
    force_custom_clip=True
)
model = model.to(device).eval()

CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
]

camera_image_embs = {}

for sample_token in tqdm(val_sample_tokens, desc="Encoding images (L/14-336)"):
    sample = nusc.get("sample", sample_token)
    cam_data = {}
    for cam_name in CAMERAS:
        cam_token = sample["data"][cam_name]
        cam_sample_data = nusc.get("sample_data", cam_token)
        img_path = os.path.join(nuscenes_dataroot, cam_sample_data["filename"])
        try:
            image = Image.open(img_path).convert("RGB")
            # Resize to 336x366 as requested, squeezing the 1280x720 input
            image = image.resize((336, 366), Image.BICUBIC)
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            cam_data[cam_name] = image_features.cpu().squeeze(0)  # [1024]
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            cam_data[cam_name] = torch.zeros(1024)
    camera_image_embs[sample_token] = cam_data

torch.save(camera_image_embs, output_path)
print(f"Saved to {output_path}")