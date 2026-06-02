# tools/encode_nuscenes_images_evaclip.py
import os
import torch
import argparse
from PIL import Image
import sys
sys.path.insert(0, "/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/EVA/EVA-CLIP/rei")
from eva_clip import create_model_and_transforms
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Encode nuScenes images with EVA-CLIP-B/16")
    parser.add_argument("--dataroot", type=str, default="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval")
    parser.add_argument("--output_path", type=str, default="./camera_image_embeddings_evaclip_val150.pth")
    parser.add_argument("--max_val_samples", type=int, default=-1,
        help="Max validation samples (-1 = all)")
    parser.add_argument("--val_samples_per_scene", type=int, default=-1,
        help="Number of evenly spaced samples per validation scene (-1 = all)")
    return parser.parse_args()

args = parse_args()
nuscenes_dataroot = args.dataroot
output_path = args.output_path

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
print(f"Selected {len(val_sample_tokens)} validation samples for image encoding.")

# Загрузка модели
model_name = "EVA02-CLIP-B-16"
pretrained = "eva_clip"
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
model = model.to(device).eval()

CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
]

camera_image_embs = {}

for sample_token in tqdm(val_sample_tokens, desc="Encoding images"):
    sample = nusc.get("sample", sample_token)
    cam_data = {}
    for cam_name in CAMERAS:
        cam_token = sample["data"][cam_name]
        cam_sample_data = nusc.get("sample_data", cam_token)
        img_path = os.path.join(nuscenes_dataroot, cam_sample_data["filename"])
        try:
            image = Image.open(img_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            cam_data[cam_name] = image_features.cpu().squeeze(0)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            cam_data[cam_name] = torch.zeros(768)
    camera_image_embs[sample_token] = cam_data

torch.save(camera_image_embs, output_path)
print(f"Saved to {output_path}")