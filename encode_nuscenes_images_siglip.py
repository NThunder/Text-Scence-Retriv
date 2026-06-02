# tools/encode_nuscenes_images_siglip.py
import os
import torch
import json
import argparse
from PIL import Image
from transformers import AutoProcessor, AutoModel
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Encode nuScenes images with SigLIP")
    parser.add_argument("--dataroot", type=str, default="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval")
    parser.add_argument("--output_path", type=str, default="./camera_image_embeddings_siglip.pth")
    parser.add_argument("--max_val_samples", type=int, default=-1,
        help="Max validation samples (-1 = all)")
    parser.add_argument("--val_samples_per_scene", type=int, default=-1,
        help="Number of evenly spaced samples per validation scene (-1 = all)")
    return parser.parse_args()

args = parse_args()
nuscenes_dataroot = args.dataroot
output_path = args.output_path

model_name = "google/siglip-base-patch16-224"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).eval().cuda()

CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
]

nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_dataroot, verbose=False)

val_scenes = set(create_splits_scenes()["val"])
val_sample_tokens = []
for scene in nusc.scene:
    if scene["name"] in val_scenes:
        scene_tokens = []
        current = scene["first_sample_token"]
        while current != "":
            scene_tokens.append(current)
            current = nusc.get("sample", current)["next"]
        if args.val_samples_per_scene > 0:
            step = max(1, len(scene_tokens) // args.val_samples_per_scene)
            scene_tokens = scene_tokens[::step][:args.val_samples_per_scene]
        val_sample_tokens.extend(scene_tokens)

if args.max_val_samples > 0:
    val_sample_tokens = val_sample_tokens[:args.max_val_samples]
print(f"Selected {len(val_sample_tokens)} validation samples.")

camera_image_embs = {}

for sample_token in tqdm(val_sample_tokens, desc="Encoding images (SigLIP)"):
    sample = nusc.get("sample", sample_token)
    cam_data = {}
    for cam_name in CAMERAS:
        cam_token = sample["data"][cam_name]
        cam_sample_data = nusc.get("sample_data", cam_token)
        img_path = os.path.join(nuscenes_dataroot, cam_sample_data["filename"])
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.vision_model(**inputs)
                emb = outputs.pooler_output  # [1, 768]
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            cam_data[cam_name] = emb.cpu().squeeze(0)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            cam_data[cam_name] = torch.zeros(768)
    camera_image_embs[sample_token] = cam_data

torch.save(camera_image_embs, output_path)
print(f"Saved image embeddings to {output_path}")