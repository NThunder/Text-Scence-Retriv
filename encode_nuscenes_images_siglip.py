# tools/encode_nuscenes_images_siglip.py
import os
import torch
import json
from PIL import Image
from transformers import AutoProcessor, AutoModel
from nuscenes import NuScenes
from tqdm import tqdm

nuscenes_dataroot = "/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval"
output_path = "./camera_image_embeddings_siglip.pth"

model_name = "google/siglip-base-patch16-224"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).eval().cuda()

CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
]

nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_dataroot, verbose=False)
camera_image_embs = {}

for scene in tqdm(nusc.scene, desc="Processing scenes"):
    current_sample_token = scene["first_sample_token"]
    while current_sample_token != "":
        sample = nusc.get("sample", current_sample_token)
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
                cam_data[cam_name] = torch.zeros(768)  # размерность SigLIP
        camera_image_embs[current_sample_token] = cam_data
        current_sample_token = sample["next"]

torch.save(camera_image_embs, output_path)
print(f"Saved image embeddings to {output_path}")