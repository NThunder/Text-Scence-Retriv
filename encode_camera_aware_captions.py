# tools/encode_camera_aware_captions.py
import json
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from nuscenes import NuScenes
from tqdm import tqdm

nuscenes_dataroot = "./"
camera_captions_path = f"{nuscenes_dataroot}/camera_aware_captions_motion_map.json"
output_path = f"{nuscenes_dataroot}/camera_text_embeddings_motion_map_clip.pth"

with open(camera_captions_path, "r") as f:
    camera_captions = json.load(f)

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").eval().cuda()

nusc = NuScenes(version='v1.0-trainval', dataroot="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval", verbose=False)

camera_text_embs = {}  # {sample_token: {cam_name: emb}}

for scene in tqdm(nusc.scene):
    current_sample_token = scene["first_sample_token"]
    while current_sample_token != "":
        if current_sample_token not in camera_captions:
            current_sample_token = nusc.get("sample", current_sample_token)["next"]
            continue

        cam_dict = camera_captions[current_sample_token]
        embs_dict = {}

        for cam_name, captions in cam_dict.items():
            if captions:
                scene_text = "The camera view contains: " + "; ".join(captions)
            else:
                continue

            inputs = tokenizer(scene_text, return_tensors="pt", padding=True, truncation=True, max_length=77).to("cuda")
            with torch.no_grad():
                emb = text_encoder(**inputs).pooler_output.squeeze(0).cpu()
            embs_dict[cam_name] = emb

        camera_text_embs[current_sample_token] = embs_dict
        current_sample_token = nusc.get("sample", current_sample_token)["next"]

torch.save(camera_text_embs, output_path)
print(f"Saved to {output_path}")

# === Отладка: вывод примеров ===

# import os

# print("\n=== Примеры (изображение + текст) ===")
# count = 0
# for sample_token, cam_dict in camera_captions.items():
#     if count >= 3:  # покажем 3 примера
#         break
#     print(f"\nSample Token: {sample_token}")
    
#     # Получим путь к изображению (например, CAM_FRONT)
#     sample = nusc.get("sample", sample_token)
#     cam_token = sample["data"]["CAM_FRONT"]
#     cam_data = nusc.get("sample_data", cam_token)
#     img_path = os.path.join(nuscenes_dataroot, cam_data["filename"])
#     print(f"Image path: {img_path}")
    
#     # Текстовое описание для CAM_FRONT
#     if "CAM_FRONT" in cam_dict and cam_dict["CAM_FRONT"]:
#         text = "The camera view contains: " + "; ".join(cam_dict["CAM_FRONT"])
#         print(f"Text: {text}")
#     else:
#         print("Text: (no objects)")
    
#     count += 1