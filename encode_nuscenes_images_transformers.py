# tools/encode_nuscenes_images_transformers.py
import os
import torch
import json
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from nuscenes import NuScenes
from tqdm import tqdm

# Конфигурация
nuscenes_dataroot = "/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval"
output_path = os.path.join("./", "camera_image_embeddings_transformers.pth")

# Загрузка CLIP модели (та же, что и для текста!)
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).eval().cuda()

# Имена камер
CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT"
]

# Инициализация NuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval", verbose=False)

# Словарь для эмбеддингов
camera_image_embs = {}

# Проход по всем сценам и сэмплам
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
                # Загрузка изображения
                image = Image.open(img_path).convert("RGB")
                
                # Препроцессинг и кодирование
                inputs = processor(images=image, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)  # [1, 512]
                    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
                cam_data[cam_name] = image_features.cpu().squeeze(0)  # [512]
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                cam_data[cam_name] = torch.zeros(512)  # fallback
        
        camera_image_embs[current_sample_token] = cam_data
        current_sample_token = sample["next"]

# Сохранение
torch.save(camera_image_embs, output_path)
print(f"Saved image embeddings to {output_path}")