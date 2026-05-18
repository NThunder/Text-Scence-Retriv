# tools/debug_camera_aware_samples.py
import os
import json
from PIL import Image
from nuscenes import NuScenes

import argparse

# Конфигурация
def parse_args():
    parser = argparse.ArgumentParser(description="Debug camera-aware samples")
    parser.add_argument("--dataroot", type=str, default="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval", help="Path to nuScenes dataset")
    parser.add_argument("--camera_captions_path", type=str, default="./camera_aware_captions_short.json", help="Path to camera captions json")
    parser.add_argument("--output_dir", type=str, default="./debug_samples", help="Path to output directory")
    return parser.parse_args()

args = parse_args()
NUSCENES_DATAROOT = args.dataroot
CAMERA_CAPTIONS_PATH = args.camera_captions_path
OUTPUT_DIR = args.output_dir

# Создание папок
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "texts"), exist_ok=True)

# Загрузка описаний
with open(CAMERA_CAPTIONS_PATH, "r") as f:
    camera_captions = json.load(f)

# Инициализация nuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot=NUSCENES_DATAROOT, verbose=False)

# Камеры (сохраним только те, где есть объекты)
CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
]

# Собираем примеры
samples_to_save = []
count = 0
max_samples = 20  # сколько сцен сохранить

for scene in nusc.scene:
    if count >= max_samples:
        break
    current_sample_token = scene["first_sample_token"]
    while current_sample_token != "" and count < max_samples:
        if current_sample_token in camera_captions:
            samples_to_save.append(current_sample_token)
            count += 1
        current_sample_token = nusc.get("sample", current_sample_token)["next"]

# Сохранение данных
html_content = "<html><body><h1>Camera-Aware Samples</h1>"

for i, sample_token in enumerate(samples_to_save):
    sample = nusc.get("sample", sample_token)
    html_content += f'<h2>Sample {i+1}: {sample_token}</h2>'
    
    for cam_name in CAMERAS:
        # Путь к изображению
        cam_token = sample["data"][cam_name]
        cam_data = nusc.get("sample_data", cam_token)
        src_img_path = os.path.join(NUSCENES_DATAROOT, cam_data["filename"])
        dst_img_path = os.path.join(OUTPUT_DIR, "images", f"{sample_token}_{cam_name}.jpg")
        
        # Сохранение изображения
        try:
            img = Image.open(src_img_path)
            img.save(dst_img_path)
        except Exception as e:
            print(f"Error saving image {src_img_path}: {e}")
            # Создаём заглушку
            img = Image.new('RGB', (224, 224), color='gray')
            img.save(dst_img_path)
        
        # Текстовое описание
        text = "No objects detected."
        if sample_token in camera_captions and cam_name in camera_captions[sample_token]:
            captions = camera_captions[sample_token][cam_name]
            if captions:
                text = "The camera view contains: " + "; ".join(captions)
        
        # Сохранение текста
        text_path = os.path.join(OUTPUT_DIR, "texts", f"{sample_token}_{cam_name}.txt")
        with open(text_path, "w") as f:
            f.write(text)
        
        # Добавление в HTML
        html_content += f'''
        <div style="display:inline-block; margin:10px; border:1px solid #ccc; padding:5px;">
            <h3>{cam_name}</h3>
            <img src="images/{os.path.basename(dst_img_path)}" width="320" style="display:block;">
            <p><small>{text[:200]}{"..." if len(text) > 200 else ""}</small></p>
        </div>
        '''

html_content += "</body></html>"

# Сохранение HTML
with open(os.path.join(OUTPUT_DIR, "index.html"), "w") as f:
    f.write(html_content)

print(f"Saved {len(samples_to_save)} samples to {OUTPUT_DIR}/")
print(f"Open {OUTPUT_DIR}/index.html in your browser to view examples.")