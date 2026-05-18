# tools/encode_nuscenes_images_llm2clip_openai_l14_336_hf.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from tqdm import tqdm

import argparse

# === Конфигурация ===
def parse_args():
    parser = argparse.ArgumentParser(description="Encode nuScenes images with LLM2CLIP")
    parser.add_argument("--dataroot", type=str, default="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval", help="Path to nuScenes dataset")
    parser.add_argument("--output_path", type=str, default="./camera_image_embeddings_llm2clip_openai_l14_336_val150_motion_map_lora4.pth", help="Path to output embeddings")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA adapter")
    return parser.parse_args()

args = parse_args()

nuscenes_dataroot = args.dataroot
output_path = args.output_path
lora_path = args.lora_path

# === Загрузка модели и процессора ===
print("Loading LLM2CLIP image model (OpenAI L/14-336)...")
model_name = "microsoft/LLM2CLIP-Openai-L-14-336"
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
model_name = lora_path
if not lora_path:
    model_name = "microsoft/LLM2CLIP-Openai-L-14-336"
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True
).cuda().eval()

# === Подготовка nuScenes (150 val samples) ===
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

# === Имена камер ===
CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
]

# === Генерация эмбеддингов ===
camera_image_embs = {}

for sample_token in tqdm(val_sample_tokens, desc="Encoding images with LLM2CLIP"):
    sample = nusc.get("sample", sample_token)
    cam_data = {}
    for cam_name in CAMERAS:
        cam_token = sample["data"][cam_name]
        cam_sample_data = nusc.get("sample_data", cam_token)
        img_path = os.path.join(nuscenes_dataroot, cam_sample_data["filename"])
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").pixel_values.to("cuda")
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.get_image_features(inputs)
                image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
            cam_data[cam_name] = image_features.cpu().squeeze(0)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            cam_data[cam_name] = torch.zeros(768)
    camera_image_embs[sample_token] = cam_data

torch.save(camera_image_embs, output_path)
print(f"Saved image embeddings to {output_path}")