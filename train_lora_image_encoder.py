# tools/train_lora_image_encoder.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor
from peft import LoraConfig, get_peft_model
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from tqdm import tqdm
import json

import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from tqdm import tqdm

import argparse

# === Конфигурация ===
def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA image encoder")
    parser.add_argument("--dataroot", type=str, default="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval", help="Path to nuScenes dataset")
    parser.add_argument("--text_emb_path", type=str, default="./camera_text_embeddings_llm2clip_openai_l14_336_train750_motion_map.pth", help="Path to text embeddings")
    parser.add_argument("--output_lora_dir", type=str, default="./lora_image_encoder_nuscenes", help="Path to output LoRA directory")
    return parser.parse_args()

args = parse_args()

NUSCENES_DATAROOT = args.dataroot
TEXT_EMB_PATH = args.text_emb_path  # ← вы должны сначала создать этот файл
OUTPUT_LORA_DIR = args.output_lora_dir
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 3

# === Загрузка текстовых эмбеддингов (предвычисленных) ===
print("Loading precomputed text embeddings...")
text_embs = torch.load(TEXT_EMB_PATH, map_location="cpu")  # dict: sample_token → {cam: emb}

# === Подготовка train-сэмплов (750 сцен) ===
nusc = NuScenes(version='v1.0-trainval', dataroot=NUSCENES_DATAROOT, verbose=False)
train_scenes = set(create_splits_scenes()["train"])

train_sample_tokens = []
for scene in nusc.scene:
    if scene["name"] in train_scenes:
        current = scene["first_sample_token"]
        while current != "" and len(train_sample_tokens) < 750:
            train_sample_tokens.append(current)
            current = nusc.get("sample", current)["next"]
        if len(train_sample_tokens) >= 750:
            break
train_sample_tokens = train_sample_tokens[:750]
print(f"Selected {len(train_sample_tokens)} train samples.")

# === Собираем список (sample_token, cam) с эмбеддингами ===
train_pairs = []
for sample_token in train_sample_tokens:
    if sample_token not in text_embs:
        continue
    for cam in ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
                "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]:
        if cam in text_embs[sample_token]:
            train_pairs.append((sample_token, cam))

print(f"Total training pairs: {len(train_pairs)}")

# === Датасет ===
class NuScenesImageTextDataset(Dataset):
    def __init__(self, nusc, pairs, text_embs, processor):
        self.nusc = nusc
        self.pairs = pairs
        self.text_embs = text_embs
        self.processor = processor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sample_token, cam = self.pairs[idx]
        # Загрузка изображения
        sample = self.nusc.get("sample", sample_token)
        cam_token = sample["data"][cam]
        cam_data = self.nusc.get("sample_data", cam_token)
        img_path = os.path.join(NUSCENES_DATAROOT, cam_data["filename"])
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        # Загрузка текстового эмбеддинга
        text_emb = self.text_embs[sample_token][cam]
        return inputs, text_emb

# === Загрузка модели ===
print("Loading EVA-CLIP vision encoder (L/14-336)...")
# Используем openai/clip-vit-large-patch14-336 как прокси (веса совместимы с EVA-CLIP)
# vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
vision_model = AutoModel.from_pretrained(
    "microsoft/LLM2CLIP-Openai-L-14-336",
    trust_remote_code=True,
    torch_dtype=torch.float16
).cuda()

# Заморозка backbone
for param in vision_model.parameters():
    param.requires_grad = False

# Применение LoRA к attention-слоям
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # стандарт для ViT
    lora_dropout=0.1,
    bias="none",
)
vision_model = get_peft_model(vision_model, lora_config)
vision_model.print_trainable_parameters()
visual_projection = torch.nn.Linear(1024, 1280).cuda()
vision_model.cuda()
optimizer = torch.optim.AdamW(vision_model.parameters(), lr=LR)

# === Даталоадер ===
dataset = NuScenesImageTextDataset(nusc, train_pairs, text_embs, processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# === Обучение ===
vision_model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, text_embs_gt = batch
        images = images.cuda()
        text_embs_gt = text_embs_gt.cuda()

        optimizer.zero_grad()
        image_features = vision_model.get_image_features(images)  # [B, 1024]
        # image_features = visual_projection(image_embeds)     
        image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
        text_embs_gt = torch.nn.functional.normalize(text_embs_gt, p=2, dim=-1)

        # Cosine similarity loss (maximize similarity)
        cos_sim = torch.sum(image_features * text_embs_gt, dim=1)  # [B]
        loss = (1 - cos_sim).mean()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

# === Сохранение LoRA ===
vision_model.save_pretrained(OUTPUT_LORA_DIR)
print(f"LoRA adapter saved to {OUTPUT_LORA_DIR}")