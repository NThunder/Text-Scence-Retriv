# tools/encode_camera_aware_captions_llm2clip.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import json
import torch
from llm2vec import LLM2Vec
from transformers import AutoModel, AutoTokenizer, AutoConfig, CLIPImageProcessor
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from tqdm import tqdm

import argparse

# === Конфигурация ===
def parse_args():
    parser = argparse.ArgumentParser(description="Encode camera aware captions with LLM2CLIP")
    parser.add_argument("--dataroot", type=str, default="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval", help="Path to nuScenes dataset")
    parser.add_argument("--camera_captions_path", type=str, default="./camera_aware_captions_motion_map.json", help="Path to camera captions")
    parser.add_argument("--output_path", type=str, default="./camera_text_embeddings_llm2clip_openai_l14_336_motion_map4.pth", help="Path to output embeddings")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA adapter")
    return parser.parse_args()

args = parse_args()

nuscenes_dataroot = args.dataroot
camera_captions_path = args.camera_captions_path
output_path = args.output_path
lora_path = args.lora_path

with open(camera_captions_path, "r") as f:
    camera_captions = json.load(f)

# === Загрузка LLM2CLIP (OpenAI версия) ===
print("Loading LLM2CLIP model (OpenAI L/14-336)...")
model_name_or_path = "microsoft/LLM2CLIP-Openai-L-14-336"
clip_model = AutoModel.from_pretrained(
    model_name_or_path,
    trust_remote_code=True
).cuda().eval()

# === Загрузка LLM2Vec (Llama-3) ===
llm_model_name = "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned"
config = AutoConfig.from_pretrained(llm_model_name, trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=False)
model_name = lora_path
if not lora_path:
    model_name = llm_model_name
llm_model = AutoModel.from_pretrained(
    model_name,
    config=config,
    trust_remote_code=False
)
llm_model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
l2v = LLM2Vec(llm_model, tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512)

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

# === Генерация текстовых эмбеддингов ===
camera_text_embs = {}

for sample_token in tqdm(val_sample_tokens, desc="Encoding with LLM2CLIP"):
    if sample_token not in camera_captions:
        continue
    cam_dict = camera_captions[sample_token]
    embs_dict = {}
    for cam_name, captions in cam_dict.items():
        if not captions:
            continue
        scene_text = "The camera view contains: " + "; ".join(captions)
        
        with torch.no_grad():
            # 1. Получаем 4096-dim от LLM2Vec
            text_emb_raw = l2v.encode([scene_text], convert_to_tensor=True).to('cuda')  # [1, 4096]
            # 2. Проецируем в CLIP-пространство через get_text_features
            text_emb_clip = clip_model.get_text_features(text_emb_raw)  # [1, 768]
            text_emb_clip = torch.nn.functional.normalize(text_emb_clip, p=2, dim=-1)
            embs_dict[cam_name] = text_emb_clip.cpu().squeeze(0)
    camera_text_embs[sample_token] = embs_dict

torch.save(camera_text_embs, output_path)
print(f"Saved to {output_path}")