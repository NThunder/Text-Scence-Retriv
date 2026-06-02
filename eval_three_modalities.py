import os
import torch
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from tqdm import tqdm
import numpy as np
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate three-modal retrieval")
    parser.add_argument("--dataroot", type=str, default="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval")
    parser.add_argument("--text_emb_path", type=str, default="./camera_text_embeddings_llm2clip_openai_l14_336_motion_map_lora.pth")
    parser.add_argument("--image_emb_path", type=str, default="./camera_image_embeddings_llm2clip_openai_l14_336_val150_motion_map_lora.pth")
    parser.add_argument("--point_emb_path", type=str, default="./camera_point_embeddings_val150_motion_map.pth")
    parser.add_argument("--attributes_path", type=str, default="./camera_aware_captions_motion_map.json")
    parser.add_argument("--max_val_samples", type=int, default=-1, help="Max validation samples (-1 = all)")
    parser.add_argument("--val_samples_per_scene", type=int, default=-1, help="Number of evenly spaced samples per validation scene (-1 = all)")
    return parser.parse_args()

args = parse_args()

# Загрузка эмбеддингов
print("Loading embeddings...")
text_embs_all = torch.load(args.text_emb_path, map_location="cpu")
image_embs_all = torch.load(args.image_emb_path, map_location="cpu")
point_embs_all = torch.load(args.point_emb_path, map_location="cpu")

with open(args.attributes_path, "r") as f:
    attributes_all = json.load(f)

# Инициализация nuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=False)
val_scenes = set(create_splits_scenes()["val"])

# Сбор sample_token'ов
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

CAMERAS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

# Сбор всех эмбеддингов
all_text_embs, all_image_embs, all_point_embs = [], [], []
index_to_sample_cam = []
attr_list = []

for sample_token in val_sample_tokens:
    for cam in CAMERAS:
        if (sample_token in text_embs_all and cam in text_embs_all[sample_token] and
            sample_token in image_embs_all and cam in image_embs_all[sample_token] and
            sample_token in point_embs_all and cam in point_embs_all[sample_token]):
            
            all_text_embs.append(text_embs_all[sample_token][cam])
            all_image_embs.append(image_embs_all[sample_token][cam])
            all_point_embs.append(point_embs_all[sample_token][cam])
            attr_list.append(tuple(sorted(attributes_all[sample_token][cam])))
            index_to_sample_cam.append((sample_token, cam))

# Конвертация в тензоры
text_embs = torch.stack(all_text_embs)
image_embs = torch.stack(all_image_embs)
point_embs = torch.stack(all_point_embs)

# Нормализация
text_embs = torch.nn.functional.normalize(text_embs, p=2, dim=1)
image_embs = torch.nn.functional.normalize(image_embs, p=2, dim=1)
point_embs = torch.nn.functional.normalize(point_embs, p=2, dim=1)

print(f"Total pairs: {len(text_embs)}")
print(f"Text emb dim: {text_embs.shape[1]}")
print(f"Image emb dim: {image_embs.shape[1]}")
print(f"Point emb dim: {point_embs.shape[1]}")

# Функция для вычисления метрик
def compute_metrics(query_embs, target_embs, attr_list, is_symmetric=False):
    similarity = torch.mm(query_embs, target_embs.T)
    ranks = []
    
    attr_to_indices = {}
    for idx, attrs in enumerate(attr_list):
        for attr in attrs:
            if attr not in attr_to_indices:
                attr_to_indices[attr] = set()
            attr_to_indices[attr].add(idx)
    
    for i in range(len(query_embs)):
        query_attrs = set(attr_list[i])
        relevant = set()
        for attr in query_attrs:
            relevant.update(attr_to_indices.get(attr, set()))
        
        if is_symmetric:
            relevant.discard(i)
        
        sorted_indices = torch.argsort(similarity[i], descending=True)
        found = False
        for rank_pos, idx in enumerate(sorted_indices.tolist()):
            if idx in relevant:
                ranks.append(rank_pos + 1)
                found = True
                break
        if not found:
            ranks.append(len(query_embs))
    
    ranks = np.array(ranks)
    return {
        'R@1': np.mean(ranks <= 1),
        'R@5': np.mean(ranks <= 5),
        'R@10': np.mean(ranks <= 10),
        'MRR': np.mean(1.0 / ranks),
        'Median': np.median(ranks)
    }

# Вычисление всех метрик
print("\n" + "="*60)
print("TEXT → IMAGE")
metrics_t2i = compute_metrics(text_embs, image_embs, attr_list)
print(f"R@1: {metrics_t2i['R@1']:.4f}")
print(f"R@5: {metrics_t2i['R@5']:.4f}")
print(f"R@10: {metrics_t2i['R@10']:.4f}")
print(f"MRR: {metrics_t2i['MRR']:.4f}")
print(f"Median: {metrics_t2i['Median']:.1f}")

print("\n" + "="*60)
print("IMAGE → TEXT")
metrics_i2t = compute_metrics(image_embs, text_embs, attr_list)
print(f"R@1: {metrics_i2t['R@1']:.4f}")
print(f"R@5: {metrics_i2t['R@5']:.4f}")
print(f"R@10: {metrics_i2t['R@10']:.4f}")
print(f"MRR: {metrics_i2t['MRR']:.4f}")

print("\n" + "="*60)
print("TEXT → POINT CLOUD")
metrics_t2p = compute_metrics(text_embs, point_embs, attr_list)
print(f"R@1: {metrics_t2p['R@1']:.4f}")
print(f"R@5: {metrics_t2p['R@5']:.4f}")
print(f"R@10: {metrics_t2p['R@10']:.4f}")
print(f"MRR: {metrics_t2p['MRR']:.4f}")

print("\n" + "="*60)
print("POINT CLOUD → TEXT")
metrics_p2t = compute_metrics(point_embs, text_embs, attr_list)
print(f"R@1: {metrics_p2t['R@1']:.4f}")
print(f"R@5: {metrics_p2t['R@5']:.4f}")
print(f"R@10: {metrics_p2t['R@10']:.4f}")
print(f"MRR: {metrics_p2t['MRR']:.4f}")

print("\n" + "="*60)
print("IMAGE → POINT CLOUD")
metrics_i2p = compute_metrics(image_embs, point_embs, attr_list)
print(f"R@1: {metrics_i2p['R@1']:.4f}")
print(f"R@5: {metrics_i2p['R@5']:.4f}")
print(f"R@10: {metrics_i2p['R@10']:.4f}")
print(f"MRR: {metrics_i2p['MRR']:.4f}")

print("\n" + "="*60)
print("POINT CLOUD → IMAGE")
metrics_p2i = compute_metrics(point_embs, image_embs, attr_list)
print(f"R@1: {metrics_p2i['R@1']:.4f}")
print(f"R@5: {metrics_p2i['R@5']:.4f}")
print(f"R@10: {metrics_p2i['R@10']:.4f}")
print(f"MRR: {metrics_p2i['MRR']:.4f}")