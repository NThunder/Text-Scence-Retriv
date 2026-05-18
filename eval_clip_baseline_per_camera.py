# tools/eval_clip_baseline_per_camera.py
import os
import torch
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from tqdm import tqdm
import numpy as np
import json

import argparse

# Конфигурация
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CLIP baseline")
    parser.add_argument("--dataroot", type=str, default="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval", help="Path to nuScenes dataset")
    parser.add_argument("--text_emb_path", type=str, default="./camera_text_embeddings_llm2clip_openai_l14_336_val150_map.pth", help="Path to text embeddings")
    parser.add_argument("--image_emb_path", type=str, default="./camera_image_embeddings_llm2clip_openai_l14_336_val150_lora.pth", help="Path to image embeddings")
    parser.add_argument("--attributes_path", type=str, default="./camera_aware_captions_short.json", help="Path to attributes json")
    parser.add_argument("--vis_dir", type=str, default="./retrieval_visualizations_evaclip", help="Directory for visualizations")
    return parser.parse_args()

args = parse_args()

NUSCENES_DATAROOT = args.dataroot
# TEXT_EMB_PATH = os.path.join("./", "camera_text_embeddings_evaclip_val150.pth")
# IMAGE_EMB_PATH = os.path.join("./", "camera_image_embeddings_evaclip_val150.pth")
TEXT_EMB_PATH = args.text_emb_path
IMAGE_EMB_PATH = args.image_emb_path

vis_dir = args.vis_dir

ATTRIBUTES_PATH = args.attributes_path  # ← НОВЫЙ ФАЙЛ

print("Loading attributes...")
with open(ATTRIBUTES_PATH, "r") as f:
    attributes_all = json.load(f)

# Загрузка эмбеддингов
print("Loading embeddings...")
text_embs_all = torch.load(TEXT_EMB_PATH, map_location="cpu")
image_embs_all = torch.load(IMAGE_EMB_PATH, map_location="cpu")

# Инициализация nuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot=NUSCENES_DATAROOT, verbose=False)

# Получение валидационных сцен
val_scenes = set(create_splits_scenes()["val"])

# Сбор sample_token'ов (первые 150)
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

sample_to_scene = {}
sample_to_next = {}  # sample_token → next_sample_token (в той же сцене)

for scene in nusc.scene:
    if scene["name"] not in val_scenes:
        continue
    current = scene["first_sample_token"]
    while current != "":
        sample_to_scene[current] = scene["token"]
        sample_record = nusc.get("sample", current)
        sample_to_next[current] = sample_record["next"]
        current = sample_record["next"]

# Камеры
CAMERA_NAMES = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
]

# Собираем все необходимые данные
all_text_embs = []
all_image_embs = []
all_attributes_list = []  # будем хранить как списки или кортежи для хеширования
index_to_sample_cam = []
query_relevant_indices = []  # ← NEW: список множеств/списков релевантных индексов
sample_cam_to_index = {}

for sample_token in tqdm(val_sample_tokens, desc="Loading per-camera embeddings"):
    for cam in CAMERA_NAMES:
        if (sample_token in text_embs_all and cam in text_embs_all[sample_token] and
            sample_token in image_embs_all and cam in image_embs_all[sample_token]):
            all_text_embs.append(text_embs_all[sample_token][cam])
            all_image_embs.append(image_embs_all[sample_token][cam])
            attrs = tuple(sorted(attributes_all[sample_token][cam]))  # хешируемый ключ
            # print("attrs:   ", attrs)
            all_attributes_list.append(attrs)
            idx = len(index_to_sample_cam)
            index_to_sample_cam.append((sample_token, cam))
            sample_cam_to_index[(sample_token, cam)] = idx

# Преобразуем в тензоры
text_embs = torch.stack(all_text_embs)
image_embs = torch.stack(all_image_embs)
text_embs = torch.nn.functional.normalize(text_embs, p=2, dim=1)
image_embs = torch.nn.functional.normalize(image_embs, p=2, dim=1)

# Предварительно создадим маппинг: attrs → список индексов
from collections import defaultdict
attr_to_indices = defaultdict(list)
for idx, attrs in enumerate(all_attributes_list):
    attr_to_indices[attrs].append(idx)

ranks = []
query_relevant_indices = []
similarity = torch.mm(text_embs, image_embs.t())

attr_to_indices_global = defaultdict(set)
for idx, attrs_tuple in enumerate(all_attributes_list):
    for attr in attrs_tuple:
        attr_to_indices_global[attr].add(idx)

# === ШАГ 2: Для каждого запроса собрать все индексы, имеющие хотя бы один общий атрибут ===
query_relevant_indices = []
temporal_window = 10  # как у вас

for i in tqdm(range(len(all_text_embs)), desc="Computing soft relevance (≥1 attr match + temporal)"):
    sample_token, cam = index_to_sample_cam[i]
    query_attrs = set(all_attributes_list[i])  # множество атрибутов запроса

    # 1. Релевантность по атрибутам: объединяем все индексы, где есть хотя бы один общий атрибут
    relevant = set()
    for attr in query_attrs:
        relevant.update(attr_to_indices_global[attr])
    
    # 2. Добавляем временно близкие кадры (следующие N в той же камере)
    current = sample_token
    for _ in range(temporal_window):
        current = sample_to_next.get(current, "")
        if not current:
            break
        if (current, cam) in sample_cam_to_index:
            relevant.add(sample_cam_to_index[(current, cam)])

    # 3. Убираем сам запрос из релевантных? → обычно не нужно, но можно
    # relevant.discard(i)  # раскомментируйте, если не хотите считать сам себя релевантным

    query_relevant_indices.append(list(relevant))

    # 3. Находим ранг первого релевантного
    sorted_indices = torch.argsort(similarity[i], descending=True)
    found = False
    for rank_pos, idx in enumerate(sorted_indices.tolist()):
        if idx in relevant:
            ranks.append(rank_pos + 1)
            found = True
            break
    if not found:
        ranks.append(len(all_text_embs))

# Метрики
ranks = np.array(ranks)
R1 = np.mean(ranks <= 1)
R5 = np.mean(ranks <= 5)
R10 = np.mean(ranks <= 10)
MRR = np.mean(1.0 / ranks)
median_rank = np.median(ranks)

print("\n=== CLIP Baseline (Attribute-Based Retrieval) ===")
print(f"R@1:  {R1:.4f}")
print(f"R@5:  {R5:.4f}")
print(f"R@10: {R10:.4f}")
print(f"MRR:  {MRR:.4f}")
print(f"Median Rank: {median_rank:.1f}")

import matplotlib.pyplot as plt
from PIL import Image
import os

os.makedirs(vis_dir, exist_ok=True)

def load_image(sample_token, cam_name):
    cam_token = nusc.get("sample", sample_token)["data"][cam_name]
    cam_data = nusc.get("sample_data", cam_token)
    img_path = os.path.join(NUSCENES_DATAROOT, cam_data["filename"])
    return Image.open(img_path).convert("RGB")

def format_attrs(attrs_tuple):
    return "\n".join(attrs_tuple) if attrs_tuple else "No attributes"

num_vis = min(20, len(all_text_embs))

for i in range( len(all_text_embs)):
    # Атрибуты запроса
    query_attrs = all_attributes_list[i]
    query_sample, query_cam = index_to_sample_cam[i]
    query_img = load_image(query_sample, query_cam)

    # Топ-1 retrieved
    top1_idx = torch.argmax(similarity[i]).item()
    retrieved_attrs = all_attributes_list[top1_idx]
    retrieved_sample, retrieved_cam = index_to_sample_cam[top1_idx]
    retrieved_img = load_image(retrieved_sample, retrieved_cam)

    # Проверка релевантности
    is_relevant = top1_idx in query_relevant_indices[i]
    match_status = "✅ MATCH" if is_relevant else "❌ MISMATCH"

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Query side
    axes[0].imshow(query_img)
    axes[0].set_title("Query Image", fontsize=12, pad=120)
    axes[0].text(
        0.5, 1.02, format_attrs(query_attrs),
        transform=axes[0].transAxes,
        fontsize=9,
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="orange")
    )
    axes[0].axis('off')

    # Retrieved side
    axes[1].imshow(retrieved_img)
    axes[1].set_title(f"Retrieved (Top-1)\n{match_status}", fontsize=12, pad=120, color='green' if is_relevant else 'red')
    axes[1].text(
        0.5, 1.02, format_attrs(retrieved_attrs),
        transform=axes[1].transAxes,
        fontsize=9,
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan" if is_relevant else "lightcoral", edgecolor="gray")
    )
    axes[1].axis('off')

    plt.suptitle("Attribute-Based Retrieval Example", fontsize=13, y=0.98)
    plt.tight_layout()

    save_path = os.path.join(vis_dir, f"example_{i:03d}.png")
    # plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

print(f"Saved {num_vis} annotated examples to {vis_dir}")