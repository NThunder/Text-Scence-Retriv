import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import argparse
from collections import defaultdict

# ========== Конфигурация ==========
def parse_args():
    parser = argparse.ArgumentParser(description="Validation with GME-VARCO-VISION-Embedding")
    parser.add_argument("--dataroot", type=str, default="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval")
    parser.add_argument("--camera_captions_path", type=str, default="./camera_aware_captions_motion_map.json")
    parser.add_argument("--output_logs", type=str, default="./vlm_validation_logs", help="Directory for logs and results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for image encoding")
    parser.add_argument("--temporal_window", type=int, default=10, help="Temporal window for relevance")
    return parser.parse_args()

args = parse_args()

# Создаём директории
os.makedirs(args.output_logs, exist_ok=True)
os.makedirs(f"{args.output_logs}/embeddings", exist_ok=True)

# ========== 1. Загрузка модели ==========
print("Loading GME-VARCO-VISION-Embedding model...")
model_name = "NCSOFT/GME-VARCO-VISION-Embedding"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",  # Автоматически распределит по 4 GPU
)

processor = AutoProcessor.from_pretrained(model_name)
tokenizer = processor.tokenizer
device = model.device
print(f"Model loaded on: {device}")

# ========== 2. Загрузка данных ==========
print("Loading camera-aware captions...")
with open(args.camera_captions_path, "r") as f:
    camera_captions = json.load(f)

print("Loading nuScenes...")
nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=False)

# Получаем валидационные сцены
val_scenes = set(create_splits_scenes()["val"])

val_sample_tokens = []
for scene in nusc.scene:
    if scene["name"] in val_scenes:
        current = scene["first_sample_token"]
        while current != "" and len(val_sample_tokens) < 150:
            val_sample_tokens.append(current)
            current = nusc.get("sample", current)["next"]
        if len(val_sample_tokens) >= 150:
            break

CAMERAS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

print(f"Selected {len(val_sample_tokens)} validation samples, {len(CAMERAS)} cameras each")

# ========== 3. Подготовка шаблонов сообщений ==========
# Шаблон для текстового запроса
qry_msg_template = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "The camera view contains: "},  # Будет дополнено
        ],
    },
]

# Шаблон для изображения
img_msg_template = [
    {
        "role": "user",
        "content": [{
            "type": "image",
            "image": "image"  # Будет заменено на реальный путь
        }]
    }
]

# Получаем текстовую часть шаблона
img_txt = processor.apply_chat_template(
    img_msg_template, tokenize=False, add_generation_prompt=True
) + tokenizer.eos_token

# ========== 4. Функции для извлечения эмбеддингов ==========
def get_text_embeddings(texts):
    """Извлекает эмбеддинги для списка текстов"""
    # Формируем сообщения
    messages = []
    for text in texts:
        msg = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            },
        ]
        txt = processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        ) + tokenizer.eos_token
        messages.append(txt)
    
    # Batch обработка
    inputs = processor(
        text=messages,
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    with torch.inference_mode():
        outputs = model(
            **inputs, output_hidden_states=True, return_dict=True
        )
        embeddings = outputs.hidden_states[-1][:, -1, :]  # [B, hidden_size]
    
    return F.normalize(embeddings, dim=-1).cpu()

def get_image_embeddings(image_paths):
    """Извлекает эмбеддинги для списка изображений"""
    # Формируем сообщения для изображений
    img_msgs = []
    for img_path in image_paths:
        img_msgs.append([
            {
                "role": "user",
                "content": [{
                    "type": "image",
                    "image": img_path
                }]
            }
        ])
    
    # Обрабатываем изображения
    images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
    
    # Получаем текст для каждого изображения
    texts = [img_txt] * len(images)
    
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    with torch.inference_mode():
        outputs = model(
            **inputs, output_hidden_states=True, return_dict=True
        )
        embeddings = outputs.hidden_states[-1][:, -1, :]
    
    return F.normalize(embeddings, dim=-1).cpu()

# ========== 5. Сбор эмбеддингов для всех образцов ==========
print("\n=== Collecting embeddings ===")

# Словари для хранения эмбеддингов
text_embeddings = {}  # {sample_token: {cam: embedding}}
image_embeddings = {}  # {sample_token: {cam: embedding}}

# Собираем все тексты для batch обработки
all_texts = []
all_text_info = []  # (sample_token, cam)
all_image_paths = []
all_image_info = []  # (sample_token, cam)

for sample_token in tqdm(val_sample_tokens, desc="Preparing data"):
    if sample_token not in camera_captions:
        continue
    sample = nusc.get("sample", sample_token)
    
    for cam in CAMERAS:
        if cam not in camera_captions[sample_token] or not camera_captions[sample_token][cam]:
            continue
        
        # Текст
        scene_text = "The camera view contains: " + "; ".join(camera_captions[sample_token][cam])
        all_texts.append(scene_text)
        all_text_info.append((sample_token, cam))
        
        # Изображение
        cam_token = sample["data"][cam]
        cam_data = nusc.get("sample_data", cam_token)
        img_path = os.path.join(args.dataroot, cam_data["filename"])
        all_image_paths.append(img_path)
        all_image_info.append((sample_token, cam))

print(f"Total pairs: {len(all_texts)}")

# Batch обработка текстов
print("Encoding texts...")
for i in tqdm(range(0, len(all_texts), args.batch_size)):
    batch_texts = all_texts[i:i+args.batch_size]
    batch_info = all_text_info[i:i+args.batch_size]
    
    embeddings = get_text_embeddings(batch_texts)
    
    for j, (sample_token, cam) in enumerate(batch_info):
        if sample_token not in text_embeddings:
            text_embeddings[sample_token] = {}
        text_embeddings[sample_token][cam] = embeddings[j]

# Batch обработка изображений
print("Encoding images...")
for i in tqdm(range(0, len(all_image_paths), args.batch_size)):
    batch_paths = all_image_paths[i:i+args.batch_size]
    batch_info = all_image_info[i:i+args.batch_size]
    
    embeddings = get_image_embeddings(batch_paths)
    
    for j, (sample_token, cam) in enumerate(batch_info):
        if sample_token not in image_embeddings:
            image_embeddings[sample_token] = {}
        image_embeddings[sample_token][cam] = embeddings[j]

# Сохраняем эмбеддинги
torch.save(text_embeddings, f"{args.output_logs}/embeddings/text_embeddings.pth")
torch.save(image_embeddings, f"{args.output_logs}/embeddings/image_embeddings.pth")
print(f"✓ Embeddings saved to {args.output_logs}/embeddings/")

# ========== 6. Построение матрицы сходства и вычисление метрик ==========
print("\n=== Computing metrics ===")

# Собираем все эмбеддинги в массивы
text_emb_list = []
image_emb_list = []
sample_cam_list = []

for sample_token in val_sample_tokens:
    if sample_token not in text_embeddings:
        continue
    for cam in CAMERAS:
        if cam in text_embeddings[sample_token] and cam in image_embeddings[sample_token]:
            text_emb_list.append(text_embeddings[sample_token][cam])
            image_emb_list.append(image_embeddings[sample_token][cam])
            sample_cam_list.append((sample_token, cam))

text_embs = torch.stack(text_emb_list)
image_embs = torch.stack(image_emb_list)

print(f"Total valid pairs: {len(text_embs)}")
print(f"Text embeddings shape: {text_embs.shape}")
print(f"Image embeddings shape: {image_embs.shape}")

# Построение матрицы сходства (text → image)
similarity = text_embs @ image_embs.T  # [N, N]

# ========== 7. Создание словарей для релевантности ==========
print("Building relevance dictionaries...")

# Словарь атрибутов для каждого индекса
attr_to_indices = defaultdict(set)
for idx, (sample_token, cam) in enumerate(sample_cam_list):
    if sample_token in camera_captions and cam in camera_captions[sample_token]:
        for attr in camera_captions[sample_token][cam]:
            attr_to_indices[attr].add(idx)

# Словарь для временных соседей
sample_to_next = {}
for scene in nusc.scene:
    current = scene["first_sample_token"]
    while current != "":
        sample_record = nusc.get("sample", current)
        sample_to_next[current] = sample_record["next"]
        current = sample_record["next"]

# Индекс для быстрого поиска
sample_cam_to_idx = {}
for idx, (token, cam) in enumerate(sample_cam_list):
    sample_cam_to_idx[(token, cam)] = idx

# ========== 8. Вычисление рангов и метрик ==========
print("Computing ranks with attribute and temporal relevance...")
ranks = []

for i in tqdm(range(len(text_embs))):
    query_token, query_cam = sample_cam_list[i]
    
    # Получаем атрибуты запроса
    query_attrs = set()
    if query_token in camera_captions and query_cam in camera_captions[query_token]:
        query_attrs = set(camera_captions[query_token][query_cam])
    
    # Релевантные индексы (по атрибутам)
    relevant = set()
    for attr in query_attrs:
        relevant.update(attr_to_indices[attr])
    
    # Добавляем временные соседи
    current = query_token
    for _ in range(args.temporal_window):
        current = sample_to_next.get(current, "")
        if not current:
            break
        if (current, query_cam) in sample_cam_to_idx:
            relevant.add(sample_cam_to_idx[(current, query_cam)])
    
    # Находим ранг первого релевантного
    sorted_indices = torch.argsort(similarity[i], descending=True)
    found = False
    for rank_pos, idx in enumerate(sorted_indices.tolist()):
        if idx in relevant:
            ranks.append(rank_pos + 1)
            found = True
            break
    if not found:
        ranks.append(len(text_embs))

# ========== 9. Вывод метрик ==========
ranks = np.array(ranks)
r1 = np.mean(ranks <= 1)
r5 = np.mean(ranks <= 5)
r10 = np.mean(ranks <= 10)
mrr = np.mean(1.0 / ranks)
median_rank = np.median(ranks)

print("\n" + "="*70)
print("GME-VARCO-VISION-Embedding Validation Results")
print("="*70)
print(f"Text → Image Retrieval (with attribute + temporal relevance):")
print(f"  R@1:   {r1:.4f}")
print(f"  R@5:   {r5:.4f}")
print(f"  R@10:  {r10:.4f}")
print(f"  MRR:   {mrr:.4f}")
print(f"  Median Rank: {median_rank:.1f}")
print("="*70)

# ========== 10. Сохранение результатов ==========
results = {
    'R@1': float(r1),
    'R@5': float(r5),
    'R@10': float(r10),
    'MRR': float(mrr),
    'Median_Rank': float(median_rank),
    'total_pairs': len(text_embs),
    'temporal_window': args.temporal_window,
    'model_name': model_name
}

with open(f"{args.output_logs}/results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to {args.output_logs}/results.json")

# ========== 11. Сохранение примеров ранжирования ==========
def save_retrieval_examples(similarity, sample_cam_list, num_examples=10):
    """Сохраняет примеры успешного и неудачного ранжирования"""
    
    # Находим лучшие и худшие запросы
    ranks = []
    for i in range(len(similarity)):
        sorted_indices = torch.argsort(similarity[i], descending=True)
        # Находим ранг правильного ответа (i)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    
    ranks = np.array(ranks)
    best_queries = np.argsort(ranks)[:num_examples]
    worst_queries = np.argsort(ranks)[-num_examples:]
    
    examples = {
        'best': [],
        'worst': []
    }
    
    for idx in best_queries:
        token, cam = sample_cam_list[idx]
        examples['best'].append({
            'sample_token': token,
            'camera': cam,
            'rank': int(ranks[idx]),
            'attributes': camera_captions.get(token, {}).get(cam, [])
        })
    
    for idx in worst_queries:
        token, cam = sample_cam_list[idx]
        examples['worst'].append({
            'sample_token': token,
            'camera': cam,
            'rank': int(ranks[idx]),
            'attributes': camera_captions.get(token, {}).get(cam, [])
        })
    
    with open(f"{args.output_logs}/retrieval_examples.json", 'w') as f:
        json.dump(examples, f, indent=2)
    
    print(f"✓ Retrieval examples saved to {args.output_logs}/retrieval_examples.json")

save_retrieval_examples(similarity, sample_cam_list)

print(f"\n✅ Validation completed!")
print(f"Results saved to: {args.output_logs}")