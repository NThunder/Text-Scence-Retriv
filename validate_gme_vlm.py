import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
    parser.add_argument("--relevance_captions_path", type=str, default=None,
        help="Path to captions used for soft-relevance (atts). "
             "If None, uses --camera_captions_path. "
             "Set to camera_aware_captions_short.json to fix relevance at L0 "
             "while varying query text level.")
    parser.add_argument("--model_name", type=str, default="NCSOFT/GME-VARCO-VISION-Embedding",
        help="HuggingFace model name or local path to fine-tuned model (e.g. ./gme_finetuned/best_model)")
    parser.add_argument("--output_logs", type=str, default="./vlm_validation_logs", help="Directory for logs and results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for image encoding")
    parser.add_argument("--temporal_window", type=int, default=10, help="Temporal window for relevance")
    parser.add_argument(
        "--disable_temporal_relevance",
        action="store_true",
        help="Disable temporal neighbors in relevance set",
    )
    parser.add_argument(
        "--relevance_mode",
        type=str,
        default="any",
        choices=["any", "all"],
        help="Attribute relevance mode: any=at least one shared attribute, all=all query attributes must be present",
    )
    parser.add_argument("--no_prefix", action="store_true",
        help="Use comma-separated attributes without 'The camera view contains: ' prefix")
    parser.add_argument("--max_val_samples", type=int, default=-1,
        help="Max validation samples (-1 = all)")
    parser.add_argument("--val_samples_per_scene", type=int, default=-1,
        help="Number of evenly spaced samples per validation scene (-1 = all)")
    parser.add_argument("--filter_pairs", type=str, default=None,
        help="Path to JSON with list of (sample_token, cam) pairs to use for evaluation")
    parser.add_argument("--save_image_embs", type=str, default=None,
        help="Path to save image embeddings for reuse")
    parser.add_argument("--load_image_embs", type=str, default=None,
        help="Path to load precomputed image embeddings (skips image encoding)")
    parser.add_argument("--save_retrieval_results", type=str, default=None,
        help="Path to save attribute-based retrieval examples for teaser")
    return parser.parse_args()

args = parse_args()

# Создаём директории
os.makedirs(args.output_logs, exist_ok=True)
os.makedirs(f"{args.output_logs}/embeddings", exist_ok=True)

# ========== 1. Загрузка модели ==========
model_name = args.model_name
print(f"Loading model: {model_name}")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)

processor = AutoProcessor.from_pretrained(model_name)
tokenizer = processor.tokenizer
device = model.device
print(f"Model loaded on: {device}")

# ========== 2. Загрузка данных ==========
print("Loading camera-aware captions...")
with open(args.camera_captions_path, "r") as f:
    camera_captions = json.load(f)

# Captions used for soft-relevance computation (atts field).
# Can be fixed to L0 (short) while query texts come from a richer level.
relevance_captions_path = args.relevance_captions_path or args.camera_captions_path
if relevance_captions_path != args.camera_captions_path:
    print(f"Loading relevance captions from: {relevance_captions_path}")
    with open(relevance_captions_path, "r") as f:
        relevance_captions = json.load(f)
else:
    relevance_captions = camera_captions
print(f"Query captions  : {args.camera_captions_path}")
print(f"Relevance captions: {relevance_captions_path}")

print("Loading nuScenes...")
nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=False)

# Получаем валидационные сцены
val_scenes = set(create_splits_scenes()["val"])

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
        attrs = camera_captions[sample_token][cam]
        if args.no_prefix:
            scene_text = " ".join(attrs)
        else:
            scene_text = "The camera view contains: " + "; ".join(attrs)
        all_texts.append(scene_text)
        all_text_info.append((sample_token, cam))
        
        # Изображение
        cam_token = sample["data"][cam]
        cam_data = nusc.get("sample_data", cam_token)
        img_path = os.path.join(args.dataroot, cam_data["filename"])
        all_image_paths.append(img_path)
        all_image_info.append((sample_token, cam))

print(f"Total pairs: {len(all_texts)}")

# Фильтрация пар (для ablation study: одинаковое множество при разных порогах)
if args.filter_pairs:
    print(f"Loading pair filter from {args.filter_pairs}")
    with open(args.filter_pairs, "r") as f:
        allowed_pairs = set(tuple(p) for p in json.load(f))
    filtered_texts = []
    filtered_text_info = []
    filtered_image_paths = []
    filtered_image_info = []
    for i in range(len(all_texts)):
        pair = all_text_info[i]
        if pair in allowed_pairs:
            filtered_texts.append(all_texts[i])
            filtered_text_info.append(all_text_info[i])
            filtered_image_paths.append(all_image_paths[i])
            filtered_image_info.append(all_image_info[i])
    all_texts = filtered_texts
    all_text_info = filtered_text_info
    all_image_paths = filtered_image_paths
    all_image_info = filtered_image_info
    print(f"After filter: {len(all_texts)} pairs")

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
if args.load_image_embs:
    print(f"Loading image embeddings from {args.load_image_embs}...")
    loaded = torch.load(args.load_image_embs, map_location="cpu")
    image_embeddings = {}
    for sample_token, cam in all_image_info:
        if sample_token in loaded and cam in loaded[sample_token]:
            if sample_token not in image_embeddings:
                image_embeddings[sample_token] = {}
            image_embeddings[sample_token][cam] = loaded[sample_token][cam]
else:
    print("Encoding images...")
    for i in tqdm(range(0, len(all_image_paths), args.batch_size)):
        batch_paths = all_image_paths[i:i+args.batch_size]
        batch_info = all_image_info[i:i+args.batch_size]

        embeddings = get_image_embeddings(batch_paths)

        for j, (sample_token, cam) in enumerate(batch_info):
            if sample_token not in image_embeddings:
                image_embeddings[sample_token] = {}
            image_embeddings[sample_token][cam] = embeddings[j]

    if args.save_image_embs:
        print(f"Saving image embeddings to {args.save_image_embs}...")
        torch.save(image_embeddings, args.save_image_embs)

# Сохраняем эмбеддинги для этого запуска
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

text_embs = torch.stack(text_emb_list).float()
image_embs = torch.stack(image_emb_list).float()
text_embs = F.normalize(text_embs, p=2, dim=1)
image_embs = F.normalize(image_embs, p=2, dim=1)

print(f"Total valid pairs: {len(text_embs)}")
print(f"Text embeddings shape: {text_embs.shape}")
print(f"Image embeddings shape: {image_embs.shape}")

# Построение матрицы сходства (text → image)
similarity = text_embs @ image_embs.T  # [N, N]

# ========== 7. Создание словарей для релевантности ==========
print("Building relevance dictionaries...")

# Словарь атрибутов для каждого индекса.
# Использует relevance_captions (может быть зафиксировано на L0),
# а не camera_captions (которые содержат тексты запросов нужного уровня).
attr_to_indices = defaultdict(set)
attrs_by_idx = []
for idx, (sample_token, cam) in enumerate(sample_cam_list):
    sample_cam_attrs = set()
    if sample_token in relevance_captions and cam in relevance_captions[sample_token]:
        for attr in relevance_captions[sample_token][cam]:
            attr_to_indices[attr].add(idx)
            sample_cam_attrs.add(attr)
    attrs_by_idx.append(sample_cam_attrs)

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
temporal_enabled = not args.disable_temporal_relevance
temporal_desc = f"+ temporal(window={args.temporal_window})" if temporal_enabled else "(temporal disabled)"
print(f"Computing ranks with attribute relevance mode='{args.relevance_mode}' {temporal_desc}...")
ranks = []
rel_sizes = []

for i in tqdm(range(len(text_embs))):
    query_token, query_cam = sample_cam_list[i]

    # Атрибуты для релевантности берутся из relevance_captions (может быть L0)
    query_attrs = set()
    if query_token in relevance_captions and query_cam in relevance_captions[query_token]:
        query_attrs = set(relevance_captions[query_token][query_cam])

    # Релевантные индексы (по атрибутам)
    relevant = set()
    if args.relevance_mode == "any":
        for attr in query_attrs:
            relevant.update(attr_to_indices[attr])
    else:
        if query_attrs:
            for idx, candidate_attrs in enumerate(attrs_by_idx):
                if query_attrs.issubset(candidate_attrs):
                    relevant.add(idx)

    # Добавляем временные соседи
    if temporal_enabled:
        current = query_token
        for _ in range(args.temporal_window):
            current = sample_to_next.get(current, "")
            if not current:
                break
            if (current, query_cam) in sample_cam_to_idx:
                relevant.add(sample_cam_to_idx[(current, query_cam)])

    rel_sizes.append(len(relevant))

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

# ========== 8.5 Сохранение retrieval примеров для teaser ==========
if args.save_retrieval_results:
    print(f"Saving retrieval results to {args.save_retrieval_results}...")
    import random
    random.seed(42)
    # Выбираем успешные (rank <= 1) и неудачные (rank >= 5) запросы
    good = [i for i, r in enumerate(ranks) if r <= 1]
    bad = [i for i, r in enumerate(ranks) if r >= 5]
    selected = random.sample(good, min(5, len(good))) + random.sample(bad, min(5, len(bad)))
    examples = []
    for i in selected:
        query_token, query_cam = sample_cam_list[i]
        query_attrs_list = camera_captions.get(query_token, {}).get(query_cam, [])
        if args.no_prefix:
            query_text = " ".join(query_attrs_list)
        else:
            query_text = "The camera view contains: " + "; ".join(query_attrs_list)
        query_attrs = set(relevance_captions.get(query_token, {}).get(query_cam, []))
        relevant_set = set()
        if args.relevance_mode == "any":
            for attr in query_attrs:
                relevant_set.update(attr_to_indices[attr])
        else:
            if query_attrs:
                for j, cand_attrs in enumerate(attrs_by_idx):
                    if query_attrs.issubset(cand_attrs):
                        relevant_set.add(j)
        sorted_idx = torch.argsort(similarity[i], descending=True)[:10].tolist()
        retrieved = []
        for pos, idx in enumerate(sorted_idx):
            tok, cam = sample_cam_list[idx]
            retrieved.append({
                "rank": pos + 1,
                "relevant": int(idx in relevant_set),
                "sample_token": tok,
                "camera": cam,
            })
        examples.append({
            "query_rank": int(ranks[i]),
            "query_token": query_token,
            "query_camera": query_cam,
            "query_text": query_text,
            "query_attributes": list(query_attrs),
            "top10": retrieved,
        })
    with open(args.save_retrieval_results, "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(examples)} retrieval examples to {args.save_retrieval_results}")

# ========== 9. Вывод метрик ==========
ranks = np.array(ranks)
r1 = np.mean(ranks <= 1)
r5 = np.mean(ranks <= 5)
r10 = np.mean(ranks <= 10)
mrr = np.mean(1.0 / ranks)
median_rank = np.median(ranks)

# Среднее попарное косинусное сходство между текстовыми запросами
sim_tt = text_embs @ text_embs.T
mask = ~torch.eye(len(text_embs), dtype=torch.bool)
avg_text_sim = sim_tt[mask].mean().item()

print("\n" + "="*70)
print("GME-VARCO-VISION-Embedding Validation Results")
print("="*70)
print(f"Query captions  : {args.camera_captions_path}")
print(f"Relevance captions: {relevance_captions_path}")
print(
    "Text → Image Retrieval "
    f"(attribute mode={args.relevance_mode}, temporal={'on' if temporal_enabled else 'off'}):"
)
print(f"  R@1:   {r1:.4f}")
print(f"  R@5:   {r5:.4f}")
print(f"  R@10:  {r10:.4f}")
print(f"  MRR:   {mrr:.4f}")
print(f"  Median Rank: {median_rank:.1f}")
print(f"Relevant set size : mean={np.mean(rel_sizes):.1f}  std={np.std(rel_sizes):.1f}  min={np.min(rel_sizes)}  max={np.max(rel_sizes)}")
print(f"Avg inter-query cosine similarity (text): {avg_text_sim:.4f}")
print("="*70)

# ========== 10. Сохранение результатов ==========
results = {
    'R@1': float(r1),
    'R@5': float(r5),
    'R@10': float(r10),
    'MRR': float(mrr),
    'Median_Rank': float(median_rank),
    'total_pairs': len(text_embs),
    'temporal_window': args.temporal_window if temporal_enabled else 0,
    'temporal_relevance_enabled': temporal_enabled,
    'relevance_mode': args.relevance_mode,
    'model_name': args.model_name,
    'query_captions': args.camera_captions_path,
    'relevance_captions': relevance_captions_path,
    'avg_relevant_set_size': float(np.mean(rel_sizes)),
    'std_relevant_set_size': float(np.std(rel_sizes)),
    'avg_inter_query_cosine_sim': float(avg_text_sim),
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
