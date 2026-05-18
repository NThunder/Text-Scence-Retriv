import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")

# ========== Конфигурация ==========
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GME-VARCO-VISION-Embedding on nuScenes")
    parser.add_argument("--dataroot", type=str, default="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval")
    parser.add_argument("--camera_captions_path", type=str, default="./camera_aware_captions_motion_map.json")
    parser.add_argument("--output_dir", type=str, default="./gme_finetuned_nuscenes")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--temporal_window", type=int, default=10, help="Temporal window for relevance")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--save_embeddings_every", type=int, default=1, help="Save embeddings every N epochs")
    return parser.parse_args()

args = parse_args()

# Создаём директории
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(f"{args.output_dir}/logs", exist_ok=True)
os.makedirs(f"{args.output_dir}/checkpoints", exist_ok=True)
os.makedirs(f"{args.output_dir}/embeddings", exist_ok=True)
os.makedirs(f"{args.output_dir}/plots", exist_ok=True)

writer = SummaryWriter(f"{args.output_dir}/logs")

# Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = 1
print(f"Using {num_gpus} GPUs")

# ========== 1. Загрузка данных ==========
print("Loading camera-aware captions...")
with open(args.camera_captions_path, "r") as f:
    camera_captions = json.load(f)

print("Loading nuScenes...")
nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=False)

# Получаем train и validation сцены
train_scenes = set(create_splits_scenes()["train"])
val_scenes = set(create_splits_scenes()["val"])

CAMERAS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

# Собираем train пары
train_pairs = []
for scene in nusc.scene:
    if scene["name"] not in train_scenes:
        continue
    current = scene["first_sample_token"]
    while current:
        if current in camera_captions:
            for cam in CAMERAS:
                if cam in camera_captions[current] and camera_captions[current][cam]:
                    text = "The camera view contains: " + "; ".join(camera_captions[current][cam])
                    train_pairs.append((current, cam, text))
        current = nusc.get("sample", current)["next"] if current else None
        if len(train_pairs) >= 10000:
            break
    if len(train_pairs) >= 10000:
        break

print(f"Selected {len(train_pairs)} training pairs")

# Собираем validation пары
val_pairs = []
for scene in nusc.scene:
    if scene["name"] not in val_scenes:
        continue
    current = scene["first_sample_token"]
    while current and len(val_pairs) < 500:
        if current in camera_captions:
            for cam in CAMERAS:
                if cam in camera_captions[current] and camera_captions[current][cam]:
                    text = "The camera view contains: " + "; ".join(camera_captions[current][cam])
                    val_pairs.append((current, cam, text))
        current = nusc.get("sample", current)["next"] if current else None
        if len(val_pairs) >= 500:
            break
    if len(val_pairs) >= 500:
        break

print(f"Selected {len(val_pairs)} validation pairs")

# Создаём словарь для временной близости (sample → next_sample)
sample_to_next = {}
for scene in nusc.scene:
    current = scene["first_sample_token"]
    while current:
        sample_record = nusc.get("sample", current)
        sample_to_next[current] = sample_record["next"]
        current = sample_record["next"]

# ========== 2. Загрузка модели ==========
print("\nLoading GME-VARCO-VISION-Embedding model...")
model_name = "NCSOFT/GME-VARCO-VISION-Embedding"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(model_name)
tokenizer = processor.tokenizer

# Настройка дообучения
if args.use_lora:
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
else:
    for param in model.parameters():
        param.requires_grad = True

# Если нужно возобновить обучение
start_epoch = 0
if args.resume_from:
    checkpoint = torch.load(args.resume_from, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed from epoch {start_epoch}")

model = model.to(device)

# Подготовка для распределённого обучения
if num_gpus > 1:
    model = torch.nn.DataParallel(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
scaler = GradScaler()

# ========== 3. Датасет ==========
class NuScenesVLMDataset(Dataset):
    def __init__(self, pairs, processor, nusc, dataroot):
        self.pairs = pairs
        self.processor = processor
        self.nusc = nusc
        self.dataroot = dataroot
        self.tokenizer = processor.tokenizer
        
        self.img_msg_template = [
            {
                "role": "user",
                "content": [{"type": "image", "image": "image"}]
            }
        ]
        self.img_txt = processor.apply_chat_template(
            self.img_msg_template, tokenize=False, add_generation_prompt=True
        ) + self.tokenizer.eos_token
    
    def __len__(self):
        return len(self.pairs)
    
    def get_image_path(self, sample_token, cam):
        sample = self.nusc.get('sample', sample_token)
        cam_token = sample['data'][cam]
        cam_data = self.nusc.get('sample_data', cam_token)
        return os.path.join(self.dataroot, cam_data['filename'])
    
    def __getitem__(self, idx):
        sample_token, cam, text = self.pairs[idx]
        img_path = self.get_image_path(sample_token, cam)
        image = Image.open(img_path).convert('RGB')
        
        return {
            'image': image,
            'text': text,
            'sample_token': sample_token,
            'camera': cam
        }

def collate_fn(batch):
    return {
        'images': [item['image'] for item in batch],
        'texts': [item['text'] for item in batch],
        'sample_tokens': [item['sample_token'] for item in batch],
        'cameras': [item['camera'] for item in batch]
    }

# Создаём датасеты и загрузчики
train_dataset = NuScenesVLMDataset(train_pairs, processor, nusc, args.dataroot)
val_dataset = NuScenesVLMDataset(val_pairs, processor, nusc, args.dataroot)

train_loader = DataLoader(
    train_dataset, 
    batch_size=args.batch_size * num_gpus, 
    shuffle=True,
    num_workers=args.num_workers, 
    collate_fn=collate_fn,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=args.batch_size * num_gpus, 
    shuffle=False,
    num_workers=args.num_workers, 
    collate_fn=collate_fn,
    pin_memory=True
)

# ========== 4. Класс для хранения истории ==========
class TrainingHistory:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_r1 = []
        self.val_r5 = []
        self.val_r10 = []
        self.val_mrr = []
        self.best_r1 = 0.0
        self.best_epoch = 0
    
    def save_plots(self, output_dir):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Training and Validation Loss
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue', linewidth=2)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Validation Loss
        axes[0, 1].plot(self.val_losses, label='Validation Loss', color='orange', linewidth=2)
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # R@1 Metric
        axes[1, 0].plot(self.val_r1, label='R@1', marker='o', linewidth=2)
        axes[1, 0].set_title('R@1 Metric')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R@1')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
        
        # R@5 and MRR
        axes[1, 1].plot(self.val_r5, label='R@5', marker='s', linewidth=2)
        axes[1, 1].plot(self.val_mrr, label='MRR', marker='^', linewidth=2)
        axes[1, 1].set_title('R@5 and MRR Metrics')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/training_history.png", dpi=150)
        plt.close()
        
        # Save history as JSON
        history_dict = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_r1': self.val_r1,
            'val_r5': self.val_r5,
            'val_r10': self.val_r10,
            'val_mrr': self.val_mrr,
            'best_r1': self.best_r1,
            'best_epoch': self.best_epoch
        }
        with open(f"{output_dir}/plots/history.json", 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"✓ Plots saved to {output_dir}/plots/")

history = TrainingHistory()

# ========== 5. Функции для извлечения эмбеддингов ==========
def get_embeddings(model, images, texts):
    """Извлекает эмбеддинги для батча изображений и текстов"""
    B = len(images)
    
    img_txt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image", "image": "image"}]}],
        tokenize=False, add_generation_prompt=True
    ) + tokenizer.eos_token
    
    txt_prompts = [
        processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": txt}]}],
            tokenize=False, add_generation_prompt=True
        ) + tokenizer.eos_token
        for txt in texts
    ]
    
    img_inputs = processor(
        text=[img_txt] * B,
        images=images,
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    txt_inputs = processor(
        text=txt_prompts,
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    with torch.no_grad():
        img_outputs = model(**img_inputs, output_hidden_states=True, return_dict=True)
        txt_outputs = model(**txt_inputs, output_hidden_states=True, return_dict=True)
        
        img_embs = img_outputs.hidden_states[-1][:, -1, :]
        txt_embs = txt_outputs.hidden_states[-1][:, -1, :]
    
    return F.normalize(img_embs, dim=-1), F.normalize(txt_embs, dim=-1)

def compute_metrics_with_relevance(similarity, sample_tokens, cameras, camera_captions, 
                                    sample_to_next, temporal_window=10):
    """Вычисляет метрики с учётом атрибутов и временной близости"""
    # Создаём индекс для быстрого поиска
    local_index = {}
    for idx, (token, cam) in enumerate(zip(sample_tokens, cameras)):
        local_index[(token, cam)] = idx
    
    # Создаём словарь атрибутов
    attr_to_indices = defaultdict(set)
    for idx, (token, cam) in enumerate(zip(sample_tokens, cameras)):
        if token in camera_captions and cam in camera_captions[token]:
            for attr in camera_captions[token][cam]:
                attr_to_indices[attr].add(idx)
    
    ranks = []
    for i in range(len(sample_tokens)):
        sample_token, cam = sample_tokens[i], cameras[i]
        
        # Релевантные индексы по атрибутам
        relevant = set()
        if sample_token in camera_captions and cam in camera_captions[sample_token]:
            for attr in camera_captions[sample_token][cam]:
                relevant.update(attr_to_indices[attr])
        
        # Добавляем временные соседи
        current = sample_token
        for _ in range(temporal_window):
            current = sample_to_next.get(current, "")
            if not current:
                break
            if (current, cam) in local_index:
                relevant.add(local_index[(current, cam)])
        
        # Находим ранг первого релевантного
        sorted_indices = torch.argsort(similarity[i], descending=True)
        found = False
        for rank_pos, idx in enumerate(sorted_indices.tolist()):
            if idx in relevant:
                ranks.append(rank_pos + 1)
                found = True
                break
        if not found:
            ranks.append(len(sample_tokens))
    
    ranks = np.array(ranks)
    return {
        'R@1': np.mean(ranks <= 1),
        'R@5': np.mean(ranks <= 5),
        'R@10': np.mean(ranks <= 10),
        'MRR': np.mean(1.0 / ranks),
        'Median': np.median(ranks)
    }

# ========== 6. Функция валидации ==========
def validate(model, val_loader, epoch, save_embeddings=False):
    model.eval()
    
    all_img_embs = []
    all_txt_embs = []
    all_sample_tokens = []
    all_cameras = []
    val_loss = 0.0
    
    # Для сохранения эмбеддингов
    if save_embeddings:
        img_embeddings_dict = {}
        txt_embeddings_dict = {}
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch['images']
            texts = batch['texts']
            sample_tokens = batch['sample_tokens']
            cameras = batch['cameras']
            B = len(images)
            
            img_embs, txt_embs = get_embeddings(model, images, texts)
            
            all_img_embs.append(img_embs.cpu())
            all_txt_embs.append(txt_embs.cpu())
            all_sample_tokens.extend(sample_tokens)
            all_cameras.extend(cameras)
            
            # Validation loss
            logits = img_embs @ txt_embs.T / args.temperature
            labels = torch.arange(B).to(device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            val_loss += loss.item()
            
            # Сохраняем эмбеддинги
            if save_embeddings:
                for j, (token, cam) in enumerate(zip(sample_tokens, cameras)):
                    if token not in img_embeddings_dict:
                        img_embeddings_dict[token] = {}
                        txt_embeddings_dict[token] = {}
                    img_embeddings_dict[token][cam] = img_embs[j].cpu()
                    txt_embeddings_dict[token][cam] = txt_embs[j].cpu()
    
    val_loss /= len(val_loader)
    
    # Конкатенация всех эмбеддингов
    all_img_embs = torch.cat(all_img_embs)
    all_txt_embs = torch.cat(all_txt_embs)
    
    # Вычисляем матрицу сходства
    similarity = all_img_embs @ all_txt_embs.T
    
    # Вычисляем метрики с учётом релевантности
    metrics = compute_metrics_with_relevance(
        similarity, all_sample_tokens, all_cameras, camera_captions,
        sample_to_next, args.temporal_window
    )
    
    print(f"\n{'='*70}")
    print(f"Validation Results (Epoch {epoch+1})")
    print(f"{'='*70}")
    print(f"Text → Image Retrieval (with attribute + temporal relevance):")
    print(f"  R@1:   {metrics['R@1']:.4f}")
    print(f"  R@5:   {metrics['R@5']:.4f}")
    print(f"  R@10:  {metrics['R@10']:.4f}")
    print(f"  MRR:   {metrics['MRR']:.4f}")
    print(f"  Median Rank: {metrics['Median']:.1f}")
    print(f"  Loss:  {val_loss:.4f}")
    print(f"{'='*70}\n")
    
    # Сохраняем эмбеддинги
    if save_embeddings:
        embeddings_dir = f"{args.output_dir}/embeddings/epoch_{epoch+1}"
        os.makedirs(embeddings_dir, exist_ok=True)
        torch.save(img_embeddings_dict, f"{embeddings_dir}/image_embeddings.pth")
        torch.save(txt_embeddings_dict, f"{embeddings_dir}/text_embeddings.pth")
        print(f"✓ Saved embeddings to {embeddings_dir}")
    
    # Логирование в TensorBoard
    writer.add_scalar('Val/R@1', metrics['R@1'], epoch)
    writer.add_scalar('Val/R@5', metrics['R@5'], epoch)
    writer.add_scalar('Val/R@10', metrics['R@10'], epoch)
    writer.add_scalar('Val/MRR', metrics['MRR'], epoch)
    writer.add_scalar('Val/Loss', val_loss, epoch)
    
    model.train()
    return metrics, val_loss

# ========== 7. Функция обучения ==========
def train_epoch(model, train_loader, optimizer, scaler, epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in progress_bar:
        images = batch['images']
        texts = batch['texts']
        B = len(images)
        
        img_txt = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": "image"}]}],
            tokenize=False, add_generation_prompt=True
        ) + tokenizer.eos_token
        
        txt_prompts = [
            processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": txt}]}],
                tokenize=False, add_generation_prompt=True
            ) + tokenizer.eos_token
            for txt in texts
        ]
        
        with autocast(dtype=torch.bfloat16):
            img_inputs = processor(
                text=[img_txt] * B,
                images=images,
                padding=True,
                return_tensors="pt",
            ).to(device)
            
            txt_inputs = processor(
                text=txt_prompts,
                padding=True,
                return_tensors="pt",
            ).to(device)
            
            img_outputs = model(**img_inputs, output_hidden_states=True, return_dict=True)
            txt_outputs = model(**txt_inputs, output_hidden_states=True, return_dict=True)
            
            img_embs = img_outputs.hidden_states[-1][:, -1, :]
            txt_embs = txt_outputs.hidden_states[-1][:, -1, :]
            
            img_embs = F.normalize(img_embs, dim=-1)
            txt_embs = F.normalize(txt_embs, dim=-1)
            
            logits = img_embs @ txt_embs.T / args.temperature
            labels = torch.arange(B).to(device)
            
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")
    
    writer.add_scalar('Train/Loss', avg_loss, epoch)
    return avg_loss

# ========== 8. Цикл обучения ==========
print(f"\n=== Starting fine-tuning for {args.epochs} epochs ===")

for epoch in range(start_epoch, args.epochs):
    # Обучение
    train_loss = train_epoch(model, train_loader, optimizer, scaler, epoch)
    
    # Валидация с сохранением эмбеддингов
    save_emb = (epoch + 1) % args.save_embeddings_every == 0 or epoch == args.epochs - 1
    metrics, val_loss = validate(model, val_loader, epoch, save_embeddings=save_emb)
    
    # Сохраняем историю
    history.train_losses.append(train_loss)
    history.val_losses.append(val_loss)
    history.val_r1.append(metrics['R@1'])
    history.val_r5.append(metrics['R@5'])
    history.val_r10.append(metrics['R@10'])
    history.val_mrr.append(metrics['MRR'])
    
    # Сохраняем чекпоинт
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': {
            'train_losses': history.train_losses,
            'val_losses': history.val_losses,
            'val_r1': history.val_r1,
            'val_r5': history.val_r5,
            'val_r10': history.val_r10,
            'val_mrr': history.val_mrr,
        }
    }
    torch.save(checkpoint, f"{args.output_dir}/checkpoints/checkpoint_epoch_{epoch}.pth")
    
    # Сохраняем лучшую модель
    if metrics['R@1'] > history.best_r1:
        history.best_r1 = metrics['R@1']
        history.best_epoch = epoch + 1
        
        save_dir = f"{args.output_dir}/best_model"
        os.makedirs(save_dir, exist_ok=True)
        
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        
        # Копируем эмбеддинги лучшей эпохи
        if save_emb:
            best_emb_dir = f"{args.output_dir}/embeddings/best"
            os.makedirs(best_emb_dir, exist_ok=True)
            import shutil
            shutil.copytree(f"{args.output_dir}/embeddings/epoch_{epoch+1}", best_emb_dir, dirs_exist_ok=True)
        
        print(f"✓ Best model saved! R@1: {history.best_r1:.4f} (epoch {history.best_epoch})")
    
    scheduler.step()
    
    # Сохраняем графики каждые 2 эпохи
    if (epoch + 1) % 2 == 0 or epoch == args.epochs - 1:
        history.save_plots(args.output_dir)

# Финальное сохранение графиков
history.save_plots(args.output_dir)

# Сохраняем финальные результаты
final_results = {
    'best_r1': history.best_r1,
    'best_epoch': history.best_epoch,
    'final_r1': history.val_r1[-1] if history.val_r1 else 0,
    'final_r5': history.val_r5[-1] if history.val_r5 else 0,
    'final_r10': history.val_r10[-1] if history.val_r10 else 0,
    'final_mrr': history.val_mrr[-1] if history.val_mrr else 0,
}

with open(f"{args.output_dir}/final_results.json", 'w') as f:
    json.dump(final_results, f, indent=2)

print(f"\n{'='*60}")
print(f"✅ Fine-tuning completed!")
print(f"{'='*60}")
print(f"Best R@1: {history.best_r1:.4f} (epoch {history.best_epoch})")
print(f"Final R@1: {final_results['final_r1']:.4f}")
print(f"Final R@5: {final_results['final_r5']:.4f}")
print(f"Final R@10: {final_results['final_r10']:.4f}")
print(f"Final MRR: {final_results['final_mrr']:.4f}")
print(f"{'='*60}")
print(f"Results saved to: {args.output_dir}")
print(f"  - Plots: {args.output_dir}/plots/")
print(f"  - Models: {args.output_dir}/best_model/")
print(f"  - Checkpoints: {args.output_dir}/checkpoints/")
print(f"  - Embeddings: {args.output_dir}/embeddings/")
print(f"  - TensorBoard: tensorboard --logdir {args.output_dir}/logs")
print(f"{'='*60}")

writer.close()