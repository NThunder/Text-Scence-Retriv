import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DataParallel
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor
from peft import LoraConfig, get_peft_model
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
import numpy as np
from tqdm import tqdm
import json
import argparse
import copy
import matplotlib.pyplot as plt
from collections import defaultdict
from llm2vec import LLM2Vec

# Импорт Utonia
import utonia
from utonia.model import PointTransformerV3

# ========== Конфигурация ==========
def parse_args():
    parser = argparse.ArgumentParser(description="Joint training with Utonia (PTv3) and Query Fusion on 4 GPUs")
    parser.add_argument("--dataroot", type=str, default="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval")
    parser.add_argument("--camera_captions_path", type=str, default="./camera_aware_captions_motion_map.json")
    parser.add_argument("--output_lora_text", type=str, default="./lora_text_encoder_utonia")
    parser.add_argument("--output_lora_image", type=str, default="./lora_image_encoder_utonia")
    parser.add_argument("--output_point_model", type=str, default="./point_encoder_utonia")
    parser.add_argument("--output_logs", type=str, default="./training_logs_utonia", help="Directory for logs and plots")
    parser.add_argument("--batch_size", type=int, default=32, help="Total batch size across all GPUs")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--max_points", type=int, default=20000)
    parser.add_argument("--fusion_dim", type=int, default=1280, help="Dimension for fused embedding")
    parser.add_argument("--num_queries", type=int, default=4, help="Number of learnable queries")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    return parser.parse_args()

args = parse_args()

# Создаём директории для логов
os.makedirs(args.output_logs, exist_ok=True)
os.makedirs(f"{args.output_logs}/plots", exist_ok=True)
os.makedirs(f"{args.output_logs}/models", exist_ok=True)
os.makedirs(f"{args.output_logs}/embeddings", exist_ok=True)

# TensorBoard writer
writer = SummaryWriter(f"{args.output_logs}/tensorboard")

# Определяем устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"Using {num_gpus} GPUs: {[f'GPU {i}' for i in range(num_gpus)]}")

# ========== 1. Загрузка captions ==========
print("Loading camera-aware captions...")
with open(args.camera_captions_path, "r") as f:
    camera_captions = json.load(f)

# ========== 2. Подготовка nuScenes ==========
nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=False)
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

# ========== 3. Сбор пар ==========
CAMERAS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

train_pairs = []
for sample_token in train_sample_tokens:
    if sample_token not in camera_captions:
        continue
    for cam in CAMERAS:
        if cam in camera_captions[sample_token] and camera_captions[sample_token][cam]:
            scene_text = "The camera view contains: " + "; ".join(camera_captions[sample_token][cam])
            train_pairs.append((sample_token, cam, scene_text))
print(f"Total training pairs: {len(train_pairs)}")

# ========== 4. Загрузка моделей ==========
print("\n=== Loading models ===")

# ---- Point encoder: Utonia ----
print("Loading Utonia (PointTransformerV3) from HuggingFace...")
from utonia import load

point_model = load("utonia", repo_id="Pointcept/Utonia")
point_model.train()
for param in point_model.parameters():
    param.requires_grad = True
print("Utonia loaded. Number of parameters:", sum(p.numel() for p in point_model.parameters()))

# ---- CLIP projection (заморожена) ----
clip_model = AutoModel.from_pretrained(
    "microsoft/LLM2CLIP-Openai-L-14-336",
    trust_remote_code=True,
    torch_dtype=torch.float32
)
clip_model.eval()
for param in clip_model.parameters():
    param.requires_grad = False

# ---- Vision encoder (CLIP + LoRA) ----
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
vision_model = AutoModel.from_pretrained(
    "microsoft/LLM2CLIP-Openai-L-14-336",
    trust_remote_code=True,
    torch_dtype=torch.float32
)
for param in vision_model.parameters():
    param.requires_grad = False
lora_config_vision = LoraConfig(r=args.lora_r, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                                lora_dropout=0.1, bias="none")
vision_model = get_peft_model(vision_model, lora_config_vision)
vision_model.print_trainable_parameters()

# ---- Text encoder (LLM2Vec + LoRA) ----
llm_model_name = "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=False)
text_model = AutoModel.from_pretrained(llm_model_name, trust_remote_code=False, torch_dtype=torch.float32)
for param in text_model.parameters():
    param.requires_grad = False
lora_config_text = LoraConfig(r=args.lora_r, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                              lora_dropout=0.1, bias="none")
text_model = get_peft_model(text_model, lora_config_text)
text_model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
l2v = LLM2Vec(text_model, tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512)
text_model.print_trainable_parameters()

# Проекционные слои
point_feat_dim = 576
point_proj = nn.Linear(point_feat_dim, args.fusion_dim)

# Fusion модель
class QueryFusion(nn.Module):
    def __init__(self, dim, num_queries, num_heads=8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim))
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.fusion_proj = nn.Linear(dim * num_queries, dim)

    def forward(self, img_emb, pt_emb):
        B = img_emb.shape[0]
        img_seq = img_emb.unsqueeze(1)
        pt_seq = pt_emb.unsqueeze(1)
        multi_modal = torch.cat([img_seq, pt_seq], dim=1)
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        fused, _ = self.cross_attn(queries, multi_modal, multi_modal)
        fused = fused.reshape(B, -1)
        fused = self.fusion_proj(fused)
        return fused

fusion_model = QueryFusion(args.fusion_dim, args.num_queries)

# ========== 5. Move models to device and wrap with DataParallel ==========
print(f"\nMoving models to {num_gpus} GPUs...")

vision_model = vision_model.to(device)
text_model = text_model.to(device)
point_model = point_model.to(device)
clip_model = clip_model.to(device)
point_proj = point_proj.to(device)
fusion_model = fusion_model.to(device)

# Оборачиваем в DataParallel
if num_gpus > 1:
    vision_model = DataParallel(vision_model, device_ids=list(range(num_gpus)))
    text_model = DataParallel(text_model, device_ids=list(range(num_gpus)))
    point_model = DataParallel(point_model, device_ids=list(range(num_gpus)))
    point_proj = DataParallel(point_proj, device_ids=list(range(num_gpus)))
    fusion_model = DataParallel(fusion_model, device_ids=list(range(num_gpus)))
    print("Models wrapped with DataParallel")

# Функция для получения исходной модели (если обёрнута)
def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model

# ========== 6. Transform для Utonia ==========
utonia_transform = utonia.transform.default(scale=1.0, apply_z_positive=False, normalize_coord=False)

# ========== 7. Датасет ==========
class JointNuScenesDataset(Dataset):
    def __init__(self, nusc, pairs, processor, max_points=20000, cache_lidar=True):
        self.nusc = nusc
        self.pairs = pairs
        self.processor = processor
        self.max_points = max_points
        self.cache_lidar = cache_lidar
        self.lidar_cache = {} if cache_lidar else None

    def __len__(self):
        return len(self.pairs)

    def load_lidar(self, sample_token):
        if self.cache_lidar and sample_token in self.lidar_cache:
            return self.lidar_cache[sample_token]
        sample = self.nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        lidar_path = os.path.join(self.nusc.dataroot, lidar_data['filename'])
        pc = LidarPointCloud.from_file(lidar_path)
        if self.cache_lidar:
            self.lidar_cache[sample_token] = pc
        return pc

    def project_points_to_camera(self, pc, sample_token, camera_name, min_dist=1.0):
        sample = self.nusc.get('sample', sample_token)
        cam_token = sample['data'][camera_name]
        cam_data = self.nusc.get('sample_data', cam_token)
        pointsensor = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])

        pc = copy.deepcopy(pc)
        cs_record_lidar = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record_lidar['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record_lidar['translation']))

        poserecord_lidar = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord_lidar['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord_lidar['translation']))

        poserecord_cam = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        pc.translate(-np.array(poserecord_cam['translation']))
        pc.rotate(Quaternion(poserecord_cam['rotation']).rotation_matrix.T)

        cs_record_cam = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record_cam['translation']))
        pc.rotate(Quaternion(cs_record_cam['rotation']).rotation_matrix.T)

        depths = pc.points[2, :]
        points_img = view_points(pc.points[:3, :], np.array(cs_record_cam['camera_intrinsic']), normalize=True)

        h, w = 900, 1600
        mask = depths > min_dist
        mask = np.logical_and(mask, points_img[0, :] >= 0)
        mask = np.logical_and(mask, points_img[0, :] < w - 1)
        mask = np.logical_and(mask, points_img[1, :] >= 0)
        mask = np.logical_and(mask, points_img[1, :] < h - 1)
        valid_indices = np.where(mask)[0]

        if len(valid_indices) == 0:
            return np.zeros((0, 4)), np.zeros((0, 2))

        points_cam = pc.points[:3, valid_indices].T
        intensity = pc.points[3, valid_indices].reshape(-1, 1)
        points_cam_with_intensity = np.concatenate([points_cam, intensity], axis=1)
        pixel_coords = points_img[:, valid_indices].T

        if self.max_points is not None and len(points_cam_with_intensity) > self.max_points:
            idx = np.random.choice(len(points_cam_with_intensity), self.max_points, replace=False)
            points_cam_with_intensity = points_cam_with_intensity[idx]
            pixel_coords = pixel_coords[idx]
        return points_cam_with_intensity, pixel_coords

    def __getitem__(self, idx):
        sample_token, cam, text = self.pairs[idx]
        sample = self.nusc.get('sample', sample_token)
        cam_token = sample['data'][cam]
        cam_data = self.nusc.get('sample_data', cam_token)
        img_path = os.path.join(self.nusc.dataroot, cam_data['filename'])
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.processor(images=image, return_tensors='pt')['pixel_values'][0]

        pc = self.load_lidar(sample_token)
        points_cam, _ = self.project_points_to_camera(pc, sample_token, cam)
        points_tensor = torch.from_numpy(points_cam).float()
        return image_tensor, text, points_tensor, sample_token, cam

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    texts = [item[1] for item in batch]
    points_list = [item[2] for item in batch]
    sample_tokens = [item[3] for item in batch]
    cameras = [item[4] for item in batch]

    all_points = []
    offsets = []
    for pts in points_list:
        all_points.append(pts)
        offsets.append(pts.shape[0])
    all_points = torch.cat(all_points, dim=0)
    offsets = torch.tensor(offsets).cumsum(dim=0).int()
    return images, texts, all_points, offsets, sample_tokens, cameras

train_dataset = JointNuScenesDataset(nusc, train_pairs, processor, args.max_points)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

# ========== 8. Валидационный датасет ==========
def prepare_val_dataset(nusc, camera_captions, processor, num_samples=150):
    val_scenes = set(create_splits_scenes()["val"])
    val_sample_tokens = []
    for scene in nusc.scene:
        if scene["name"] in val_scenes:
            current = scene["first_sample_token"]
            while current != "" and len(val_sample_tokens) < num_samples:
                val_sample_tokens.append(current)
                current = nusc.get("sample", current)["next"]
            if len(val_sample_tokens) >= num_samples:
                break
    val_sample_tokens = val_sample_tokens[:num_samples]

    val_pairs = []
    for sample_token in val_sample_tokens:
        if sample_token not in camera_captions:
            continue
        for cam in CAMERAS:
            if cam in camera_captions[sample_token] and camera_captions[sample_token][cam]:
                scene_text = "The camera view contains: " + "; ".join(camera_captions[sample_token][cam])
                val_pairs.append((sample_token, cam, scene_text))
    return JointNuScenesDataset(nusc, val_pairs, processor), len(val_pairs)

val_dataset, num_val_pairs = prepare_val_dataset(nusc, camera_captions, processor, num_samples=150)
print(f"Prepared validation dataset with {num_val_pairs} pairs")
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

# ========== 9. Оптимизатор ==========
params = [
    {'params': vision_model.parameters(), 'lr': args.lr},
    {'params': text_model.parameters(), 'lr': args.lr},
    {'params': point_model.parameters(), 'lr': args.lr / 10},
    {'params': point_proj.parameters(), 'lr': args.lr},
    {'params': fusion_model.parameters(), 'lr': args.lr},
]
optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# ========== 10. Функции ==========
def offset2batch(offset):
    batch = torch.zeros(offset[-1].item(), dtype=torch.long, device=offset.device)
    for i in range(1, len(offset)):
        batch[offset[i-1]:offset[i]] = i
    return batch

class TrainingHistory:
    def __init__(self):
        self.train_losses = []
        self.val_r1 = []
        self.best_r1 = 0.0
    
    def save_plots(self, output_dir):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/plots/loss.png", dpi=150)
        plt.close()
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.val_r1, label='Validation R@1', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('R@1')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/plots/r1.png", dpi=150)
        plt.close()

history = TrainingHistory()

# ========== 11. Валидация ==========
def validate(epoch):
    vision_model.eval()
    text_model.eval()
    point_model.eval()
    point_proj.eval()
    fusion_model.eval()
    
    # Получаем оригинальные модели для методов
    vision_model_raw = unwrap_model(vision_model)
    text_model_raw = unwrap_model(text_model)
    point_model_raw = unwrap_model(point_model)
    point_proj_raw = unwrap_model(point_proj)
    fusion_model_raw = unwrap_model(fusion_model)
    
    # Для LLM2Vec используем оригинальную модель
    l2v.model = text_model_raw

    all_fused_embs = []
    all_txt_embs = []

    with torch.no_grad():
        for images, texts, all_points, offsets, sample_tokens, cameras in tqdm(val_loader, desc="Validation", leave=False):
            images = images.to(device)
            all_points = all_points.to(device)
            offsets = offsets.to(device)
            B = images.shape[0]

            # Image embedding - используем оригинальную модель
            img_feat = vision_model_raw.get_image_features(images)
            img_emb = F.normalize(img_feat, p=2, dim=-1)

            # Text embedding
            txt_raw = l2v.encode(texts, convert_to_tensor=True).to(device)
            txt_clip = clip_model.get_text_features(txt_raw)
            txt_emb = F.normalize(txt_clip, p=2, dim=-1)
            all_txt_embs.append(txt_emb.cpu())

            # Point embedding
            pt_embs = []
            for i in range(B):
                start = offsets[i-1].item() if i > 0 else 0
                end = offsets[i].item()
                pts = all_points[start:end]
                if pts.shape[0] == 0:
                    pt_embs.append(torch.zeros(args.fusion_dim, device=device))
                    continue
                    
                coord = pts[:, :3]
                color = torch.zeros_like(coord)
                normal = torch.zeros_like(coord)
                
                point_dict = {
                    "coord": coord.cpu().numpy(),
                    "color": color.cpu().numpy(),
                    "normal": normal.cpu().numpy(),
                }
                point_dict = utonia_transform(point_dict)
                
                for k in point_dict.keys():
                    if isinstance(point_dict[k], np.ndarray):
                        point_dict[k] = torch.from_numpy(point_dict[k]).to(device)
                
                out = point_model_raw(point_dict)
                if isinstance(out, dict):
                    feat = out['feat'].mean(dim=0)
                else:
                    feat = out.mean(dim=0)
                pt_emb = F.normalize(point_proj_raw(feat), p=2, dim=-1)
                pt_embs.append(pt_emb)
            pt_emb = torch.stack(pt_embs)

            # Fused embedding
            fused_emb = fusion_model_raw(img_emb, pt_emb)
            fused_emb = F.normalize(fused_emb, p=2, dim=-1)
            all_fused_embs.append(fused_emb.cpu())

    all_fused_embs = torch.cat(all_fused_embs)
    all_txt_embs = torch.cat(all_txt_embs)
    
    sim = all_fused_embs @ all_txt_embs.T
    ranks = []
    for i in range(len(sim)):
        rank = (torch.argsort(sim[i], descending=True) == i).nonzero()[0].item() + 1
        ranks.append(rank)
    
    ranks = np.array(ranks)
    r1 = np.mean(ranks <= 1)
    
    print(f"\nValidation Epoch {epoch+1}: R@1 = {r1:.4f}")
    
    vision_model.train()
    text_model.train()
    point_model.train()
    point_proj.train()
    fusion_model.train()
    
    return r1

# ========== 12. Цикл обучения ==========
print(f"\n=== Starting training on {num_gpus} GPUs for {args.epochs} epochs ===")
best_r1 = 0.0

for epoch in range(args.epochs):
    vision_model.train()
    text_model.train()
    point_model.train()
    point_proj.train()
    fusion_model.train()
    
    # Получаем оригинальные модели для методов
    vision_model_raw = unwrap_model(vision_model)
    text_model_raw = unwrap_model(text_model)
    point_model_raw = unwrap_model(point_model)
    point_proj_raw = unwrap_model(point_proj)
    fusion_model_raw = unwrap_model(fusion_model)
    
    epoch_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
    for images, texts, all_points, offsets, sample_tokens, cameras in progress_bar:
        images = images.to(device)
        all_points = all_points.to(device)
        offsets = offsets.to(device)
        B = images.shape[0]

        # Image embedding - используем оригинальную модель
        img_feat = vision_model_raw.get_image_features(images)
        img_emb = F.normalize(img_feat, p=2, dim=-1)

        # Text embedding - используем оригинальную модель
        l2v.model = text_model_raw
        txt_raw = l2v.encode(texts, convert_to_tensor=True).to(device)
        txt_clip = clip_model.get_text_features(txt_raw)
        txt_emb = F.normalize(txt_clip, p=2, dim=-1)

        # Point embedding
        pt_embs = []
        for i in range(B):
            start = offsets[i-1].item() if i > 0 else 0
            end = offsets[i].item()
            pts = all_points[start:end]
            if pts.shape[0] == 0:
                pt_embs.append(torch.zeros(args.fusion_dim, device=device))
                continue
            coord = pts[:, :3]
            color = torch.zeros_like(coord)
            normal = torch.zeros_like(coord)
            point_dict = {
                "coord": coord.cpu().numpy(),
                "color": color.cpu().numpy(),
                "normal": normal.cpu().numpy(),
            }
            point_dict = utonia_transform(point_dict)
            for k in point_dict.keys():
                if isinstance(point_dict[k], np.ndarray):
                    point_dict[k] = torch.from_numpy(point_dict[k]).to(device)
            out = point_model_raw(point_dict)
            if isinstance(out, dict):
                feat = out['feat'].mean(dim=0)
            else:
                feat = out.mean(dim=0)
            pt_emb = F.normalize(point_proj_raw(feat), p=2, dim=-1)
            pt_embs.append(pt_emb)
        pt_emb = torch.stack(pt_embs)

        # Fused embedding
        fused_emb = fusion_model_raw(img_emb, pt_emb)
        fused_emb = F.normalize(fused_emb, p=2, dim=-1)

        labels = torch.arange(B).to(device)
        
        # Losses
        logits_fused_text = fused_emb @ txt_emb.T / args.temperature
        logits_it = img_emb @ txt_emb.T / args.temperature
        logits_ip = img_emb @ pt_emb.T / args.temperature
        logits_tp = txt_emb @ pt_emb.T / args.temperature

        loss_fused_text = (F.cross_entropy(logits_fused_text, labels) +
                           F.cross_entropy(logits_fused_text.T, labels)) / 2
        loss_it = (F.cross_entropy(logits_it, labels) + F.cross_entropy(logits_it.T, labels)) / 2
        loss_ip = (F.cross_entropy(logits_ip, labels) + F.cross_entropy(logits_ip.T, labels)) / 2
        loss_tp = (F.cross_entropy(logits_tp, labels) + F.cross_entropy(logits_tp.T, labels)) / 2

        loss = loss_fused_text * 3.0 + loss_it * 0.5 + loss_ip * 0.5 + loss_tp * 0.5

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for group in params for p in group['params']], max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = epoch_loss / len(train_loader)
    history.train_losses.append(avg_loss)
    
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
    
    # Validation
    r1 = validate(epoch)
    history.val_r1.append(r1)
    
    # Save best model
    if r1 > best_r1:
        best_r1 = r1
        best_dir = f"{args.output_logs}/models/best"
        os.makedirs(best_dir, exist_ok=True)
        
        vision_model_raw.save_pretrained(f"{best_dir}/vision_lora")
        text_model_raw.save_pretrained(f"{best_dir}/text_lora")
        tokenizer.save_pretrained(f"{best_dir}/text_lora")
        processor.save_pretrained(f"{best_dir}/vision_lora")
        torch.save(point_model_raw.state_dict(), f"{best_dir}/point_model.pth")
        torch.save(point_proj_raw.state_dict(), f"{best_dir}/point_proj.pth")
        torch.save(fusion_model_raw.state_dict(), f"{best_dir}/fusion_model.pth")
        
        print(f"✓ Best model saved! R@1: {best_r1:.4f}")
    
    scheduler.step()
    
    if (epoch + 1) % 5 == 0:
        history.save_plots(args.output_logs)

# ========== 13. Сохранение ==========
print("\n=== Saving final models ===")
final_dir = f"{args.output_logs}/models/final"
os.makedirs(final_dir, exist_ok=True)

vision_model_raw = unwrap_model(vision_model)
text_model_raw = unwrap_model(text_model)
point_model_raw = unwrap_model(point_model)
point_proj_raw = unwrap_model(point_proj)
fusion_model_raw = unwrap_model(fusion_model)

vision_model_raw.save_pretrained(f"{final_dir}/vision_lora")
text_model_raw.save_pretrained(f"{final_dir}/text_lora")
tokenizer.save_pretrained(f"{final_dir}/text_lora")
processor.save_pretrained(f"{final_dir}/vision_lora")
torch.save(point_model_raw.state_dict(), f"{final_dir}/point_model.pth")
torch.save(point_proj_raw.state_dict(), f"{final_dir}/point_proj.pth")
torch.save(fusion_model_raw.state_dict(), f"{final_dir}/fusion_model.pth")

history.save_plots(args.output_logs)

print(f"\n✅ Training completed! Best R@1: {best_r1:.4f}")
print(f"Results saved to: {args.output_logs}")
writer.close()