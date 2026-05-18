import os
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, logging
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

# Подавляем warnings transformers
logging.set_verbosity_error()

# Импорт Utonia
import utonia
from utonia.model import PointTransformerV3

# ========== Конфигурация ==========
def parse_args():
    parser = argparse.ArgumentParser(description="Joint training with Utonia (PTv3)")
    parser.add_argument("--dataroot", type=str, default="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval")
    parser.add_argument("--camera_captions_path", type=str, default="./camera_aware_captions_motion_map.json")
    parser.add_argument("--output_logs", type=str, default="./training_logs_utonia", help="Directory for logs and plots")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--max_points", type=int, default=20000)
    parser.add_argument("--fusion_dim", type=int, default=1280, help="Dimension for fused embedding")
    parser.add_argument("--fusion_type", type=str, default="query", choices=["query", "weighted", "moe"], 
                        help="Fusion type: query (QueryAttention), weighted (Weighted sum), moe (Mixture of Experts)")
    parser.add_argument("--num_queries", type=int, default=4, help="Number of learnable queries (for query fusion)")
    parser.add_argument("--num_experts", type=int, default=4, help="Number of experts (for MoE fusion)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--log_interval", type=int, default=10)
    return parser.parse_args()

args = parse_args()

# Создаём директории
os.makedirs(args.output_logs, exist_ok=True)
os.makedirs(f"{args.output_logs}/plots", exist_ok=True)
os.makedirs(f"{args.output_logs}/models", exist_ok=True)
os.makedirs(f"{args.output_logs}/embeddings", exist_ok=True)
os.makedirs(f"{args.output_logs}/checkpoints", exist_ok=True)

# TensorBoard writer
writer = SummaryWriter(f"{args.output_logs}/tensorboard")

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
print("Loading Utonia (PointTransformerV3)...")
from utonia import load

point_model = load("utonia", repo_id="Pointcept/Utonia").cuda()
point_model.train()
for param in point_model.parameters():
    param.requires_grad = True
print(f"Utonia loaded. Parameters: {sum(p.numel() for p in point_model.parameters()):,}")

# ---- CLIP projection (заморожена) ----
clip_model = AutoModel.from_pretrained(
    "microsoft/LLM2CLIP-Openai-L-14-336",
    trust_remote_code=True,
    torch_dtype=torch.float32
).cuda()
clip_model.eval()
for param in clip_model.parameters():
    param.requires_grad = False

# ---- Vision encoder (CLIP + LoRA) ----
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
vision_model = AutoModel.from_pretrained(
    "microsoft/LLM2CLIP-Openai-L-14-336",
    trust_remote_code=True,
    torch_dtype=torch.float32
).cuda()
for param in vision_model.parameters():
    param.requires_grad = False
lora_config_vision = LoraConfig(r=args.lora_r, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                                lora_dropout=0.1, bias="none")
vision_model = get_peft_model(vision_model, lora_config_vision).cuda()

# ---- Text encoder (LLM2Vec + LoRA) ----
llm_model_name = "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=False)
text_model = AutoModel.from_pretrained(llm_model_name, trust_remote_code=False, torch_dtype=torch.float32)
for param in text_model.parameters():
    param.requires_grad = False
lora_config_text = LoraConfig(r=args.lora_r, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                              lora_dropout=0.1, bias="none")
text_model = get_peft_model(text_model, lora_config_text).cuda()
text_model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
l2v = LLM2Vec(text_model, tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512)

# Проекционные слои
point_feat_dim = 576
point_proj = nn.Linear(point_feat_dim, args.fusion_dim).cuda()

# ========== 5. Fusion модули ==========
class WeightedFusion(nn.Module):
    """Взвешенная сумма"""
    def __init__(self, dim):
        super().__init__()
        self.weight_img = nn.Parameter(torch.ones(1) * 0.5)
        self.weight_pt = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, img_emb, pt_emb):
        w_img = torch.sigmoid(self.weight_img)
        w_pt = torch.sigmoid(self.weight_pt)
        total = w_img + w_pt
        fused = (w_img / total) * img_emb + (w_pt / total) * pt_emb
        return fused

class MoEFusion(nn.Module):
    """Mixture of Experts"""
    def __init__(self, dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(dim * 2, num_experts)
    
    def forward(self, img_emb, pt_emb):
        concat = torch.cat([img_emb, pt_emb], dim=-1)
        gate_weights = F.softmax(self.gate(concat), dim=-1)
        fused = torch.zeros_like(img_emb)
        for i, expert in enumerate(self.experts):
            fused += gate_weights[:, i:i+1] * expert(concat)
        return fused

class QueryFusion(nn.Module):
    """Query-based Attention Fusion"""
    def __init__(self, dim, num_queries=4, num_heads=8):
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

# Выбираем fusion модуль
print(f"\nUsing fusion type: {args.fusion_type}")
if args.fusion_type == "weighted":
    fusion_model = WeightedFusion(args.fusion_dim).cuda()
elif args.fusion_type == "moe":
    fusion_model = MoEFusion(args.fusion_dim, args.num_experts).cuda()
else:
    fusion_model = QueryFusion(args.fusion_dim, args.num_queries).cuda()

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
                          num_workers=4, pin_memory=True, collate_fn=collate_fn)

# ========== 8. Подготовка валидации ==========
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
                        num_workers=2, pin_memory=True, collate_fn=collate_fn)

# ========== 9. Класс для хранения истории ==========
class TrainingHistory:
    def __init__(self):
        self.train_losses = []
        self.train_losses_fused_text = []
        self.val_r1 = []
        self.val_r5 = []
        self.val_r10 = []
        self.val_mrr = []
        self.val_losses = []
        self.best_r1 = 0.0
        self.best_epoch = 0
        self.start_epoch = 0
    
    def save_plots(self, output_dir):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Training loss
        axes[0, 0].plot(self.train_losses, label='Total Loss', color='blue', linewidth=2)
        axes[0, 0].set_title('Total Training Loss')
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
        
        # R@1 metrics
        axes[1, 0].plot(self.val_r1, label='Fused→Text R@1', marker='o', linewidth=2)
        axes[1, 0].set_title('R@1 Metric')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R@1')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
        
        # R@5 and MRR
        axes[1, 1].plot(self.val_r5, label='Fused→Text R@5', marker='o')
        axes[1, 1].plot(self.val_mrr, label='Fused→Text MRR', marker='s')
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
    
    def save_checkpoint(self, output_dir, epoch, model, optimizer, scheduler):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_r1': self.val_r1,
                'val_r5': self.val_r5,
                'val_r10': self.val_r10,
                'val_mrr': self.val_mrr,
                'best_r1': self.best_r1,
                'best_epoch': self.best_epoch
            }
        }
        torch.save(checkpoint, f"{output_dir}/checkpoints/checkpoint_epoch_{epoch}.pth")
    
    def load_checkpoint(self, checkpoint_path, model, optimizer, scheduler):
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.train_losses = checkpoint['history']['train_losses']
        self.val_losses = checkpoint['history']['val_losses']
        self.val_r1 = checkpoint['history']['val_r1']
        self.val_r5 = checkpoint['history']['val_r5']
        self.val_r10 = checkpoint['history']['val_r10']
        self.val_mrr = checkpoint['history']['val_mrr']
        self.best_r1 = checkpoint['history']['best_r1']
        self.best_epoch = checkpoint['history']['best_epoch']
        print(f"Resumed from epoch {self.start_epoch}, best R@1: {self.best_r1:.4f}")
        return self.start_epoch

history = TrainingHistory()

# ========== 10. Оптимизатор ==========
params = [
    {'params': vision_model.parameters(), 'lr': args.lr},
    {'params': text_model.parameters(), 'lr': args.lr},
    {'params': point_model.parameters(), 'lr': args.lr / 10},
    {'params': point_proj.parameters(), 'lr': args.lr},
    {'params': fusion_model.parameters(), 'lr': args.lr},
]
optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# ========== 11. Функция валидации ==========
def offset2batch(offset):
    batch = torch.zeros(offset[-1].item(), dtype=torch.long, device=offset.device)
    for i in range(1, len(offset)):
        batch[offset[i-1]:offset[i]] = i
    return batch

def validate(vision_model, text_model, point_model, clip_model, point_proj, fusion_model,
             val_loader, utonia_transform, sample_to_next, epoch=0, save_embeddings=False):
    vision_model.eval()
    text_model.eval()
    point_model.eval()
    point_proj.eval()
    fusion_model.eval()
    l2v.model.eval()

    all_fused_embs = []
    all_txt_embs = []
    all_sample_tokens = []
    all_cameras = []
    val_loss = 0.0
    
    # Для сохранения эмбеддингов
    if save_embeddings:
        fused_embeddings_dict = {}
        text_embeddings_dict = {}

    with torch.no_grad():
        for images, texts, all_points, offsets, sample_tokens, cameras in tqdm(val_loader, desc="Validation", leave=False):
            images = images.cuda()
            all_points = all_points.cuda()
            offsets = offsets.cuda()
            B = images.shape[0]

            img_feat = vision_model.get_image_features(images)
            img_emb = F.normalize(img_feat, p=2, dim=-1)

            txt_raw = l2v.encode(texts, convert_to_tensor=True).to('cuda')
            txt_clip = clip_model.get_text_features(txt_raw)
            txt_emb = F.normalize(txt_clip, p=2, dim=-1)
            all_txt_embs.append(txt_emb.cpu())
            all_sample_tokens.extend(sample_tokens)
            all_cameras.extend(cameras)

            pt_embs = []
            for i in range(B):
                start = offsets[i-1].item() if i > 0 else 0
                end = offsets[i].item()
                pts = all_points[start:end].cuda()
                if pts.shape[0] == 0:
                    pt_embs.append(torch.zeros(args.fusion_dim, device='cuda'))
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
                        point_dict[k] = torch.from_numpy(point_dict[k]).cuda()
                    elif isinstance(point_dict[k], torch.Tensor):
                        point_dict[k] = point_dict[k].cuda()
                
                out = point_model(point_dict)
                feat = out['feat'].mean(dim=0)
                pt_emb = F.normalize(point_proj(feat), p=2, dim=-1)
                pt_embs.append(pt_emb)
            pt_emb = torch.stack(pt_embs)

            fused_emb = fusion_model(img_emb, pt_emb)
            fused_emb = F.normalize(fused_emb, p=2, dim=-1)
            all_fused_embs.append(fused_emb.cpu())
            
            # Сохраняем эмбеддинги для этой эпохи
            if save_embeddings:
                for j, (token, cam) in enumerate(zip(sample_tokens, cameras)):
                    if token not in fused_embeddings_dict:
                        fused_embeddings_dict[token] = {}
                        text_embeddings_dict[token] = {}
                    fused_embeddings_dict[token][cam] = fused_emb[j].cpu()
                    text_embeddings_dict[token][cam] = txt_emb[j].cpu()
            
            # Validation loss
            labels = torch.arange(B).cuda()
            logits = fused_emb @ txt_emb.T / args.temperature
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            val_loss += loss.item()

    val_loss /= len(val_loader)
    
    # Сохраняем эмбеддинги на диск
    if save_embeddings:
        embeddings_dir = f"{args.output_logs}/embeddings/epoch_{epoch+1}"
        os.makedirs(embeddings_dir, exist_ok=True)
        torch.save(fused_embeddings_dict, f"{embeddings_dir}/fused_embeddings.pth")
        torch.save(text_embeddings_dict, f"{embeddings_dir}/text_embeddings.pth")
        print(f"✓ Saved embeddings to {embeddings_dir}")
    
    all_fused_embs = torch.cat(all_fused_embs)
    all_txt_embs = torch.cat(all_txt_embs)
    
    # Create local index
    local_index = {}
    for idx, (token, cam) in enumerate(zip(all_sample_tokens, all_cameras)):
        local_index[(token, cam)] = idx
    
    # Attributes for relevance
    attr_to_indices = defaultdict(set)
    for idx, (token, cam) in enumerate(zip(all_sample_tokens, all_cameras)):
        if token in camera_captions and cam in camera_captions[token]:
            for attr in camera_captions[token][cam]:
                attr_to_indices[attr].add(idx)
    
    similarity = all_fused_embs @ all_txt_embs.T
    ranks = []
    
    for i in range(len(all_fused_embs)):
        sample_token, cam = all_sample_tokens[i], all_cameras[i]
        
        relevant = set()
        if sample_token in camera_captions and cam in camera_captions[sample_token]:
            for attr in camera_captions[sample_token][cam]:
                relevant.update(attr_to_indices[attr])
        
        current = sample_token
        for _ in range(10):
            current = sample_to_next.get(current, "")
            if not current:
                break
            if (current, cam) in local_index:
                relevant.add(local_index[(current, cam)])
        
        sorted_indices = torch.argsort(similarity[i], descending=True)
        found = False
        for rank_pos, idx in enumerate(sorted_indices.tolist()):
            if idx in relevant:
                ranks.append(rank_pos + 1)
                found = True
                break
        if not found:
            ranks.append(len(all_fused_embs))
    
    ranks = np.array(ranks)
    r1 = np.mean(ranks <= 1)
    r5 = np.mean(ranks <= 5)
    r10 = np.mean(ranks <= 10)
    mrr = np.mean(1.0 / ranks)
    
    print(f"\n{'='*70}")
    print(f"Validation Results (Epoch {epoch+1})")
    print(f"{'='*70}")
    print(f"Fused → Text:")
    print(f"  R@1:  {r1:.4f}")
    print(f"  R@5:  {r5:.4f}")
    print(f"  R@10: {r10:.4f}")
    print(f"  MRR:  {mrr:.4f}")
    print(f"  Loss: {val_loss:.4f}")
    print(f"{'='*70}\n")
    
    writer.add_scalar('Val/Fused2Text_R1', r1, epoch)
    writer.add_scalar('Val/Fused2Text_R5', r5, epoch)
    writer.add_scalar('Val/Fused2Text_R10', r10, epoch)
    writer.add_scalar('Val/Fused2Text_MRR', mrr, epoch)
    writer.add_scalar('Val/Loss', val_loss, epoch)
    
    vision_model.train()
    text_model.train()
    point_model.train()
    point_proj.train()
    fusion_model.train()
    l2v.model.train()
    
    return r1, r5, r10, mrr, val_loss

# ========== 12. Создание словаря для временной близости ==========
sample_to_next = {}
for scene in nusc.scene:
    current = scene["first_sample_token"]
    while current != "":
        sample_record = nusc.get("sample", current)
        sample_to_next[current] = sample_record["next"]
        current = sample_record["next"]

# ========== 13. Загрузка чекпоинта (если указан) ==========
start_epoch = 0
if args.resume:
    print(f"\nResuming from checkpoint: {args.resume}")
    start_epoch = history.load_checkpoint(args.resume, fusion_model, optimizer, scheduler)
    # Также загружаем состояние для других моделей
    checkpoint = torch.load(args.resume, map_location='cuda')
    vision_model.load_state_dict(checkpoint['vision_model_state_dict']) if 'vision_model_state_dict' in checkpoint else None
    text_model.load_state_dict(checkpoint['text_model_state_dict']) if 'text_model_state_dict' in checkpoint else None
    point_model.load_state_dict(checkpoint['point_model_state_dict']) if 'point_model_state_dict' in checkpoint else None
    point_proj.load_state_dict(checkpoint['point_proj_state_dict']) if 'point_proj_state_dict' in checkpoint else None

# ========== 14. Цикл обучения ==========
print(f"\n=== Starting training for {args.epochs} epochs ===")

for epoch in range(start_epoch, args.epochs):
    vision_model.train()
    text_model.train()
    point_model.train()
    point_proj.train()
    fusion_model.train()
    
    epoch_loss = 0.0
    epoch_loss_fused_text = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
    for batch_idx, (images, texts, all_points, offsets, sample_tokens, cameras) in enumerate(progress_bar):
        images = images.cuda()
        all_points = all_points.cuda()
        offsets = offsets.cuda()
        B = images.shape[0]

        img_feat = vision_model.get_image_features(images)
        img_emb = F.normalize(img_feat, p=2, dim=-1)

        txt_raw = l2v.encode(texts, convert_to_tensor=True).to('cuda')
        txt_clip = clip_model.get_text_features(txt_raw)
        txt_emb = F.normalize(txt_clip, p=2, dim=-1)

        pt_embs = []
        for i in range(B):
            start = offsets[i-1].item() if i > 0 else 0
            end = offsets[i].item()
            pts = all_points[start:end]
            if pts.shape[0] == 0:
                pt_embs.append(torch.zeros(args.fusion_dim, device='cuda'))
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
                    point_dict[k] = torch.from_numpy(point_dict[k]).cuda()
                elif isinstance(point_dict[k], torch.Tensor):
                    point_dict[k] = point_dict[k].cuda()
            out = point_model(point_dict)
            feat = out['feat'].mean(dim=0)
            pt_emb = F.normalize(point_proj(feat), p=2, dim=-1)
            pt_embs.append(pt_emb)
        pt_emb = torch.stack(pt_embs)

        fused_emb = fusion_model(img_emb, pt_emb)
        fused_emb = F.normalize(fused_emb, p=2, dim=-1)

        labels = torch.arange(B).cuda()
        
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
        epoch_loss_fused_text += loss_fused_text.item()
        
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = epoch_loss / len(train_loader)
    avg_loss_fused_text = epoch_loss_fused_text / len(train_loader)
    
    history.train_losses.append(avg_loss)
    history.train_losses_fused_text.append(avg_loss_fused_text)
    
    writer.add_scalar('Train/Total_Loss', avg_loss, epoch)
    writer.add_scalar('Train/Fused_Text_Loss', avg_loss_fused_text, epoch)
    
    print(f"\nEpoch {epoch+1} | Total Loss: {avg_loss:.4f} | Fused-Text Loss: {avg_loss_fused_text:.4f}")
    
    # Validation
    r1, r5, r10, mrr, val_loss = validate(
        vision_model, text_model, point_model, clip_model, point_proj, fusion_model,
        val_loader, utonia_transform, sample_to_next, epoch=epoch,
        save_embeddings=True  # Добавьте этот параметр
    )
    
    history.val_r1.append(r1)
    history.val_r5.append(r5)
    history.val_r10.append(r10)
    history.val_mrr.append(mrr)
    history.val_losses.append(val_loss)
    
    writer.add_scalar('Val/R1', r1, epoch)
    writer.add_scalar('Val/R5', r5, epoch)
    writer.add_scalar('Val/R10', r10, epoch)
    writer.add_scalar('Val/MRR', mrr, epoch)
    writer.add_scalar('Val/Loss', val_loss, epoch)
    
    if r1 > history.best_r1:
        history.best_r1 = r1
        history.best_epoch = epoch + 1
        
        best_dir = f"{args.output_logs}/models/best"
        os.makedirs(best_dir, exist_ok=True)
        
        vision_model.save_pretrained(f"{best_dir}/vision_lora")
        text_model.save_pretrained(f"{best_dir}/text_lora")
        tokenizer.save_pretrained(f"{best_dir}/text_lora")
        processor.save_pretrained(f"{best_dir}/vision_lora")
        torch.save(point_model.state_dict(), f"{best_dir}/point_model.pth")
        torch.save(point_proj.state_dict(), f"{best_dir}/point_proj.pth")
        torch.save(fusion_model.state_dict(), f"{best_dir}/fusion_model.pth")
        
        print(f"✓ Best model saved! Fused→Text R@1: {history.best_r1:.4f} (epoch {history.best_epoch})")
    
    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint = {
            'epoch': epoch,
            'vision_model_state_dict': vision_model.state_dict(),
            'text_model_state_dict': text_model.state_dict(),
            'point_model_state_dict': point_model.state_dict(),
            'point_proj_state_dict': point_proj.state_dict(),
            'fusion_model_state_dict': fusion_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': {
                'train_losses': history.train_losses,
                'val_losses': history.val_losses,
                'val_r1': history.val_r1,
                'val_r5': history.val_r5,
                'val_r10': history.val_r10,
                'val_mrr': history.val_mrr,
                'best_r1': history.best_r1,
                'best_epoch': history.best_epoch
            }
        }
        torch.save(checkpoint, f"{args.output_logs}/checkpoints/checkpoint_epoch_{epoch}.pth")
        print(f"✓ Checkpoint saved to {args.output_logs}/checkpoints/")
    
    scheduler.step()
    
    if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
        history.save_plots(args.output_logs)
        print(f"✓ Plots saved to {args.output_logs}/plots/")

# ========== 15. Сохранение финальных моделей ==========
print("\n=== Saving final models ===")
final_dir = f"{args.output_logs}/models/final"
os.makedirs(final_dir, exist_ok=True)

vision_model.save_pretrained(f"{final_dir}/vision_lora")
text_model.save_pretrained(f"{final_dir}/text_lora")
tokenizer.save_pretrained(f"{final_dir}/text_lora")
processor.save_pretrained(f"{final_dir}/vision_lora")
torch.save(point_model.state_dict(), f"{final_dir}/point_model.pth")
torch.save(point_proj.state_dict(), f"{final_dir}/point_proj.pth")
torch.save(fusion_model.state_dict(), f"{final_dir}/fusion_model.pth")

history.save_plots(args.output_logs)

final_results = {
    'best_r1': history.best_r1,
    'best_epoch': history.best_epoch,
    'final_r1': history.val_r1[-1] if history.val_r1 else 0,
    'final_r5': history.val_r5[-1] if history.val_r5 else 0,
    'final_r10': history.val_r10[-1] if history.val_r10 else 0,
    'final_mrr': history.val_mrr[-1] if history.val_mrr else 0,
    'fusion_type': args.fusion_type
}

with open(f"{args.output_logs}/final_results.json", 'w') as f:
    json.dump(final_results, f, indent=2)

print(f"\n{'='*60}")
print(f"✅ Training completed!")
print(f"{'='*60}")
print(f"Best Fused→Text R@1: {history.best_r1:.4f} (epoch {history.best_epoch})")
print(f"Final R@1: {final_results['final_r1']:.4f}")
print(f"Final R@5: {final_results['final_r5']:.4f}")
print(f"Final R@10: {final_results['final_r10']:.4f}")
print(f"Final MRR: {final_results['final_mrr']:.4f}")
print(f"Fusion type: {args.fusion_type}")
print(f"{'='*60}")
print(f"Results saved to: {args.output_logs}")
print(f"  - Plots: {args.output_logs}/plots/")
print(f"  - Models: {args.output_logs}/models/")
print(f"  - Checkpoints: {args.output_logs}/checkpoints/")
print(f"  - TensorBoard: tensorboard --logdir {args.output_logs}/tensorboard")
print(f"{'='*60}")

writer.close()