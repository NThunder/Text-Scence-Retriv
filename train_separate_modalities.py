import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
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
    parser = argparse.ArgumentParser(description="Separate training for text, image and point cloud encoders (no fusion)")
    parser.add_argument("--dataroot", type=str, default="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval")
    parser.add_argument("--camera_captions_path", type=str, default="./camera_aware_captions_motion_map.json")
    parser.add_argument("--output_dir", type=str, default="./separate_training_output")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--max_points", type=int, default=20000)
    parser.add_argument("--point_weight", type=float, default=1.0, help="Weight for point cloud losses")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--max_train_samples", type=int, default=-1,
        help="Max training samples (-1 = all)")
    parser.add_argument("--max_val_samples", type=int, default=-1,
        help="Max validation samples (-1 = all)")
    parser.add_argument("--val_samples_per_scene", type=int, default=-1, help="Number of evenly spaced samples per validation scene (-1 = all)")
    parser.add_argument("--no_prefix", action="store_true", help="Omit 'The camera view contains: ' prefix")
    return parser.parse_args()

args = parse_args()

# Создаём директории
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(f"{args.output_dir}/plots", exist_ok=True)
os.makedirs(f"{args.output_dir}/models", exist_ok=True)

# TensorBoard writer
writer = SummaryWriter(f"{args.output_dir}/logs")

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
        while current != "":
            train_sample_tokens.append(current)
            current = nusc.get("sample", current)["next"]

if args.max_train_samples > 0:
    train_sample_tokens = train_sample_tokens[:args.max_train_samples]
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
            if args.no_prefix:
                scene_text = " ".join(camera_captions[sample_token][cam])
            else:
                scene_text = "The camera view contains: " + "; ".join(camera_captions[sample_token][cam])
            train_pairs.append((sample_token, cam, scene_text))
print(f"Total training pairs: {len(train_pairs)}")

# ========== 4. Загрузка моделей ==========
print("\n=== Loading models ===")

# ---- Point encoder: Utonia ----
print("Loading Utonia (PointTransformerV3)...")
point_model = utonia.load("utonia", repo_id="Pointcept/Utonia").cuda()
point_model.train()
print("Utonia loaded. Number of parameters:", sum(p.numel() for p in point_model.parameters()))

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
print("Vision LoRA trainable params:")
vision_model.print_trainable_parameters()

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
print("Text LoRA trainable params:")
text_model.print_trainable_parameters()

# Проекционные слои (приводим к единой размерности для контрастивного обучения)
point_feat_dim = 576  # Utonia output dimension
EMBED_DIM = 1280  # Общая размерность для всех эмбеддингов

# img_proj = nn.Linear(1280, EMBED_DIM).cuda()
# txt_proj = nn.Linear(1280, EMBED_DIM).cuda()
point_proj = nn.Linear(point_feat_dim, EMBED_DIM).cuda()

print(f"\nAll embeddings projected to {EMBED_DIM} dimensions")

# ========== 5. Transform для Utonia ==========
utonia_transform = utonia.transform.default(scale=1.0, apply_z_positive=False, normalize_coord=False)

# ========== 6. Датасет ==========
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

# ========== 7. Валидационный датасет ==========
def prepare_val_dataset(nusc, camera_captions, processor, num_samples=150, val_samples_per_scene=-1):
    val_scenes = set(create_splits_scenes()["val"])
    val_sample_tokens = []
    for scene in nusc.scene:
        if scene["name"] in val_scenes:
            scene_tokens = []
            current = scene["first_sample_token"]
            while current != "":
                scene_tokens.append(current)
                current = nusc.get("sample", current)["next"]
            if val_samples_per_scene > 0:
                step = max(1, len(scene_tokens) // val_samples_per_scene)
                scene_tokens = scene_tokens[::step][:val_samples_per_scene]
            val_sample_tokens.extend(scene_tokens)

    if num_samples > 0:
        val_sample_tokens = val_sample_tokens[:num_samples]

    val_pairs = []
    for sample_token in val_sample_tokens:
        if sample_token not in camera_captions:
            continue
        for cam in CAMERAS:
            if cam in camera_captions[sample_token] and camera_captions[sample_token][cam]:
                if args.no_prefix:
                    scene_text = " ".join(camera_captions[sample_token][cam])
                else:
                    scene_text = "The camera view contains: " + "; ".join(camera_captions[sample_token][cam])
                val_pairs.append((sample_token, cam, scene_text))
    return JointNuScenesDataset(nusc, val_pairs, processor), len(val_pairs)

val_dataset, num_val_pairs = prepare_val_dataset(nusc, camera_captions, processor, num_samples=args.max_val_samples, val_samples_per_scene=args.val_samples_per_scene)
print(f"Prepared validation dataset with {num_val_pairs} pairs")
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True, collate_fn=collate_fn)

# ========== 8. Оптимизатор ==========
params = [
    {'params': vision_model.parameters(), 'lr': args.lr},
    {'params': text_model.parameters(), 'lr': args.lr},
    {'params': point_model.parameters(), 'lr': args.lr / 10},
    # {'params': img_proj.parameters(), 'lr': args.lr},
    # {'params': txt_proj.parameters(), 'lr': args.lr},
    {'params': point_proj.parameters(), 'lr': args.lr},
]
optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

# ========== 9. Функции для сохранения истории ==========
class TrainingHistory:
    def __init__(self):
        self.train_losses = []
        self.train_losses_it = []
        self.train_losses_ip = []
        self.train_losses_tp = []
        
        self.val_i2t_r1 = []
        self.val_t2i_r1 = []
        self.val_i2p_r1 = []
        self.val_p2i_r1 = []
        self.val_t2p_r1 = []
        self.val_p2t_r1 = []
        
        self.val_i2t_r5 = []
        self.val_i2p_r5 = []
        self.val_t2p_r5 = []
    
    def save_plots(self, output_dir):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot train losses
        axes[0, 0].plot(self.train_losses, label='Total Loss', color='blue')
        axes[0, 0].set_title('Total Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot individual losses
        axes[0, 1].plot(self.train_losses_it, label='Image-Text', alpha=0.7)
        axes[0, 1].plot(self.train_losses_ip, label='Image-Point', alpha=0.7)
        axes[0, 1].plot(self.train_losses_tp, label='Text-Point', alpha=0.7)
        axes[0, 1].set_title('Individual Losses')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot R@1 metrics for all pairs
        axes[1, 0].plot(self.val_i2t_r1, label='Image→Text R@1', marker='o')
        axes[1, 0].plot(self.val_t2i_r1, label='Text→Image R@1', marker='s')
        axes[1, 0].plot(self.val_i2p_r1, label='Image→Point R@1', marker='^')
        axes[1, 0].plot(self.val_p2i_r1, label='Point→Image R@1', marker='v')
        axes[1, 0].plot(self.val_t2p_r1, label='Text→Point R@1', marker='d')
        axes[1, 0].plot(self.val_p2t_r1, label='Point→Text R@1', marker='p')
        axes[1, 0].set_title('R@1 Metrics')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R@1')
        axes[1, 0].legend(loc='lower right', fontsize=8)
        axes[1, 0].grid(True)
        
        # Plot R@5 metrics
        axes[1, 1].plot(self.val_i2t_r5, label='Image→Text R@5', marker='o')
        axes[1, 1].plot(self.val_i2p_r5, label='Image→Point R@5', marker='^')
        axes[1, 1].plot(self.val_t2p_r5, label='Text→Point R@5', marker='d')
        axes[1, 1].set_title('R@5 Metrics')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('R@5')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/training_history.png", dpi=150)
        plt.close()
        
        # Save history as JSON
        history_dict = {
            'train_losses': self.train_losses,
            'train_losses_it': self.train_losses_it,
            'train_losses_ip': self.train_losses_ip,
            'train_losses_tp': self.train_losses_tp,
            'val_i2t_r1': self.val_i2t_r1,
            'val_t2i_r1': self.val_t2i_r1,
            'val_i2p_r1': self.val_i2p_r1,
            'val_p2i_r1': self.val_p2i_r1,
            'val_t2p_r1': self.val_t2p_r1,
            'val_p2t_r1': self.val_p2t_r1,
        }
        with open(f"{output_dir}/plots/history.json", 'w') as f:
            json.dump(history_dict, f, indent=2)

history = TrainingHistory()

# ========== 10. Функция валидации (попарное сравнение) ==========
def offset2batch(offset):
    batch = torch.zeros(offset[-1].item(), dtype=torch.long, device=offset.device)
    for i in range(1, len(offset)):
        batch[offset[i-1]:offset[i]] = i
    return batch

def compute_metrics(sim_matrix):
    """Compute R@1, R@5, R@10, MRR from similarity matrix"""
    N = len(sim_matrix)
    ranks = []
    for i in range(N):
        sorted_indices = torch.argsort(sim_matrix[i], descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    
    ranks = np.array(ranks)
    return {
        'R@1': np.mean(ranks <= 1),
        'R@5': np.mean(ranks <= 5),
        'R@10': np.mean(ranks <= 10),
        'MRR': np.mean(1.0 / ranks),
        'Median': np.median(ranks)
    }

def validate(epoch):
    vision_model.eval()
    text_model.eval()
    point_model.eval()
    # img_proj.eval()
    # txt_proj.eval()
    point_proj.eval()
    l2v.model.eval()

    all_img_embs = []
    all_txt_embs = []
    all_pt_embs = []

    with torch.no_grad():
        for images, texts, all_points, offsets, sample_tokens, cameras in tqdm(val_loader, desc="Validation", leave=False):
            images = images.cuda()
            B = images.shape[0]

            # Image embeddings
            img_feat = vision_model.get_image_features(images)
            img_emb = F.normalize((img_feat), p=2, dim=-1)
            all_img_embs.append(img_emb.cpu())

            # Text embeddings
            txt_raw = l2v.encode(texts, convert_to_tensor=True).to('cuda')
            txt_clip = clip_model.get_text_features(txt_raw)
            txt_emb = F.normalize((txt_clip), p=2, dim=-1)
            all_txt_embs.append(txt_emb.cpu())

            # Point embeddings
            pt_embs = []
            for i in range(B):
                start = offsets[i-1].item() if i > 0 else 0
                end = offsets[i].item()
                pts = all_points[start:end].cuda()
                if pts.shape[0] == 0:
                    pt_embs.append(torch.zeros(EMBED_DIM, device='cuda'))
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
            all_pt_embs.append(pt_emb.cpu())

    # Конкатенируем все эмбеддинги
    all_img_embs = torch.cat(all_img_embs)
    all_txt_embs = torch.cat(all_txt_embs)
    all_pt_embs = torch.cat(all_pt_embs)
    
    # Вычисляем все попарные метрики
    metrics = {}
    
    # Image ↔ Text
    sim_it = all_img_embs @ all_txt_embs.T
    metrics['i2t'] = compute_metrics(sim_it)
    metrics['t2i'] = compute_metrics(sim_it.T)
    
    # Image ↔ Point
    sim_ip = all_img_embs @ all_pt_embs.T
    metrics['i2p'] = compute_metrics(sim_ip)
    metrics['p2i'] = compute_metrics(sim_ip.T)
    
    # Text ↔ Point
    sim_tp = all_txt_embs @ all_pt_embs.T
    metrics['t2p'] = compute_metrics(sim_tp)
    metrics['p2t'] = compute_metrics(sim_tp.T)
    
    # Выводим результаты
    print(f"\n{'='*70}")
    print(f"Validation Results (Epoch {epoch+1})")
    print(f"{'='*70}")
    print(f"Image → Text:  R@1={metrics['i2t']['R@1']:.4f}, R@5={metrics['i2t']['R@5']:.4f}, R@10={metrics['i2t']['R@10']:.4f}")
    print(f"Text → Image:  R@1={metrics['t2i']['R@1']:.4f}, R@5={metrics['t2i']['R@5']:.4f}, R@10={metrics['t2i']['R@10']:.4f}")
    print(f"Image → Point: R@1={metrics['i2p']['R@1']:.4f}, R@5={metrics['i2p']['R@5']:.4f}, R@10={metrics['i2p']['R@10']:.4f}")
    print(f"Point → Image: R@1={metrics['p2i']['R@1']:.4f}, R@5={metrics['p2i']['R@5']:.4f}, R@10={metrics['p2i']['R@10']:.4f}")
    print(f"Text → Point:  R@1={metrics['t2p']['R@1']:.4f}, R@5={metrics['t2p']['R@5']:.4f}, R@10={metrics['t2p']['R@10']:.4f}")
    print(f"Point → Text:  R@1={metrics['p2t']['R@1']:.4f}, R@5={metrics['p2t']['R@5']:.4f}, R@10={metrics['p2t']['R@10']:.4f}")
    print(f"{'='*70}\n")
    
    # Log to TensorBoard
    writer.add_scalar('Val/Image2Text_R1', metrics['i2t']['R@1'], epoch)
    writer.add_scalar('Val/Text2Image_R1', metrics['t2i']['R@1'], epoch)
    writer.add_scalar('Val/Image2Point_R1', metrics['i2p']['R@1'], epoch)
    writer.add_scalar('Val/Point2Image_R1', metrics['p2i']['R@1'], epoch)
    writer.add_scalar('Val/Text2Point_R1', metrics['t2p']['R@1'], epoch)
    writer.add_scalar('Val/Point2Text_R1', metrics['p2t']['R@1'], epoch)
    
    vision_model.train()
    text_model.train()
    point_model.train()
    # img_proj.train()
    # txt_proj.train()
    point_proj.train()
    l2v.model.train()
    
    return metrics

# ========== 11. Цикл обучения ==========
print(f"\n=== Starting separate training for {args.epochs} epochs ===")
best_i2t_r1 = 0.0

for epoch in range(args.epochs):
    vision_model.train()
    text_model.train()
    point_model.train()
    # img_proj.train()
    # txt_proj.train()
    point_proj.train()
    
    epoch_loss = 0.0
    epoch_loss_it = 0.0
    epoch_loss_ip = 0.0
    epoch_loss_tp = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
    for batch_idx, (images, texts, all_points, offsets, sample_tokens, cameras) in enumerate(progress_bar):
        images = images.cuda()
        all_points = all_points.cuda()
        offsets = offsets.cuda()
        B = images.shape[0]

        # ---- Image embedding ----
        img_feat = vision_model.get_image_features(images)
        img_emb = F.normalize((img_feat), p=2, dim=-1)

        # ---- Text embedding ----
        txt_raw = l2v.encode(texts, convert_to_tensor=True).to('cuda')
        txt_clip = clip_model.get_text_features(txt_raw)
        txt_emb = F.normalize((txt_clip), p=2, dim=-1)

        # ---- Point embedding ----
        # ---- Point embedding (batch обработка по одному образцу) ----
        pt_embs = []
        for i in range(B):
            start = offsets[i-1].item() if i > 0 else 0
            end = offsets[i].item()
            pts = all_points[start:end]
            if pts.shape[0] == 0:
                pt_embs.append(torch.zeros(FUSION_DIM, device='cuda'))
                continue
            coord = pts[:, :3]
            color = torch.zeros_like(coord)
            normal = torch.zeros_like(coord)
            point_dict = {
                "coord": coord.cpu().numpy(),  # конвертируем в numpy
                "color": color.cpu().numpy(),
                "normal": normal.cpu().numpy(),
            }
            point_dict = utonia_transform(point_dict)  # теперь работает с numpy
            for k in point_dict.keys():
                if isinstance(point_dict[k], np.ndarray):
                    point_dict[k] = torch.from_numpy(point_dict[k]).cuda(non_blocking=True)
            for k in point_dict:
                if isinstance(point_dict[k], np.ndarray):
                    point_dict[k] = torch.from_numpy(point_dict[k])
                if isinstance(point_dict[k], torch.Tensor):
                    point_dict[k] = point_dict[k].cuda()
            out = point_model(point_dict)
            feat = out['feat'].mean(dim=0)
            pt_emb = F.normalize(point_proj(feat), p=2, dim=-1)
            pt_embs.append(pt_emb)
        pt_emb = torch.stack(pt_embs)

        labels = torch.arange(B).cuda()
        
        # Контрастивные лоссы для всех трёх пар
        logits_it = img_emb @ txt_emb.T / args.temperature
        logits_ip = img_emb @ pt_emb.T / args.temperature
        logits_tp = txt_emb @ pt_emb.T / args.temperature
        
        loss_it = (F.cross_entropy(logits_it, labels) + F.cross_entropy(logits_it.T, labels)) / 2
        loss_ip = (F.cross_entropy(logits_ip, labels) + F.cross_entropy(logits_ip.T, labels)) / 2
        loss_tp = (F.cross_entropy(logits_tp, labels) + F.cross_entropy(logits_tp.T, labels)) / 2

        # Суммарный лосс с весом для point cloud
        loss = loss_it + loss_ip * args.point_weight + loss_tp * args.point_weight

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for group in params for p in group['params']], max_norm=1.0)
        optimizer.step()

        # Accumulate losses
        epoch_loss += loss.item()
        epoch_loss_it += loss_it.item()
        epoch_loss_ip += loss_ip.item()
        epoch_loss_tp += loss_tp.item()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    # Average losses
    num_batches = len(train_loader)
    avg_loss = epoch_loss / num_batches
    avg_loss_it = epoch_loss_it / num_batches
    avg_loss_ip = epoch_loss_ip / num_batches
    avg_loss_tp = epoch_loss_tp / num_batches
    
    # Store history
    history.train_losses.append(avg_loss)
    history.train_losses_it.append(avg_loss_it)
    history.train_losses_ip.append(avg_loss_ip)
    history.train_losses_tp.append(avg_loss_tp)
    
    # Log to TensorBoard
    writer.add_scalar('Train/Total_Loss', avg_loss, epoch)
    writer.add_scalar('Train/Image_Text_Loss', avg_loss_it, epoch)
    writer.add_scalar('Train/Image_Point_Loss', avg_loss_ip, epoch)
    writer.add_scalar('Train/Text_Point_Loss', avg_loss_tp, epoch)
    
    print(f"\nEpoch {epoch+1} | Loss: {avg_loss:.4f} (IT:{avg_loss_it:.4f}, IP:{avg_loss_ip:.4f}, TP:{avg_loss_tp:.4f})")
    
    # Validation
    metrics = validate(epoch)
    
    # Store validation metrics
    history.val_i2t_r1.append(metrics['i2t']['R@1'])
    history.val_t2i_r1.append(metrics['t2i']['R@1'])
    history.val_i2p_r1.append(metrics['i2p']['R@1'])
    history.val_p2i_r1.append(metrics['p2i']['R@1'])
    history.val_t2p_r1.append(metrics['t2p']['R@1'])
    history.val_p2t_r1.append(metrics['p2t']['R@1'])
    history.val_i2t_r5.append(metrics['i2t']['R@5'])
    history.val_i2p_r5.append(metrics['i2p']['R@5'])
    history.val_t2p_r5.append(metrics['t2p']['R@5'])
    
    # Save best model (by Image→Text R@1)
    if metrics['i2t']['R@1'] > best_i2t_r1:
        best_i2t_r1 = metrics['i2t']['R@1']
        best_dir = f"{args.output_dir}/models/best"
        os.makedirs(best_dir, exist_ok=True)
        
        vision_model.save_pretrained(f"{best_dir}/vision_lora")
        text_model.save_pretrained(f"{best_dir}/text_lora")
        tokenizer.save_pretrained(f"{best_dir}/text_lora")
        processor.save_pretrained(f"{best_dir}/vision_lora")
        torch.save(point_model.state_dict(), f"{best_dir}/point_model.pth")
        # torch.save(img_proj.state_dict(), f"{best_dir}/img_proj.pth")
        # torch.save(txt_proj.state_dict(), f"{best_dir}/txt_proj.pth")
        torch.save(point_proj.state_dict(), f"{best_dir}/point_proj.pth")
        
        print(f"✓ Best model saved (Image→Text R@1: {best_i2t_r1:.4f})")
    
    # Save plots every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
        history.save_plots(args.output_dir)
        print(f"✓ Plots saved to {args.output_dir}/plots/")

# ========== 12. Сохранение финальных моделей ==========
print("\n=== Saving final models ===")
final_dir = f"{args.output_dir}/models/final"
os.makedirs(final_dir, exist_ok=True)

vision_model.save_pretrained(f"{final_dir}/vision_lora")
text_model.save_pretrained(f"{final_dir}/text_lora")
tokenizer.save_pretrained(f"{final_dir}/text_lora")
processor.save_pretrained(f"{final_dir}/vision_lora")
torch.save(point_model.state_dict(), f"{final_dir}/point_model.pth")
# torch.save(img_proj.state_dict(), f"{final_dir}/img_proj.pth")
# torch.save(txt_proj.state_dict(), f"{final_dir}/txt_proj.pth")
torch.save(point_proj.state_dict(), f"{final_dir}/point_proj.pth")

# Final plots
history.save_plots(args.output_dir)

print(f"\n✅ Training completed!")
print(f"Best Image→Text R@1: {best_i2t_r1:.4f}")
print(f"Results saved to {args.output_dir}")

# Close TensorBoard writer
writer.close()