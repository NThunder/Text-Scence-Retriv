import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
import shutil
from llm2vec import LLM2Vec

import sys
sys.path.insert(0, '/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/pointcept')
sys.path.insert(0, '/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net')
sys.path.insert(0, '/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/libs')

from pointcept.models.point_transformer_v2p.point_transformer_v6m5_random_shift import PointTransformerV2P, Point

# ========== Конфигурация ==========
def parse_args():
    parser = argparse.ArgumentParser(description="Joint training with Query-based Fusion")
    parser.add_argument("--dataroot", type=str, default="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval")
    parser.add_argument("--camera_captions_path", type=str, default="./camera_aware_captions_motion_map.json")
    parser.add_argument("--output_lora_text", type=str, default="./lora_text_encoder_fused")
    parser.add_argument("--output_lora_image", type=str, default="./lora_image_encoder_fused")
    parser.add_argument("--output_point_model", type=str, default="./point_encoder_fused")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--max_points", type=int, default=20000)
    parser.add_argument("--fusion_dim", type=int, default=768, help="Dimension for fused embedding")
    parser.add_argument("--num_queries", type=int, default=4, help="Number of learnable queries in fusion")
    parser.add_argument("--max_train_samples", type=int, default=-1,
        help="Max training samples (-1 = all)")
    parser.add_argument("--max_val_samples", type=int, default=-1,
        help="Max validation samples (-1 = all)")
    parser.add_argument("--val_samples_per_scene", type=int, default=-1, help="Number of evenly spaced samples per validation scene (-1 = all)")
    parser.add_argument("--no_prefix", action="store_true", help="Omit 'The camera view contains: ' prefix")
    return parser.parse_args()

args = parse_args()

NUSCENES_DATAROOT = args.dataroot
CAMERA_CAPTIONS_PATH = args.camera_captions_path
OUTPUT_LORA_TEXT = args.output_lora_text
OUTPUT_LORA_IMAGE = args.output_lora_image
OUTPUT_POINT_MODEL = args.output_point_model
BATCH_SIZE = args.batch_size
LR = args.lr
EPOCHS = args.epochs
TEMPERATURE = args.temperature
LORA_R = args.lora_r
MAX_POINTS = args.max_points
FUSION_DIM = args.fusion_dim
NUM_QUERIES = args.num_queries

# ========== 1. Загрузка описаний ==========
print("Loading camera-aware captions...")
with open(CAMERA_CAPTIONS_PATH, "r") as f:
    camera_captions = json.load(f)

# ========== 2. Подготовка nuScenes (750 train samples) ==========
nusc = NuScenes(version='v1.0-trainval', dataroot=NUSCENES_DATAROOT, verbose=False)
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

# ========== 4. Загрузка моделей с LoRA ==========
print("\n=== Loading models ===")

# ---- Замороженная LLM2CLIP для проекций ----
clip_model = AutoModel.from_pretrained(
    "microsoft/LLM2CLIP-Openai-L-14-336",
    trust_remote_code=True,
    torch_dtype=torch.float16
).cuda()
clip_model.eval()
for param in clip_model.parameters():
    param.requires_grad = False

# ---- Визуальный энкодер (LoRA) ----
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
vision_model = AutoModel.from_pretrained(
    "microsoft/LLM2CLIP-Openai-L-14-336",
    trust_remote_code=True,
    torch_dtype=torch.float16
).cuda()
for param in vision_model.parameters():
    param.requires_grad = False
lora_config_vision = LoraConfig(r=LORA_R, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                                lora_dropout=0.1, bias="none")
vision_model = get_peft_model(vision_model, lora_config_vision).cuda()
vision_model.print_trainable_parameters()

# ---- Текстовый энкодер (LLM2Vec + LoRA) ----
llm_model_name = "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=False)
text_model = AutoModel.from_pretrained(llm_model_name, trust_remote_code=False, torch_dtype=torch.float16)
for param in text_model.parameters():
    param.requires_grad = False
lora_config_text = LoraConfig(r=LORA_R, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                              lora_dropout=0.1, bias="none")
text_model = get_peft_model(text_model, lora_config_text).cuda()
text_model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
l2v = LLM2Vec(text_model, tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512)
text_model.print_trainable_parameters()

# ---- PointTransformerV2P ----
print("\nLoading PointTransformerV2P...")
point_model = PointTransformerV2P(
    in_channels=1,
    order=["z", "z-trans", "hilbert", "hilbert-trans"],
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(32, 64, 128, 256, 512),
    enc_num_head=(2, 4, 8, 16, 32),
    enc_patch_size=(256, 256, 256, 256, 256),
    dec_depths=(2, 2, 2, 2),
    dec_channels=(64, 64, 128, 256),
    dec_num_head=(4, 4, 8, 16),
    dec_patch_size=(256, 256, 256, 256),
    mlp_ratio=4, qkv_bias=True, enable_rpe=False, enable_flash=False,
    upcast_attention=True, upcast_softmax=True, cls_mode=True,
    pdnorm_bn=False, pdnorm_ln=False, pdnorm_decouple=False,
    pdnorm_adaptive=False, pdnorm_affine=False,
    pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
).cuda()
point_model.train()
print("Point model trainable parameters:", sum(p.numel() for p in point_model.parameters() if p.requires_grad))

# ========== 5. Проекционные слои к единой размерности ==========
with torch.no_grad():
    dummy_img = torch.randn(1, 3, 336, 336).cuda()
    image_feat_dim = vision_model.get_image_features(dummy_img).shape[-1]
    print(f"Image feature dimension: {image_feat_dim}")
    dummy_txt = l2v.encode(["test"], convert_to_tensor=True).to('cuda').half()
    text_feat_dim = clip_model.get_text_features(dummy_txt).shape[-1]
    print(f"Text feature dimension: {text_feat_dim}")
    point_feat_dim = 512   # из point_model (cls_mode=True)

img_proj = nn.Linear(image_feat_dim, FUSION_DIM).cuda()
txt_proj = nn.Linear(text_feat_dim, FUSION_DIM).cuda()
point_proj = nn.Linear(point_feat_dim, FUSION_DIM).cuda()

# ========== 6. Query‑based Fusion ==========
class QueryFusion(nn.Module):
    def __init__(self, dim=FUSION_DIM, num_queries=NUM_QUERIES, num_heads=8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim))
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.fusion_proj = nn.Linear(dim * num_queries, dim)

    def forward(self, img_emb, pt_emb):
        # img_emb, pt_emb: [B, dim]
        B = img_emb.shape[0]
        img_seq = img_emb.unsqueeze(1)       # [B, 1, dim]
        pt_seq = pt_emb.unsqueeze(1)         # [B, 1, dim]
        multi_modal = torch.cat([img_seq, pt_seq], dim=1)  # [B, 2, dim]

        queries = self.queries.unsqueeze(0).expand(B, -1, -1)   # [B, num_queries, dim]
        fused, _ = self.cross_attn(queries, multi_modal, multi_modal)  # [B, num_queries, dim]
        fused = fused.reshape(B, -1)          # [B, num_queries * dim]
        fused = self.fusion_proj(fused)       # [B, dim]
        return fused

fusion_model = QueryFusion(dim=FUSION_DIM, num_queries=NUM_QUERIES).cuda()

# ========== 7. Датасет (без изменений) ==========
class JointNuScenesDataset(Dataset):
    def __init__(self, nusc, pairs, processor, max_points=MAX_POINTS, cache_lidar=True):
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
        return image_tensor, text, points_tensor, sample_token

# ========== 8. Коллатор ==========
def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    texts = [item[1] for item in batch]
    points_list = [item[2] for item in batch]
    sample_tokens = [item[3] for item in batch]

    all_points = []
    offsets = []
    for pts in points_list:
        all_points.append(pts)
        offsets.append(pts.shape[0])
    all_points = torch.cat(all_points, dim=0)
    offsets = torch.tensor(offsets).cumsum(dim=0).int()
    return images, texts, all_points, offsets, sample_tokens

# ========== 9. Датасеты и загрузчики ==========
train_dataset = JointNuScenesDataset(nusc, train_pairs, processor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, collate_fn=collate_fn)

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
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True, collate_fn=collate_fn)

# ========== 10. Оптимизатор ==========
params = (list(vision_model.parameters()) + list(text_model.parameters()) +
          list(point_model.parameters()) + list(img_proj.parameters()) +
          list(txt_proj.parameters()) + list(point_proj.parameters()) +
          list(fusion_model.parameters()))
optimizer = torch.optim.AdamW(params, lr=LR)

# ========== 11. Функция валидации (с сохранением эмбеддингов) ==========
def offset2batch(offset):
    batch = torch.zeros(offset[-1].item(), dtype=torch.long, device=offset.device)
    for i in range(1, len(offset)):
        batch[offset[i-1]:offset[i]] = i
    return batch

def validate(vision_model, text_model, point_model, clip_model,
            img_proj, txt_proj, point_proj, fusion_model,
            val_loader, save_embeddings=False, epoch=0):
    # vision_model.train()
    # text_model.eval()
    # point_model.eval()
    # img_proj.eval()
    # txt_proj.eval()
    # point_proj.eval()
    # fusion_model.eval()
    # l2v.model.eval()

    all_img_embs, all_txt_embs, all_pt_embs, all_fused_embs = [], [], [], []
    if save_embeddings:
        camera_fused_embs = {}

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        for images, texts, all_points, offsets, sample_tokens in tqdm(val_loader, desc="Validation", leave=False):
            images = images.cuda()
            all_points = all_points.cuda()
            offsets = offsets.cuda()
            B = images.shape[0]

            # ---- Image ----
            img_feat = vision_model.get_image_features(images)
            img_emb = F.normalize(img_proj(img_feat), p=2, dim=-1)

            # ---- Text ----
            txt_raw = l2v.encode(texts, convert_to_tensor=True).to('cuda')
            txt_clip = clip_model.get_text_features(txt_raw)
            txt_emb = F.normalize(txt_proj(txt_clip), p=2, dim=-1)

            # ---- Point cloud ----
            data_dict = Point({
                'coord': all_points[:, :3].contiguous(),
                'feat': all_points[:, 3:].contiguous(),
                'offset': offsets,
                'grid_size': 0.1,
                'condition': 'nuScenes'
            })
            data_dict['batch'] = offset2batch(offsets)
            pt_out = point_model(data_dict)
            pt_feat = pt_out[0] if isinstance(pt_out, tuple) else pt_out
            pt_emb = F.normalize(point_proj(pt_feat), p=2, dim=-1)

            # ---- Fused embedding ----
            fused_emb = fusion_model(img_emb, pt_emb)
            fused_emb = F.normalize(fused_emb, p=2, dim=-1)

            all_img_embs.append(img_emb.cpu())
            all_txt_embs.append(txt_emb.cpu())
            all_pt_embs.append(pt_emb.cpu())
            all_fused_embs.append(fused_emb.cpu())

            if save_embeddings:
                for i, token in enumerate(sample_tokens):
                    # найти камеру по тексту
                    cam = None
                    for c in CAMERAS:
                        if c in texts[i]:
                            cam = c
                            break
                    if cam is None:
                        cam = CAMERAS[i % len(CAMERAS)]
                    if token not in camera_fused_embs:
                        camera_fused_embs[token] = {}
                    camera_fused_embs[token][cam] = fused_emb[i].cpu()

    all_img_embs = torch.cat(all_img_embs)
    all_txt_embs = torch.cat(all_txt_embs)
    all_pt_embs = torch.cat(all_pt_embs)
    all_fused_embs = torch.cat(all_fused_embs)

    # Метрики для fused ↔ text
    sim_fused_text = all_fused_embs @ all_txt_embs.T
    sim_text_fused = sim_fused_text.T
    def r1(sim):
        return (sim.argmax(dim=1) == torch.arange(len(sim))).float().mean().item()
    fused2text_r1 = r1(sim_fused_text)
    text2fused_r1 = r1(sim_text_fused)

    if save_embeddings:
        os.makedirs(f"./embeddings_epoch_{epoch}", exist_ok=True)
        torch.save(camera_fused_embs, f"./embeddings_epoch_{epoch}/fused_embeddings.pth")
        print(f"✓ Saved fused embeddings to ./embeddings_epoch_{epoch}/")

    # Возвращаем режимы тренировки
    vision_model.train()
    text_model.train()
    point_model.train()
    img_proj.train()
    txt_proj.train()
    point_proj.train()
    fusion_model.train()
    l2v.model.train()

    return fused2text_r1, text2fused_r1

# ========== 12. Цикл обучения с Query Fusion ==========
print(f"\n=== Starting training with Query Fusion for {EPOCHS} epochs ===")
best_fused2text = 0.0
best_epoch = 0

for epoch in range(EPOCHS):
    vision_model.train()
    text_model.train()
    point_model.train()
    img_proj.train()
    txt_proj.train()
    point_proj.train()
    fusion_model.train()
    total_loss = 0.0

    for batch_idx, (images, texts, all_points, offsets, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        images = images.cuda()
        all_points = all_points.cuda()
        offsets = offsets.cuda()
        B = images.shape[0]

        # ---- Point data ----
        data_dict = Point({
            'coord': all_points[:, :3].contiguous(),
            'feat': all_points[:, 3:].contiguous(),
            'offset': offsets,
            'grid_size': 0.1,
        })
        data_dict['batch'] = offset2batch(offsets)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(dtype=torch.float16):
            # Image
            img_feat = vision_model.get_image_features(images)
            img_emb = F.normalize(img_proj(img_feat), p=2, dim=-1)

            # Text
            txt_raw = l2v.encode(texts, convert_to_tensor=True).to('cuda')
            txt_clip = clip_model.get_text_features(txt_raw)
            txt_emb = F.normalize(txt_proj(txt_clip), p=2, dim=-1)

            # Point
            pt_out = point_model(data_dict)
            pt_feat = pt_out[0] if isinstance(pt_out, tuple) else pt_out
            pt_emb = F.normalize(point_proj(pt_feat), p=2, dim=-1)

            # Fused
            fused_emb = fusion_model(img_emb, pt_emb)
            fused_emb = F.normalize(fused_emb, p=2, dim=-1)

            labels = torch.arange(B).cuda()

            # Contrastive losses
            logits_fused_text = fused_emb @ txt_emb.T / TEMPERATURE
            logits_fused_img = fused_emb @ img_emb.T / TEMPERATURE
            logits_fused_pt  = fused_emb @ pt_emb.T / TEMPERATURE

            loss_fused_text = (F.cross_entropy(logits_fused_text, labels) +
                               F.cross_entropy(logits_fused_text.T, labels)) / 2
            loss_fused_img  = (F.cross_entropy(logits_fused_img, labels) +
                               F.cross_entropy(logits_fused_img.T, labels)) / 2
            loss_fused_pt   = (F.cross_entropy(logits_fused_pt, labels) +
                               F.cross_entropy(logits_fused_pt.T, labels)) / 2

            # Дополнительные парные лоссы (регуляризация)
            logits_it = img_emb @ txt_emb.T / TEMPERATURE
            logits_ip = img_emb @ pt_emb.T / TEMPERATURE
            logits_tp = txt_emb @ pt_emb.T / TEMPERATURE

            loss_it = (F.cross_entropy(logits_it, labels) + F.cross_entropy(logits_it.T, labels)) / 2
            loss_ip = (F.cross_entropy(logits_ip, labels) + F.cross_entropy(logits_ip.T, labels)) / 2
            loss_tp = (F.cross_entropy(logits_tp, labels) + F.cross_entropy(logits_tp.T, labels)) / 2

            # Итоговый лосс (можно подобрать веса)
            loss = (loss_fused_text * 3.0 +
                    loss_fused_img * 1.0 +
                    loss_fused_pt * 1.0 +
                    loss_it * 0.5 +
                    loss_ip * 0.5 +
                    loss_tp * 0.5)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

    if epoch % 5 == 0:
        # Валидация на fused → text
        fused2text_r1, text2fused_r1 = validate(
            vision_model, text_model, point_model, clip_model,
            img_proj, txt_proj, point_proj, fusion_model,
            val_loader, save_embeddings=True, epoch=epoch
        )
        print(f"Validation: Fused→Text R@1 = {fused2text_r1:.2%} | Text→Fused R@1 = {text2fused_r1:.2%}")

        # Сохраняем лучшую модель по fused→text
        if fused2text_r1 > best_fused2text:
            best_fused2text = fused2text_r1
            best_epoch = epoch + 1
            shutil.copytree(f"./embeddings_epoch_{epoch}", "./best_fused_embeddings", dirs_exist_ok=True)

            # Сохраняем веса
            vision_model.save_pretrained(OUTPUT_LORA_IMAGE + "_best")
            text_model.save_pretrained(OUTPUT_LORA_TEXT + "_best")
            torch.save(point_model.state_dict(), os.path.join(OUTPUT_POINT_MODEL + "_best", 'point_model.pth'))
            torch.save(point_proj.state_dict(), os.path.join(OUTPUT_POINT_MODEL + "_best", 'point_proj.pth'))
            torch.save(img_proj.state_dict(), os.path.join(OUTPUT_POINT_MODEL + "_best", 'img_proj.pth'))
            torch.save(txt_proj.state_dict(), os.path.join(OUTPUT_POINT_MODEL + "_best", 'txt_proj.pth'))
            torch.save(fusion_model.state_dict(), os.path.join(OUTPUT_POINT_MODEL + "_best", 'fusion_model.pth'))
            print(f"✓ Best model saved (fused→text R@1 = {best_fused2text:.2%})")

# ========== 13. Сохранение финальных моделей ==========
print("\n=== Saving final models ===")
os.makedirs(OUTPUT_POINT_MODEL, exist_ok=True)
vision_model.save_pretrained(OUTPUT_LORA_IMAGE)
text_model.save_pretrained(OUTPUT_LORA_TEXT)
tokenizer.save_pretrained(OUTPUT_LORA_TEXT)
processor.save_pretrained(OUTPUT_LORA_IMAGE)
torch.save(point_model.state_dict(), os.path.join(OUTPUT_POINT_MODEL, 'point_model.pth'))
torch.save(point_proj.state_dict(), os.path.join(OUTPUT_POINT_MODEL, 'point_proj.pth'))
torch.save(img_proj.state_dict(), os.path.join(OUTPUT_POINT_MODEL, 'img_proj.pth'))
torch.save(txt_proj.state_dict(), os.path.join(OUTPUT_POINT_MODEL, 'txt_proj.pth'))
torch.save(fusion_model.state_dict(), os.path.join(OUTPUT_POINT_MODEL, 'fusion_model.pth'))

print(f"\n✅ Training completed! Best Fused→Text R@1: {best_fused2text:.2%} at epoch {best_epoch}")