# retriv/encode_nuscenes_pointclouds_simple.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.insert(0, '/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/pointcept')
sys.path.insert(0, '/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net')

import torch
import torch.nn as nn
import torch.nn.functional as F
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
import numpy as np
from tqdm import tqdm
import argparse
import copy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval")
    parser.add_argument("--point_model_path", type=str, default="./point_encoder_joint_all/point_model.pth")
    parser.add_argument("--point_proj_path", type=str, default="./point_encoder_joint_all/point_proj.pth")
    parser.add_argument("--output_path", type=str, default="./camera_point_embeddings.pth")
    parser.add_argument("--max_points", type=int, default=10000)
    return parser.parse_args()

args = parse_args()

class SimplePointEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=512):
        super().__init__()
        # Простой MLP для глобального признака облака точек
        self.encoder = nn.Sequential(
            nn.Linear(4, 64),  # x,y,z,intensity
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, out_channels),
        )
        
    def forward(self, points):
        # points: list of [N_i, 4] tensors
        batch_features = []
        for p in points:
            if p.shape[0] == 0:
                batch_features.append(torch.zeros(512, device=p.device))
                continue
            # Применяем MLP к каждой точке
            feat = self.encoder(p)  # [N, 256]
            # Глобальный пулинг
            global_feat = feat.mean(dim=0)  # [256]
            # Финальный MLP
            out = self.fc(global_feat)  # [512]
            batch_features.append(out)
        return torch.stack(batch_features)

def project_points_to_camera(nusc, pc, sample_token, camera_name, max_points=10000, min_dist=1.0):
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data'][camera_name]
    cam_data = nusc.get('sample_data', cam_token)
    pointsensor = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

    pc = copy.deepcopy(pc)

    cs_record_lidar = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record_lidar['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record_lidar['translation']))

    poserecord_lidar = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord_lidar['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord_lidar['translation']))

    poserecord_cam = nusc.get('ego_pose', cam_data['ego_pose_token'])
    pc.translate(-np.array(poserecord_cam['translation']))
    pc.rotate(Quaternion(poserecord_cam['rotation']).rotation_matrix.T)

    cs_record_cam = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
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
        return np.zeros((0, 4))

    points_cam = pc.points[:3, valid_indices].T
    intensity = pc.points[3, valid_indices].reshape(-1, 1)
    points_cam_with_intensity = np.concatenate([points_cam, intensity], axis=1)

    if max_points is not None and len(points_cam_with_intensity) > max_points:
        idx = np.random.choice(len(points_cam_with_intensity), max_points, replace=False)
        points_cam_with_intensity = points_cam_with_intensity[idx]

    return points_cam_with_intensity

print("Loading models...")
point_model = SimplePointEncoder(in_channels=1, out_channels=512)
point_model.cuda().eval()

# Пытаемся загрузить веса (если есть)
try:
    state_dict = torch.load(args.point_model_path, map_location='cpu')
    # Фильтруем только совместимые слои
    model_state = point_model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    point_model.load_state_dict(filtered, strict=False)
    print(f"Loaded {len(filtered)} layers from {args.point_model_path}")
except Exception as e:
    print(f"Could not load weights: {e}, using random init")

point_proj = nn.Linear(512, 1280)
try:
    point_proj.load_state_dict(torch.load(args.point_proj_path, map_location='cpu'))
    print(f"Loaded projection from {args.point_proj_path}")
except Exception as e:
    print(f"Could not load projection: {e}, using random init")

point_proj.cuda().eval()

# Подготовка nuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=False)
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

# Кэширование LiDAR
lidar_cache = {}
print("Loading LiDAR data...")
for sample_token in tqdm(val_sample_tokens):
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    lidar_path = os.path.join(args.dataroot, lidar_data['filename'])
    pc = LidarPointCloud.from_file(lidar_path)
    lidar_cache[sample_token] = pc

# Генерация эмбеддингов
camera_point_embs = {}
print("Encoding point clouds...")

for sample_token in tqdm(val_sample_tokens):
    pc = lidar_cache[sample_token]
    cam_embs = {}
    
    for cam in CAMERAS:
        points = project_points_to_camera(nusc, pc, sample_token, cam, args.max_points)
        
        if len(points) == 0:
            cam_embs[cam] = torch.zeros(1280)
            continue
        
        points_tensor = torch.from_numpy(points).float().cuda()
        
        # Нормализация интенсивности
        if points_tensor.shape[1] == 4:
            intensity = points_tensor[:, 3]
            intensity_min, intensity_max = intensity.min(), intensity.max()
            if intensity_max > intensity_min:
                points_tensor[:, 3] = (intensity - intensity_min) / (intensity_max - intensity_min)
        
        with torch.no_grad():
            point_feat = point_model([points_tensor])  # [1, 512]
            point_emb = point_proj(point_feat)  # [1, 1280]
            point_emb = F.normalize(point_emb, p=2, dim=-1)
        
        cam_embs[cam] = point_emb.cpu().squeeze(0)
    
    camera_point_embs[sample_token] = cam_embs

torch.save(camera_point_embs, args.output_path)
print(f"✓ Saved to {args.output_path}")