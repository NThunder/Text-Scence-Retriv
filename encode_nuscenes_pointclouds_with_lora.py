import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.insert(0, '/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/pointcept')
sys.path.insert(0, '/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net')

import torch
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
from pointcept.models.point_transformer_v2p.point_transformer_v6m5_random_shift import PointTransformerV2P, Point

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval")
    parser.add_argument("--point_model_path", type=str, default="./point_encoder_joint_all/point_model.pth")
    parser.add_argument("--point_proj_path", type=str, default="./point_encoder_joint_all/point_proj.pth")
    parser.add_argument("--output_path", type=str, default="./camera_point_embeddings_fixed.pth")
    parser.add_argument("--max_points", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_val_samples", type=int, default=-1, help="Max validation samples (-1 = all)")
    parser.add_argument("--val_samples_per_scene", type=int, default=-1, help="Number of evenly spaced samples per validation scene (-1 = all)")
    return parser.parse_args()

args = parse_args()

def offset2batch(offset):
    batch = torch.zeros(offset[-1].item(), dtype=torch.long, device=offset.device)
    for i in range(1, len(offset)):
        batch[offset[i-1]:offset[i]] = i
    return batch

def project_points_to_camera(nusc, pc, sample_token, camera_name, max_points=20000, min_dist=1.0):
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

print("Loading PointTransformerV2P...")
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
    mlp_ratio=4,
    qkv_bias=True,
    enable_rpe=False,
    enable_flash=False,
    upcast_attention=True,
    upcast_softmax=True,
    cls_mode=True,
    pdnorm_bn=False,
    pdnorm_ln=False,
    pdnorm_decouple=False,
    pdnorm_adaptive=False,
    pdnorm_affine=False,
    pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
)

# Загрузка весов
state_dict = torch.load(args.point_model_path, map_location='cpu')
point_model.load_state_dict(state_dict)
point_model.cuda().train()
print("✓ Point model loaded")

point_proj = torch.nn.Linear(512, 1280)
point_proj.load_state_dict(torch.load(args.point_proj_path, map_location='cpu'))
point_proj.cuda().train()
print("✓ Projection loaded")

# Подготовка nuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=False)
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

# Собираем все пары для обработки
all_pairs = []
for sample_token in val_sample_tokens:
    pc = lidar_cache[sample_token]
    for cam in CAMERAS:
        points = project_points_to_camera(nusc, pc, sample_token, cam, args.max_points)
        if len(points) > 0:
            all_pairs.append((sample_token, cam, points))

print(f"Processing {len(all_pairs)} point clouds in batches of {args.batch_size}...")

camera_point_embs = {}

# Обработка батчами
for i in tqdm(range(0, len(all_pairs), args.batch_size)):
    batch_pairs = all_pairs[i:i+args.batch_size]
    
    # Собираем точки в батч
    batch_points = []
    batch_offsets = []
    batch_info = []
    
    for sample_token, cam, points in batch_pairs:
        points_tensor = torch.from_numpy(points).float()
        batch_points.append(points_tensor)
        batch_offsets.append(points_tensor.shape[0])
        batch_info.append((sample_token, cam))
    
    # Объединяем все точки
    all_points = torch.cat(batch_points, dim=0).cuda()
    offsets = torch.tensor(batch_offsets).cumsum(dim=0).int().cuda()
    
    # Подготовка данных (как в обучении)
    coords = all_points[:, :3].float().contiguous()
    intensity = all_points[:, 3:4].float().contiguous()
    
    # Нормализация интенсивности
    intensity_min, intensity_max = intensity.min(), intensity.max()
    if intensity_max > intensity_min:
        intensity = (intensity - intensity_min) / (intensity_max - intensity_min)
    
    # Создаем Point объект (как в тесте)
    data_dict = Point({
        'coord': coords,
        'feat': intensity,  # float32
        'offset': offsets,
        'grid_size': 0.1,
        'condition': 'nuScenes',
    })
    data_dict['batch'] = offset2batch(offsets)
    
    # Сериализация и sparsify
    data_dict.serialization(order=point_model.order)
    data_dict.sparsify(pad=96)
    
    # Forward pass
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        point_output = point_model(data_dict)
        point_feat = point_output[0] if isinstance(point_output, tuple) else point_output
        point_emb = point_proj(point_feat)
        point_emb = F.normalize(point_emb, p=2, dim=-1)
    
    # Сохраняем результаты
    for j, (sample_token, cam) in enumerate(batch_info):
        if sample_token not in camera_point_embs:
            camera_point_embs[sample_token] = {}
        camera_point_embs[sample_token][cam] = point_emb[j].cpu()

torch.save(camera_point_embs, args.output_path)
print(f"✓ Saved to {args.output_path}")