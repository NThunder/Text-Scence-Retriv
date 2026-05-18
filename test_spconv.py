import torch
import spconv.pytorch as spconv
import numpy as np

import sys
sys.path.insert(0, '/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/pointcept')
sys.path.insert(0, '/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net')
from pointcept.models.point_transformer_v2p.point_transformer_v6m5_random_shift import PointTransformerV2P, Point

# 1. Проверка базового импорта spconv
# print(f"✅ SpConv version: {spconv.__version__}")

# 2. Простое объявление модели (как в вашем train-скрипте)
model = PointTransformerV2P(
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
).cuda()

# 3. Создание "фейковых" данных, чтобы увидеть ожидаемый тип
# Генерируем 1000 случайных точек
fake_points = torch.randn(1000, 4).cuda()  # [N, 4] (xyz + 1 фича)
fake_offsets = torch.tensor([1000]).int().cuda()  # [B]

point_data = Point({
    'coord': fake_points[:, :3].contiguous(),
    'feat': fake_points[:, 3:].contiguous(),  # <- Вот здесь ключевой момент
    'offset': fake_offsets,
    'grid_size': 0.1,
    'condition': 'nuScenes',
})
point_data['batch'] = torch.zeros(1000, dtype=torch.long).cuda()
point_data.serialization(order=model.order)

# Смотрим, какой тип у признаков (feat)
print(f"Тип 'feat' внутри 'point_data': {point_data['feat'].dtype}")