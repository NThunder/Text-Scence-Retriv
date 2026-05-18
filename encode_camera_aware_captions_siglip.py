# tools/encode_camera_aware_captions_siglip.py
import json
import torch
from transformers import AutoTokenizer, AutoModel
from nuscenes import NuScenes
from tqdm import tqdm

nuscenes_dataroot = "./"
camera_captions_path = f"{nuscenes_dataroot}/camera_aware_captions_short.json"
output_path = f"{nuscenes_dataroot}/camera_text_embeddings_siglip.pth"

with open(camera_captions_path, "r") as f:
    camera_captions = json.load(f)

model_name = "google/siglip-base-patch16-224"
tokenizer = AutoTokenizer.from_pretrained(model_name)
text_encoder = AutoModel.from_pretrained(model_name).eval().cuda()

nusc = NuScenes(
    version='v1.0-trainval',
    dataroot="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval",
    verbose=False
)

camera_text_embs = {}

for scene in tqdm(nusc.scene):
    current_sample_token = scene["first_sample_token"]
    while current_sample_token != "":
        if current_sample_token not in camera_captions:
            current_sample_token = nusc.get("sample", current_sample_token)["next"]
            continue

        cam_dict = camera_captions[current_sample_token]
        embs_dict = {}

        for cam_name, captions in cam_dict.items():
            if not captions:
                continue
            scene_text = "The camera view contains: " + "; ".join(captions)

            inputs = tokenizer(
                scene_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64  # SigLIP использует 64 токена по умолчанию
            ).to("cuda")

            with torch.no_grad():
                # SigLIP возвращает `text_embeds` через `get_text_features`
                # Но AutoModel не имеет этого метода → используем напрямую
                outputs = text_encoder.text_model(**inputs)
                # Pooler: берём [CLS] токен (первый)
                emb = outputs.pooler_output  # [1, 768]
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
                embs_dict[cam_name] = emb.cpu().squeeze(0)

        camera_text_embs[current_sample_token] = embs_dict
        current_sample_token = nusc.get("sample", current_sample_token)["next"]

torch.save(camera_text_embs, output_path)
print(f"Saved to {output_path}")