# tools/encode_nuscenes_scene_captions.py
import json
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from nuscenes import NuScenes
from tqdm import tqdm

# Путь к данным
nuscenes_dataroot = "/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval"
caption_json_path = f"./final_caption_bbox_token.json"
output_path = f"./scene_text_embeddings_clip_vitb32.pth"

# Загрузка аннотаций
with open(caption_json_path, "r") as f:
    captions = json.load(f)

# Инициализация CLIP
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").eval().cuda()

# Инициализация NuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_dataroot, verbose=False)

scene_text_embs = {}

def build_final_caption(data):
    attrs = []
    if "attribute_caption" in data and isinstance(data["attribute_caption"], dict):
        attr = data["attribute_caption"].get("attribute_caption", "").strip()
        if attr and attr.lower() != "none":
            attrs.append(attr)
    
    loc = ""
    if "localization_caption" in data and isinstance(data["localization_caption"], dict):
        loc = data["localization_caption"].get("localization_caption", "").strip()
        if loc.lower() == "none": loc = ""
    
    depth = ""
    if "depth_caption" in data and isinstance(data["depth_caption"], dict):
        depth = data["depth_caption"].get("depth_caption", "").strip()
        if depth.lower() == "none": depth = ""
    
    motion = ""
    if "motion_caption" in data and isinstance(data["motion_caption"], dict):
        motion = data["motion_caption"].get("motion_caption", "").strip()
        if motion.lower() == "none": motion = ""
    
    map_desc = ""
    if "map_caption" in data and isinstance(data["map_caption"], dict):
        map_desc = data["map_caption"].get("map_caption", "").strip()
        if map_desc.lower() == "none": map_desc = ""
    
    relation = data.get("relation_caption", "none")
    if isinstance(relation, str):
        relation = relation.strip() if relation.lower() != "none" else ""
    
    # Собираем осмысленное предложение
    parts = []
    if attrs:
        obj = " and ".join(attrs)
        parts.append(f"{obj}")
    if loc:
        parts.append(f"is located {loc}")
    if depth:
        parts.append(f"at {depth}")
    if motion:
        parts.append(f"and is {motion}")
    if map_desc:
        parts.append(f"in {map_desc}")
    if relation:
        parts.append(relation)
    
    if parts:
        desc = "A " + " ".join(parts) + "."
        return desc
    else:
        return "An object with no description."

# Проход по всем сценам и сэмплам
for scene in tqdm(nusc.scene, desc="Processing scenes"):
    current_sample_token = scene["first_sample_token"]
    while current_sample_token != "":
        sample = nusc.get("sample", current_sample_token)
        ann_tokens = sample["anns"]
        
        scene_descriptions = []
        for ann_token in ann_tokens:
            if ann_token in captions:
                # Генерируем описание объекта
                obj_caption = build_final_caption(captions[ann_token])
                scene_descriptions.append(obj_caption)
        
        # Формируем описание всей сцены
        if scene_descriptions:
            scene_text = "The scene contains: " + "; ".join(scene_descriptions)
        else:
            scene_text = "An empty scene with no objects."

        # Кодируем текст
        inputs = tokenizer(
            scene_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to("cuda")
        
        with torch.no_grad():
            emb = text_encoder(**inputs).pooler_output.squeeze(0).cpu()
        
        scene_text_embs[current_sample_token] = emb
        current_sample_token = sample["next"]

# Сохранение
torch.save(scene_text_embs, output_path)
print(f"Saved scene text embeddings to {output_path}")