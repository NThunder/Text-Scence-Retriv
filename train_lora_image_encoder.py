# tools/train_lora_image_encoder.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor
from peft import LoraConfig, get_peft_model
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from tqdm import tqdm
import json

import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from tqdm import tqdm

import argparse
from retrieval_metrics import attribute_retrieval_metrics
from sampling import sample_scenes_uniformly

# === Конфигурация ===
def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA image encoder")
    parser.add_argument("--dataroot", type=str, default="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval", help="Path to nuScenes dataset")
    parser.add_argument("--text_emb_path", type=str, default="./camera_text_embeddings_llm2clip_openai_l14_336_train750_motion_map.pth", help="Path to text embeddings")
    parser.add_argument("--val_text_emb_path", type=str, default=None, help="Path to validation text embeddings (optional)")
    parser.add_argument("--output_lora_dir", type=str, default="./lora_image_encoder_nuscenes", help="Path to output LoRA directory")
    parser.add_argument("--validate_every", type=int, default=1, help="Run validation every N epochs")
    parser.add_argument("--max_train_samples", type=int, default=-1,
        help="Max training samples (-1 = all)")
    parser.add_argument("--max_val_samples", type=int, default=-1,
        help="Max validation samples (-1 = all)")
    parser.add_argument("--val_samples_per_scene", type=int, default=-1,
        help="Number of evenly spaced samples per validation scene (-1 = all)")
    parser.add_argument("--sample_seed", type=int, default=0,
        help="Seed for the scene-level training subsample")
    parser.add_argument("--camera_captions_path", type=str, default=None,
        help="Caption JSON for attribute-based validation metrics. Without it "
             "the checkpoint falls back to validation loss, which is not the "
             "metric the paper reports.")
    return parser.parse_args()

args = parse_args()

NUSCENES_DATAROOT = args.dataroot
TEXT_EMB_PATH = args.text_emb_path  # ← вы должны сначала создать этот файл
VAL_TEXT_EMB_PATH = args.val_text_emb_path
CAMERA_CAPTIONS = None
if args.camera_captions_path:
    with open(args.camera_captions_path, "r") as _f:
        CAMERA_CAPTIONS = json.load(_f)
OUTPUT_LORA_DIR = args.output_lora_dir
VALIDATE_EVERY = args.validate_every
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 5
TEMPERATURE = 0.07

# === Загрузка текстовых эмбеддингов (предвычисленных) ===
print("Loading precomputed text embeddings...")
text_embs = torch.load(TEXT_EMB_PATH, map_location="cpu")  # dict: sample_token → {cam: emb}

# === Подготовка train-сэмплов (750 сцен) ===
nusc = NuScenes(version='v1.0-trainval', dataroot=NUSCENES_DATAROOT, verbose=False)
train_scenes = set(create_splits_scenes()["train"])

tokens_by_scene = {}
for scene in nusc.scene:
    if scene["name"] in train_scenes:
        bucket = tokens_by_scene.setdefault(scene["name"], [])
        current = scene["first_sample_token"]
        while current != "":
            bucket.append(current)
            current = nusc.get("sample", current)["next"]

train_sample_tokens, used_scenes = sample_scenes_uniformly(
    tokens_by_scene, args.max_train_samples, seed=args.sample_seed)
print(f"Selected {len(train_sample_tokens)} train samples "
      f"from {len(used_scenes)} scenes (seed {args.sample_seed}).")

# === Собираем список (sample_token, cam) с эмбеддингами ===
train_pairs = []
for sample_token in train_sample_tokens:
    if sample_token not in text_embs:
        continue
    for cam in ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
                "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]:
        if cam in text_embs[sample_token]:
            train_pairs.append((sample_token, cam))

print(f"Total training pairs: {len(train_pairs)}")

if len(train_pairs) == 0:
    raise ValueError(
        "No training pairs found. Ensure --text_emb_path contains train-split embeddings "
        "(e.g. produced with encode_camera_aware_captions_llm2clip.py --split train)."
    )

# === Датасет ===
class NuScenesImageTextDataset(Dataset):
    def __init__(self, nusc, pairs, text_embs, processor):
        self.nusc = nusc
        self.pairs = pairs
        self.text_embs = text_embs
        self.processor = processor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sample_token, cam = self.pairs[idx]
        # Загрузка изображения
        sample = self.nusc.get("sample", sample_token)
        cam_token = sample["data"][cam]
        cam_data = self.nusc.get("sample_data", cam_token)
        img_path = os.path.join(NUSCENES_DATAROOT, cam_data["filename"])
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        # Загрузка текстового эмбеддинга
        text_emb = self.text_embs[sample_token][cam]
        return inputs, text_emb

# === Загрузка модели ===
print("Loading EVA-CLIP vision encoder (L/14-336)...")
# Используем openai/clip-vit-large-patch14-336 как прокси (веса совместимы с EVA-CLIP)
# vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
vision_model = AutoModel.from_pretrained(
    "microsoft/LLM2CLIP-Openai-L-14-336",
    trust_remote_code=True,
    torch_dtype=torch.float16
).cuda()

# Заморозка backbone
for param in vision_model.parameters():
    param.requires_grad = False

# Применение LoRA к attention-слоям
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # стандарт для ViT
    lora_dropout=0.1,
    bias="none",
)
vision_model = get_peft_model(vision_model, lora_config)
vision_model.print_trainable_parameters()
# Проекция будет инициализирована динамически при первом батче,
# если размерности image/text эмбеддингов не совпадают.
visual_projection = None
vision_model.cuda()
optimizer = torch.optim.AdamW(vision_model.parameters(), lr=LR)

# === Даталоадер ===
dataset = NuScenesImageTextDataset(nusc, train_pairs, text_embs, processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# === Валидационный датасет (опционально) ===
val_dataloader = None
if VAL_TEXT_EMB_PATH:
    print("Loading validation text embeddings...")
    val_text_embs = torch.load(VAL_TEXT_EMB_PATH, map_location="cpu")

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

    val_pairs = []
    for sample_token in val_sample_tokens:
        if sample_token not in val_text_embs:
            continue
        for cam in ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
                    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]:
            if cam in val_text_embs[sample_token]:
                val_pairs.append((sample_token, cam))

    print(f"Validation pairs: {len(val_pairs)}")
    if len(val_pairs) > 0:
        val_dataset = NuScenesImageTextDataset(nusc, val_pairs, val_text_embs, processor)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


def run_validation(vision_model, visual_projection, val_dataloader, val_pairs=None):
    vision_model.eval()
    all_img = []
    all_txt = []
    total_val_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for images, text_embs_gt in tqdm(val_dataloader, desc="Validation", leave=False):
            images = images.cuda(non_blocking=True)
            text_embs_gt = text_embs_gt.cuda(non_blocking=True)

            image_features = vision_model.get_image_features(images)
            image_features = visual_projection(image_features.float())
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
            text_embs_gt = torch.nn.functional.normalize(text_embs_gt.float(), p=2, dim=-1)

            logits = (image_features @ text_embs_gt.T) / TEMPERATURE
            labels = torch.arange(len(images), device=images.device)
            loss_i2t = torch.nn.functional.cross_entropy(logits, labels)
            loss_t2i = torch.nn.functional.cross_entropy(logits.T, labels)
            val_loss = (loss_i2t + loss_t2i) / 2

            total_val_loss += val_loss.item()
            total_batches += 1

            all_img.append(image_features.cpu())
            all_txt.append(text_embs_gt.cpu())

    img = torch.cat(all_img)
    txt = torch.cat(all_txt)
    sim = img @ txt.T

    i2t_r1 = (sim.argmax(dim=1) == torch.arange(len(sim))).float().mean().item()
    t2i_r1 = (sim.argmax(dim=0) == torch.arange(len(sim))).float().mean().item()
    avg_val_loss = total_val_loss / max(total_batches, 1)

    # Метрика бенчмарка: strict "all", без temporal, направление text -> image.
    # sim выше построен как image @ text.T, поэтому берём транспонированную.
    attr = None
    if CAMERA_CAPTIONS is not None and val_pairs:
        attr = attribute_retrieval_metrics(txt @ img.T, val_pairs[:len(sim)], CAMERA_CAPTIONS)

    vision_model.train()
    return i2t_r1, t2i_r1, avg_val_loss, attr

# === Обучение ===
vision_model.train()
best_val_loss = float("inf")
best_attr_mrr = -1.0
best_epoch = -1
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, text_embs_gt = batch
        images = images.cuda()
        text_embs_gt = text_embs_gt.cuda()

        optimizer.zero_grad()
        image_features = vision_model.get_image_features(images)  # [B, 1024]
        if visual_projection is None:
            img_dim = image_features.shape[-1]
            txt_dim = text_embs_gt.shape[-1]
            if img_dim != txt_dim:
                visual_projection = torch.nn.Linear(img_dim, txt_dim).cuda()
                optimizer.add_param_group({"params": visual_projection.parameters(), "lr": LR})
                print(f"Initialized visual projection: {img_dim} -> {txt_dim}")
            else:
                visual_projection = torch.nn.Identity().cuda()
                print(f"Projection not required: image/text dim = {img_dim}")

        image_features = visual_projection(image_features.float())
        image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
        text_embs_gt = torch.nn.functional.normalize(text_embs_gt.float(), p=2, dim=-1)

        # InfoNCE (симметричный CLIP-лосс)
        logits = (image_features @ text_embs_gt.T) / TEMPERATURE  # [B, B]
        labels = torch.arange(len(images), device=images.device)
        loss_i2t = torch.nn.functional.cross_entropy(logits, labels)
        loss_t2i = torch.nn.functional.cross_entropy(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

    if val_dataloader is not None and ((epoch + 1) % VALIDATE_EVERY == 0 or epoch == EPOCHS - 1):
        i2t_r1, t2i_r1, val_loss, attr = run_validation(
            vision_model, visual_projection, val_dataloader, val_pairs)
        print(f"Validation epoch {epoch+1}: loss={val_loss:.4f}, I->T R@1={i2t_r1:.4f}, T->I R@1={t2i_r1:.4f}")
        if attr:
            print(f"  Attribute-based (all, no temporal): R@1={attr['R@1']:.4f} "
                  f"R@10={attr['R@10']:.4f} MRR={attr['MRR']:.4f} med={attr['median_rank']:.0f}")

        # Отбор по репортируемой метрике; без --camera_captions_path откатываемся на лосс.
        if attr:
            improved = attr["MRR"] > best_attr_mrr
            if improved:
                best_attr_mrr = attr["MRR"]
        else:
            improved = val_loss < best_val_loss
        if improved:
            best_val_loss = min(best_val_loss, val_loss)
            best_epoch = epoch + 1
            best_dir = OUTPUT_LORA_DIR + "_best"
            os.makedirs(best_dir, exist_ok=True)
            vision_model.save_pretrained(best_dir)
            torch.save(visual_projection.state_dict(), os.path.join(best_dir, "visual_projection.pt"))
            print(f"New best model saved to {best_dir}")

# === Сохранение LoRA ===
vision_model.save_pretrained(OUTPUT_LORA_DIR)
torch.save(visual_projection.state_dict(), os.path.join(OUTPUT_LORA_DIR, "visual_projection.pt"))
print(f"LoRA adapter saved to {OUTPUT_LORA_DIR}")

if best_epoch > 0:
    if best_attr_mrr >= 0:
        print(f"Best attribute MRR: {best_attr_mrr:.4f} at epoch {best_epoch}")
    else:
        print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
