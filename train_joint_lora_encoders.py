# tools/train_joint_lora_encoders.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor
from peft import LoraConfig, get_peft_model
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from tqdm import tqdm
import json
import argparse
from llm2vec import LLM2Vec
import numpy as np
from tqdm import tqdm
import json
import argparse
import copy
import matplotlib.pyplot as plt
from collections import defaultdict

# === Конфигурация ===
def parse_args():
    parser = argparse.ArgumentParser(description="Joint training of text and image encoders with LoRA")
    parser.add_argument("--dataroot", type=str, default="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval", help="Path to nuScenes dataset")
    parser.add_argument("--camera_captions_path", type=str, default="./camera_aware_captions_motion_map.json", help="Path to camera captions JSON")
    parser.add_argument("--output_lora_text", type=str, default="./lora_text_encoder_joint_all", help="Path to output text LoRA directory")
    parser.add_argument("--output_lora_image", type=str, default="./lora_image_encoder_joint_all", help="Path to output image LoRA directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs")
    parser.add_argument("--temperature", type=float, default=0.07, help="Contrastive loss temperature")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--validate_every", type=int, default=1, help="Run validation every N epochs")
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
BATCH_SIZE = args.batch_size
LR = args.lr
EPOCHS = args.epochs
TEMPERATURE = args.temperature
LORA_R = args.lora_r
VALIDATE_EVERY = args.validate_every

# === 1. Загрузка описаний сцен ===
print("Loading camera-aware captions...")
with open(CAMERA_CAPTIONS_PATH, "r") as f:
    camera_captions = json.load(f)

# === 2. Подготовка nuScenes (750 train samples) ===
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

# === 3. Сбор тренировочных пар ===
CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
]

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

# === 4. Загрузка моделей с LoRA ===
print("\n=== Loading models ===")

# --- Общая модель LLM2CLIP для проекций (полностью заморожена) ---
print("Loading frozen LLM2CLIP projection layers...")
clip_model = AutoModel.from_pretrained(
    "microsoft/LLM2CLIP-Openai-L-14-336",
    trust_remote_code=True,
    torch_dtype=torch.float16
).cuda()
clip_model.eval()
for param in clip_model.parameters():
    param.requires_grad = False

# --- Визуальный энкодер (только vision часть из LLM2CLIP) ---
print("Loading CLIP vision encoder with LoRA...")
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
vision_model = AutoModel.from_pretrained(
    "microsoft/LLM2CLIP-Openai-L-14-336",
    trust_remote_code=True,
    torch_dtype=torch.float16
).cuda()

for param in vision_model.parameters():
    param.requires_grad = False

lora_config_vision = LoraConfig(
    r=LORA_R,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
)
vision_model = get_peft_model(vision_model, lora_config_vision)
vision_model.cuda()
print("Vision LoRA trainable params:")
vision_model.print_trainable_parameters()

# --- Текстовый энкодер (внешний LLM2Vec) ---
print("\nLoading LLM2Vec text encoder with LoRA...")
llm_model_name = "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=False)
text_model = AutoModel.from_pretrained(
    llm_model_name,
    trust_remote_code=False,
    torch_dtype=torch.float16
)
for param in text_model.parameters():
    param.requires_grad = False

lora_config_text = LoraConfig(
    r=LORA_R,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
)
text_model = get_peft_model(text_model, lora_config_text)
text_model.cuda()

text_model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
l2v = LLM2Vec(text_model, tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512)

print("Text LoRA trainable params:")
text_model.print_trainable_parameters()


# === 6. Датасет ===
class JointNuScenesDataset(Dataset):
    def __init__(self, nusc, pairs, processor, return_metadata=False):
        self.nusc = nusc
        self.pairs = pairs
        self.processor = processor
        self.return_metadata = return_metadata
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        sample_token, cam, text = self.pairs[idx]
        
        # Загрузка изображения
        sample = self.nusc.get("sample", sample_token)
        cam_token = sample["data"][cam]
        cam_data = self.nusc.get("sample_data", cam_token)
        img_path = os.path.join(NUSCENES_DATAROOT, cam_data["filename"])
        image = Image.open(img_path).convert("RGB")
        image_inputs = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        
        if self.return_metadata:
            return image_inputs, text, sample_token, cam
        return image_inputs, text

def prepare_val_dataset(nusc, camera_captions, num_samples=150, val_samples_per_scene=-1):
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

    return JointNuScenesDataset(nusc, val_pairs, processor, return_metadata=True), len(val_pairs)

val_dataset, num_val_pairs = prepare_val_dataset(nusc, camera_captions, num_samples=args.max_val_samples, val_samples_per_scene=args.val_samples_per_scene)
print(f"Prepared validation dataset with {num_val_pairs} pairs")

# === 8. Функция валидации ===
def validate(vision_model, text_model, clip_model, val_dataset, batch_size=16):
    vision_model.eval()
    text_model.eval()
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(batch_size, len(val_dataset)),
        pin_memory=True
    )
    
    all_image_embs = []
    all_text_embs = []
    all_sample_tokens = []
    all_cameras = []
    total_val_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                images, text_inputs = batch[0], batch[1]
                images = images.cuda(non_blocking=True)

                image_features = vision_model.get_image_features(images)
                img_emb = torch.nn.functional.normalize(image_features, p=2, dim=-1)

                text_emb_raw = l2v.encode(text_inputs, convert_to_tensor=True).to('cuda')
                text_emb_clip = clip_model.get_text_features(text_emb_raw.float())
                txt_emb = F.normalize(text_emb_clip, p=2, dim=-1)

                logits_per_image = (img_emb @ txt_emb.T) / TEMPERATURE
                logits_per_text = logits_per_image.T
                labels = torch.arange(len(images), device=images.device)
                loss_i2t = F.cross_entropy(logits_per_image, labels)
                loss_t2i = F.cross_entropy(logits_per_text, labels)
                val_loss = (loss_i2t + loss_t2i) / 2

                total_val_loss += val_loss.item()
                total_batches += 1

                all_image_embs.append(img_emb.cpu())
                all_text_embs.append(txt_emb.cpu())
                if len(batch) > 2:
                    all_sample_tokens.extend(batch[2])
                    all_cameras.extend(batch[3])
    
    all_image_embs = torch.cat(all_image_embs)
    all_text_embs = torch.cat(all_text_embs)
    
    # Стандартные метрики (exact match)
    sim_matrix = all_image_embs @ all_text_embs.T
    i2t_r1 = (sim_matrix.argmax(dim=1) == torch.arange(len(sim_matrix))).float().mean().item()
    t2i_r1 = (sim_matrix.argmax(dim=0) == torch.arange(len(sim_matrix))).float().mean().item()
    
    # Метрики с учётом атрибутов (all mode, без temporal)
    attr_r1, attr_r5, attr_r10, attr_mrr = 0.0, 0.0, 0.0, 0.0
    if all_sample_tokens:
        attrs_by_idx = []
        attr_to_indices = defaultdict(set)
        for idx, (token, cam) in enumerate(zip(all_sample_tokens, all_cameras)):
            attrs = set(camera_captions.get(token, {}).get(cam, []))
            attrs_by_idx.append(attrs)
            for attr in attrs:
                attr_to_indices[attr].add(idx)

        ranks = []
        for i in range(len(all_sample_tokens)):
            sample_token, cam = all_sample_tokens[i], all_cameras[i]
            query_attrs = set(camera_captions.get(sample_token, {}).get(cam, []))
            relevant = set()
            if query_attrs:
                for idx, candidate_attrs in enumerate(attrs_by_idx):
                    if query_attrs.issubset(candidate_attrs):
                        relevant.add(idx)

            sorted_indices = torch.argsort(sim_matrix[i], descending=True)
            found = False
            for rank_pos, idx in enumerate(sorted_indices.tolist()):
                if idx in relevant:
                    ranks.append(rank_pos + 1)
                    found = True
                    break
            if not found:
                ranks.append(len(all_sample_tokens))

        ranks = np.array(ranks)
        attr_r1 = np.mean(ranks <= 1)
        attr_r5 = np.mean(ranks <= 5)
        attr_r10 = np.mean(ranks <= 10)
        attr_mrr = np.mean(1.0 / ranks)
    
    vision_model.train()
    text_model.train()
    
    avg_val_loss = total_val_loss / max(total_batches, 1)

    return i2t_r1, t2i_r1, attr_r1, attr_r5, attr_r10, attr_mrr, sim_matrix, avg_val_loss



dataset = JointNuScenesDataset(nusc, train_pairs, processor)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# === 7. Оптимизатор ===
params = list(vision_model.parameters()) + list(text_model.parameters())
optimizer = torch.optim.AdamW(params, lr=LR)

# === 8. Совместное обучение ===
print(f"\n=== Starting joint training for {EPOCHS} epochs ===")
print(f"Batch size: {BATCH_SIZE}, LR: {LR}, Temperature: {TEMPERATURE}")

best_val_loss = float("inf")
best_epoch = -1

for epoch in range(EPOCHS):
    vision_model.train()
    text_model.train()
    total_loss = 0
    total_i2t_acc = 0
    total_t2i_acc = 0
    
    for batch_idx, (images, text_inputs) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        images = images.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(dtype=torch.float16):
            # --- Визуальные эмбеддинги (как в оригинальном скрипте) ---
            # Получаем 1024-dim features от vision encoder
            
            image_features = vision_model.get_image_features(images)
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
            
            # --- Текстовые эмбеддинги (как в оригинальном скрипте) ---
            # Получаем 4096-dim features от LLM2Vec

            text_emb_raw = l2v.encode(text_inputs, convert_to_tensor=True).to('cuda')  # [1, 4096]
            text_emb_clip = clip_model.get_text_features(text_emb_raw.float())
            text_features = torch.nn.functional.normalize(text_emb_clip, p=2, dim=-1)
            
            # --- Симметричный контрастивный лосс ---
            logits_per_image = (image_features @  text_features.T) / TEMPERATURE
            logits_per_text = logits_per_image.T
            
            labels = torch.arange(len(images)).cuda()
            
            loss_i2t = F.cross_entropy(logits_per_image, labels)
            loss_t2i = F.cross_entropy(logits_per_text, labels)
            loss = (loss_i2t + loss_t2i) / 2
            
            # --- Метрики ---
            i2t_preds = logits_per_image.argmax(dim=1)
            t2i_preds = logits_per_text.argmax(dim=1)
            i2t_acc = (i2t_preds == labels).float().mean()
            t2i_acc = (t2i_preds == labels).float().mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_i2t_acc += i2t_acc.item()
        total_t2i_acc += t2i_acc.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_i2t_acc = total_i2t_acc / len(dataloader)
    avg_t2i_acc = total_t2i_acc / len(dataloader)
    
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | I→T Acc: {avg_i2t_acc:.2%} | T→I Acc: {avg_t2i_acc:.2%}")

    if len(val_dataset) > 0 and ((epoch + 1) % VALIDATE_EVERY == 0 or epoch == EPOCHS - 1):
        print(f"\n{'='*50}")
        print(f"Running validation at epoch {epoch+1}...")
        print(f"{'='*50}")

        i2t_r1, t2i_r1, attr_r1, attr_r5, attr_r10, attr_mrr, sim_matrix, val_loss = validate(
            vision_model, text_model, clip_model, val_dataset, batch_size=BATCH_SIZE
        )

        mean_pos_sim = torch.diag(sim_matrix).mean().item()
        mean_neg_sim = (sim_matrix.sum() - torch.diag(sim_matrix).sum()) / (len(sim_matrix) * (len(sim_matrix) - 1))

        print(f"\nValidation Results (epoch {epoch+1}):")
        print(f"  Val loss: {val_loss:.4f}")
        print(f"  I→T R@1 (exact): {i2t_r1:.2%}")
        print(f"  T→I R@1 (exact): {t2i_r1:.2%}")
        print(f"  Attribute-based (all, no temporal):")
        print(f"    R@1: {attr_r1:.4f}  R@5: {attr_r5:.4f}  R@10: {attr_r10:.4f}  MRR: {attr_mrr:.4f}")
        print(f"  Mean positive similarity: {mean_pos_sim:.4f}")
        print(f"  Mean negative similarity: {mean_neg_sim:.4f}")
        print(f"  Separation gap: {mean_pos_sim - mean_neg_sim:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            print("  New best validation loss! Saving checkpoint...")

            vision_model.save_pretrained(OUTPUT_LORA_IMAGE + "_best")
            processor.save_pretrained(OUTPUT_LORA_IMAGE + "_best")
            text_model.save_pretrained(OUTPUT_LORA_TEXT + "_best")
            tokenizer.save_pretrained(OUTPUT_LORA_TEXT + "_best")
            print(f"  Best model saved to {OUTPUT_LORA_IMAGE}_best and {OUTPUT_LORA_TEXT}_best")

        print(f"{'='*50}\n")

# === 11. Сохранение финальных адаптеров ===
print("\n=== Saving final LoRA adapters ===")
vision_model.save_pretrained(OUTPUT_LORA_IMAGE)
processor.save_pretrained(OUTPUT_LORA_IMAGE)
print(f"Final image LoRA saved to {OUTPUT_LORA_IMAGE}")

text_model.save_pretrained(OUTPUT_LORA_TEXT)
tokenizer.save_pretrained(OUTPUT_LORA_TEXT)
print(f"Final text LoRA saved to {OUTPUT_LORA_TEXT}")

if best_epoch > 0:
    print(f"\n✅ Training completed! Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
else:
    print("\n✅ Training completed! Validation was not run or no improvement recorded.")
