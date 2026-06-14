# tools/encode_camera_aware_captions_siglip.py
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModel
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Encode camera captions with SigLIP")
    parser.add_argument("--dataroot", type=str, default="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval")
    parser.add_argument("--camera_captions_path", type=str, default="./camera_aware_captions_short.json")
    parser.add_argument("--output_path", type=str, default="./camera_text_embeddings_siglip.pth")
    parser.add_argument("--max_val_samples", type=int, default=-1,
        help="Max validation samples (-1 = all)")
    parser.add_argument("--val_samples_per_scene", type=int, default=-1,
        help="Number of evenly spaced samples per validation scene (-1 = all)")
    parser.add_argument("--no_prefix", action="store_true", help="Omit 'The camera view contains: ' prefix")
    return parser.parse_args()

args = parse_args()
nuscenes_dataroot = args.dataroot
camera_captions_path = args.camera_captions_path
output_path = args.output_path

with open(camera_captions_path, "r") as f:
    camera_captions = json.load(f)

model_name = "google/siglip-base-patch16-224"
tokenizer = AutoTokenizer.from_pretrained(model_name)
text_encoder = AutoModel.from_pretrained(model_name).eval().cuda()

nusc = NuScenes(
    version='v1.0-trainval',
    dataroot=nuscenes_dataroot,
    verbose=False
)

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
print(f"Selected {len(val_sample_tokens)} validation samples.")

camera_text_embs = {}

for current_sample_token in tqdm(val_sample_tokens, desc="Encoding text (SigLIP)"):
    if current_sample_token not in camera_captions:
        continue

    cam_dict = camera_captions[current_sample_token]
    embs_dict = {}

    for cam_name, captions in cam_dict.items():
        if not captions:
            continue
        if args.no_prefix:
            scene_text = " ".join(captions)
        else:
            scene_text = "The camera view contains: " + "; ".join(captions)

        inputs = tokenizer(
            scene_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to("cuda")

        with torch.no_grad():
            outputs = text_encoder.text_model(**inputs)
            emb = outputs.pooler_output  # [1, 768]
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            embs_dict[cam_name] = emb.cpu().squeeze(0)

    camera_text_embs[current_sample_token] = embs_dict

torch.save(camera_text_embs, output_path)
print(f"Saved to {output_path}")