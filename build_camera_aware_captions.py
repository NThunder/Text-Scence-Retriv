# tools/build_camera_aware_captions.py
import json
import os

import argparse
import os

# Конфигурация
def parse_args():
    parser = argparse.ArgumentParser(description="Build camera-aware captions")
    
    # Determine default path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming script is in retriv/ and json is in root/
    default_caption_path = os.path.join(os.path.dirname(script_dir), "final_caption_bbox_token.json")
    
    parser.add_argument("--dataroot", type=str, default="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval", help="Path to nuScenes dataset")
    parser.add_argument("--caption_path", type=str, default=default_caption_path, help="Path to caption json")
    parser.add_argument("--output_path", type=str, default="./camera_aware_captions_map.json", help="Path to output json")
    return parser.parse_args()

args = parse_args()

caption_path = args.caption_path
output_path = args.output_path

with open(caption_path, "r") as f:
    captions = json.load(f)

camera_captions = {}

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
        parts.append(f"is {motion}")
    if map_desc:
        parts.append(f"{map_desc}")
    if relation:
        parts.append(relation)
    
    if parts:
        desc = "A " + " ".join(parts) + "."
        return desc, attrs
    else:
        return "An object with no description.", attrs
from nuscenes import NuScenes
nuscenes_dataroot = args.dataroot
nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_dataroot, verbose=False)

for bbox_token, data in captions.items():
    sample_token = data["sample_token"]
    ann = nusc.get("sample_annotation", bbox_token)
    if ann["num_lidar_pts"] + ann["num_radar_pts"] < 20:
        # print("sample_annotation", data)
        continue

    cam_file = data["cam_file"]  # e.g., "samples/CAM_BACK/..."

    # Извлекаем имя камеры: CAM_BACK, CAM_FRONT и т.д.
    cam_name = cam_file.split("/")[1]  # → "CAM_BACK"

    caption, atts = build_final_caption(data)

    if caption == "An object with no description.":
        print("An object with no description")

    # Добавляем в структуру
    if sample_token not in camera_captions:
        camera_captions[sample_token] = {
            "CAM_FRONT": [],
            "CAM_FRONT_RIGHT": [],
            "CAM_FRONT_LEFT": [],
            "CAM_BACK": [],
            "CAM_BACK_LEFT": [],
            "CAM_BACK_RIGHT": [],
            "atts": []
        }
    
    if cam_name in camera_captions[sample_token]:
        f = False
        for a in atts:
            if a in camera_captions[sample_token]["atts"]:
                f = True
                break
            else:
                camera_captions[sample_token]["atts"].append(a)
        if not f:
            camera_captions[sample_token][cam_name].append(caption)
    # print("atts:    ", camera_captions[sample_token]["atts"])

# Сохраняем
with open(output_path, "w") as f:
    json.dump(camera_captions, f, indent=2)

print(f"Saved to {output_path}")