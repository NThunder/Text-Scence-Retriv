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
    parser.add_argument("--level", type=str, default="attr_motion_map",
        choices=["attr", "attr_motion", "attr_map", "attr_motion_map", "full"],
        help=(
            "Detalization level of captions:\n"
            "  attr            — attribute only         (e.g. 'A blue car.')\n"
            "  attr_motion     — attribute + motion     (e.g. 'A blue car is moving quickly.')\n"
            "  attr_map        — attribute + map        (e.g. 'A blue car in the stop lane.')\n"
            "  attr_motion_map — attribute + motion + map (default, = motion_map)\n"
            "  full            — all fields incl. depth + localization"
        )
    )
    parser.add_argument(
        "--min_lidar_points",
        type=int,
        default=20,
        help="Minimum num_lidar_pts + num_radar_pts to keep annotation",
    )
    parser.add_argument(
        "--dedup_scope",
        type=str,
        default="camera",
        choices=["camera", "keyframe"],
        help=(
            "Where duplicate descriptions are collapsed:\n"
            "  camera   - within a single camera view, on the caption itself (default).\n"
            "             One view never lists the same description twice, but an\n"
            "             identical object seen from another camera is preserved.\n"
            "  keyframe - legacy behaviour: an object is dropped if its attribute\n"
            "             string already appeared in ANY camera of the same keyframe."
        ),
    )
    return parser.parse_args()

args = parse_args()

caption_path = args.caption_path
output_path = args.output_path

with open(caption_path, "r") as f:
    captions = json.load(f)

camera_captions = {}

def build_final_caption(data, level="attr_motion_map"):
    """
    Build a text caption from TOD3Cap annotation data.

    level controls which fields are included:
        attr            — attribute only
        attr_motion     — attribute + motion
        attr_map        — attribute + map
        attr_motion_map — attribute + motion + map  (original behaviour)
        full            — all fields (attr + loc + depth + motion + map + relation)
    """
    attrs = []
    if "attribute_caption" in data and isinstance(data["attribute_caption"], dict):
        attr = data["attribute_caption"].get("attribute_caption", "").strip()
        if attr and attr.lower() != "none":
            attrs.append(attr)

    loc = ""
    if level == "full":
        if "localization_caption" in data and isinstance(data["localization_caption"], dict):
            loc = data["localization_caption"].get("localization_caption", "").strip()
            if loc.lower() == "none": loc = ""

    depth = ""
    if level == "full":
        if "depth_caption" in data and isinstance(data["depth_caption"], dict):
            depth = data["depth_caption"].get("depth_caption", "").strip()
            if depth.lower() == "none": depth = ""

    motion = ""
    if level in ("attr_motion", "attr_motion_map", "full"):
        if "motion_caption" in data and isinstance(data["motion_caption"], dict):
            motion = data["motion_caption"].get("motion_caption", "").strip()
            if motion.lower() == "none": motion = ""

    map_desc = ""
    if level in ("attr_map", "attr_motion_map", "full"):
        if "map_caption" in data and isinstance(data["map_caption"], dict):
            map_desc = data["map_caption"].get("map_caption", "").strip()
            if map_desc.lower() == "none": map_desc = ""

    relation = ""
    if level == "full":
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

CAMERAS_LIST = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
                "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

n_kept = 0
n_dropped = 0

for bbox_token, data in captions.items():
    sample_token = data["sample_token"]
    ann = nusc.get("sample_annotation", bbox_token)
    if ann["num_lidar_pts"] + ann["num_radar_pts"] < args.min_lidar_points:
        # print("sample_annotation", data)
        continue

    cam_file = data["cam_file"]  # e.g., "samples/CAM_BACK/..."

    # Извлекаем имя камеры: CAM_BACK, CAM_FRONT и т.д.
    cam_name = cam_file.split("/")[1]  # → "CAM_BACK"

    caption, atts = build_final_caption(data, level=args.level)

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
    
    if cam_name in CAMERAS_LIST:
        rec = camera_captions[sample_token]

        if args.dedup_scope == "keyframe":
            # Legacy rule, kept only so older runs stay reproducible: the object is
            # discarded if its attribute string was already seen in any camera of
            # this keyframe, so one of two white cars in different views is lost.
            dup = False
            for a in atts:
                if a in rec["atts"]:
                    dup = True
                    break
                rec["atts"].append(a)
            if dup:
                n_dropped += 1
            else:
                rec[cam_name].append(caption)
                n_kept += 1
        else:
            # Default rule: collapse duplicates inside one camera view only, keyed on
            # the caption itself. A view never lists "A white car." twice, while the
            # same object type seen from another camera is kept.
            if caption in rec[cam_name]:
                n_dropped += 1
            else:
                rec[cam_name].append(caption)
                n_kept += 1
            for a in atts:
                if a not in rec["atts"]:
                    rec["atts"].append(a)

# Сохраняем
with open(output_path, "w") as f:
    json.dump(camera_captions, f, indent=2)

print(f"Saved to {output_path}")

# ========== Статистика ==========
total_samples = len(camera_captions)
cam_counts = {cam: 0 for cam in CAMERAS_LIST}
total_pairs = 0
total_captions = 0
samples_with_any = 0

for sample_token, data in camera_captions.items():
    has_any = False
    for cam in CAMERAS_LIST:
        n = len(data.get(cam, []))
        if n > 0:
            cam_counts[cam] += 1
            total_pairs += 1
            total_captions += n
            has_any = True
    if has_any:
        samples_with_any += 1

print(f"\n{'='*50}")
print(f"Statistics:")
print(f"  Total samples in output:              {total_samples}")
print(f"  Samples with at least one caption:    {samples_with_any}")
print(f"  Total camera-view pairs (non-empty):  {total_pairs}")
for cam in CAMERAS_LIST:
    print(f"    {cam:<22}: {cam_counts[cam]}")
print(f"  Total captions:                       {total_captions}")
print(f"  Descriptions kept:                    {n_kept}")
print(f"  Duplicates dropped:                   {n_dropped}"
      f"  ({n_dropped / max(1, n_kept + n_dropped):.1%} of annotations above threshold)")
print(f"  Level: {args.level},  min_lidar_points: {args.min_lidar_points},"
      f"  dedup_scope: {args.dedup_scope}")
print(f"{'='*50}")
