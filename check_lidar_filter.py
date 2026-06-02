import json
import os
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from collections import defaultdict

dataroot = "/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval"
caption_path = "../final_caption_bbox_token.json"

CAMERAS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)

with open(caption_path, "r") as f:
    captions = json.load(f)

train_scenes = set(create_splits_scenes()["train"])
val_scenes = set(create_splits_scenes()["val"])

train_sample_tokens = []
for scene in nusc.scene:
    if scene["name"] in train_scenes:
        current = scene["first_sample_token"]
        while current != "":
            train_sample_tokens.append(current)
            current = nusc.get("sample", current)["next"]

val_sample_tokens = []
for scene in nusc.scene:
    if scene["name"] in val_scenes:
        current = scene["first_sample_token"]
        while current != "":
            val_sample_tokens.append(current)
            current = nusc.get("sample", current)["next"]

total_train_frames = len(train_sample_tokens)
total_val_frames = len(val_sample_tokens)

print(f"Train scenes: {len(train_scenes)}")
print(f"Train sample tokens (all frames): {total_train_frames}")
print(f"Max possible train pairs: {total_train_frames * len(CAMERAS)}")
print()
print(f"Val scenes: {len(val_scenes)}")
print(f"Val sample tokens (all frames): {total_val_frames}")
print(f"Max possible val pairs: {total_val_frames * len(CAMERAS)}")
print()

for min_pts in [1, 5, 20]:
    # Собираем, какие аннотации проходят фильтр
    # {sample_token: {cam: [atts]}}
    camera_captions = {}
    for bbox_token, data in captions.items():
        sample_token = data["sample_token"]
        ann = nusc.get("sample_annotation", bbox_token)
        if ann["num_lidar_pts"] + ann["num_radar_pts"] < min_pts:
            continue
        cam_file = data["cam_file"]
        cam_name = cam_file.split("/")[1]

        if sample_token not in camera_captions:
            camera_captions[sample_token] = {}
        if cam_name not in camera_captions[sample_token]:
            camera_captions[sample_token][cam_name] = set()

        # extract atts
        if "attribute_caption" in data and isinstance(data["attribute_caption"], dict):
            attr = data["attribute_caption"].get("attribute_caption", "").strip()
            if attr and attr.lower() != "none":
                camera_captions[sample_token][cam_name].add(attr)

    def count_pairs(sample_tokens):
        total = 0
        by_cam = {c: 0 for c in CAMERAS}
        tokens_with_any = 0
        for token in sample_tokens:
            has = False
            if token in camera_captions:
                for cam in CAMERAS:
                    if cam in camera_captions[token] and camera_captions[token][cam]:
                        by_cam[cam] += 1
                        total += 1
                        has = True
            if has:
                tokens_with_any += 1
        return total, by_cam, tokens_with_any

    train_total, train_by_cam, train_tokens = count_pairs(train_sample_tokens)
    val_total, val_by_cam, val_tokens = count_pairs(val_sample_tokens)

    print(f"{'='*55}")
    print(f"min_lidar_points = {min_pts}")
    print(f"{'='*55}")
    print(f"  TRAIN:")
    print(f"    Frames with any caption: {train_tokens} / {total_train_frames}")
    print(f"    Total valid pairs:       {train_total} / {total_train_frames * 6}")
    for cam in CAMERAS:
        print(f"      {cam:<22}: {train_by_cam[cam]}")
    print(f"  VAL:")
    print(f"    Frames with any caption: {val_tokens} / {total_val_frames}")
    print(f"    Total valid pairs:       {val_total} / {total_val_frames * 6}")
    for cam in CAMERAS:
        print(f"      {cam:<22}: {val_by_cam[cam]}")
    print(f"  Unique sample tokens in captions JSON: {len(camera_captions)}")
    print()
