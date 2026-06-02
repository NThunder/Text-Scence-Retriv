import json
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from collections import defaultdict
import numpy as np

dataroot = "/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval"
CAMERAS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)
val_scenes = set(create_splits_scenes()["val"])

val_tokens = []
for scene in nusc.scene:
    if scene["name"] in val_scenes:
        t = scene["first_sample_token"]
        while t != "":
            val_tokens.append(t)
            t = nusc.get("sample", t)["next"]

for level in ["L0", "L1", "L2", "L3", "L4"]:
    path = f"./retriv/camera_aware_captions_{level}_p5.json"
    with open(path) as f:
        captions = json.load(f)

    pairs = []
    for token in val_tokens:
        if token not in captions:
            continue
        for cam in CAMERAS:
            if cam in captions[token] and captions[token][cam]:
                attrs = set(captions[token][cam])
                pairs.append((token, cam, attrs))

    attr_to_idx = defaultdict(set)
    for idx, (_, _, attrs) in enumerate(pairs):
        for a in attrs:
            attr_to_idx[a].add(idx)

    rel_sizes = []
    for idx, (_, _, query_attrs) in enumerate(pairs):
        relevant = set()
        for a in query_attrs:
            relevant.update(attr_to_idx[a])
        rel_sizes.append(len(relevant))

    rel_sizes = np.array(rel_sizes)
    print(f"{level}: val_pairs={len(pairs):>6}  avg={rel_sizes.mean():.1f}  std={rel_sizes.std():.1f}  min={rel_sizes.min()}  max={rel_sizes.max()}")
