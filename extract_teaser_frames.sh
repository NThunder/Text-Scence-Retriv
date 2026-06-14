#!/bin/bash
# Extract retrieval examples from GME ft. L3 L0→L0 results for the teaser.
# Run on the server in the gme_finetune environment.
# Saves 3 frames to figures/ of the WACV 2027 template.

set -e

export VALDIR="./results_4perscene/gme_ftL3_L0"
export NUSCENES="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval"
export OUTDIR="./teaser_frames"

mkdir -p "$OUTDIR"

python3 << 'PYEOF'
import json, os
from PIL import Image
from nuscenes import NuScenes

VALDIR = os.environ["VALDIR"]
NUSCENES = os.environ["NUSCENES"]
OUTDIR = os.environ["OUTDIR"]

nusc = NuScenes(version="v1.0-trainval", dataroot=NUSCENES, verbose=False)

with open(f"{VALDIR}/retrieval_examples.json") as f:
    examples = json.load(f)

def load_image(sample_token, cam_name):
    cam_token = nusc.get("sample", sample_token)["data"][cam_name]
    cam_data = nusc.get("sample_data", cam_token)
    img_path = os.path.join(NUSCENES, cam_data["filename"])
    return Image.open(img_path).convert("RGB")

# Pick: top-1, top-2 (exact match), top-5+ (exact miss)
best_r1 = [e for e in examples.get("best", []) if e["rank"] == 1]
best_r2 = [e for e in examples.get("best", []) if e["rank"] == 2]
worst_r5 = [e for e in examples.get("worst", []) if e["rank"] >= 5]

# Use the first available or fall back to any best/worst
hit1 = best_r1[0] if best_r1 else (examples.get("best", [None])[0])
hit2 = best_r2[0] if best_r2 else (examples.get("best", [None])[1] if len(examples.get("best", [])) > 1 else hit1)
miss = worst_r5[0] if worst_r5 else (examples.get("worst", [None])[0])

for name, e in [("frame_hit1", hit1), ("frame_hit2", hit2), ("frame_miss", miss)]:
    if e is None:
        print(f"WARNING: no example for {name}, using placeholder")
        continue
    img = load_image(e["sample_token"], e["camera"])
    path = os.path.join(OUTDIR, f"{name}.png")
    img.save(path)
    print(f"{name}: token={e['sample_token']}, cam={e['camera']}, rank={e['rank']}, saved to {path}")

print("Done!")
PYEOF
