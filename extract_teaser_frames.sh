#!/bin/bash
# Extract teaser frames: same query described at L0–L4, top-1 retrieved image per level.
set -e

export CAPFILE="./retriv/camera_aware_captions_L0_p5.json"
export NUSCENES="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval"
export OUTDIR="./teaser_frames"

mkdir -p "$OUTDIR"

python3 << 'PYEOF'
import json, os
from PIL import Image
from nuscenes import NuScenes

CAPFILE = os.environ["CAPFILE"]
NUSCENES = os.environ["NUSCENES"]
OUTDIR = os.environ["OUTDIR"]

nusc = NuScenes(version="v1.0-trainval", dataroot=NUSCENES, verbose=False)
with open(CAPFILE) as f:
    camera_captions = json.load(f)

def make_text(sample_token, cam, capfile):
    attrs = capfile.get(sample_token, {}).get(cam, [])
    return "The camera view contains: " + "; ".join(attrs) if attrs else "(no description)"

def load_image(sample_token, cam_name):
    cam_token = nusc.get("sample", sample_token)["data"][cam_name]
    cam_data = nusc.get("sample_data", cam_token)
    img_path = os.path.join(NUSCENES, cam_data["filename"])
    return Image.open(img_path).convert("RGB")

# Load all 5 level results
levels = ["L0", "L1", "L2", "L3", "L4"]
all_data = {}
for lvl in levels:
    with open(f"./results_4perscene/teaser_{lvl}.json") as f:
        all_data[lvl] = json.load(f)

# Find a common query that exists in all 5 levels
# Index 0 is the first query in each file; they use same val set, so same index = same pair
query_info = []
for lvl in levels:
    d = all_data[lvl][0]
    query_info.append((lvl, d["query_token"], d["query_camera"], d["query_text"]))

# Verify all 5 point to the same (token, cam)
tokens = set((d[1], d[2]) for d in query_info)
print(f"Common query pairs across levels: {len(tokens)}")
for lvl, tok, cam, txt in query_info:
    print(f"  {lvl}: {tok} / {cam}")

# Extract the query image (from any level, e.g., L3)
q_tok, q_cam = query_info[3][1], query_info[3][2]
query_img = load_image(q_tok, q_cam)
query_img.save(os.path.join(OUTDIR, "query_frame.png"))
query_text = query_info[3][3]
print(f"\nQuery saved: {q_tok} / {q_cam}")
print(f"Query text (L3): {query_text}")

# For each level, save the top-1 retrieved image
saved = []
for lvl in levels:
    d = all_data[lvl][0]
    top1 = d["top10"][0]
    ret_img = load_image(top1["sample_token"], top1["camera"])
    fname = f"ret_{lvl}.png"
    ret_img.save(os.path.join(OUTDIR, fname))

    ret_text = make_text(top1["sample_token"], top1["camera"], camera_captions)
    mark = "✓" if top1["relevant"] else "✗"
    print(f"\n{lvl} → rank=1 {mark}")
    print(f"  Retrieved: {top1['sample_token']} / {top1['camera']}")
    print(f"  Text: {ret_text}")

    saved.append({
        "level": lvl,
        "file": fname,
        "relevant": bool(top1["relevant"]),
        "sample_token": top1["sample_token"],
        "camera": top1["camera"],
    })

# Save metadata
meta = {
    "query": {"token": q_tok, "camera": q_cam, "text": query_text},
    "results": saved
}
with open(os.path.join(OUTDIR, "teaser_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"\nDone! Frames in {OUTDIR}/")
PYEOF
