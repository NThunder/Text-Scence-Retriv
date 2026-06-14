#!/bin/bash
# Extract retrieval examples from GME ft. L3 L0→L0 results for the teaser.
# Saves frames + query/prediction text annotations.

set -e

export VALDIR="./results_4perscene/gme_ftL3_L0"
export CAPFILE="./retriv/camera_aware_captions_L0_p5.json"
export NUSCENES="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval"
export OUTDIR="./teaser_frames"

mkdir -p "$OUTDIR"

python3 << 'PYEOF'
import json, os
from PIL import Image
from nuscenes import NuScenes

VALDIR = os.environ["VALDIR"]
CAPFILE = os.environ["CAPFILE"]
NUSCENES = os.environ["NUSCENES"]
OUTDIR = os.environ["OUTDIR"]

nusc = NuScenes(version="v1.0-trainval", dataroot=NUSCENES, verbose=False)

with open(CAPFILE) as f:
    camera_captions = json.load(f)

with open(f"{VALDIR}/retrieval_examples.json") as f:
    examples = json.load(f)

def make_text(sample_token, cam):
    attrs = camera_captions.get(sample_token, {}).get(cam, [])
    return "The camera view contains: " + "; ".join(attrs) if attrs else "(no description)"

def load_image(sample_token, cam_name):
    cam_token = nusc.get("sample", sample_token)["data"][cam_name]
    cam_data = nusc.get("sample_data", cam_token)
    img_path = os.path.join(NUSCENES, cam_data["filename"])
    return Image.open(img_path).convert("RGB")

# Pick: top-1, top-2, top-5+ (exact match rank)
best_r1 = [e for e in examples.get("best", []) if e["rank"] == 1]
best_r2 = [e for e in examples.get("best", []) if e["rank"] == 2]
worst_r5 = [e for e in examples.get("worst", []) if e["rank"] >= 5]

hit1 = best_r1[0] if best_r1 else (examples.get("best", [None])[0])
hit2 = best_r2[0] if best_r2 else (examples.get("best", [None])[1] if len(examples.get("best", [])) > 1 else hit1)
miss = worst_r5[0] if worst_r5 else (examples.get("worst", [None])[0])

saved = []
for name, e in [("frame_hit1", hit1), ("frame_hit2", hit2), ("frame_miss", miss)]:
    if e is None:
        print(f"WARNING: no example for {name}")
        continue
    img = load_image(e["sample_token"], e["camera"])
    path = os.path.join(OUTDIR, f"{name}.png")
    img.save(path)

    text = make_text(e["sample_token"], e["camera"])
    print(f"\n{'='*60}")
    print(f"{name} (rank={e['rank']})")
    print(f"  Token: {e['sample_token']}, Cam: {e['camera']}")
    print(f"  Text: {text}")

    saved.append({
        "file": f"{name}.png",
        "rank": e["rank"],
        "sample_token": e["sample_token"],
        "camera": e["camera"],
        "text": text,
        "attributes": e.get("attributes", [])
    })

# Query text: use the first retrieved example's token as query
if hit1:
    query_text = make_text(hit1["sample_token"], hit1["camera"])
    print(f"\n{'='*60}")
    print(f"QUERY TEXT (used for retrieval):")
    print(f"  {query_text}")

# Save metadata
meta = {
    "query": {"text": make_text(hit1["sample_token"], hit1["camera"])} if hit1 else {},
    "results": saved
}
with open(os.path.join(OUTDIR, "teaser_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"\nMetadata saved to {OUTDIR}/teaser_metadata.json")
print("Done!")
PYEOF
