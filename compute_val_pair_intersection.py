import json
from collections import defaultdict

CAMERAS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

thresholds = [0, 1, 5, 20]
all_pairs = {}  # threshold -> set of (sample_token, cam)

for p in thresholds:
    path = f"./camera_aware_captions_L0_p{p}.json"
    print(f"Loading {path}...")
    with open(path, "r") as f:
        captions = json.load(f)

    pairs = set()
    for sample_token, cams in captions.items():
        for cam in CAMERAS:
            if cam in cams and cams[cam]:
                pairs.add((sample_token, cam))
    all_pairs[p] = pairs
    print(f"  Threshold {p}: {len(pairs)} pairs")

# Intersection across all thresholds
intersection = all_pairs[0] & all_pairs[1] & all_pairs[5] & all_pairs[20]
intersection = sorted(intersection)  # sort for reproducibility

print(f"\nIntersection across all thresholds: {len(intersection)} pairs")
print(f"  Per threshold coverage:")
for p in thresholds:
    kept = sum(1 for pair in intersection if pair in all_pairs[p])
    print(f"    p={p}: {kept}/{len(intersection)} ({100*kept/len(intersection):.1f}%)")

# Save
output = "./val_pairs_intersection.json"
with open(output, "w") as f:
    json.dump(intersection, f, indent=2)
print(f"\nSaved {len(intersection)} pairs to {output}")
