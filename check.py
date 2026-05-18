import json
with open("camera_aware_captions.json", "r") as f:
    data = json.load(f)
first_key = list(data.keys())[0:2]
print(data[first_key].keys())