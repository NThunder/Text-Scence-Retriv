"""
make_qualitative_examples.py

Generates real examples for the qualitative figure (Figure 6, fig:examples):
top-1 retrieval of the fine-tuned GME at a fixed caption level (default L2),
with a few SUCCESS cases (top-1 attribute match) and a few informative FAILURE
cases (top-1 has the right object class but the wrong scene/place, so it is
NOT relevant under the strict criterion).

Same conventions (model, embeddings, image paths, relevance) as
validate_gme_vlm.py / make_teaser_retrieval.py. Relevance is the strict paper
criterion with the empty-guard: a candidate is relevant iff the query's L0
attributes are a non-empty subset of the candidate's L0 attributes.

Outputs into --out_dir:
  succ{k}_query.png, succ{k}_ret.png   (success examples)
  fail{k}_query.png, fail{k}_ret.png   (failure examples)
  qual_meta.json                       (query/retrieved text, attrs, relevant, rank)

Example (server):
    conda activate gme_finetune
    python retriv/make_qualitative_examples.py \
        --model_name ./gme_finetuned_L3_p5_v2/best_model \
        --captions_pattern "./retriv/camera_aware_captions_{level}_p5.json" \
        --level L2 \
        --relevance_captions_path "./retriv/camera_aware_captions_L0_p5.json" \
        --val_samples_per_scene 4 \
        --batch_size 2 \
        --load_image_embs "./results_4perscene/img_cache/gme_ftL3.pth" \
        --n_success 2 --n_fail 1 --max_attrs 3 \
        --out_dir ./results_4perscene/qual_frames
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("TEASER_GPU", "2"))

import argparse
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from PIL import Image
from tqdm import tqdm

CAMERAS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

# object classes used to detect "right object, wrong scene" failures
CLASS_KEYWORDS = [
    "construction vehicle", "trailer", "truck", "bus", "car",
    "motorcycle", "bicycle", "bike", "pedestrian", "adult", "child",
    "person", "barrier", "traffic cone", "cone", "animal", "police",
    "ambulance", "emergency", "trash can", "stroller", "wheelchair",
    "debris", "pushable", "movable",
]


def classes_in(attr_list):
    text = " ".join(attr_list).lower()
    return {k for k in CLASS_KEYWORDS if k in text}


def parse_args():
    p = argparse.ArgumentParser(description="Generate qualitative retrieval examples (Figure 6)")
    p.add_argument("--dataroot", type=str,
                   default="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval")
    p.add_argument("--model_name", type=str, default="NCSOFT/GME-VARCO-VISION-Embedding")
    p.add_argument("--captions_pattern", type=str,
                   default="./retriv/camera_aware_captions_{level}_p5.json")
    p.add_argument("--level", type=str, default="L2", help="Caption level for the query text")
    p.add_argument("--relevance_captions_path", type=str, default=None,
                   help="Captions used for relevance (default: the L0 level file)")
    p.add_argument("--filter_pairs", type=str, default=None)
    p.add_argument("--load_image_embs", type=str, default=None)
    p.add_argument("--save_image_embs", type=str, default=None)
    p.add_argument("--val_samples_per_scene", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_success", type=int, default=6,
                   help="How many success candidates to save for browsing")
    p.add_argument("--n_fail", type=int, default=6,
                   help="How many failure candidates to save for browsing")
    p.add_argument("--max_attrs", type=int, default=3,
                   help="Only use queries with at most this many L0 attributes "
                        "(keeps the overlaid caption short)")
    p.add_argument("--min_self_sim", type=float, default=0.0,
                   help="Min cosine(query text, its OWN image): higher = the "
                        "caption clearly matches the query frame (good GT)")
    p.add_argument("--exclude_tokens", type=str, nargs="*", default=[],
                   help="Query sample_tokens to skip (e.g. to re-roll past bad examples)")
    p.add_argument("--rank_cap", type=int, default=50,
                   help="Cap when searching the rank of the first relevant frame")
    p.add_argument("--out_dir", type=str, default="./qual_frames")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    L0 = "L0"
    cap_query = args.captions_pattern.format(level=args.level)
    rel_path = args.relevance_captions_path or args.captions_pattern.format(level=L0)

    # ---- model ----
    print(f"Loading model: {args.model_name}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="cuda:0")
    processor = AutoProcessor.from_pretrained(args.model_name)
    tokenizer = processor.tokenizer
    device = model.device
    img_msg = [{"role": "user", "content": [{"type": "image", "image": "image"}]}]
    img_txt = processor.apply_chat_template(img_msg, tokenize=False, add_generation_prompt=True) \
        + tokenizer.eos_token

    @torch.inference_mode()
    def encode_texts(texts):
        msgs = []
        for t in texts:
            m = [{"role": "user", "content": [{"type": "text", "text": t}]}]
            msgs.append(processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                        + tokenizer.eos_token)
        inp = processor(text=msgs, padding=True, return_tensors="pt").to(device)
        out = model(**inp, output_hidden_states=True, return_dict=True)
        return F.normalize(out.hidden_states[-1][:, -1, :], dim=-1).cpu()

    @torch.inference_mode()
    def encode_images(paths):
        images = [Image.open(p).convert("RGB") for p in paths]
        inp = processor(text=[img_txt] * len(images), images=images,
                        padding=True, return_tensors="pt").to(device)
        out = model(**inp, output_hidden_states=True, return_dict=True)
        return F.normalize(out.hidden_states[-1][:, -1, :], dim=-1).cpu()

    # ---- captions ----
    with open(cap_query) as f:
        qcaps = json.load(f)
    with open(rel_path) as f:
        rel_caps = json.load(f)
    print(f"Query captions ({args.level}): {cap_query}")
    print(f"Relevance captions (L0): {rel_path}")

    # ---- nuScenes + gallery ----
    print("Loading nuScenes...")
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.dataroot, verbose=False)

    def img_path(token, cam):
        sample = nusc.get("sample", token)
        sd = nusc.get("sample_data", sample["data"][cam])
        return os.path.join(args.dataroot, sd["filename"])

    if args.filter_pairs:
        with open(args.filter_pairs) as f:
            gallery = [tuple(p) for p in json.load(f)]
    else:
        val_scenes = set(create_splits_scenes()["val"])
        val_tokens = []
        for scene in nusc.scene:
            if scene["name"] not in val_scenes:
                continue
            toks, cur = [], scene["first_sample_token"]
            while cur:
                toks.append(cur)
                cur = nusc.get("sample", cur)["next"]
            if args.val_samples_per_scene > 0:
                step = max(1, len(toks) // args.val_samples_per_scene)
                toks = toks[::step][:args.val_samples_per_scene]
            val_tokens.extend(toks)
        gallery = [(t, c) for t in val_tokens if t in rel_caps
                   for c in CAMERAS if rel_caps.get(t, {}).get(c)]
    print(f"Gallery (pre image-emb check): {len(gallery)} pairs")

    # ---- gallery image embeddings ----
    if args.load_image_embs:
        bank = torch.load(args.load_image_embs, map_location="cpu")
        gallery = [(t, c) for (t, c) in gallery if t in bank and c in bank[t]]
        img_embs = torch.stack([bank[t][c] for (t, c) in gallery]).float()
    else:
        embs, bank = [], defaultdict(dict)
        for i in tqdm(range(0, len(gallery), args.batch_size), desc="Encoding images"):
            chunk = gallery[i:i + args.batch_size]
            e = encode_images([img_path(t, c) for (t, c) in chunk])
            for j, (t, c) in enumerate(chunk):
                bank[t][c] = e[j]
            embs.append(e)
        img_embs = torch.cat(embs).float()
        if args.save_image_embs:
            torch.save({t: dict(d) for t, d in bank.items()}, args.save_image_embs)
    img_embs = F.normalize(img_embs, p=2, dim=1)
    print(f"Gallery: {len(gallery)} pairs")

    # L0 attribute sets per gallery index
    attrs_by_idx = [set(rel_caps.get(t, {}).get(c, [])) for (t, c) in gallery]

    # ---- query texts at the chosen level for every gallery pair ----
    def qtext(t, c):
        parts = qcaps.get(t, {}).get(c, [])
        return "The camera view contains: " + "; ".join(parts) if parts else None

    valid = [i for i, (t, c) in enumerate(gallery) if qtext(t, c) is not None]
    texts = [qtext(*gallery[i]) for i in valid]
    print(f"Encoding {len(texts)} query texts at level {args.level}...")
    chunks = []
    for i in tqdm(range(0, len(texts), args.batch_size)):
        chunks.append(encode_texts(texts[i:i + args.batch_size]))
    qembs = F.normalize(torch.cat(chunks).float(), p=2, dim=1)

    sims = qembs @ img_embs.T  # [Nq, Ngallery]

    # ---- score every query: top-1 + relevance + rank ----
    def first_rel_rank(order, relset):
        for k, ix in enumerate(order[:args.rank_cap]):
            if ix in relset:
                return k + 1
        return args.rank_cap + 1  # ">cap"

    row_of = {gi: r for r, gi in enumerate(valid)}   # gallery index -> sims row
    exclude = set(args.exclude_tokens)

    successes, failures = [], []
    for row, i in enumerate(tqdm(valid, desc="Scoring")):
        t, c = gallery[i]
        if t in exclude:
            continue
        qattrs = attrs_by_idx[i]
        if not (1 <= len(qattrs) <= args.max_attrs):
            continue
        # how well the caption matches its OWN image (GT text <-> GT image)
        q_self = float(sims[row, i])
        if q_self < args.min_self_sim:
            continue
        order = torch.argsort(sims[row], descending=True).tolist()
        top1 = order[0]
        rtok, rcam = gallery[top1]
        is_rel = bool(qattrs) and qattrs.issubset(attrs_by_idx[top1])
        # alignment of the retrieved frame with its own caption
        ret_self = float(sims[row_of[top1], top1]) if top1 in row_of else 0.0
        rec = {
            "level": args.level,
            "query_token": t, "query_camera": c,
            "query_text": qtext(t, c),
            "query_L0_attributes": sorted(qattrs),
            "query_self_sim": round(q_self, 4),
            "retrieved_token": rtok, "retrieved_camera": rcam,
            "retrieved_L0_attributes": sorted(attrs_by_idx[top1]),
            "retrieved_self_sim": round(ret_self, 4),
            "relevant": is_rel,
        }
        if is_rel:
            successes.append(rec)
        else:
            shared = classes_in(rec["query_L0_attributes"]) & classes_in(rec["retrieved_L0_attributes"])
            if shared:  # informative failure: same object class, wrong scene
                rec["shared_classes"] = sorted(shared)
                relset = {j for j, a in enumerate(attrs_by_idx) if qattrs.issubset(a)}
                rec["first_relevant_rank"] = first_rel_rank(order, relset)
                failures.append(rec)

    # Prefer examples whose captions clearly match their frames (high self-sim),
    # so the GT text agrees with the GT image. Successes: both sides must align;
    # failures: at least the query side must align (so the miss is real, not a
    # mislabeled query).
    successes.sort(key=lambda r: -(r["query_self_sim"] + r["retrieved_self_sim"]))
    failures.sort(key=lambda r: -r["query_self_sim"])
    chosen = ([("succ", r) for r in successes[:args.n_success]]
              + [("fail", r) for r in failures[:args.n_fail]])
    if len(successes) < args.n_success or len(failures) < args.n_fail:
        print(f"WARNING: found {len(successes)} successes, {len(failures)} failures "
              f"(requested {args.n_success}/{args.n_fail}). Relax --max_attrs if too few.")

    # ---- save a pool of candidates + meta (browse, then pick) ----
    meta = []
    counters = defaultdict(int)
    for kind, r in chosen:
        counters[kind] += 1
        k = counters[kind]
        cid = f"{kind}{k:02d}"
        Image.open(img_path(r["query_token"], r["query_camera"])).convert("RGB") \
            .save(os.path.join(args.out_dir, f"{cid}_query.png"))
        Image.open(img_path(r["retrieved_token"], r["retrieved_camera"])).convert("RGB") \
            .save(os.path.join(args.out_dir, f"{cid}_ret.png"))
        r = dict(r)
        r["id"] = cid
        r["kind"] = kind
        r["query_file"] = f"{cid}_query.png"
        r["ret_file"] = f"{cid}_ret.png"
        meta.append(r)
        flag = "OK" if r["relevant"] else "x"
        print(f"  [{cid}] {flag} self(q={r['query_self_sim']}, ret={r['retrieved_self_sim']})  "
              f"q='{'; '.join(r['query_L0_attributes'])}'  "
              f"-> ret='{'; '.join(r['retrieved_L0_attributes'])}'"
              + ("" if r["relevant"] else f"  (rank {r.get('first_relevant_rank')})"))

    with open(os.path.join(args.out_dir, "qual_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nDone. {len(meta)} candidates + qual_meta.json in {args.out_dir}/")
    print("Browse them (or render a contact sheet with make_qualitative_contact.py),"
          " then build the final figure from the chosen ids.")


if __name__ == "__main__":
    main()
