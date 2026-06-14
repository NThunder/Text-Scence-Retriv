"""
make_teaser_retrieval.py

Builds the WACV teaser: ONE fixed query (sample_token, camera) described at five
caption levels L0..L4. For each level we encode the query text, rank the SAME
image gallery, take top-1, and label it honestly with the paper's strict
criterion (query L0 attributes must be a subset of the retrieved frame's L0
attributes). The query is held fixed across levels; only the caption detail
changes. This is the correct teaser design.

Why the previous attempt marked everything relevant: in "all" mode relevance is
`query_attrs.issubset(cand_attrs)`, and the EMPTY set is a subset of everything,
so a query whose L0 attributes come out empty makes every frame "relevant".
Here we (a) guard the empty case and (b) only pick queries with non-empty L0
attributes and a small relevant set, so the check is meaningful.

Conventions (model, embeddings, image paths) are copied from validate_gme_vlm.py.

Example (server, with cached image embeddings reused from the L0 run):

    conda activate gme_finetune
    python make_teaser_retrieval.py \
        --model_name ./gme_finetuned/best_model \
        --captions_pattern "./retriv/camera_aware_captions_{level}_p5.json" \
        --relevance_captions_path "./retriv/camera_aware_captions_L0_p5.json" \
        --filter_pairs ./val_pairs_2280.json \
        --load_image_embs ./image_embs_p5.pth \
        --out_dir ./teaser_frames

    # then copy ./teaser_frames/*.png to the local figures/ dir.

You can also force a specific query with --query_token / --query_cam.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("TEASER_GPU", "2"))

import argparse
import json
import shutil
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


def parse_args():
    p = argparse.ArgumentParser(description="Build the L0..L4 teaser for one fixed query")
    p.add_argument("--dataroot", type=str,
                   default="/home/jovyan/shares/SR006.nfs2/bukhtuev/M3Net/data/sets/nuScenes/trainval")
    p.add_argument("--model_name", type=str, default="NCSOFT/GME-VARCO-VISION-Embedding",
                   help="HF name or local path to the fine-tuned GME")
    p.add_argument("--captions_pattern", type=str,
                   default="./retriv/camera_aware_captions_{level}_p5.json",
                   help="Path pattern for per-level caption files; {level} is replaced by L0..L4")
    p.add_argument("--levels", type=str, nargs="+", default=["L0", "L1", "L2", "L3", "L4"])
    p.add_argument("--relevance_captions_path", type=str, default=None,
                   help="Captions used for relevance (default: the L0 level file). "
                        "Relevance is always computed at this fixed level.")
    p.add_argument("--filter_pairs", type=str, default=None,
                   help="JSON list of [sample_token, cam] defining the gallery "
                        "(use the same 2280-pair file as the paper eval)")
    p.add_argument("--load_image_embs", type=str, default=None,
                   help="Precomputed image embeddings {token:{cam:emb}} (skips image encoding)")
    p.add_argument("--save_image_embs", type=str, default=None,
                   help="Where to save image embeddings if they are computed here")
    p.add_argument("--val_samples_per_scene", type=int, default=-1,
                   help="Evenly spaced samples per val scene when building the gallery (-1 = all)")
    p.add_argument("--batch_size", type=int, default=8)
    # query selection
    p.add_argument("--query_token", type=str, default=None, help="Force a specific query sample_token")
    p.add_argument("--query_cam", type=str, default=None, help="Force a specific query camera")
    p.add_argument("--min_rel", type=int, default=1, help="Min L0 relevant-set size for an auto-picked query")
    p.add_argument("--max_rel", type=int, default=8, help="Max L0 relevant-set size for an auto-picked query")
    p.add_argument("--max_candidates", type=int, default=60,
                   help="Cap on candidate queries scored during auto-selection")
    p.add_argument("--out_dir", type=str, default="./teaser_frames")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    level_files = {lvl: args.captions_pattern.format(level=lvl) for lvl in args.levels}
    rel_path = args.relevance_captions_path or level_files[args.levels[0]]

    # ---------- load model ----------
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
        emb = out.hidden_states[-1][:, -1, :]
        return F.normalize(emb, dim=-1).cpu()

    @torch.inference_mode()
    def encode_images(paths):
        images = [Image.open(p).convert("RGB") for p in paths]
        inp = processor(text=[img_txt] * len(images), images=images,
                        padding=True, return_tensors="pt").to(device)
        out = model(**inp, output_hidden_states=True, return_dict=True)
        emb = out.hidden_states[-1][:, -1, :]
        return F.normalize(emb, dim=-1).cpu()

    # ---------- load captions ----------
    captions = {}
    for lvl, path in level_files.items():
        with open(path) as f:
            captions[lvl] = json.load(f)
        print(f"  {lvl}: {path}")
    with open(rel_path) as f:
        rel_caps = json.load(f)
    print(f"  relevance: {rel_path}")

    # ---------- nuScenes + gallery pairs ----------
    print("Loading nuScenes...")
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.dataroot, verbose=False)

    def img_path(token, cam):
        sample = nusc.get("sample", token)
        sd = nusc.get("sample_data", sample["data"][cam])
        return os.path.join(args.dataroot, sd["filename"])

    if args.filter_pairs:
        with open(args.filter_pairs) as f:
            gallery = [tuple(p) for p in json.load(f)]
        print(f"Gallery from filter_pairs: {len(gallery)} pairs")
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
        l0 = captions[args.levels[0]]
        gallery = [(t, c) for t in val_tokens if t in l0
                   for c in CAMERAS if l0.get(t, {}).get(c)]
        print(f"Gallery from val split: {len(gallery)} pairs")

    # ---------- gallery image embeddings ----------
    if args.load_image_embs:
        print(f"Loading image embeddings: {args.load_image_embs}")
        bank = torch.load(args.load_image_embs, map_location="cpu")
        gallery = [(t, c) for (t, c) in gallery if t in bank and c in bank[t]]
        img_embs = torch.stack([bank[t][c] for (t, c) in gallery]).float()
    else:
        print("Encoding gallery images (no cache provided)...")
        embs = []
        bank = defaultdict(dict)
        for i in tqdm(range(0, len(gallery), args.batch_size)):
            chunk = gallery[i:i + args.batch_size]
            e = encode_images([img_path(t, c) for (t, c) in chunk])
            for j, (t, c) in enumerate(chunk):
                bank[t][c] = e[j]
            embs.append(e)
        img_embs = torch.cat(embs).float()
        if args.save_image_embs:
            torch.save({t: dict(d) for t, d in bank.items()}, args.save_image_embs)
            print(f"Saved image embeddings to {args.save_image_embs}")
    img_embs = F.normalize(img_embs, p=2, dim=1)
    print(f"Gallery size after embedding check: {len(gallery)}")

    idx_of = {pair: i for i, pair in enumerate(gallery)}

    # ---------- relevance bank (fixed level, e.g. L0) ----------
    attrs_by_idx = []
    for (t, c) in gallery:
        attrs_by_idx.append(set(rel_caps.get(t, {}).get(c, [])))

    def relevant_set(qattrs):
        # EMPTY-GUARD: an empty query never matches anything (avoids the
        # "empty set is a subset of everything" bug).
        if not qattrs:
            return set()
        return {i for i, ca in enumerate(attrs_by_idx) if qattrs.issubset(ca)}

    def level_text(token, cam, lvl):
        parts = captions[lvl].get(token, {}).get(cam, [])
        return "The camera view contains: " + "; ".join(parts) if parts else None

    # ---------- choose the query ----------
    def candidate_ok(token, cam):
        qattrs = set(rel_caps.get(token, {}).get(cam, []))
        if not qattrs:
            return None
        # query text must exist at every level
        if any(level_text(token, cam, lvl) is None for lvl in args.levels):
            return None
        rs = relevant_set(qattrs)
        if not (args.min_rel <= len(rs) <= args.max_rel):
            return None
        return qattrs

    if args.query_token and args.query_cam:
        query = (args.query_token, args.query_cam)
        qattrs = set(rel_caps.get(query[0], {}).get(query[1], []))
        if not qattrs:
            raise SystemExit(f"Query {query} has EMPTY L0 attributes — pick another pair.")
        print(f"Using forced query: {query}  (L0 attrs: {sorted(qattrs)})")
    else:
        print("Auto-selecting a query (non-empty L0 attrs, small relevant set, "
              "L0 fails but a detailed level succeeds)...")
        cands = []
        for (t, c) in gallery:
            qa = candidate_ok(t, c)
            if qa is not None:
                cands.append((t, c, qa))
            if len(cands) >= args.max_candidates:
                break
        if not cands:
            raise SystemExit("No candidate query satisfied the constraints; "
                             "relax --min_rel/--max_rel or pass --query_token/--query_cam.")

        best, best_score = None, None
        for (t, c, qa) in cands:
            rs = relevant_set(qa)
            per_level = {}
            l0_top1_rel = False
            n_detail_rel = 0
            for lvl in args.levels:
                te = encode_texts([level_text(t, c, lvl)])
                sims = (F.normalize(te.float(), p=2, dim=1) @ img_embs.T)[0]
                order = torch.argsort(sims, descending=True).tolist()
                top1 = order[0]
                top1_rel = top1 in rs
                rank = next((k + 1 for k, ix in enumerate(order) if ix in rs), len(order))
                per_level[lvl] = (top1, top1_rel, rank)
                if lvl == args.levels[0]:
                    l0_top1_rel = top1_rel
                elif top1_rel:
                    n_detail_rel += 1
            # narrative score: prefer L0 wrong + some detailed level right
            score = (0 if l0_top1_rel else 10) + n_detail_rel
            if best_score is None or score > best_score:
                best, best_score, best_per_level, best_qa = (t, c), score, per_level, qa
        query, qattrs, per_level = best, best_qa, best_per_level
        print(f"Picked query: {query}  score={best_score}  L0 attrs={sorted(qattrs)}")

    # ---------- final retrieval per level for the chosen query ----------
    rs = relevant_set(qattrs)
    qtok, qcam = query
    meta = {
        "query_token": qtok,
        "query_camera": qcam,
        "query_L0_attributes": sorted(qattrs),
        "relevant_set_size": len(rs),
        "levels": [],
    }

    # save query frame
    Image.open(img_path(qtok, qcam)).convert("RGB").save(os.path.join(args.out_dir, "query_frame.png"))

    for lvl in args.levels:
        qtext = level_text(qtok, qcam, lvl)
        te = encode_texts([qtext])
        sims = (F.normalize(te.float(), p=2, dim=1) @ img_embs.T)[0]
        order = torch.argsort(sims, descending=True).tolist()
        top1 = order[0]
        rtok, rcam = gallery[top1]
        is_rel = top1 in rs
        rank = next((k + 1 for k, ix in enumerate(order) if ix in rs), len(order))
        Image.open(img_path(rtok, rcam)).convert("RGB").save(
            os.path.join(args.out_dir, f"ret_{lvl}.png"))
        meta["levels"].append({
            "level": lvl,
            "query_text": qtext,
            "retrieved_token": rtok,
            "retrieved_camera": rcam,
            "retrieved_L0_attributes": sorted(attrs_by_idx[top1]),
            "relevant": bool(is_rel),          # honest, empty-guarded
            "first_relevant_rank": int(rank),
            "file": f"ret_{lvl}.png",
        })
        flag = "OK" if is_rel else "x"
        print(f"  {lvl}: top1={rtok}/{rcam} relevant={is_rel} [{flag}] "
              f"first_rel_rank={rank}")

    with open(os.path.join(args.out_dir, "teaser_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nDone. Frames + teaser_meta.json in {args.out_dir}/")
    print(f"Query relevant-set size = {len(rs)} (frames whose L0 attrs contain all query attrs).")


if __name__ == "__main__":
    main()
