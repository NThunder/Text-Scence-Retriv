#!/usr/bin/env python3
"""Score text-to-scene retrieval from cached embeddings.

Reads the ``.pth`` embedding dicts that the encode/validate scripts already
write, so no model and no nuScenes devkit are needed. Emits the aggregate
metrics the paper reports plus a per-query dump of (rank, |Rel|) that the
difficulty-normalised metrics are computed from.

The pair set is the intersection of the text and image embedding keys, which
is exactly the set the evaluation scripts rank over.
"""
import argparse
import json
import math
from collections import Counter, defaultdict

import numpy as np
import torch

CAMERAS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text_emb", required=True, help="{token: {cam: tensor}} .pth")
    p.add_argument("--image_emb", required=True, help="{token: {cam: tensor}} .pth")
    p.add_argument("--relevance_captions", required=True,
                   help="caption JSON defining A(v); the Ly of Lx->Ly")
    p.add_argument("--relevance_mode", default="all", choices=["all", "any", "graded"],
                   help="all: A(q) subset of A(v); any: non-empty overlap; "
                        "graded: |A(q) & A(v)|/|A(q)| >= --tau")
    p.add_argument("--tau", type=float, default=1.0,
                   help="threshold for --relevance_mode graded")
    p.add_argument("--save_per_query", default=None,
                   help="write per-query (rank, |Rel|) as JSON")
    p.add_argument("--label", default="", help="tag printed with the results")
    return p.parse_args()


def build_pairs(text_embs, image_embs):
    """Ordered pair list = keys present in both embedding dicts."""
    pairs, txt, img = [], [], []
    for token in sorted(text_embs):
        if token not in image_embs:
            continue
        for cam in CAMERAS:
            if cam in text_embs[token] and cam in image_embs[token]:
                pairs.append((token, cam))
                txt.append(text_embs[token][cam])
                img.append(image_embs[token][cam])
    t = torch.nn.functional.normalize(torch.stack(txt).float(), p=2, dim=1)
    v = torch.nn.functional.normalize(torch.stack(img).float(), p=2, dim=1)
    return pairs, t, v


def relevant_indices(query_attrs, posting, n, mode, tau):
    """Indices of candidates relevant to one query, via an inverted index.

    Avoids the O(N^2) subset scan the original evaluation used, which is what
    makes the full-validation-split protocol tractable.
    """
    if not query_attrs:
        return set()
    if mode == "all":
        lists = sorted((posting.get(a, set()) for a in query_attrs), key=len)
        out = set(lists[0])
        for s in lists[1:]:
            out &= s
            if not out:
                break
        return out
    if mode == "any":
        out = set()
        for a in query_attrs:
            out |= posting.get(a, set())
        return out
    counts = Counter()
    for a in query_attrs:
        counts.update(posting.get(a, ()))
    need = tau * len(query_attrs)
    return {i for i, c in counts.items() if c >= need - 1e-9}


def first_relevant_rank(sim_row, relevant):
    """Rank (1-based) of the highest-scoring relevant candidate.

    Ties resolve optimistically; with float embeddings exact ties do not occur
    in practice, so this matches the argsort-based original.
    """
    idx = torch.tensor(sorted(relevant), dtype=torch.long)
    best = sim_row[idx].max()
    return int((sim_row > best).sum().item()) + 1


def chance_recall_at_k(rel_sizes, n, k):
    """Mean over queries of P(a relevant item lands in the top k) at random.

    Computed per query, because 1-(1-R/N)^k is concave in R: using the mean
    |Rel| instead overestimates the baseline (by ~22% at k=10 on our spread).
    """
    vals = []
    for r in rel_sizes:
        p = 1.0
        for j in range(k):
            num = n - r - j
            if num <= 0:
                p = 0.0
                break
            p *= num / (n - j)
        vals.append(1.0 - p)
    return float(np.mean(vals))


def main():
    args = parse_args()
    text_embs = torch.load(args.text_emb, map_location="cpu")
    image_embs = torch.load(args.image_emb, map_location="cpu")
    relevance = json.load(open(args.relevance_captions))

    pairs, t, v = build_pairs(text_embs, image_embs)
    n = len(pairs)

    attrs_by_idx, posting = [], defaultdict(set)
    for i, (token, cam) in enumerate(pairs):
        a = set(relevance.get(token, {}).get(cam, []))
        attrs_by_idx.append(a)
        for x in a:
            posting[x].add(i)

    sim = t @ v.T

    ranks, rel_sizes = [], []
    for i in range(n):
        rel = relevant_indices(attrs_by_idx[i], posting, n, args.relevance_mode, args.tau)
        rel_sizes.append(len(rel))
        ranks.append(first_relevant_rank(sim[i], rel) if rel else n)

    ranks = np.array(ranks, dtype=float)
    rel_sizes = np.array(rel_sizes, dtype=float)

    # rho: observed rank over the rank expected by chance for that same query
    rho = ranks * (rel_sizes + 1.0) / (n + 1.0)

    sim_tt = t @ t.T
    mask = ~torch.eye(n, dtype=torch.bool)
    inter_q = float(sim_tt[mask].mean())

    out = {
        "label": args.label,
        "pairs": n,
        "relevance_mode": args.relevance_mode,
        "tau": args.tau if args.relevance_mode == "graded" else None,
        "R@1": float((ranks <= 1).mean()),
        "R@5": float((ranks <= 5).mean()),
        "R@10": float((ranks <= 10).mean()),
        "MRR": float((1.0 / ranks).mean()),
        "median_rank": float(np.median(ranks)),
        "query_attrs_mean": float(np.mean([len(a) for a in attrs_by_idx])),
        "rel_set_mean": float(rel_sizes.mean()),
        "rel_set_std": float(rel_sizes.std()),
        "rel_set_min": int(rel_sizes.min()),
        "rel_set_max": int(rel_sizes.max()),
        "inter_query_cos": inter_q,
        "median_rho": float(np.median(rho)),
        "frac_better_than_chance": float((rho < 1).mean()),
    }
    for k in (1, 5, 10):
        c = chance_recall_at_k(rel_sizes, n, k)
        out[f"chance_R@{k}"] = c
        out[f"norm_R@{k}"] = (out[f"R@{k}"] - c) / (1.0 - c) if c < 1 else float("nan")

    width = max(len(k) for k in out)
    print(f"=== {args.label or args.text_emb} ===")
    for k, val in out.items():
        if k == "label":
            continue
        print(f"  {k:<{width}}  {val}")

    if args.save_per_query:
        with open(args.save_per_query, "w") as f:
            json.dump({
                "pairs": n,
                "relevance_mode": args.relevance_mode,
                "queries": [
                    {"sample_token": tok, "camera": cam,
                     "rank": int(r), "rel_size": int(s)}
                    for (tok, cam), r, s in zip(pairs, ranks, rel_sizes)
                ],
            }, f)
        print(f"  per-query dump -> {args.save_per_query}")


if __name__ == "__main__":
    main()
