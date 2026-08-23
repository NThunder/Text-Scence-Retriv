"""Attribute-based retrieval metrics, shared by the training scripts.

Mirrors what ``eval_clip_baseline_per_camera.py`` and ``validate_gme_vlm.py``
report: strict "all" relevance (A(q) subset of A(v)) with temporal neighbours
disabled, scored text -> image. Keeping one implementation stops the training
criterion from drifting away from the metric the paper reports.
"""
from collections import defaultdict

import numpy as np
import torch


def attribute_retrieval_metrics(sim_t2i, pairs, captions):
    """Metrics for a text->image similarity matrix.

    Args:
        sim_t2i: [N, N] tensor; row i is text query i scored against all images.
                 Note the orientation - an image->text matrix must be
                 transposed first, since A(q) subset of A(v) is not symmetric.
        pairs: list of (sample_token, camera), same order as the matrix.
        captions: {sample_token: {camera: [description, ...]}}.

    Returns:
        dict with R@1, R@5, R@10, MRR, median_rank and mean relevant-set size.
    """
    n = len(pairs)
    if n == 0:
        return {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "MRR": 0.0,
                "median_rank": 0.0, "rel_set_mean": 0.0}

    attrs_by_idx, posting = [], defaultdict(set)
    for idx, (token, cam) in enumerate(pairs):
        attrs = set(captions.get(token, {}).get(cam, []))
        attrs_by_idx.append(attrs)
        for a in attrs:
            posting[a].add(idx)

    ranks, rel_sizes = [], []
    for i in range(n):
        query = attrs_by_idx[i]
        if query:
            # intersect the shortest posting lists first
            lists = sorted((posting.get(a, set()) for a in query), key=len)
            relevant = set(lists[0])
            for s in lists[1:]:
                relevant &= s
                if not relevant:
                    break
        else:
            relevant = set()
        rel_sizes.append(len(relevant))

        if relevant:
            idx_t = torch.tensor(sorted(relevant), dtype=torch.long)
            best = sim_t2i[i][idx_t].max()
            ranks.append(int((sim_t2i[i] > best).sum().item()) + 1)
        else:
            ranks.append(n)

    ranks = np.asarray(ranks, dtype=float)
    return {
        "R@1": float((ranks <= 1).mean()),
        "R@5": float((ranks <= 5).mean()),
        "R@10": float((ranks <= 10).mean()),
        "MRR": float((1.0 / ranks).mean()),
        "median_rank": float(np.median(ranks)),
        "rel_set_mean": float(np.mean(rel_sizes)),
    }
