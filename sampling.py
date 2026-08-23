"""Seeded, scene-level subsampling of the nuScenes training split.

The training scripts used to take a prefix of the item list built by walking
``nusc.scene`` in order. That is not a uniform subsample: on nuScenes the first
~60 training scenes are 70% singapore-onenorth and 0% queenstown/hollandvillage,
while the full split is 64% boston-seaport. Training therefore ran on a
different location mix than the validation split it was scored on.

Sampling happens at scene level, not item level: consecutive frames inside one
scene are highly correlated, so drawing items independently would overstate how
much of the split is actually covered.
"""
import random


def sample_scenes_uniformly(items_by_scene, budget, seed=0):
    """Take whole scenes in a seeded random order until ``budget`` items.

    Args:
        items_by_scene: {scene_name: [item, ...]} in any order.
        budget: maximum number of items; <= 0 or None means take everything.
        seed: RNG seed, so a run is reproducible from its arguments alone.

    Returns:
        (items, scene_names_used). The final scene is truncated so the item
        count matches ``budget`` exactly.
    """
    names = sorted(items_by_scene)
    if not budget or budget <= 0:
        return [x for n in names for x in items_by_scene[n]], names

    random.Random(seed).shuffle(names)
    items, used = [], []
    for name in names:
        if len(items) >= budget:
            break
        items.extend(items_by_scene[name])
        used.append(name)
    return items[:budget], used
