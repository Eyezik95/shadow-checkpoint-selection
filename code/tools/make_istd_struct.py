"""Build symlinked ISTD views for FDRNet's SBUDataset layout (ShadowImages/ShadowMasks).

train_heldout / val_heldout reproduce the scene-disjoint split of the paper's fair protocol
(scripts/obj2_retrain_siran.py: scene = filename prefix before '-', val_frac 0.10, split seed 0)
and are checked against the val_scenes recorded in the paper's result files. test is the full ISTD
test partition. Nothing is copied; the PhD project is only read.
"""
import json, random, sys
from pathlib import Path

ISTD = Path(sys.argv[1]); OUT = Path(sys.argv[2]); RECORDED = Path(sys.argv[3])

def scene_of(p): return p.stem.split("-")[0]

def split_scenes(paths, val_frac, seed):  # identical to obj2_retrain_siran.split_scenes
    scenes = sorted({scene_of(p) for p in paths}, key=lambda s: (len(s), s))
    rng = random.Random(seed)
    val = set(rng.sample(scenes, max(1, round(val_frac * len(scenes)))))
    return [p for p in paths if scene_of(p) not in val], [p for p in paths if scene_of(p) in val], sorted(val)

train = sorted((ISTD / "train/train_A").glob("*.png"))
tr, va, val_scenes = split_scenes(train, 0.10, 0)
recorded = json.loads(RECORDED.read_text())["config"]["val_scenes"]
assert val_scenes == recorded, (val_scenes, recorded)
assert not ({scene_of(p) for p in tr} & {scene_of(p) for p in va}), "scene leak between train and val"
test = sorted((ISTD / "test/test_A").glob("*.png"))

def link(split, files, mask_dir):
    for sub in ("ShadowImages", "ShadowMasks"):
        (OUT / split / sub).mkdir(parents=True, exist_ok=False)
    for p in files:
        m = mask_dir / p.name
        assert m.exists(), m
        (OUT / split / "ShadowImages" / p.name).symlink_to(p)
        (OUT / split / "ShadowMasks" / p.name).symlink_to(m)
    return len(files)

counts = {"train_heldout": link("train_heldout", tr, ISTD / "train/train_B"),
          "val_heldout": link("val_heldout", va, ISTD / "train/train_B"),
          "test": link("test", test, ISTD / "test/test_B")}
(OUT / "split.json").write_text(json.dumps({"counts": counts, "val_scenes": val_scenes, "rule": "scene = stem before '-', val_frac 0.10, random.Random(0)"}, indent=1))
print(counts, val_scenes)
