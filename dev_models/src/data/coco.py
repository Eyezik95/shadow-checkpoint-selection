"""Loader for the CVAT COCO exports of the University of Ibadan corpus (Section 3.2.3).

One JSON per sequence, named `<n>.json`. Two quirks of the actual exports are
handled here and must not be "simplified" away:

  * Category names differ between files ('person', 'pedestrian', 'Pedestrian',
    'shadow', 'Shadow') and in 12.json the ids are swapped (1 = Shadow,
    2 = Pedestrian). Classes are therefore resolved by name, never by id.
  * CVAT stores rotated boxes as the unrotated (x, y, w, h) plus a `rotation`
    attribute in degrees, rotated about the box centre. The centre c(s) used
    by Eq 3.1 is invariant to that rotation; the foot point f(p) is not, so
    rotated pedestrian boxes are counted and reported separately.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Box", "Frame", "Sequence", "load_sequence", "file_sha256"]

_CLASS_BY_NAME = {"person": "person", "pedestrian": "person", "shadow": "shadow"}


@dataclass(frozen=True)
class Box:
    ann_id: int
    track_id: int
    bbox: tuple[float, float, float, float]    # COCO (x, y, w, h), pixels
    keyframe: bool                             # False = CVAT-interpolated
    occluded: bool
    rotation: float                            # degrees about the box centre


@dataclass
class Frame:
    image_id: int
    file_name: str
    persons: list[Box] = field(default_factory=list)
    shadows: list[Box] = field(default_factory=list)


@dataclass
class Sequence:
    name: str
    path: Path
    sha256: str
    width: int
    height: int
    frames: list[Frame]
    category_map: dict[int, str]               # as found in the file


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_sequence(path: str | Path) -> Sequence:
    path = Path(path)
    data = json.loads(path.read_text())

    category_map = {}
    for c in data["categories"]:
        name = c["name"].strip().lower()
        if name not in _CLASS_BY_NAME:
            raise ValueError(f"{path.name}: unknown category {c['name']!r}")
        category_map[c["id"]] = _CLASS_BY_NAME[name]
    if sorted(category_map.values()) != ["person", "shadow"]:
        raise ValueError(f"{path.name}: expected one person and one shadow class, "
                         f"got {category_map}")

    sizes = {(im["width"], im["height"]) for im in data["images"]}
    if len(sizes) != 1:
        raise ValueError(f"{path.name}: mixed image sizes {sizes}")
    width, height = sizes.pop()

    frames = {im["id"]: Frame(im["id"], im["file_name"]) for im in data["images"]}
    for a in data["annotations"]:
        attrs = a.get("attributes", {})
        if "track_id" not in attrs:
            raise ValueError(f"{path.name}: annotation {a['id']} has no track_id")
        box = Box(
            ann_id=a["id"],
            track_id=int(attrs["track_id"]),
            bbox=tuple(float(v) for v in a["bbox"]),
            keyframe=bool(attrs.get("keyframe", False)),
            occluded=bool(attrs.get("occluded", False)),
            rotation=float(attrs.get("rotation", 0.0)),
        )
        cls = category_map[a["category_id"]]
        (frames[a["image_id"]].persons if cls == "person"
         else frames[a["image_id"]].shadows).append(box)

    return Sequence(
        name=path.stem, path=path, sha256=file_sha256(path),
        width=width, height=height,
        frames=sorted(frames.values(), key=lambda f: f.image_id),
        category_map=category_map,
    )
