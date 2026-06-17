"""Dataset for the continual-learning setting — matches the CIDM/CIFC training data exactly.

Each concept (= one CL task) is trained on the SAME per-image captions as CIDM
(`caption/<concept>/<stem>.txt`), with the class word replaced the same way CIDM does via
`replace_mapping` — except instead of CIDM's learned pseudo-tokens we substitute our identifier
phrase `V<k> <class>` (e.g. `dog -> V1 dog`, `duck toy -> V2 duck toy`, `painting -> V6 painting`).
So the images AND captions are identical to CIDM; only the conditioning mechanism differs.

If a concept has no `caption_dir`, the fixed `prompt` is used for every image (fallback).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class ConceptSpec:
    concept_id: str
    images_dir: str
    class_word: str
    identifier: str                       # e.g. "V1"
    caption_dir: Optional[str] = None      # CIDM per-image captions; None -> use `prompt`
    prompt: Optional[str] = None           # fallback + canonical diagnostic prompt
    category: Optional[str] = None         # eval category (pet/plushy/style), carried downstream

    @property
    def replacement(self) -> str:
        return f"{self.identifier} {self.class_word}"        # "V1 dog"

    @property
    def diag_prompt(self) -> str:
        return self.prompt or f"a photo of {self.replacement}"


def _load_image(path: str, resolution: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    img = img.resize((resolution, resolution), Image.BICUBIC)
    arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return arr * 2.0 - 1.0  # [-1,1], [3,H,W]


class ConceptDataset(Dataset):
    """One concept's images, each paired with its CIDM caption (class word -> 'V<k> <class>')."""

    def __init__(self, spec: ConceptSpec, resolution: int = 512):
        self.spec = spec
        self.resolution = resolution
        self.paths = sorted(
            p for p in glob.glob(os.path.join(spec.images_dir, "*")) if p.lower().endswith(_IMG_EXTS)
        )
        if not self.paths:
            raise FileNotFoundError(f"no images for concept '{spec.concept_id}' in {spec.images_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def _caption(self, stem: str) -> str:
        if self.spec.caption_dir:
            cap_path = os.path.join(self.spec.caption_dir, stem + ".txt")
            if os.path.exists(cap_path):
                with open(cap_path) as f:
                    cap = f.read().strip()
                repl, cls = self.spec.replacement, self.spec.class_word
                # CIDM-style: replace the class word with the identifier phrase; if the class word
                # is absent, prepend the phrase so the identifier is always present.
                return cap.replace(cls, repl) if cls in cap else f"{repl}, {cap}"
        return self.spec.diag_prompt

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx % len(self.paths)]
        stem = os.path.splitext(os.path.basename(path))[0]
        return {"pixel_values": _load_image(path, self.resolution), "caption": self._caption(stem)}


def collate_fn(batch: List[dict]) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch], 0),
        "captions": [b["caption"] for b in batch],
    }


def specs_from_config(concept_cfgs: List[dict]) -> List[ConceptSpec]:
    """Each entry: {concept_id, images_dir, class_word, [caption_dir], [category], [prompt]}.
    The identifier is V<k> by task order (1-based), matching the eval substitution in gen_cifc."""
    specs = []
    for i, c in enumerate(concept_cfgs):
        specs.append(ConceptSpec(
            concept_id=c["concept_id"],
            images_dir=c["images_dir"],
            class_word=c.get("class_word", c["concept_id"]),
            identifier=c.get("identifier", f"V{i + 1}"),
            caption_dir=c.get("caption_dir"),
            prompt=c.get("prompt"),
            category=c.get("category"),
        ))
    return specs
