"""Custom-Diffusion-style metrics + a CIL forgetting metric.

* CLIP-T : text-alignment  = cos(CLIP image emb, CLIP text emb)         [higher better]
* CLIP-I : image-alignment = cos(CLIP gen emb, CLIP mean-reference emb)  [higher better]
* DINO   : DINOv2 image similarity to the reference set                  [higher better]
* Forgetting (CIL): for each old concept, how much its score dropped from its best (right
  after it was added) to its final value after later concepts were added. Lower is better.

CLIP uses HF transformers; DINO uses timm. Both are loaded lazily (eval only — not exercised
by the smoke test).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from PIL import Image


def _load_images(paths_or_dir) -> List[Image.Image]:
    if isinstance(paths_or_dir, str) and os.path.isdir(paths_or_dir):
        paths = sorted(glob.glob(os.path.join(paths_or_dir, "*")))
    else:
        paths = list(paths_or_dir)
    out = []
    for p in paths:
        if str(p).lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
            out.append(Image.open(p).convert("RGB"))
    return out


class CLIPScorer:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def image_embeds(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inp = self.proc(images=list(images), return_tensors="pt").to(self.device)
        e = self.model.get_image_features(**inp)
        return F.normalize(e, dim=-1)

    @torch.no_grad()
    def text_embeds(self, texts: Sequence[str]) -> torch.Tensor:
        inp = self.proc(text=list(texts), return_tensors="pt", padding=True, truncation=True).to(self.device)
        e = self.model.get_text_features(**inp)
        return F.normalize(e, dim=-1)

    def clip_t(self, images, texts) -> float:
        ie = self.image_embeds(images)
        te = self.text_embeds(texts if len(texts) == len(images) else texts * len(images))
        return float((ie * te).sum(-1).mean().item())

    def clip_i(self, gen, ref) -> float:
        ge = self.image_embeds(gen)
        re = self.image_embeds(ref).mean(0, keepdim=True)
        re = F.normalize(re, dim=-1)
        return float((ge * re).sum(-1).mean().item())


class DinoScorer:
    def __init__(self, model_name: str = "vit_small_patch14_dinov2.lvd142m", device: str = "cuda"):
        import timm
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0).to(self.device).eval()
        cfg = timm.data.resolve_model_data_config(self.model)
        self.tf = timm.data.create_transform(**cfg, is_training=False)

    @torch.no_grad()
    def _embeds(self, images: Sequence[Image.Image]) -> torch.Tensor:
        x = torch.stack([self.tf(im) for im in images]).to(self.device)
        return F.normalize(self.model(x), dim=-1)

    def dino_i(self, gen, ref) -> float:
        ge = self._embeds(gen)
        re = F.normalize(self._embeds(ref).mean(0, keepdim=True), dim=-1)
        return float((ge * re).sum(-1).mean().item())


def compute_forgetting(history: Dict[str, List[float]]) -> float:
    """history[concept] = scores measured after each stage that concept existed.

    Forgetting = mean over concepts of (best_seen - final). >0 means net degradation.
    """
    drops = []
    for _, scores in history.items():
        if len(scores) >= 2:
            drops.append(max(scores[:-1]) - scores[-1])
    return float(sum(drops) / len(drops)) if drops else 0.0


def parse_args():
    p = argparse.ArgumentParser(description="ContinualHyper evaluation")
    p.add_argument("--gen_dir", required=True, help="generated images for one concept")
    p.add_argument("--ref_dir", required=True, help="reference (training) images for that concept")
    p.add_argument("--prompt", required=True, help="text prompt used for CLIP-T")
    p.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    p.add_argument("--dino_model", default="vit_small_patch14_dinov2.lvd142m")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    gen = _load_images(args.gen_dir)
    ref = _load_images(args.ref_dir)
    clip = CLIPScorer(args.clip_model)
    dino = DinoScorer(args.dino_model)
    metrics = {
        "clip_t": clip.clip_t(gen, [args.prompt]),
        "clip_i": clip.clip_i(gen, ref),
        "dino_i": dino.dino_i(gen, ref),
        "n_gen": len(gen),
        "n_ref": len(ref),
    }
    print(json.dumps(metrics, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
