"""Segmentacja semantyczna konceptow CIFC: Grounded-SAM w transformers.
GroundingDINO (tekst -> ramka klasy) + SAM (ramka -> maska calej instancji).
Naprawia oba typy defektow isnet: 'za duzo' (deska przy misiu) i 'za malo' (brak brzucha).
Wyjscie: ciasne wycinki RGBA z featherem -> data/seg_gsam/<concept_id>/.
Run: python -m scripts._seg_gsam --config <cfg>
"""
import argparse, glob, os
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import (AutoProcessor, GroundingDinoForObjectDetection,
                          SamModel, SamProcessor)
from src.common import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/phaseP/P_ground_seg.yaml")
ap.add_argument("--out", default="data/seg_gsam")
ap.add_argument("--box_thr", type=float, default=0.3)
ap.add_argument("--min_alpha_frac", type=float, default=0.05)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
gd_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
gd = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(dev).eval()
sam_proc = SamProcessor.from_pretrained("facebook/sam-vit-huge")
sam = SamModel.from_pretrained("facebook/sam-vit-huge").to(dev).eval()

cfg = load_config(a.config)
for c in cfg["concepts"]:
    if c.get("category") == "style":
        print(f"[gsam] {c['concept_id']}: styl -> pomijam", flush=True)
        continue
    d = os.path.join(a.out, c["concept_id"])
    os.makedirs(d, exist_ok=True)
    text = c["class_word"].lower().strip()
    if not text.endswith("."):
        text = text + "."
    paths = sorted(sum((glob.glob(os.path.join(c["images_dir"], e))
                        for e in ("*.jpg", "*.jpeg", "*.png")), []))
    kept = 0
    for p in paths:
        img = Image.open(p).convert("RGB")
        with torch.no_grad():
            gi = gd_proc(images=img, text=text, return_tensors="pt").to(dev)
            go = gd(**gi)
            det = gd_proc.post_process_grounded_object_detection(
                go, gi.input_ids, box_threshold=a.box_thr, text_threshold=0.25,
                target_sizes=[img.size[::-1]])[0]
        if len(det["scores"]) == 0:
            print(f"[gsam]  BRAK DETEKCJI ({text}): {p}", flush=True)
            continue
        box = det["boxes"][det["scores"].argmax()].tolist()
        with torch.no_grad():
            si = sam_proc(img, input_boxes=[[box]], return_tensors="pt").to(dev)
            so = sam(**si)
            masks = sam_proc.image_processor.post_process_masks(
                so.pred_masks.cpu(), si["original_sizes"].cpu(),
                si["reshaped_input_sizes"].cpu())[0][0]        # [3, H, W]
            best = int(so.iou_scores[0, 0].argmax())
            mask = masks[best].numpy().astype(np.uint8) * 255
        soft = np.asarray(Image.fromarray(mask).filter(ImageFilter.GaussianBlur(2)))
        alpha = np.where(mask > 0, np.maximum(soft, 128), soft).astype(np.uint8)
        ys, xs = np.where(alpha > 16)
        if len(xs) == 0 or len(xs) < a.min_alpha_frac * alpha.size:
            print(f"[gsam]  ODRZUCONE (pusta maska): {p}", flush=True)
            continue
        arr = np.dstack([np.asarray(img), alpha])
        cut = Image.fromarray(arr, "RGBA").crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        cut.save(os.path.join(d, os.path.splitext(os.path.basename(p))[0] + ".png"))
        kept += 1
    print(f"[gsam] {c['concept_id']}: {kept}/{len(paths)} wycinkow -> {d}", flush=True)
print("GSAM_DONE", flush=True)
