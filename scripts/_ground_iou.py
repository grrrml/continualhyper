"""Placement UCZCIWIE: IoU wykrytej ramki obiektu z ramka zadana (+ stare cwiartki i DINO).

Po co: metryka cwiartek (argmax DINO po cwiartkach) mierzy, GDZIE koncept jest najbardziej
widoczny, a nie czy MIESCI SIE w ramce. Na siatce @kappa=4 dog2/cat2 maja 100% cwiartek,
podczas gdy zwierzak wypelnia caly kadr, a ramka lezy na jego glowie. Konwencja literatury
groundingu (GLIGEN, LayoutDiffusion) to detektor: IoU wykrytej ramki z zadana i odsetek
IoU>0.5. Detektor: torchvision Faster R-CNN R50-FPN v2 (COCO) -- zero nowych zaleznosci.
Klasa jest podpowiadana per koncept, z fallbackiem na najlepsza detekcje dowolnej klasy;
sciezka jest zliczana i drukowana, zeby liczba byla audytowalna, a nie magiczna.

Trzy liczby, bo mierza trzy rozne rzeczy:
  IoU         -- pozycja I skala razem (to, czego wymaga twierdzenie o ramce)
  zawarcie    -- ile obiektu wpadlo do ramki (przekroj / pole detekcji): pozycja bez skali
  wypelnienie -- pole detekcji / pole ramki: >1 znaczy "obiekt przelewa sie z ramki"

--gain_res wylacza wstrzyk na wybranych rozdzielczosciach map attn2 (np. "64:0,32:0.5"):
uklad rozstrzyga sie na mapach 8/16, kolor i tekstura na 32/64.

Run: python -u scripts/_ground_iou.py --grid 4.0:0.15 --gain_res 64:0
"""
import os, sys, glob, argparse, torch
sys.path.insert(0, ".")
import torch.nn.functional as Fn
from PIL import Image, ImageDraw
from src.common import load_config, load_hyper
from src.sd_loader import load_sd
from src.manager import build_hyper
from src.injection import DEFAULT_TARGETS
from src.sampling import ddim_sample
from src.tokens import token_span_mask
from src.regional import set_grounded
from src.cifc_metrics import _Dino

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/phaseP/P_ground_gsa.yaml")
ap.add_argument("--ckpt", default="outputs/phaseP/P_ground_gsa/hyper.pt")
ap.add_argument("--grid", default="4.0:0.15", help="pary kappa:sched po przecinku")
ap.add_argument("--gain_res", default="", help="mnozniki kappa per strona mapy, np. '64:0,32:0.5'")
ap.add_argument("--boxes", default="quads", choices=["quads", "halves"])
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--scale", type=float, default=0.7)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--det_thr", type=float, default=0.3)
ap.add_argument("--out", default="", help="katalog na podglady z ramkami (pusty = nie zapisuj)")
a = ap.parse_args()

GRID = [tuple(float(v) for v in p.split(":")) for p in a.grid.split(",")]
GAIN_RES = ({int(k): float(v) for k, v in (p.split(":") for p in a.gain_res.split(",") if p)}
            if a.gain_res else None)
BOXES = ({"TL": (0.25, 0.25, 0.5, 0.5), "TR": (0.75, 0.25, 0.5, 0.5),
          "BL": (0.25, 0.75, 0.5, 0.5), "BR": (0.75, 0.75, 0.5, 0.5)} if a.boxes == "quads"
         else {"L": (0.25, 0.5, 0.5, 1.0), "R": (0.75, 0.5, 0.5, 1.0)})
# Klasa COCO podpowiadana per koncept. Kaczka gumowa nie ma swojej klasy, "bird" to
# najblizsze, co COCO ma -- dlatego fallback na dowolna klase jest tu regula, nie wyjatkiem.
HINT = {"cifc_dog": "dog", "cifc_dog2": "dog", "cifc_cat": "cat", "cifc_cat2": "cat",
        "cifc_backpack": "backpack", "cifc_teddybear": "teddy bear", "cifc_duck_toy": "bird"}

cfg = load_config(a.config)
bundle = load_sd(device="cuda", dtype=torch.float16)
manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                      n_tasks=len(cfg["concepts"]), task_cond=cfg.get("task_cond"),
                      **cfg.get("hyper", {}))
load_hyper(manager, a.ckpt, map_location="cuda")
manager.eval(); manager.lora_scale = a.scale
manager.ground_gain_res = GAIN_RES
set_grounded(bundle.unet, manager)
dino = _Dino("cuda")

from torchvision.models.detection import (fasterrcnn_resnet50_fpn_v2,
                                          FasterRCNN_ResNet50_FPN_V2_Weights)
_w = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
det = fasterrcnn_resnet50_fpn_v2(weights=_w).eval().to("cuda")
CATS = _w.meta["categories"]

print(f"[env] torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
      f"host {os.uname().nodename} {os.uname().machine}", flush=True)
print(f"[cfg] ckpt {a.ckpt} | boxes {a.boxes} | n {a.n} | gain_res {GAIN_RES} | "
      f"detektor Faster R-CNN R50-FPN-v2 (COCO), prog {a.det_thr}", flush=True)
if a.out:
    os.makedirs(a.out, exist_ok=True)


def to_xyxy(box, W, H):
    cx, cy, bw, bh = box
    return ((cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H)


def iou_parts(p, q):
    """(IoU, pole przekroju, pole p) dla dwoch ramek xyxy."""
    ix0, iy0 = max(p[0], q[0]), max(p[1], q[1])
    ix1, iy1 = min(p[2], q[2]), min(p[3], q[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ap_ = max(0.0, p[2] - p[0]) * max(0.0, p[3] - p[1])
    aq = max(0.0, q[2] - q[0]) * max(0.0, q[3] - q[1])
    union = ap_ + aq - inter
    return (inter / union if union > 0 else 0.0), inter, ap_


@torch.no_grad()
def detect(img, hint):
    """img [3,H,W] w [0,1] -> (ramka xyxy, sciezka) albo (None, 'brak')."""
    out = det([img.float()])[0]
    keep = out["scores"] >= a.det_thr
    boxes, labels, scores = out["boxes"][keep], out["labels"][keep], out["scores"][keep]
    if len(boxes) == 0:
        return None, "brak"
    if hint is not None and hint in CATS:
        sel = labels == CATS.index(hint)
        if bool(sel.any()):
            i = int(scores[sel].argmax())
            return tuple(float(v) for v in boxes[sel][i]), "klasa"
    i = int(scores.argmax())
    return tuple(float(v) for v in boxes[i]), "fallback:" + CATS[int(labels[i])]


def crops(img, mode):
    H, W = img.shape[-2:]
    if mode == "quads":
        return {"TL": img[:, :H // 2, :W // 2], "TR": img[:, :H // 2, W // 2:],
                "BL": img[:, H // 2:, :W // 2], "BR": img[:, H // 2:, W // 2:]}
    return {"L": img[:, :, :W // 2], "R": img[:, :, W // 2:]}


@torch.no_grad()
def dino_sim(t, ref):
    pil = Image.fromarray((t.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())
    f = Fn.normalize(dino.m(dino.tf(pil).unsqueeze(0).to("cuda")), dim=-1)
    return float((f @ ref.t()).item()), pil


for kap, sched in GRID:
    manager.ground_gain_base = kap
    manager.ground_sched_frac = sched
    print(f"=== kappa={kap} sched={sched}", flush=True)
    agg = {"iou": 0.0, "hit": 0, "con": 0.0, "fill": 0.0, "q": 0, "n": 0, "dino": 0.0}
    for j, c in enumerate(cfg["concepts"]):
        if c.get("category") == "style":
            continue
        ref = dino.img_feats(sorted(glob.glob(os.path.join(c["images_dir"], "*")))).mean(0, keepdim=True)
        cls = c["class_word"]
        prompt = "a photo of " + cls
        ch, pooled, _ = bundle.encode_text([prompt])
        uh, _, _ = bundle.encode_text([""])
        tm = token_span_mask(bundle.tokenizer, [prompt], cls).cuda() if cfg.get("token_mask_lora") else None
        st = {"iou": 0.0, "hit": 0, "con": 0.0, "fill": 0.0, "q": 0, "n": 0, "dino": 0.0}
        paths = {}
        for bname, box in BOXES.items():
            manager.cond_box = box
            for i in range(a.n):
                g = torch.Generator(device="cuda").manual_seed(31337 + i)
                img = ddim_sample(bundle, manager, ch, uh, pooled, num_inference_steps=a.steps,
                                  guidance_scale=7.5, generator=g, task_idx=j, token_mask=tm)[0]
                H, W = img.shape[-2:]
                sims = {k: dino_sim(v, ref)[0] for k, v in crops(img, a.boxes).items()}
                st["q"] += int(max(sims, key=sims.get) == bname)
                d, pil = dino_sim(img, ref)
                st["dino"] += d
                req = to_xyxy(box, W, H)
                dbox, path = detect(img, HINT.get(c["concept_id"]))
                paths[path] = paths.get(path, 0) + 1
                if dbox is not None:
                    iou, inter, adet = iou_parts(dbox, req)
                    abox = (req[2] - req[0]) * (req[3] - req[1])
                    st["iou"] += iou
                    st["hit"] += int(iou > 0.5)
                    st["con"] += inter / adet if adet > 0 else 0.0
                    st["fill"] += adet / abox if abox > 0 else 0.0
                st["n"] += 1
                if a.out and i == 0:
                    dr = ImageDraw.Draw(pil)
                    dr.rectangle(req, outline=(255, 0, 0), width=3)
                    if dbox is not None:
                        dr.rectangle(dbox, outline=(0, 255, 0), width=3)
                    pil.save(os.path.join(a.out, c["concept_id"] + f"_k{kap}_s{sched}_{bname}.png"))
        n = max(1, st["n"])
        print(f"  {c['concept_id']:<16} cwiartki {st['q']}/{st['n']} = {st['q']/n:.0%} | "
              f"IoU {st['iou']/n:.3f} | IoU>0.5 {st['hit']/n:.0%} | zawarcie {st['con']/n:.2f} | "
              f"wypelnienie {st['fill']/n:.2f} | DINO {st['dino']/n:.4f} | det {paths}", flush=True)
        for k in agg:
            agg[k] += st[k]
    n = max(1, agg["n"])
    print(f"  RAZEM            cwiartki {agg['q']}/{agg['n']} = {agg['q']/n:.0%} | "
          f"IoU {agg['iou']/n:.3f} | IoU>0.5 {agg['hit']/n:.0%} | zawarcie {agg['con']/n:.2f} | "
          f"wypelnienie {agg['fill']/n:.2f} | DINO {agg['dino']/n:.4f}", flush=True)
print("IOU_DONE", flush=True)
