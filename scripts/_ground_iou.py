"""Placement i kolor UCZCIWIE: detektor + maska obiektu wobec ramki zadanej.

Po co: metryka cwiartek (argmax DINO po cwiartkach) mierzy, GDZIE koncept jest najbardziej
widoczny, a nie czy MIESCI SIE w ramce. Zmierzone @kappa=2: kaczka ma 83% cwiartek i 33%
IoU>0.5, a wykryty obiekt jest 1.9x wiekszy od zadanej ramki. Konwencja literatury
groundingu (GLIGEN, LayoutDiffusion) to detektor: IoU wykrytej ramki z zadana + odsetek
IoU>0.5. Drugi brak: DINO ViT-S/16 jest na kolor prawie slepy (zielona kaczka 0.690 vs
zolta 0.670), wiec dryf koloru nie mial dotad ZADNEJ liczby.

Detektor: torchvision Mask R-CNN R50-FPN v2 (COCO) -- zero nowych zaleznosci, a maska daje
kolor obiektu bez tla. Wybor detekcji: najpierw klasa podpowiedziana per koncept; jesli jej
nie ma, kandydat najbardziej podobny (DINO) do zdjec referencyjnych, a nie ten o najwyzszym
score -- bo dla kaczki gumowej COCO strzela "dining table" i "vase", czyli mebel pod
obiektem. Ponizej progu podobienstwa liczymy "brak" i raportujemy wykrywalnosc osobno,
zeby porazka detektora nie udawala porazki placementu.

Metryki, bo mierza rozne rzeczy:
  IoU         -- pozycja I skala razem (to, czego wymaga twierdzenie o ramce)
  IoU>0.5     -- prog uzywany w literaturze groundingu
  zawarcie    -- frakcja pikseli MASKI obiektu wpadajaca w ramke: pozycja bez skali
  wypelnienie -- pole detekcji / pole ramki; >1 znaczy "obiekt przelewa sie z ramki"
  kolor dRGB  -- odleglosc sredniego koloru obiektu (piksele maski) od koloru referencji
                 liczonego z WYCINKOW data/seg/<cid> (alfa>0.5), nie z calych zdjec
  cwiartki    -- stara metryka, zostawiona do porownania z poprzednimi werdyktami

Knoby badane tym skryptem:
  --grid kappa:sched[:confine]  -- confine to kara logitu dla tokenow konceptu poza ramka
  --gain_res 64:0               -- wylacza wstrzyk na mapach 64x64 (kolor/tekstura),
                                   zostawiajac go na 8/16/32 (uklad)

Run: python -u scripts/_ground_iou.py --grid 4.0:0.15,4.0:0.15:5 --n 3
"""
import os, sys, glob, argparse
import numpy as np
import torch
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
ap.add_argument("--grid", default="4.0:0.15", help="kappa:sched[:confine], po przecinku")
ap.add_argument("--gain_res", default="", help="mnozniki kappa per strona mapy, np. '64:0,32:0.5'")
ap.add_argument("--boxes", default="quads", choices=["quads", "halves", "full"])
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--scale", type=float, default=0.7)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--det_thr", type=float, default=0.3)
ap.add_argument("--dino_floor", type=float, default=0.35,
                help="ponizej tego podobienstwa kandydat nie jest uznawany za obiekt")
ap.add_argument("--seg_dir", default="data/seg", help="wycinki referencyjne do koloru")
ap.add_argument("--prefix", action="store_true",
                help="promptuj z eval_prefix ('yellow rubber duck toy') zamiast golo")
ap.add_argument("--bg_ref", action="store_true",
                help="dla kazdej generacji policz TE SAMA bez ramki (to samo ziarno) i podaj "
                     "podobienstwo DINO TLA (obiekt zamaskowany): 1.0 = ramka nie ruszyla tla. "
                     "Energia gradientu mierzy lokalny kontrast, nie zawartosc sceny - ostra "
                     "krawedz pustej sciany punktuje wyzej niz rozmyty park.")
ap.add_argument("--seed0", type=int, default=31337,
                help="baza ziaren generacji; inna wartosc = niezalezna probka tej samej "
                     "konfiguracji, czyli szum SAMPLINGOWY metryki (nie treningowy)")
ap.add_argument("--confine_tail", action="store_true",
                help="kara confine do konca sekwencji (CLIP przyczynowy), nie tylko na spanie")
ap.add_argument("--out", default="", help="katalog na podglady z ramkami (pusty = nie zapisuj)")
a = ap.parse_args()


def _pt(p):
    v = [float(x) for x in p.split(":")]
    return (v + [0.0])[:3] if len(v) < 3 else v[:3]


GRID = [_pt(p) for p in a.grid.split(",")]
GAIN_RES = ({int(k): float(v) for k, v in (p.split(":") for p in a.gain_res.split(",") if p)}
            if a.gain_res else None)
BOXES = {"quads": {"TL": (0.25, 0.25, 0.5, 0.5), "TR": (0.75, 0.25, 0.5, 0.5),
                   "BL": (0.25, 0.75, 0.5, 0.5), "BR": (0.75, 0.75, 0.5, 0.5)},
         "halves": {"L": (0.25, 0.5, 0.5, 1.0), "R": (0.75, 0.5, 0.5, 1.0)},
         "full": {"F": (0.5, 0.5, 1.0, 1.0)}}[a.boxes]
# Klasy COCO podpowiadane per koncept (lista, bo kaczka gumowa nie ma swojej klasy).
HINT = {"cifc_dog": ("dog",), "cifc_dog2": ("dog",), "cifc_cat": ("cat",),
        "cifc_cat2": ("cat",), "cifc_backpack": ("backpack", "handbag"),
        "cifc_teddybear": ("teddy bear",), "cifc_duck_toy": ("bird", "teddy bear")}

cfg = load_config(a.config)
bundle = load_sd(device="cuda", dtype=torch.float16)
manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                      n_tasks=len(cfg["concepts"]), task_cond=cfg.get("task_cond"),
                      **cfg.get("hyper", {}))
load_hyper(manager, a.ckpt, map_location="cuda")
manager.eval()
manager.lora_scale = a.scale
manager.ground_gain_res = GAIN_RES
manager.ground_confine_tail = a.confine_tail
set_grounded(bundle.unet, manager)
dino = _Dino("cuda")

from torchvision.models.detection import (maskrcnn_resnet50_fpn_v2,
                                          MaskRCNN_ResNet50_FPN_V2_Weights)
_w = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
det = maskrcnn_resnet50_fpn_v2(weights=_w).eval().to("cuda")
CATS = _w.meta["categories"]

print(f"[env] torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
      f"host {os.uname().nodename} {os.uname().machine}", flush=True)
print(f"[cfg] ckpt {a.ckpt} | boxes {a.boxes} | n {a.n} | gain_res {GAIN_RES} | "
      f"prefix {a.prefix} | confine_tail {a.confine_tail} | seed0 {a.seed0} | "
      f"Mask R-CNN R50-FPN-v2, prog {a.det_thr}, dino_floor {a.dino_floor}",
      flush=True)
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


def ref_color(concept_id):
    """Sredni kolor OBIEKTU z wycinkow referencyjnych (alfa>0.5). Cale zdjecie nie nadaje
    sie na referencje koloru: tlo (zwir, dywan) wchodzi do sredniej mocniej niz obiekt."""
    tot, wsum = np.zeros(3), 0.0
    for p in sorted(glob.glob(os.path.join(a.seg_dir, concept_id, "*.png"))):
        arr = np.asarray(Image.open(p).convert("RGBA"), dtype=np.float32) / 255.0
        m = arr[..., 3] > 0.5
        if not m.any():
            continue
        tot += arr[..., :3][m].sum(0)
        wsum += float(m.sum())
    return (tot / wsum) if wsum > 0 else None


def ref_fill(c):
    """Sredni udzial obiektu w kadrze ZDJEC REFERENCYJNYCH (pole ciasnego wycinka / pole
    zdjecia). Hipoteza do sprawdzenia wobec kolumny "wypelnienie": adapter uczy sie skali
    z referencji, wiec koncepty fotografowane jako zblizenie beda przelewac sie z ramki
    niezaleznie od jakosci adresowania."""
    vals = []
    for p in sorted(glob.glob(os.path.join(a.seg_dir, c["concept_id"], "*.png"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        orig = glob.glob(os.path.join(c["images_dir"], stem + ".*"))
        if not orig:
            continue
        cw, ch = Image.open(p).size
        ow, oh = Image.open(orig[0]).size
        vals.append((cw * ch) / float(ow * oh))
    return (sum(vals) / len(vals)) if vals else None


def bg_stats(img, dmask):
    """Jakosc TLA: srednia energia gradientu i odchylenie jasnosci na pikselach POZA maska
    obiektu. Zaden inny licznik w tym skrypcie tla nie widzi (IoU dotyczy ramki, dRGB pikseli
    obiektu, DINO jest zdominowane przez obiekt), a plaskie tlo to obserwowany artefakt kappa.
    Liczby maja sens tylko W PORNANIU miedzy konfiguracjami na tych samych ziarnach."""
    g = img.float().mean(0)
    gx = (g[:, 1:] - g[:, :-1]).abs()
    gy = (g[1:, :] - g[:-1, :]).abs()
    grad = torch.zeros_like(g)
    grad[:, :-1] += gx
    grad[:, 1:] += gx
    grad[:-1, :] += gy
    grad[1:, :] += gy
    bg = (~dmask) if dmask is not None else torch.ones_like(g, dtype=torch.bool)
    if int(bg.sum()) < 1000:
        return None
    return float(grad[bg].mean()), float(g[bg].std())


def mask_out(img, dmask):
    """Obiekt zamalowany szarym: zostaje samo tlo, wiec DINO patrzy na scene, nie na podmiot."""
    out = img.clone()
    if dmask is not None:
        out[:, dmask] = 0.5
    return out


@torch.no_grad()
def dino_feat(pil):
    return Fn.normalize(dino.m(dino.tf(pil).unsqueeze(0).to("cuda")), dim=-1)


def to_pil(t):
    return Image.fromarray((t.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())


@torch.no_grad()
def detect(img, pil, hints, ref):
    """-> (ramka xyxy, maska bool [H,W], sciezka) albo (None, None, 'brak')."""
    out = det([img.float()])[0]
    keep = out["scores"] >= a.det_thr
    boxes, labels, masks = out["boxes"][keep], out["labels"][keep], out["masks"][keep]
    scores = out["scores"][keep]
    if len(boxes) == 0:
        return None, None, "brak"
    hid = [CATS.index(x) for x in (hints or ()) if x in CATS]
    if hid:
        sel = torch.isin(labels, torch.tensor(hid, device=labels.device)).nonzero().flatten()
        if len(sel) > 0:
            i = int(sel[int(scores[sel].argmax())])
            return tuple(float(v) for v in boxes[i]), masks[i, 0] > 0.5, "klasa"
    # zadna podpowiedziana klasa nie wypadla: wybieramy kandydata NAJBARDZIEJ PODOBNEGO
    # do referencji (nie o najwyzszym score - ten pokazuje mebel pod obiektem)
    best, bsim = None, -1.0
    for i in range(len(boxes)):
        x0, y0, x1, y1 = [int(v) for v in boxes[i]]
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        sim = float((dino_feat(pil.crop((x0, y0, x1, y1))) @ ref.t()).item())
        if sim > bsim:
            best, bsim = i, sim
    if best is None or bsim < a.dino_floor:
        return None, None, "brak"
    return (tuple(float(v) for v in boxes[best]), masks[best, 0] > 0.5,
            "dino:" + CATS[int(labels[best])])


def crops(img, mode):
    H, W = img.shape[-2:]
    if mode == "quads":
        return {"TL": img[:, :H // 2, :W // 2], "TR": img[:, :H // 2, W // 2:],
                "BL": img[:, H // 2:, :W // 2], "BR": img[:, H // 2:, W // 2:]}
    if mode == "halves":
        return {"L": img[:, :, :W // 2], "R": img[:, :, W // 2:]}
    return {"F": img}


KEYS = ("iou", "hit", "con", "fill", "q", "n", "ndet", "dino", "dcol", "ncol",
        "bgg", "bgs", "nbg", "bgsim", "nbgsim")
for kap, sched, conf in GRID:
    manager.ground_gain_base = kap
    manager.ground_sched_frac = sched
    manager.ground_confine = conf
    print(f"=== kappa={kap} sched={sched} confine={conf}", flush=True)
    agg = dict.fromkeys(KEYS, 0.0)
    for j, c in enumerate(cfg["concepts"]):
        if c.get("category") == "style":
            continue
        ref = dino.img_feats(sorted(glob.glob(os.path.join(c["images_dir"], "*")))).mean(0, keepdim=True)
        rcol = ref_color(c["concept_id"])
        cls = c["class_word"]
        pref = (c.get("eval_prefix", "").strip() + " ") if a.prefix else ""
        prompt = "a photo of " + pref + cls
        ch, pooled, _ = bundle.encode_text([prompt])
        uh, _, _ = bundle.encode_text([""])
        tm = token_span_mask(bundle.tokenizer, [prompt], cls).cuda() if cfg.get("token_mask_lora") else None
        st = dict.fromkeys(KEYS, 0.0)
        paths, gcol_sum, gcol_n = {}, np.zeros(3), 0
        for bname, box in BOXES.items():
            manager.cond_box = box
            for i in range(a.n):
                g = torch.Generator(device="cuda").manual_seed(a.seed0 + i)
                img = ddim_sample(bundle, manager, ch, uh, pooled, num_inference_steps=a.steps,
                                  guidance_scale=7.5, generator=g, task_idx=j, token_mask=tm)[0]
                H, W = img.shape[-2:]
                pil = to_pil(img)
                sims = {k: float((dino_feat(to_pil(v)) @ ref.t()).item())
                        for k, v in crops(img, a.boxes).items()}
                st["q"] += int(max(sims, key=sims.get) == bname)
                st["dino"] += float((dino_feat(pil) @ ref.t()).item())
                st["n"] += 1
                req = to_xyxy(box, W, H)
                dbox, dmask, path = detect(img, pil, HINT.get(c["concept_id"]), ref)
                paths[path] = paths.get(path, 0) + 1
                if dbox is not None:
                    iou, inter, adet = iou_parts(dbox, req)
                    abox = (req[2] - req[0]) * (req[3] - req[1])
                    st["ndet"] += 1
                    st["iou"] += iou
                    st["hit"] += int(iou > 0.5)
                    st["fill"] += adet / abox if abox > 0 else 0.0
                    inb = torch.zeros_like(dmask)
                    inb[int(req[1]):int(req[3]), int(req[0]):int(req[2])] = True
                    tot = float(dmask.sum())
                    st["con"] += float((dmask & inb).sum()) / tot if tot > 0 else 0.0
                    if rcol is not None and tot > 0:
                        gc = img.permute(1, 2, 0)[dmask].clamp(0, 1).float().cpu().numpy().mean(0)
                        gcol_sum += gc
                        gcol_n += 1
                        st["dcol"] += float(np.linalg.norm(gc - rcol))
                        st["ncol"] += 1
                if a.bg_ref:
                    # to samo ziarno, ten sam prompt, ramka = pelny kadr => grounding bez adresu
                    keep = (manager.cond_box, manager.ground_confine)
                    manager.cond_box = None
                    manager.ground_confine = 0.0
                    g2 = torch.Generator(device="cuda").manual_seed(a.seed0 + i)
                    ref_img = ddim_sample(bundle, manager, ch, uh, pooled,
                                          num_inference_steps=a.steps, guidance_scale=7.5,
                                          generator=g2, task_idx=j, token_mask=tm)[0]
                    manager.cond_box, manager.ground_confine = keep
                    _, rmask, _ = detect(ref_img, to_pil(ref_img),
                                         HINT.get(c["concept_id"]), ref)
                    both = dmask if rmask is None else (
                        dmask | rmask if dmask is not None else rmask)
                    f1 = dino_feat(to_pil(mask_out(img, both)))
                    f2 = dino_feat(to_pil(mask_out(ref_img, both)))
                    st["bgsim"] += float((f1 @ f2.t()).item())
                    st["nbgsim"] += 1
                bs = bg_stats(img, dmask)
                if bs is not None:
                    st["bgg"] += bs[0]
                    st["bgs"] += bs[1]
                    st["nbg"] += 1
                if a.out and i == 0:
                    dr = ImageDraw.Draw(pil)
                    dr.rectangle(req, outline=(255, 0, 0), width=3)
                    if dbox is not None:
                        dr.rectangle(dbox, outline=(0, 255, 0), width=3)
                    pil.save(os.path.join(a.out, c["concept_id"] + f"_k{kap}_s{sched}_c{conf}_{bname}.png"))
        n, nd = max(1.0, st["n"]), max(1.0, st["ndet"])
        print(f"  {c['concept_id']:<16} cwiartki {int(st['q'])}/{int(st['n'])} = {st['q']/n:.0%} | "
              f"IoU {st['iou']/nd:.3f} | IoU>0.5 {st['hit']/nd:.0%} | zawarcie {st['con']/nd:.2f} | "
              f"wypelnienie {st['fill']/nd:.2f} | DINO {st['dino']/n:.4f} | "
              f"det {int(st['ndet'])}/{int(st['n'])} {paths}", flush=True)
        nbg = max(1.0, st["nbg"])
        extra = (f" | tlo sim {st['bgsim']/max(1.0, st['nbgsim']):.4f}" if st["nbgsim"] else "")
        print(f"  {'':<16} tlo grad {st['bgg']/nbg:.4f} | tlo std {st['bgs']/nbg:.4f}{extra}",
              flush=True)
        rf = ref_fill(c)
        if rf is not None:
            print(f"  {'':<16} obiekt w kadrze referencji: {rf:.2f}", flush=True)
        if rcol is not None and gcol_n > 0:
            gm, ncol = gcol_sum / gcol_n, max(1.0, st["ncol"])
            print(f"  {'':<16} kolor dRGB {st['dcol']/ncol:.3f} | "
                  f"gen ({gm[0]:.2f},{gm[1]:.2f},{gm[2]:.2f}) vs ref "
                  f"({rcol[0]:.2f},{rcol[1]:.2f},{rcol[2]:.2f})", flush=True)
        for k in KEYS:
            agg[k] += st[k]
    n, nd = max(1.0, agg["n"]), max(1.0, agg["ndet"])
    print(f"  RAZEM            cwiartki {int(agg['q'])}/{int(agg['n'])} = {agg['q']/n:.0%} | "
          f"IoU {agg['iou']/nd:.3f} | IoU>0.5 {agg['hit']/nd:.0%} | zawarcie {agg['con']/nd:.2f} | "
          f"wypelnienie {agg['fill']/nd:.2f} | DINO {agg['dino']/n:.4f} | "
          f"kolor dRGB {agg['dcol']/max(1.0, agg['ncol']):.3f} | "
          f"tlo grad {agg['bgg']/max(1.0, agg['nbg']):.4f} | "
          f"tlo std {agg['bgs']/max(1.0, agg['nbg']):.4f} | "
          + (f"tlo sim {agg['bgsim']/max(1.0, agg['nbgsim']):.4f} | " if agg['nbgsim'] else "")
          + f"det {int(agg['ndet'])}/{int(agg['n'])}", flush=True)
print("IOU_DONE", flush=True)
