"""Jednorazowa segmentacja konceptow CIFC (rembg/U2Net): dla kazdego zdjecia treningowego
zapisujemy CIASNY wyciety obiekt RGBA (miekka alfa z matowania) do data/seg/<concept_id>/.
Style pomijamy (globalne, nie maja obiektu). Run: python -m scripts._seg_concepts --config <cfg>
"""
import argparse, glob, os
import numpy as np
from PIL import Image
from rembg import remove, new_session
from src.common import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/phaseP/P_ground.yaml")
ap.add_argument("--out", default="data/seg")
ap.add_argument("--min_alpha_frac", type=float, default=0.05,
                help="odrzuc segmentacje pokrywajaca <5% obrazu (prawdopodobnie pusta)")
a = ap.parse_args()

cfg = load_config(a.config)
sess = new_session("isnet-general-use")
for c in cfg["concepts"]:
    if c.get("category") == "style":
        print(f"[seg] {c['concept_id']}: styl -> pomijam", flush=True)
        continue
    d = os.path.join(a.out, c["concept_id"])
    os.makedirs(d, exist_ok=True)
    paths = sorted(sum((glob.glob(os.path.join(c["images_dir"], e))
                        for e in ("*.jpg", "*.jpeg", "*.png")), []))
    kept = 0
    for p in paths:
        img = Image.open(p).convert("RGB")
        cut = remove(img, session=sess)                       # RGBA, miekka alfa
        # utwardzenie alfy: prog 0.5 -> binarna + feather 2px na krawedzi; usuwa
        # polprzezroczyste "duchy" wewnatrz obiektu (matowanie pada na jasnym futrze),
        # zostawia miekki brzeg po konturze
        arr = np.asarray(cut).copy()
        from PIL import ImageFilter
        hard = (arr[:, :, 3] > 38).astype(np.uint8) * 255   # prog 0.15: isnet daje ostre alfy
        soft = np.asarray(Image.fromarray(hard).filter(ImageFilter.GaussianBlur(2)))
        arr[:, :, 3] = np.where(hard > 0, np.maximum(soft, 128), soft)
        cut = Image.fromarray(arr)
        al = np.asarray(cut)[:, :, 3]
        ys, xs = np.where(al > 16)
        if len(xs) == 0 or len(xs) < a.min_alpha_frac * al.size:
            print(f"[seg]  ODRZUCONE (pusta maska): {p}", flush=True)
            continue
        cut = cut.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))   # ciasny bbox
        cut.save(os.path.join(d, os.path.splitext(os.path.basename(p))[0] + ".png"))
        kept += 1
    print(f"[seg] {c['concept_id']}: {kept}/{len(paths)} wycinkow -> {d}", flush=True)
print("SEG_DONE", flush=True)
