"""Build the forgetting-matrix image grid (rows = concepts V1..V10, columns = model after each
task, then a separated reference column). Cell (concept j, after-task k) = the eval sample from
the after-task-k model for concept j (only k>=j exists; earlier cells are left blank/white).

Run: python scripts/make_forgetting_grid.py --config configs/cl_unhype.yaml \
        --eval_root outputs/cl_unhype/cifc_eval --out outputs/cl_unhype/forgetting_grid.jpg
"""
import argparse, glob, os, sys
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.getcwd())
from src.common import load_config

THUMB, PAD, LABEL_W, HEADER_H, GAP = 95, 5, 58, 30, 40


def _font(sz):
    cands = ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
             "/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf"]
    try:  # matplotlib bundles DejaVuSansMono-Bold.ttf in the venv
        import matplotlib
        cands.insert(0, os.path.join(os.path.dirname(matplotlib.__file__),
                                     "mpl-data/fonts/ttf/DejaVuSansMono-Bold.ttf"))
    except Exception:
        pass
    for p in cands:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def _thumb(path):
    if path and os.path.exists(path):
        return Image.open(path).convert("RGB").resize((THUMB, THUMB), Image.BICUBIC)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval_root", default="outputs/cl_unhype/cifc_eval")
    ap.add_argument("--out", default="outputs/cl_unhype/forgetting_grid.jpg")
    ap.add_argument("--sample", default="0.jpg")
    args = ap.parse_args()

    concepts = load_config(args.config)["concepts"]
    n = len(concepts)
    ref_x = LABEL_W + n * (THUMB + PAD) + GAP
    W = ref_x + THUMB + PAD
    H = HEADER_H + n * (THUMB + PAD) + PAD
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    fnt = fnt_h = _font(24)

    sep = LABEL_W + n * (THUMB + PAD) + GAP // 2
    draw.line([(sep, 4), (sep, H - PAD)], fill=(200, 200, 200), width=2)

    # column headers, bottom-aligned right above the grid
    hy = HEADER_H - 8
    for k in range(n):
        x = LABEL_W + k * (THUMB + PAD) + THUMB // 2
        draw.text((x, hy), f"t{k+1}", fill=(0, 0, 0), font=fnt_h, anchor="mb")
    draw.text((ref_x + THUMB // 2, hy), "ref", fill=(170, 0, 0), font=fnt_h, anchor="mb")

    for j, c in enumerate(concepts):
        cid = c["concept_id"]
        y = HEADER_H + j * (THUMB + PAD)
        # row label, right-aligned right next to the grid
        draw.text((LABEL_W - 8, y + THUMB // 2), f"V{j+1}", fill=(0, 0, 0), font=fnt, anchor="rm")
        for k in range(j, n):
            im = _thumb(os.path.join(args.eval_root, f"after_task{k:02d}",
                                     f"task{j:02d}_{cid}", "samples", args.sample))
            if im is not None:
                canvas.paste(im, (LABEL_W + k * (THUMB + PAD), y))
        refs = sorted(glob.glob(os.path.join(c["images_dir"], "*")))
        ref = next((p for p in refs if p.lower().endswith((".jpg", ".jpeg", ".png"))), None)
        im = _thumb(ref)
        if im is not None:
            canvas.paste(im, (ref_x, y))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    canvas.save(args.out, quality=92)
    print(f"saved {args.out}  ({W}x{H})")


if __name__ == "__main__":
    main()
