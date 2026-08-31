import torch, glob, os, yaml
from transformers import CLIPTextModel, CLIPTokenizer
mid = "stable-diffusion-v1-5/stable-diffusion-v1-5"
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = CLIPTokenizer.from_pretrained(mid, subfolder="tokenizer")
te = CLIPTextModel.from_pretrained(mid, subfolder="text_encoder").eval().to(dev)

cfg = yaml.safe_load(open("configs/phaseF/F_base.yaml"))
words = sorted({(c.get("eval_prefix", "") + " " + c["class_word"]).strip() for c in cfg["concepts"]})
print("frazy klasowe:", words)

# --- KORPUS BAZOWY: wlasne szablony, ROZLACZNE z promptami CIFC
CTX = ["on a wooden table","in a forest clearing","under neon lights","beside a river",
 "in a busy market","on a snowy hill","inside a library","at sunset on a beach",
 "in a glass greenhouse","on a rooftop at night","in heavy rain","among autumn leaves",
 "in a subway station","on a marble floor","next to a bicycle","in a desert canyon",
 "behind a wire fence","on a red carpet","in a bamboo grove","near an old windmill",
 "wearing a scarf","made of origami","as a bronze statue","in watercolor style",
 "with dramatic shadows","in soft morning light","photographed from above","in extreme close-up",
 "surrounded by candles","on a chessboard","in a cardboard box","against a brick wall"]
TPL = ["a photo of {w} {c}","{w} {c}","a picture of {w} {c}","{w}, {c}","an image of {w} {c}"]
corpus = [t.format(w=w, c=c) for w in words for c in CTX for t in TPL]

ev = []
for f in sorted(glob.glob("data/CIFC/datasets/evaluation_prompts/*.txt")):
    ev += [l.strip() for l in open(f) if l.strip()]
eval_prompts = [p.replace("<TOK>", w) for w in words for p in ev]
caps = []
for c in cfg["concepts"]:
    d = c.get("caption_dir")
    if d and os.path.isdir(d):
        caps += [open(f).read().strip() for f in glob.glob(os.path.join(d, "*.txt"))]

IDS = {w: tok(w, add_special_tokens=False).input_ids for w in words}
def span_embs(prompts, bs=128):
    out = []
    for i in range(0, len(prompts), bs):
        ids = tok(prompts[i:i+bs], padding="max_length", max_length=77,
                  truncation=True, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            H = te(ids)[0]
        for n, seq in enumerate(ids.tolist()):
            for w, wid in IDS.items():
                hit = next((j for j in range(len(seq)-len(wid)+1) if seq[j:j+len(wid)] == wid), None)
                if hit is not None:
                    out.append(H[n, hit:hit+len(wid)].cpu()); break
    return torch.cat(out, 0) if out else None

# --- baza PER KONCEPT: adapter widzi wylacznie wlasny span tokenow
def per_word(prompts, w):
    wid = IDS[w]; out = []
    for i in range(0, len(prompts), 128):
        ids = tok(prompts[i:i+128], padding="max_length", max_length=77,
                  truncation=True, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad(): H = te(ids)[0]
        for n, seq in enumerate(ids.tolist()):
            j = next((j for j in range(len(seq)-len(wid)+1) if seq[j:j+len(wid)] == wid), None)
            if j is not None: out.append(H[n, j:j+len(wid)].cpu())
    return torch.cat(out, 0) if out else None
print(f"{'fraza':<26}{'rzad 99%':>9}{'P=8':>9}{'P=16':>9}{'P=32':>9}   (blad na promptach eval, najgorszy)")
for w in words:
    Ew = per_word([t.format(w=w, c=c) for c in CTX for t in TPL], w)
    Xw = per_word([pp.replace("<TOK>", w) for pp in ev], w)
    if Ew is None or Xw is None: print(f"{w:<26} brak"); continue
    muw = Ew.mean(0, keepdim=True)
    _, Sw, Vw = torch.linalg.svd((Ew-muw).float(), full_matrices=False)
    e=(Sw**2)/(Sw**2).sum(); c99=int((torch.cumsum(e,0)<0.99).sum())+1
    Xc=(Xw-muw).float(); row=[]
    for k in (8,16,32):
        P=Vw[:k]; R=Xc@P.t()@P
        row.append(float(((Xc-R).norm(dim=1)/Xw.float().norm(dim=1)).max())*100)
    print(f"{w:<26}{c99:>9}{row[0]:>8.2f}%{row[1]:>8.2f}%{row[2]:>8.2f}%")
import sys; sys.exit(0)
E = span_embs(corpus)
print(f"korpus bazowy: {len(corpus)} promptow -> {len(E)} zanurzen tokenow")
mu = E.mean(0, keepdim=True)
U, S, V = torch.linalg.svd((E - mu).float(), full_matrices=False)
en = (S**2)/(S**2).sum(); cum = torch.cumsum(en, 0)
print(f"rzad korpusu bazowego: 90%={int((cum<0.90).sum())+1}  95%={int((cum<0.95).sum())+1}  99%={int((cum<0.99).sum())+1}\n")

def err(name, prompts):
    X = span_embs(prompts)
    if X is None:
        print(f"{name}: brak"); return
    Xc = (X - mu).float()
    print(f"{name} (n={len(X)}):")
    for k in (8, 16, 32, 64):
        P = V[:k]
        R = Xc @ P.t() @ P
        rel = (Xc - R).norm(dim=1) / X.float().norm(dim=1)
        print(f"   P={k:>3}: sredni blad wzgl. {rel.mean()*100:6.3f}%   najgorszy {rel.max()*100:6.3f}%")
err("PROMPTY EWALUACYJNE (held-out)", eval_prompts)
err("captiony treningowe (held-out)", caps)
