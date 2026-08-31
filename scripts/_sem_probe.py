import torch, yaml, glob, os, itertools
from src.cifc_metrics import _Clip
dev="cuda" if torch.cuda.is_available() else "cpu"
clip=_Clip(dev)
cfg=yaml.safe_load(open("configs/phaseK/K_sem128.yaml"))
E={}
for c in cfg["concepts"]:
    paths=sorted(glob.glob(os.path.join(c["images_dir"],"*")))
    E[c["concept_id"]]=clip.img_feats(paths).mean(0).cpu()
names=list(E)
cos=lambda a,b: float(torch.nn.functional.cosine_similarity(E[a],E[b],dim=0))
same=[("cifc_dog","cifc_dog2"),("cifc_cat","cifc_cat2")]
styles=[("cifc_painting","cifc_ink_painting"),("cifc_painting","cifc_drawing"),("cifc_ink_painting","cifc_drawing")]
allp=[(a,b) for a,b in itertools.combinations(names,2)]
import statistics as st
print(f"srednia po WSZYSTKICH parach: {st.mean(cos(a,b) for a,b in allp):+.3f}\n")
print("pary TEJ SAMEJ klasy:")
for a,b in same: print(f"   {a:<16} / {b:<18} {cos(a,b):+.3f}")
print("pary stylow:")
for a,b in styles: print(f"   {a:<16} / {b:<18} {cos(a,b):+.3f}")
print("\n5 najbardziej podobnych par ogolem:")
for v,a,b in sorted(((cos(a,b),a,b) for a,b in allp), reverse=True)[:5]:
    print(f"   {a:<16} / {b:<18} {v:+.3f}")
