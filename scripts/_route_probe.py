import torch, glob, statistics as st, itertools
from transformers import CLIPTextModel, CLIPTokenizer
mid="stable-diffusion-v1-5/stable-diffusion-v1-5"
dev="cuda" if torch.cuda.is_available() else "cpu"
tok=CLIPTokenizer.from_pretrained(mid,subfolder="tokenizer")
te=CLIPTextModel.from_pretrained(mid,subfolder="text_encoder").eval().to(dev)

ev=[l.strip() for l in open("data/CIFC/datasets/evaluation_prompts/test_pet.txt") if l.strip()]

def span_emb(prompt, word):
    """kontekstowe zanurzenie tokenu slowa klasy (ostatnia pozycja spanu)"""
    wid=tok(word,add_special_tokens=False).input_ids
    ids=tok([prompt],padding="max_length",max_length=77,truncation=True,return_tensors="pt").input_ids.to(dev)
    with torch.no_grad(): H=te(ids)[0][0]
    seq=ids[0].tolist()
    js=[j for j in range(len(seq)-len(wid)+1) if seq[j:j+len(wid)]==wid]
    return H[js[-1]+len(wid)-1].cpu() if js else None   # ostatnie wystapienie (po identyfikatorze)

cos=lambda a,b: float(torch.nn.functional.cosine_similarity(a,b,dim=0))

print("=== SONDA 1: separacja tozsamosci vs wariancja szablonu (slowo 'dog') ===")
E={}
for ident in ("V1","V7"):
    E[ident]=[span_emb(p.replace("<TOK>",f"{ident} dog"),"dog") for p in ev]
    E[ident]=[e for e in E[ident] if e is not None]
n=min(len(E["V1"]),len(E["V7"]))
within=[cos(a,b) for a,b in itertools.combinations(E["V1"][:n],2)]
across=[cos(E["V1"][i],E["V7"][i]) for i in range(n)]
print(f"szablonow: {n}")
print(f"within (V1, rozne szablony):        cos {st.mean(within):.4f}  (min {min(within):.4f})")
print(f"across (V1 vs V7, ten sam szablon): cos {st.mean(across):.4f}  (max {max(across):.4f})")
print(f"-> sygnal tozsamosci {'JEST' if st.mean(across) < st.mean(within) else 'GINIE w wariancji szablonu'}")
d_id=1-st.mean(across); d_tpl=1-st.mean(within)
print(f"   dystans tozsamosci {d_id:.4f} vs dystans szablonu {d_tpl:.4f}  (stosunek {d_id/max(d_tpl,1e-9):.2f}x)")

print("\n=== kontrola: bez identyfikatora (musi byc cos=1) ===")
a=span_emb(ev[0].replace("<TOK>","dog"),"dog"); b=span_emb(ev[1].replace("<TOK>","dog"),"dog")
c=span_emb(ev[0].replace("<TOK>","dog"),"dog")
print(f"ten sam prompt: {cos(a,c):.4f} | rozne szablony bez ident: {cos(a,b):.4f}")

print("\n=== SONDA 2: kompozycja - czy blizszy identyfikator dominuje? ===")
p1="a photo of V1 dog and V7 dog in a park"
wid=tok("dog",add_special_tokens=False).input_ids
ids=tok([p1],padding="max_length",max_length=77,truncation=True,return_tensors="pt").input_ids.to(dev)
with torch.no_grad(): H=te(ids)[0][0]
seq=ids[0].tolist()
js=[j for j in range(len(seq)) if seq[j:j+1]==wid]
d1,d2=H[js[0]].cpu(),H[js[1]].cpu()
solo1=span_emb("a photo of V1 dog in a park","dog")
solo7=span_emb("a photo of V7 dog in a park","dog")
print(f"1. 'dog' (po V1): cos do solo-V1 {cos(d1,solo1):.4f}  do solo-V7 {cos(d1,solo7):.4f}")
print(f"2. 'dog' (po V7): cos do solo-V7 {cos(d2,solo7):.4f}  do solo-V1 {cos(d2,solo1):.4f}")
print(f"cos miedzy oboma 'dog' w zdaniu: {cos(d1,d2):.4f}")
