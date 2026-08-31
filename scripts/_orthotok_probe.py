import torch, itertools, statistics as st
from transformers import CLIPTextModel, CLIPTokenizer
mid="stable-diffusion-v1-5/stable-diffusion-v1-5"
dev="cuda" if torch.cuda.is_available() else "cpu"
tok=CLIPTokenizer.from_pretrained(mid,subfolder="tokenizer")
te=CLIPTextModel.from_pretrained(mid,subfolder="text_encoder").eval().to(dev)

# --- dodaj <V1>..<V10> i ustaw wiersze na ORTOGONALNE klucze (norma dopasowana do slownika)
names=[f"<V{k}>" for k in range(1,11)]
tok.add_tokens(names)
te.resize_token_embeddings(len(tok))
emb=te.get_input_embeddings().weight
with torch.no_grad():
    base_norm=emb[:49408].norm(dim=1).mean()
    print(f"srednia norma wiersza slownika: {base_norm:.4f}")
    Q,_=torch.linalg.qr(torch.randn(768,10,generator=torch.Generator().manual_seed(1234)))
    for i,n in enumerate(names):
        tid=tok.convert_tokens_to_ids(n)
        emb[tid]=(Q[:,i]*float(base_norm)).to(dtype=emb.dtype, device=emb.device)

ev=[l.strip() for l in open("data/CIFC/datasets/evaluation_prompts/test_pet.txt") if l.strip()]
def span_emb(prompt, word="dog"):
    wid=tok(word,add_special_tokens=False).input_ids
    ids=tok([prompt],padding="max_length",max_length=77,truncation=True,return_tensors="pt").input_ids.to(dev)
    with torch.no_grad(): H=te(ids)[0][0]
    seq=ids[0].tolist()
    js=[j for j in range(len(seq)-len(wid)+1) if seq[j:j+len(wid)]==wid]
    return [H[j+len(wid)-1].cpu() for j in js]
cos=lambda a,b: float(torch.nn.functional.cosine_similarity(a,b,dim=0))

print("\n=== SONDA 1 z tokenami ortogonalnymi: separacja vs szablon ===")
E={}
for ident in ("<V1>","<V7>"):
    E[ident]=[span_emb(p.replace("<TOK>",f"{ident} dog"))[0] for p in ev if span_emb(p.replace("<TOK>",f"{ident} dog"))]
n=min(len(E["<V1>"]),len(E["<V7>"]))
within=[cos(a,b) for a,b in itertools.combinations(E["<V1>"][:n],2)]
across=[cos(E["<V1>"][i],E["<V7>"][i]) for i in range(n)]
print(f"within (ten koncept, rozne szablony): {st.mean(within):.4f}")
print(f"across (rozne koncepty, ten szablon): {st.mean(across):.4f}")
d_id,d_tpl=1-st.mean(across),1-st.mean(within)
print(f"stosunek dystansow: {d_id/max(d_tpl,1e-9):.2f}x   (stringi 'V1'/'V7' dawaly 1.48x)")

print("\n=== SONDA 2 z tokenami ortogonalnymi: kompozycja ===")
comp=span_emb("a photo of <V1> dog and <V7> dog in a park")
solo1=span_emb("a photo of <V1> dog in a park")[0]
solo7=span_emb("a photo of <V7> dog in a park")[0]
d1,d2=comp[0],comp[1]
print(f"1. 'dog' (po <V1>): do solo-V1 {cos(d1,solo1):.4f}   do solo-V7 {cos(d1,solo7):.4f}")
print(f"2. 'dog' (po <V7>): do solo-V7 {cos(d2,solo7):.4f}   do solo-V1 {cos(d2,solo1):.4f}")
print(f"   (stringi dawaly: 0.7777 vs 0.7790 - rownoodlegle)")
marg=cos(d2,solo7)-cos(d2,solo1)
print(f"margines routingu drugiego konceptu: {marg:+.4f}  -> {'BLIZSZY SWOJEMU' if marg>0.01 else 'nadal nierozroznialne'}")
