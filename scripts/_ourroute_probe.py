"""Nasz routing wzorem CIDM (pow4 + L1), dwa warianty: na wejsciowych embeddingach
(wiersz slownika = klucz) i na kontekstowych stanach encodera (jak u nich)."""
import torch, statistics as st
from transformers import CLIPTextModel, CLIPTokenizer
mid="stable-diffusion-v1-5/stable-diffusion-v1-5"; dev="cuda" if torch.cuda.is_available() else "cpu"
tok=CLIPTokenizer.from_pretrained(mid,subfolder="tokenizer")
te=CLIPTextModel.from_pretrained(mid,subfolder="text_encoder").eval().to(dev)
KEY_DIM, T = 128, 10
names=[f"<V{k+1}>" for k in range(T)]
tok.add_tokens(names); te.resize_token_embeddings(len(tok))
emb=te.get_input_embeddings().weight
with torch.no_grad():
    base=float(emb[:49408].norm(dim=1).mean())
    keys=[]
    for k in range(T):                                     # identycznie jak register_ortho_tokens
        g=torch.Generator().manual_seed(1234+k); key=torch.randn(KEY_DIM,generator=g)
        row=torch.zeros(emb.shape[1]); row[:KEY_DIM]=key/key.norm()*base
        emb[tok.convert_tokens_to_ids(names[k])]=row.to(dtype=emb.dtype,device=emb.device)
        keys.append(row)
K=torch.stack(keys)                                        # [10,768] wzorce = same wiersze
Kn=K/K.norm(dim=-1,keepdim=True)

CLS=["dog","duck toy","cat","backpack","teddy bear","painting","dog","drawing","cat","ink painting"]
tmpl=["a {} in the swimming pool","a {} on the beach","a {} in the swimming pool","a {} on the beach",
      "a {} on the beach","a mountain in the style of {}","a {} in the swimming pool",
      "a mountain in the style of {}","a {} in the swimming pool","a mountain in the style of {}"]

def weights(prompt, contextual):
    ids=tok([prompt],padding="max_length",max_length=77,truncation=True,return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        H = te(ids)[0][0] if contextual else te.get_input_embeddings()(ids)[0]
    Hn=H/H.norm(dim=-1,keepdim=True)
    ref = (te(tok([" ".join(names)],padding="max_length",max_length=77,truncation=True,
                 return_tensors="pt").input_ids.to(dev))[0][0][1:T+1] if contextual else Kn.to(dev))
    ref = ref/ref.norm(dim=-1,keepdim=True)
    lw = torch.einsum('w n, c n -> w c', ref, Hn)          # [10, 77]
    w,_ = torch.max(lw, dim=-1)                            # max po pozycjach promptu
    w = torch.pow(w.clamp_min(0), 4)
    return (w/w.sum()).cpu()

for contextual, tag in [(False,"WEJSCIOWE embeddingi (wiersz = klucz)"),
                        (True, "KONTEKSTOWE stany encodera (jak CIDM)")]:
    own,leak=[],[]
    print(f"\n=== {tag} ===")
    print(f"{'koncept':<18}{'waga wlasciwego':>16}{'najw. obcy':>12}")
    for j in range(T):
        w=weights(tmpl[j].format(f"<V{j+1}> {CLS[j]}"), contextual)
        o=float(w[j]); t=float(max(w[i] for i in range(T) if i!=j))
        own.append(o); leak.append(1-o)
        if j in (0,6,8,4): print(f"{CLS[j]+(' (dup)' if j in (6,8) else ''):<18}{o:>16.4f}{t:>12.4f}")
    print(f"{'SREDNIA':<18}{st.mean(own):>16.4f}   wyciek {st.mean(leak):.4f}   (CIDM: 0.613 / 0.387)")
