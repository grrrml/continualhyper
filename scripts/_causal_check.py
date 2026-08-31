import torch
from transformers import CLIPTextModel, CLIPTokenizer
mid="stable-diffusion-v1-5/stable-diffusion-v1-5"
dev="cuda" if torch.cuda.is_available() else "cpu"
tok=CLIPTokenizer.from_pretrained(mid,subfolder="tokenizer")
te=CLIPTextModel.from_pretrained(mid,subfolder="text_encoder").eval().to(dev)
def emb(p, w="dog"):
    wid=tok(w,add_special_tokens=False).input_ids
    ids=tok([p],padding="max_length",max_length=77,truncation=True,return_tensors="pt").input_ids.to(dev)
    with torch.no_grad(): H=te(ids)[0][0]
    seq=ids[0].tolist()
    j=next(j for j in range(len(seq)-len(wid)+1) if seq[j:j+len(wid)]==wid)
    return H[j].cpu()
base=emb("A dog, in the swimming pool")
print("TEN SAM prefiks 'A ', rozne konce:")
for p in ["A dog, in front of Eiffel tower","A dog, near the mount fuji","A dog on a chessboard in heavy rain"]:
    d=(emb(p)-base).norm()/base.norm()
    print(f"   {p[:44]:<46} roznica wzgl. {d*100:7.4f}%")
print("\nROZNE prefiksy:")
for p in ["a photo of dog, in the swimming pool","A jumping dog, in the swimming pool","A painting of a dog, in the swimming pool"]:
    d=(emb(p)-base).norm()/base.norm()
    print(f"   {p[:44]:<46} roznica wzgl. {d*100:7.4f}%")
