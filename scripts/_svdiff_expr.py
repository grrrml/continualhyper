import torch, statistics as st
from src.sd_loader import load_sd
dev = "cuda" if torch.cuda.is_available() else "cpu"
b = load_sd(model_id="stable-diffusion-v1-5/stable-diffusion-v1-5", device=dev, dtype="fp32")
blob = torch.load("outputs/cl_noidA2F_b100/baked_lora.pt", map_location="cpu")["baked"]
tasks = sorted(blob); layers = list(blob[tasks[0]])
mods = dict(b.unet.named_modules())
def base_weight(name):
    m = mods.get(name)
    if m is None: return None
    w = getattr(m, "original", m)
    return getattr(w, "weight", None)
print(f"{'warstwa':<46}{'ksztalt':>14}{'energia na przekatnej':>24}")
share, done = [], 0
for lay in layers:
    W = base_weight(lay)
    if W is None: continue
    W = W.detach().float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    for t in tasks[:3]:
        xl, xr = blob[t][lay]
        # ΔW stosowana jako x @ x_L @ x_R  ==>  ΔW^T w konwencji nn.Linear (out,in)
        dW = (xl.float() @ xr.float()).t().to(W.device)      # [out,in]
        M = U.t() @ dW @ Vh.t()                              # w bazie wlasnej W0
        frac = float((torch.diagonal(M)**2).sum() / (M**2).sum())
        share.append(frac)
    done += 1
    if done <= 6:
        print(f"{lay[-44:]:<46}{str(tuple(W.shape)):>14}{share[-1]*100:>22.3f}%")
    if done >= 12: break
rnd = 1.0/min(W.shape)
print(f"\nsrednio po {len(share)} parach (warstwa,zadanie): {st.mean(share)*100:.3f}%")
print(f"poziom odniesienia dla macierzy losowej: ~{rnd*100:.3f}%")
print(f"=> SVDiff odtwarza ~{st.mean(share)*100:.1f}% energii naszej delty")
