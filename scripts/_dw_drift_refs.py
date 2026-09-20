"""Wyciaga z checkpointow czynniki WLASNEGO konceptu -- male pliki referencyjne dla _dw_drift.py.

Po co. Strumien p022_v2 jest wznawiany, wiec jego checkpointy leza na dwoch klastrach: zadania
00-09 zostaly na Heliosie, 09-49 powstaly na Athenie. `_dw_drift.py` potrzebuje dW_j(phi_j) dla
KAZDEGO j, wiec bez zadan 00-08 przerywa na k=0 i zapisuje pusty `by_lag` (job 3184647).
Przeniesienie brakujacych checkpointow to 9 x 104 MB przez lacze domowe: zmierzone 6m43s na plik,
czyli okolo dwoch godzin w kazda strone.

Ale dryf liczy sie z macierzy r x r, a z checkpointu j potrzebny jest wylacznie JEDEN koncept --
ten wlasnie nauczony. To jest 1/50 zawartosci, okolo 2.6 MB. Ten skrypt zapisuje dokladnie to,
a `_dw_drift.py --ref_factors` doklada je do `refs` i pomija brakujace checkpointy.

Kontrola poprawnosci jest wbudowana w zakres: checkpoint 09 istnieje na OBU klastrach, wiec jego
czynniki wyciagniete tutaj musza zgadzac sie z tymi, ktore Athena policzy u siebie. Rozjazd
oznaczalby, ze jakis parametr nie siedzi w checkpointcie i zostaje z losowej inicjalizacji --
a wtedy caly pomysl przenoszenia referencji miedzy klastrami jest niewazny. Dlatego 09 jest
zapisywany mimo ze Athena go ma, i dlatego `_dw_drift.py` porownuje go glosno.

Run:  python scripts/_dw_drift_refs.py --config configs/phaseT/T50_v2.yaml \
          --ckpt_dir outputs/sweep/p022_v2/ckpts --first 0 --last 9 \
          --out outputs/sweep/p022_v2/dw_refs_00_09.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _dw_drift import factors                           # noqa: E402  ta sama definicja, nie kopia
from src.common import load_config, load_hyper          # noqa: E402
from src.injection import DEFAULT_TARGETS               # noqa: E402
from src.manager import build_hyper                     # noqa: E402
from src.sd_loader import load_sd                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, required=True, help="wlacznie")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = load_config(a.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_sd(model_id=cfg.get("sd_model_id", "") or None, device=device, dtype="fp32")
    manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                          n_tasks=len(cfg["concepts"]), task_cond=cfg.get("task_cond"),
                          **cfg.get("hyper", {}))

    refs = {}
    for j in range(a.first, a.last + 1):
        ck = os.path.join(a.ckpt_dir, f"hyper_after_task{j:02d}.pt")
        if not os.path.exists(ck):
            raise SystemExit(f"BLAD: brak {ck}")
        load_hyper(manager, ck, map_location=str(device))
        # dokladnie ten sam wywolanie co w _dw_drift.py: checkpoint po zadaniu j niesie
        # warunkowanie kanoniczne tylko dla zadan 0..j
        F = factors(manager, bundle, j + 1, device)
        refs[j] = {n: (v[0][j:j + 1].cpu().clone(), v[1][j:j + 1].cpu().clone())
                   for n, v in F.items()}
        print(f"[refs] zadanie {j}: {len(refs[j])} warstw", flush=True)

    out_dir = os.path.dirname(os.path.abspath(a.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(refs, a.out)
    print(f"[refs] zapisane do {a.out} "
          f"({os.path.getsize(a.out) / 1e6:.1f} MB, zadania {a.first}-{a.last})")


if __name__ == "__main__":
    main()
