"""Captiony dla CustomConcept101 w rezimie CIFC -- i kalibracja tego rezimu na ich danych.

Po co: CC101 nie ma zadnych captionow (zero plikow .txt w calym zbiorze), a CIFC ma je
per zdjecie. Przebieg T=50 na CC101 zestawiony z T=10 na CIFC mieszalby wiec wplyw liczby
konceptow z wplywem zmiany rezimu captionow. Ten skrypt usuwa ten czynnik: nadaje CC101
captiony tego samego rodzaju co CIFC.

Czym sa captiony CIFC: to wyjscie BLIP-a, krotkie zdanie "obiekt + czynnosc + miejsce",
np. "cat sitting on a stair rail", "teddy bear sitting on the ground in the dirt". Uzywamy
wiec BLIP-a, a nie modelu, ktory pisze akapitami.

KALIBRACJA, i to jest powod, dla ktorego mozna temu ufac: 12 zdjec CIFC jest bajt w bajt
identycznych ze zdjeciami CC101 (cifc/cat = pet_cat1, 5 zdjec; cifc/teddybear =
plushie_teddybear, 7 zdjec). Dla tych dwunastu znamy caption CIFC, wiec `--calibrate`
generuje nasz caption i pokazuje oba obok siebie. Jesli nasze rozjezdzaja sie stylem albo
dlugoscia, to widac przed policzeniem czegokolwiek, zamiast tlumaczyc potem roznice w
wynikach czyms, czego nie sprawdzilismy.

Uwaga o atrybutach: BLIP wstawia przymiotniki ("fluffy cat"), tak samo jak captiony CIFC.
Metoda strippuje je przez `concepts[].attr_strip`, bo atrybut opisany tekstem to atrybut,
ktorego adapter sie nie nauczy. Skrypt raportuje wiec, w ktorych captionach przed slowem
klasy stoi przymiotnik, zeby dalo sie te listy wypelnic bez czytania 634 plikow.

Run (login node ma internet, wezly maja HF_HUB_OFFLINE=1, wiec model prefetchujemy tutaj):
    python -u scripts/_caption_cc101.py --calibrate
    python -u scripts/_caption_cc101.py --out data/cc101_caption
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, ".")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Zdjecia wspolne dla CIFC i CC101, znalezione po md5. Sluza wylacznie do kalibracji.
OVERLAP = {
    "pet_cat1": ("cat", {
        "jeanie-de-klerk-av2WGfogjqg-unsplash": "01",
        "jeanie-de-klerk-bhonzdJMVjY-unsplash": "02",
        "jeanie-de-klerk-QfOnteIcGZU-unsplash": "03",
        "jeanie-de-klerk-SGFLjexLlaI-unsplash": "04",
        "jeanie-de-klerk-t_Z5ND4Ce3k-unsplash": "05",
    }),
    "plushie_teddybear": ("teddybear", None),  # te same nazwy plikow po obu stronach
}


def load_blip(model_id, device):
    from transformers import BlipForConditionalGeneration, BlipProcessor
    proc = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device).eval()
    return proc, model


def caption(paths, proc, model, device, batch=16, max_new_tokens=24):
    import torch
    from PIL import Image
    out = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        ims = [Image.open(p).convert("RGB") for p in chunk]
        inp = proc(images=ims, return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model.generate(**inp, max_new_tokens=max_new_tokens, num_beams=3)
        out.extend(t.strip() for t in proc.batch_decode(ids, skip_special_tokens=True))
    return out


def images_of(d):
    return sorted(p for p in glob.glob(os.path.join(d, "*"))
                  if p.lower().endswith(IMG_EXTS))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/benchmark_dataset")
    ap.add_argument("--cifc", default="data/CIFC/datasets")
    ap.add_argument("--out", default="", help="katalog na captiony; pusty = tylko raport")
    ap.add_argument("--model", default="Salesforce/blip-image-captioning-base")
    ap.add_argument("--calibrate", action="store_true",
                    help="tylko 12 zdjec wspolnych z CIFC, nasz caption obok ich")
    ap.add_argument("--concepts", default="", help="ogranicz do tych katalogow, po przecinku")
    a = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[cap] {a.model} na {device}", flush=True)
    proc, model = load_blip(a.model, device)

    if a.calibrate:
        n_ok = 0
        for cc_dir, (cifc_dir, stem_map) in OVERLAP.items():
            paths = images_of(os.path.join(a.data, cc_dir))
            caps = caption(paths, proc, model, device)
            print(f"\n=== {cc_dir}  (CIFC: {cifc_dir})", flush=True)
            for p, c in zip(paths, caps):
                stem = os.path.splitext(os.path.basename(p))[0]
                their_stem = stem_map.get(stem, stem) if stem_map else stem
                tp = os.path.join(a.cifc, "caption", cifc_dir, their_stem + ".txt")
                theirs = open(tp).read().strip() if os.path.exists(tp) else "(brak)"
                print(f"  CIFC : {theirs}")
                print(f"  nasz : {c}")
                n_ok += 1
        print(f"\n[cap] porownano {n_ok} par. Oceniamy dlugosc, obecnosc przymiotnika "
              f"przed slowem klasy i to, czy opisywane jest tlo.", flush=True)
        return

    concepts = ([c for c in a.concepts.split(",") if c] or
                sorted(d for d in os.listdir(a.data)
                       if os.path.isdir(os.path.join(a.data, d))))
    attr_hits = {}
    total = 0
    for cid in concepts:
        paths = images_of(os.path.join(a.data, cid))
        if not paths:
            print(f"[cap] {cid}: brak zdjec, pomijam", flush=True)
            continue
        caps = caption(paths, proc, model, device)
        if a.out:
            od = os.path.join(a.out, cid)
            os.makedirs(od, exist_ok=True)
            for p, c in zip(paths, caps):
                stem = os.path.splitext(os.path.basename(p))[0]
                with open(os.path.join(od, stem + ".txt"), "w") as f:
                    f.write(c)
        total += len(caps)
        # przymiotnik przed slowem klasy: heurystyka, tylko do wypelnienia attr_strip
        head = re.sub(r"[0-9_]+$", "", cid).split("_")[-1] or cid
        pat = re.compile(r"\b(\w+)\s+" + re.escape(head) + r"\b", re.I)
        hits = sorted({m.group(0).lower() for c in caps for m in [pat.search(c)] if m})
        if hits:
            attr_hits[cid] = hits
        print(f"[cap] {cid}: {len(caps)} | np. {caps[0]!r}", flush=True)

    print(f"\n[cap] {total} captionow"
          + (f" -> {a.out}" if a.out else " (nic nie zapisano)"), flush=True)
    if attr_hits:
        print("\n[cap] fraza przed slowem klasy, kandydaci do attr_strip:", flush=True)
        for cid, hits in attr_hits.items():
            print(f"  {cid}: {', '.join(hits[:4])}", flush=True)


if __name__ == "__main__":
    main()
