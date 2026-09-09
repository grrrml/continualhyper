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

CO POKAZALA PIERWSZA KALIBRACJA i co z tego wynika dla tego skryptu:

    CIFC : cat standing on a set of stairs        CIFC : teddy bear sitting on a rock in the woods
    nasz : a small kitten walking up a set of stairs   nasz : a brown teddy bear sitting on a rock

  1. Rodzajnik na poczatku. CIFC go nie ma, my mielismy. Kazdy token z przodu przesuwa
     pozycje wszystkich pozostalych, a trening liczy po captionie maske span-u tokenow --
     wiec obcinamy wiodace "a "/"an "/"the " (`_clean`).
  2. Wazniejsze: dryf rzeczownika glownego ("kitten" tam, gdzie slowem klasy jest "cat").
     To nie kosmetyka. `data.ConceptDataset._caption` podmienia slowo klasy na fraze
     identyfikatora, a gdy slowa klasy w captionie NIE MA, DOKLEJA fraze z przodu:
     "V1 cat, a small kitten walking up a set of stairs". Adapter uczylby sie wtedy jednego
     konceptu pod dwiema konkurencyjnymi nazwami. Dlatego captionujemy WARUNKOWO: BLIP
     dostaje slowo klasy jako prompt i go kontynuuje, wiec slowo klasy jest w wyniku
     z definicji, a nie z nadziei. Skrypt sprawdza to i tak po fakcie (`cw in caption`,
     dokladnie ten sam test, ktory zrobi `_caption`) i liczy wpadki per koncept --
     captioning warunkowy zmniejsza ryzyko, nie kasuje go.

Uwaga o atrybutach: BLIP wstawia przymiotniki ("fluffy cat"), tak samo jak captiony CIFC.
Metoda strippuje je przez `concepts[].attr_strip`, bo atrybut opisany tekstem to atrybut,
ktorego adapter sie nie nauczy. Skrypt raportuje wiec, w ktorych captionach przed slowem
klasy stoi przymiotnik. Po przejsciu na captioning warunkowy caption ZACZYNA sie od slowa
klasy, wiec ta lista powinna byc juz prawie pusta -- wpis w niej znaczy, ze slowo klasy
wystapilo w zdaniu drugi raz, i wtedy warto na niego spojrzec.

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

# --- slowo klasy z nazwy katalogu ----------------------------------------------------
# JEDNO zrodlo prawdy: `class_word_of` nizej. Prompt podany BLIP-owi i pole `class_word`
# w configu musza byc tym samym napisem, bo `_caption` szuka slowa klasy w captionie
# dokladnie tak: `cls in cap`, z uwzglednieniem wielkosci liter.
#
# UWAGA: configs/phaseT/T50_cc101.yaml zostal napisany RECZNIE -- tamte 50 wartosci
# class_word powstalo z reguly opisanej w tym configu w komentarzu "CLASS WORDS", nie
# z tego kodu. Ta funkcja MUSI sie z nimi zgadzac co do znaku i zostala z nimi zestawiona
# programowo (wszystkie 50 sie zgadza). Jesli ruszasz ktorakolwiek tabele nizej, przejedz
# config jeszcze raz: rozjazd niczego nie wywali, tylko cicho zepsuje trening -- patrz
# punkt 2 w docstringu.
#
# Regula: wez czesc po pierwszym "_", utnij koncowe cyfry indeksu i uzyj jej, gdy nazywa
# obiekt. Gdy nie nazywa (sam indeks, atrybut, nazwa wlasna) -- wez slowo klasy prefiksu.

# Slowo klasy prefiksu. Tylko te prefiksy, ktorym ktorys ze 101 katalogow naprawde tego
# fallbacku potrzebuje; nieznany prefiks konczy sie wyjatkiem, a nie zgadywaniem.
_PREFIX_CLASS = {
    "actionfigure": "action figure",     # actionfigure_1..3: reszta to sam indeks
    "dish": "dish",                      # dish_1, dish_2
    "flower": "flower",                  # flower_1, flower_2
    "instrument": "musical instrument",  # instrument_1, instrument_music1..3
    "person": "person",                  # person_1..3
    "plushie": "plushie",                # plushie_2, plushie_pink, plushie_happysad, ...
    "toy": "toy figure",                 # toy_pikachu1
}
# Reszta nazwy nie nazywa obiektu: atrybut ("plushie_pink") albo slowo obok rzeczy
# ("instrument_music1" -- muzyka to nie instrument).
_NOT_OBJECT = {"music", "pink", "happysad", "sadangry"}
# Nazwa wlasna nie moze byc slowem klasy: znak towarowy w slowie klasy oddaje tozsamosc
# enkoderowi tekstu, a to jest dokladnie ta praca, ktora ma wykonac adapter.
_PROPER_NOUNS = {"pikachu"}
# Zlepki dwoch osobnych slow. Pojedyncze slowa slownikowe ZOSTAJA calo: houseplant,
# lighthouse, waterfall, corkscrew, keychain, backpack, sunglasses, motorbike, earring.
_SPLIT = {"teddybear": "teddy bear", "woodenpot": "wooden pot", "tablechair": "table chair"}
# plushie_<x>: <x> mowi, co maskotka przedstawia, wiec zostaja oba czlony ("cow plushie").
# Wyjatek: reszta, ktora juz sama nazywa maskotke ("teddy bear", nie "teddy bear plushie").
_ALREADY_PLUSH = {"teddybear"}


def class_word_of(concept_dir):
    """Nazwa katalogu CC101 -> slowo klasy: 'pet_cat1' -> 'cat', 'toy_bear' -> 'toy bear'."""
    prefix, _, rest = concept_dir.partition("_")
    rest = re.sub(r"\d+$", "", rest)
    if not rest or rest in _NOT_OBJECT or rest in _PROPER_NOUNS:
        if prefix not in _PREFIX_CLASS:
            raise ValueError(
                f"{concept_dir}: reszta nazwy nie nazywa obiektu, a prefiks '{prefix}' nie "
                f"ma slowa klasy w _PREFIX_CLASS. Dopisz je i sprawdz zgodnosc z "
                f"configs/phaseT/T50_cc101.yaml.")
        return _PREFIX_CLASS[prefix]
    word = _SPLIT.get(rest, rest)
    if prefix == "toy":                                     # toy_bear -> "toy bear"
        return f"toy {word}"
    if prefix == "plushie" and rest not in _ALREADY_PLUSH:  # plushie_cow -> "cow plushie"
        return f"{word} plushie"
    return word


_ARTICLE = re.compile(r"^(?:an?|the)\s+", re.I)


def _clean(text, cw):
    """Wyjscie BLIP-a -> rezim CIFC: bez wiodacego rodzajnika i bez echa promptu."""
    text = _ARTICLE.sub("", text.strip())
    # Captioning warunkowy potrafi powtorzyc prompt: "cat cat sitting on ...", "cat a cat
    # sitting on ...". \b na koncu, zeby "cat cathedral" nie zostalo sklejone w "cathedral".
    echo = re.compile(r"^(" + re.escape(cw) + r")(?:\s+(?:an?|the)?\s*\1\b)+", re.I)
    return echo.sub(r"\1", text).strip()


def load_blip(model_id, device):
    from transformers import BlipForConditionalGeneration, BlipProcessor
    proc = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device).eval()
    return proc, model


def caption(paths, proc, model, device, cw, batch=16, max_new_tokens=24):
    """Captiony warunkowane slowem klasy: BLIP dostaje `cw` jako prompt i go kontynuuje,
    wiec `cw` jest w wyniku, zamiast liczyc na to, ze model sam trafi w 'cat', nie
    'kitten'. Zwraca captiony juz przepuszczone przez `_clean`."""
    import torch
    from PIL import Image
    out = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        ims = [Image.open(p).convert("RGB") for p in chunk]
        # Ten sam prompt dla calej partii, wiec tokeny maja rowna dlugosc i padding nie
        # wystepuje. To nie jest kosmetyka: BLIP.generate obcina ostatni token wejscia
        # (input_ids[:, :-1]), wiec padding po prawej obcinalby pad zamiast [SEP].
        inp = proc(images=ims, text=[cw] * len(chunk), return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model.generate(**inp, max_new_tokens=max_new_tokens, num_beams=3)
        # przy captioningu warunkowym dekodowanie zwraca prompt razem z kontynuacja
        out.extend(_clean(t, cw) for t in proc.batch_decode(ids, skip_special_tokens=True))
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
        n_pairs = n_has = our_words = n_theirs = their_words = 0
        for cc_dir, (cifc_dir, stem_map) in OVERLAP.items():
            cw = class_word_of(cc_dir)
            paths = images_of(os.path.join(a.data, cc_dir))
            caps = caption(paths, proc, model, device, cw)
            print(f"\n=== {cc_dir}  (CIFC: {cifc_dir}, slowo klasy: {cw!r})", flush=True)
            for p, c in zip(paths, caps):
                stem = os.path.splitext(os.path.basename(p))[0]
                their_stem = stem_map.get(stem, stem) if stem_map else stem
                tp = os.path.join(a.cifc, "caption", cifc_dir, their_stem + ".txt")
                theirs = open(tp).read().strip() if os.path.exists(tp) else None
                print(f"  CIFC : {theirs if theirs is not None else '(brak)'}")
                print(f"  nasz : {c}")
                n_pairs += 1
                n_has += cw in c        # ten sam test co data.ConceptDataset._caption
                our_words += len(c.split())
                if theirs is not None:
                    n_theirs += 1
                    their_words += len(theirs.split())
        if not n_pairs:
            print("\n[cap] zero zdjec do kalibracji -- sprawdz --data", flush=True)
            return
        theirs_avg = f"{their_words / n_theirs:.1f} (z {n_theirs})" if n_theirs else "(brak)"
        print(f"\n[cap] porownano {n_pairs} par | slowo klasy w {n_has}/{n_pairs} naszych "
              f"captionach | srednia dlugosc: nasze {our_words / n_pairs:.1f} slowa, "
              f"CIFC {theirs_avg}", flush=True)
        print("[cap] poza tym oceniamy przymiotnik przed slowem klasy i to, czy opisywane "
              "jest tlo.", flush=True)
        return

    concepts = ([c for c in a.concepts.split(",") if c] or
                sorted(d for d in os.listdir(a.data)
                       if os.path.isdir(os.path.join(a.data, d))))
    attr_hits = {}
    missing = {}
    total = 0
    for cid in concepts:
        paths = images_of(os.path.join(a.data, cid))
        if not paths:
            print(f"[cap] {cid}: brak zdjec, pomijam", flush=True)
            continue
        cw = class_word_of(cid)
        caps = caption(paths, proc, model, device, cw)
        if a.out:
            od = os.path.join(a.out, cid)
            os.makedirs(od, exist_ok=True)
            for p, c in zip(paths, caps):
                stem = os.path.splitext(os.path.basename(p))[0]
                with open(os.path.join(od, stem + ".txt"), "w") as f:
                    f.write(c)
        total += len(caps)
        # ten sam test, ktory zdecyduje w treningu: `cls in cap` z ConceptDataset._caption
        bad = [c for c in caps if cw not in c]
        if bad:
            missing[cid] = (len(bad), bad[0])
        # przymiotnik przed slowem klasy: heurystyka, tylko do wypelnienia attr_strip
        pat = re.compile(r"\b(\w+)\s+" + re.escape(cw) + r"\b", re.I)
        hits = sorted({m.group(0).lower() for c in caps for m in [pat.search(c)] if m})
        if hits:
            attr_hits[cid] = hits
        print(f"[cap] {cid}: {len(caps)} | {cw!r} | np. {caps[0]!r}", flush=True)

    print(f"\n[cap] {total} captionow"
          + (f" -> {a.out}" if a.out else " (nic nie zapisano)"), flush=True)
    if missing:
        n_bad = sum(k for k, _ in missing.values())
        print(f"\n[cap] UWAGA: {n_bad}/{total} captionow BEZ slowa klasy. Dla nich "
              f"data.ConceptDataset._caption doklei fraze identyfikatora z przodu, czyli "
              f"koncept dostanie dwie nazwy naraz:", flush=True)
        for cid, (k, ex) in sorted(missing.items()):
            print(f"  {cid}: {k} | np. {ex!r}", flush=True)
    else:
        print(f"[cap] slowo klasy obecne we wszystkich {total} captionach", flush=True)
    if attr_hits:
        print("\n[cap] fraza przed slowem klasy, kandydaci do attr_strip:", flush=True)
        for cid, hits in attr_hits.items():
            print(f"  {cid}: {', '.join(hits[:4])}", flush=True)


if __name__ == "__main__":
    main()
