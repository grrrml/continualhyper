"""Generacja z checkpointow METODY BENCHMARKU (CIDM) do NASZEGO ukladu katalogow.

Po co. Ich `inference.py` generuje poprawnie, ale w swoim ukladzie
(`<out>/<method>/<replace_prompt>/samples`), a nasz `cifc_metrics.py` czyta
`<root>/after_task{k}/task{j}_{cid}/{samples,prompts.json}`. Oba zapisuja `prompts.json`
w TYM SAMYM formacie (lista {"<numer>": "<tekst kandydata>"}), a ich `concept_name` to
slowo klasy, czyli dokladnie nasza konwencja kandydata do CLIP-T. Rozjazd jest wiec
wylacznie w sciezkach i ten skrypt go zasypuje, nie dotykajac ich kodu.

Co jest zachowane z ich protokolu, bo o to chodzi w tym porownaniu:
  * ich `inference.py` bez zadnej zmiany, wolany jako podproces;
  * alpha 0.8 -- ich domyslna wartosc z `inference.py`, czyli punkt pracy z artykulu;
  * ich pliki promptow z `datasets/evaluation_prompts/test_<kategoria>.txt`, te same,
    ktore czyta nasz `gen_cifc.py`;
  * fuzja adapterow zadan 1..k+1 przy ocenie checkpointu po zadaniu k, bo ich metoda
    trzyma adapter na koncept i skleja je przy generacji.

Uwaga na koszt: ich skrypt laduje caly pipeline przy kazdym wywolaniu, a my wolamy go raz
na komorke (checkpoint, koncept). Dla samego wiersza modelu koncowego to 10 wywolan,
dla pelnej macierzy 55.

Run (wiersz modelu koncowego dla dziesieciu konceptow benchmarku):
  python scripts/_cidm_gen.py --cidm_root data/CIFC --output_dir output_bench10 \
      --stream data/CIFC/datasets/data_cfgs/task10.json --tasks 10 --final_only \
      --out_root outputs/cidm_bench10/final --per_prompt 10
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

# kategoria promptow per koncept benchmarku; te same pliki czyta nasz gen_cifc.py
CATEGORY = {
    "dog": "pet", "cat": "pet", "dog2": "pet", "cat2": "pet",
    "duck toy": "plushy", "backpack": "plushy", "teddy bear": "plushy",
    "painting": "style", "drawing": "style", "ink painting": "style",
}
# concept_id w naszej konwencji, po kolei jak zadania 1..10 w ich task10.json
CIDS = ["cifc_dog", "cifc_duck_toy", "cifc_cat", "cifc_backpack", "cifc_teddybear",
        "cifc_painting", "cifc_dog2", "cifc_drawing", "cifc_cat2", "cifc_ink_painting"]


def category_of(concept_name, idx):
    if concept_name in CATEGORY:
        return CATEGORY[concept_name]
    # cc101 i reszta: rzeczowniki traktujemy jak obiekty
    return "object"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cidm_root", default="data/CIFC", help="katalog z ich train.py/inference.py")
    ap.add_argument("--output_dir", default="output_bench10",
                    help="gdzie leza ICH checkpointy, wzglednie do --cidm_root")
    ap.add_argument("--stream", required=True, help="ich json z konceptami (task10.json)")
    ap.add_argument("--tasks", type=int, required=True, help="ile zadan w strumieniu")
    ap.add_argument("--out_root", required=True, help="nasz katalog wynikowy")
    ap.add_argument("--per_prompt", type=int, default=10, help="obrazow na prompt")
    ap.add_argument("--alpha", type=float, default=0.8, help="ich domyslna waga adaptera")
    ap.add_argument("--sd", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--final_only", action="store_true",
                    help="tylko checkpoint po ostatnim zadaniu (wiersz modelu koncowego)")
    ap.add_argument("--eval_first", type=int, default=0,
                    help="oceniaj tylko pierwsze N konceptow strumienia (0 = wszystkie). "
                         "Fuzja adapterow zostaje pelna, 1..k+1 -- ogranicza sie WYLACZNIE "
                         "zbior mierzony. Potrzebne do krzywej retencji: w kazdym punkcie "
                         "T mierzymy te same dziesiec konceptow, wiec koszt punktu nie "
                         "rosnie z T (10 komorek zamiast T).")
    ap.add_argument("--cids_from", default=None,
                    help="nasz config yaml, z ktorego brac concept_id; MUSI byc ten sam, "
                         "ktorym potem liczymy metryki, bo `cifc_metrics.py` sklada sciezke "
                         "jako after_task{k}/task{j}_{concept_id} i inaczej nic nie znajdzie")
    a = ap.parse_args()

    cids = CIDS
    if a.cids_from:
        import yaml
        cids = [c["concept_id"] for c in
                yaml.safe_load(open(os.path.abspath(a.cids_from), encoding="utf-8"))["concepts"]]

    root = os.path.abspath(a.cidm_root)
    stream = json.load(open(os.path.abspath(a.stream), encoding="utf-8"))
    if len(stream) < a.tasks:
        raise SystemExit(f"{a.stream} ma {len(stream)} konceptow, a --tasks to {a.tasks}")
    out_root = os.path.abspath(a.out_root)
    tmp = os.path.join(out_root, "_tmp")

    ks = [a.tasks - 1] if a.final_only else list(range(a.tasks))
    bs = min(a.per_prompt, 5)
    it = max(1, round(a.per_prompt / bs))

    for k in ks:
        # adaptery zadan 1..k+1, sciezki wzgledne do ich katalogu, jak w ich task10.json
        cfg = []
        for j in range(k + 1):
            c = dict(stream[j])
            c["lora_path"] = f"./{a.output_dir}/task_{j + 1}/models/edlora_model-latest.pth"
            cfg.append(c)
        # Nazwa MUSI byc unikalna per bieg. Wczesniej byla `_cfg_k{k}.json`, czyli
        # zalezna tylko od k -- a trzy macierze CIDM (ziarna 0, 1, 2) chodza rownolegle
        # na TYM SAMYM `data/CIFC`, wiec jedna kasowala plik, ktorego druga jeszcze
        # uzywala: `FileNotFoundError` na `os.remove` (bieg 3186848, po 12 min).
        # PID wystarcza, bo kolizja jest miedzy procesami na jednym wezle lub klastrze.
        cfg_path = os.path.join(root, f"_cfg_k{k}_p{os.getpid()}.json")
        json.dump(cfg, open(cfg_path, "w"), indent=1)

        n_eval = min(k + 1, a.eval_first) if a.eval_first else k + 1
        for j in range(n_eval):
            cid = cids[j] if j < len(cids) else f"task{j:02d}"
            dst = os.path.join(out_root, f"after_task{k:02d}", f"task{j:02d}_{cid}")
            if os.path.exists(os.path.join(dst, "prompts.json")):
                print(f"[cidm-gen] k={k} j={j}: gotowe, pomijam", flush=True)
                continue
            cat = category_of(stream[j]["concept_name"], j)
            txt = f"./datasets/evaluation_prompts/test_{cat}.txt"
            # przez shim, nie wprost: ich inference.py przewraca sie na attn1 na wlasnej
            # przypietej wersji diffusers -- powod i uzasadnienie w _cidm_attn1_shim.py
            shim = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "_cidm_attn1_shim.py")
            cmd = [sys.executable, "-u", shim,
                   "--concept_cfg", cfg_path,
                   "--text_path", txt,
                   "--pretrained_path", a.sd,
                   "--output_path", tmp,
                   "--replace_prompt", stream[j]["replace_mapping"],
                   "--method", "ours",
                   "--alpha", str(a.alpha),
                   "--batch_size", str(bs),
                   "--iter", str(it)]
            print(f"[cidm-gen] k={k} j={j} ({cid}, {cat}): {' '.join(cmd[-8:])}", flush=True)
            r = subprocess.run(cmd, cwd=root)
            if r.returncode != 0:
                raise SystemExit(f"inference.py padl dla k={k} j={j}")
            src = os.path.join(tmp, "ours", stream[j]["replace_mapping"])
            if not os.path.isdir(src):
                raise SystemExit(f"brak wyjscia {src}")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(src, dst)
        os.remove(cfg_path)
    shutil.rmtree(tmp, ignore_errors=True)
    print("CIDM_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
