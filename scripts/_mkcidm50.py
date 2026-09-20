"""Generuje 50 configow zadan dla KODU BENCHMARKU (data/CIFC, arXiv 2410.17594) z naszego
strumienia T50_mixed, zeby ich metode dalo sie wytrenowac na tej samej sekwencji co nasza.

Co sie podmienia wobec ich `options/cidm/task_1.yml`:
  * `data_dir` / `caption_dir` -- katalogi konceptu (sciezki wzgledne do data/CIFC, bo ich
    train.py odpalamy z tego katalogu; nasze dane leza wyzej, wiec ida jako ../../);
  * `replace_mapping` -- slowo klasy -> dwa nowe tokeny konceptu (ich ED-LoRA uczy pary);
  * `new_concept_token` / `initializer_token` -- para tokenow i ich inicjalizacja: pierwszy
    losowy (ich `<rand-0.013>`), drugi ze slowa klasy. Slowo klasy wielowyrazowe skracamy do
    ostatniego wyrazu, bo initializer musi byc POJEDYNCZYM tokenem;
  * `batch_size_per_gpu: 2` -- oni licza na dwoch GPU po 1, my na jednym; `total_iter` u nich
    to `len(dataset)/total_batch_size`, wiec przy 2 liczba krokow zostaje taka sama;
  * walidacja wizualna wylaczona (`val_during_save: false`) -- to tylko podglad, a kosztuje
    kilka minut na zadanie razy 50.

Run:  python scripts/_mkcidm50.py --stream configs/phaseT/T50_mixed.yaml \
          --template data/CIFC/options/cidm/task_1.yml --out_dir data/CIFC/options/cidm50
"""
import argparse
import os
import re

import yaml

CIFC_REL = "../.."          # z data/CIFC do korzenia repo


def tokens(k):
    return f"<c{k:02d}a>", f"<c{k:02d}b>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="configs/phaseT/T50_mixed.yaml")
    ap.add_argument("--template", default="data/CIFC/options/cidm/task_1.yml")
    ap.add_argument("--out_dir", default="data/CIFC/options/cidm50")
    ap.add_argument("--val_prompts", default="./datasets/validation_prompts/test_dog.txt")
    a = ap.parse_args()

    stream = yaml.safe_load(open(a.stream, encoding="utf-8"))
    tpl = open(a.template, encoding="utf-8").read()
    os.makedirs(a.out_dir, exist_ok=True)
    rows = []
    for k, c in enumerate(stream["concepts"], start=1):
        ta, tb = tokens(k)
        cls = c["class_word"]
        init = cls.split()[-1]                       # initializer musi byc jednym tokenem
        # ...a "jeden token" znaczy jeden token CLIP-a, nie jedno slowo. Ich
        # trainer_edlora.py:176 rzuca ValueError, gdy tokenizer rozbije initializer na
        # wiecej niz jeden id. Sprawdzone w vocab.json z SD1.5: "houseplant" i "plushie"
        # sie rozbijaja, pozostale 23 slowa klas z T50_mixed nie. Podmieniamy je na
        # najblizsze jednotokenowe slowo - initializer tylko inicjuje embedding drugiego
        # tokenu konceptu, wiec liczy sie sasiedztwo semantyczne, nie doslownosc.
        init = {'houseplant': 'plant', 'plushie': 'plush'}.get(init, init)
        img = os.path.join(CIFC_REL, c["images_dir"])
        cap = os.path.join(CIFC_REL, c["caption_dir"]) if c.get("caption_dir") else ""
        s = tpl
        s = re.sub(r"^name: .*$", f"name: task_{k}", s, flags=re.M)
        s = re.sub(r"^    data_dir: .*$", f"    data_dir: {img}", s, flags=re.M)
        s = re.sub(r"^    caption_dir: .*$", f"    caption_dir: {cap}", s, flags=re.M)
        s = re.sub(r"^    use_caption: .*$", f"    use_caption: {'true' if cap else 'false'}", s, flags=re.M)
        s = re.sub(r"^      dog: <dog1> <dog2>$", f"      {cls}: {ta} {tb}", s, flags=re.M)
        s = re.sub(r"^    prompts: .*$", f"    prompts: {a.val_prompts}", s, flags=re.M)
        s = re.sub(r"^  new_concept_token: .*$", f"  new_concept_token: {ta}+{tb}", s, flags=re.M)
        s = re.sub(r"^  initializer_token: .*$", f"  initializer_token: <rand-0.013>+{init}", s, flags=re.M)
        s = re.sub(r"^    batch_size_per_gpu: 1$", "    batch_size_per_gpu: 2", s, flags=re.M)
        s = re.sub(r"^  val_during_save: .*$", "  val_during_save: false", s, flags=re.M)
        s = re.sub(r"^  compose_visualize: .*$", "  compose_visualize: false", s, flags=re.M)
        # enable_xformers: na Heliosie xformers jest swiadomie pominiety
        # (slurm/clusters/helios.sh, SKIP_PACKAGES): kolo 0.0.28 jest zbudowane pod
        # torch 2.3.1 i pod 2.6 nie laduje rozszerzen CUDA. Ich trainer robi z tej flagi
        # tylko `assert is_xformers_available()` (trainer_edlora.py:52) i nigdzie nie
        # wlacza memory-efficient attention jawnie, a lib/models/edlora.py ma pelny
        # fallback na natywna uwage. Wylaczenie flagi nie zmienia numeryki, a bez tego
        # zadanie 1 padloby na assercie zaraz po starcie.
        s = re.sub(r"^  enable_xformers: .*$", "  enable_xformers: false", s, flags=re.M)
        out = os.path.join(a.out_dir, f"task_{k}.yml")
        open(out, "w", encoding="utf-8", newline="\n").write(s)
        rows.append((k, c["concept_id"], cls, f"{ta} {tb}"))
    man = os.path.join(a.out_dir, "mapping.tsv")
    with open(man, "w", encoding="utf-8", newline="\n") as f:
        f.write("task\tconcept_id\tclass_word\ttokens\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"zapisane {len(rows)} configow w {a.out_dir}, mapowanie w {man}")
    for r in rows[:3] + rows[-2:]:
        print("  ", r)


if __name__ == "__main__":
    main()
