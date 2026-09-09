"""Buduje strumien T=50: dziesiatka CIFC na poczatku, potem 40 konceptow CustomConcept101.

Dlaczego tak, a nie 50 konceptow z jednego zbioru: punkt T=10 tej krzywej ma BYC wynikiem
benchmarkowym, a nie punktem do niego podobnym. Skoro pierwsze dziesiec zadan, ich kolejnosc
i wszystkie hiperparametry sa identyczne jak w P_paper, to trajektoria treningu dla zadan 0-9
jest bitowo identyczna -- `n_tasks` nie zuzywa dodatkowej losowosci przy budowie sieci
(task_emb to jedynki, ortho_basis to zera, klucze maja wlasny generator 1234+k). Checkpoint
po zadaniu 9 musi wiec odtworzyc headline co do znaku; jesli nie odtworzy, cos jest zle i
mamy to od razu.

Wykluczone z CC101, kazde z powodu:
  * pet_cat1, plushie_teddybear -- BAJT W BAJT te same zdjecia co cifc/cat i cifc/teddybear
    (sprawdzone md5). Bez tego ten sam koncept wystapilby w strumieniu dwa razy pod dwoma
    kluczami, co jest najgorszym mozliwym testem ortogonalnosci kluczy: nie zmierzylibysmy
    interferencji, tylko wlasny blad.
  * scene_* -- barn, canal, castle, garden, lighthouse, waterfall, sculpture. To sceny, nie
    personalizowane obiekty. Prompty ewaluacyjne dla nich nie istnieja ("a broken canal",
    "a child is playing a barn"), a captioning warunkowy wymusza rzeczownik glowny, ktorego
    BLIP by tam sam nie postawil.

Kategoria ewaluacyjna: pet_* i person_* dostaja test_pet.txt, reszta CC101 nowy
test_object.txt. Dziesiatka CIFC zachowuje swoje wlasne kategorie, w tym trzy style.

UWAGA o stylach: CC101 nie ma ani jednego konceptu stylu (101 katalogow, 16 prefiksow, same
obiekty, sceny, zwierzeta i osoby). Strumien ma wiec 3 style na 50, wszystkie w pierwszej
dziesiatce. Krzywa powyzej T=10 mowi o konceptach obiektowych i tylko o nich.

Run:  python scripts/_build_t50_mixed.py
"""
import os
import re
import sys

sys.path.insert(0, ".")
from scripts._caption_cc101 import class_word_of  # noqa: E402  jedno zrodlo slowa klasy

P_BASE = "configs/phaseP/P_paper.yaml"
OUT = "configs/phaseT/T50_mixed.yaml"
CC101 = "data/benchmark_dataset"
CAPTIONS = "data/cc101_caption"
N_CC101 = 40

# Pelna lista 101 katalogow CustomConcept101, wpisana na stale: zbior jest niezmienny,
# a `data/benchmark_dataset` to symlink na klaster, wiec skrypt ma dzialac takze lokalnie.
CC101_DIRS = [
    "actionfigure_1", "actionfigure_2", "actionfigure_3", "decoritems_houseplant1",
    "decoritems_houseplant2", "decoritems_houseplant3", "decoritems_lamp1", "decoritems_vase2",
    "decoritems_woodenpot", "dish_1", "dish_2", "flower_1",
    "flower_2", "furniture_chair1", "furniture_chair2", "furniture_chair3",
    "furniture_sofa1", "furniture_sofa2", "furniture_table1", "instrument_1",
    "instrument_music1", "instrument_music2", "instrument_music3", "jewelry_earring",
    "jewelry_ring", "luggage_backpack1", "luggage_purse1", "luggage_purse2",
    "luggage_purse3", "luggage_purse4", "person_1", "person_2",
    "person_3", "pet_cat1", "pet_cat2", "pet_cat3",
    "pet_cat4", "pet_cat5", "pet_cat6", "pet_cat7",
    "pet_dog1", "pet_dog2", "pet_dog3", "pet_dog4",
    "plushie_2", "plushie_bunny", "plushie_cow", "plushie_dice",
    "plushie_happysad", "plushie_lobster", "plushie_panda", "plushie_penguin",
    "plushie_pink", "plushie_sadangry", "plushie_teddybear", "plushie_tortoise",
    "plushie_unicorn", "scene_barn", "scene_canal", "scene_castle",
    "scene_garden", "scene_lighthouse", "scene_sculpture1", "scene_waterfall",
    "things_book", "things_book2", "things_bottle1", "things_corkscrew",
    "things_cup1", "things_cup2", "things_cup3", "things_headphone1",
    "things_headphone2", "things_helmet", "things_keychain1", "toy_bear",
    "toy_gnome", "toy_pikachu1", "toy_tablechair", "toy_unicorn",
    "transport_bike", "transport_car1", "transport_car10", "transport_car11",
    "transport_car2", "transport_car3", "transport_car4", "transport_car5",
    "transport_car6", "transport_car7", "transport_car8", "transport_car9",
    "transport_motorbike1", "transport_tank", "wearable_glasses", "wearable_jacket1",
    "wearable_jacket2", "wearable_shoes1", "wearable_shoes2", "wearable_sunglasses1",
    "wearable_sunglasses2",
]

# Te same zdjecia co w CIFC (md5), wiec koncept powtorzylby sie w strumieniu.
DUPLICATES = {"pet_cat1", "plushie_teddybear"}


def cc101_dirs():
    """Nazwy katalogow CC101 w kolejnosci round-robin po prefiksach.

    Round-robin, a nie alfabetycznie: alfabetycznie dostalibysmy siedem kotow pod rzad,
    czyli dokladnie taki ciag jednej klasy, jakiego kolejnosc zadan w CL miec nie powinna.
    """
    names = sorted(d for d in CC101_DIRS
                   if d not in DUPLICATES and not d.startswith("scene_"))
    buckets = {}
    for n in names:
        buckets.setdefault(n.split("_")[0], []).append(n)
    out, i = [], 0
    while len(out) < N_CC101:
        added = False
        for p in sorted(buckets):
            if i < len(buckets[p]):
                out.append(buckets[p][i])
                added = True
                if len(out) == N_CC101:
                    return out
        if not added:
            raise RuntimeError(f"za malo konceptow: {len(out)} < {N_CC101}")
        i += 1
    return out


def main():
    base = open(P_BASE, encoding="utf-8").read()
    head, _, tail = base.partition("concepts:\n")
    body, _, rest = tail.partition("training:\n")

    lines = [head.replace("output_dir: ./outputs/phaseP/P_paper",
                          "output_dir: ./outputs/phaseT/T50_mixed"),
             "concepts:\n",
             "# --- zadania 0-9: dziesiatka CIFC, dokladnie jak w P_paper ---\n",
             body.rstrip("\n") + "\n",
             "# --- zadania 10-49: CustomConcept101, captiony BLIP w rezimie CIFC ---\n"]

    for d in cc101_dirs():
        cw = class_word_of(d)
        cat = "pet" if d.startswith(("pet_", "person_")) else "object"
        lines.append(
            f"- concept_id: cc101_{d}\n"
            f"  images_dir: {CC101}/{d}\n"
            f"  caption_dir: {CAPTIONS}/{d}\n"
            f"  class_word: {cw}\n"
            f"  category: {cat}\n"
            f"  identifier: ''\n"
            f"  prompt: a photo of {cw}\n")

    lines.append("training:\n")
    lines.append(rest.replace("  diagnostic_freq: 100", "  diagnostic_freq: 0")
                     .replace("name: P_paper", "name: T50_mixed"))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(lines))

    n = len(re.findall(r"^- concept_id:", "".join(lines), re.M))
    print(f"{OUT}: {n} konceptow")
    if n != 50:
        raise SystemExit(f"BLAD: oczekiwano 50, jest {n}")


if __name__ == "__main__":
    main()
