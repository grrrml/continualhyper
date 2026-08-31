# ContinualHyper — raport stanu projektu

> Żywy dokument-pamięć projektu: syntetyczny obraz "gdzie jesteśmy i skąd to wiemy".
> Aktualizowany po każdym domknięciu wątku (werdykt, faza, decyzja ramowa).
> Szczegółowy dziennik pomiarów i odrzuceń: `assets/STATUS.md`. Stan na: **2026-08-31**.

---

## 1. Teza i rama

**Teza:** jedna hipersieć generująca LoRA per koncept (z deterministycznego klucza) rozwiązuje
ciągłą personalizację dyfuzji lepiej niż klasyczne mechanizmy CL — przy pamięci **O(1)**
(stała hipersieć) zamiast **O(T)** (magazyn adapterów per task) i przy praktycznie zerowym
zapominaniu.

**Rama (decyzja 2026-08-10):** setting = **ścisłe CL** — model o ograniczonej pamięci, taski
sekwencyjnie, bez rosnącego składowania. Liga porównawcza: finetune, EWC, LwF, C-LoRA, L2DM
(wszystkie zaimplementowane wg prac źródłowych, wspólny backbone, strojone λ).
`lora_solo` (10 niezależnych LoRA + oracle) i CIDM (pamięć O(T), miękki routing) = referencje
**poza settingiem**. Benchmark: CIFC/CIDM (10 konceptów: 7 obiektów + 3 style, 4–7 zdjęć każdy);
metryki TA (CLIP-T), IA (CLIP-I, protokół CIDM), **DINO** (główna oś tożsamości, ViT-S/16 —
ten sam ekstraktor co ich `evaluate.py`); forgetting z pełnej macierzy 55 komórek (peak−final).

## 2. Metoda (wersja główna)

**`F_base`**: wejście hipersieci = 128-wym. losowy wektor ortonormalny per task (deterministyczny
z seeda 1234+k) → per-warstwowe heady (h50) → LoRA rank 4 na `attn2.{to_q,to_k,to_v,to_out.0}`
(64 warstwy SD-1.5), maska tokenowa ON, regularyzacja von Oswalda β=100 (kotwice), 400 kroków/task,
**21.1M parametrów** niezależnie od liczby tasków. Punkt pracy: reguła 3c (max DINO przy
TA≥0.748 ∧ IA≥0.780). Inferencja: 1 forward hipersieci na generację (delta cache'owana), potem
czysty UNet — koszt niezależny od T (CIDM: pętla po wszystkich adapterach + routing).

**Grounding (nowe, GO 2026-08-20):** `+GSA hipersieciowa` — hipersieć generuje 4 tokeny
groundingu z (klucz ⊕ Fourier(ramka)) + FiLM z klucza; wspólna wąska uwaga czytająca per attn2
(projekcje 64-wym., ~1.2M) × **analityczna** maska inside(pos, ramka) × tanh(gate) zero-init.
Trening: segmentowana wklejka (isnet, próg 0.15) całego obiektu na ~100 naturalnych teł
(generowanych raz bazowym SD), strata bez maski, t∈[T/2,T) na krokach z ramką. Razem +1.5M.

## 3. Wyniki główne — DOMKNIĘTE

### SD-1.5 (liga, matched-TA, 3 seedy)
- **F_base @TA=0.748: DINO 0.6415±0.0012, IA 0.8054±0.0017** (3 seedy; szum międzyseedowy
  mechanizmów: 0.0091).
- Przewaga DINO nad ligą (pełne krzywe, zwycięskie λ): **LwF +0.070 | EWC +0.081 | C-LoRA +0.122 |
  finetune +0.238 | L2DM +0.377** — 6–40× ponad szum.
- **Forgetting (pełne macierze): F_base 0.0015** vs LwF 0.018 | EWC 0.036 | C-LoRA 0.037 |
  L2DM 0.159 | finetune 0.236 — 12–150× mniej.
- Wobec CIDM: reprodukcja ich kodu ±0.005 (walidacja protokołu); flagship w pełni dopasowanym
  protokołem: **+0.036 IA, +0.065 DINO** (prefiksowany; natywnie +0.020/+0.042 — raportujemy oba).
  Ciekawostka: prefiksy (język ICH captionów) POGARSZAJĄ CIDM — kolizja z uczonymi tokenami.

### SDXL (port 1:1, bez strojenia)
- **@50, 3 seedy, s0.7: TA 0.7921±0.0091 | IA 0.7988±0.0050 | DINO 0.6262±0.0076** —
  **parytet** z opublikowanym CIDM-SDXL (0.795/0.800) na ich protokole. (Kod CIDM-SDXL nie
  istnieje publicznie — repo/branche/PR-y/fork współautora sprawdzone; wiersz = liczby z pracy.)
- **Forgetting: DINO +0.0012, CLIP-I +0.0019** (pełna macierz; grid: `assets/figures/fgt_grid_F_sdxl.jpg`).
- Liga CL na SDXL (nasz harness, @10): najlepszy baseline LwF DINO 0.519 vs nasze 0.634 —
  **+0.115 DINO, +0.056 IA**; 4/5 metod policzone (L2DM: OOM na A100 40GB — decyzja w toku).
- Hipernet SDXL: 87.5M (280 warstw), hiperparametry przeniesione 1:1.

### Pamięć (uczciwie)
@T=10 jesteśmy 5× więksi od magazynu CIDM (21.1M vs 4.26M); teza to **skalowanie O(1) vs O(T)**
(próg opłacalności ~49 konceptów) — twierdzenie "dowolnie wiele tasków" wymaga sweepu h @T=35
(niezrobione, nie stawiać).

## 4. Ramki / grounding — saga i przełom

Decyzja użytkownika: ramki = must-have, hipersieć w centrum. Siedem wariantów, ten sam
pre-rejestrowany protokół (sonda połówkowa 84 próbki, próg 80%; test A/B lewo/prawo/bez):

| # | wariant | sonda | \|L−R\| | lekcja |
|---|---|---|---|---|
| 1 | ramka→wagi (box_emb w hipersieci) | 48.8% | — | delta wag jest globalna, nie adresuje "gdzie" |
| 2 | bramka σ(⟨q,Ke⟩)·Ve w attn2 | 48.8% | 0.005 | q_i w SD nie niesie pozycji |
| 3 | 5× więcej augmentacji (long) | 47.6% | 0.002 | to nie głód sygnału |
| 4 | uczona geometria (LayoutDiffusion-style) | 51.2% | 0.002 | pos/box_proj zostały na inicie — pętla gate↔geometria |
| 5 | objective v2 (seg-wklejka, strata pełna) | 50.0% | 0.003 | objective sam nie wystarcza |
| 6 | maska analityczna + skalarna bramka | 47.6% | 0.029 | adres działa, zastrzyk skalar·wektor za ubogi |
| 7 | **GSA hipersieciowa** | **84.5% GO** | **0.179** | **wąskim gardłem była STRUKTURA odczytu** |

**Lekcja mechanistyczna (materiał do artykułu):** przez 6 porażek problemem nie była amplituda
(gate'y zawsze ~0.02, także w wariancie GO) ani adresowanie (analityczna maska od kroku 0),
tylko **pojemność odczytu**: pozycja obrazu musi *czytać* z tokenów groundingu treść zależną
od własnego stanu (uwaga), a nie dostawać skalar·stały wektor. Do tego dwie pułapki
architektoniczne: martwy punkt podwójnego zero-initu (grad(gate)∝V·e=0 i grad(head)∝tanh(gate)=0)
i pętla startowa uczonej geometrii (gradient dławiony przez tanh(gate)≈0).

**Gałka κ na gate'ach przy inferencji** (odpowiednik lora_scale dla groundingu; stosowana
TYLKO przy podanej ramce — pełny kadr = κ=1, protokół bazowy nietknięty):
sonda połówkowa 84.5% (κ=1) → **94.0% (κ=2)**; ćwiartkowa (przypadek 25%): 40.5% → **59.5%
(κ=2)** → 54.8% (κ=4, artefakty) → 28.6% (κ=8, rozpad) — krzywa odwróconego U jak przy guidance.
Grid wizualny @κ=2 (`assets/figures/ground_grid_gsa_k2.jpg`): placement widoczny w ~25/28
komórek.

**Trzy koszty zaobserwowane @κ=2** (grid): (1) mały obiekt traci tożsamość (kaczka→blob),
(2) sporadyczne artefakty kolorystyczne, (3) spłaszczone tła. Wspólny rdzeń: zastrzyk działał
przez CAŁE odszumianie i w OBU gałęziach CFG. **Poprawki (inference-only, 2026-08-20):**
(a) harmonogram κ — grounding aktywny tylko przez początkową frakcję kroków
(`ground_sched_frac`; układ rozstrzyga się przy wysokim szumie, detale maluje czysty
model+LoRA; wzorzec GLIGEN γ-frac/ReGround); (b) grounding tylko w gałęzi warunkowej —
uncond liczony pod `no_lora()` jest teraz czysty, grounding wchodzi do różnicy guidance (×7.5
jak każdy warunek). **Sweep trade-offu po poprawkach (2026-08-20):** poprawki odblokowały skalowanie κ —
ćwiartki: (κ=2, bez harm.) 61.9% → (2, 0.4) 65.5% → (3, 0.3) 75.0% → **(4, 0.3) 89.3%**
przy DINO(cały obraz vs ref) 0.730 → 0.694. Przed poprawkami κ=4 NISZCZYŁO generację (54.8%,
bloby) — z harmonogramem 0.3 zastrzyk ustawia układ przy wysokim szumie i oddaje pędzel
czystemu modelowi. Kaczka @(4, 0.3): kształt/dziób/oczy zachowane (koniec blobów), ale
kolor zdryfował do oliwkowego — to jest zmierzony koszt −0.036 DINO. Kandydaci na punkt
pracy: (3, 0.3) zbalansowany lub (4, 0.3) placement-first; do zbadania (4–6, 0.2).
**Kryterium 2 pre-rejestracji: ZALICZONE (2026-08-20)** — eval pełnokadrowy @50: krzywa GSA
przesunięta wzdłuż trade-offu (s05: TA 0.762/IA 0.790/DINO 0.621; s07: 0.729/0.816/0.650;
s10: 0.687/0.826/0.647); przy matched-TA różnice wobec P_ground: DINO −0.003…−0.008,
IA ±0.005 — w granicach szumu jednoseedowego (0.0091). Punkt pracy 3c: s05.
**Test (a) — prompty ze sceną: tła wracają** przy zachowanym placemencie
(`ground_grid_gsa_k4_scene.jpg`) → płaskie tło było artefaktem gołego promptu × prior
kompozytów; retrening na bogatych tłach niepilny. Pozostałe skazy: kaczka blobowata @κ=4
(fix: κ per koncept, małe obiekty → κ=2), dog2/cat2 miejscami płaskie.
OBA kryteria GO spełnione. Dalej: drugi seed (formalizacja), potem kompozycja wielokonceptowa.

### Polerowanie GSA na Heliosie (2026-08-31) — dwa przecieki i uczciwa metryka

**(a) Przeciek atrybutu w captionach treningowych — POTWIERDZONY, fix w treningu.**
Captiony CIFC to dosłownie `yellow rubber duck toy sitting on a gravel surface` (wszystkie 4)
i `red backpack sitting on a rock in the woods` (wszystkie 6); `cat` ma `fluffy` w 1 z 5.
`P_ground_gsa.yaml` **nie ustawia `attr_strip`**, więc `src/data.py` nic nie zjadał i kolor
wchodził do modelu przez prompt. Gorzej: przy `token_mask_lora: true` delta LoRA aplikuje się
tylko na pozycjach tokenów `class_word`, więc tokeny `yellow`/`rubber` idą przez **zamrożone**
K/V — hipersieć koloru nie musiała się nauczyć **i nie mogła go dotknąć**. Po stronie
ewaluacji `gen_cifc.py` dokłada `eval_prefix`, więc artefakt był niewidoczny do momentu,
gdy sonda promptuje goło. Korelacja pełna: dwa koncepty z kolorem w captionie to dokładnie
dwa koncepty z artefaktem (kaczka zielenieje, plecak ma najgorsze DINO: 0.43–0.52).
Trening bez atrybutów: **job 21592961** (`P_ground_gsa_nocap`, seed 2024 — sparowany
z `P_ground_gsa`), log dowodzi stripu (`captiony cifc_duck_toy:  duck toy on a blue carpet`).
Od commita `e442b47` każdy trening drukuje captiony per task — cichy błąd w podmianie nie ma
żadnego innego objawu niż atrybut, którego adapter się nie uczy.

**(b) Metryka ćwiartek mierzy saliency, nie zawieranie — placement jest słabszy, niż mówiła.**
`scripts/_ground_iou.py` (Mask R-CNN R50-FPN-v2 COCO, wybór detekcji przez podobieństwo DINO
do referencji, nie przez score — dla kaczki COCO strzela `dining table`/`vase`, czyli mebel pod
obiektem). Obecny checkpoint, ćwiartki, 84 generacje, **job 21594432**:

| @κ=2, s=0.3 | ćwiartki | IoU | IoU>0.5 | zawarcie | wypełnienie | DINO |
|---|---|---|---|---|---|---|
| dog | 42% | 0.273 | 0% | 0.31 | 2.38 | 0.8414 |
| duck_toy | 83% | 0.482 | 33% | 0.58 | 1.89 | 0.6890 |
| cat | 83% | 0.427 | 33% | 0.48 | 2.13 | 0.8338 |
| backpack | 75% | 0.349 | 25% | 0.46 | 1.63 | 0.4622 |
| teddybear | 42% | 0.289 | 8% | 0.31 | 2.52 | 0.7206 |
| dog2 | 58% | 0.403 | 25% | 0.43 | 2.28 | 0.7995 |
| cat2 | 58% | 0.400 | 17% | 0.41 | 2.45 | 0.7422 |
| **RAZEM** | **63%** | **0.375** | **20%** | **0.43** | **2.18** | 0.7270 |

Obiekt jest średnio **2.2× większy od żądanej ramki**, a mniej niż połowa jego pikseli wpada
do środka. Pre-rejestrowane „84.5% GO" to saliency-argmax; kryterium detektorowe z literatury
groundingu daje **20% IoU>0.5**. To nie unieważnia GO (|L−R| = 0.179 i sonda połówkowa stoją),
ale **zawęża twierdzenie**: mechanizm steruje *położeniem najwyrazistszej części* obiektu, nie
jego *rozciągłością*. Do artykułu obie liczby, z tym rozróżnieniem nazwanym wprost.

**Diagnoza konstrukcyjna:** wstrzyk GSA tylko **dodaje** treść w ramce i nic nie **tłumi**
konceptu poza nią. Brakująca połowa dołożona jako `ground_confine` (commit `5cecfe7`): kara
logitu dla tokenów konceptu na pozycjach poza ramką, ten sam analityczny adres `inside()`,
harmonogram wspólny z κ (kara żyje, dopóki `ground_gain > 0`). Pełny kadr ⇒ `inside`=1 ⇒
kara 0, więc protokół bazowy jest nietknięty. Sweep κ=4/s=0.15 × confine ∈ {0,3,6,10}:
**job 21597878**. Druga gałka, `ground_gain_res` (mnożnik κ per rozdzielczość mapy attn2):
układ rozstrzyga się na mapach 8/16, kolor i tekstura na 32/64, a wstrzyk był jednakowy na
wszystkich 16 warstwach — **job 21597884** (`--gain_res 64:0`).

**Uwaga do κ-aware treningu:** amplituda κ podczas treningu jest **redundantna z magnitudą
gate'a** (uczy się iloczyn `gain·tanh(gate)`), więc trening ze stałym κ to reparametryzacja
i nie da silniejszego mechanizmu. Sensowna wersja to trening z κ **losowanym** — to nie
kalibracja punktu pracy, a regularyzacja odporności na skalę: cel to placement z κ=4 bez
szkód z κ=4. Peak placement się od tego nie poprawi.

## 5. Pozostałe wyniki analityczne (do artykułu)

- **Lekcje maskowania nie przenoszą się między backbone'ami**: nomask +0.019 DINO na SD-1.5,
  −0.018 na SDXL (odwrócenie znaku).
- **Kompozycja CIDM niereprodukowalna z równań pracy**: rów. 4-5 wymaga bootstrapu wnętrza ramki
  (MultiDiffusion), nieopisanego; z bootstrapem 2/3 próbek na SD-1.5. Regionalna uwaga trasuje
  treść, nie wymusza liczby podmiotów.
- **Placement wymaga skali lub struktury**: GLIGEN/TC-LoRA uczą przestrzenności na 10⁵–10⁶
  obrazów; w reżimie personalizacji (4–7 zdjęć) działa dopiero attention-level read (nasze GO).
- Rozrzut międzyseedowy 0.0091 → jednoseedowe różnice <0.01 niepotwierdzalne (bramka i K_sem64
  zmieniły znak na drugim seedzie).
- Adaptery nieskompresowalne (4 osie, bez kolana); encoder CLIP przyczynowy; klucz semantyczny
  zbędny (treść klucza nieistotna — wystarczy ortonormalność).

## 6. W toku / otwarte

- **Helios, 2026-08-31:** trening `P_ground_gsa_nocap` (21592961) + sondy IoU/confine
  (21597878) i ablacja rozdzielczości (21597884). Czekają decyzje: drugi seed GSA,
  kompozycja wielokonceptowa, κ per koncept, retrening κ-losowany.
- L2DM na SDXL: OOM (A100 40GB) — opcje: gradient checkpointing / batch 1+akum. / liga 4-metodowa.
- Parked: test szybkości adaptacji (warm-start control, ~5 GPU-h); R_tail "obiecujące,
  niepotwierdzone" (+0.011, 2 seedy, brak trzeciego); sweep h @T=35 (warunek tezy O(1));
  kompozycja wielokonceptowa na GSA (cel ramek); tabele @50 refresh (`scripts/make_tables.py`,
  fix: ours-Fgt z pełnej macierzy).

## 7. Infrastruktura (twarde lekcje)

- GPU przez `sbatch`, konto z `scripts/pick_account.sh`. **Zmiana 2026-08-31:** `plgideascvgroup1`
  odblokowany, a skrypt rozpoznaje klaster po hostname (`-gpu-a100` na Athenie, `-gpu-gh200` na
  Heliosie) i preferuje grant wygasający najszybciej. Dotąd był martwym kodem — żaden z 11 runnerów
  go nie wywoływał, wszystkie miały zaszyte `--account=plgideascv1cl-gpu-a100`, **a ten grant
  zakończył się 2026-08-26**, więc te zadania nie wystartują.
- Węzły MAJĄ internet (wandb online); `HF_HUB_OFFLINE=1` w runnerach celowo → nowe checkpointy
  wymagają prefetchu na login-nodzie.
- `TMPDIR=/tmp` we wszystkich runnerach — quota inode'ów $SCRATCH bywa pełna (inne projekty).
- Skrypty sbatch TYLKO w `scripts/` (scratchpad sesji jest czyszczony/niewidoczny z węzłów).
- venv wspólny z UnHype (python 3.11; instalacje przez `uv pip`, venv nie ma pipa; uwaga na
  cudze pakiety w ~/.local dla pythona 3.9 — mylą `pip list`). **To jest bomba zegarowa:**
  `scripts/sbatch_cl.sh` ma `VENV=../unlearning/UnHype/.venv`, a `unlearning` leży w katalogu
  grantu `plggrecontext`, który **wygasa 2026-09-08**. Na Athenie nie ma zamiennika — trzeba zbudować
  własny venv w `$SCRATCH/venvs/`, jak zrobione na Heliosie (patrz niżej).
- Smoke przed każdym pełnym treningiem + strażnik (weryfikacja gradientów w checkpoincie,
  auto-scancel łańcucha); łańcuchy jobów na `--dependency=afterok`.
- Nie commitować/pushować bez zgody. Przy zmianie headline'u lub dużych wydatkach GPU — pytać.
- Ocena wizualna: nigdy z jednej próbki; tożsamość tylko wobec zdjęć referencyjnych; przed
  diagnozą modelu sprawdzić pipeline renderowania (SDXL@512 = kafelki).

### Port na Helios (GH200) — 2026-08-31

Działa, z jednym zastrzeżeniem o numeryce. Stan:

| | |
|---|---|
| klaster | `ssh helios`, partycja `plgrid-gpu-gh200`, limit 2 dni, 110 węzłów × 4 GPU |
| konto | `plgideascvgroup1-gpu-gh200`, ważne do 2027-03-23, ~35 450 z 50 000 h |
| GPU | NVIDIA GH200 120GB, capability 9.0 |
| venv | `$SCRATCH/venvs/continualhyper-helios` — **samodzielny**, nie dzielony z UnHype |
| runnery | `scripts/sbatch_helios_venv.sh` (budowa), `scripts/sbatch_helios_smoke.sh` |

Helios ma **login node x86_64 (AMD EPYC 9654), a węzły GPU aarch64 (Grace)** — dlatego venv musi
powstawać w jobie, nie na login-nodzie, i dlatego Anaconda tam nie działa. `sbatch` wymaga
`#!/bin/bash -l`, bez tego `module load` nie inicjalizuje się. Moduł: `ML-bundle/24.06a`
(**nie** domyślny `25.04` — ten ma 4 koła i nie ma torchvision).

**Numeryka — ZWERYFIKOWANA 2026-08-31.** Ten sam checkpoint `P_ground_gsa/hyper.pt`, te same
ziarna (31337+i), 30 kroków, `lora_scale` 0.7: ćwiartki per koncept @κ=2/s=0.3 zgadzają się
z Ateną **co do próbki** (42/83/83/75/42/58/58, job 21594432 vs percept 3043021), a DINO do
≤0.0015 na koncept. Sonda połówkowa: κ=1 **86.9%** (73/84, job 21590216) vs 84.5% na Atenie,
κ=2 **91.7%** (77/84, job 21590217) vs 94.0% — rozjazd 2–3 próbki z 84 siedzi prawie cały na
kaczce, czyli na koncepcie o najniższym i najbardziej chybotliwym DINO wobec referencji.
Wniosek: torch 2.6.0+cu124/aarch64 na GH200 nadaje się do tych pomiarów, werdykty się nie
zmieniają. GH200 jest ~3× szybszy od A100 na tym kodzie (84 generacje @30 kroków: 5 min).

**Szczegóły wersji (kontekst powyższej weryfikacji).**
`requirements.txt` pinuje `torch==2.7.1` / `torchvision==0.22.1`; kół aarch64 dla tych wersji nie ma
(max `2.7.0rc9` / `0.21.0`). Spójna dostępna para to **torch 2.6.0+cu124.post3 + torchvision
0.21.0+cu124torch260**, czyli zejście o wersję minor. Reszta stosu trafia w piny co do numeru:
diffusers 0.30.0, transformers 4.44.2, timm 1.0.24, numpy 1.26.4. Zanim uznamy liczby z Heliosa
za porównywalne z Atheną, trzeba przepuścić config o znanym wyniku i porównać — zejście wersji jest
mniej bezpieczne niż podniesienie, mimo że `requirements.txt` dopuszcza „newer minor versions".

Smoke (job 21530405): `src.*` importuje się, SD-1.5 wczytany, 25 kroków 512² w **1,7 s**.

**Dane i checkpointy są na Heliosie od 2026-08-31**: `$SCRATCH/continualhyper/data` (6.9 GB —
`CIFC/`, `seg/` 7 konceptów, `backgrounds/` 100 teł) i `outputs/` (24 GB, w tym
`phaseP/P_ground_gsa/hyper.pt`). Klon `$HOME/projekty/continualhyper` ma symlinki na scratch.
Zadania idą przez `bash scripts/run.sh scripts/sbatch_py.sh <skrypt.py> ...` i
`... scripts/sbatch_cl.sh <config>` — oba sourcują `slurm/env.sh`, więc konto i partycja
Heliosa wchodzą z CLI. Wagi torch.hub (detektory) idą do `$SCRATCH/.cache/torch` przez
`TORCH_HOME` z `env.sh` — `$HOME` ma 100 000 inodów na wszystko.

**Pułapka w proweniencji:** `run.sh` oznacza każdy run jako `(DIRTY)`, bo `git status` widzi
nieśledzone symlinki `wandb` i `outputs`, które sam tworzy — a wykluczone są tylko
`results/logs/data`. Flaga DIRTY straciła więc znaczenie sygnalizacyjne; do naprawy jednym
wykluczeniem więcej.

## 8. Konwencja aktualizacji

Po każdym domkniętym wątku: zaktualizować sekcję wyników lub sagę ramek (tabela wariantów),
przenieść pozycje z "W toku" do właściwych sekcji, datę w nagłówku. Szczegóły i pełne liczby
zawsze w `assets/STATUS.md`; tu tylko synteza, którą da się przeczytać w 5 minut.
