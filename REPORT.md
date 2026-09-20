# ContinualHyper — raport stanu projektu

> Żywy dokument-pamięć projektu: syntetyczny obraz "gdzie jesteśmy i skąd to wiemy".
> Aktualizowany po każdym domknięciu wątku (werdykt, faza, decyzja ramowa).
> Szczegółowy dziennik pomiarów i odrzuceń: `assets/STATUS.md`. Stan na: **2026-09-14 (noc)**.

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
  **parytet** z opublikowanym CIDM-SDXL na ich protokole. UWAGA: ich Table 2 podaje
  **TA 80.0 / IA 79.5** (wcześniej mieliśmy tu te liczby zamienione). Wobec tego jesteśmy
  −0.79 TA i +0.38 IA, czyli remis bez dominacji. **To checkpoint BEZ groundingu** —
  wersja z groundingiem jest niżej, patrz §4. (Kod CIDM-SDXL nie
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

### Polerowanie GSA na Heliosie (2026-08-31) — dwa przecieki, uczciwa metryka, nowy punkt pracy

**(a) Przeciek atrybutu w captionach treningowych — POTWIERDZONY i NAPRAWIONY.**
Captiony CIFC to dosłownie `yellow rubber duck toy sitting on a gravel surface` (4/4)
i `red backpack sitting on a rock in the woods` (6/6); `cat` ma `fluffy` w 1 z 5.
`P_ground_gsa.yaml` **nie ustawiał `attr_strip`**, więc `src/data.py` nic nie zjadał i kolor
wchodził promptem. Gorzej: przy `token_mask_lora: true` delta LoRA aplikuje się tylko na
pozycjach tokenów `class_word`, więc tokeny `yellow`/`rubber` idą przez **zamrożone** K/V —
hipersieć koloru nie musiała się nauczyć **i nie mogła go dotknąć**. `gen_cifc.py` dokłada przy
ewaluacji `eval_prefix`, więc artefakt był niewidoczny, dopóki sonda nie promptowała goło.
Trening bez atrybutów: `P_ground_gsa_nocap`, job 21592961, seed 2024 (sparowany z bazą).
Wynik @κ=2/s=0.3, te same ziarna, ten sam instrument:

| koncept | DINO base → nocap | dRGB base → nocap | gen RGB base → nocap | ref RGB |
|---|---|---|---|---|
| duck_toy | 0.6890 → **0.7377** | 0.272 → **0.181** | (0.55,0.45,0.28) → (0.62,0.50,0.21) | (0.75,0.60,0.20) |
| backpack | 0.4622 → **0.5582** | 0.228 → **0.139** | (0.29,0.30,0.32) → (0.34,0.17,0.25) | (0.40,0.16,0.24) |

Plecak przeszedł z achromatycznej szarości (R≈G≈B) na czerwień z kanałami G i B trafiającymi
w referencję. Wzrokowo (`assets/figures/ground_compare_k4.jpg`): kaczka **zielona** w base,
**żółta** w nocap, na obu ziarnach; plecak czarny → karmazynowy. DINO +0.049 i +0.096 to
5–10× szum międzyseedowy. Koszt: pięć pozostałych konceptów nie zyskuje, a trzy późne taski
tracą (teddy −0.028, cat2 −0.035, dog −0.010) — podpis interferencji trajektorii CL, bo zmiana
captionów tasków 1 i 3 przestawia kotwice von Oswalda i wszystko po nich. Przy jednym seedzie
nieodróżnialne od szumu trajektorii; drugi seed to rozstrzygnie.
Od commita `e442b47` każdy trening drukuje captiony per task.

**(b) Metryka ćwiartek zawyżała placement.** `scripts/_ground_iou.py` (Mask R-CNN R50-FPN-v2,
wybór detekcji przez podobieństwo DINO do referencji, nie przez score — dla kaczki COCO strzela
`dining table`/`vase`). Ćwiartki mierzą argmax podobieństwa, czyli GDZIE koncept jest
najwyrazistszy, a nie czy MIESCI SIE w ramce: @κ=2 kaczka ma 83% ćwiartek i 33% IoU>0.5.
Do artykułu obie liczby z nazwanym rozróżnieniem.

**(c) Brakująca połowa mechanizmu: tłumienie poza ramką — i dlaczego musi objąć ogon.**
Wstrzyk GSA tylko DODAJE treść w ramce, nic nie TŁUMI konceptu poza nią. `ground_confine`
(kara logitu dla tokenów konceptu na pozycjach poza ramką, ten sam analityczny adres
`inside()`, harmonogram wspólny z κ) na samym spanie konceptu jest **no-op**: kara 0/3/6/10 →
IoU>0.5 62/64/64/65%, wypełnienie bez zmian. Wyjaśnienie było już zmierzone w tym repo
(docstring `RegionKVAttnProcessor`, audit 2885915): **CLIP jest przyczynowy**, więc koncept
wycieka do każdego kolejnego tokenu, a kara na pozycjach `dog` zostawia EOS i padding niosące
całe zdanie. `ground_confine_tail` (`cummax` po osi tokenów — kara od pierwszego tokenu
konceptu do końca sekwencji) przy karze 3: **IoU>0.5 62% → 79%**, zawarcie 0.67 → 0.78,
wypełnienie 1.60 → 1.43, koszt DINO 0.0087. Nasyca się natychmiast (3/6/10 → 79/77/78%), więc
punkt pracy to kara 3. **Maska kary to nie maska LoRA**: `token_mask_lora` zostaje na spanie,
bo celowo nie rusza reszty promptu (prompt-following). Dwie różne maski, dotąd przypadkiem
identyczne.

**(d) Punkt pracy: `nocap + κ=4/s=0.15 + tail-confine 3`.** Wszystkie osie wobec stanu
z 2026-08-20 (n=3, ziarna 31337):

| konfiguracja | IoU>0.5 | zawarcie | wypełn. | DINO | dRGB | tło grad | tło std |
|---|---|---|---|---|---|---|---|
| base, bez kary (stan przed) | 62% | 0.67 | 1.60 | 0.7001 | 0.180 | 0.1104 | 0.1385 |
| base + tail3 | **79%** | 0.78 | 1.43 | 0.6914 | 0.183 | 0.0969 | 0.1250 |
| nocap, bez kary | 38% | 0.57 | 1.76 | **0.7167** | 0.146 | 0.1099 | **0.1502** |
| **nocap + tail3** | 75% | 0.77 | **1.35** | 0.7076 | **0.161** | 0.0993 | 0.1307 |
| nocap + κ=2/s=0.3 + tail3 | 56% | 0.70 | 1.57 | 0.7268 | 0.128 | — | — |

**ROZSTRZYGNIĘCIE PARAMI (κ=4/s=0.15 + tail3, n=6, DWIE partie ziaren, 336 generacji na
konfigurację).** Kluczowa obserwacja metodologiczna: **efekt partii ziaren jest WSPÓLNY dla
obu wariantów** (oba spadają z ~76% na ~67% między partiami), więc czytać należy różnice
sparowane, nie wartości bezwzględne.

| metryka | base: 31337 / 41337 | nocap: 31337 / 41337 | Δ(base−nocap) |
|---|---|---|---|
| IoU>0.5 | 76% / 68% | 75% / 66% | +1, +2 pp → **remis** |
| zawarcie | 0.78 / 0.73 | 0.76 / 0.73 | ≈0 |
| wypełnienie | 1.44 / 1.60 | **1.35 / 1.35** | −0.09, −0.25 |
| DINO | 0.6970 / 0.6960 | **0.7125 / 0.7109** | −0.0155, −0.0149 |
| kolor dRGB | 0.192 / 0.197 | **0.159 / 0.162** | +0.033, +0.035 |
| tło std | 0.1249 / 0.1305 | **0.1323 / 0.1399** | −0.0074, −0.0094 |

Różnice sparowane powtarzają się co do trzeciego miejsca na obu partiach, choć wartości
bezwzględne wahają się o 8–9 pp. **`nocap + tail3` dominuje wariant bazowy na zawieraniu,
tożsamości, kolorze i tle, przy remisie na placemencie.** Wcześniejsze „base+tail ma lepszy
placement (79% vs 75%)" było artefaktem pojedynczego pomiaru n=3. Wypełnienie 1.35 powtórzyło
się w trzech niezależnych pomiarach nocap. Zgadza się to z siatką: w kolumnie base+tail są
duplikacje podmiotu (kaczka z dwoma dziobami, podwojony kot) i płaskie tła, których żadna
liczba nie karze.

**PUNKT PRACY: `P_ground_gsa_nocap` + κ=4, sched 0.15, tail-confine 3.**

**Kontrola bez ramki** (κ=1, pełny kadr = protokół bazowy, 42 generacje): base DINO 0.7744,
dRGB 0.171, tło grad 0.0702, tło std 0.1763 · nocap DINO 0.7898, dRGB **0.103**, tło grad
0.0724, tło std **0.1836**. Czyli **`attr_strip` poprawia protokół BAZOWY, nie tylko wersję
z ramką**: kolor −40%, DINO +0.015 bez żadnego groundingu. To jest poprawka metody, nie
groundingu, i to ona uzasadnia uczynienie nocap wersją główną.

**(e) Ostrzeżenia metodologiczne, które dotyczą wstecz.**
1. **Szum IoU>0.5 to do ~10 pp**, nie 1 pp: ten sam checkpoint na świeżych ziarnach dał base
   62% → 52%, nocap 37% → 38%. Przy 168 próbkach błąd dwumianowy to ~3.9 pp, a próbki w obrębie
   konceptu są skorelowane. Pojedyncze pomiary n=3 są orientacyjne; różnica base/nocap na
   placemencie po przejściu na n=6 spadła z 25 pp do **14 pp** (52% vs 38%), a różnica
   wypełnienia (1.60 vs 1.78) **zniknęła** (1.75 vs 1.76) — była szumem.
2. **`tło grad` jest metryką mylącą, używać `tło std`.** Rozstrzygnęła kontrola bez ramki:
   `tło grad` **bez** ramki wynosi 0.070, czyli MNIEJ niż każda konfiguracja z ramką
   (0.099–0.110) — płaska ściana z ostrą krawędzią bije rozmyte bokeh zbliżenie, więc gradient
   nie mierzy bogactwa tła. `tło std` układa się zgodnie z oceną wzrokową i z kierunkiem
   artefaktu: bez ramki 0.176–0.184 > nocap 0.150 > base 0.139 > nocap+tail 0.131 >
   base+tail 0.125. Wniosek liczbowy: ramka z wstrzykiem κ zabiera **15–25% zróżnicowania
   tła**, kara na ogonie kolejne ~10%, a nocap część odzyskuje. Artefakt płaskich teł jest
   zmniejszony, nie usunięty. Dodatkowo `--bg_ref` (podobieństwo DINO tła do generacji bez
   ramki na tym samym ziarnie) — w toku.
3. **Duplikacja podmiotu jest niewidoczna dla IoU.** W `base+tail` na drugim ziarnie kaczka ma
   dwa dzioby, a kot jest podwojony; dobra ramka detekcji tego nie karze. To dodatkowy powód,
   dla którego `base+tail` (79%) jest gorszy od `nocap+tail` (75%) wbrew liczbie.

**(f) Odrzucone dzisiaj, z pomiarem.**
- **`gain_res` (κ per rozdzielczość):** wyłączenie wstrzyku na mapach 64² zabiera 16 pp IoU>0.5
  (62% → 46%), a kolor poprawia o 0.007 (w szumie). Najdrobniejsze warstwy attn2 współpracują
  w układzie, a dryf koloru nie pochodzi ze wstrzyku.
- **Prior skali z referencji:** kolumna „obiekt w kadrze referencji" nie koreluje z wypełnieniem
  (kot: referencja 0.76 → wypełnienie 1.25; kaczka: 0.23 → 2.19). Wcześniejszy wniosek z porządku
  wypełnień był konfudowany podziałem zwierzę/przedmiot. Zdejmuje to `box_aug_p` z priorytetów.
- **„Nocap ma słabszy grounding":** normy z checkpointów mówią odwrotnie — `ground_gates`
  ×1.086, `Σ tanh(gate)·||o||` ×1.099, `ground_gsa_mods` ×0.999, LoRA ×1.000. Grounding jest
  ~10% silniejszy, a wagi LoRA mają identyczną normę (zmieniły się kierunki, nie amplituda).
- **κ-aware trening jako kalibracja punktu pracy:** uczy się iloczyn `gain·tanh(gate)`, więc
  stałe κ w treningu to reparametryzacja. Sens ma tylko κ **losowane**, jako regularyzacja
  odporności na skalę — i nie podniesie szczytowego placementu.

**(h) Pułapka: udany trening ze statusem TIMEOUT zabija łańcuch `afterok` (2026-09-01).** Job
21746202 zrobił CAŁĄ robotę — wypisał `[CL] DONE`, zapisał `hyper.pt` (100 MB) — a Slurm dał mu
`State=TIMEOUT` przy `Elapsed 01:00:07`, przy kroku `batch=COMPLETED`. Python wyszedł (runner
zdążył wypisać `DONE`), ale **serwis wandb zostaje w grupie procesów joba** i trzyma go do ściany
czasu; `wandb.finish()` jest wywoływane (`train_cl.py:562`) i nie wystarcza. Skutek: dwie sondy
dowiązane przez `--dependency=afterok` przesiedziały **12 godzin** w `DependencyNeverSatisfied`
i nigdy by nie wystartowały, choć checkpoint był poprawny od pierwszej minuty. Wnioski:
(1) łańcuchy po treningu wiązać przez **`afterany`**, nie `afterok`, i pozwolić sondzie paść na
braku pliku; (2) przy diagnozie zawieszonej kolejki czytać `squeue -o "%R"`, bo
`DependencyNeverSatisfied` w `sacct` wygląda identycznie jak zwykłe `PENDING`; (3) gdyby to miało
się nie powtarzać: `WANDB_MODE=offline` w treningach i `wandb sync` z login-noda — kosztem
konwencji „W&B online", więc decyzja użytkownika.

**(g) Pułapka infrastrukturalna, kosztowała trzy padłe zadania.** Job zapisał figurę do katalogu
**śledzonego gitem** na klastrze (`assets/figures/`), a ta sama ścieżka została zacommitowana
lokalnie — `git pull --ff-only` odmawia nadpisania pliku nieśledzonego, więc klon został 5
commitów z tyłu i **każdy `run.sh` startował po cichu na starym kodzie** (jedna linijka
ostrzeżenia ginęła w wyjściu; padły dopiero Python i brak configu). Naprawione trzema zmianami:
(1) `_ground_compare.py` pisze na scratch, nie do `assets/`; (2) `run.sh` **przerywa** przy
nieudanym pullu, świadome uruchomienie to `SKIP_PULL=1`; (3) flaga `(DIRTY)` wyklucza
`wandb`/`outputs`, bo to symlinki, które ten sam launcher tworzy — dotąd była włączona ZAWSZE
i przestała cokolwiek znaczyć. Efekt uboczny do zapamiętania: pull podmienia `run.sh` **w trakcie
jego własnego wykonania**, a bash czyta skrypt przyrostowo — pierwsza submisja po naprawie użyła
jeszcze starej logiki i wpisała mylące `(DIRTY)` do `run-info.txt` joba 21620332. Commit
w proweniencji pozostaje prawdziwy (brany po pullu), ale podmiana w locie może zrobić więcej
szkody niż mylna flaga.

### Zawieranie i tło — łańcuch mechanizmów (2026-08-31, wieczór)

Wymaganie użytkownika przestało być kompromisem: **obiekt prawie zawsze w ramce** ORAZ
**sensowne tło**, z celem kompozycji wielokonceptowej. Kolejność odkryć:

**(a) Rodzina kar w uwadze ma pułap.** `tail-confine` 0/3/6/10 → IoU>0.5 62/64/64/65%,
wypełnienie bez zmian. Kara na samym spanie konceptu to no-op (CLIP przyczynowy).

**(b) Izolacja self-attention NISZCZY placement**: 75% → **4%** IoU>0.5, zawarcie 0.77 → 0.40.
Wniosek mechanistyczny: **zewnętrze ramki nie dziedziczy obiektu biernie — generuje go samo**
z promptu i globalnej delty LoRA. Odcięcie komunikacji pozwoliło obu regionom rysować
niezależnie. To zamyka rodzinę „ograniczaj kanał" i przekierowuje na „zajmij miejsce".
(Pułapka: `leak` w `RegionalSelfAttnProcessor` jest bezczynne przy `strength=None` — kara
`(1-allow)*(1-leak)*1e4` po softmaxie jest −inf dla każdego leak<1; leak=0.5 dał liczby
identyczne z leak=0.0.)

**(c) Bootstrap latentu działa, ale liczy się CO stoi poza ramką.** Zawarcie rośnie
monotonicznie z długością K. Trzy warianty zewnętrza, κ=2/s=0.3, prompt ze sceną, 84 gen.:

| poza ramką | zawarcie | wypełn. | TA | DINOwyc | tło std |
|---|---|---|---|---|---|
| nic (bez bootstrapu) | 0.61 | 1.48 | 0.8073 | 0.7244 | 0.1775 |
| guidance wyłączone (bezwarunkowo) | 0.55 | 1.60 | — | 0.6891 | 0.1617 |
| bank teł `data/backgrounds` K=10 | 0.88 | 0.91 | 0.7485 | 0.6973 | 0.1881 |
| bank teł K=20 | 0.98 | 0.75 | 0.7080 | 0.6962 | 0.1848 |
| **rusztowanie z promptu K=20** | **0.90** | 0.80 | **0.8278** | 0.6775 | 0.1246 |
| **rusztowanie z promptu K=15, s25** | **0.93** | 0.77 | **0.8175** | 0.6849 | 0.1302 |

- **Wyłączenie guidance poza ramką jest GORSZE od braku bootstrapu** (0.55 vs 0.77 zawarcia):
  odebranie promptu to nie to samo, co zajęcie miejsca. Ta sama lekcja co w (b).
- **Bank teł psuł TA o 0.06** — nie przez zawieranie, a przez niezgodność scen: prompt mówił
  „on a beach", rusztowanie dawało łąkę. Mój wcześniejszy rozkład („kara 0.061, bootstrap
  0.006") był błędny metodologicznie — sekwencyjny zamiast czynnikowego; bootstrap SAM
  kosztuje 0.059, mechanizmy się nakładają.
- **Rusztowanie generowane z promptu użytkownika** (ten sam prompt z wyciętym konceptem,
  10 kroków, bez LoRA i groundingu, `lora_scale=0`, `ground_gain_base=0`) daje TA **0.828**,
  czyli LEPIEJ niż bez bootstrapu (0.807), przy zawarciu 0.90–0.94. Zero danych na wejściu
  (warunek użytkownika). Liczba kroków rusztowania nie ma znaczenia (10/25/30 → tło std
  0.125/0.125/0.126).

**(d) Metryki tła: dwa proxy z rzędu wprowadziły w błąd.** `tło grad` bez ramki wynosi 0.070,
czyli MNIEJ niż każda konfiguracja z ramką — ostra krawędź pustej ściany bije rozmyte bokeh.
`tło std` stawia rusztowanie z promptu (0.125) niżej niż brak bootstrapu (0.178), a na
obrazach jest odwrotnie: plaża z falami, fakturą piasku i cieniem pod obiektem vs płaski
szary kąt. Wiarygodne przesłanki: **siatka wizualna, `tło sim` i TA**. TA jest tu najlepsza,
bo mierzy wprost „czy tło realizuje prompt".

**(e) Pełnokadrowe DINO nagradza wypełnianie kadru.** Bootstrap K=10: DINO pełne spada 0.101,
na wycinku 0.027, na masce 0.020 — **75–80% pozornego kosztu tożsamości to artefakt skali**.
To dotyczy też metryk CIDM (IA i TA są pełnokadrowe), więc porównując kompozycję ich osiami
karalibyśmy się za spełnienie warunku o ramce. Raportować obie wersje.

**(f) Bez ramek metoda jest NIETKNIĘTA — zmierzone.** Pełny kadr, wszystkie gałki włączone
(kara 3, bootstrap 20, rusztowanie 25) vs czysty przebieg: DINO 0.7916 vs 0.7898, kolor
0.097 vs 0.103, tło std 0.1826 vs 0.1836, wypełnienie 0.59 vs 0.60, IoU>0.5 71% vs 71% —
wszystko w szumie. `tail-confine` i `bootstrap` są zerowe z konstrukcji (kara `(1-inside)`,
blok pod `cond_box is not None`); miękka maska `geo_inside` (stromość 40) daje przy pełnym
kadrze `inside`≈0.33 w narożniku, ale efekt pierścienia brzegowego jest poniżej progu.

**(g) CIDM: brak jakiejkolwiek liczby dla kompozycji** (arXiv 2410.17594, wersja HTML +
ich `evaluate.py`). Metryki w pracy: **tylko IA i TA** (DINO jest w ich kodzie, nie w
tabelach). Kompozycja to figury 3, 12 i ablacja 6 — wyłącznie jakościowo, bez tabeli ani
badania użytkowników. **Ich protokół to ITP + RTP**: globalny prompt sceny plus prompt per
region z ramką od użytkownika. Czyli „konteksty per region" (`RegionKVAttnProcessor` w repo)
nie są naszym wynalazkiem, tylko protokołem porównania — i rozwiązują problem TA z
konstrukcji, bo sceneria poza ramką ma własny prompt. To jest właściwy tor dalej, a kara i
rusztowanie są obejściami problemu, którego w tym protokole nie ma.

**PUNKT PRACY (zawieranie + tło):** `P_ground_gsa_nocap` + κ=2, sched 0.3, **bez kary**,
bootstrap 10–15 kroków z rusztowaniem z promptu (10 kroków rusztowania wystarcza).
Zawarcie 0.90–0.94, ćwiartki 99–100%, TA 0.818–0.828, DINO na wycinku 0.68–0.70, kolor
0.125–0.135, bez ramek bezczynne. Wypełnienie 0.73–0.87 (obiekt respektuje ramkę, ale jej
nie wypełnia) — dlatego IoU>0.5 zostaje na 73–76% mimo zawarcia 0.9+; K=10 daje wypełnienie
0.87 i jest lepszym kompromisem niż K=20.

### Bootstrap: trzy maski, trzy skazy, jeden korzeń (2026-08-31, noc)

Konfiguracja `P_ground_gsa_nocap_all` + κ=2/s=0.3, bootstrap 10 z rusztowaniem z promptu,
prompt „on a beach", 84 generacje. Skazy widoczne TYLKO na siatkach — wskazane przez
użytkownika, nie przez metryki:

| maska | IoU>0.5 | zawarcie | wypełn. | TA | DINOwyc | kolor | det | skaza wizualna |
|---|---|---|---|---|---|---|---|---|
| twarda | 60% | 0.82 | 0.87 | 0.8115 | 0.7008 | 0.135 | 84/84 | **amputacje**: miś bez nóg |
| miękka (feather 2) | 34% | 0.75 | 0.69 | 0.7722 | 0.6508 | 0.171 | **74/84** | **szare, wyprane obrazy** |
| twarda + dylatacja 3 | 57–58% | 0.74–0.76 | **1.05–1.07** | 0.8124 | 0.704–0.714 | 0.122–0.127 | 84/84 | prostokątny szew, przycięcie kadrem |

**Miękka maska — ODRZUCONA z mechanizmem.** Mieszanka `latents*m + bg_t*(1-m)` składa DWA
niezależne losowania szumu, więc wariancja to `m²+(1-m)²` — przy m=0.5 połowa właściwej.
Denoiser dostaje wejście „za mało zaszumione" i zwraca obraz o zdławionym kontraście: tła
wychodzą szare, a detektor nie znajduje podmiotu w 10 z 84 obrazów. Nie da się tego naprawić
szerokością rozmycia; binarność maski jest warunkiem poprawnej statystyki szumu.

**Dylatacja naprawia amputację i kolor, ale nie skalę.** Kot w ramce TR ma nadal uciętą głowę
**krawędzią OBRAZKA**: ramka sięga górnej krawędzi kadru, więc gdy model wygeneruje obiekt za
duży, jego górna część musiałaby leżeć poza kadrem — tam nie ma „zewnętrza", którym maska
mogłaby zarządzić. To porażka SKALI, nie maski: żaden z badanych mechanizmów nie mówi modelowi
„zrób obiekt mniejszy".

**Wniosek strukturalny:** bootstrap daje zawieranie **przez rzeźbienie w kadrze**, a artefakty
rzeźbienia (amputacja, przycięcie, szew) są jego podpisem, nie kwestią strojenia — i metryki je
NAGRADZAJĄ, bo zawarcie rośnie, gdy wystające fragmenty znikają. Dalsze warianty maski to
polerowanie wady wpisanej w konstrukcję. Właściwy kierunek: **ITP/RTP** (własny kontekst
tekstowy per region, protokół CIDM, `RegionKVAttnProcessor` w repo) — zmienia to, o co model
jest proszony, zamiast wycinać to, co wygenerował — plus brakująca przesłanka o skali.

Figury: `assets/figures/grid_nocapall_k2_boot10_TLTR.jpg` (twarda, tylko TL/TR — usterka
szerokości arkusza, naprawiona), `grid_nocapall_soft.jpg` (miękka, szare tła),
`grid_nocapall_dil.jpg` (dylatacja, 4 kolumny).

### Obwódki obiektów: poświata z wycinków treningowych — NAPRAWIONE (2026-09-01)

Użytkownik zauważył na siatce, że obiekty „wyglądają jakby miały obrysy". Rozkład na trzy
przypadki (ten sam checkpoint, te same ziarna) wskazał winowajcę: rant jest **obecny z ramką**
(κ=1 i κ=2, z bootstrapem i bez) i **nieobecny bez ramki** — czyli wchodzi ze ścieżką groundingu,
nie z podmianą latentu i nie z amplitudy.

Przyczyna jest w danych. Wycinki z isnet mają **~10-pikselowy gradient alfy** (zmierzony profil
`[1,3,6,13,26,49,81,122,164,201,228,244]`), który wmieszuje kolor tła **ZE ZDJĘCIA ŹRÓDŁOWEGO**
w brzeg obiektu — miś ma jasną poświatę po sylwetce, bo jego oryginalne tło było jasne. Ta
poświata jest potem wklejana na inne tło, a `box_aug_p: 0.5` znaczy, że **połowa kroków
treningu** pokazywała koncept jako wycinek z rantem. GSA było trenowane właśnie na tych
wklejkach, więc im mocniej sterujemy tą ścieżką, tym wierniej model odtwarza wyuczony wygląd
wklejki — łącznie z obwódką.

Poprawka: `training.alpha_erode` (min-pooling alfy przed wklejką) cofa brzeg do wnętrza obiektu,
więc skrajne piksele mają kolor OBIEKTU. `alpha_erode: 0` zachowuje stare zachowanie, więc
poprzednie configi i wyniki pozostają odtwarzalne. Trening: `P_ground_gsa_erode` (job 21746202).

Wynik przy κ=2/s=0.3 bez bootstrapu, prompt ze sceną, 84 generacje:

| | ćwiartki | IoU>0.5 | zawarcie | wypełn. | TA | DINOwyc | kolor | tło std |
|---|---|---|---|---|---|---|---|---|
| `nocap_all` | 62% | 30% | 0.53 | 1.74 | 0.8057 | 0.7077 | 0.111 | 0.1638 |
| **`erode`** | 73% | 24% | 0.54 | 1.68 | 0.8095 | **0.7180** | 0.121 | 0.1694 |

Obwódki zniknęły wizualnie (`assets/figures/grid_erode.jpg` vs `grid_nocapall_dil.jpg`), a
metryki stoją albo lekko rosną: DINO na wycinku +0.010, TA i zawieranie bez zmian, detekcja
84/84 w obu. Kolor −0.010 i IoU>0.5 −6 pp mieszczą się w zmierzonym szumie samplingowym
(0.015 dla koloru, ~10 pp dla IoU>0.5). **Erozja jest darmowa, a `P_ground_gsa_erode` jest
obecnie najlepszym checkpointem: pełny strip atrybutów obiektu + brak poświaty wklejkowej.**

Do zapamiętania metodologicznie: to **trzeci** silent bug w obróbce danych znaleziony w tym
projekcie (po przecieku atrybutu i po niespójności zestawu referencyjnego) i **czwarty raz**,
gdy defekt wskazało oko na siatce, a nie liczba. Przy twierdzeniach o sterowalności tabela bez
siatki jest niewystarczająca.

### SUFIT BENCHMARKU: per-koncept DINO jest ograniczone spójnością referencji (2026-08-31)

`scripts/_ref_selfsim.py` (job 21684620). `self_par` = średnie podobieństwo DINO **par zdjęć
referencyjnych**, `min_par` = najgorsza para. Kolumna DINO to `P_ground_gsa_nocap_all`,
protokół bez ramki:

| koncept | n | self_par | min_par | DINO | DINO/self_par |
|---|---|---|---|---|---|
| dog2 | 5 | 0.913 | 0.873 | 0.863 | 0.95 |
| cat | 5 | 0.898 | 0.840 | 0.864 | 0.96 |
| dog | 5 | 0.849 | 0.759 | 0.851 | 1.00 |
| cat2 | 5 | 0.782 | 0.677 | 0.799 | 1.02 |
| teddybear | 7 | 0.774 | 0.673 | 0.774 | 1.00 |
| duck_toy | 4 | 0.721 | 0.637 | 0.785 | 1.09 |
| **backpack** | 6 | **0.604** | **0.428** | 0.617 | 1.02 |
| drawing | 6 | 0.555 | 0.362 | — | — |
| ink_painting | 5 | 0.432 | 0.317 | — | — |
| painting | 7 | 0.423 | 0.239 | — | — |

**Kolejność per-koncept DINO JEST kolejnością spójności danych**, a stosunek DINO/self_par
wynosi 0.95–1.09 dla każdego obiektu: nasze generacje są tak podobne do średniej referencji,
jak same referencje są podobne do siebie. **Na tym benchmarku nie ma już zapasu na tożsamość
mierzoną DINO.**

Dotyczy to wprost pytania „czemu plecak wypada gorzej" (0.62 przy 0.72–0.89 reszty): jego
zestaw jest najmniej spójny w benchmarku — dwie referencje są od siebie oddalone o **0.428**,
bo zdjęcie 00 to plecak **na plecach kobiety** (włosy, ramię, jeansy, chmury w kadrze), a 03
to ten sam plecak **sam na mchu w lesie**. DINO osadza cały obraz, więc średnia referencji
jest rozmyciem między „człowiek + niebo" i „plecak + las". Nasze 0.617 leży **powyżej**
wewnętrznej spójności zestawu. Dodatkowo tożsamość tego konceptu siedzi w drobiazgach
(metka Herschel, trzy przypinki), których rank-4 LoRA przy 512² nie odtworzy, a dwa z sześciu
captionów opisują człowieka (`woman with a red backpack`).

Sprostowanie: wcześniejsza uwaga, że nocap dał plecak „przesaturowany różowo", była błędna —
referencja jest karmazynowo-magentowa (ref RGB 0.40,0.16,0.24), więc magenta była BLIŻEJ
prawdy niż czarny plecak z bazy.

**Do artykułu:** per-koncept DINO raportować **razem z sufitem** self_par, inaczej 0.62
czyta się jako porażkę metody, a jest granicą zestawu. To jest też argument, że dalsze
polerowanie tożsamości na CIFC jest bezcelowe i różnicowanie musi iść przez forgetting,
skalowanie pamięci i sterowalność.

### Zapominanie z groundingiem: sprzężenie, nie usterka (2026-09-02/03)

Forgetting mierzony pełną macierzą 55 komórek, checkpoint z groundingiem:

| skala | ziarno 2024 | 2025 | 2026 |
|---|---|---|---|
| 0.5 | 0.0084 | −0.0017 | 0.0136 |
| 0.7 | 0.0100 | 0.0019 | — |
| 0.8 | 0.0151 | 0.0039 | — |

**Rośnie monotonicznie ze skalą w obu ziarnach.** W punkcie pracy s=0.5 na trzech ziarnach
wychodzi **0.0068 ± 0.0078**, czyli nieodróżnialne od zera; sprzed groundingu było 0.0015.
Twierdzenia „o rząd niżej od baseline'ów" nie da się obronić — CIDM ma 0.0174, LwF 0.0183.

**Gałąź groundingu niesie tożsamość, a nie tylko adresowanie.** Wyłączenie jej na inferencji
(`--ground_gain 0`) sprowadza forgetting z 0.0151 do 0.0014, ale kosztuje **8.2 IA i 11.1 DINO**
i ląduje 5.6 punktu PONIŻEJ naszej krzywej — czyli gorzej niż samo obniżenie skali.

**Rodzina kotwic — monotoniczna wymiana** (s=0.8, IA wobec krzywej przy zrównanym TA):

| wariant | forgetting | ΔIA wobec krzywej |
|---|---|---|
| bez kotwicy (2 ziarna) | 0.0095 | — |
| kotwica na tokeny | 0.0075 | −0.29 |
| kotwica + bramki | 0.0009 | −2.77 |
| gałąź wyłączona | 0.0014 | −5.6 |

Im mocniej przypinamy gałąź, tym mniej zapominania i tym mniej tożsamości. **Wkład gałęzi w
tożsamość i jej wkład w zapominanie to ta sama rzecz.** Decyzja: nie ograniczać gałęzi,
raportować forgetting w punkcie pracy i opisać sprzężenie jako własność metody.

Dryf mierzony deterministycznie z checkpointów (`scripts/_ground_drift.py`): `ground_head`
przemieszcza się o 3.34 własnej normy przez strumień, amplituda bramek z 0.011 na 0.0395.
Kotwica to dusi (3.34 → 0.15), ale **wzrost bramek to uczenie się, nie dryf** — zamrożenie ich
kaleczy gałąź. Uwaga metodologiczna: `regG≈0` przy WŁĄCZONEJ kotwicy nie dowodzi braku dryfu,
bo kotwica i brak dryfu wyglądają identycznie; stąd pomiar z checkpointów.

### Bank wycinków miał ucięte obiekty — NAPRAWIONE (2026-09-06)

Na siatkach SDXL dwie z czterech próbek psa miały **odcięty fragment obiektu zawieszony w
kadrze**. Trzy hipotezy odpadły po pomiarze (maski nie są prostokątne, alfa jest binarna bez
mgławicy, ramka nie wycieka do sweepu diagnostycznego). Przyczyna: **3 z 5 wycinków `cifc_dog`
jest przeciętych granicą cropu** (alfa na dolnej krawędzi 0.88 i 0.49, na prawej 0.40), przy
0–1 z 5 dla pozostałych konceptów. Wklejony w losowe miejsce daje obiekt ucięty płaską linią.

Dwie poprawki, obie domyślnie wyłączone:
- **`training.paste_flush`** — wycinek ucięty na krawędzi jest dosuwany tą krawędzią do brzegu
  kadru. Krawędzie liczone PRZED erozją alfy (erozja to min-pool z zerowym paddingiem, po niej
  każda krawędź wygląda na przezroczystą).
- **ramka per próbka** — `set_ground` przyjmuje listę ramek; tokeny `[B,M,D]`, FiLM `[B,128]`,
  `geo_inside` → `[B,n,1]`, maska przez `repeat_interleave(heads)` (head_to_batch_dim jest
  b-major). Ścieżka jednoramkowa bitowo identyczna. **Dotąd druga próbka w partii dostawała
  ramkę o rozmiarze pierwszej, a obiekt wklejony we własnej skali — na ~25% wszystkich kroków
  nadzór umiejscowienia był jawnie niezgodny z obrazem.**

Efekt, ten sam instrument i te same flagi, checkpointy finalne:

| | przed | po |
|---|---|---|
| IoU > 0.5 | 43% | **83%** |
| zawarcie | 0.64 | **0.83** |
| wypełnienie | 1.56 | **1.30** |
| IoU | 0.466 | **0.669** |
| TA (instrument) | 0.7203 | 0.7067 |
| DINO na wycinku | 0.7277 | 0.7190 |

Najmocniejsza pojedyncza poprawa kontroli przestrzennej w projekcie. Koszt: −0.014 TA i
−0.009 DINO na wycinku (na granicy rozdzielczości 0.0091). Jedno ziarno — do pracy potrzebne drugie.

### Protokół CIDM: co naprawdę robią (2026-09-03)

Z ich configów (`data/CIFC/options/cidm/task_*.yml`) i źródła Mix-of-Show:
- **800 kroków**, `batch_size_per_gpu: 1` na dwóch GPU = efektywnie 2, **tyle co u nas**.
  Nasze headline'owe SD-1.5 ma 400 — czyli **połowę ich budżetu**.
- **`manual_seed: 0` we wszystkich dziesięciu configach** — każda ich liczba, także dla
  baseline'ów, jest z jednego ziarna.
- Augmentacje: `HumanResizeCropFinalV3` (letterbox z maską straty), `ShuffleCaption`
  (u nas no-op: 54 z 55 podpisów bez przecinka), `EnhanceText` (27 szablonów obiektowych).
  **Nasze headline'owe configi nie mają żadnej augmentacji obrazu.**
- Pojemność per koncept większa niż sądziliśmy: **osobne embeddingi tekstowe per warstwa**
  transformera i LoRA na CAŁEJ uwadze (`where: Attention`), gdy my owijamy tylko `attn2`.

### SDXL: budżet, nie podział danych (2026-09-03/04)

Grounding kosztował na SDXL 3.1 TA i stawiał nas pod CIDM. Dwa warianty rozdzieliły przyczyny:
- `box_aug_p 0.25` (więcej kroków rekonstrukcji przy stałym budżecie): **null**.
- **800 kroków: +2.1 IA przy zrównanym TA**; z augmentacją obrazu **+3.35**.

Na SD-1.5 to samo podwojenie daje +0.07 i +0.75 IA, czyli praktycznie nic — spójne z
sufitem danych. SDXL ma 87.5 M parametrów hipersieci wobec 21.1 M i po prostu nie domykał się
w 400 krokach.

**Stan wobec CIDM (2026-09-06):**

| | ich TA/IA | nasze przy ich TA |
|---|---|---|
| SD-1.5 | 74.8 / 78.0 | **80.19** (+2.19), a przy s=0.5 bijemy na obu osiach naraz |
| SDXL, 2 ziarna `800aug` | 80.0 / 79.5 | 78.94 (**−0.56**) |

Rozrzut ziaren na SDXL to 1.20 IA, więc dwa ziarna to minimum. W toku: 1600 kroków
(`box_aug_p 0.5` znaczy, że nasze 800 to 400 kroków na koncepcie wobec ich 800 — 1600
zrównuje tę wielkość) plus obie poprawki wklejania, sześć skal.

### Negatywy z tej fali (wszystkie tanie, wszystkie zamknięte)

- **`boxonly`** (gałąź milczy bez ramki): tor LoRA nie przejmuje tożsamości. DINO 53.57 przy
  400 krokach i 54.95 przy 800, wobec 62.30 dla F_base — a `g0b` na normalnym checkpointcie
  daje 53.88. Przyczyną nie jest budżet: F_base ma te same 400 kroków na zdjęciach.
- **`paste_scale_full_p 0.3`** (drugi tryb skali przy 0.9–1.0): **−2.4 IA** przy zrównanym TA.
- **`proto`** (27 szablonów + podpis kompozytu z prawdziwego tła): TA −4.2 przy tej samej
  skali, 0.47 IA poniżej krzywej. **SPROSTOWANIE 2026-09-06:** przyczyną był BŁĄD w
  `prompt_aug` (stosowany po `encode_text`, patrz przegląd kodu niżej), nie `paste_caption`.
  Oba zostały potem zmierzone osobno na naprawionym kodzie — patrz „Układ 2×2".
- **Kotwice na gałąź**: patrz tabela wyżej — działają, ale płacą tożsamością.

### Przegląd kodu — trzy błędy, jeden ciężki (2026-09-06)

Przegląd na prośbę, bez awarii jako powodu. Znalezione i naprawione:
1. **`prompt_aug` po `encode_text`** (`2e50858`) — cichy i ciężki. UNet dostawał podpis BEZ
   szablonu (flaga martwa), a `token_span_mask` liczył się na łańcuchu z szablonem, przesuniętym
   o ~4 tokeny — **LoRA trafiała w złe pozycje tokenów przez cały trening**. Dotyczył tylko
   przebiegów z tą flagą (`proto`, dwa anulowane). Żaden raportowany wynik nie był dotknięty.
2. **`ddim_sample` zostawiał zmutowane `manager.ground_gain`** (`4baf4ab`) — trening czyta ten
   atrybut i mnoży przez niego wstrzyknięcie GSA. Bezobjawowe tylko przez domyślne
   `ground_sched_frac=1.0`. Teraz przywracane po samplingu.
3. **Kara konfinująca niegotowa na ramkę per próbka** (`2da92f9`) — `geo_inside` zwraca `[B,n,1]`
   przy liście ramek; ścieżka `ground_confine` zakładała `[n,1]`. Głośny błąd, nieaktywny w
   naszych configach.

Zgłoszone, nienaprawione (martwe ścieżki): `geo_logit`, `_ground_boxvec`, `box_emb` nie są
listowo-świadome (`ground_geo*`, `box_cond` wyłączone); `if False` w `set_ground` sprzed zmian.
Sprawdzone i czyste: blok wklejania, kolejność ramka→`set_ground`→forward, wzór na forgetting,
`--final_only` → forgetting 0 z konstrukcji, indeksowanie zadań w macierzy, seedowanie.

**Lekcja infrastrukturalna:** `run.sh` robi `git pull` przy WYSŁANIU, a job czyta drzewo klonu
przy STARCIE. Push między wysłaniem a startem zmienia kod czekającego zadania, a `run-info.txt`
zapisuje stary commit. Raz anulowaliśmy i wysłaliśmy ponownie z tego powodu (21925099 → 21926692).

### Transfer w przód umiejscowienia — NOWY WYNIK POZYTYWNY (2026-09-07)

Instrument umiejscowienia na checkpointach `P_ground_gsa_erode` po zadaniu 0, 4 i 9, koncept 0
(pies), te same flagi (κ=2, sched 0.3, bootstrap 10, rusztowanie 10, s=0.7, ćwiartki):

| po zadaniu | ćwiartka | IoU | IoU>0.5 | zawarcie | DINO |
|---|---|---|---|---|---|
| 0 | 75% | 0.33 | 8% | 0.44 | 0.753 |
| 4 | 83% | 0.57 | 83% | 0.70 | 0.815 |
| 9 | 92% | 0.57 | 75% | 0.72 | 0.816 |

**Pierwszy koncept nigdy więcej nie widział własnych danych, a jego umiejscowienie poprawia się
dzięki temu, co gałąź nauczyła się na dziewięciu innych.** Lustrzane odbicie zapominania; nikt na
tym benchmarku tego nie raportuje. Konsekwencja: **kotwica na gałąź jest błędem kierunkowym**
(zamraża stan, w którym pies miał 8%). Przebieg SDXL z kotwicą (22087630) anulowany przed startem.

**Zaprojektowane, niezaimplementowane rozwiązanie napięcia kotwica/transfer:** `gate =
g_warstwa + h(ramka)` z małym MLP na Fourierze ramki. Wtedy da się kotwiczyć tokeny **i**
`g + h(pełna klatka)` starych konceptów (wkład bezramkowy), zostawiając `h(inne ramki)` wolne.
Dziś bramka to jeden skalar na warstwę, więc amplituda pełnoklatkowa i ramkowa są nierozdzielne
— stąd kotwica na tokeny (dryf uciekł w bramki: Δ 0.0285→0.0589) i na bramki (−2.77 IA,
zamrożone umiejscowienie). Zysk ograniczony: forgetting w punkcie pracy już ≈0.

### `paste_flush` i ramka per próbka: SD-1.5 za darmo, SDXL płaci (2026-09-06/07)

SD-1.5, eval CIFC `P_ground_flush` wobec `erode` (2 ziarna): **+0.09 / +0.25 IA** przy zrównanym
TA — skok umiejscowienia 43%→83% nie kosztował nic.

SDXL, `X_sdxl_best` (= `800aug` + flush + per próbka) wobec `800aug`: **−1.70 do −2.03 IA** przy
zrównanym TA; przy ich TA 80.0 → 77.69 (−1.81) wobec 78.94 (−0.56) dla `800aug`. Przy tej samej
skali IA stoi (−0.23), a **TA spada o 3.1** (7 z 10 konceptów, ~3σ). Atrybucja per koncept
niemożliwa: różnice `best`/`800aug` per koncept (śr. |ΔIA| 2.24, |ΔTA| 3.52) są tej wielkości
co szum między ziarnami `800aug` (2.14, 3.11); `painting` ma −8.3 TA w `best` i +8.4 w drugim
ziarnie. **Liczby per koncept na SDXL wymagają ≥3 ziaren.**

Mechanizm (pomiar parametrów, `scripts/_ground_drift.py`, jobs 22087531/32):

| | `800aug` | `best` |
|---|---|---|
| dryf `ground_head` | 8.15× | **17.74×** |
| bramki absmax koniec | 0.0672 | **0.0856** |

Ramka per próbka wytrenowała gałąź mocniej, a **wstrzyknięcie GSA nie jest skalowane przez
`s_lora`** (`regional.py`: `κ·tanh(gate)·inside·read`) — więc ciągnie TA w dół na całej krzywej
i skalą LoRA nie da się tego oddać. Test κ=0.5/0.7 na inferencji: **22087569/70** (w kolejce).
Instrument umiejscowienia na SDXL: **22094511** (pierwszy raz; poprzedni 22087510 padł, bo sondy
ładowały SD-1.5 na sztywno — naprawione `7dcdf96`). Bez niego nie wiadomo, czy za −2 IA cokolwiek
kupujemy; jeśli nie — na SDXL `flush` wyłączyć i wrócić do −0.56.

**Zmiana architektoniczna, która z tego wynika (niezaimplementowana):** mnożyć wstrzyknięcie gałęzi
także przez `lora_scale`, żeby oba tory adaptera słabły razem — jedno pokrętło zamiast dwu.

### Układ 2×2 na naprawionym kodzie (2026-09-07)

Kontrola `P_best` = `erode` + augment + flush + per próbka (400 kroków). Krzywa: s=0.4
(75.80/79.30), **0.45 (75.13/79.95)**, 0.5 (74.61/80.39), 0.6 (73.57/81.23), 0.7 (72.66/81.81),
0.8 (71.78/81.85). **Sufit IA na SD-1.5 ≈ 81.85** — dalsze wzmacnianie nic nie daje.
**Przy s=0.45 bijemy CIDM na obu osiach**, przy ich TA 74.8 → IA **80.23 (+2.23)**.

| flaga (SD-1.5, wobec `P_best`, zrównane TA) | ΔIA | werdykt |
|---|---|---|
| `prompt_aug` (`P_best_pa`) | +0.07 | neutralny |
| `paste_caption` (`P_best_pc`) | +0.11 | neutralny; zostawić dla poprawności podpisów |
| `prompt_aug` na SDXL (`X_sdxl_best_pa`) | **−1.40** (dwa punkty) | **odrzucony** |

`800aug` na SD-1.5 (augmentacja obrazu na 800 krokach): kasuje ślizg z 800, netto ≈0.
`X_sdxl_1600` (1600 + aug + flush + per próbka): przy ich TA → 78.01 (−1.49); +0.32 nad 800 —
**nasyca się, nie ratuje**.

### Współdzielone głowice pod groundingiem — ODRZUCONE (2026-09-07)

Faza C (`configs/phaseC/`, wyniki na Athenie `outputs/phaseC/`, 52 evale, BEZ groundingu)
porównywała warianty WEWNĄTRZ rodziny `share_heads` — `C_cap_base` sam ma `share_heads: true`.
Headline z groundingiem tej flagi nie ustawia → 64 niezależne głowice, 21.1 M. Z fazy C przy
zrównanym TA wobec `C_cap_base`: `C_cap_role` **+0.57/+1.20**, `C_share_nomask_nonorm`
+0.47/+0.66/+1.31, `C_share` (z `preserve_norm: true`) −0.3/−0.4 i załamanie przy s=1.0 →
**`preserve_norm` szkodzi współdzieleniu**. Szerokość: 50 −0.31, 100 → 0, 200 +0.65, 400 +0.64
(nasycenie ~200). `P_best_share` (= `P_best` + `head_hidden 100, share_heads, share_by_role`; trening 22087727,
evale 22087728/30/33, instrument 22087736) **wobec `P_best`, jedno ziarno:**

| | share | P_best | Δ |
|---|---|---|---|
| IA @TA≈75.1 (s=0.45) | 78.32 | 79.95 | **−1.6** |
| IA @TA≈74.6 (s=0.5) | 78.68 | 80.39 | **−1.7** |
| IA @TA≈73.6 (s=0.6) | 79.34 | 81.23 | **−1.9** |
| instrument: IoU>0.5 / IoU | 67% / 0.577 | 83% / 0.669 | −16 pp / −0.09 |
| zawarcie / wypełnienie | 0.76 / 1.44 | 0.83 / 1.30 | gorzej |
| DINO wycinek | 0.675 | 0.719 | −0.044 |
| TA (instrument) | 0.711 | 0.707 | ≈ |

Spójnie na trzech skalach i na umiejscowieniu (placement w granicach szumu 10 pp, ale zgodny
kierunkiem). **Lekcja z fazy C (+0.57/+1.20) była bez groundingu i nie przenosi się**: gałąź GSA
sama zjada pojemność sterowania, więc per-warstwowe głowice są potrzebne — trzeci przypadek
lekcji nieprzenośnej między rodzinami (po `nomask` SD-1.5↔SDXL i `preserve_norm`). `share_heads`
zostaje wyłącznie jako opcja pamięciowa przy dużym T (sweep T=35, CustomConcept101 jest na
klastrze: `data/benchmark_dataset`, 101 konceptów), NIE w bieżącej wersji.

### SDXL: diagnoza w całości (2026-09-07)

Dwa problemy, nie jeden:
- **A. Baza bez zapasu.** SD-1.5 baza jest +2.5 IA nad CIDM przy ich TA, SDXL baza w remisie
  (−0.79 TA / +0.38 IA, ale mierzona przy 400 krokach). CIDM zyskał na SDXL 5.2 TA, my 3.7 —
  hipoteza: adaptacja po stronie tekstu (ich embeddingi per warstwa; nasz enkoder zamrożony).
  **Baza SDXL@800 bez groundingu nigdy nie mierzona** → `X_sdxl_base800` **22095781**, evale
  22095788/99/806 (s=0.4/0.5/0.7). To rozstrzyga, czy sufit tekstowy istnieje.
- **B. Podatek groundingu 10× większy niż na SD-1.5** (2–3 IA wobec 0.3): gałąź mocniejsza i
  nieskalowana przez `s_lora` (wyżej); **wycinki powiększane przy 1024²** (500–850 px wobec celu
  460–870 → do 1.7×, rozmyty obiekt na połowie kroków; przy 512² zawsze zmniejszane) →
  `paste_no_upscale` (`5ceb9e7`, `r = min(r, 1.0)`, no-op na SD-1.5) → `X_sdxl_best_nu`
  **22095463**, evale 22095464/65/71. Predykcja: odzyska głównie DINO.
- **C. Statystyka:** ich liczby z jednego ziarna; nasz rozrzut 1.20 IA na dwu. Brakujące 0.56
  mieści się w jednym odchyleniu.

**Adaptacja tekstu, jeśli sufit się potwierdzi:** Option C (`learned_tokens.enabled`, cały wiersz
uczony, `init_from_class`, routing przez `key_prompt: index` — klucz ortogonalny NIETKNIĘTY), a
NIE `R_tail`/`ortho_tokens`, bo tam klucz siedzi w pierwszych 128 wymiarach wiersza embeddingu i
jest szumem dla enkodera. `R_tail` dał +0.011 DINO na 2 ziarnach (SD-1.5). **Obie ścieżki są
SD-1.5-only** — `tokens.py` nie zna `text_encoder_2`/`tokenizer_2`; SDXL wymaga rejestracji i
treningu w obu enkoderach. O(T) pamięci (768+1280 floatów/koncept) — jawnie policzyć w pracy.

### Przegląd kodu toru SDXL — trzy odstępstwa, żadne na SD-1.5 (2026-09-07, wieczór)

Przegląd na pytanie „czy gorsze wyniki SDXL to bug". Sprawdzone i czyste: `encode_text` SDXL
(penultimate layers obu enkoderów, pooled z TE2 — jak pipeline), ramka per próbka i
`repeat_interleave` (b-major zgodne z `head_to_batch_dim`), maska tokenowa między tokenizerami
(wspólne BPE), `ground_tok_dim`=1280, bank teł 1024², scheduler z configu SDXL, VAE fp32,
preprocessing metryk. Trzy rzeczy są SDXL-specyficzne i niezamierzone — **naprawione, jeszcze
niezmierzone**:

1. **Gałąź uncond w CFG była niespójna** (`sampling.py`, `gen_cifc.py`): sekwencja = zakodowany
   prompt negatywny (`NEG`), pooled `text_embeds` = zera. Pipeline SDXL zeruje OBA naraz (i tylko
   przy pustym negatywie); przy podanym negatywie oba pochodzą z enkoderów. Model nigdy nie widział
   kombinacji (NEG, 0), a błąd uncond mnoży się w CFG przez (1−7.5). Dotyczy KAŻDEJ liczby SDXL
   (baza, grounding, baseline'y) — kandydat na problem A („baza bez zapasu"). Fix: pooled
   negatywu idzie do `added_cond` uncond (`uncond_pooled` w `ddim_sample`; `gen_cifc`, `_gen_one`,
   `infer`, `_ground_iou`). Stare zachowanie: `gen_cifc --uncond_legacy_zero`. **Inference-only** →
   izolowalne na istniejącym checkpoincie.
2. **`GroundedAttnProcessor` liczył logity uwagi w bf16** (`baddbmm` w `q.dtype`), gdy domyślny
   SDPA trzyma QK^T i softmax w fp32 (`upcast_attention: null` w UNecie SDXL nic tu nie zmienia).
   Ulp bf16 przy |logit|≈8 to 0.0625 → ~3% błędu wag uwagi na wszystkich 70 attn2 przez cały
   trening bf16. Baza (bez `ground_cond`) trenuje przez SDPA → „podatek groundingu" na SDXL był
   zmieszany z numeryką kernela; na SD-1.5 trening jest fp32, więc różnicy nie było (podatek 0.3).
   Fix: logity w fp32. Zmienia numerykę eval fp16 także na SD-1.5 z groundingiem (w granicach
   szumu; headline `P_best` liczony przed zmianą).
3. **Mikro-warunkowanie SDXL w treningu było stałe** `(1024,1024,0,0,1024,1024)` dla KAŻDEGO
   obrazu. Źródła: `dog` 687–781 px, `ink_painting` 512 px, `drawing` 645–910 px, plus crop
   80–100% z `augment` → powiększane 1.3–2× i deklarowane jako natywne, nieprzycięte 1024.
   `original_size` w SDXL istnieje dokładnie po to, by model nie kojarzył rozmycia z natywną
   rozdzielczością (Podell et al. §2.2; `train_dreambooth_lora_sdxl.py` podaje prawdziwe wartości).
   Fix: `data._load_image` zwraca rozmiar źródła i offset cropu (w układzie po przeskalowaniu),
   `added_cond(orig_size, crop)`; kompozyt na tle 1024 = (1024,1024,0,0). Baseline'y SDXL
   (`train_baselines`, `train_l2dm`) NIE zmienione — używają stałych jak dotąd.

Poza tym: `guidance_scale 7.5` na SDXL (pipeline domyślnie 5.0) — protokół, do sweepu
inference-only; eval fp16 po treningu bf16 — niespójność bez znanego skutku; `grad_clip 1.0` na
normie globalnej przy 87.5 M parametrów tnie częściej niż przy 21 M → trening loguje teraz normę
PRZED klipem (`gnorm`, W&B `grad_norm`). Metryki @224 px słabo widzą rozmycie z (3), więc jego
wpływ na IA jest niepewny; (1) i (2) mają mechanizm działający wprost na IA/TA.

**(1) ZMIERZONE — NULL (2026-09-08, jobs 22132419/23/26).** `X_sdxl_ground_800aug`, ten sam
checkpoint i ziarna, `eval10f` (stara gałąź) vs `eval10f_uc` (pooled negatywu): s=0.4
80.15/79.27/59.67 → 80.18/79.13/59.57; s=0.5 77.35/80.92/62.08 → 77.32/80.83/61.93; s=0.6
75.16/82.09/63.98 → 75.16/81.97/63.82. ΔTA ≤ 0.0003, ΔIA −0.001, ΔDINO −0.0015 — poniżej szumu
samplingowego, znak minimalnie ujemny. Odstępstwo od pipeline'u było realne, ale model jest na
nie niewrażliwy: **luka do CIDM na SDXL NIE siedzi w gałęzi uncond.** Poprawka zostaje (zgodność
z protokołem SDXL), ale nie jest wyjaśnieniem problemu A. `grad_clip` też odpada jako hipoteza:
`gnorm` w treningach SDXL to 0.003–0.34, klip przy 1.0 nigdy nie działa.

**Plan pomiaru (rozłącznie):** (a) ✓ powyżej; (b) retrening `X_sdxl_base800`
(dotknięty tylko (3)) i `X_sdxl_best_nu` ((2)+(3)+`paste_no_upscale`) na naprawionym kodzie —
kolejka z 2026-09-07 anulowana przed startem, bo `run.sh` pulluje przy wysyłce, a job czyta drzewo
przy starcie, więc push zmieniłby kod czekających zadań pod starym `run-info.txt`.

**Stan wobec CIDM (2026-09-07):**

| | config | wobec CIDM przy ich TA |
|---|---|---|
| SD-1.5 | `P_best` s=0.45 | **+2.23 IA**, dominacja na obu osiach |
| SDXL | `800aug` (bez flush) | −0.56 IA (2 ziarna, rozrzut 1.20) |

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

## 5b. Faza T — skalowanie do 50 konceptów (2026-09-10…14)

**Strumień.** `configs/phaseT/T50_mixed.yaml`: zadania 0–9 to dziesiątka CIFC **bitowo identyczna**
z `P_paper` (ta sama kolejność, te same hiperparametry), zadania 10–49 to 40 konceptów
CustomConcept101 z captionami BLIP w reżimie CIFC. Wykluczone: `pet_cat1` i `plushie_teddybear`
(md5-identyczne z CIFC) oraz wszystkie `scene_*`. Punkt T=10 tej krzywej **musi** odtworzyć
headline — i odtwarza.

### 5b.1 Krzywa przy stałej skali s=0.45 (średnia po konceptach 0..k)

| T | TA | IA | DINO |
|---|---|---|---|
| 10 | 0.7483 | 0.8011 | 0.6253 |
| 20 | 0.7583 | 0.7397 | 0.5206 |
| 30 | 0.7633 | 0.7358 | 0.5266 |
| 40 | 0.7571 | 0.7338 | 0.5185 |
| 50 | 0.7628 | 0.7171 | 0.4929 |

Cały spadek siedzi w skoku 10→20, czyli tam, gdzie wchodzi CC101; między 20 a 40 jest **płasko**
(−0.006 IA, −0.002 DINO). To zmiana składu zbioru, nie zapominanie.

**Rozkład luki T=10→50** (−0.084 IA / −0.132 DINO). Te same 10 konceptów CIFC ocenione
checkpointem 49 dają IA 0.7471 / DINO 0.5501, więc:
- **zapominanie/dryf na ustalonym zbiorze:** −0.054 IA / −0.075 DINO
- **zmiana składu zbioru** (40 trudniejszych konceptów CC101): −0.030 IA / −0.057 DINO

Czterdziestka CC101 wychodzi na IA 0.710 / DINO 0.479, czyli −0.037 / −0.071 poniżej dziesiątki
CIFC przy tym samym checkpoincie. To 36% luki w IA i 43% w DINO, i **żadna metoda CL tego nie
ruszy** — w pracy trzeba raportować krzywą na jednorodnym podzbiorze albo nazwać ten człon wprost.

**Zapominanie na ustalonych podzbiorach** (s=0.45) rozkłada się skrajnie nierówno:

| podzbiór | horyzont | ΔIA | ΔDINO |
|---|---|---|---|
| CIFC 0–9 | T=10→50 (40 zadań) | −0.054 | −0.075 |
| CC101 10–19 | T=20→50 (30 zadań) | −0.000 | −0.015 |
| CC101 20–29 | T=30→50 (20 zadań) | −0.013 | −0.026 |

Na porównywalnym horyzoncie 30 zadań CIFC traci 3× więcej niż CC101 nawet po znormalizowaniu na
poziom wyjściowy (12% wobec 3.5%). Hipoteza robocza: człon kohortowy śledzi **o ile urósł strumień
od punktu odniesienia** (CIFC 5×, CC101 10–19 2.5×), czyli rozcieńczanie pojemności, nie długość
łańcucha kotwic.

**Koncept zerowy jest anomalią.** Per koncept, ΔDINO T=10→50: `dog` **−28.7%**, `dog2` −12.5%,
`drawing` −13.2%, `duck_toy` −12.0%, `cat2` −10.9%, reszta −6…−9%. Pozycje 1–9 **nie mają żadnego
uporządkowania** — to nie jest „im starszy, tym gorzej", tylko konkretnie zadanie 0. Style wypadają
**lepiej** niż obiekty (−9.6% wobec −12.3%), więc „style są kruche" odpada. Przyczyna niewyjaśniona;
sprawdzone i odrzucone: geometria klucza (patrz 5b.4) i zimny emiter (`dog` ma **najwyższe** DINO
z dziesiątki przy T=10, więc nie jest niedouczony). Jedyny czysty test to ta sama pięćdziesiątka
w innej kolejności — nie zrobione.

### 5b.2 PRÓG SZUMU — dotyczy wstecz wszystkiego przy T=50

Trzy ziarna **identycznej** konfiguracji bazowej (2024/2025/2026), wszystkie w pełni interpolowane
do wspólnego TA=0.747:

| wielkość | średnia | sd | rozstęp |
|---|---|---|---|
| IA @T=10 | 0.8025 | **0.0006** | 0.0012 |
| DINO @T=10 | 0.6296 | **0.0037** | 0.0069 |
| IA @T=50 | 0.7748 | 0.0034 | 0.0066 |
| DINO @T=50 | 0.5737 | **0.0133** | 0.0242 |
| ΔDINO (zapominanie) | −0.0560 | **0.0161** | 0.0310 |

**T=10 jest praktycznie deterministyczne, T=50 nie.** Szum narasta wzdłuż strumienia: 50
sekwencyjnych zadań kumuluje drobne różnice. Konsekwencje, wszystkie wsteczne:
- każde twierdzenie o headlinie wobec CIDM (T=10) jest twarde — sd 0.0006 na IA;
- każde twierdzenie o T=50 wymaga 3 ziaren albo efektu > 0.03;
- **najgorsza jest sama różnica ΔDINO** (sd 0.0161), bo odejmuje dwie zaszumione liczby.
  Raportować **poziom DINO@T=50**, nie deltę;
- wcześniejszy wniosek, że β=300 daje „−15% zapominania" (0.0323 wobec 0.0380), **opisywał szum**.

### 5b.3 Warianty metody — jedenaście prób, dwie wygrane

Wszystko przy zrównanym TA=0.747, te same 10 konceptów CIFC, odczyt interpolowany po skalach
(`scripts/_curve.py`). Baza = ziarno 2024.

| wariant | DINO @T=10 | DINO @T=50 | werdykt |
|---|---|---|---|
| **baza** (rolling, β=100, czynniki) | 0.6269 | 0.5890 | — |
| **key_renorm** | 0.6282 | **0.5965** | **jedyna wygrana**, patrz 5b.5 |
| ground_anchor | 0.6394 | 0.6001 | poziom wyżej, **nachylenie bez zmian** |
| β=300 (czynniki) | 0.6219 | 0.5896 | w granicach szumu |
| q192h128 (gardło 128, baza 192) | 0.5960 | 0.5511 | gorsze też przy T=10 |
| kotwica na ΔW, β=100 | 0.6169 | 0.4499 | za słaby więz |
| kotwica na ΔW, β=500 | 0.6260 | 0.5051 | wciąż za słaby |
| **kotwica na ΔW, β=2500** | 0.6200 | **0.6106** | **najlepszy wynik**, patrz niżej |
| kotwica na ΔW, β=10000 | 0.6143 | 0.5909 | za mocny, optimum minięte |
| era N=10, β=100 | 0.4232 | 0.3823 | adapter zduszony |
| ema 0.9, β=100 | 0.4912 | 0.4286 | jw. |
| era β=35 / ema β=40 | 0.525 / 0.578 | 0.362 / 0.429 | poziom wraca, nachylenie się psuje |
| q64h128 (baza wiąże) | 0.5219 | 0.2344 | kontrola, potwierdza pomiar `basis_L99` |

**Kotwice era/ema są zamknięte z obu stron β.** Przy β=100 duszą magnitudę adaptera 2.4–2.9×
(`dw_mag`: rolling 9.37/12.33, era 3.23/5.53, ema 3.84/5.05 przy T=10/50) przy prawie nienaruszonym
zróżnicowaniu (`dw99` przy T=50: rolling 47, ema 47, era 42). Ale **przywrócenie magnitudy nie
przywraca jakości**: przy s=1.5 (3.3× skali) era daje DINO 0.452 wobec 0.625 bazy, a jej krzywa
kompromisu biegnie **wstecz** — zejście z TA obniża IA. Przy β dobranym pod magnitudę (35/40)
poziom wraca, ale nachylenie się psuje. Więz bezwzględny jest albo za ciasny, albo za luźny.

**Kotwica na dW: argument poprawny, i po skalibrowaniu beta ROWNIEZ empirycznie najlepszy wynik.**
Rozklad `dW = x_L @ x_R` nie jest jednoznaczny -- `(x_L R, R^-1 x_R)` daje te sama funkcje -- wiec
MSE na czynnikach karze tez czysta zmiane cechowania. Zweryfikowane liczbowo: reparametryzacja
zostawiajaca `dW` bez zmian (max roznica 9.5e-07) przesuwa `_reg_mse` z 3.974 na 10.658, a `_reg_dw`
zostaje na 7.643927 co do ostatniej cyfry. Argument z LoRAGen (ICLR 2026, `tsinghua-fib-lab/LoRAGen`).

Czlon na dW jest przy tej samej wadze **okolo piec razy mniejszy liczbowo** (log zadania 49:
reg 0.0005 na czynnikach wobec 0.0001 na dW), wiec beta=100 i 500 mierzyly za slaby wiez, a nie zla
przestrzen. Sweep po beta jest **niemonotoniczny i ma optimum przy 2500**:

| beta (dW) | DINO @T=10 | DINO @T=50 | dDINO |
|---|---|---|---|
| 100 | 0.6169 | 0.4499 | −0.167 |
| 500 | 0.6260 | 0.5051 | −0.121 |
| **2500** | 0.6200 | **0.6106** | **−0.0094** |
| 10000 | 0.6143 | 0.5909 | −0.0234 |
| baza (czynniki, beta=100) | 0.6269 | 0.5890 | −0.0380 |

**0.6106 to najwyzszy `DINO@T=50` ze wszystkiego, co zmierzylismy** — +0.0216 nad baza przy tym
samym ziarnie, wiecej niz sigma (0.0133), przy zapominaniu czterokrotnie mniejszym i minimalnym
koszcie plastycznosci. Niemonotonicznosc (10000 gorsze od 2500) znaczy, ze to optimum, a nie
plateau. **To jest jeden przebieg** — dwa ziarna potwierdzajace sa w kolejce (`22402991/92` plus
lancuchy koncow). Odczyt T=10 przy beta=2500 byl minimalnie ekstrapolowany (TA siega 0.746 przy
celu 0.747), dlatego nowym ziarnom dosylamy skale s=0.3.

Implementacja: `reg.space`, domyslnie `factors`; `_reg_dw` liczy odleglosc bez materializowania
`dW` (tozsamosc sladu, koszt O(r^2(in+out)), zgodnosc z rachunkiem wprost 0.0e+00).

**Jesli to bedzie headline — co opisujemy w pracy** (ustalone 2026-09-14):
* **Teza, nie implementacja.** Regularyzacja wyjscia von Oswalda zastosowana do LoRA karze
  wolnosc cechowania: `(x_L R, R^-1 x_R)` to ta sama funkcja, a MSE na czynnikach traktuje je
  jak rozne. Kotwica powinna wiec dzialac na `dW`. Do tego obserwacja empiryczna: wymaga to
  bety o **dwa rzedy** wiekszej niz na czynnikach i ma **niemonotoniczne optimum** (2500 lepsze
  od 500 i od 10000) — sama zmiana przestrzeni bez rekalibracji wyglada jak porazka, i tak nam
  sie to najpierw pokazalo.
* **Atrybucja.** Argument o niejednoznacznosci rozkladu nalezy sie LoRAGen (ICLR 2026) i musi byc
  zacytowany. Nasze jest przeniesienie go na kotwice w uczeniu ciaglym oraz kalibracja bety.
* **Tozsamosc sladu: jedno zdanie w opisie metody albo przypis, NIE osobny wklad.** To
  standardowa algebra (cyklicznosc sladu + `||M||_F^2 = tr(M^T M)`), wiec podawanie jej jako
  nowosci wygladaloby naiwnie. Potrzebna jest wylacznie po to, zeby uprzedzic pytanie
  o koszt: `||A1 B1 - A2 B2||_F^2 = tr(G1 H1) - 2 tr(Gx Hx) + tr(G2 H2)` przy `G = A^T A`
  i `H = B B^T` rozmiaru `[r, r]`, czyli `O(r^2(in+out))` zamiast materializowania `[in, out]`.
  Dla typowej warstwy (in=768, out=1280, r=4) to 32.8 tys. mnozen zamiast 3.9 mln i 32 liczby
  w pamieci zamiast 983 tys. Bez tego regularyzacja przy 50 kotwicach i 64 warstwach po prostu
  by sie nie zmiescila — i to jest jedyny powod, dla ktorego o niej wspominamy.

### 5b.4 Pomiary architektury (bez GPU-godzin, `scripts/_spectrum.py`)

**Rząd efektywny emitowanych adapterów** (99% energii, mediana po 64 warstwach, macierz centrowana
po konceptach). Liczony na `dW` przez macierz Grama — **liczony na `x_L` jest skażony cechowaniem
i zaniża wynik**:

| T | sufit danych (T−1) | `dw99` | `lora_L99` (czynniki) | % sufitu architektury (50) |
|---|---|---|---|---|
| 10 | 9 | 9 | 9 | 18% |
| 30 | 29 | 28 | 27 | 56% |
| 50 | 49 | **47** | 38 | **94%** |

Koncepty **nie zlewają się** — 47 z 49 możliwych kierunków. Ale `head_hidden=50` jest zajęte
w 94%. Pierwotny odczyt (38, „wykorzystanie spada do 78%") był artefaktem cechowania.
**Mimo to poszerzenie gardła nie pomogło** (`q192h128` gorsze także przy T=10), więc wysokie
zajęcie nie znaczy, że to ono jest ograniczeniem.

**Zapotrzebowanie na bazę** (`basis_L99` — ile wymiarów `R^in` zajmują łącznie kolumny `x_L`):
39 przy T=10, 104 przy T=30, **144 przy T=50**. Dlatego `basis_q=64` wiąże mocniej niż samo gardło
i `q64h128` wypadło najgorzej z całej serii. To domyka też Fazę F: q=32 przy zapotrzebowaniu 40
kosztowało −0.057, q=128 tylko −0.012.

**Normy kluczy.** Gram-Schmidt odejmuje jeden wymiar na zadanie, więc reszta kurczy się jak
`sqrt(key_dim − k)`: zmierzone 11.31 przy k=0 i mediana 9.92 przy 50 zadaniach. Ekstrapolacja:
6.2 przy T=90 i **dokładnie 0 przy T = key_dim = 128** — twarda ściana architektury, niezależna od
`head_hidden`. Do benchmarku T=90 trzeba podnieść `key_dim` (koszt: tylko pierwsza warstwa głowic,
+3 M parametrów). Klucz zadania 0 **nie jest** odstępstwem (11.135 wobec 11.326 dla zadania 1),
więc geometria klucza nie tłumaczy anomalii `dog`.

### 5b.7 Ranga 8 kupuje jakosc za PARYTET PAMIECIOWY (2026-09-15)

Sweep pokazal, ze czolowka ma wspolny mianownik: piec najlepszych punktow ma `hyper.rank = 8`
(DINO@T50 0.5715-0.6314 wobec bazy 0.5917/0.5708/0.5664 na trzech ziarnach), a retencja
T=10 -> T=50 poprawia sie z −0.034…−0.040 na −0.009…−0.018.

**Ale ranga wchodzi wprost w stala pamieciowa.** Glowica to `Linear(cond,h) -> SiLU ->
Linear(h, in*r)`, wiec czlon zalezny od rangi dominuje:

```
dense:  4.92 M + 51*r*S          S = suma(in+out) po 64 warstwach ~ 79 300
basis:  4.92 M + q*(S + 6528*r)
```

| wariant | parametry | prog oplacalnosci vs magazyn CIDM (0.426 M/koncept) |
|---|---|---|
| dense r=4 (dzis) | 21.1 M | **~49 konceptow** |
| dense r=8 | **37.3 M** | ~88 |
| r=8 + basis q=144 | 23.8 M | ~56 |
| r=8 + basis q=288 | 42.8 M | ~100 |

Przy T=50 magazyn CIDM wazy 21.3 M. Ranga 4 jest wiec z nim **na styk**, a ranga 8 przegrywa
pamieciowo **dokladnie w punkcie, ktory demonstrujemy**. Teza asymptotyczna `O(1)` vs `O(T)`
przezywa przy kazdej randze; psuje sie teza praktyczna, bo prog wychodzi poza pokazywany zakres.

**`basis_q` jako mitygacja: prawdopodobnie nie.** Zapotrzebowanie `basis_L99` przy randze 4
wynosi 144 przy T=50 (5b.4), ale przy randze 8 kazde zadanie wnosi dwa razy wiecej kolumn
`x_L`, wiec moze wyjsc ~288 — a przy `q = 246` baza `q*S` zrownuje sie z kosztem dense i
oszczednosc znika. Do tego zmierzona jakosc `basis_q` jest zla: `T50_q192h128` daje przy T=50
DINO 0.4970 wobec bazy 0.5501 (−0.053), czyli WIECEJ niz ranga 8 daje na plus, i bylo gorsze
takze przy T=10, gdzie zapotrzebowanie to ledwie 39. Pomiar `basis_L99` przy randze 8 (zadania
22427932 dla p001/`factors` i 22428011 dla p029/`dw`) rozstrzyga, czy kombinacja jest w ogole
mozliwa pamieciowo.

**Rekomendacja robocza: headline zostaje przy randze 4, ranga 8 idzie jako ablacja skalowania.**
Kosztuje zero (dane na oba warianty juz sa), zdejmuje przetrenowanie `P_paper` ze sciezki
krytycznej i jest samo w sobie wynikiem: podniesienie rangi kupuje +0.04 DINO i trzykrotnie
lepsza retencje przy koszcie, ktory NIE ROSNIE z liczba konceptow — gałki, ktorej bank
adapterow nie ma.

### 5b.8 `head_hidden >= T-1` to warunek architektury, nie hiperparametr (2026-09-15)

Emitowana LoRA to `W2*SiLU(W1*c+b1) + b2`, wiec nieliniowosc stoi PRZED warstwa wyjsciowa
i obraz lezy w zbiorze afinicznym `b2 + span(kolumny W2)` o wymiarze **co najwyzej `h`**.
Macierz adapterow wycentrowana po konceptach ma rzad co najwyzej `T-1`. Zeby koncepty nie
byly ZMUSZONE dzielic kierunkow, potrzeba wiec `h >= T-1`.

Przy T=50 i `h=50` mieszcimy sie o jeden wymiar — i dokladnie to mierzy 5b.4 (`dw99` = 47 z 49,
gardlo zajete w 94%). **Ograniczenie jest niezalezne od rangi**: granica jest liczba kolumn
`W2`, nie szerokosc wyjscia. Wniosek praktyczny: `head_hidden` NIE jest dzwignia oszczednosciowa
przy T=50 — zwezenie do 32 zamknęloby 50 konceptow w 32 wymiarach.

**Luka w planie na T=90.** Sekcja 6 zapowiada benchmark CC101 do T=90 i wymienia jako warunek
tylko wiekszy `key_dim` (bo przy 128 norma klucza spada do zera przy T=128). Tym samym
argumentem `head_hidden=50` jest sciana TWARDSZA: przy T=90 trzeba `h >= 89`. Klucz wytrzymuje
do 128 zadan, gardlo peka juz przy 51. Oba trzeba podniesc razem, nie zamiast siebie.

### 5b.9 Stan odczytu sweepu po usunieciu `scale_cond` (2026-09-15)

Na 12 punktach czystego podzbioru **sigma reszt DINO@T50 wynosi 0.0158**, wobec zakladanych
0.035. SE spadly do 0.008-0.010, czyli **juz przy 12 punktach mamy dokladnosc, ktora miala
wymagac 47**. Prog decyzyjny to ~0.018, nie 0.020.

Zaden efekt glowny nie przekracza jeszcze progu, ale kolejnosc jest pouczajaca:
`reg.weight` **+0.019** (1.9 SE, i TA bez zmian), `hyper.rank` +0.013, `reg.space` (dw) −0.007.
Czyli **wniosek „ranga 8 wygrywa" pochodzi z rankingu czolowki, a nie z modelu** — a model
wskazuje raczej sile regularyzacji.

Wspiera to sklad danych: obydwa punkty rangi 4, ktore mamy (p000 beta=176, p036 beta=59), maja
kotwice `dw` przy beta ponizej 200, czyli praktycznie wylaczona (czlon na `dW` jest ~5x mniejszy
liczbowo, punkt pracy to 2500). To samo widac w randze 8: p003 (`dw`, beta=151) ma 0.5656,
najgorszy wynik calej rangi 8. **Nie mamy wiec ANI JEDNEGO punktu rangi 4 z dzialajaca
regularyzacja** — a czekaja trzy takie: p027 (`factors`, 1460), p031 (`factors`, 5810)
i p022 (`dw`, 7120, czyli pierwszy `dw` powyzej progu). Dlatego 15.09 wstrzymano czekajace
punkty rangi 2 i 8, zeby te trzy weszly pierwsze.

**Macierze zapominania przy T=50 NIE SA policzone dla zadnego przebiegu.** Trening zapisuje
`fresh/`, `forgetting/`, `final/`, ale `cifc_metrics` nigdy na nich nie poszlo; krzywe licza sie
z `--only_tasks`, wiec raportuja forgetting 0.0. Liczby o retencji powyzej to roznica
T=10 -> T=50 na tych samych 10 konceptach CIFC, co jest dobra miara utrzymania na strumieniu,
ale NIE jest peak-final z macierzy 55 komorek i tak trzeba to opisac.

### 5b.5 `preserve_norm` był no-opem — `key_renorm` jest jego naprawą

`manager.py` liczył `h / ||h|| * ||h||`, czyli dzielił i mnożył przez **tę samą** normę po
rzutowaniu: zwracał `h` bez zmian. Docstring mówi, że miało skalować „back to ||h||", czyli do
normy **sprzed** Grama-Schmidta. Flaga nie robiła nic, więc wcześniejszy wniosek „preserve_norm
nie pomaga" opisywał szum.

**Nie naprawione w miejscu:** 55 configów ma `preserve_norm: true` i ich wyniki powstały z no-opem,
więc zmiana znaczenia flagi uczyniłaby je nieodtwarzalnymi z commita. Zamierzone zachowanie żyje
pod nową flagą `key_renorm`, domyślnie wyłączoną.

**Wynik, porównanie sparowane** (to samo ziarno = ta sama kolejność danych i inicjalizacja):

| ziarno | baza DINO@T=50 | key_renorm | różnica |
|---|---|---|---|
| 2024 | 0.5890 | 0.5965 | **+0.0075** |
| 2025 | 0.5672 | 0.5762 | **+0.0090** |

Dwie pary, zgodny znak, praktycznie ta sama wielkość — rozrzut różnicy 0.0015 wobec 0.0133
rozrzutu niesparowanego. Przy T=10 obie pary neutralne (+0.0013, +0.0006), więc bez kosztu
plastyczności. **Zysk na IA się nie powtórzył** (+0.0106 przy 2024, +0.0001 przy 2025) — powtarzalny
jest tylko DINO. Trzecie ziarno w toku. Zastrzeżenie: to leczy skalę, nie utracone kierunki —
przy T = key_dim reszta jest zerowa i nie ma czego normalizować.

### 5b.6 Ewaluacja na wszystkich 50 konceptach i przekątna

Checkpoint końcowy, 10 000 obrazów na skalę: s=0.45 → TA 0.7628 / IA 0.7171 / DINO 0.4929;
s=0.60 → TA 0.7374 / IA 0.7504 / DINO 0.5414.

Przekątna (każdy koncept oceniony checkpointem tuż po swojej nauce, `--diagonal` w `gen_cifc`):
s=0.45 → TA 0.7499 / IA 0.6669 / DINO 0.4101; s=0.60 → TA 0.7025 / IA 0.7783 / DINO 0.5309.
Po zrównaniu TA na 0.744 przekątna daje 0.681 IA / 0.425 DINO wobec końcowych 0.742 / 0.529, czyli
**+0.061 IA i +0.104 DINO na korzyść checkpointu końcowego**. Nie wpisywać tego jako „ujemnego
zapominania": przekątna ma wadę konstrukcyjną, bo koncept 0 jest oceniany siecią po 400 krokach
treningu w ogóle. Wczesne wpisy mierzą „ledwo nauczoną sieć", nie „świeżo nauczony koncept".

---

## 6. W toku / otwarte

### Sweep: pierwszy istotny efekt — `reg.weight` (odczyt 2026-09-15, 15:20)

Model efektów głównych na podzbiorze `scale_cond=false`, odczyt przy wspólnej skali s=0.45,
**n=14** (doszły p039, p041), sigma reszt 0.0132:

| oś | efekt na DINO@T50 | SE | | efekt na TA@T50 | SE |
|---|---|---|---|---|---|
| `reg.weight` (log10) | **+0.0198** | 0.0064 | **3.1 SE** | −0.0041 | 0.0041 |
| `hyper.rank` (log2) | +0.0130 | 0.0067 | 1.9 SE | +0.0076 | 0.0043 |
| pozostałe | ≤ 0.007 | — | szum | ≤ 0.004 | — |

Wniosek: **większa waga kotwicy podnosi tożsamość przy T=50 bez mierzalnego kosztu TA** — to
jedyna oś, która przekroczyła 2 SE. `hyper.rank` stoi na granicy, ale rangi 8 nie bierzemy
(parytet pamięciowy, 5b.7). Spójne z rankingiem: p039 (ranga 2, w=453) 0.5680 wyprzedza
najlepszy punkt rangi 4 (p000, w=176) 0.5617 — waga liczy się bardziej niż ranga.

Rozstrzygające są trzy punkty rangi 4 o dużej wadze, wciąż w kolejce: p022 (w=7120), p027
(1460), p031 (5810). Jeśli efekt się utrzyma, kandydat na headline to ranga 4 z wagą rzędu
10³–10⁴ i drugie ziarno na nim. Kolejka: 7 elementów PD (Priority), nic nie liczy.

Aktualizacja 23:20: kolejka ruszyła — p022/p027/p031 liczą się od ~20:40 (2 h 36 min w chwili
odczytu), `release_rank.sh` zwolnił rangę 2 (p043/p044/p045 w toku). Ranking kandydatów przy
T=50 po dojściu p039/p041 **z odfiltrowaną rangą 8** (p001, p003, p006, p021, p029, p038, p041
odpadają — parytet pamięciowy): p039 (r2) 0.568, p009 (r2) 0.563, p000 (r4) 0.562, p023 (r2)
0.558, p018 (r2) 0.554, p007 (r2) 0.553, p036 (r4) 0.544; przy T=10: p039 0.610, p036 0.602,
p000 0.600. Ranga 8 prowadzi o ~0.06 DINO@50, ale jest poza dyskusją. Decyzja o headline czeka
na trzy punkty rangi 4.

**`reg.weight` z podziałem na `reg.space` (23:30, `_sweep_split.py` w $SCRATCH; jednowymiarowe
nachylenie DINO@50 na log10(w) przy s045, bo połówki sweepu nie mają stopni swobody na pełny
model).** Wszystkie rangi: `factors` (n=7) +0.024 ± 0.014, `dw` (n=7) **+0.040 ± 0.009** (4.3 SE);
TA w obu płaskie (±0.002 ± 0.005). Efekt jest więc w obu przestrzeniach, ale mocniejszy i czystszy
w `dw`. Zastrzeżenie: w obu grupach duże wagi to prawie wyłącznie ranga 8 (factors: p001/p006/p038;
dw: p029/p021/p041), więc na całości nachylenie miesza wagę z rangą. Po odcięciu rangi 8 zostaje
7 punktów: `factors` (ranga 2, n=4: p007/p009/p023/p018, w=96…4880) +0.007 ± 0.010 — **brak
efektu**, a p018 z w=4880 nie wyprzedza p009 z w=386; `dw` (n=3: p036/p000 r4, p039 r2,
w=59…453) +0.028 ± 0.009, ale to trzy punkty na jednym rzędzie wielkości i TA spada tam
−0.0125 na dekadę (SE 0.0003 — z trzech punktów, nie ufać). Przy T=10 w rangach ≤4 nic
istotnego na DINO. Wniosek: dowód na `reg.weight` w rangach 2/4 jest na razie **słaby i tylko w
`dw`**; ranga 2 przy dużej wadze (p018) go nie potwierdza. Rozstrzygną p022/p027/p031 (ranga 4,
w=7120/1460/5810): p022 jest w `dw`, p027 i p031 w `factors`, więc po ich dojściu obie
przestrzenie będą miały rangę 4 z dużą wagą i da się rozdzielić przestrzeń od wagi.

### Sweep DOMKNIĘTY — 21 punktów `scale_cond=false`, odczyt 2026-09-16 rano

Ostatnie siedem punktów doszło w nocy (p022/p027/p031 ranga 4 po 7–8 h; p043/p044/p045 ranga 2/4
po 6–7.5 h; p025). Pełny model efektów głównych (n=21, s=0.45):

| oś | DINO@T50 | SE | | TA@T50 | SE | | DINO@T10 | TA@T10 |
|---|---|---|---|---|---|---|---|---|
| `reg.weight` (log10) | **+0.0176** | 0.0030 | 5.9 SE | −0.0022 | 0.0019 | | +0.0051 ± 0.0032 | −0.0009 |
| `hyper.rank` (log2) | **+0.0151** | 0.0037 | 4.1 SE | +0.0041 | 0.0024 | | −0.0051 ± 0.0040 | **+0.0133 ± 0.0021** |
| `reg.space` (dw=+1) | **−0.0086** | 0.0037 | 2.3 SE | **+0.0050** | 0.0024 | | +0.0045 | +0.0008 |
| `key_dim` (log2) | −0.0006 | — | | **+0.0045** | 0.0019 | | −0.0045 | **+0.0053 ± 0.0017** |
| pozostałe | ≤ 0.005 | — | szum | | | | | |

**`reg.weight` z podziałem na przestrzeń, tylko rangi 2/4 (n=7+7), nachylenie DINO@50 na
log10(w):** `factors` **+0.027 ± 0.010**, `dw` **+0.029 ± 0.006** — teraz istotne w OBU, a więc
efekt wagi jest realny, nie artefakt rangi 8. TA: `factors` płaskie (+0.001 ± 0.004), `dw` spada
−0.011 ± 0.002 na dekadę. Przy T=10 waga też pomaga (factors +0.015 ± 0.006, dw +0.008 ± 0.004),
mniej — T=10 blisko nasycenia, ale nie „zero", jak wyglądało przy n=14.

**Kandydaci rangi ≤ 4 przy zrównanym TA=0.747 (`score.json`, objective = DINO@50, kara za
ekstrapolację):**

| punkt | ranga | space | w | DINO@50 | DINO@10 | interp. | uwaga |
|---|---|---|---|---|---|---|---|
| **p031** | 4 | factors | 5810 | **0.6086** | 0.633 | tak/tak | najlepszy w klasie pamięciowej |
| p027 | 4 | factors | 1460 | 0.6036 | 0.635 | tak/tak | |
| p022 | 4 | dw | 7120 | 0.6034 | 0.630 | tak/tak | TA niżej (0.781 przy s045) |
| p045 | 4 | dw | 148 | (0.611) | 0.635 | **nie**/tak | odczyt ekstrapolowany — nie ufać |
| p043 | 4 | dw | 196 | (0.611) | 0.644 | **nie**/tak | jw. |
| p000 | 4 | dw | 176 | (0.589) | 0.627 | nie/tak | dotychczasowy lider rangi 4 |
| p039 | 2 | dw | 453 | (0.585) | 0.625 | nie/tak | |
| p044 | 2 | dw | 556 | 0.527 | 0.616 | tak/nie | ranga 2 przy dużej wadze — słabo, jak p018 |

Dla porównania ranga 8: p001 0.631, p029 0.624 — o ~0.02 nad p031, ale poza parytetem pamięci.
Wobec `T50_mixed` (checkpoint pracy, DINO@50 0.550 przy s=0.45; 0.571 przy s045 dla p031 przy tej
samej skali) p031 daje **+0.02 przy wspólnej skali i +0.06 przy zrównanym TA** — przy σ=0.013 na
tej skali to 1.5 i 4.5 σ, drugie ziarno rozstrzygnie. Ranga 2 przy dużej wadze (p018 w=4880 0.531,
p044 w=556 0.527) potwierdza brak pojemności rangi 2 na 50 konceptów.

**Rekomendacja (do decyzji użytkownika):** headline = ranga 4, `factors`, waga ~5·10³ (p031);
następny krok to drugie i trzecie ziarno p031 (i ewentualnie p027 jako kontrola wagi) na pełnym
protokole T=10+T=50, a potem ten config jako checkpoint pracy zamiast `P_paper`/`T50_mixed`.
Skrypty tymczasowe: `_sweep_effects_k.py`, `_sweep_split.py` w `$SCRATCH/continualhyper`.

### DECYZJA: headline = p022 (2026-09-16, ~11:30)

Najlepszy punkt klasy pamięciowej, p031 (0.609), ma `learn_v: true` — wejściem hipersieci jest
`V_t ⊙ z_t`, uczony wektor per zadanie razy klucz Gram–Schmidta (`manager.py:217,469`). To
sprzeczne z tezą pracy („embedding reads no data, orthogonal by construction"), a oś `learn_v`
jest w sweepie zerowa (−0.001 ± 0.004). **Wybrany headline: p022** — ranga 4, `reg.space=dw`
(kotwica na ΔW), β=7120, d_v=256, 800 kroków/zadanie, wd=1e-2, `learn_v=false`,
`scale_cond=false`; DINO@50 przy zrównanym TA **0.603** (interpolowany), DINO@10 0.630 — od p031
o 0.005 niżej, czyli w szumie (σ 0.013). Koszt w tekście: Eq. 7 i dwa zdania o targetach kotwicy
(ΔW zamiast czynników), Setup/Appendix (β, d_v, kroki, wd). Uwaga ze sweepu: `dw` kosztuje TA
(+0.005 dla `factors` na TA@50, 2 SE) — przy zrównanym TA bez różnicy, ale krzywa p022 leży na
niższym TA przy tej samej skali (0.781 przy s=0.45).

Odwołane (zgoda użytkownika): 22476070/71/72 (ziarna i kontrola p031), 22476073 (macierz p031,
~1.6 h policzone), 22476105 (SDXL p031, ~50 min). Wysłane na p022: 22479373 `p022_s2025`,
22479374 `p022_s2026`, 22479377 `p022_w1460` (kontrola wagi), 22479378 macierz forgetting
(zadania 0–9, skale 0.5/0.45), 22479379 generacje do teasera (s=0.6/0.45), 22479380 siatka
placementu (ogród), SDXL `X_sdxl_p022` (config w `outputs/sdxl/X_sdxl_p022/`).
Zostają jako materiał poboczny: `p031/matrix` (niedokończona), `teaser/p031_final`,
`teaser/grid_p031_garden50.jpg`.

Dalsze zadania na p022 (2026-09-16, po południu): 22482151 `p022-topup` (skale 0.3/0.9/1.05 przy
T=10, 0.3/1.05 przy T=50 — pod fig:tradeoff i odczyt bez ekstrapolacji), 22483181 `p022-all50`
(ewaluacja 50 konceptów przy 0.45/0.6, `outputs/sweep/p022/all50`), 22483500 `p022-tab-main` i
22483502 `p022-tab-op` (tab:grounding na **`hyper_after_task09.pt`**, 7 obiektów CIFC przez nową
flagę `--only_concepts`, 50 kroków, K=15). Commity 3a13202 + 29cb70a: `_ground_iou.py
--only_concepts`, `_lora_ortho.py`.

### Ortogonalność wygenerowanych LoR (2026-09-16, 22483498, 15 s)

`scripts/_lora_ortho.py` na p022: kosinus par $\langle\Delta W_i,\Delta W_j\rangle_F$ per warstwa
(tożsamość śladowa, bez materializacji), średnia |cos| poza przekątną; odniesienie: losowe A, B o tych
samych kształtach.

| | dW (T=10) | dW (T=50) | czynnik A | czynnik B | losowe dW |
|---|---|---|---|---|---|
| to_q | 0.039 | 0.030 | 0.16 | 0.18–0.24 | 0.0013 |
| to_k | 0.070 | 0.055 | 0.34 | 0.15–0.20 | 0.0011 |
| to_v | 0.088 | 0.051 | 0.34–0.36 | 0.14–0.24 | 0.0011 |
| to_out | 0.026 | 0.022 | 0.21–0.26 | 0.08–0.11 | 0.0014 |
| **wszystkie** | **0.056** | **0.040** | 0.27 | 0.14–0.20 | 0.0012 |

Odczyt: wygenerowane **aktualizacje ΔW różnych konceptów są bliskie ortogonalności** (średni |cos|
0.04–0.06; najbliższe pary to koty: (cat2, cat) 0.07, (cat, dog) 0.06–0.07 — słaby ślad semantyki),
mimo że nikt tej ortogonalności nie wymusza: ortogonalizujemy tylko wejścia. Nie jest to poziom
losowy (0.001) — sieć współdzieli część kierunków — ale o rząd niżej niż na czynnikach (A: 0.27–0.35),
które mają swobodę cechowania i dzielą wspólny „bias" głowicy. Przy T=50 kosinusy dW są *niższe*
niż przy T=10 (0.040 vs 0.056), czyli więcej konceptów nie zagęszcza wyjść. To argument wobec
Orthogonal Adaptation: ortogonalność wag wychodzi *z* ortogonalności wejść, nie trzeba jej narzucać.
Do dodatku: tabela jak wyżej + mapa cieplna 10×10 (`mean_cos_dw` w `lora_ortho.json`).

### Ten sam checkpoint na T=10 i T=50? Tak — T=10 jest nasycone (odczyt 2026-09-15)

Ten sam model efektów na odpowiedzi przy T=10 (s=0.45, n=14): `reg.weight` +0.0003 ± 0.0062 na
DINO@10 i +0.0014 ± 0.0035 na TA@10 — **zero**. Wyraz wolny DINO@10 = 0.606, σ 0.013, żadna oś
nie przekracza 2 SE: przy dziesięciu konceptach tożsamość jest nasycona (por. sufit danych,
sec:pitfalls). Korelacja DINO@10 z DINO@50 = +0.02; log10(reg.weight) z DINO@50 = +0.78.
Wniosek: **konfiguracja wybrana pod T=50 nic nie kosztuje przy T=10** — jeden checkpoint na oba.
Uwagi: `hyper.rank` +0.0153 ± 0.0037 na TA@10 (4.1 SE) i `key_dim` +0.0069 ± 0.0025 (2.8 SE) —
ranga poza dyskusją (pamięć); p018 (ranga 2, w=4880) traci −0.092 mimo silnej kotwicy, więc
ranga 2 może nie mieć pojemności na 50 konceptów. Skrypty: `_sweep_effects_k.py` (SWEEP_K=9),
`_sweep_pairs.py` w $SCRATCH/continualhyper (tymczasowe).

### Protokół generacji: WSZYSTKO na 50 krokach — decyzja 2026-09-15

Do tej pory tabele raportowane szły na 50 krokach DDIM, a sondy placementu (`_ground_iou.py`,
`_ground_grid.py`, cała `tab:grounding`) na 30 — z bootstrapem K=10 z 30 i harmonogramem GSA na
pierwszych 30%. Decyzja użytkownika: **jeden protokół, 50 kroków, wszędzie**. Konsekwencje:

- bootstrap podajemy jako 30% kroków, czyli **K=15 z 50** (nie 10); harmonogram GSA zostaje 0.3
  (frakcja, skaluje się sam); rusztowanie zostaje 10-krokowe (to osobna mini-generacja, nie
  frakcja głównej — założenie, do zweryfikowania wizualnie);
- `tab:grounding` trzeba **zmierzyć ponownie** przy 50 krokach — liczby z 30 (100% / 0.686 /
  0.802) nie są przenośne, bo K i harmonogram zmieniają punkt pracy; do czasu pomiaru praca
  nadal cytuje 30-krokowe;
- `_ground_grid.py` dostał flagę `--steps` (domyślnie 30, więc stare siatki są odtwarzalne);
  `_ground_iou.py` miał ją od dawna;
- zdanie w Setup pracy („30 steps for the placement probes") do zmiany po pomiarze;
- teaser: siatka placementu ma powstać z `P_paper` przy 50 krokach i K=15, nie z `grid_FINAL`
  (ten jest z `P_ground_gsa_erode`, poprzednika bez augmentacji).

Pomiar: sonda jak `ch-tab2-main`/`ch-tab2-op` (22223351/22223362) z `--steps 50 --bootstrap 15`.

**Pomiar wykonany (2026-09-15, 22437419 `ch-tab50-main` 19 min, 22437420 `ch-tab50-op` 15 min).**
`P_paper`, quads, 7 obiektów × 4 ćwiartki × 3 = 84 generacje, seed0 31337, 50 kroków DDIM,
bootstrap K=15, rusztowanie 10-krokowe, Mask R-CNN R50-FPN-v2:

| wiersz | ćwiartka | IoU>0.5 | zawarcie | wypełnienie | TA | DINO maska | DINO cały | kolor dRGB | det |
|---|---|---|---|---|---|---|---|---|---|
| bez ramki (κ=0) | 25% | 0% | 0.25 | 1.43 | 0.800 | 0.627 | 0.637 | 0.175 | 84/84 |
| κ=1, sched 1.0 | 85% | 45% | 0.65 | 1.68 | 0.784 | 0.714 | 0.704 | 0.097 | 84/84 |
| κ=2, sched 0.3 | 98% | 85% | 0.81 | 1.39 | 0.776 | 0.702 | 0.685 | 0.114 | 84/84 |
| κ=2 + bootstrap 15 | **100%** | **95%** | **0.90** | **1.14** | **0.816** | 0.703 | 0.661 | 0.114 | 84/84 |

Wobec 30 kroków (86% / 0.83 / 1.09 / 0.802 / 0.686) punkt pracy przy 50 krokach jest lepszy na
IoU>0.5 (+9 pkt), zawarciu (+0.07) i TA (+0.014), gorszy o 0.05 na wypełnieniu; tożsamość na masce
0.627→0.703 (było 0.610→0.686). Nowość względem 30 kroków: bootstrap **podnosi** IoU>0.5
(85→95%) i zawarcie (0.81→0.90), nie tylko wypełnienie — zdanie „leaves placement where it was" w
pracy zostało przepisane. Koszt bootstrapu w DINO: −0.024 na całym obrazie, 0.000 na wycinku,
+0.001 na masce — to liczby do zdania o „wrong axis" w sec:grounding-results. Liczby wpisane do
`tab:grounding`, abstraktu, wstępu, §5.5 i Setup (K=15, bez „30 steps for the placement probes").
Dwa wiersze porównawcze (pozycja w promptcie, training-free layout) nadal do policzenia.

Przy okazji: 22425158 `ch-sdxl-lfilm` padł po 9 s — `train_cl.py: error: argument --config:
expected one argument`, czyli runner nie podał ścieżki configu (błąd w lokalnym skrypcie sbatch,
nie w kodzie). Do poprawienia przed ponownym wysłaniem.


**Stan kolejki (Helios, 2026-09-14 wieczor).** Wszystko ma zmniejszone zadanie zasobow do
**8 rdzeni / 32 GB** (zmierzony szczyt: 7.2 GB trening, 9.3 GB ewaluacja). Stare 32/120 GB bylo
realna przeszkoda w planowaniu: wezel ma 489 GB, wiec cztery zadania po 120 GB stawaly na styk
i nie mieszczily sie po cztery na wezel.

- **`22402991/92`** — trening `ΔW` beta=2500 na ziarnach 2025/2026, start szacowany na 15.09
  07:11 i 08:08, limit sciety do 4:00 (zmierzony przebieg 3:16).
- **`22408433/35/36/39` + `22408748/54`** — konce krzywej dla tych ziaren, podpiete **`afterok`**
  do trenigow, wiec ida same. Dodatkowa skala s=0.3 dla T=10, zeby odczyt nie byl ekstrapolowany.
- **`22408427/28`** — konce trzeciego ziarna `key_renorm`; domykaja decyzje z 5b.5.
- **`22403877` (23 punkty x 400 krokow, 6:30) i `22403880` (24 x 800, 10:30)** — sweep, throttle
  12+12, oba `(Priority)`. Szacunek 16.09; dla tablic Slurm liczy go po ostatnim zadaniu, wiec
  pierwsze punkty wejda wczesniej.

**Sweep przepadl raz i to warto pamietac przy restarcie.** 14.09 tablice ruszyly o 12:09, a
naprawiony runner dotarl na klaster o 13:51 — w efekcie 24 z 26 punktow mialy
`output_dir: ./outputs/sweep/config` i 17 rownoleglych zadan nadpisywalo sobie checkpointy
w jednym katalogu. Anulowane i wyslane od nowa; kosztowalo dzien obliczen i wiek w kolejce
(~40 h priorytetu). Lekcje w sekcji 7. Katalog `outputs/sweep/config` (pomieszane checkpointy,
~5 GB) czeka na skasowanie.

**Jak czytac sweep (ustalone zawczasu).** Nie „ktory punkt wyszedl najlepiej" — maksimum z 47
losowan z czystego szumu lezy srednio 2.4 sigma ponad srednia, czyli +0.029 DINO z przypadku.
Decyzja z **efektow glownych**: model liniowy po osmiu osiach na wszystkich punktach naraz
(przy LHS osie sa prawie ortogonalne), SE = 0.29 sigma_total ~ 0.010, dzialamy tylko na tym, co
przekracza 2 SE (~0.020). Wybor konfiguracji **nigdy wprost z rankingu** — 3-4 najwyzsze punkty
ida na dwa dodatkowe ziarna i decyduje porownanie sparowane. Analize prowadzic na `dino_t50`,
a nie na polu `objective`: kara 0.05 za ekstrapolacje byla pomyslana pod optymalizacje bayesowska
i przy modelu liniowym psulaby estymaty. Punktom z `interp_t50: false` dosylac brakujaca skale
i przeliczac.

**Kompozycja wielokonceptowa — POTOK ZBUDOWANY (2026-09-14), czeka na GPU.** Decyzja z 14.09:
robimy ja rownolegle do sweepu, doklejajac sie do scen z pracy CIDM. Stan:

- `assets/composition/scenes.json` — 11 scen z Rys. 3 i 12 w postaci maszynowej: ITP/RTP
  przepisane doslownie, geometria ramek z wektorow ich PDF-ow, parowanie koncept↔ramka po
  KOLORZE (w scenach 3.2, 3.4, 3.5 i 12.1 rozni sie od kolejnosci w RTP — przepisanie promptu
  „z figury" daje zle przypisanie). Mapowanie `V<k>` → task `k−1` jest bitowe wobec kolejnosci
  konceptow w `P_paper` / `X_sdxl_*` / `R_tail_*`; sterownik to sprawdza i przerywa przy
  rozjezdzie. Ramki sa rozlaczne we wszystkich 11 scenach (zweryfikowane) — wagi scalania
  w rowaniu 5 sumuja sie wtedy do 1 wszedzie.
- `src/sampling.py::compose_sample_regions` — **dziala teraz na SDXL**. Dotad nie mogl:
  nie podawal `added_cond_kwargs`, wiec na SDXL wywalal sie od razu, mial zaszyte 512 i
  dekodowal VAE bez dzielenia na kawalki (~1 GB aktywacji na obraz przy 1024). Doszly:
  pooled promptu negatywnego w galezi uncond (ta sama poprawka co w `ddim_sample` z 07.09)
  oraz `--ground`, czyli NASZ grounding GSA zaadresowany ramka regionu — to, czego ich
  rownania nie maja.
- `scripts/_compose_scenes.py` — sterownik. `--dry_run 1` buduje prompty, manifesty i
  `layout.png` (nasz odpowiednik ich panelu „Region Boxes") **bez GPU i bez wag**; przebieg
  zapisuje obok obrazow `manifest.json` z kazda nasza decyzja (rozdzielczosc, sampler, kroki,
  ziarno, negatyw), bo praca zadnej z nich nie podaje.
- **Scena 3.5 rozstrzygnieta OBRAZEM, nie domyslem (14.09).** RTP i etykieta ramki mowia
  „V9 dog", a V9 to u nich kot. W ICH WLASNYM panelu „Ours" w prawej ramce jest **pies**:
  gladkowlosy corgi w skafandrze, zgodny z miniatura Task 7 (V7) z tej samej figury i wyraznie
  inny od duzego, puchatego V1 w srodkowej ramce; szary kot rosyjski z miniatury Task 9 (V9)
  nie wystepuje w kadrze w ogole. Czyli „V9" to blad etykiety zamiast **V7** — i w RTP,
  i przy ramce. Notatki mialy to jako `[INFERENCE]`; teraz jest `[FIGURE]`. Domyslny odczyt
  `v7`, `--scene35 v9|literal` odtwarza figure jak wydrukowana. Rys. 3 NIE MA w repo CIDM
  (jest tylko Rys. 12) — sprawdzone na renderze HTML arXiv, kopia w `data/cidm_figs/`
  (poza gitem; arXiv nie daje praw redystrybucji, do publikacji tylko kopia Apache-2.0 z repo).
- Pary tej samej klasy (dwa psy, dwa koty) w 5 z 11 scen **nie wymagaja identyfikatorow**:
  przy rowaniu 4-5 kazdy region ma wlasny prompt i wlasny klucz adaptera, wiec brak
  `<V1>` na SDXL nie blokuje ani jednej sceny. Wczesniejsze „`F_base` nie potrafi pary
  same-class" dotyczylo JEDNEGO przebiegu ze wspolnym promptem.

**JEDNO PRZEJSCIE vs U+1 — to jest osia calej sprawy (14.09).** Pierwsza wersja potoku szla
wylacznie `compose_sample_regions`, czyli ICH rownaniem 4-5: `2+U` wywolan UNeta na krok
(dla naszych scen 4-6). To jest ich model kosztu. Nasza teza to `O(1)` — jeden forward
hipersieci, potem czysty UNet — wiec figura zrobiona tym torem pokazuje **nasze adaptery
w ich schemacie probkowania** i sama z siebie kasuje argument kosztowy. Dlatego doszedl
`--mode single`:

| tryb | wywolan UNeta / krok | co pokazuje |
|---|---|---|
| `single` (`compose_sample_single`) | **2**, niezaleznie od U | teze `O(1)` — wiersz do pracy |
| `unp1` (`compose_sample_regions`) | 2+U | ich schemat, gorna granica i ablacja |

**Protokol wejscia jest w obu ten sam i dokladnie ich**: ITP jako prompt globalny, kazdy RTP
zakodowany OSOBNO plus ramka. Osobne kodowanie nie jest wygoda, tylko koniecznoscia — CLIP
jest przyczynowy, wiec w jednym scalonym prompcie drugi span niesie kontekst pierwszego
(audyt 2885915: drugi span rownoodlegly od obu wzorcow, 0.778 vs 0.779). Sekwencje tekstowe
nie maja w UNecie ograniczenia dlugosci, wiec U+1 blokow kontekstu miesci sie w jednym
przebiegu — rozne sa tylko K/V w 16 (SD-1.5) / 70 (SDXL) warstwach attn2, a nie caly UNet.

Tor jednoprzebiegowy to rozszerzony `RegionKVAttnProcessor`. Trzy zmiany:
- **wstrzyk GSA per region** (`_gsa`, adresowany ramka regionu). Sierpniowy werdykt
  („regionalna uwaga trasuje tresc, ale nie wymusza liczby podmiotow", region_rewrite
  redukowal dwa podmioty do jednego) dotyczyl GOLEGO maskowania uwagi. Ta sciezka byla
  testowana poprawnie w tym sensie, ze `_compose_unp1.py --kv 1` odinstalowuje procesor wokol
  przebiegu uncond, wiec blad z `RegionalAttnProcessor` jej nie dotyczyl — ale mierzyla tor
  z POLOWA adaptera (patrz nizej), wiec jako dowod przeciwko jednemu przejsciu jest slabsza,
  niz wygladala. Do tego galaz
  groundingu powstala PO tamtym werdykcie (GO 20.08 wobec kompozycji zaparkowanej 09.08)
  i jest UCZONA pchac mase konceptu do ramki, czyli robi to, czego samo maskowanie nie umie.
  To jest strzal, ktorego nigdy nie oddalismy.
- **bramka `lora_enabled`**, taka sama jak w poprawce wyzej: przebieg uncond widzi czysty
  negatyw w calym kadrze. Dotad wolajacy musial odinstalowywac procesor recznie wokol
  kazdego uncond — latwo zapomniec, skutek cichy.
- **`manager.snapshot_lora`/`restore_lora`**: LoRA jest niezalezna od kroku, wiec liczymy ja
  raz na koncept i podmieniamy wskaznik cache'a. Bez tego procesor przeliczal cala hipersiec
  w KAZDEJ warstwie attn2, dla kazdego regionu i kazdego kroku — na SDXL 70 x U x 50 pelnych
  forwardow hipersieci 87.5 M na jeden obraz.

**POLOWA ADAPTERA BYLA NIEUZYWANA — naprawione 14.09.** `RegionKVAttnProcessor` liczyl `q`
raz globalnie i `to_out` raz na juz scalonym wyjsciu, oba pod `no_lora()`. Powody byly rozne
i tylko jeden byl decyzja: `q` globalne to wiernosc wobec `region_rewrite` z Mix-of-Show
(ktory podmienia wylacznie K/V), a `to_out` pod `no_lora()` to **zabezpieczenie przed bledem** —
wykonywalo sie PO wklejeniu wszystkich regionow, wiec aktywny bylby adapter OSTATNIEGO z nich
i rozsmarowalby delte jednego konceptu po calym kadrze. Skutek laczny: w torze
jednoprzebiegowym dzialala wylacznie TEKSTOWA polowa naszej LoRA (`to_k`/`to_v`), podczas gdy
hipersiec generuje delty na wszystkich czterech projekcjach.

Teraz kazda galaz (tlo + kazdy region) przechodzi `to_q`, `to_k`, `to_v`, uwage, wstrzyk GSA
i `to_out` **pod swoim adapterem**, a scalanie jest PO `to_out`. Dla tla nic to nie zmienia,
bo `to_out` jest afiniczne (scalenie przed i po jest przy tych samych wagach tozsame), a
regionom daje ich wlasna projekcje wyjsciowa. Koszt: U+1 mnozen d x d na warstwe, czyli nic
wobec uwagi, ktora i tak liczy sie U+1 razy. Kontrola 9 w `_verify_regional.py` sprawdza to
wprost: loguje, ktory adapter byl aktywny przy kazdej z czterech projekcji.

**MASKA `attn1` ODCINALA PODMIOTY OD TLA — znalezione 14.09 po obejrzeniu kolorow.**
Pierwszy punkt z miekka separacja dal dwa rozdzielone podmioty (czego nie dala zadna inna
proba), ale obraz byl przesycony, pasiasty i wygladal jak wycinanka. To nie byla kalibracja,
tylko semantyka maski. W `RegionalSelfAttnProcessor._bias` jest
`free = 1 - max(covered_i, covered_j)`, czyli para jest wolna tylko wtedy, gdy OBA konce sa
tlem — mimo komentarza „wolne tlo laczy wszystko". Para podmiot<->tlo byla wiec karana:

| scena | pokrycie ramkami | zakazanych par uwagi | po poprawce (`min` zamiast `max`) |
|---|---|---|---|
| 3.1 (2 koncepty) | 45% | **59.8%** | 10.2% |
| 12.4 (4 koncepty) | 49% | **67.8%** | 17.8% |

Podmiot nie widzial sceny ani scena jego, wiec kazda strefa dorabiala sobie wlasna palete
i oswietlenie. Stad tez skalowanie szkody z liczba konceptow: wiecej ramek to wieksze
pokrycie, czyli wiekszy odsetek ciętych par. Przy efektywnej karze 2 scena o dwoch konceptach
byla brzydka, a o czterech rozpadala sie calkiem.

**To NIE jest zwykly bug — ta semantyka jest poprawna dla JEDNEJ ramki**, bo tam ciecie
podmiot<->tlo to wlasnie zawieranie obiektu w ramce (tak uzywa jej `_ground_iou.py`).
Nigdy nie zostala zaadaptowana do kompozycji. Poprawka jest wiec trybem (`bg_shared`),
domyslnie wlaczonym w torze kompozycji i wylaczonym wszedzie indziej. Kontrola 10
w `_verify_regional.py` liczy odsetek ciętych par w obu trybach.

**Konsekwencja dla sierpniowego werdyktu** „izolacja attn1 niszczy generacje, samo-uwaga jest
tym, co skleja obraz": byl mierzony ta sama maska, czyli z podmiotami odcietymi od sceny.
Do tego szedl ze `strength=None`, wiec byl to dodatkowo przypadek TWARDY. Ten werdykt nie
mowi zatem nic o miekkiej separacji region-region, ktora jest jedyna wersja, jakiej chcemy.

**`leak` i `strength` to jedno pokretlo.** Punkty `s2_l0` i `s4_l0.5` wyszly BAJTOWO
identyczne, bo kara to `(1-allow)*(1-leak)*strength` — oba parametry wystepuja wylacznie
jako iloczyn. Docstring obiecuje, ze `leak` zachowuje globalna spojnosc; zeby to robil,
musialby byc PODLOGA na uwage miedzyregionowa, a nie przeskalowaniem kary. W siatkach
zmiatac efektywna sile, nie oba parametry osobno.

**Potok jest niezalezny od checkpointu** — zalezne sa tylko same obrazy. Czego brakuje:
sklejka rysunku, ktora wymaga decyzji, ktore sceny i skad panele CIDM
(patrz `assets/composition/README.md`).

**`X_sdxl_800aug_fix` JEST WYRAZNIE GORSZY NIZ `X_sdxl_ground_800aug` — do wyjasnienia.**
Wypadlo przy wyborze checkpointu do kompozycji. Ten sam przepis (`diff` configow to tylko
`output_dir` i `wandb.name`), retrenowany 09.09 na kodzie po trzech poprawkach SDXL:

| ckpt | s=0.4 | s=0.5 | s=0.6 |
|---|---|---|---|
| `X_sdxl_ground_800aug` (stary kod) | 0.8015 / 0.7927 / 0.5967 | 0.7735 / 0.8092 / 0.6208 | 0.7516 / 0.8209 / 0.6398 |
| `X_sdxl_800aug_fix` (nowy kod) | 0.7692 / 0.7505 / 0.5243 | 0.7509 / 0.7678 / 0.5628 | 0.7370 / 0.7831 / 0.5912 |

(TA / IA / DINO, `eval10f`, `average_final`.) Cala krzywa lezy nizej: przy zrownanym TA to
okolo **−6 IA i −7 DINO**, czyli 5x ponad zmierzony rozrzut miedzyseedowy SDXL (1.20 IA).
Poprawka (1) (pooled uncond) jest inference-only i zmierzona jako NULL, wiec zostaja (2) logity
fp32 w groundingu i (3) mikro-warunkowanie z prawdziwego rozmiaru zrodla — albo cos poza ta
trojka. **Nie scigam tego teraz**, ale kazde zdanie o SDXL na naprawionym kodzie stoi na tej
liczbie, wiec to jest do rozbrojenia przed pisaniem. Do kompozycji biore `ground_800aug`
(najlepszy punkt SDXL wobec CIDM, ten z wiersza „Stan wobec CIDM") przy **s=0.4**, czyli
zrownanym TA wobec ich 80.0.

**Przeciek do galezi bezwarunkowej w `RegionalAttnProcessor` — NAPRAWIONE (2026-09-14).**
Kara ukladu miala dzialac tylko w przebiegu warunkowym, a podzial `full[B//2:]` zakladal
sklejony batch `[uncond, cond]`. Nasze samplery licza cond i uncond **oddzielnymi** wywolaniami
UNetu z B=1, wiec `full[0:] = bias` — kara ladowala takze w predykcji bezwarunkowej, a CFG
mnozy jej blad przez (1 − guidance). Po tej samej sciezce szla akumulacja map uwagi. Teraz
bramka to `manager.lora_enabled` (przebieg uncond zawsze idzie pod `no_lora()`), a manager
jest przekazywany we wszystkich miejscach instalacji. Audyt na CPU bez SD:
`scripts/_verify_regional.py` (5 kontroli, przechodzi). **Zaden raportowany wynik nie jest
tym dotkniety** — `_ground_iou.py --layout regional` nigdy nie dal liczby do REPORT-u,
a `_verify_unp1.py` instaluje procesor dopiero PO przebiegu uncond. Odblokowuje to wiersz
„layout bez treningu" do Tabeli 2.

**Uwaga o headlinie.** Sweep chodzi po T=50, a headline to `P_paper` przy T=10. Parametry
architektury i treningu stosuja sie do obu, wiec zwyciezca **moze wymusic przetrenowanie
`P_paper`** (~6 h) i regeneracje kompozycji. Receptura wygrywajaca przy T=50 nie musi wygrywac
przy T=10 — przy dziesieciu konceptach nie ma presji na pojemnosc. Zwyciezce weryfikowac osobno
przy T=10, zanim cokolwiek ruszy w headline. To samo dotyczy rysunku skalowania
(`scripts/_fig_scaling.py`): skrypt jest niezalezny od receptury, ale pelna krzywa dla nowej
wymaga trzech dodatkowych zadan (punkty T=20/30/40, ~3.5 h GPU) — dla wariantow liczymy tylko
konce.

**Do zrobienia przed wysylka (sciezka krytyczna to pisanie, nie kolejka):**
- Kompozycja wielokonceptowa: smoke na GPU, potem obrazy na zwycieskiej recepturze (wyzej).
- Dwa wiersze porownawcze do Tabeli 2 (podloga „sam prompt", layout bez treningu) — poprawka
  galezi uncond w `RegionalAttnProcessor` zrobiona 14.09, wiersz jest odblokowany.
- `figures/tradeoff.pdf` przerobiony na liczby z `P_paper`.
- Drugie ziarno `P_best` + pelna macierz forgettingu (tabela forgettingu w pracy jest wciaz
  z `erode`).
- Trzecie ziarno SDXL wybranego przepisu — bez tego zadne zdanie o SDXL nie ma wagi.
- Abstract, Introduction, Related work, Limitations, Conclusion — puste `\todo{}`.
- Transfer w przod — zmierzony, **ani zdania w tekscie**.

**Nastepna wersja (nie ta):** osobny benchmark skalowania na pelnym CustomConcept101 do T=90
(92 koncepty po odrzuceniu `scene_*` i duplikatow) — jednorodny strumien usuwa czlon skladu
zbioru z 5b.1, ale **wymaga wiekszego `key_dim`**, bo przy T=90 zostaje 55% normy klucza, a przy
128 dokladnie zero. CC101 nie ma ani jednego konceptu stylu, wiec taki benchmark bylby wylacznie
obiektowy. Dalej: bramka `g + h(ramka)`; skalowanie galezi przez `s_lora`; Option C dwuenkoderowe
na SDXL; letterbox z bboxem przez korelacje wzorca; L2DM na SDXL (OOM); `R_tail` trzecie ziarno
na SD-1.5.

**`task_cond.scale_cond` — ODRZUCONE, i to dwa razy kosztownie (2026-09-15).** Os sweepu
`scale_cond` (skala LoRA jako WEJSCIE glowicy zamiast mnoznika) rozdziela punkty bez jednego
wyjatku: `false` daje DINO@T50 w zakresie 0.487-0.588, `true` w 0.254-0.451. Zakresy sie nie
nakladaja. Efekt glowny to **-0.096 DINO (8 SE)** i **-0.050 TA (8 SE)** — szkodzi na obu osiach
naraz, przy zadnej wartosci nie wygrywa.

Do tego ta os **psula sam odczyt sweepu**: warunkujac hipersiec na wartosci `s`, model sam
produkuje adapter „wlasciwy dla tego s", wiec kręcenie `lora_scale` przestaje przesuwac punkt
pracy i **TA robi sie plaskie w trzeciej cyfrze**. Wszystkie punkty ekstrapolujace przy odczycie
matched-TA (wspolczynnik do +2250, p016 dostal tak `dino_t50` = 0.808, czyli wiecej niz
cokolwiek zmierzonego w projekcie) maja `scale_cond: true`; wszystkie interpolujace maja
`false`. Awaria odczytu byla wiec OBJAWEM tej osi, nie wada celu.

**Koszt: polowa projektu sweepu.** 24 z 48 punktow LHS mialo te flage. Anulowane 15.09:
13 elementow `sw800` (10 czekajacych, 3 w polowie) i 6 elementow `sw400` w 75-83% ukonczenia —
razem okolo **119 GPU-godzin**. Zostaly 24 punkty, czyli komplet `scale_cond: false`.
Paradoksalnie projekt na tym zyskuje: `scale_cond` dawalo najwieksza czesc wariancji reszt
(sigma 0.039 przy poolingu), wiec przy N=24 i siedmiu osiach SE powinno zejsc do ~0.008,
ponizej pierwotnie zakladanych 0.010.

**Lekcja procesowa, wazniejsza od samej osi.** Ta flaga miala configi (`phaseS/S_kappa*`,
`phaseT/T_time_scale`, `_scale_smoke`, wszystkie z 31.08) i smoke na klastrze, ale **ani jednego
zdania w REPORT ani w STATUS** — sprawdzone, slowo `scale_cond` nie wystepuje w zadnym `.md`
tego repo. Nie bylo tez na liscie „nie robic". Jesli werdykt kiedykolwiek zapadl, zginal razem
z sesja. **Kazdy odrzucony wariant ma trafiac na te liste w tym samym commicie, w ktorym
zapada werdykt** — inaczej wraca, i tym razem wrocil jako polowa osi sweepu.

**Nie robic (zmierzone albo rozstrzygniete):** **`task_cond.scale_cond` (wyzej)**;
`share_heads` pod groundingiem (−1.6…−1.9 IA,
−16 pp placementu); kotwica na galaz (zamraza transfer w przod);
`prompt_aug` (−1.40 IA na SDXL); `paste_scale_full_p` (−2.4 IA); `boxonly` (LoRA nie przejmuje
tozsamosci: 53.6/55.0 wobec 62.3); 1600 krokow (nasyca sie); tokeny groundingu bez klucza
(= GLIGEN bez semantyki, claim (ii) upada); rozdzielanie tozsamosci od umiejscowienia przez
wejscie galezi (wyciek bierze sie z sygnalu treningowego, nie z wejscia); podnoszenie rangi /
szerokosci glowicy per warstwa; gonienie wypelnienia kadru pod metryke; **kotwice era i ema przy
kazdym beta** (5b.3); **poszerzanie `head_hidden`** (5b.3, gorsze takze przy T=10);
**`basis_q` ponizej zmierzonego `basis_L99`** (5b.4).

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
- **Athena ma własny venv** `$SCRATCH/venvs/continualhyper-athena` (5.5 GB, torch 2.7.1,
  diffusers 0.30.0; sprawdzone 2026-09-07), `slurm/clusters/athena.sh` na niego wskazuje i `env.sh`
  eksportuje `VENV`, który wygrywa z `${VENV:-../unlearning/UnHype/.venv}` w runnerach. Wcześniejsza
  „bomba zegarowa" (venv UnHype w grancie `plggrecontext` wygasającym 2026-09-08) jest więc
  rozbrojona. Uwaga na cudze pakiety w ~/.local dla pythona 3.9 — mylą `pip list`.
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


**Lekcje z fazy T (2026-09-10…14) — wszystkie tego samego rodzaju: działanie na wnioskowaniu tam,
gdzie weryfikacja kosztuje jedną komendę.**

- **`SKIP_PULL=1` wstawione dla ciszy w logach nie dostarczyło configów** — cztery zadania padły po
  10 sekundach z `FileNotFoundError`. Ten flag wyłącza jedyny mechanizm transportu kodu.
- **Plik z punktami sweepu wygenerowany na Windows miał CRLF**, a pętla basha doklejała znak
  powrotu karetki do **wartości ostatniego parametru w linii**. Komunikat (`'false:bool<CR>'`)
  w żaden sposób nie wskazywał na końce linii. Ostrzeżenie z `CLAUDE.md` dotyczy `.sh`/`.py`,
  ale to samo dotyczy **każdego pliku czytanego przez shell**.
- **Poprawka na CRLF sama wstawiła literalny CR do źródła** — patch pisany przez heredok
  zamienił escape na prawdziwy znak. Dwa razy, w dwóch plikach. Po każdej łatce: `py_compile`
  **i** `grep -c $'\r'`, zanim cokolwiek pójdzie na klaster.
- **Runner nigdy nie przeszedł end-to-end przed postawieniem za nim 47 zadań.** `_mkcfg.py` brał
  nazwę punktu z `basename(--out)`, a runner zapisuje config jako `<katalog>/config.yaml` — więc
  **każdy punkt pisałby do `outputs/sweep/config`**. Wykryte po 3 h 20 GPU, na pierwszej metryce.
- **„Test dymny" trwający tyle co zadanie produkcyjne nie jest testem.** Stąd `SMOKE=1` w
  `sbatch_sweep.sh`: 5 kroków na zadanie, jedna skala, jeden obraz na prompt.
- **`grep -c` zwraca kod 1, gdy nie znajdzie dopasowań** — czyli łańcuch
  `grep -c $'\r' plik && git add && git commit` **cicho pomija commit**, gdy plik jest czysty.
  Zadanie poszło wtedy na config, którego nie było w repo.
- **Zależność `afterok` uratowała 47 punktów.** Trzy razy z rzędu smoke padał i tablice nie
  ruszały. To jedyny mechanizm, który w tej serii zadziałał tak, jak zaplanowano.
- **Utrata danych: 12 120 obrazów z `phaseP/P_best/eval10f`.** Skrypt pakujący skończył archiwum
  i skasował oryginały, po czym zerwało się ssh **zanim linia „OK" doszła**. Uznałem tar za obcięty
  i poleciłem go skasować. Był kompletny — 571 351 040 B / 12 120 plików = 47 141 B na plik, przy
  typowym `.jpg` 46 212 B. Metryki i checkpointy ocalały, obrazy nie. **Nie wnioskować o stanie
  artefaktu z tego, czego nie ma w logu — sprawdzić artefakt** (`tar -tf | wc -l`). Skrypt ma teraz
  budować do `.tar.part` i nadawać właściwą nazwę dopiero po weryfikacji liczby wpisów.
- **Limity `--time` z pomiaru, nie z oszacowania.** Dwa razy pod rząd smoke padł na limicie, bo
  przy 5 krokach na zadanie dominuje narzut (50 zapisów checkpointu po 85 MB i próbkowanie do
  `fresh/`), a nie sam trening.
- **Inody, nie miejsce, są ograniczeniem `$SCRATCH`.** Przy 4.4% zajętości dysku było 97.8%
  z miliona inodów. `scripts/_pack_outputs.sh` zwinął 392 000 plików z `phaseP` i `sdxl` do
  archiwów (obrazy zostają, metryki i checkpointy poza tarem). `sbatch_sweep.sh` kasuje obrazy
  **od razu po metrykach każdej skali** — punkt trzyma wtedy 2 tys. plików zamiast 14 tys., co
  pozwala na 24 równoległe punkty zamiast 8.

## 8. Konwencja aktualizacji

Po każdym domkniętym wątku: zaktualizować sekcję wyników lub sagę ramek (tabela wariantów),
przenieść pozycje z "W toku" do właściwych sekcji, datę w nagłówku. Szczegóły i pełne liczby
zawsze w `assets/STATUS.md`; tu tylko synteza, którą da się przeczytać w 5 minut.

### Dalsze zadania na p022 (2026-09-16, popołudnie, c.d.)

Commity 33eb1d4 + 9870ef6: `task_cond.gram_schmidt` (domyślnie true; `false` = klucze bez rzutowania
na dopełnienie), `scripts/_embed_probe.py` (interpolacja między embeddingami, wektory spoza bazy,
skalowanie normy; gałąź umiejscowienia wyłączona, prompt „a photo"). Pierwszy commit przekodował
komentarze `manager.py` (plik ma jeden bajt cp1250, 0xb3 w „Gałki"), drugi przywrócił bajty.

| job | co | wyjście |
|---|---|---|
| 22484020 `p022-nogs` | ablacja: p022 bez Grama–Schmidta (losowe, nieortogonalne klucze) | `outputs/sweep/p022_nogs` |
| 22484021 `p022-embed` | sonda przestrzeni embeddingów (pary 0:2, 0:6, 2:8, 1:4; normy 0–2; 4 wektory spoza bazy) | `outputs/sweep/p022/embed_probe` |
| 22483851 → 22484074 `p022-fwd00` | umiejscowienie po zadaniu 0; pierwsze zgłoszenie padło (`canonical conditioning for task 1 not set` — po zadaniu 0 istnieje tylko embedding psa), powtórka z `--only_concepts cifc_dog` | `outputs/teaser/p022_fwd_t00` |
| 22483854 `p022-fwd04` | j.w. po zadaniu 4; **padnie na dog2** (indeks 5, jeszcze nienauczony) po policzeniu pięciu pierwszych konceptów — wiersze per koncept zostają w logu, podsumowanie RAZEM trzeba złożyć ręcznie | `outputs/teaser/p022_fwd_t04` |
| 22483855 `p022-fwd49` | j.w. po zadaniu 49, wszystkie 7 | `outputs/teaser/p022_fwd_t49` |

**tab:grounding, wiersz „op" na `hyper_after_task09.pt` (22483502, 7 min, 50 kroków, K=15):**
ćwiartki 84/84 = 100 %, IoU 0.766, IoU>0.5 96 %, zawarcie 0.93, wypełnienie 1.01, TA 0.828,
DINO 0.655, DINO-wycinek 0.720, DINO-maska 0.717. Wiersz „main" (22483500) i „no box" w toku.

### Umiejscowienie wzdłuż strumienia, p022 (2026-09-16; 22484074 / 22483854 / 22483502 / 22483855)

Protokół jak wiersz „op" tab:grounding: grid 2:0.3, bootstrap 15, tło 10 kroków, scale 0.7, 50 kroków,
n=3 (12 generacji na koncept), jeden seed treningowy. Checkpointy `hyper_after_task{00,04,09,49}.pt`.
Po zadaniu 4 skrypt padł na dog2 (nienauczony), więc średnia z pięciu pierwszych konceptów policzona
ręcznie z wierszy per koncept; brak `RAZEM` dla tego punktu.

IoU per koncept:

| koncept | po t=0 | po t=4 | po t=9 | po t=49 |
|---|---|---|---|---|
| dog | 0.731 | 0.828 | 0.850 | 0.786 |
| duck_toy | – | 0.827 | 0.780 | 0.473 |
| cat | – | 0.579 | 0.749 | 0.592 |
| backpack | – | 0.553 | 0.702 | 0.478 |
| teddybear | – | 0.766 | 0.789 | 0.736 |
| dog2 | – | – | 0.788 | 0.492 |
| cat2 | – | – | 0.702 | 0.537 |
| **średnia (5 pierwszych)** | – | 0.711 | 0.774 | 0.613 |
| **RAZEM (7)** | – | – | **0.766** / IoU>0.5 96 % / DINO 0.655 / TA 0.828 | **0.586** / IoU>0.5 70 % / DINO 0.618 / TA 0.833 |

Odczyt: do t=9 umiejscowienie starych konceptów **nie pogarsza się, a raczej poprawia** (pies 0.73 →
0.85; cat/backpack z 0.55–0.58 do 0.70–0.75) — moduł umiejscowienia uczy się dalej na kolejnych
zadaniach i to przenosi się wstecz. Między t=9 a t=49 **spadek**: IoU 0.766 → 0.586, IoU>0.5 96 % → 70 %,
wypełnienie 1.01 → 0.83 (obiekty za małe względem ramki), ćwiartka nadal 100 %. Zgodność treści spada
umiarkowanie (DINO 0.655 → 0.618). Czyli przy T=50 gorzej cierpi *gdzie/jak duży* niż *co*; pod §5.6
trzeba to napisać wprost. Jeden seed, 12 generacji na koncept — różnice ~0.05 IoU per koncept są w szumie,
spadek średniej 0.18 nie.

Sonda embeddingów 22484021 padła (Half vs Float: wstrzykiwany klucz rzutowany do fp16 `pooled`, a głowice
są fp32); commit efa9890, powtórka zgłoszona.

### tab:grounding na p022, `hyper_after_task09.pt` (2026-09-16; 22483500 main, 22483502 op; 50 kroków, K=15, scale 0.7, „on a beach", 84 gen./wiersz)

| wiersz | ćwiartka | IoU | IoU>0.5 | zawarcie | wypełnienie | TA | DINO cały | DINO wycinek | DINO maska | kolor dRGB | det |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bez ramki (κ=0) | 25 % | 0.134 | 0 % | 0.25 | 1.07 | 0.778 | 0.524 | 0.606 | 0.599 | 0.215 | 72/84 |
| κ=1 | 93 % | 0.551 | 52 % | 0.74 | 1.34 | 0.780 | 0.685 | 0.721 | 0.713 | 0.119 | 84/84 |
| κ=2, sched 0.3 | 100 % | 0.761 | 96 % | 0.93 | 1.07 | 0.795 | 0.662 | 0.730 | 0.724 | 0.112 | 82/84 |
| κ=2 + bootstrap 15 (op) | 100 % | 0.766 | 96 % | 0.93 | 1.01 | 0.828 | 0.655 | 0.720 | 0.717 | 0.122 | 84/84 |

Różnice wobec P_paper (stara tabela): bez ramki p022 jest słabszy (DINO maska 0.599 vs 0.627, det 72/84
vs 84/84 — 12 generacji bez wykrywalnego obiektu), za to κ=2 sam już domyka umiejscowienie (96 %,
zawarcie 0.93, wypełnienie 1.07; u P_paper 85 %, 0.81, 1.39). Bootstrap na p022 nie zmienia już IoU ani
zawarcia, daje +0.033 TA (0.795 → 0.828, powyżej wiersza bez ramki 0.778), wypełnienie 1.07 → 1.01 i
det 82 → 84. Koszt bootstrapu na tożsamości mały i jednakowy na trzech osiach (−0.007 cały, −0.010
wycinek, −0.007 maska) — argument „koszt znika na masce" z P_paper na p022 nie zachodzi, usunięty z tekstu.
Tożsamość: maska 0.599 → 0.724 (κ=2) / 0.717 (op), kolor 0.215 → 0.112 / 0.122, cały obraz 0.524 → 0.655.
Wpisane do main.tex (tab:grounding, §5.5, abstrakt, wstęp).

### Sonda embeddingów: pierwsza wersja bezużyteczna (2026-09-16, 22484366 → 22486238)

Arkusze z `embed_probe/` (prompt „a photo", bez maski tokenów) dają tekstury i ulice nawet dla
czystego embeddingu psa (alfa 0, norma 1). Powód: aktualizacja to_k/to_v jest uczona wyłącznie na
pozycji słowa klasy; przyłożona do wszystkich tokenów jest poza rozkładem. Wynik nie mówi nic
o przestrzeni embeddingów, tylko o masce. Poprawka c244f78: prompt „a photo of <klasa>" z maską
na słowie klasy jak w treningu; dla pary o różnych klasach (pies → kot) po jednym wierszu na słowo
klasy. Powtórka 22486238 → `outputs/sweep/p022/embed_probe2`. Stary katalog `embed_probe` do
skasowania po obejrzeniu.

### „Puść wszystko, czego nie mamy" (2026-09-16, ~14:30)

Wszystko na przepisie p022; zależności `afterok` tam, gdzie wejście jeszcze się liczy.

| job | co | zależy od | wyjście |
|---|---|---|---|
| 22487416 / 22487421 | tab:grounding, dwa wiersze porównawcze na `hyper_after_task09.pt`: `--layout prompt` (pozycja słowami, bez ramki) i `--layout regional` (uwaga regionalna na tych samych adapterach); bootstrap 0, reszta jak wiersz „op" | – | `outputs/teaser/p022_tab_{prompt,regional}` |
| 22487427, 22487443 | macierze 55 komórek dla seedów 2025 / 2026 (skale 0.5, 0.45) | 22479373 / 22479374 | `outputs/sweep/p022_s{2025,2026}/matrix` |
| 22487429–22487435, 22487453–22487462 | umiejscowienie wzdłuż strumienia dla obu seedów: po zadaniu 1 (pies), 5 (pięć pierwszych), 10 i 50 (siedem) | j.w. | `outputs/teaser/p022_s{S}_fwd_t{00,04,09,49}` |
| 22487470 `p022_sem128` | kontrola „co niesie embedding": połowa klucza (128 wym.) = zamrożony CLIP-image embedding konceptu, połowa losowa ortogonalna (`task_cond.sem_dim=128`) | – | `outputs/sweep/p022_sem128` |
| 22487472 | ewaluacja SDXL p022 przy 0.4 / 0.5 / 0.6 (final only) | 22479391 | `outputs/sdxl/X_sdxl_p022/eval` |
| 22487476 | sonda umiejscowienia na SDXL p022 (1024², scale 0.5, reszta jak „op"); wcześniej nigdy nie doszła do końca | 22479391 | `outputs/teaser/sdxl_p022_tab_op` |
| 22487511/15, 22487518/20, 22487523/28, 22487532/36, 22487538/40 | **drugi seed (2025) baseline'ów**: finetune, C-LoRA, EWC, LwF (`train_baselines`), L2DM (`train_l2dm`); każdy trening + macierz na skali 1.0. Stare wyjścia baseline'ów zostały wyczyszczone ze scratcha, więc to trening od zera. CIDM (osobny pipeline, 2 GPU) pominięty. | trening → macierz | `outputs/baseline/<m>_s2025/{ckpts,matrix}` |
| 22487541/50, 22487551/52 | **oracle**: `lora_solo` (10 niezależnych LoRA, rank 4, 800 kroków, wybór po indeksie zadania), seedy 2024 i 2025; ewaluacja tylko modelu końcowego przy 0.45/0.6/0.8/1.0 (`FINAL_ONLY=1`) | trening → eval | `outputs/baseline/lora_solo_s{2024,2025}/final` |

Niepuszczone, z powodem: ablacje §5.7 (sześć wariantów wymagałoby sześciu treningów na p022; zostają na
starych checkpointach jako tabela w dodatku), sweep szerokości i wykres pamięci vs T (poza planem),
CIDM drugi seed (zewnętrzny kod). Sufit spójności referencji dla p022 to analiza z wyjść macierzy,
nie job GPU.

### Dlaczego umiejscowienie spada po zadaniu 50 — hipoteza i diagnostyka (2026-09-16, ~15:00)

Fakt z configu p022: `reg.space=dw`, bez `reg.ground_anchor`. Regularyzator wyjściowy kotwiczy tylko ΔW;
głowica tokenów, FiLM i bramki g^(ℓ) uczą się bez ochrony przez 40 kolejnych zadań (plus wd 1e-2).
Objawy pasują do dryfu *treści* tokenów, nie do osłabienia gałęzi: ćwiartka 100 %, zawarcie 0.89,
wypełnienie 0.83 (obiekt w ramce, ale za mały); słabsza gałąź daje odwrotność (κ=1: wypełnienie 1.34,
zawarcie 0.74). Zmiana rozkładu danych (CC101: małe obiekty, osoby) mogła przestroić mapowanie
„ramka → rozmiar" dla wszystkich embeddingów naraz.

| job | co | wyjście |
|---|---|---|
| 22488470 | `_ground_drift.py` na `p022/ckpts`: dryf parametrów ground_head / ground_film / bramek między checkpointami (górne ograniczenie dryfu wyjścia) | log |
| 22488479 | sonda po zadaniu 50 przy κ = 1.5 i 3 (bootstrap 15) — czy wypełnienie reaguje na κ | `outputs/teaser/p022_fwd_t49_kappa` |
| 22488484 | sonda po zadaniu 50 przy κ = 2 **bez** bootstrapu | `outputs/teaser/p022_fwd_t49_noboot` |
| 22488488 → 22488491 / 22488496 | **p022_ganchor**: p022 + `reg.ground_anchor=true` (kotwica także na placement tokens); po treningu sondy po zadaniu 10 i 50 | `outputs/sweep/p022_ganchor`, `outputs/teaser/p022_ganchor_fwd_t{09,49}` |
| 22488500 → 22488503 / 22488514 | **p022_ganchor_g**: j.w. + `reg.ground_anchor_gates=true` (kotwica także na bramki) | `outputs/sweep/p022_ganchor_g`, `outputs/teaser/p022_ganchor_g_fwd_t{09,49}` |

Odczyt, gdy przyjdą: jeśli ganchor trzyma IoU po zadaniu 50 blisko 0.77 przy niezmienionym T=10 i
niezmienionym DINO/TA na krzywej, to zmiana headline'u na p022_ganchor jest uzasadniona i domyka historię:
ten sam regularyzator wyjściowy, rozszerzony na drugie wyjście hipersieci. Jeśli nie pomaga — dryf
jest w danych (rozkład rozmiarów CC101), nie w parametrach, i zostaje jako ograniczenie.

**Wynik dryfu (22488470, 13 s), 50 checkpointów p022, dryf względny pierwszy → ostatni (||Δ||/||param||):**

| grupa | tensorów | ||param|| | krok po kroku | pierwszy → ostatni |
|---|---|---|---|---|
| ground_head (tokeny) | 4 | 10.03 | 1.01 | **6.28** |
| ground_film | 2 | 1.16 | 0.74 | **7.55** |
| ground_gates (16 skalarów) | 16 | 0.034 | 48.8 | 567 |
| ground_gsa_mods (q/k/v/o gałęzi) | 64 | 72.1 | 0.37 | 1.82 |
| heads (LoRA) | 576 | 263.7 | 1.96 | 12.64 |

Bramki: pierwszy ckpt mean −0.002, absmax 0.015; ostatni mean −0.019, absmax **0.078** — efektywna siła
gałęzi (tanh g) urosła ok. **5×** przez strumień. Głowice LoRA dryfują względnie najbardziej (12.6), ale
ich *wyjścia* dla starych embeddingów są kotwiczone (swoboda cechowania); głowica tokenów i FiLM
(6–7.5) nie mają żadnej kotwicy. Odczyt: przy κ = 2 dobranym na T=10 model po zadaniu 50 działa jak
κ ≈ 10 — stąd zawężenie obiektu (wypełnienie 0.83) przy trafionej ćwiartce. Test wprost: 22488546
`p022_fwd_t49_lowk` (κ = 0.5 i 1 po zadaniu 50); jeśli wypełnienie i IoU wrócą do ~1.0 / ~0.75, to
jest to dryf siły bramek + tokenów, a `ground_anchor_gates` (22488500) powinien to usunąć u źródła.

### Wyniki diagnostyki po zadaniu 50 — hipoteza „za mocna gałąź" OBALONA (2026-09-16, ~15:40)

Sondy na `hyper_after_task49.pt`, 7 obiektów CIFC, protokół jak „op" (bootstrap 15, chyba że napisano):

| κ | bootstrap | ćwiartka | IoU | IoU>0.5 | zawarcie | wypełnienie | TA | DINO | det |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 15 | 79 % | 0.279 | 18 % | 0.61 | 0.63 | 0.827 | 0.523 | 80/84 |
| 1.0 | 15 | 99 % | 0.453 | 40 % | 0.82 | 0.77 | 0.841 | 0.575 | 84/84 |
| 1.5 | 15 | 100 % | 0.528 | 57 % | 0.83 | 0.84 | 0.837 | 0.607 | 84/84 |
| 2.0 | 15 (op) | 100 % | 0.586 | 70 % | 0.89 | 0.83 | 0.833 | 0.618 | 84/84 |
| 3.0 | 15 | 88 % | 0.599 | 78 % | 0.82 | 0.74 | 0.827 | 0.544 | 81/84 |
| 2.0 | **0** | 100 % | 0.602 | 67 % | 0.87 | **0.97** | 0.813 | **0.631** | 82/84 |

Niższa κ jest **gorsza** na każdej osi, więc wzrost bramek 5× nie oznacza przestrzelenia — gałąź po 50
zadaniach nie jest za mocna, tylko *inna*: przy każdej κ wypełnienie < 0.85 i IoU>0.5 ≤ 78 %, wobec 96 % po
zadaniu 10. Bez bootstrapu wypełnienie wraca do 0.97 i DINO rośnie (0.631), IoU>0.5 zostaje 67 % — czyli
po 50 zadaniach to bootstrap ściska obiekt, a niecelność ramki zostaje niezależnie od niego. Wniosek:
dryf *treści* tokenów/FiLM (6–7.5 normy, bez kotwicy), nie siły. Reguła normalizacji κ **odpada**.
Rozstrzygną warianty z kotwicą: `p022_ganchor` (tokeny) i `p022_ganchor_g` (tokeny + bramki), 22488488 /
22488500, z sondami po zadaniu 10 i 50.

### tab:grounding — dwa wiersze porównawcze (22487416 / 22487421, p022 po zadaniu 10, bez bootstrapu)

| wiersz | ćwiartka | IoU | IoU>0.5 | zawarcie | wypełnienie | TA | DINO maska | kolor | det |
|---|---|---|---|---|---|---|---|---|---|
| pozycja słowami w promptcie (κ=0) | 21 % | 0.135 | 0 % | 0.25 | 1.01 | 0.743 | 0.597 | 0.199 | 67/84 |
| regional attention, te same adaptery (κ=0) | 49 % | 0.164 | 2 % | 0.37 | 0.92 | 0.770 | 0.528 | 0.289 | 65/84 |

Prompt nic nie daje (jak bez ramki). Regional trafia ćwiartkę w połowie przypadków, ale gubi obiekt w ¼
generacji i psuje tożsamość (maska 0.528 vs 0.599 bez ramki, kolor 0.289). Wpisane do main.tex, todo
o wierszach usunięte.

### Sonda embeddingów v2 (22486238, `embed_probe2`, prompt „a photo of <klasa>" + maska) — działa

* **Interpolacja dog → dog2**: gładki morf korgi → drugi pies, każda komórka to sensowny pies; tożsamość
  zmienia się ciągle, nie skokowo.
* **dog → cat** (dwa wiersze: słowo „dog" / „cat"): kategorię wybiera słowo klasy, embedding steruje
  tożsamością w obrębie kategorii; punkty pośrednie to „generyczny pies/kot", nie mieszaniec.
* **Norma** (0, 0.5, 1, 1.5, 2 × embedding psa): 0 i 0.5 = generyczny pies (prior modelu), 1 = korgi,
  1.5 = korgi z artefaktami, 2 = szum. Długość działa jak siła adaptera do ok. 1, potem się rozpada.
* **Spoza bazy** (4 losowe wektory ⊥ do całej bazy, norma = mediana): za każdym razem generyczny,
  poprawny pies. Nieznany embedding = prior klasy, nie śmieci. Dobry materiał do dodatku (jeden arkusz).

### Długość tekstu (2026-09-16, ~16:30)

Pomiar bez czerwonych notatek (skrypt `tmp/method/nodraft.py`: draftnotesfalse + wycięty todolist; sama
zmiana przełącznika nie działa, bo `todolist` ma `\ifdraftnotes` rozpięte między begin/end i TeX
nie umie tego przeskoczyć). Przed cięciami: Conclusion kończył się na stronie 11. Po dwóch rundach:
Conclusion kończy się dokładnie na dole strony 9, przy pustych Related work i Limitations (razem ok.
0.6 strony do dopisania, więc kolejne ~0.6 strony cięć będzie potrzebne).

Co poszło: §5.7 (alternatywy modułu) → dodatek A.5 jako tabela left/right + skrócona proza; §5.9 z
trzech akapitów do dwóch (zdanie o protokole bez ramki → dodatek); Preliminaries skrócone o ~40 %;
Setup: szczegóły enkoderów tekstu → A.1, akapit o metrykach krótszy; §5.5: wstęp o detektorze i
akapity „Second/Third" skrócone; zdanie-przegląd na początku Results usunięte; intro: liczby pamięci
w jednym zdaniu; §4.2: GLIGEN w jednym zdaniu, dwa zdania uzasadnień usunięte; podpisy Tab. 1, Tab. 2,
Fig. 3 skrócone; Fig. 3 szerokość 0.62 → 0.50. Backupy `main.tex.bak-length-*`, `bak-length2-*`.
Bibliografia: poprawione też Hervé Jégou (Caron et al.).

### Krzywa p022 (22482151 `p022-topup`, 2 h 21 min; `curveA_t9`, `curveA_t49`, jeden seed, 10 próbek/koncept)

| s_lora | TA@10 | IA@10 | DINO@10 | TA@50 | IA@50 | DINO@50 |
|---|---|---|---|---|---|---|
| 0.30 | 77.24 | 77.44 | 0.578 | 78.71 | 73.59 | 0.522 |
| 0.45 | 75.27 | 79.46 | 0.621 | 78.07 | 75.19 | 0.557 |
| 0.60 | 73.66 | 80.97 | 0.648 | 76.98 | 76.56 | 0.581 |
| 0.75 | 71.92 | 82.17 | 0.661 | 75.66 | 77.78 | 0.598 |
| 0.90 | 70.54 | 82.95 | 0.664 | 74.14 | 78.52 | 0.606 |
| 1.05 | 68.96 | 83.17 | 0.655 | 72.84 | 78.34 | 0.595 |

Odczyt przy TA CIDM (74.8): interpolacja między 0.45 i 0.6 daje IA **79.9** wobec 78.0 CIDM → **+1.9 pkt**
(P_paper miało +2.4; `score.json` sweepu podaje 80.0 przy swoim celu TA). Krzywa saturuje przy 0.9
(DINO 0.664, IA 82.95), 1.05 już spada. Przy T=50 cała krzywa przesunięta: przy tym samym s TA wyżej
o ~3 pkt, IA niżej o ~4, DINO o ~0.06 — adaptery słabną, a nie „psują się". Wpisane: fig:tradeoff (krzywa
p022 0.3–0.9, jeden seed, bez słupków), tab:curve (sześć skal), „2.4 points" → „1.9 points" we wstępie,
§5.2, podpisie. Akapit o dwóch seedach (80.37 / 80.43) w A.4 dotyczy starego checkpointu — oznaczony
do przeliczenia po `p022_s2025/6`.

### Baseline'y na strumieniu T=50 (2026-09-16, ~18:00; commit 18c9695)

`scripts/_mkbl50.py` buduje `configs/bl50/bl50_<m>.yaml`: baza = `baseline_<m>.yaml` (metoda, budżet 800
kroków/zadanie, sekcja `baseline`), koncepty = 50 z `T50_mixed.yaml`, każdy z identyfikatorem `<Vk>`
i promptem „a photo of <Vk> <klasa>" (baseline'y uczą token per koncept, jak CIDM). Siła regularyzacji
z CLI `--lam`, więc jeden config na metodę. Uwaga od WG: CIDM też ma `<v1>` w promptcie, więc „handed the
task index" dotyczy wszystkich metod — podpis Fig. 3 poprawiony (CIDM nie „identyfikuje konceptu sam").

| metoda | λ | trening → eval (FINAL_ONLY, skale 0.6/0.8/1.0) | wyjście |
|---|---|---|---|
| finetune | 0 | 22491902 → 22491903 | `outputs/baseline50/finetune_l0` |
| EWC | 100 / 1000 / 10000 | 22491904→07, 22491910→15, 22491916→17 | `ewc_l{100,1000,10000}` |
| LwF | 0.3 / 1 / 3 | 22491924→32, 22491938→39, 22491949→51 | `lwf_l{0.3,1,3}` |
| C-LoRA | 0.3 / 1 / 3 | 22491952→54, 22491957→58, 22491966→67 | `clora_l{0.3,1,3}` |
| lora_solo (oracle, 50 LoRA) | 0 | 22491968 → 22491969 | `lora_solo_l0` |
| L2DM (α=β=γ=1) | – | 22491970 → 22491971 | `l2dm` |

Selekcja „najlepszy config per metoda": na dziesięciu konceptach benchmarku po 50 zadaniach, przy TA
dopasowanym do p022@50 (78.1 przy s=0.45), interpolując po trzech skalach; drugorzędnie DINO na
wszystkich 50. Bez pełnych macierzy 50×50 (1275 komórek/config — za drogo); zapominanie na T=50 tylko dla
p022 (macierz `p022/matrix` liczy się). Koszt: 12 treningów × ~3 h + 12 ewaluacji × ~2 h ≈ 60 GPU-h.

### Teaser na s=0.8 (2026-09-16, ~18:40; z 22483866 `p022_final/s08`)

Wybór ręczny z próbek 30–49 (te same prompty co przy s=0.6), wszystkie przy s=0.8 dla spójności
(s=1.0 dawało mocniejszą tożsamość tylko gitarze, kosztem artefaktów u innych):
dog 38, painting 47, actionfigure_1 33, instrument_1 40, plushie_2 33, actionfigure_2 34,
instrument_music1 40, plushie_bunny 40, actionfigure_3 40, person_3 37. Podpis Fig. 1 podaje s=0.8.
Arkusze wyboru: `tmp/teaser/sheets_hi/`. Wyraźnie lepsze niż s=0.6: pies, Meowth, króliczek, sukienka,
osoba; wzmacniacz Marshall i figurka 2 podobne.

Kompozycja: brak nowych wyjść od 15.09 22:10 (`outputs/compose_scenes/unp1_ground_k2`) i brak nowych
wpisów w REPORT — wątek prowadzony w drugiej sesji; §5.7 w tekście to nadal akapit wprowadzający + todo.

### Teaser na s=0.8 (2026-09-16, ~18:40; z 22483866 `p022_final/s08`)

Wybór ręczny z próbek 30–49 (te same prompty co przy s=0.6), wszystkie przy s=0.8 dla spójności
(s=1.0 dawało mocniejszą tożsamość tylko gitarze, kosztem artefaktów u innych):
dog 110 (po uwadze WG o kolorze; 38 było wyblakłe, 110 ma pomarańcz głowy jak referencja), painting 47, actionfigure_1 33, instrument_1 40, plushie_2 33, actionfigure_2 34,
instrument_music1 40, plushie_bunny 40, actionfigure_3 40, person_3 37. Podpis Fig. 1 podaje s=0.8.
Arkusze wyboru: `tmp/teaser/sheets_hi/`. Wyraźnie lepsze niż s=0.6: pies, Meowth, króliczek, sukienka,
osoba; wzmacniacz Marshall i figurka 2 podobne.

Kompozycja: brak nowych wyjść od 15.09 22:10 (`outputs/compose_scenes/unp1_ground_k2`) i brak nowych
wpisów w REPORT — wątek prowadzony w drugiej sesji; §5.7 w tekście to nadal akapit wprowadzający + todo.

### Żniwa 2026-09-16 wieczorem: macierz p022, oracle, baseline'y seed 2025, skala CIDM

**1. Macierz zapominania p022 (22479378) — jest.** Pełne 55 komórek, jeden seed:

| skala | zapominanie (DINO) | TA | IA | DINO (model końcowy) |
|---|---|---|---|---|
| 0.45 | **0.0094** | 75.27 | 79.46 | 0.621 |
| 0.50 | 0.0108 | 74.73 | 79.96 | 0.631 |

Zgodne z poprzednim checkpointem (0.0068 ± 0.0078 na trzech ziarnach) i z regułą „rośnie ze skalą".

**2. Oracle: dziesięć niezależnych LoRA (`lora_solo`, rank 4, 800 kroków, wybór po indeksie).**
Dwa ziarna, model końcowy, protokół bez ramki:

| s_lora | TA (2024/2025) | IA | DINO |
|---|---|---|---|
| 0.45 | 78.14 / 77.75 | 73.77 / 74.38 | 0.486 / 0.500 |
| 0.60 | 76.99 / 76.07 | 75.23 / 75.72 | 0.519 / 0.529 |
| 0.80 | 74.51 / 72.71 | 76.81 / 77.06 | 0.545 / 0.548 |
| 1.00 | 70.09 / 68.74 | 76.54 / 76.79 | 0.526 / 0.524 |

Przy TA 75.27 (nasz punkt z p022 s=0.45) interpolacja daje oracle **IA 76.2, DINO 0.535** wobec naszych
**79.5 i 0.621**. Czyli **generowane adaptery biją niezależnie trenowane** o 3.3 IA i 0.086 DINO przy
zrównanym TA, mimo T× mniejszej pamięci. Zastrzeżenie do tekstu: nasz przepis ma też gałąź umiejscowienia
i kompozyty (`box_aug_p 0.5`, `augment: true`), których solo-LoRA nie ma — część przewagi może stąd
pochodzić. Wpisane do §5.2.

**3. Baseline'y seed 2025, macierz przy skali 1.0** (22487515/20/28/36): zapominanie finetune 0.4056,
EWC 0.3196, LwF 0.2526, C-LoRA 0.1638; TA 69.2–70.6, DINO 0.30–0.44. To **10× więcej** niż liczby
w tabeli pracy (LwF 0.0183, EWC 0.0363…), bo tamte były przy niskiej skali. Zapominanie rośnie ze skalą
u wszystkich metod, więc porównanie musi iść przy **zrównanym TA**, nie przy zrównanej skali.

**4. Skala odniesienia CIDM = 0.8.** Ich `inference.py` ma `--alpha` domyślnie **0.8**, a
`scripts/inference.sh` nigdy go nie nadpisuje (domyślny katalog wyników to `./results_08`). Trening:
rank 4, alpha 1.0, `alpha_list: [0, 0.7, 1.0]` tylko do podglądu w walidacji. Czyli opublikowane
liczby CIDM są przy 0.8 — to jest właściwy punkt dla baseline'ów.

Dołożone macierze baseline'ów: skale 0.5 i 0.7 (22496871/72/75/79/80) oraz 0.8 (22496917–21).
Po nich wiersz zapominania w Fig. 3 (prawy panel) idzie z odczytu przy zrównanym TA.

### Odtworzenie baseline'ów z papieru CIDM przy T=10 (2026-09-16, ~17:20; commit 6d8927c)

Cel: nasze reimplementacje mają trafiać w opublikowane TA/IA przy **ich** skali (alpha 0.8), zanim
z nich policzymy zapominanie. Dziś przy skali 1.0 (seed 2025) jesteśmy poniżej na obu osiach:

| metoda | opublikowane TA/IA | nasze @1.0 | luka |
|---|---|---|---|
| Finetuning | 70.0 / 73.7 | 69.2 / 66.2 | −7.5 IA |
| EWC | 72.7 / 75.9 | 69.7 / 70.2 | −3.0 TA, −5.7 IA |
| LwF | 73.4 / 74.1 | 70.3 / 72.9 | −3.1 TA, −1.2 IA |
| C-LoRA | 73.6 / 76.9 | 70.6 / 74.8 | −3.0 TA, −2.1 IA |

Trzy różnice wobec ich przepisu (z `options/cidm/task_*.yml` i kodu ED-LoRA): (a) **my nie
augmentowaliśmy** — `train_baselines.py` w ogóle nie czytał `training.augment`, podczas gdy nasz
przepis ma `augment: true`, więc porównanie było częściowo o danych, nie o mechanizmie; (b) oni
wkładają LoRA w **całą uwagę** (`where: Attention`, czyli attn1 + attn2), my tylko w attn2; (c) mają
**osobne embeddingi tekstowe per warstwa** (ED-LoRA) — tego nie odwzorowujemy.

Commit 6d8927c: `training.augment` działa w `train_baselines`, `scripts/_mkblvar.py` generuje warianty
`aug` i `aug_allattn`. Osiem treningów (seed 2025) + ewaluacje modelu końcowego przy 0.8 i 0.6:
22497270/72, 22497274/78 (finetune), 22497279/80, 22497281/82 (EWC), 22497283/84, 22497285/89 (LwF),
22497293/94, 22497295/98 (C-LoRA). Wybór per metoda: wariant najbliższy opublikowanym TA/IA przy 0.8;
z niego dopiero macierz zapominania.

### Co papier CIDM naprawdę podaje o baseline'ach (2026-09-16, ~18:00; arXiv 2410.17594)

Z pracy (sekcja implementacyjna): **rank 4 dla wszystkich metod**, lr **1e-3 dla embeddingów tekstowych
i 1e-4 dla UNetu**, **800 kroków na koncept**, Adam, „the same backbone" dla wszystkich; ewaluacja
**20 promptów × 50 obrazów na koncept**, metryki TA/IA na CLIP-ie, guidance 7.5. Porównują: finetuning,
EWC, LwF, RPY (replay), C-LoRA, L2DM, LoRA-M, LoRA-C.

Czego w pracy **nie ma**: siły regularyzacji dla EWC/LwF/C-LoRA (podają tylko własne γ1=0.1, γ2=1.0),
tego, które warstwy obejmuje LoRA baseline'ów, tego czy baseline'y dostają ich ED-LoRA z embeddingami
per warstwa, **żadnej metryki zapominania** i skali LoRA przy inferencji (ta jest tylko w kodzie: 0.8).
Repo zawiera wyłącznie ich metodę — implementacji baseline'ów nie opublikowali. **Idealne odtworzenie
ich baseline'ów jest więc niemożliwe z zasady**; odtwarzalne jest wszystko poza λ i ED-LoRA.

**Co u nas nie zgadzało się z ich przepisem** (i zostało naprawione, commit eab6992):
1. `LoraDataset` + `HumanResizeCropFinalV3(size=512, crop_p=0.5)`: krótszy bok do 512, z p=0.5 losowy
   kwadratowy crop, dłuższy bok do ≤512 i wklejenie w **losowe miejsce czarnego płótna**, a strata
   liczona tylko na realnych pikselach (`img_mask`). U nas baseline'y miały crop centralny bez
   augmentacji. Tryb `training.augment: cifc` odwzorowuje to wiernie, razem z maską straty.
2. `EnhanceText(enhance_type='object')`: podpisy wkładane w losowy szablon obiektowy. U nas było to
   stosowane **tylko w `train_cl`** (nasza metoda), baseline'y dostawały surowe podpisy —
   asymetria na naszą korzyść. Flaga `training.enhance_text`.
3. `ShuffleCaption(keep_token_num=1)`: na tych danych no-op (54 z 55 podpisów bez przecinka).
4. LoRA na **całej uwadze** (ich `where: Attention`), u nas tylko attn2. Wariant `cifc` dokłada attn1.

Puszczone: `finetune|ewc|lwf|clora _cifc` (22499186/88/91/93 + ewaluacje 22499187/90/92/94, skale 0.8
i 0.6, seed 2025). Wcześniejsze warianty `aug` i `aug_allattn` (22497270…22497298) zostają jako punkt
odniesienia, ile daje sama augmentacja bez letterboxa. Po nich: wybór wariantu per metoda po zgodności
TA/IA z opublikowanymi przy 0.8, potem sweep λ dla zwycięzcy i dopiero macierz zapominania.

Zastrzeżenie do tekstu: zapominania nie da się porównać z liczbą z literatury, bo **nikt go dla tego
benchmarku nie publikuje**. Nasze wiersze to nasze pomiary na naszych reimplementacjach i tak muszą
być opisane.

### Dwa ramiona baseline'ów — decyzja protokolarna (2026-09-16, ~18:40; commit 22cc91d)

Pytanie WG: czy baseline'y niezgodne z CIDM mają sens. Mają, ale muszą być zgodne **z nami**, nie z nimi.
Dwie poprawne pozycje, mieszanie ich jest najgorszym wyborem:

* **A, kontrola wewnętrzna.** Wszystko identyczne poza mechanizmem CL. Zdanie „przy zrównanym TA nasz
  mechanizm zapomina mniej" jest wtedy twierdzeniem o mechanizmach. Nie wolno przy tym mówić, że to są
  liczby z ich pracy.
* **B, zgodność zewnętrzna.** Baseline'y po ICH przepisie. Zgodności i tak nie osiągniemy (λ i ED-LoRA
  nieznane), a dodatkowo **tracimy porównywalność z naszą metodą**, bo ona trenuje na innym pipelinie.

Decyzja: **A w tekście głównym, B w dodatku jako kontrola „czy nie osłabiliśmy baseline'ów"**. Lewy panel
Fig. 3 zostaje na ich opublikowanych liczbach, więc zgodność zewnętrzna jest tam załatwiona bez
reimplementacji.

Dziura, którą przy okazji wykryto: warianty `aug` (22497270…) mają nasz crop i flip, ale **nie mają
EnhanceText**, którego nasza metoda używa — więc nie były zgodne z nami. Stąd czwarte ramię:

| ramię | augmentacja | szablony promptów | warstwy | po co |
|---|---|---|---|---|
| `_ours` (22500208/12/15/20 + macierze 22500210/13/17/22, skale 0.8 i 0.6) | nasz crop+flip | tak | attn2 | **główne**: jedyna różnica to mechanizm CL |
| `_cifc` (22499186/88/91/93) | ich letterbox + maska straty | tak | attn1+attn2 | dodatek: baseline po ich przepisie |
| `_aug`, `_aug_allattn` (22497270…) | nasz crop+flip | nie | attn2 / +attn1 | pośrednie, pokazują wkład samej augmentacji |
| bez wariantu (22487511…) | brak | nie | attn2 | stan wyjściowy z rana |

### Awaria: wyczerpany limit inodów na $SCRATCH (2026-09-16, ~18:50)

`lfs quota -u plgrrrml /net/scratch/hscra` pokazał **999 999 / 1 000 000 plików** przy 1 TB z 12 TB
zajętego miejsca. Od tego momentu **każdy zapis padał** z `OSError: [Errno 122] Disk quota exceeded`,
także `scp` jednego skryptu i `tar -cf` (nowe archiwum też potrzebuje inoda).

**Co padło:** p022_s2025 i p022_s2026 (po 7 h 20 min, ale **trening zdążył się skończyć** — padła dopiero
generacja krzywej), p022_w1460, macierz L2DM, ewaluacja bl50-finetune, bl50 EWC ×2 i LwF ×3.
**Osobno, przez mój za krótki `--time` (4 h na punkt sweepa, potrzeba 8–10):** nogs, sem128, ganchor,
ganchor_g, all50 (6 h). Dwadzieścia dwa zadania zależne dostały `DependencyNeverSatisfied`.

**Spis inodów w `outputs` (762 tys.):** phaseT 441 tys., baseline 154 tys., sweep 136 tys., reszta 31 tys.
Winowajcą są obrazy ewaluacyjne: jedna macierz 55 komórek × 10 obrazów × 10 konceptów to ~5.5 tys.
plików, a `sbatch_matrix.sh` (w odróżnieniu od `sbatch_sweep.sh`) ich **nie kasuje** po policzeniu metryk.

**Naprawa** (za zgodą WG): `_pack_phaseT.sh` pakuje obrazy każdego przebiegu w `outputs/phaseT` do
`<run>.images.tar` i kasuje pliki (`tar --remove-files`), zostawiając `cifc_metrics.json`, checkpointy
i logi — konwencja jak `.images.tar` w `phaseP`. Żeby tar w ogóle mógł powstać, najpierw skasowano
obrazy w `outputs/sweep/p022/matrix` (metryki policzone i zapisane w REPORT, obrazy odtwarzalne;
22 tys. inodów). Skrypt i log muszą leżeć w `$HOME`, bo scratch nie przyjmował nowych plików.

**Wnioski na przyszłość:**
1. `sbatch_matrix.sh` powinien kasować obrazy po policzeniu metryk, tak jak robi to `sbatch_sweep.sh`,
   albo pakować je do tara. Bez tego każda macierz zostawia 5.5 tys. plików na zawsze.
2. Przed serią zgłoszeń sprawdzać `lfs quota -u plgrrrml /net/scratch/hscra` — limit plików, nie miejsca,
   jest wiążący (CLAUDE.md mówi o tym wprost, zignorowałem).
3. Limit `--time` dla punktu sweepa to 10 h, nie 4 h.

**Odtworzone bez powtarzania treningu:** seedy 2025/2026 i w1460 miały gotowe `hyper.pt` oraz pełne
`ckpts`, więc puszczono same brakujące ewaluacje (macierze 22501069/22501080, po cztery sondy
umiejscowienia). Krzywe przy 0.45 i 0.6 dla obu seedów były już policzone przed awarią.

### Umiejscowienie wzdłuż strumienia na TRZECH ziarnach (2026-09-16, ~20:30)

Sondy 22501071/74/77/79 (ziarno 2025) i 22501081/84/85/88 (2026) obok wcześniejszych dla 2024.
Protokół jak wiersz „op" tab:grounding.

**Pierwszy koncept (pies), IoU:**

| ziarno | po zadaniu 1 | po 5 | po 10 | po 50 |
|---|---|---|---|---|
| 2024 | 0.731 | 0.828 | 0.850 | 0.786 |
| 2025 | 0.537 | 0.668 | 0.763 | 0.632 |
| 2026 | 0.530 | 0.714 | 0.643 | 0.574 |
| **średnia** | **0.599 ± 0.114** | **0.737 ± 0.082** | **0.752 ± 0.104** | **0.664 ± 0.110** |

Transfer wsteczny (1 → 10) dodatni na **każdym** ziarnie: +0.119, +0.226, +0.113.

**Siedem obiektów, model po 10 i po 50 zadaniach:**

| ziarno | IoU@10 | IoU@50 | IoU>0.5 @10 | @50 | wypełnienie @10 | @50 | DINO @10 | @50 |
|---|---|---|---|---|---|---|---|---|
| 2024 | 0.766 | 0.586 | 96 % | 70 % | 1.01 | 0.83 | 0.655 | 0.618 |
| 2025 | 0.739 | 0.566 | 96 % | 70 % | 1.00 | 0.84 | 0.668 | 0.611 |
| 2026 | 0.628 | 0.546 | 73 % | 55 % | 1.21 | 1.03 | 0.669 | 0.623 |
| **średnia** | **0.711 ± 0.073** | **0.566 ± 0.020** | 88 % | 65 % | 1.07 | 0.90 | 0.664 | 0.617 |

Spadek 10 → 50 występuje na każdym ziarnie, ćwiartka zostaje przy 96 %, wypełnienie spada do 0.90.
§5.6 przepisana na trzy ziarna; ten sam zestaw liczb wszedł do Limitations.

### Zapominanie baseline'ów przy ZRÓWNANYM TA — zmiana obrazu (2026-09-16, ~22:00)

Pełne macierze 55 komórek, seed 2025, kilka skal na metodę (22496871…, 22501349…):

| metoda | s=0.5 | s=0.7 | s=0.8 | s=1.0 |
|---|---|---|---|---|
| C-LoRA | 0.0554 (TA 78.2) | 0.0869 (76.4) | 0.1086 (74.9) | 0.1638 (70.6) |
| fine-tuning | 0.0999 (79.0) | – | 0.2302 (76.4) | 0.4056 (69.2) |
| LwF | – | – | 0.1019 (76.0) | 0.2526 (70.3) |
| EWC | – | – | 0.2023 (74.1) | 0.3196 (69.7) |

**Zapominanie rośnie monotonicznie ze skalą u każdej metody**, więc porównanie przy jednej skali nie
znaczy nic. Przy naszym punkcie pracy (TA 75.27, zapominanie **0.0094**) interpolacja daje:
C-LoRA ≈ 0.103, LwF ≈ 0.121, EWC ≈ 0.18, fine-tuning ≈ 0.257. To **rząd wielkości** różnicy, a nie
kilkadziesiąt procent jak w starej tabeli (LwF 0.0183, CIDM 0.0174), której protokołu nie da się
odtworzyć — tamte liczby musiały być czytane przy bardzo niskiej skali, gdzie baseline prawie nie
adaptuje i nie ma czego zapominać, ale jego tożsamość jest daleko poniżej naszej.

Do dokończenia: niższe skale dla EWC, LwF i fine-tuningu oraz macierz L2DM (liczą się). Wtedy prawy
panel Fig. 3 idzie z odczytu przy zrównanym TA, z wyraźnym podpisem, że to nasze reimplementacje.

### SDXL: waga regularyzatora NIE przenosi sie miedzy backbone'ami (2026-09-17, ~00:15)

Objaw: `X_sdxl_p022` (przepis headline'u) przy skali 0.4 daje TA 72.13 / IA 76.43 / DINO 0.536,
podczas gdy stary `X_sdxl_ground_800aug` przy tej samej skali mial TA 80.2 / IA 79.3 / DINO 0.597.
Gorzej na OBU osiach naraz, czyli nie przesuniecie po krzywej, tylko krzywa nizej.

Przyczyna (diff configow): stary przebieg ma `reg.weight: 100` bez klucza `space`, czyli
**przestrzen czynnikow** (`train_cl.py:170`, domyslnie `factors`). Nowy ma `reg.weight: 7120`
i `space: dw`. To sa dwa rozne cele o innej skali naturalnej -- liczby 100 i 7120 nie sa
porownywalne, a 7120 wyszlo ze sweepa na SD-1.5.

Diagnostyka z logow: surowe `reg` jest podobne na obu backbone'ach (1e-4), wiec czlon kotwicy
wnosi ~1.4 przy stracie ~0.15, czyli **dominuje dziesieciokrotnie**. Do tego norma gradientu na
SDXL to 0.04 wobec 0.17 na SD-1.5, czyli czlon dopasowania jest tam czterokrotnie slabszy --
przy tej samej wadze regularyzator przygniata go jeszcze mocniej.

Puszczony sweep: `X_sdxl_b300`, `_b1000`, `_b3000` (22505808/12/26) po 10 h, kazdy z ewaluacja
przy skalach 0.3/0.4/0.5 (22505831/32/33). UWAGA: `_mkcfg.py` wymusza `output_dir` na
`outputs/sweep/<name>`, wiec ewaluacje MUSZA wskazywac `outputs/sweep/X_sdxl_b<B>/ckpts`;
pierwsze trzy zgloszenia (22505810/18/27) mialy `outputs/sdxl/...` i padna natychmiast.

Do tekstu: jesli ktoras waga trafi w ich wiersz (TA 80.0 / IA 79.5), wiersz SDXL idzie z niej,
a w Setup ladnie sie pisze, ze przepis przenosi sie po przeskalowaniu jednego hiperparametru.
Stary przebieg 22487472 dokonczy trzy skale i zostaje jako udokumentowany wynik negatywny.

### Noc 16/17.09: zapominanie na trzech ziarnach, kompozycja negatywna, srodowisko CIDM

**Zapominanie p022, pelna macierz 55 komorek, skala 0.45 (punkt odczytu krzywej):**

| ziarno | zapominanie DINO |
|---|---|
| 2024 | 0.0094 |
| 2025 | 0.0051 |
| 2026 | 0.0035 |
| **srednia** | **0.0060 ± 0.0031** |

Wpisane do wstepu i do 5.3, slupek "nasz" w Fig. 3 zaktualizowany. UWAGA: katalogi `matrix/s05`
obu ziaren sa ZANIECZYSZCZONE -- zostaly w nich obrazy z anulowanego biegu na 1275 komorek, wiec
`cifc_metrics.json` policzyl tam 120 komorek zamiast 55 i jego `forgetting` (0.0184) jest bez
znaczenia. Skala 0.45 jest czysta (55 komorek). Przed uzyciem s05 trzeba skasowac stare obrazy
i przeliczyc metryki.

**Kompozycja: wynik negatywny.** `_compose_scenes.py` na p022 (SD-1.5, skala 0.6, sceny 3.1-3.4
przepisane z ich Rys. 3) konczy sie bez bledu, ale obrazy nie skladaja sie: w scenie palacowej
jest sam pies bez misia i kota, w sypialni kilka rozmytych psow zamiast czterech konceptow.
Potok domyslnie celuje w checkpoint SDXL i tam byl strojony; na SD-1.5 nie dziala bez dostrojenia.
Decyzja: nie wydawac na to nocy. 5.7 zostaje bez liczby, do skrocenia albo wyciecia.

**Srodowisko dla kodu benchmarku dziala** (22505970, 2 min): torch 2.6.0+cu124 na GH200,
diffusers 0.20.0, transformers 4.25.1, accelerate 0.24.0. Jedyny blad importu to `_tkinter`
(matplotlib siega po backend Tk na wezle bez X). Naprawione w `sbatch_cidm_train.sh` przez
`MPLBACKEND=Agg`. Agent dostal to srodowisko do treningu ich metody na 50 zadaniach.

**Baseline'y na 50 konceptach (model koncowy, n=50)** -- pierwsze komplety:
EWC λ=10000 s08: TA 75.66 / IA 69.09 / DINO 0.4328; C-LoRA λ=3 s08: 68.95 / 70.01 / 0.4184;
oracle (50 niezaleznych adapterow) s08: 71.84 / 68.62 / 0.4199; fine-tuning s06: 74.85 / 67.43 /
0.3963. Nasz punkt na wszystkich 50 konceptach czeka na `p022-all50`.

## 2026-09-17, rano -- odblokowanie i-wezlow, wznowienia, CIDM na 50 zadaniach

**Limit i-wezlow byl scisla przyczyna wszystkich awarii tej nocy.** 928 152 / 1 000 000 przy
starcie. To dlatego padl trening `l2dm` na 50 konceptach (22491970, `shifts_after_task15.pt
cannot be opened` przy zadaniu 15 z 50) i dlatego agent od CIDM nie zglosil ani jednego bloku.
Spakowane, nie skasowane: `_pack_scored.sh` (nowy, na `~` obok `_cleanup_scored.sh`) wklada obrazy
do `<katalog>/images.tar` tylko tam, gdzie lezy juz policzony `cifc_metrics.json`, pomija
`/teaser/`. 455 840 obrazow, bajty zostaja, odtworzenie to `tar -xf <katalog>/images.tar -C outputs`.
Zuzycie spadlo ponizej 650 000 w trakcie pisania tego wpisu.

**Wyslane zadania** (wszystkie przez `run.sh`, wiec z proweniencja):

| id | co | uwagi |
|---|---|---|
| 22510640 | macierz 55 komorek, `ewc_cifc_l300`, skale 0.8 i 0.6 | ramie kalibrowane pod ich liczby |
| 22510641 | macierz, `lwf_cifc_l0.3`, 0.8 i 0.6 | j.w. |
| 22510642 | macierz, `clora_cifc_l3`, 0.8 i 0.6 | j.w. |
| 22510643 | macierz, `finetune_cifc`, 0.6 | 0.8 i 1.0 juz byly |
| 22510649/50/51/53 | brakujaca skala 1.0 dla bl50 lwf/ewc/clora/lora_solo | poprzednie padly na TIMEOUT 5 h, teraz 8 h |
| 22510654 -> 22510656 | `l2dm` na 50 konceptach od zera + eval (afterok) | 16 h, poprzedni padl na kwocie przy zadaniu 15 |
| 22510713 | `p022-all50` skala 0.6 | poprzedni (22501097) zjadl limit 12 h na s06 |
| 22510734 -> 735 -> 737 -> 739 -> 740 | CIDM na naszych 50 zadaniach, bloki 1-10 ... 41-50 | lancuch `afterok`, po 24 h |

**Wybor lambdy pod odtworzenie ich tabeli** (ramie `cifc`, skala 0.8, L1 = |dTA| + |dIA| wobec
opublikowanych): EWC 300 (2.37), LwF 0.3 (3.95), C-LoRA 3 (4.67), fine-tuning bez lambdy (3.31).
Na tych czterech checkpointach licza sie teraz macierze 55-komorkowe -- to jest to zapominanie
"na checkpointach odtwarzajacych CIDM/CIFC", o ktore prosiles.

**Srodowisko benchmarku: drugi smieciowy import.** Po zaslepce `_tkinter` wyszedl
`lib/utils/ptp_util.py:7 from IPython.display import display`, ktory przewracal
`trainer_edlora`. Prawdziwy IPython to ~3000 plikow, wiec `sbatch_cidm_venv.sh` pisze teraz
dwuplikowa zaslepke `IPython/` z no-opowym `display`. Job 22510710: **5/5 importow OK**.

**Zadanie wstrzymane, do Twojej decyzji.** 22510712 (`ch-p022-all50-s06`) wyslalem z bledna
sciezka configu -- wskazywal `configs/phaseT/T50_mixed.yaml` zamiast `outputs/sweep/p022/config.yaml`,
a oba istnieja, wiec job by ruszyl i zapisal cudze obrazy do `outputs/sweep/p022/all50/s06`.
Zamiast `scancel` (bez Twojej zgody) dalem `scontrol hold`. Poprawny to 22510713. Zwolnic albo
anulowac 22510712 -- Twoja decyzja.

**Papier.** Wpisane liczby: 5.5 "po wszystkich 50 konceptach TA 75.1 / IA 72.0 / DINO 0.502"
(z `outputs/sweep/p022/all50/s045`, n=50); 5.6 SDXL "0.2 punktu od opublikowanej" (ich SDXL IA
to 79.5). 5.7 kompozycja opisana jako wynik negatywny, plus akapit w Limitations. 5.8 "What these
metrics resolve" przeniesione do appendixu (etykieta `sec:pitfalls` bez zmian, odsylacze
poprawione). Dlugosc bez czerwonych notatek spadla z ~9,75 do ~9,4 strony; do 9 brakuje jeszcze
okolo 19 linii.

**SDXL wciaz zle** przy przeszczepionej becie: `X_sdxl_p022` daje TA 72.13 / IA 76.43 przy 0.4
i 69.89 / 78.34 przy 0.5, przy opublikowanym punkcie CIDM 80.0 / 79.5. Sweep bety 300/1000/3000
liczy sie dalej; krzywa w `figures/make_tradeoff_sdxl.py` czeka na te punkty.

### Ablacje osadzen: dwa wyniki negatywne dla opisu metody

Piec biegow wariantow skonczonych (22501091/93/94/96, 22501502). Liczby z `score.json`,
model koncowy, jeden seed, skala 0.45 / 0.6:

| wariant | DINO t10 @0.45 | @0.6 | DINO t50 @0.45 | @0.6 | TA t10 @0.45 |
|---|---|---|---|---|---|
| p022 (odniesienie) | 0.6208 | 0.6480 | 0.5565 | 0.5814 | 75.27 |
| nogs (bez Grama-Schmidta) | **0.6321** | **0.6587** | **0.5712** | **0.6034** | 75.74 |
| sem128 (osadzenie z CLIP-a) | 0.6071 | 0.6225 | 0.5492 | 0.5763 | 74.24 |
| ganchor (clip 1.0) | 0.4933 | 0.5214 | 0.4816 | 0.5121 | 75.07 |
| ganchor-clip (clip 25) | 0.5234 | 0.5501 | 0.4935 | 0.5204 | 74.90 |
| ganchor_g | 0.5502 | 0.5836 | 0.5433 | 0.5791 | **78.40** |

**To jest problem dla opisu metody.** Abstrakt i 4.1 mowia, ze ortogonalizacja Grama-Schmidta
"lagodzi interferencje i zapominanie", a bieg bez niej jest lepszy we wszystkich czterech
komorkach przy nienaruszonym TA. Zastrzezenie: jeden seed, roznica 0.011-0.015, a rozrzut
miedzyseedowy p022 przy 0.45 to ok. 0.005 DINO -- czyli dwa odchylenia, na granicy. Dwie rzeczy
rozstrzygna: (a) macierz 55 komorek dla nogs, **22510778**, bo prawdziwe pytanie dotyczy
interferencji, a nie tozsamosci modelu koncowego; (b) drugi seed nogs, jesli (a) tez wyjdzie
na remis. Do tego czasu paragraf "What the embedding carries" w 5.2 opisuje to jako zmierzone,
z wyraznym `\todo` o liczbie ziaren.

Sem128 przegrywa wszedzie i jako jedyny ma `feasible_t10 = false`. Razem: ani geometria klucza,
ani jego tresc nie sa nosne -- liczy sie tylko to, ze klucze sa rozne.

Kotwica gradientowa (`ganchor*`) sprzeczna z hipoteza o przycinaniu: podniesienie `grad_clip`
z 1.0 do 25 odzyskuje tylko +0.030 DINO z luki -0.128, wiec strata tozsamosci jest wlasnoscia
kotwicy, nie konfiguracji. Najlepszy wariant (`ganchor_g`) po 50 zadaniach jedynie dogania
bieg glowny, placac -0.071 po 10. Nie wlaczac. Macierze 55-komorkowe dla wszystkich pieciu
wariantow wyslane: 22510778 (nogs, 0.45 i 0.5), 22510779 (sem128), 22510780 (ganchor),
22510783 (ganchor_g), 22510786 (ganchor-clip).

Drobiazg do poprawy po deadline: `train_cl.py:301` drukuje "task_cond ON (learned V_t +
Gram-Schmidt ortho)" niezaleznie od configu, wiec log biegu `nogs` klamie. Wyniki sa dobre
(`manager.py:493` czyta flage poprawnie), ale to zla proweniencja.

Wstrzymane duplikaty: 22510712 (zly config) i 22510751 (dublet 22510713, zlozony golym sbatch,
bez run-info). Oba `scontrol hold`, decyzja o anulowaniu Twoja.

### Ostatnie trzy `\pend` w appendixie: wypelnione, jedno zdanie bylo odwrotne do danych

Zrodla odnalezione w `results/*/run-info.txt` + logi:

* **Tlumienie konceptu poza ramka, caly ogon przyczynowy.** Para dopasowana, ta sama siatka
  ($\kappa = 4$, sched $0.15$), jedyna roznica to flaga `--confine_tail`: job 21597878
  (confine 0, zawarcie **0.67**, IoU>0.5 62%) kontra 21602457 (kary 3/6/10, zawarcie
  **0.78 / 0.79 / 0.81**, IoU>0.5 79/77/78%).
* **Koszt TA.** Skrypt `_ground_iou.py` drukowal TA dopiero w pozniejszej wersji, wiec para
  z TA jest w innej konfiguracji ($\kappa = 2$, sched $0.3$, tlo z `--bg_dir`): 21649595
  (confine 0, **TA 0.8073**) kontra 21649672 (confine 3, **TA 0.7464**). Kontrolowana, ale nie
  ta sama siatka co zawarcie -- odnotowane w `\todo` przy zdaniu.
* **Tlo z gotowego zbioru obrazow: zdanie w papierze bylo odwrotne do danych.** Twierdzilo, ze
  taki background "zawiera obiekt nieco lepiej" niz tlo z promptu. Para 21649718 (`--bg_dir`)
  kontra 21637497 (tlo z promptu), oba `--bootstrap 10`, `2.0:0.3:3`: zawarcie **0.92 w obu**,
  ale IoU **0.630 kontra 0.728** i IoU>0.5 **76% kontra 90%**. Czyli nie zawiera lepiej,
  a umieszcza gorzej. Zdanie poprawione wedlug danych.

Po tej poprawce w `main.tex` nie ma juz zadnego zywego `\pend{}` -- jedyny pozostaly siedzi
w zakomentowanym starym abstrakcie.

### Anulowane na Twoja prosbe

22510712 (zly config) i 22510751 (dublet 22510713, zlozony golym sbatch bez run-info) --
oba `scancel`, potwierdzone w `sacct` jako CANCELLED.

### Papier miesci sie w dziewieciu stronach

Bez czerwonych notatek cialo pracy konczy sie dokladnie na dole strony 9; strona 10 zaczyna sie
od Reproducibility Statement, ktory do limitu nie liczy. Bylo 9,75 strony. Co zostalo wyciete
albo przeniesione, w kolejnosci wielkosci:

* **5.8 "What these metrics resolve" -> appendix.** To instrukcja czytania metryk, nie wynik,
  a jej drugi akapit mowil to samo co A.5. Etykieta `sec:pitfalls` bez zmian, cztery odsylacze
  poprawione na `Appendix~\ref`.
* **5.7 kompozycja -> appendix.** Wynik negatywny zostaje w calosci, ale w Wynikach caly
  podrozdzial o nim tylko przyciagal na niego uwage. Akapit w Limitations niesie wniosek.
* **Protokol testu rozmieszczenia (5.5) -> A.5.** Opis metody, nie wynik.
* Podpis Fig. 3: zdanie o CIDM powtarzalo akapit z Limitations. Proza w 5.5 powtarzala wiersze
  Tabeli 2 liczba po liczbie. Akapit o pamieci w Limitations powtarzal liczby z 5.4.
  Zdanie "our placement table rests on one training seed" usuniete jako nieaktualne.
* Conclusion: wypadlo z niego "kept orthogonal to those of earlier concepts", bo ablacja tego
  nie potwierdza (patrz wyzej).

W `main.tex` nie ma juz zadnego zywego `\pend{}`. Zostaly `\todo` i one nadal znacza
miejsca, gdzie liczby sa jednoziarnowe albo prowizoryczne -- glownie slupki baseline'ow
w Fig. 3, ktore czekaja na macierze 22510640/41/42/43.

### Arkusze przestrzeni osadzen maja wreszcie obrazek (A.6)

Akapit "What lies between and outside the embeddings" opisywal same obserwacje wizualne --
interpolacje, skalowanie normy, wektory spoza bazy -- i nie mial **zadnego** rysunku, wiec
czytelnik musial wziac je na wiare. Obrazy istnialy od 2026-09-16 w
`outputs/sweep/p022/embed_probe2` (job 22486238, skala 0.6, 50 krokow, dwa ziarna na komorke),
tylko nikt ich nie zlozyl.

Nowy `figures/make_embedding_sheets.py` sklada z nich Figure 5: piec wierszy po piec kolumn,
naglowki alfa 0..1 dla interpolacji i 0..2x dla normy, zrodla w `figures/embed_src/`.
Wnioski z obrazow, wszystkie zgodne z tekstem:

* **Slowo klasy decyduje o kategorii, osadzenie o tozsamosci.** Ta sama interpolacja
  dog -> cat, przeczytana raz ze slowem "dog" i raz ze slowem "cat", daje caly czas psa
  albo caly czas kota. Zadnej chimery w zadnym punkcie. To najmocniejszy obraz z calego
  zestawu i wczesniej nie bylo go w papierze.
* **Interpolacja w obrebie klasy** (dog -> dog2) morfuje tozsamosc plynnie, bez przeskoku.
* **Skalowanie normy** dziala jak sila adaptera: 0 i 0.5 to prior klasy, 1 to koncept,
  1.5 juz sie rozpada (wycinanka z halo), 2 to szum. Poprawilem tekst, ktory mowil
  "beyond 1.5 degrades" -- rozpad widac juz przy 1.5.
* **Cztery losowe wektory ortogonalne do calej bazy** daja czysty, ale generyczny egzemplarz
  slowa klasy, czyli nieuzyte osadzenie nic nie kosztuje.

Figura idzie do appendixu, wiec limit dziewieciu stron nietkniety: cialo pracy nadal konczy
sie na stronie 9. Z listy brakow w appendixie znika pozycja "embedding-space sheets".

### Figure 5 przepisana, 5.6 przestawione

**Figura.** Piaty obrazek w ostatnim rzedzie wymagal dogenerowania -- sonda robila cztery
losowania spoza bazy. Job 22511879 (`--offbasis 5`, 6 minut) do `embed_probe3`. Generator
wektorow jest zaseedowany na stale (4242), wiec pierwsze cztery losowania sa te same co
poprzednio; potwierdzone tym, ze arkusze interpolacji i normy wyszly bajt w bajt tej samej
wielkosci (350628, 163594, 191588).

Uklad przerobiony na trzy grupy z osobnymi naglowkami kolumn, bo kolumny znacza w kazdej
grupie co innego (alfa, mnoznik normy, numer losowania) i bez rozdzielenia czytalo sie to
jak jedna siatka. Do tego cienkie ramki kafelkow i linie miedzy grupami. Linia nie jest
w polowie paska tylko na 0.35, bo pod nia dochodza naglowki nastepnej grupy i srodek
optyczny lezy nizej niz srodek pola.

**5.6 przestawione.** Sekcja czytala sie pol na pol: zysk do zadania 10 i spadek do 50
dostawaly tyle samo miejsca. Teraz teza sekcji to transfer wstecz (+0.15 IoU na kazdym
z trzech ziaren), a spadek zostaje, ale jako jedno zdanie z odsylaczem do Limitations,
gdzie i tak stoi razem ze srodkiem zaradczym. **Nic nie zostalo usuniete** -- pytanie
brzmialo, czy raportowac tylko do zadania 10, i odpowiedz byla nie: 5.4 raportuje modul
tresci do zadania 50, wiec urwanie rozmieszczenia na dziesiatce byloby wybiorczym
raportowaniem akurat na slabszej osi, a spadek (0.711 -> 0.566, udzial powyzej 0.5
z 88% na 65%) odtwarza sie na kazdym ziarnie.

Pozycja planu o rozmieszczeniu zaktualizowana: trzy ziarna sa, spadek sie odtworzyl.
Jednym checkpointem stoi juz tylko sama Tabela 2.

### Figure 5 przepisana na TikZ-a, spojna z teaserem

Zaokraglenie naroznikow samo w sobie dalo sie zrobic maska w PIL-u, ale to nie zalatwialoby
"uspojnienia z teaserem": teaser sklada sie w TikZ-u, wiec jego podpisy sa zlozone fontem
dokumentu, a matplotlib tylko go przybliza. Dlatego figura poszla ta sama droga co
`figures/method_tikz.tex`:

* `figures/embed_tikz.tex` (nowy) rysuje uklad, wczytywany przez `\input` z `main.tex`;
* `figures/make_embedding_sheets.py` nie sklada juz arkusza, tylko wycina 25 pojedynczych
  kafelkow do `figures/embed_cells/`;
* obrazki ida przez `\clip[rounded corners=2.5pt]`, czyli **dokladnie ta sama komenda
  i ten sam promien** co `placed_corgi.jpg` w teaserze (method_tikz.tex:121);
* hairline kafelka to `frozengrey!45, line width=0.4pt` -- jak glif bounding boxa
  w teaserze (method_tikz.tex:14); linie miedzy grupami `frozengrey!30`;
* podpisy w stylach `lbl` / `rowlbl`: `\scriptsize`, `text=black!75`, jak w teaserze.

Pulapka po drodze: `\newcommand{\cx}[1]{{(#1) * (\ih + \gx)}}` z zewnetrznymi klamrami
wywracal parser TikZ-a ("Missing number, treated as zero", 8 razy), bo arytmetyke liczy on
tylko w `{...}` i zagniezdzona klamra w srodku takiego wyrazenia jest bledem. Definicja bez
klamer, klamry na miejscu uzycia.

Kafelki 2.30 cm, caly blok miesci sie w szerokosci tekstu (zero Overfull hbox).
`figures/embedding_sheets.jpg` z poprzedniej wersji zostaje na dysku, ale **nie jest juz
przez nic uzywany** -- do skasowania, jesli ma nie lezec.

### Dwa bledy w Figure 5 i koniec z "headline checkpoint"

**Kafelki nie wypelnialy ramek**, bo `sheet()` w `_embed_probe.py` (linia 42) wkleja obrazek
**256 px w komorke o skoku 260 px** -- kazda komorka ma 4 px bialego marginesu po prawej
i u dolu. Wycinalem cala komorke, wiec kazdy kafelek mial bialy pasek. Teraz wycinamy sam
obrazek (`PITCH = 260`, `IMG = 256`). Zmierzone w PDF-ie: kafelki to 2.30 x 2.30 cm, tyle
samo co ramka.

**Linie rozdzielajace wchodzily na naglowki kolumn.** Liczylem je jako polowe przerwy
miedzy grupami, ale pod linia stoi jeszcze naglowek nastepnej grupy (ok. 0.28 cm), wiec
srodek pola to nie jest srodek optyczny. Teraz linia jest `\gr = 0.24` cm ponizej dolu
poprzedniej grupy, a przerwa podniesiona do `\gg = 0.74`.

**"headline checkpoint" wyciete z calego tekstu dla czytelnika** (6 miejsc: podpis tabeli
per skala, akapit o ortogonalnosci w A.6, akapit o przestrzeni osadzen, podpis Figure 5,
A.8 o kompozycji, lista brakow). To byl nasz zargon: czytelnik widzi jeden model, wiec
wyroznik nic mu nie mowi, a sugeruje, ze sa gdzies inne, o ktorych nie piszemy.
Zostalo tylko w `\todo` i w liscie roboczej, gdzie i tak operujemy nazwami biegow (p022).
Sprawdzone skryptem liczacym klamry, nie greppem: zero wystapien w tekscie, zero w PDF-ie.

## Nie odtwarzamy liczb CIDM/CIFC i nie da sie tego naprawic strojeniem

**Ustalenie.** Ramie wierne ich potokowi, z lambda strojona osobno pod kazdy opublikowany
wiersz (skala 0.8, ich punkt pracy):

| metoda | TA opub. | TA nasz | dTA | IA opub. | IA nasz | dIA |
|---|---|---|---|---|---|---|
| Finetuning | 70.0 | 72.97 | +2.97 | 73.7 | 74.04 | +0.34 |
| EWC λ=300 | 72.7 | 73.67 | +0.97 | 75.9 | 74.50 | -1.40 |
| LwF λ=0.3 | 73.4 | 72.78 | -0.62 | 74.1 | 77.43 | +3.33 |
| C-LoRA λ=3 | 73.6 | 72.19 | -1.41 | 76.9 | 73.64 | -3.26 |

Sredni blad bezwzgledny 1.49 TA / 2.08 IA. **Cala ich tabela rozpina sie na 3.6 TA i 3.2 IA**,
wiec blad na wiersz to polowa rozpietosci tabeli, ktora mamy odtworzyc.

**Kolejnosc metod nie zgadza sie w ZADNYM z czterech ramion.** Opublikowana po IA:
C-LoRA > EWC > LwF > Finetuning. U nas LwF wychodzi pierwszy zawsze. Blad jest systematyczny
per metoda, nie losowy: LwF **zawsze za wysoko** (+2.30, +3.33, +3.39, +5.06), C-LoRA
**zawsze za nisko** (-1.63, -2.36, -3.26, -3.30), EWC zawsze lekko za nisko. Strojenie lambdy
przesunelo LwF tylko z +5.06 na +3.33, wiec to nie jest kwestia hiperparametru.

**Dlaczego tego nie naprawimy: ich repozytorium nie zawiera baselineow.** Sprawdzone
wprost w `data/CIFC`: `lib/` ma cztery katalogi i wszystkie dotycza ED-LoRA
(`models/edlora.py`, `pipelines/pipeline_edlora.py`, `trainer_edlora.py`,
`data/lora_dataset.py`), `options/` ma tylko `cidm`, `scripts/` trzy skrypty do ich metody.
Grep po calym repo za `ewc|lwf|c-?lora|l2dm|fisher|distill|baseline` daje **jedno** trafienie:
README opisujacy ICH WLASNY modul agregacji wag. Kod tych czterech wierszy nigdy nie zostal
opublikowany, wiec nie ma zrodla prawdy do dopasowania -- kazda reprodukcja jest
re-implementacja z prozy artykulu.

**Ich wlasny `evaluate.py` przyznaje, ze nie odtwarza artykulu**: ostrzezenie w kodzie mowi,
ze przy numpy >= 1.21 normalizacja liczy sie inaczej i "to exactly replicate paper results,
please use numpy version less than 1.21, e.g., 1.20.3".

**Co jeszcze mozna dopasowac** (wszystko wspolne dla metod, wiec nie naprawi kolejnosci):
preprocessing DINO juz mamy identyczny (Resize 256 bicubic + CenterCrop 224 + norma ImageNet,
`cifc_metrics.py:93`); zostaje CLIP -- my przez HuggingFace w fp32, oni przez pakiet OpenAI
`clip` z rzutowaniem na fp16 na GPU, oraz ich CLIPScore bez prefiksu "A photo depicts"
(`append=False` domyslnie w `clipeval`).

**Blad w poprzednim raporcie agenta, do odnotowania.** Twierdzil, ze IA odtwarzamy bardzo
dobrze, "sredni blad +0.07 na 20 porownaniach, bez obciazenia". To zla statystyka: usrednia
bledy ZE ZNAKIEM po metodach, wiec +3.3 LwF-a kasuje sie z -3.3 C-LoRA i wychodzi zero.
Srednia ze znakiem ukrywa dokladnie te awarie, ktora ma znaczenie -- trzeba patrzec na blad
bezwzgledny per metoda i na kolejnosc.

**Konsekwencja dla pracy.** Akapit "The comparisons rest on two different footings"
w Limitations mowi teraz wprost, ze baseline'y nie odtwarzaja opublikowanej tabeli, z ta
liczba i z powodem, i ze zapominanie czytamy jako rzad wielkosci. Wniosek to przezywa,
bo 0.006 przeciw 0.16-0.25 to trzydziesci do czterdziestu razy, a nie dwa punkty.

### Pierwszy odczyt przy zrownanym TA z ramienia kalibrowanego

Fine-tuning ma juz komplet trzech skal, wiec jego zapominanie da sie **odczytac**, a nie tylko
ograniczyc od dolu:

| skala | TA | zapominanie | komorek |
|---|---|---|---|
| 0.6 | 76.12 | 0.1697 | 55 |
| 0.8 | 72.97 | 0.2540 | 55 |
| 1.0 | 67.00 | 0.3713 | 55 |

Interpolacja po TA do naszego 75.59 daje **0.1839**. Nasze 0.0060, czyli **31 razy mniej**.

Pozostale trzy maja na razie sama skale 0.8 (EWC 0.2335 przy TA 73.67, C-LoRA 0.1982 przy
72.19, LwF 0.1638 przy 72.78) -- wszystkie przy TA nizszym niz nasze, wiec to ograniczenia
dolne. Macierze przy 0.6 licza sie.

Widac tez, jak bardzo skala rzadzi tym pomiarem: u samego fine-tuningu zapominanie zmienia sie
z 0.17 na 0.37 miedzy skala 0.6 a 1.0. Dlatego porownanie "przy tej samej skali adaptera" nie
znaczy nic, a `_forget_matched.py` odmawia odczytu, gdy nasze TA lezy poza zmierzonym zakresem
metody.

### Czy uzywamy ICH metryki

Tak, co do wzorow i wag; nie co do kodu. Sprawdzone obustronnie:

* CLIP-T `2.5 * mean(max(0, cos))`, bez prefiksu "A photo depicts" (u nich `append=False`
  domyslnie w `clipeval`) -- zgodne.
* CLIP-I i DINO: srednia po WSZYSTKICH parach gen x ref, czyli ich `feats @ feats_ref.T`
  i `np.mean` -- zgodne.
* DINO `dino_vits16`, preprocessing Resize 256 bicubic + CenterCrop 224 + norma ImageNet --
  zgodne co do linijki (`cifc_metrics.py:93`).
* CLIP ViT-B/32 OpenAI, shortest edge 224 bicubic + CenterCrop 224 -- zgodne.

Roznice, wszystkie wspolne dla metod i dlatego niezdolne wytlumaczyc rozjazdu per metoda:
oni ladauja przez pakiet `clip` i `torch.hub`, my przez `transformers` i `timm` (zeby dzialalo
offline); `clip.load` na CUDA daje model w **fp16** i ich kod jawnie rzutuje wsad obrazowy na
fp16, my liczymy CLIP w fp32. DINO licza w fp32 tak jak my, bo rzutowanie siedzi w galezi
`hasattr(model, 'encode_image')`.

Wniosek: puszczanie ich `evaluate.py` na naszych obrazach ma mniejsza wartosc, niz wygladalo --
metryka juz jest ich, a ich wlasny skrypt i tak nie odtworzy artykulu na numpy >= 1.21.
Rozjazd siedzi w treningu baselineow.

### SDXL: beta 3000 potwierdza diagnoze, ale porownania miedzy betami wciaz nie ma

Jedyny zmierzony punkt sweepu, przy tej samej skali 0.4 co bieg odrzucony:

| bieg | beta | TA | IA | DINO |
|---|---|---|---|---|
| p022 (przeszczep z SD-1.5) | 7120 | 72.13 | 76.43 | 0.5356 |
| X_sdxl_b3000 | 3000 | **75.54** | **80.13** | **0.5931** |

Beta 3000 poprawia **wszystkie trzy metryki naraz**: +3.41 TA, +3.70 IA, +0.058 DINO. To
potwierdza diagnoze, ze 7120 (dobrane w przestrzeni delta-wag na SD-1.5) jest na SDXL o rzad
za mocne, i ze kierunek w dol jest wlasciwy.

Czego **nie wiadomo**: b300 i b1000 nie maja jeszcze ani jednego punktu (ewaluacje ruszyly
pozno i pierwsza skala konczy generacje po ~1h50), wiec **nie ma porownania miedzy betami**
i nie da sie powiedziec, czy 3000 jest optimum. Nie ma tez odczytu IA przy ich TA 80.0 --
do interpolacji trzeba dwoch punktow obejmujacych 80.0, a b3000 ma jeden, i to przy TA 75.54,
czyli ponizej. **IA 80.13 przekracza opublikowane 79.5, ale przy niedopasowanym TA, wiec
niczego nie rozstrzyga.**

`figures/make_tradeoff_sdxl.py` rozdzielone na `TABLE_P022` (beta 7120, odrzucone) i `TABLE`
(beta 3000, kandydat), z komentarzem o przyczynie.

Dwie rzeczy do napisania w tekscie, gdy SDXL wejdzie do pracy:

* `cifc_backpack` przy 0.4 ma DINO 0.070 przy sredniej 0.5356 -- jeden koncept ciagnie srednia
  w dol i trzeba to powiedziec, a nie zostawiac czytelnikowi.
* Sonda umiejscowienia na SDXL (22487476) ma **detekcje 55/84 = 65%**, a IoU liczy sie tylko
  po udanych detekcjach. `cifc_backpack` raportuje tam IoU 0.902 przy **1 detekcji na 12** --
  bez znaczenia statystycznego. Zagregowane IoU 0.597 bez podanego wskaznika detekcji
  wprowadza w blad. Tabela 2 w pracy (SD-1.5) ma kolumne Det., wiec obecne wiersze sa w
  porzadku; chodzi o to, zeby ewentualny wiersz SDXL tez ja mial.

### Beta na SDXL jest wyczerpana; maly sweep na tym, co przeszczepilismy bez sprawdzenia

**Szczyt po becie jest ZAMKNIETY i lezy PONIZEJ 3000.** Parabola w log10(beta) przez trzy
punkty obejmujace maksimum (1000, 3000, 7120), przy tej samej skali 0.4:

| metryka | szczyt przy becie | wartosc w szczycie |
|---|---|---|
| TA | ~2207 | 75.79 |
| IA | ~2128 | 80.46 |
| DINO | ~2156 | 0.598 |

Trzy niezalezne metryki daja ten sam wierzcholek, wiec to nie jest artefakt dopasowania.
Sugestia "jeszcze wiecej bety" idzie w zla strone. Puszczone: beta 2000 (22515457 -> eval
22515459, skale 0.4/0.3/0.25).

**Ale przewidywany zysk to ~0.3 IA, a luka do CIDM przy ich TA to ~2.0.** Beta przestala byc
dzwignia: maksimum jest obustronnie ograniczone, a wysokosc szczytu znamy z dokladnoscia do
ulamka punktu.

**Gdzie luka moze siedziec.** Na SDXL przeszczepilismy z SD-1.5 wszystko poza beta, a siec ma
tam **280 adaptowanych warstw zamiast 64** przy tym samym budzecie krokow i tym samym lr.
Trzy ramiona, kazde rozni sie od `X_sdxl_b3000` **dokladnie jedna rzecza** (baza to 3000, bo
jest zmierzona; 2000 dopiero leci):

| ramie | zmiana | hipoteza | job |
|---|---|---|---|
| `kv` | tylko `attn2.to_k` i `to_v` zamiast czterech projekcji | objaw to WYSOKIE IA przy NISKIM TA, czyli adapter przykrywa warunkowanie tekstem; wezsza adaptacja powinna podniesc TA | 22516060 -> 22516074 |
| `lr3e4` | lr 1e-4 -> 3e-4 | niedouczenie przy 4.4x wiekszej liczbie glow | 22516061 -> 22516080 |
| `s1600` | 800 -> 1600 krokow na zadanie | to samo, drugim kanalem, bez zmiany kroku uczenia | 22516063 -> 22516082 |

Stawiam na `kv`, bo objaw jest specyficznie niskie TA, a szerokosc adaptacji jest tym, co
tlumi warunkowanie tekstem. To zarazem powrot do oryginalnego projektu z CLAUDE.md
("swaps attn2.to_k/to_v") -- cztery projekcje sa pozniejsza zmiana i nikt jej na SDXL
nie zweryfikowal.

Ewaluacje wszystkich ramion na skalach 0.4/0.3/0.25, zeby od razu byly porownywalne miedzy
soba i zeby domknac odczyt przy ich TA 80.0 bez ekstrapolacji.

### Sredniki wyciete z prozy, "load-bearing" wyrzucone

78 srednikow zamienionych w `main.tex` plus 2 w podpisie rysunku metody
(`figures/method_fig.tex`, ktory jest osobnym plikiem i nie lapie sie na patche do main).
Zasada: srednik laczacy dwa zdania -> kropka i wielka litera; przecinek tylko tam, gdzie
druga czesc zaczyna sie od spojnika albo gdzie srednik siedzi w nawiasie.

**Czego NIE wolno bylo ruszyc i co zostalo swiadomie** (5 w wyrenderowanym PDF-ie):

* 3 separatory cytowan, `(Hu et al., 2021; Kumari et al., 2022)` -- generuje je natbib
  z `\citep{a,b}`, to styl bibliografii, nie pisanie;
* 2 notacje matematyczne: warunkowanie `eps_theta(x_s, s, c; h_phi(v_t))` i konkatenacja
  `[v_t ; gamma(b)]`. Przed zamiana maskowalem `\;`, `;\,` i `\,;\,`, bo pierwsze to
  odstep w matematyce, a pozostale to wlasnie ta notacja.

Srednikow w prozie: **zero**. "load-bearing" usuniete (jedno wystapienie w 5.2, zamienione
na "matters").

Pulapka po drodze, warta zapamietania: `cmd && grep -c ... && cp ...` urywa lancuch, bo
`grep -c` zwraca kod 1 przy zerze trafien. Przez to kopia do katalogu testowego sie nie
wykonala i pomiar pokazywal stary stan. Do liczenia w lancuchach `&&` uzywac `grep -c ... || true`
albo rozdzielac srednikiem powloki.

## Dwie awarie infrastruktury, obie ciche

### 1. Macierze ablacji liczyly 1275 komorek zamiast 55 (moj blad, powtorzony)

`sbatch_matrix.sh` ogranicza macierz do pierwszych dziesieciu zadan **tylko** gdy w srodowisku
zgloszenia stoi `GEN_EXTRA="--only_tasks 0,1,2,3,4,5,6,7,8,9"`. Zmienna nie jest zapisywana
w `run-info.txt` (jest tam `komenda`, nie srodowisko), wiec z proweniencji nie da sie
odtworzyc, ze poprawny bieg p022 ja mial, a moj nie. To ten sam tryb awarii, ktory jest juz
odnotowany wyzej dla macierzy ziaren.

Pieciu jobom (22510778/79/80/83/86) pozwolilem liczyc 3-4 h, zanim to zauwazylem. Anulowane
za zgoda uzytkownika, skasowane ich wyniki czastkowe (101 060 obrazow, **zero**
`cifc_metrics.json`, wiec nic policzonego) i wyslane od nowa: 22517823 (nogs, skale 0.45
i 0.5), 22517825 (sem128), 22517826 (ganchor), 22517830 (ganchor_g), 22517839 (ganchor_clip).

Kasowanie bylo konieczne, nie kosmetyczne: `cifc_metrics.py` skanuje caly katalog eval, wiec
zostawienie `after_task10..49` z porzuconego biegu dalo by macierz o zlej liczbie komorek --
dokladnie tak jak zanieczyszczone `sweep/p022_s2025/matrix/s05` (120 komorek zamiast 55).

### 2. Efektywny limit czasu na plgrid-gpu-gh200 jest DUZO nizszy niz deklarowany

`sinfo` i `scontrol show partition` podaja `MaxTime=2-00:00:00`. W praktyce zadania z limitem
**20 h i 24 h stoja bezterminowo** z `Reason=PartitionConfig`, a 14 h i mniej idzie normalnie.
Nie jest to QOS: wszystkie te zadania maja `QOS=normal`, ktore nie ma `MaxWall`; nie jest to
tez pamiec ani CPU (blokowane byly zarowno `cpu=32,mem=120G`, jak i `cpu=8,mem=80G`).

**`PartitionConfig` znaczy "nigdy nie wystartuje", nie "czeka w kolejce"** -- i to jest
pulapka, bo w `squeue` wyglada jak zwykly PENDING. Caly lancuch CIDM-50 (22510734/35/37/39/40)
stal tak od rana i nikt by tego nie zauwazyl bez sprawdzenia kolumny REASON.

Naprawione przez `scontrol update jobid=... TimeLimit=12:00:00` na wszystkich pieciu blokach
CIDM oraz na `ch-sdxl-s1600`; wszystkie natychmiast przeszly w `Priority`. Uzytkownik moze
limit tylko OBNIZAC, wiec to dziala bez administratora.

**Wniosek na przyszlosc: na Heliosie nie zglaszac zadan dluzszych niz ~12 h.** Trener CIDM
pomija zadania juz wytrenowane (`DONE_MARK` w `sbatch_cidm_train.sh`), wiec dlugi blok da sie
bezpiecznie podzielic na wznowienia.

### Wstrzymane, do anulowania

* `22516146 ch-p022-kv` -- wariant z samym `to_k`/`to_v`. **Po cichu wylacza modul
  umiejscowienia**: `manager.py:392` tworzy bramki tylko dla warstw konczacych sie na `to_q`,
  a `get_ground` kluczuje po `attn2_name + ".to_q"`. Bez `to_q` w `target_modules` jest zero
  bramek, zaden blad, i dopiero wywrotka w logowaniu (`train_cl.py:681`, `gv[-1]` na pustej
  liscie) po ~300 krokach. Tak padl `ch-sdxl-kv` (22516060). Zastapione wariantem `noout`
  (bez `attn2.to_out.0`, z zachowanym `to_q`): 22517531 na SDXL i 22517536 na SD-1.5.
* `22516074 ch-sdxl-kv-eval` -- `DependencyNeverSatisfied` po awarii swojego treningu.

Poprawka kluczowania bramek jest warta zrobienia **po deadline**, bo zmiana klucza rozwali
ladowanie wszystkich istniejacych `hyper.pt` (bramki sa zapisanymi parametrami).

### Figure 3 ma wreszcie prawdziwe slupki

Zapominanie przy **zrownanym TA 75.59**, ramie kalibrowane pod ich opublikowana tabele,
macierze 22510640/41/42/43, po 55 komorek kazda:

| metoda | zapominanie | ile razy wiecej niz my |
|---|---|---|
| EWC λ=300 | 0.1874 | 31x |
| fine-tuning | 0.1839 | 31x |
| C-LoRA λ=3 | 0.1484 | 25x |
| LwF λ=0.3 | 0.1199 | 20x |
| **nasze** | **0.0060 ± 0.0031** | — |

**L2DM i CIDM celowo bez slupka.** L2DM ma macierze tylko w naszym potoku, nie w ramieniu
odtwarzajacym ich, wiec nie bylby porownywalny z pozostalymi czterema. Dla CIDM nie mamy
zadnego wlasnego pomiaru zapominania, a opublikowane tabele go nie podaja. Podpis mowi to
wprost, zamiast zostawiac czytelnika z pytaniem, czemu ich nie ma.

Poprawione przy okazji dwa zdania, ktore byly napisane przed pomiarem:

* Conclusion mowil "forget as little as the strongest baselines" -- **zanizal wlasny wynik**,
  bo to dwadziescia razy mniej niz NAJLEPSZY baseline. Teraz "forgets twenty times less".
* Related work obiecywal, ze zmierzymy, co daje ortogonalizacja wejsc. Pomiar wyszedl zerowy,
  wiec obietnica znika, a cytowania zostaja.

### Dlaczego skracanie o 2 linijki nie dzialalo przez pol godziny

Kazde ciecie wyzej bylo natychmiast pochlaniane przez przelamanie akapitow, a na stronie 10
uparcie zostawaly dokladnie **dwie** linijki Conclusion. To nie byl przypadek: to
`\widowpenalty` -- LaTeX nie zostawi jednej sierocej linijki na stronie, wiec przy kazdym
cieciu trzymal dwie zamiast jednej. Dopiero zwolnienie tylu miejsca, zeby **caly** ostatni
akapit zmiescil sie na stronie 9, przelamalo pat.

Wniosek: przy dopasowywaniu do limitu stron liczy sie, czy caly ostatni akapit sie miesci,
a nie ile linijek wystaje. Mierzenie "ile linijek za duzo" wprowadza w blad przy 1-2.

Przeniesione do appendixu przy tej okazji: akapit o SDXL z 5.2 (jego liczby sa i tak
prowizoryczne, bo strojenie bety trwa) -- teraz `\subsection{The second backbone}`
z odsylaczem z ciala pracy.

## Dwie rzeczy, ktorych praca nie miala, a powinna

### Siatka zapominania (Figure 6), zero nowych generacji

`train_cl.py` sam zapisuje `outputs/phaseT/T50_mixed/forgetting/after_task{NN}/sample_{S}.png`
-- PIERWSZY koncept re-samplowany po **kazdym** z pieciudziesieciu zadan, tym samym promptem
i tym samym ziarnem. Po spakowaniu i-wezlow leza w `T50_mixed.images.tar`. Wyciagniecie
szesciu checkpointow (po zadaniu 1, 10, 20, 30, 40, 50) i dwoch ziaren dalo gotowa figure,
**bez generowania ani jednego obrazu**.

Nowe pliki: `figures/make_forget_grid.py` (wycina kafelki), `figures/forget_tikz.tex` (uklad,
te same zaokraglone narozniki i hairline co teaser), `figures/forget_cells/` (12 kafelkow).
Figura w appendixie; do ciala pracy tylko jesli zapadnie taka decyzja, bo limit stron napiety.

### Dryf generowanych aktualizacji -- glowny mechanizm zmierzony BEZPOSREDNIO

Nowy `scripts/_dw_drift.py`. Dla konceptu j i checkpointu k >= j liczy

    || dW_j(phi_k) - dW_j(phi_j) ||_F / || dW_j(phi_j) ||_F

rozwijajac kwadrat normy roznicy na trzy iloczyny skalarne, kazdy z macierzy r x r (ta sama
tozsamosc sladu co `_spectrum.dw_gram`, uogolniona na DWA zestawy czynnikow przez `dw_cross`).
Zero generacji, 18 sekund na dziesieciu checkpointach.

Wynik dla p022:

| zadan od wlasnego | dryf dW |
|---|---|
| 1 | 0.0074 |
| 3 | 0.0138 |
| 5 | 0.0185 |
| 7 | 0.0230 |
| 9 | 0.0245 |

Po dziewieciu kolejnych zadaniach aktualizacja generowana dla pierwszego konceptu odplywa
o **niecale 2.5%** swojej normy, a krzywa sie wyplaszcza. To jest pierwszy w tej pracy
bezposredni dowod, ze regularyzator wyjscia robi to, co obiecujemy -- dotad mielismy
wylacznie skutki na obrazach, czyli trzy warstwy od mechanizmu.

Pulapka po drodze: checkpoint po zadaniu k niesie warunkowanie kanoniczne **tylko dla zadan
0..k**, wiec pytanie o wszystkie T konceptow na kazdym checkpointcie konczy sie
`RuntimeError: canonical conditioning for task 1 not set`. Skrypt pyta o `k + 1`.

### Luka, ktora przy tym wyszla: NIE MA biegu bez regularyzatora

Przegladajac `reg.weight` we wszystkich punktach sweepu: najnizsza wartosc to p013 z 54.5,
i ten punkt rozni sie od p022 takze innymi rzeczami, wiec nie jest ablacja. **Praca nie ma
ani jednego biegu z wylaczonym regularyzatorem wyjscia**, czyli bez swojego glownego
mechanizmu -- a to jest pierwsze pytanie, ktore zada recenzent. Puszczone: `p022_noreg`
(22559556, `reg.weight=0.0`, reszta identyczna z p022).

Docelowo figura z dryfem ma trzy krzywe: p022, p022_noreg i p022_nogs. Wtedy jednoczesnie
waliduje regularyzator i odpowiada na pytanie o Grama-Schmidta, bez przebierania w wynikach.

## Dryf na pelnych 50 zadaniach: Gram-Schmidt w koncu cos pokazuje

`scripts/_dw_drift.py` przepuszczony przez oba ramiona na calym strumieniu (`dw_drift50.json`,
49 opoznien, minuta na bieg, zero generacji).

| opoznienie | p022 (z GS) | p022_nogs | iloraz |
|---|---|---|---|
| 1 | 0.0114 | 0.0114 | 1.003 |
| 10 | 0.0386 | 0.0395 | 1.023 |
| 20 | 0.0562 | 0.0589 | 1.048 |
| 30 | 0.0716 | 0.0754 | 1.053 |
| 40 | 0.0903 | 0.0955 | 1.057 |
| 49 | 0.1062 | 0.1170 | 1.102 |

Wynik nie jest w poziomie krzywej, tylko w jej **ilorazie**: przy opoznieniu 1 ramiona sa
nierozroznialne, a przewaga ortogonalizacji rosnie MONOTONICZNIE z dlugoscia strumienia.
To jest dokladnie ta teza, ktora juz stoi w tekscie ("mechanizm nie jest jeszcze potrzebny
w skali, ktora mierzymy"), tylko teraz poparta pomiarem zamiast argumentem wymiarowym.

Uwaga metodologiczna zapisana w podpisie figury: przy opoznieniu L usredniamy po dokladnie
`50 - L` parach, wiec ogon ma slabe poparcie (jedna para przy 49). Zacieniony, nie obciety.

Nowe pliki: `figures/make_drift.py`, `figures/drift.pdf`, `figures/drift_src/*.json` (w repo pracy).

### Co to zmienilo w tekscie

Akapit o bliskiej ortogonalnosci konczyl sie zdaniem, ze GS trzymamy "rather than for any
effect we can measure at these lengths". To przestalo byc prawda i zostalo poprawione --
bliska ortogonalnosc *generowanych aktualizacji* faktycznie nie zalezy od GS, ale ich *dryf*
zalezy. To dwa rozne pytania i teraz sa rozdzielone.

## Blad rzeczowy w opisie ablacji osadzen

Praca opisywala ramie `p022_sem128` jako "a CLIP **text** embedding of the concept".
`train_cl.py:347` liczy tam **srednie CLIP-owe osadzenie OBRAZOW** referencyjnych konceptu
(komentarz w kodzie mowi wprost: "Images, not text"). Poprawione.

## Tabela osadzen ma tylko jedna z trzech kolumn, ktore obiecalismy

Sprawdzenie sweepu: **zero** punktow z `task_cond.key_prompt: canonical`, a zaden punkt
z `learn_v: true` nie rozni sie od p022 tylko tym jednym parametrem (`~/_find_learnv.sh`).
Czyli kolumny "CLIP" i "uczone" nie istnialy. Zgloszone:

| job | punkt | roznica wobec p022 |
|---|---|---|
| 22592763 | `p022_clipkey` | `task_cond.key_prompt=canonical` |
| 22592772 | `p022_learnv` | `task_cond.learn_v=true` |
| 22592777 | macierz clipkey | `afterok:22592763`, z `GEN_EXTRA="--only_tasks 0..9"` |
| 22592778 | macierz learnv | `afterok:22592772`, z `GEN_EXTRA="--only_tasks 0..9"` |

`GEN_EXTRA` NIE ma wartosci domyslnej w `sbatch_matrix.sh` -- bez niego macierz liczy 1275
komorek zamiast 55. Ten sam blad juz raz kosztowal ~18 GPU-h.

Przy `key_prompt: canonical` na CIFC dwa koncepty tej samej klasy (dog/dog2, cat/cat2) dostana
**identyczny klucz**, bo identyfikator jest zdejmowany i prompty kanoniczne sa tym samym tekstem.
To nie jest usterka konfiguracji, tylko wlasnie ten wynik, ktory uzasadnia projekt -- ale trzeba
go opisac jako kolizje, a nie jako "CLIP wypada gorzej".

### Zebrane macierze ablacji (55 komorek, s_lora = 0.45, jedno ziarno kazda)

| ramie | TA | IA | DINO | zapominanie |
|---|---|---|---|---|
| p022 (glowne, 3 ziarna) | 75.59 +- 0.91 | 79.07 +- 0.34 | 0.6165 | 0.0060 +- 0.0031 |
| p022_nogs | 75.74 | 79.52 | 0.6321 | 0.0023 |
| p022_sem128 | 74.24 | 79.23 | 0.6071 | 0.0239 |
| p022_ganchor | 75.07 | 72.89 | 0.4933 | 0.0839 |
| p022_ganchor_clip | 74.90 | 74.30 | 0.5234 | 0.0852 |
| p022_ganchor_g | 78.40 | 74.91 | 0.5502 | 0.0007 |

**Korekta wobec tego, co mowilem wczesniej.** Twierdzilem, ze nogs zapomina "4x mniej"
(0.0023 wobec 0.0094). To bylo porownanie z NAJGORSZYM ziarnem p022. Przy trzech ziarnach
srednia to 0.0060 +- 0.0031, wiec 0.0023 lezy jedna trzecia odchylenia ponizej -- to jest
szum, nie roznica. Dwa dodatkowe ziarna nogs (22578809, 22578906) to domkna.

Za to `sem128` przy 0.0239 jest szesc odchylen powyzej i to jest realne: tresc klucza ma
znaczenie, jego geometria nie.

Trzy ramiona `ganchor*` to ablacja **kotwicy regularyzatora** (`reg.ground_anchor`), a nie
osadzen zadan -- nazwa mylila. Wszystkie trzy wyraznie traca IA/DINO.

## Pulapka w preambule pracy, ktora czekala na dzien zgloszenia

`\draftnotesfalse` -- przelacznik istniejacy wylacznie po to, zeby uzyc go przed wyslaniem --
wywalal build: `! Incomplete \iffalse; all text was ignored`. Srodowisko `todolist` otwieralo
`\ifdraftnotes` w czesci poczatkowej i domykalo `\fi` w koncowej, wiec przy falszu warunek
nie mial pary. Zaden PDF nie powstawal.

Naprawione bez nowego pakietu (`comment.sty` nie ma w TinyTeX): definicja srodowiska jest teraz
bezwarunkowa, a `\ifdraftnotes ... \fi` obejmuje jedyne miejsce uzycia w calosci.

Przy okazji zmierzone na wersji zgloszeniowej (`\iclrfinalcopy` + `\draftnotesfalse`,
skrypt `mk_submission.py`): **cialo pracy ma dokladnie 9 stron**, bibliografia zaczyna sie
na stronie 10, calosc 18 stron. Figura z dryfem zmiescila sie w appendixie bez kosztu.

## `to_out.0` nie jest opcjonalne -- i to obala moja wlasna hipoteze o polowie sieci

Mowilem, ze jesli `p022_noout` utrzyma jakosc, to siec jest o cwierc mniejsza i przeciecie
kosztow pamieci przesuwa sie z 51 konceptow na okolo 26. **Nie utrzymala.** Porownanie
jak z jak (`curveA_t9`, s_lora = 0.45, ta sama sciezka scoringowa):

| ramie | TA | IA | DINO |
|---|---|---|---|
| p022 | 75.27 | 79.46 | 0.6208 |
| p022_noout | 68.70 | 68.90 | 0.3466 |

Przy s_lora = 0.75 noout dochodzi do DINO 0.379, czyli podniesienie skali tego nie ratuje.
Po pieciudziesieciu zadaniach to samo (0.271 przy 0.45).

Sprawdzone, ze to wynik, a nie zepsuty bieg: `diff` configow pokazuje JEDNA roznice --
usuniety `attn2.to_out.0` z `target_modules`. Nie jest to powtorka pulapki z `ch-sdxl-kv`,
gdzie usuniecie `to_q` po cichu wylaczalo modul lokalizacji (`manager.py:392` zaklada warstwy
konczace sie na `to_q`) -- tu `to_q` zostaje.

Dlaczego to ciekawe, a nie tylko negatywne: ED-LoRA i cala rodzina metod benchmarku adaptuja
**wylacznie** `to_k` i `to_v`. Adapter dopasowywany bezposrednio moze sobie na to pozwolic.
Hipersiec, ktora adapter GENERUJE, nie moze -- obie warstwy czytajace podpis sa maskowane
tokenowo, wiec dzialaja na kilku pozycjach, i bez `to_out.0` zostaje za malo strumienia
rezydualnego, w ktory mozna pisac. Dopisane do appendixu jako akapit "All four projections
are required".

Ewaluacja ramienia SDXL bez `to_out` stoi w kolejce i powie, czy efekt jest specyficzny dla SD-1.5.

## Nowy skrypt: sufit DINO ze zbioru referencyjnego

`scripts/_ref_ceiling.py` (job 22594237). Praca twierdzi, ze per-konceptowej liczby DINO nie da
sie czytac bez sufitu, jaki narzucaja same zdjecia referencyjne, i podaje "0.95 do 1.09 razy
sufit" -- ale zmierzone na `P_ground_gsa_nocap_all`, czyli NIE na glownym checkpointcie.
Skrypt przelicza to na p022.

Jedna pulapka zapisana w skrypcie: `_cross_cos` z `cifc_metrics.py` usrednia po calej macierzy
iloczynow, wiec uzyta na zbiorze z samym soba wliczylaby przekatna (same jedynki). Przy czterech
zdjeciach referencyjnych to 25% wyrazow i sufit wyszedlby zawyzony tym mocniej, im mniej zdjec
ma koncept -- czyli dokladnie tam, gdzie zalezy nam na dokladnosci. Liczymy po gornej trojkatnej.

## Stan klastra, 17.09 wieczorem

Kolejka stoi. Cztery zadania biegna, dwadziescia jeden czeka, i przez ponad pol godziny nie
wystartowalo nic -- lacznie z jobem na dwadziescia minut, ktory normalnie wchodzi w backfill
od razu. `sinfo`: 89 wezlow w stanie `mix-`, 11 `alloc`, reszta `drain`/`fail`/`resv`. Wezly
maja wolne rdzenie (np. 32/256 zajetych), wiec waskim gardlem sa GPU, nie CPU. Fairshare
0.449, czyli to nie my jestesmy spychani. Nie ma zadnej rezerwacji serwisowej.

Wniosek dla harmonogramu: szacunek "wiekszosc wynikow w ciagu 24 godzin" trzeba traktowac
jako optymistyczny. Sciezka krytyczna (CIDM-50) nadal nie ruszyla.

## Sufit referencyjny przeliczony na p022 -- liczba w tekscie byla z innego checkpointu

Joby 22594237 (s_lora = 0.45, punkt pracy) i 22597584 (s_lora = 0.9, koniec krzywej tozsamosci),
oba po 15 sekund.

| koncept | ref | sufit | s=0.45 | s=0.9 |
|---|---|---|---|---|
| painting | 7 | 0.423 | 0.319 (0.75) | 0.409 (0.97) |
| ink painting | 5 | 0.432 | 0.347 (0.80) | 0.398 (0.92) |
| drawing | 6 | 0.555 | 0.421 (0.76) | 0.513 (0.92) |
| backpack | 6 | 0.604 | 0.546 (0.90) | 0.594 (0.98) |
| duck toy | 4 | 0.721 | 0.679 (0.94) | 0.670 (0.93) |
| teddy bear | 7 | 0.774 | 0.745 (0.96) | 0.771 (1.00) |
| cat2 | 5 | 0.782 | 0.768 (0.98) | 0.794 (1.01) |
| dog | 5 | 0.849 | 0.800 (0.94) | 0.803 (0.95) |
| cat | 5 | 0.898 | 0.836 (0.93) | 0.869 (0.97) |
| dog2 | 5 | 0.913 | 0.747 (0.82) | 0.825 (0.90) |

**Co sie zmienilo wobec tekstu.** Praca mowila "0.95 do 1.09 razy sufit dla kazdego konceptu
obiektowego". Na p022 w punkcie pracy to jest **0.82 do 0.98**, czyli sufitu NIE osiagamy.
Stara liczba pochodzila z `P_ground_gsa_nocap_all`, modelu o mocniejszej tozsamosci.

Ale to nie jest zla wiadomosc, tylko inne miejsce na tej samej krzywej: przy s_lora = 0.9,
ten sam checkpoint, wychodzi **0.90 do 1.01** i cat2 sufit przekracza. Czyli roznica w punkcie
pracy to swiadomie oddane dopasowanie tekstowe, a nie tozsamosc, ktorej adapter nie umie
osiagnac. Zmierzone na obu koncach, nie wywnioskowane ze sredniej -- drugi job byl po to,
zeby tego nie ekstrapolowac.

**Mocniejszy wynik, ktory przy okazji wyszedl.** Sufit tlumaczy praktycznie caly per-konceptowy
porzadek DINO: Spearman **0.93**, Pearson **0.97**, z dog2 jako jedynym konceptem poza rankingiem
(najwyzszy sufit 0.913, a najnizszy stosunek 0.82). To jest wlasnie argument, ktory appendix
chcial postawic, tylko teraz z liczba zamiast z przykladem. Nowa Tabela 3 w appendixie.

Zakres sufitu 0.60-0.91 z tekstu byl poprawny, ale dotyczyl WYLACZNIE konceptow obiektowych --
po wszystkich dziesieciu jest 0.42 (painting) do 0.91 (dog2). Style siedza nisko, bo styl nie
ma jednego podmiotu, ktory DINO moze dopasowac. Dopisane wprost.

## Porownanie przy T = 50 na wszystkich piecdziesieciu konceptach

Cztery baseline'y `ch-bl50-*-s10` sie domknely, wiec `outputs/baseline50/` ma juz pelny
przeglad po skalach. Nasza liczba: `outputs/sweep/p022/all50/s045`.

| metoda (skala) | TA | IA | DINO |
|---|---|---|---|
| **nasza (0.45)** | **75.10** | **72.01** | **0.5023** |
| EWC l=10000 (0.6) | 75.87 | 68.11 | 0.4193 |
| EWC l=10000 (0.8) | 75.66 | 69.09 | 0.4328 |
| LwF l=1 (0.6) | 76.08 | 66.75 | 0.3929 |
| LwF l=3 (0.6) | 74.98 | 68.11 | 0.3974 |
| fine-tuning (0.6) | 74.85 | 67.43 | 0.3963 |
| C-LoRA l=3 (0.6) | 72.29 | 69.18 | 0.4227 |
| C-LoRA l=3 (0.8) | 68.95 | 70.01 | 0.4184 |
| LoRA solo (0.6) | 74.46 | 66.57 | 0.3946 |
| LoRA solo (1.0) | 67.70 | 69.40 | 0.4193 |

Przy TA rzedu 75 mamy IA o **prawie trzy punkty** wyzej i DINO o **0.07** wyzej niz najlepszy
baseline. Zadne ramie nie dochodzi do naszej tozsamosci przy naszym dopasowaniu tekstowym:
C-LoRA zbliza sie na IA, ale placi za to szescioma punktami TA.

Nawet kontrola bez interferencji (osobny adapter na koncept, `lora_solo`) zatrzymuje sie na
DINO 0.419. Nie przeciagam tego w teze o transferze -- jedno ziarno i nie sprawdzilem, czy
`lora_solo` nie jest po prostu niedouczony przy tym budzecie. Zapisane jako obserwacja.

### Pulapka, ktora o malo nie weszla do pracy

`outputs/sweep/p022/curveA_t49` liczy **n = 10** (dziesiec konceptow benchmarku ocenianych
checkpointem po 50 zadaniach), a `outputs/baseline50/*/final` liczy **n = 50**. Zestawienie
tych dwoch kolumn obok siebie wygladaloby sensownie i byloby bez sensu -- inne zbiory
konceptow. Wlasciwa nasza liczba to `all50/s045`, i to jej uzylem. Dopisane jako `\todo`
przy zdaniu w sekcji skalowania, zeby nikt tego pozniej nie pomylil.

Job `ch-p022-all50-s06c` (22518394) dolicza to samo przy skali 0.6 -- wtedy porownanie bedzie
przy tej samej skali po obu stronach, a nie tylko przy tym samym TA.

## Noc z 17 na 18.09

Policzyly sie trzy joby, wszystkie przed 00:52, i **od tamtej pory do 08:30 nie wystartowal
zaden**. To jest glowna wiadomosc z tej nocy, nie wyniki.

| job | stan | czas |
|---|---|---|
| ch-p022-all50-s06c | COMPLETED | 08:51 |
| ch-sdxl-b2000-eval | COMPLETED | 05:13 |
| ch-sdxl-s1600 (trening) | COMPLETED | 11:14 |
| ch-bl50-l2dm | **TIMEOUT** | 14:00:13 |

`ch-bl50-l2dm` wyczerpal czternastogodzinny limit i nie doszedl do konca, przez co jego
ewaluacja (22510656) wpadla w `DependencyNeverSatisfied`. Niezaleznie od tego L2DM przy T = 50
i tak wypadl z planu -- przy pieciudziesieciu zadaniach raportujemy fine-tuning, CIDM i nasza.

### `all50/s06`: porownanie przy TEJ SAMEJ skali adaptera

Dotad mielismy nasza liczbe przy T = 50 tylko dla s_lora = 0.45, wiec porownanie bylo przy
zrownanym TA, ale przy roznych skalach po obu stronach. Teraz jest jedno przy tej samej skali:

| metoda, s = 0.6 | TA | IA | DINO |
|---|---|---|---|
| **nasza** | **72.50** | **75.30** | **0.5527** |
| C-LoRA l=3 | 72.29 | 69.18 | 0.4227 |
| fine-tuning | 74.85 | 67.43 | 0.3963 |
| EWC l=10000 | 75.87 | 68.11 | 0.4193 |
| LwF l=3 | 74.98 | 68.11 | 0.3974 |
| LoRA solo | 74.46 | 66.57 | 0.3946 |

C-LoRA trafia w nasze TA z dokladnoscia do dwoch dziesiatych punktu, wiec to jest porownanie
zrownane JEDNOCZESNIE po skali i po TA -- i przewaga jest tam **szersza** niz przy 0.45:
+6.1 IA i +0.13 DINO, wobec +2.9 IA i +0.07 DINO przy s = 0.45. Roznica bierze sie stad, ze
krzywe baseline'ow sa plaskie. W pracy podany jest **zakres** (3-6 punktow IA, 0.07-0.13 DINO),
zeby nie wybierac sobie korzystniejszego punktu.

## Przepiecie lancucha CIDM-50 i czyszczenie kolejki

Lancuch b02..b05 (cztery joby po dziesiec zadan, kazdy z osobnym oczekiwaniem w kolejce)
zastapiony dwoma: **22751156** (zadania 11-30, 7h) -> **22751188** (31-50, 10h) ->
**22751399** (generacja + scoring, 8h). Podstawa: z czasow per zadanie w logu b01 (4:01 przy
k = 2 rosnac do ~7:00 przy k = 10) wychodzi t(k) ~ 3.3 + 0.38k minut, czyli 11-30 to ~3.6h,
a 31-50 ~6.2h. Podzial na cztery bloki byl zabezpieczeniem z czasow, gdy nie znalem kosztu.

Nic policzonego nie przepadlo: `sbatch_cidm_train.sh` pomija zadania, ktore maja juz
`edlora_model-latest.pth`, a b01 pisal do domyslnego `./output`, wiec scalone bloki wznawiaja
od zadania 11 z tego samego katalogu. To wazne, bo `trainer_edlora.py:79` szuka poprzednich
checkpointow po zaszytej sciezce -- inny `CIDM_OUT` zerwalby konsolidacje.

Anulowane po uzgodnieniu: 22510656 (martwy), 22578809 i 22578906 (dwa dodatkowe ziarna nogs)
oraz ich macierze 22579088 i 22579232 -- macierz bez swojego treningu bylaby dokladnie tym
samym `DependencyNeverSatisfied`, ktory sprzatalismy. Kolejka 21 -> 16.

Wniosek o ziarnach nogs zapisany w pracy zamiast pomiaru: 0.0023 lezy jedna trzecia odchylenia
ponizej sredniej trzech ziaren p022 (0.0060 +- 0.0031), wiec kolejne ziarna potwierdzilyby
odczyt, a nie go rozstrzygnely.

## Debugging kompozycji: negatywny wynik z 5.8 jest prawdopodobnie samobojem

Punkt wyjscia: obrazki z `outputs/compose_scenes/p022_fig3` wygladaja zle, a praca raportuje
to jako **negatywny wynik o metodzie**. Trzeba bylo sprawdzic, czy to metoda, czy nasz harness.

### Dowod rzeczowy: scena 3.3

Gorna polowa obrazu to poprawny stadion (prompt globalny). Dolna polowa to **plaska szara
plyta**, granica jest idealnie pozioma na y ~ 0.49. Wszystkie trzy boxy sceny 3.3 maja
y = [0.488, 0.974]. Szara plyta to dokladnie suma pudelek regionow.

Mechanizm, `sampling.compose_sample_regions`:

    flat = torch.full((1, 3, height, width), 0.5, ...)      # PLASKA SZAROSC
    z_bg = vae.encode(flat * 2 - 1).latent_dist.mean * vae_scale_factor
    ...
    if z_bg is not None and i < bootstrap_steps:
        inp_r = inp * mm + bg_t * (1 - mm)

Przez pierwsze 15 z 50 krokow (`bootstrap_steps=15` w manifescie) kazdy przebieg regionu widzi
latent, ktorego wszystko poza boxem to zaszumiona szarosc 0.5. Siec przewiduje wiec szum dla
"podmiot na jednolitym szarym tle", predykcja idzie do latentu wewnatrz maski z waga 0.9, i po
15 krokach szara plyta jest wypalona w ukladzie. Obszar nieobjety zadnym boxem idzie czystym
`eps_global` -- stad poprawny stadion u gory.

`alpha = 0.1` jest przy tym POPRAWNE: wewnatrz regionu 0.1*global + 0.9*region, w tle
0.1*global + 0.9*global, oba sumuja sie do 1. Moje pierwsze podejrzenie (globalny prompt
wygrywa w boxie) bylo chybione.

Help `--boot_grid` mowi wprost "szare tlo w bootstrapie wychodzi na obrazie, wiec to jest os
do zmiatania" -- czyli ktos to juz podejrzewal i nie domknal.

### Drugi podejrzany: brak zszywania szwu

`regional_steps = null`, czyli twarde maski regionow dzialaja przez WSZYSTKIE 50 krokow.
Standardowo obcina sie je wczesniej, zeby koncowe kroki zharmonizowaly obraz globalnie. Bez
tego szew na granicy boxa nie ma jak zniknac -- i na scenie 3.1 pies jest wycieta glowa
z twarda krawedzia dokladnie na granicy pudelka.

### Najwazniejsze: to NIE jest nasza sciezka

Figura 3 leciala domyslnym `--mode unp1`, czyli odtworzeniem ICH rownan 4-5. Nasza teza to
`--mode single`, jedno przejscie UNetu, i `compose_sample_single` **nie ma parametru bootstrap
w ogole** -- jest strukturalnie odporne na te usterke. Czyli negatywny wynik, jesli sie
potwierdzi, dotyczy naszego odtworzenia ich metody, a nie naszej.

### Sprzeczne docstringi, ktore podwazaly wazny wynik

`compose_sample_single` ostrzegal: "RegionKVAttnProcessor liczy q globalnie i to_out bez
adaptera, wiec w tym torze dziala WYLACZNIE tekstowa polowa naszej LoRA". Docstring samej
klasy mowil odwrotnie: "galaz regionu przechodzi przez WSZYSTKIE cztery projekcje".

Kod rozstrzyga na korzysc klasy -- `_branch` liczy cala galaz attn2 pod adapterem regionu
i scala PO `to_out`. Ostrzezenie bylo prawdziwe do commitu **aa787eb (2026-09-14)**,
"fix: w torze jednoprzebiegowym dzialala tylko polowa adaptera", ktory jest przodkiem commitu
biegu figury 3. Docstring `compose_sample_single` poprawiony.

To nie jest kosmetyka: ablacja `p022_noout` wlasnie pokazala, ze `to_out.0` jest dla tej metody
krytyczne (DINO 0.62 -> 0.35 bez niego). Czyli **kazdy wynik kompozycji sprzed aa787eb powstal
z polowa metody wylaczona** i nie jest porownywalny z pozniejszymi. Dopisane do docstringa.

### Dwie usterki uspione, sprawdzone ze nie gryza TU

1. Wklady regionow sumuja sie z twardymi maskami, a `clamp(sum(masks), 0, 1)` zabezpiecza
   TYLKO czlon tla. Zachodzace boxy dostalyby w czlonie regionowym podwojna wage. Sprawdzone
   wszystkie cztery sceny: 3.1 dwa regiony rozlaczne w x, 3.2 cztery rozlaczne (V1 i V3
   rozchodza sie w y: [0.205, 0.583] wobec [0.629, 0.972]), 3.3 i 3.4 po trzy rozlaczne w x.
   Nieaktywne tutaj, aktywne przy pierwszym ukladzie z zachodzeniem.
2. `_spatial_mask` przy `side * side != n_tokens` zwraca same jedynki, czyli brak maskowania
   bez zadnego ostrzezenia. Przy 512^2 latenty sa kwadratowe, wiec nieaktywne.

### Czwarta os, gdyby pierwsze trzy nie wystarczyly

Wedlug docstringa `compose_sample_single` MIEKKA separacja attn1 nie ma ani jednego pomiaru:
wszystkie dotychczasowe proby szly ze `strength=None`, czyli kara 1e4, przy ktorej `leak` jest
bezczynny (zmierzone 2026-08-31: leak=0.5 dalo liczby identyczne z leak=0.0). Czyli werdykt
"izolacja attn1 niszczy generacje" dotyczy wylacznie TWARDEJ izolacji.

### Puszczone (kazdy bieg to ~3 minuty GPU)

| job | co sprawdza | wyjscie |
|---|---|---|
| 22752320 | `--boot_grid 0,4,8,15` | `compose_scenes/p022_boot` |
| 22752322 | `--mode single`, nasza sciezka | `compose_scenes/p022_single` |
| 22752323 | `--solo 1`, sufit jednokonceptowy | `compose_scenes/p022_solo` |
| 22752327 | `--bootstrap 0 --regional_steps 30` | `compose_scenes/p022_heal` |

Zadne nie nadpisuje `p022_fig3`.

## Przeglad WSZYSTKICH wynikow kompozycji: odpowiedz byla na dysku od 14.09

Trzynascie katalogow w `outputs/compose_scenes/`. Kazdy bieg zapisuje `commit` w manifescie,
wiec wersje kodu da sie ustalic wiarygodnie -- czas katalogu by nie wystarczyl, bo job moze
ruszyc przed commitem, ktory widac w `ls`.

### 1. Fix polowy adaptera nie uniewaznia niczego

`aa787eb` (14.09, 18:45) wyprzedza NAJSTARSZY bieg kompozycji (`single_gsa_s04`, commit
`aa787eb-dirty`). Czyli zaden zachowany wynik kompozycji nie powstal z polowa metody
wylaczona. Moja wczesniejsza obawa byla nieuzasadniona i to jest dobra wiadomosc.

### 2. `bootstrap=15` w KAZDYM biegu poza `boot_grid` -- ale to ma znaczenie tylko dla `unp1`

`compose_sample_single` nie przyjmuje bootstrapu w ogole. Manifest i tak zapisuje
`bootstrap_steps: 15`, bo bierze to z argumentow, a nie z tego, co sampler faktycznie robi.
To jest mylace pole w manifescie: w torze `single` ta liczba nic nie znaczy.

### 3. `boot_grid` zawieral rozstrzygniecie od 14.09

SDXL, `mode=unp1`, `self=0`, `ground=False` -- czyli ten sam uklad co figura 3, rozny tylko
bootstrapem (i backbonem). Scena 3.1:

| bootstrap | co widac |
|---|---|
| 0 | piekny, spojny obraz: zamek, kot w czarodziejskim plaszczu. **Psa NIE MA** |
| 8 | oba podmioty obecne, scena spojna, wokol psa lekka prostokatna aureola |
| 15 (figura 3) | szara plyta na sumie pudelek, obraz zniszczony |

Czyli to nie jest usterka z jednym poprawnym ustawieniem, tylko **kompromis zle nastrojony
na obu koncach**: przy 0 podmiot nie ma sie gdzie uformowac i znika, przy 15 szare tlo wychodzi
na obraz. Srodek dziala. Dokladnie to, po co ten trik powstal -- tylko z wartoscia dobrana
w ciemno.

I rzecz najgorsza: ta siatka istniala **od 14.09**, a bieg figury 3 poszedl 17.09 z domyslnym
`--bootstrap 15`, czyli z wartoscia, o ktorej juz bylo wiadomo, ze jest zla.

### 4. Nasz tor dziala i mamy na to dowod sprzed tygodnia

`sched2/s4_r0_h1` (commit `fc4a381`, `mode=single`, `ground=True`, `self=4.0`): pies i kot oba
obecne, scena spojna, zamek w tle, zero szwow i zero szarosci. To jest dobra kompozycja.
Formalnie ma `bootstrap_steps=15` w manifescie, ale w torze `single` to pole jest martwe.

Wniosek: **figura 3 trafila w najgorsza komorke calej przestrzeni** -- tor odtwarzajacy ICH
rownania, przy bootstrapie, ktory niszczy obraz, na backbonie i skali, dla ktorych ten tor
nie byl sprawdzony.

### Co to znaczy dla 5.8

Negatywny wynik kompozycji nie jest wlasnoscia metody. Mamy zachowane obrazy, na ktorych
nasz tor komponuje poprawnie. Sekcja w obecnej formie musi zniknac albo zostac przepisana
na to, co naprawde pokazuja dane.

Czego jeszcze NIE wiem i co domkna joby 22752320/22/23/27: `sched2` jest na SDXL
(`X_sdxl_ground_800aug`), a nie na p022/SD-1.5, i nie zweryfikowalem TOZSAMOSCI konceptow na
tych obrazach -- kot moze byc generycznym kotem, a nie naszym. Dopiero `--mode single` na p022
daje porownanie jak z jak wobec figury 3.

## Poprawki kompozycji (commity c373cd3 i 3eab252, wypchniete i pobrane na klastrze)

### Poprawka 1: tlo bootstrapu losowane zamiast stalej szarosci

`compose_sample_regions` kodowalo JEDEN staly szary obraz 0.5 i pokazywalo go regionowi przez
wszystkie `bootstrap_steps` krokow, wiec model go wypalal. Teraz kodujemy szesc losowych
stalych kolorow raz przed petla i wybieramy z palety po `(i + ri)`, czyli inny kolor na krok
I na region. Izolacja zostaje (region nadal nie widzi sasiadow, bo po to ten trik istnieje),
ale zaden kolor nie jest uprzywilejowany. Tak jest tez w oryginalnym MultiDiffusion, gdzie
tlo jest losowym stalym obrazem.

**Rozwazane i odrzucone:** podstawienie pod tlo BIEZACEJ SCENY. Skasowaloby caly efekt --
region znow widzialby sasiadow, czyli wyszedlby `bootstrap_steps = 0`, a to jest konfiguracja,
w ktorej drugi podmiot znika. Zapisane, bo brzmi sensownie i latwo do tego wrocic.

### Poprawka 1b: widok o zerowym kroku szedl do VAE

Zlapane przed uruchomieniem, nie po. `col.expand(1,3,H,W)` daje widok o kroku 0, a `.to(dtype)`
przy TYM SAMYM dtype zwraca ten sam obiekt -- sprawdzone: `data_ptr` identyczny, strides
`(3,1,0,0)`. VAE w SD-1.5 siedzi w fp32, dokladnie jak losowany kolor, wiec do `vae.encode`
trafialby tensor niesciagly. Na SDXL (VAE fp32, `dtype` bf16) konwersja by go skopiowala
i problem by sie nie ujawnil -- czyli usterka wybuchlaby wylacznie na SD-1.5, czyli dokladnie
w biegach, ktore wlasnie zakolejkowalem. Dodane `.contiguous()`.

### Poprawka 2: separacja attn1 tylko tam, gdzie rozstrzyga sie uklad

Bez zmiany kodu -- pokretla juz byly. `sched2` poszedl w `s4_r0_h1`, czyli sila 4 na WSZYSTKICH
rozdzielczosciach i przez WSZYSTKIE kroki, i kupil drugi podmiot kosztem atrybutow (kot bez
kaptura i plaszcza). Uklad rozstrzyga sie przy wysokim szumie na malych mapach, atrybuty wiaza
sie pozniej i na wiekszych, wiec ciecie attn1 tam, gdzie wiaze sie kaptur, to wlasnie utrata
kaptura. Job 22753226: `--self_grid 4:0:16,2:0:16,4:0:32 --self_sched 0.3`.

### Konsekwencja dla zakolejkowanych jobow

Wszystkie piec biegow kompozycji ruszy juz z `3eab252`. W szczegolnosci **22752320 nie zmierzy
juz starego zachowania** -- zamiast siatki bootstrapu po szarym tle bedzie siatka po losowym.
To jest bardziej przydatne, ale trzeba wiedziec, ze pomiaru starego zachowania na p022/SD-1.5
nie bedzie. Stare zachowanie jest udokumentowane przez `boot_grid` z 14.09 (SDXL) i przez
sama figure 3.

## Czemu nic nie leci, 18.09 rano

Nie z naszej winy. Partycja `plgrid-gpu-gh200`: 440 GPU skonfigurowanych, **399 zajetych**.
Z 41 pozostalych, po rozbiciu na stany wezlow:

| stan wezla | wolnych GPU | uzyteczne? |
|---|---|---|
| IDLE+RESERVED | 20 (5 wezlow) | nie, rezerwacja nie nasza |
| DOWN+DRAIN / +NOT_RESPONDING / DOWN+FAIL | 20 (5 wezlow) | **nie, zepsute** |
| MIXED+PLANNED | 3 (3 wezly) | tylko w backfillu |

Czyli realnie schedulowalne sa **trzy GPU**. Piec wezlow jest awaryjnych, co wyjmuje
20 GPU ze sprzetu. Fairshare 0.449, priorytety normalne, zgloszenia poprawne.

Uwaga do czytania `sinfo`: sufiks `-` w stanie (np. `mix-`) to **PLANNED**, czyli wezel
zaklepany przez backfill pod przyszly job, a nie usterka. `scontrol show node` pokazuje to
wprost jako `State=MIXED+PLANNED`. Przez ten sufiks przez pol dnia czytalem 89 wezlow jako
uszkodzone.

### Szacowany start zalezy WYLACZNIE od limitu czasu

| limit | szacowany start |
|---|---|
| 1 h | dzis ~15:02 |
| 6-8 h | 20.09 |
| 10-12 h | **25.09 o 08:51** |

25.09 to dzien terminu. Skrocilem dwa limity tam, gdzie zmierzone czasy na to pozwalaja:

* **22518946** (`ch-cidm-bench10`) 12 h -> **3 h**. Porownywalny `ch-cidm50-b01`, dziesiec
  zadan ich metoda, zajal 1:08. Osmiokrotne przeszacowanie.
* **22751156** (`ch-cidm50-b11_30`) 7 h -> **5 h**, przy szacunku 3.6 h.

Czego NIE skrocilem i dlaczego: punkty sweepu (`clipkey`, `learnv`, `noreg`) potrzebuja
swoich 10 h. Zmierzone: `sem128` 7:54, `noout` 8:32, a **`nogs` 9:35 przy limicie 10 h**.
Zejscie do 8 h, zeby trafic w lepszy koszyk backfillu, wywaliloby czesc z nich na TIMEOUT --
a w tej sesji stracilem tak wyniki juz dwa razy (piec jobow przy 5 h, l2dm przy 14 h).

## SDXL: czemu p022 wypada slabo i co z tym robimy

Cel: ich opublikowany wiersz SDXL to TA 80.0 / IA ~79.5.

| config | regularyzator | TA@0.4 | IA@0.4 | DINO |
|---|---|---|---|---|
| ich wiersz | -- | 80.0 | ~79.5 | -- |
| X_sdxl_ground_800aug | **czynniki**, beta=100 | 80.15 | 79.27 | 0.597 |
| X_sdxl_b3000 | dW, beta=3000 | 75.54 | 80.13 | 0.593 |
| X_sdxl_b2000 | dW, beta=2000 | 73.60 | 80.08 | 0.577 |
| X_sdxl_b1000 | dW, beta=1000 | 74.12 | 78.88 | 0.572 |
| X_sdxl_b300 | dW, beta=300 | 72.85 | 76.26 | 0.535 |
| X_sdxl_p022 | dW, beta=7120 | 72.13 | 76.43 | 0.536 |

### Dwie osobne przyczyny, nie jedna

**1. Beta nie przenosi sie miedzy backbone'ami.** Punkty X_sdxl_b* roznia sie od X_sdxl_p022
WYLACZNIE wartoscia reg.weight (zweryfikowane diffem configow). Zejscie 7120 -> 3000 odzyskuje
3.4 TA i 3.7 IA. Wartosc 7120 dostrojono na SD-1.5, gdzie warstw adaptowanych jest 64
i parametrow 21.9M; SDXL ma 280 warstw i 91.1M.

**2. Reszta luki to NIE beta.** ground_800aug ma reg.space domyslny, a domyslna wartosc to
"factors" (train_cl.py:170) -- czyli beta=100 i beta=3000 sa w ROZNYCH jednostkach
i zestawianie ich w jednej kolumnie jest mylace (zrobilem tak raz, poprawione). Do tego
key_dim 128 zamiast 256 i weight_decay 0.0 zamiast 0.01. Trzy roznice naraz, wiec zadnej
nie da sie przypisac przewagi osobno.

### Problem, ktorego wczesniej nie widzialem

Sekcja 4 uzasadnia regularyzator na dW, a nie na czynnikach: czynniki nie sa jednoznaczne,
bo (A R, R^-1 B) daje to samo dW. Tymczasem appendix SDXL raportuje ground_800aug, czyli
config z regularyzatorem na CZYNNIKACH. Czyli pokazujemy transfer metody na drugi backbone
przy uzyciu innego objectivu niz ten, ktorego bronimy w metodzie.

Moja wczesniejsza rada w tej sesji -- "skasowac todo przy akapicie SDXL i zostawic
ground_800aug, bo p022 jest gorszy" -- byla zla i ja odwoluje. Lepsze liczby nie sa warte
niespojnosci mechanizmu.

### Bieg rozstrzygajacy

22842641: X_sdxl_b3000_kd128wd0 -- b3000 (dW, beta=3000) plus key_dim=128 plus
weight_decay=0.0, czyli trzy pozostale roznice wobec ground_800aug zebrane razem PRZY
ZACHOWANIU przestrzeni dW. Config wygenerowany przez scripts/_mkcfg.py, diff pokazuje
dokladnie cztery zmiany (te trzy plus output_dir i nazwa w wandb). Ewaluacja 22842707
na afterok, skale 0.4 / 0.5 / 0.3.

Jesli wyjdzie TA ~80 / IA ~79, mamy liczby ground_800aug z mechanizmem pracy i problem
znika. Jesli nie -- wiemy, ze przewaga siedzi w przestrzeni czynnikow, i wtedy uczciwa droga
jest zaraportowac OBA wiersze i nazwac roznice wprost, zamiast milczec.

### Limity ewaluacji SDXL

Sciete z 8 h na 6 h (22517539, 22516080, 22516082). Zmierzone czasy porownywalnych biegow:
5:11, 5:15, 5:14, 5:33. Ryzyko male, bo sbatch_eval_sdxl.sh w linii 25 pomija skale, ktora
ma juz cifc_metrics.json -- TIMEOUT kosztuje jedna skale, nie caly bieg. Ta sama asymetria
co przy treningu CIDM i odwrotna niz przy punktach sweepu, ktorych dlatego nie tykam.

## CIDM-50 padl na zadaniu 18: powtorzone slowo klasy w podpisie

Job 22751156 (`ch-cidm50-b18_30`... wlasciwie `b11_30`) FAILED po 1:28, na zadaniu 18.
Zdazyl domknac zadania 11-17, wiec na dysku jest **17 checkpointow**.

Blad jest w ICH kodzie, `trainer_edlora.py:315`:

    cross_map = torch.stack([batch_map[..., batch_pos] for batch_pos, batch_map in ...])
    RuntimeError: stack expects each tensor to be equal size,
                  but got [64, 64, 2] at entry 0 and [64, 64, 4] at entry 1

`cal_attn_reg` sklada mapy uwagi po batchu, zakladajac, ze kazda probka ma tyle samo pozycji
nowych tokenow. Jedna mial 2, druga 4.

### Przyczyna: nasze wlasne podpisy, ale nie pomylka w zrodle

Podpis zadania 18 (`cc101_luggage_backpack1`, klasa "backpack") brzmial:

    backpacks & bags - leather backpacks & bags

Slowo klasy dwa razy, a ich `lora_dataset.process_text` robi zwykle `str.replace(k, v)` --
bez granic slow i WSZYSTKIE wystapienia -- wiec probka dostala 4 pozycje zamiast 2.

To nie jest zly plik zrodlowy. `_caption_cc101.py` captionuje WARUNKOWO: BLIP dostaje slowo
klasy jako prompt i je kontynuuje, dokladnie po to, zeby slowo klasy na pewno bylo w wyniku
(bez tego `ConceptDataset._caption` doklejalby fraze identyfikatora z przodu i koncept mialby
dwie konkurencyjne nazwy). W trzech przypadkach kontynuacja uzyla slowa klasy DRUGI raz.

**Luka byla w walidacji, i jest waska:** skrypt sprawdzal `cw not in c`, czyli OBECNOSC slowa
klasy -- ten sam test, ktory potem robi trening -- ale nigdy jego KROTNOSCI.

### Skan calego strumienia: trzy koncepty z piecdziesieciu

| zadanie | koncept | podpis przed | po |
|---|---|---|---|
| 18 | cc101_luggage_backpack1 | `backpacks & bags - leather backpacks & bags` | `backpacks & bags` |
| 29 | cc101_flower_2 | `flower in the wildflower garden` | `flower in the garden` |
| 47 | cc101_pet_cat4 | `cat in a cat house` | `cat in a house` |

Zadanie 29 NIE jest falszywym trafieniem: `str.replace` nie zna granic slow, wiec "flower"
lapie sie takze wewnatrz "wildflower". Gdyby nic nie zrobic, bieg wywalilby sie jeszcze
dwa razy, przy 29 i 47.

Poprawka jawna (trzy podmiany z asercja na stan przed), po niej przeskan calego zbioru:
**zero** podpisow z powtorzonym slowem klasy i zero bez slowa klasy.

Do `_caption_cc101.py` dolozona brakujaca walidacja `caps.count(cw) > 1` z raportem per
koncept, zeby przy nastepnym strumieniu to nie przeszlo. Zmiana LOKALNA, niecommitowana --
do wznowienia nie jest potrzebna, bo podpisy sa juz wygenerowane.

### Jak to dzialalo u NAS -- i dlaczego nic nie padlo

`data.ConceptDataset._caption` ma dokladnie te sama podmiane wszystkich wystapien
(`cap.replace(cls, repl)`), wiec problem jest ten sam. Ale my nie skladamy map uwagi przez
`torch.stack`, tylko budujemy MASKE, a `token_span_mask` z zalozenia zaznacza wszystkie
wystapienia (jej docstring mowi to wprost). Czyli u nas adapter po prostu dzialal takze na
drugim spanie -- na "cat" w "cat house" -- zamiast sie wywalic. Ciche zanieczyszczenie,
jeden podpis na 4-7 w trzech konceptach na piecdziesiat.

**Konsekwencja dla odtwarzalnosci, ktora trzeba znac:** nasze policzone biegi T=50 (p022,
all50 i reszta) trenowaly sie na STARYCH podpisach. Pliki na dysku juz sie z nimi nie zgadzaja.
NIE przetrenowujemy -- trzy podpisy przy piecdziesieciu konceptach nie sa warte 10 h GPU --
ale roznica jest tutaj zapisana.

Nierozstrzygniete: czy nasz `token_span_mask`, ktory dopasowuje wzorzec TOKENOW a nie
podciagow, lapal sie wewnatrz "wildflower". Zalezy od BPE i nie sprawdzilem tego.

### Wznowiony lancuch

| job | zakres | limit |
|---|---|---|
| 22866703 | zadania 18-30 | 4 h |
| 22866759 | 31-40 | 4 h, afterok |
| 22866874 | 41-50 | 5 h, afterok |
| 22866942 | generacja + scoring | 8 h, afterok |

Trzy bloki zamiast jednego, kazdy w "szybkim" koszyku: dzis 3 h weszlo natychmiast, 5 h
w godzine, a 8 h i wiecej nadal stoi. Trening pomija zadania z gotowym
`edlora_model-latest.pth`, wiec wznowienie nic nie powtarza i TIMEOUT kosztuje jedno zadanie.

Anulowane martwe: 22751188 i 22751399 (`DependencyNeverSatisfied` po awarii 22751156).

### Poprawka poprawki: podpis 18 zaczynal sie od przymiotnika

Pierwsza wersja dla zadania 18 brzmiala `leather backpacks & bags`. Usuwala powtorzenie, ale
lamala konwencje CALEGO zbioru: `_caption_cc101.py` captionuje warunkowo wlasnie po to, zeby
podpis ZACZYNAL sie od slowa klasy. Do tego `luggage_backpack1` nie ma `attr_strip`, wiec
"leather" zostaloby jako atrybut opisany tekstem -- czyli atrybut, ktorego adapter sie nie uczy.
Poprawione na `backpacks & bags`. Zlapane, zanim job wystartowal.

Przy okazji dolozony TRZECI warunek do kontroli, ktorego dotad nikt nie testowal: czy podpis
zaczyna sie od slowa klasy. Pelny skan 249 podpisow po poprawkach:

    bez slowa klasy: 0 | z powtorzeniem: 0 | niezaczynajacych sie od slowa klasy: 0

Czyli captioning warunkowy zadzialal jak zaprojektowano na calym zbiorze, a jedynym naruszeniem
konwencji byla moja wlasna poprawka.

## Maska tokenowa byla WYLACZONA dla dwoch konceptow, i nikt tego nie widzial

Znalezione przy okazji sprzatania podpisow, ale to jest osobny blad i dotyczy biegow,
ktore juz policzylismy.

`token_span_mask` (src/tokens.py) dopasowuje CIAG TOKENOW slowa klasy do stokenizowanego
promptu. Gdy nie trafi, dziala fallback z linii 70-71:

    if mask[b].sum() == 0:
        mask[b] = 1.0

czyli maska staje sie CALYM promptem. A maskowanie istnieje po to, zeby delta LoRA szla
WYLACZNIE na tokeny konceptu, a reszta kontekstu zachowala bazowe K/V -- praca opisuje to
w sekcji 4 jako czesc metody. Fallback odwraca wiec dzialanie mechanizmu, po cichu.

Kiedy nie trafia: gdy podpis uzywa ODMIENIONEJ formy slowa klasy. "earring" to token
`earring</w>`, a "earrings" to `earrings</w>` -- inny token, wzorzec nie pasuje.

Zmierzone na zywym tokenizerze SD-1.5 (job 22861965, `scripts/_tok_dupes.py`):

| podpis | klasa | zaznaczonych pozycji |
|---|---|---|
| `cat in a cat house` | cat | 2 (dwa spany -- tego sie spodziewalismy) |
| `flower in the wildflower garden` | flower | 1 (CLIP ma "wildflower" jako JEDEN token, wiec nasza maska tokenowa go NIE lapie -- inaczej niz ich `str.replace`) |
| `backpacks & bags - leather backpacks & bags` | backpack | **77 z 77, czyli fallback** |

Skan calego starego zbioru: **21 z 249 podpisow** ma slowo klasy tylko w formie odmienionej.

| zadanie | koncept | ile podpisow |
|---|---|---|
| 17 | cc101_jewelry_earring | **7 z 7** ("earrings") |
| 42 | cc101_decoritems_houseplant3 | **7 z 7** ("houseplantia") |
| 27 | cc101_decoritems_houseplant2 | 5 z 8 |
| 18 | cc101_luggage_backpack1 | 1 z 5 |
| 45 | cc101_luggage_purse2 | 1 z 4 |

Czyli **dwa koncepty strumienia trenowaly sie z calkowicie wylaczonym maskowaniem**, a trzy
czesciowo. Przy T=10 (koncepty CIFC) problem nie wystepuje -- tam podpisy sa benchmarku
i uzywaja formy podstawowej.

Wniosek do walidacji, nie tylko do naprawy: warunek na podpis brzmi "slowo klasy dokladnie raz
JAKO OSOBNE SLOWO", a nie "jako podciag". Dotad sprawdzalismy `cw in caption`, czyli podciag,
co przepuszcza i "earrings", i "houseplantia". Trzeci warunek, ktory dolozylem dzis rano
("zaczyna sie od slowa klasy"), okazal sie z kolei ZA OSTRY -- patrz nizej.

## Podpisy przepisane z ogladania obrazow

Powod: `attr_strip` zdejmuje przymiotnik STOJACY PRZED slowem klasy, wiec nie ma szans zlapac
"jacket made from recycled materials" ani "chair slip covers for sofas". A atrybut opisany
w tekscie idzie przez zamrozony encoder i adapter sie go nie uczy -- to jest teza z sekcji 4
pracy, wiec podpis opisujacy wyglad konceptu psuje dokladnie to, czego chcemy nauczyc.

Czterech agentow (Sonnet) obejrzalo wszystkie 225 obrazow `cc101_v2` i napisalo podpisy od
nowa. Instrukcja: najpierw obejrzec WSZYSTKIE zdjecia konceptu naraz, ustalic co jest STALE
(kolor, material, ksztalt, znaki szczegolne, twarz -- czyli tozsamosc, ktorej adapter ma sie
uczyc z pikseli) a co ZMIENNE (gdzie lezy, na czym, co obok, poza), i opisywac wylacznie to
drugie.

Zabezpieczenia w instrukcji: slowo klasy dokladnie raz i w formie podstawowej; kolory
OTOCZENIA dozwolone ("backpack on a red bench" jest w porzadku, bo czerwona jest lawka);
zakaz slow klasowych INNYCH konceptow strumienia, chyba ze rzecz naprawde jest na zdjeciu;
zakaz zmyslania lokalizacji (przy tle studyjnym maja napisac wprost "on a plain background").

Druga runda: pierwsza wersja miala wszystkie 225 podpisow zaczynajacych sie od slowa klasy,
bo tak naprowadzily przyklady. To robilo nasz strumien OSTRZEJSZYM niz benchmark -- podpisy
CIFC tej konwencji nie trzymaja ("woman with a red backpack standing on a hill with clouds
in the background"). Agenci przepisali skladnie tak, zeby okolo polowy zaczynala sie od czegos
innego, z klasa przeniesiona w glab zdania.

## Przeniesienie na Athene (18.09 wieczorem)

Helios stanal: przed nami weszlo szesnascie cudzych zadan o priorytecie 8812-9182 z limitami
8-24 h, szacunki dla naszej sciezki krytycznej cofnely sie na 25.09, czyli dzien terminu.
Athena mala pusta kolejke, wiec przeniesienie.

Co bylo, a czego nie: venv `continualhyper-athena`, klon repo, `benchmark_dataset`, CIFC,
SD-1.5 i SDXL w cache HF, aktywny grant. Brakowalo dowiazan `data` i `outputs` w klonie
(nie byl uzywany od dawna) oraz nowych danych.

`cc101_v2` odtworzone na miejscu zamiast kopiowane: obrazy juz tam byly, wiec przeniesione
zostaly tylko podpisy, a strukture odbudowal skrypt, w ktorym **decyduje zbior podpisow** --
dla kazdego podpisu szuka odpowiadajacego obrazu. Dzieki temu nie ma dwoch zrodel prawdy
o tym, co jest w strumieniu. 40 konceptow, 225 obrazow, 225 podpisow, zgodnie z Heliosem.

`$HOME` na Athenie blokowal zgloszenia przy 99.8% zajetosci. `~/.cache/pip` mial 6.1 GB
czystego cache kol -- skasowany, kwota spadla do 39.3%.

### Ablacje przy T = 10 zamiast 50: piec razy taniej

Akapit "What the embedding carries" raportuje ramiona clipkey/learnv/noreg WYLACZNIE
z macierzy 55-komorkowej przy T = 10. Trenowalismy je na pelnym strumieniu piecdziesieciu
zadan, czyli piec razy drozej, niz wymaga wynik, ktory z nich bierzemy. Nowy config
`T10_v2.yaml` (asercja sprawdza, ze wszystkie dziesiec zadan to koncepty CIFC) skraca
kazde ramie z ~8 h do ~1.6 h. Przy okazji te biegi nie dotykaja danych cc101 w ogole.

### Wznawianie treningu

`--init_ckpt` i `--start_task` (c0f84a5) plus `--end_task` (f1a2627). Zadania 1-10 to koncepty
CIFC, wiec checkpoint po dziesiatym jest ten sam co poprzednio i nie trzeba go liczyc.
Trening ciety na kawalki, ktore mieszcza sie w backfillu.

Sprawdzone przed uzyciem, bo to najwrazliwszy plik w repo: optymalizator powstaje w petli
(linia 362), wiec nie ma stanu do odtworzenia; `weight_decay` jest bezstanowy; harmonogramu
LR nie ma w ogole. Jawnie odtwarzane sa `anchor_conds` -- kotwice regularyzatora, ktorych
NIE ma w checkpointcie. Bez nich regularyzator nie chronilby zadan sprzed wznowienia,
a w logu nie bylo by tego widac.

### Trzy awarie zlapane i naprawione

1. **`kd128wd0` PADL** po 3:19: `bg_bank` pusty, bo config SDXL uzywa `data/backgrounds_sdxl`,
   a na Athenie byl tylko `backgrounds`. Brakujace dane, nie blad kodu. Przeniesione, bieg
   wznowiony.
2. **Venv CIDM zbudowal sie pod zla nazwa i bez torchvision.** Moja detekcja klastra szla
   po `hostname`, a wezel Atheny nazywa sie `t0022` -- nie zawiera "athena", wiec skrypt
   poszedl galezia Heliosa. `slurm/env.sh` rozwiazuje to od dawna: hostname, a jak nie, to
   po sciezce `$SCRATCH`. Wszystkie trzy runnery CIDM uzywaja teraz tej samej reguly.
3. **Wygladzenie masek rozbilo obraz** -- patrz nizej.

## Kompozycja: co zadzialalo i co nie

### Zadzialalo: SDXL + grounding + szumowy bootstrap

Pierwsza konfiguracja, w ktorej sa naraz oba podmioty, atrybut z promptu i spojna scena
(`xl_b3000_boot`, scena 3.1, bootstrap 4 i 8). Trzy skladniki, kazdy konieczny:

* **wlasciwy backbone** -- ich figury 3 i 12 sa SDXL-owe, a pol dnia stroilem to na SD-1.5;
* **wlaczony nasz grounding** -- byl wylaczony we WSZYSTKICH biegach SD-1.5;
* **szumowy bootstrap** zamiast stalego tla.

Ta trzecia poprawka byla trzecia proba. Dwie pierwsze byly bledne i obie widac na obrazach:
stala szarosc 0.5 zostawiala plyte w ksztalcie sumy pudelek, losowe kolory zamienily ja na
kolorowe prostokaty. Przyczyna nie byla powtarzalnosc koloru, tylko to, ze **kazdy staly obraz
ma skladowa stala**, ktora przecieka do wnetrza ramki przez globalne pole recepcyjne UNetu.
`add_noise(zeros, eps, t)` ma wartosc oczekiwana zero.

### Nie zadzialalo: powyzej dwoch regionow

| scena | regionow | wynik |
|---|---|---|
| 3.1 | 2 | dziala |
| 3.4 | 3 | trzeci podmiot to bezksztaltna plama |
| 3.3 | 3 | pionowe pasy dokladnie na granicach pudelek, brak jednego podmiotu |
| 3.2 | 4 | pies jako latajaca glowa ucieta ramka |

Pasy w 3.3 sa diagnostyczne: trzy pudelka leza obok siebie w osi x, a pasy pojawiaja sie na
ich stykach. Czyli problemem przestalo byc tlo, a stalo sie twarde sklejanie masek.

### Wygladzenie masek: poprawka, ktora najpierw rozbila obraz

Dodane wygladzenie granic plus normalizacja przez `max(sum, 1)` (dotad `clamp(sum, 0, 1)`
chronil WYLACZNIE czlon tla, a czlony regionow sumowaly sie bez ograniczenia).

Pierwsza wersja rozbila obraz: przy feather 2 wnetrza pudelek renderowaly sie poprawnie,
ale strefa przejscia i tlo byly czystym szumem; przy feather 4 caly obraz. Przyczyna:
ta sama miekka maska szla do mieszania WEJSCIA bootstrapu, wiec region widzial w strefie
przejscia pol prawdziwego latentu i pol swiezego szumu -- co nie jest poprawnym x_t dla
zadnego t. Naprawione (0a86174): dwa komplety masek, twardy do bootstrapu, wygladzony
wylacznie do scalania predykcji. Biegi 3182287 i 3182288 sprawdzaja feather 1.5 i 3.0.

## Kompozycja, runda druga: dwie moje bledne diagnozy i to, co naprawde widac

### Wygladzenie masek: poprawka wymagala dwoch podejsc

Pierwsza proba (0a86174) dawala twarda maske tylko do mieszania WEJSCIA bootstrapu. Nie
wystarczylo -- obraz nadal rozpadal sie w szum. Sprawdzone porzadnie zamiast zgadywania:

1. manifesty potwierdzily, ze szum jest na `0a86174`, a dobry obraz na `d2c624c`, przy
   identycznych alpha, bootstrap i rozdzielczosci -- czyli winna jest moja zmiana;
2. wagi scalania policzone LICZBOWO na zabawkowym przykladzie: sumuja sie do 1 wszedzie
   (bf16 daje +-0.008), wiec normalizacja jest poprawna.

Prawdziwa przyczyna to interakcja z bootstrapem: przy bootstrapie region widzi poza swoja
ramka czysty szum, wiec jego predykcja jest sensowna WYLACZNIE w jej wnetrzu. Wygladzona
maska scalajaca siega poza ramke i dokladala tam predykcje policzona na szumie z waga rzedu
polowy. Przy sigma 3 `wsum` nie dochodzi nigdzie do 1 (maks 0.76), wiec zanieczyszczenie
wchodzilo w cale plotno. Naprawione w eb50956: w fazie bootstrapu scalamy maskami TWARDYMI,
wygladzone wchodza dopiero po niej.

### Druga bledna diagnoza: "pasy na szwach"

Pisalem, ze pionowe srebrne pasy w scenie 3.3 to artefakt twardego sklejania masek. **Nie sa.**
Zostaja przy feather 1.5 i 3.0 tak samo jak bez wygladzania. Po przeczytaniu manifestu sceny
widac, czym sa: ITP brzmi "a football field with audience background", wiec to **kolumny
stadionu domalowane przez galaz globalna** tam, gdzie region nie wyprodukowal podmiotu.
Scena, nie usterka.

### Co naprawde widac przy trzech regionach

Scena 3.3 prosi o psa (lewy), misia (srodek) i kota z medalem (prawy).

* miś i kot renderuja sie poprawnie, **kot ma zloty medal** -- atrybut z promptu sie zwiazal;
* pies w lewym pudelku nie powstaje;
* drugie ziarno pokazuje, ze to nie jest "brak podmiotu", tylko **slabosc**: mis
  w NAJSZERSZYM pudelku (0.353 szerokosci) wychodzi ostro, a dwa wezsze (~0.27) daja szare,
  wyprane zjawy;
* caly dolny pas, czyli suma pudelek, jest wyprany wobec gornej polowy obrazu.

To przesuwa podejrzenie z granic masek na to, ze **dolny pas nigdy nie dostaje dopracowania
od galezi globalnej** -- regiony pracuja przez wszystkie 50 krokow (`regional_steps=None`).
Biegi 3182677 i 3182678 sprawdzaja obciecie regionow na kroku 25 i 35.

### Stan, ktory juz mamy i ktory jest publikowalny

Scena 3.1 (dwa regiony, SDXL, grounding, szumowy bootstrap): oba podmioty, czarodziejski
kapelusz, spojna scena z zamkiem, bez szwow. To jest pierwsza konfiguracja w calej serii,
w ktorej wszystkie trzy warunki sa spelnione naraz.

## Venv CIDM na Athenie

Druga proba (3182286) po naprawie detekcji klastra: 5:56 zamiast 1:08, torch 2.6.0+cu124,
i **wszystkie piec ich modulow importuje sie OK**, lacznie z `lib.data.lora_dataset`, ktory
za pierwszym razem padal na braku torchvision. Gotowe do puszczenia lancucha CIDM-50.

## Kompozycja: konfiguracja, ktora dziala

Po piciu osiach sprawdzonych niezaleznie na scenie 3.3 (trzy regiony: pies, mis, kot z medalem):

| konfiguracja | podmioty | artefakty |
|---|---|---|
| bootstrap 8 | 2 z 3 | **kolumny** na sumie pudelek |
| bootstrap 0 | 2 z 3 | brak |
| bootstrap 8 + skala 0.7 | 2 z 3 | panele |
| bootstrap 8 + kappa 2.5 | pies wrocil, kot padl | panele |
| **bootstrap 0 + kappa 2.5** | **3 z 3** | duplikaty przy krawedziach |
| **bootstrap 0 + kappa 4.0** | **3 z 3** | prawie czysto |

Zwycieska: `--scale 0.4 --ground 1 --kappa 4.0 --ground_sched 0.5 --bootstrap 0 --feather 1.5`
na `X_sdxl_b3000`.

### Czym byly te "kolumny", i dlaczego dwa razy je zle zdiagnozowalem

Najpierw napisalem, ze to artefakt twardego sklejania masek. Nie byl -- zostawaly przy
feather 1.5 i 3.0. Potem, po przeczytaniu ITP ("a football field with audience background"),
ze to architektura stadionu domalowana przez galaz globalna. Tez nie.

Rozstrzygnal pomiar, nie rozumowanie: **przy `bootstrap 0` kolumn nie ma w ogole**, przy 8 sa.
To jest twarda nieciaglosc maski bootstrapu, ktora model rozwiazuje w prawdopodobnie
wygladajace obiekty -- walce, panele, billboardy, zaleznie od sceny. Kontrola byla dostepna
przez caly czas w danych, ktore juz mialem.

### Dlaczego kappa, a nie skala

Silniejszy adapter (skala 0.7) NIE wydobyl brakujacego psa -- mis i kot wyszly ostrzej, lewy
region dalej pusty. Kappa przy tej samej skali go wydobyla. Czyli brakujacy podmiot to nie
jest kwestia sily adaptera, tylko tego, ze nic nie zmusza masy konceptu do wejscia w ramke.
Robi to galaz groundingu -- NASZ uczony mechanizm -- i robi to bez wprowadzania nieciaglosci,
inaczej niz bootstrap z ich potoku.

To jest wynik wart zdania w pracy: ich rownanie 5 potrzebuje niezapisanego triku
(bootstrapu), ktory zostawia slad na obrazie, a nasz grounding zalatwia to samo czysto.

### Co zostaje

Duplikaty przy krawedziach (male przy kappa 4.0), niewidoczny medal na kocie, podmioty duze
wzgledem ramek. Biegi 3182744 i 3182745 licza pelne JEDENASCIE scen (3.1-3.5, 12.1-12.6),
wiec bedzie widac, czy to sie utrzymuje poza scena 3.3.

## Trening p022_v2: pierwszy kawalek gotowy

`ch-p022v2-t10_29-ath` COMPLETED w 2:23:36. Log potwierdza, ze wznowienie zadzialalo jak
zaprojektowane: `basis_count=10, kotwic=10, start od zadania 10` -- czyli baza ortogonalna
wrocila z checkpointu, a kotwice regularyzatora zostaly odtworzone (tych w checkpointcie nie
ma). Kawalek zatrzymal sie na zadaniu 29 i drugi juz liczy.

Bliizniaki na Heliosie (22916846, 22916974) anulowane.

### Pelne jedenascie scen przy kappa 4.0, bootstrap 0

| scena | regionow | wynik |
|---|---|---|
| 3.3 | 3 | wszystkie trzy ostre, scena spojna, bez artefaktow |
| 3.2 | 4 | trzy wyrazne, czwarty niepewny; wnetrze spojne |
| 12.4 | 4 | kot dobry, pies plywajacy bez tulowia, po prawej futrzana masa bez glowy |
| 12.6 | 2 | scena ladna, ale w prawym pudelku DWA koty zamiast jednego |

Czyli po usunieciu nieciaglosci zostal jeden systematyczny problem i nie jest nim ani szew,
ani brak podmiotu: **liczba podmiotow**. Region renderuje swoj koncept wiecej niz raz, a przy
czterech regionach czesc wychodzi zdeformowana.

To jest dokladnie ograniczenie zapisane w tym repo od sierpnia: "regionalna uwaga trasuje
tresc, ale nie wymusza liczby podmiotow". Bootstrap to wymuszal -- i po to powstal -- tylko
placil nieciagloscia, ktora zostawiala walce i panele.

Biegi 3183237 i 3183238: bootstrap 2 i 4 przy kappa 4.0, czyli mala dawka ponizej progu
nieciaglosci plus mocny grounding. Jesli to zadziala, mamy oba mechanizmy naraz przy koszcie
zadnego z nich osobno.

### Kappa nie jest monotoniczna -- zalezy od liczby regionow

Porownanie tej samej sceny przy dwoch wzmocnieniach groundingu, bootstrap 0 w obu:

| scena | regionow | kappa 2.5 | kappa 4.0 |
|---|---|---|---|
| 12.6 | 2 | **czysto**: jeden pies, jeden kot, bez duplikatow | dwa koty w prawym pudelku |
| 12.3 | 2 | **czysto**: pies i mis, nocna uliczka, bez szwow | (nie sprawdzone) |
| 3.3 | 3 | trzy podmioty, duplikaty przy krawedziach | trzy podmioty, prawie czysto |

Czyli kappa ma OPTIMUM, ktore rosnie z liczba (i waskoscia) regionow: przy dwoch szerokich
pudelkach 2.5 wystarcza, a 4.0 zaczyna mnozyc podmioty; przy trzech waskich jest odwrotnie.
Przepchniecie ponad optimum objawia sie duplikatami, nie brakiem podmiotu.

To jest sensowne mechanistycznie -- grounding pcha mase konceptu do ramki, wiec za mocny
pcha jej tyle, ze starcza na drugi egzemplarz -- i jest zgodne z sierpniowym werdyktem
z tego repo, ze uwaga regionalna "nie wymusza liczby podmiotow".

**Obrazki na poziomie publikacyjnym juz mamy**: 12.6 i 12.3 przy kappa 2.5 (dwa regiony)
oraz 3.3 przy kappa 4.0 (trzy regiony). Zaden nie ma szwow, paneli ani kolumn.

### Mala dawka bootstrapu NIE tlumi duplikatow

Hipoteza byla taka, ze bootstrap 2-4 wymusi pojedynczy egzemplarz, nie wprowadzajac
nieciaglosci. Obalona pomiarem, scena 12.6 (dwa regiony):

| konfiguracja | wynik |
|---|---|
| bootstrap 0, kappa 2.5 | **czysto**: jeden pies, jeden kot |
| bootstrap 0, kappa 4.0 | dwa koty |
| bootstrap 4, kappa 4.0 | **dwa psy I dwa koty** -- najgorzej |

Czyli na tej scenie bootstrap i kappa OBA nasilaja duplikacje, a nie tylko kappa. Najlepsza
konfiguracja ogolna zostaje `bootstrap 0, kappa 2.5, feather 1.5`.

Nastepna i ostatnia os bez zmian w kodzie: obciecie przebiegow regionalnych (`regional_steps`)
w DOBREJ konfiguracji -- wczesniej testowane tylko przy bootstrapie 8 i domyslnym kappa, wiec
bez wartosci. Jesli koncowe kroki beda czysto globalne, drugi egzemplarz ma szanse zostac
wchloniety. Bieg z `--regional_steps 35`.

## Kompozycja: konfiguracja koncowa

`X_sdxl_b3000 --scale 0.4 --ground 1 --kappa 2.5 --ground_sched 0.5 --bootstrap 0
--feather 1.5 --regional_steps 35`

| scena | regionow | wynik |
|---|---|---|
| 12.6 | 2 | **czysto**: jeden pies, jeden kot, spojna scena z zamkiem |
| 12.3 | 2 | **czysto**: pies i mis, nocna uliczka |
| 3.3 | 3 | wszystkie trzy ostre; duplikaty przy krawedziach |
| 12.4 | 4 | **wszystkie cztery obecne i ostre**; gorny pies bez tulowia |

Zaden obraz nie ma juz szwow, paneli ani kolumn. Przy czterech regionach wczesniej (kappa 4,
bootstrap 0) byla tam bezglowa futrzana masa -- `regional_steps 35` to naprawilo, bo koncowe
pietnascie krokow idzie czysto globalnie i domyka sylwetki.

Co zostaje: duplikaty przy krawedziach na czesci scen i pojedyncze uciecia sylwetki. To jest
jakosciowo inny poziom niz stan wyjsciowy (szara plyta w ksztalcie sumy pudelek).

### Sciezka, ktora do tego doprowadzila -- i trzy bledne diagnozy po drodze

1. "szara plyta to blad maskowania" -> NIE, to `bootstrap_steps=15` ze stalym szarym tlem;
2. "losowy kolor to naprawi" -> NIE, kazdy staly obraz ma skladowa stala; naprawilo dopiero
   tlo szumowe `add_noise(zeros, eps, t)`;
3. "pasy na stykach to szwy masek" -> NIE, to nieciaglosc maski bootstrapu; kontrola:
   przy `bootstrap 0` znikaja calkowicie;
4. "mala dawka bootstrapu stlumi duplikaty" -> NIE, nasila je (12.6: b4k4 dal dwa psy i dwa koty).

Kazda z tych czterech zostala obalona POMIAREM, nie rozumowaniem -- i w kazdej kontrola byla
juz w danych, ktore mialem.

## Trzy ramiona ablacji: TIMEOUT, ale trening ocalal

`t10_clipkey`, `t10_learnv`, `t10_noreg` padly na limicie 3 h (moj szacunek 1.6 h byl liczony
dla GH200, A100 jest wolniejszy). Ale log pokazuje, ze TRENING sie skonczyl -- wszystkie
dziesiec checkpointow i `hyper.pt` sa na miejscu, timeout zlapal je w fazie generacji krzywej.

Zamiast powtarzac trening puszczone same macierze (3183942/43/44) na istniejacych
checkpointach. Przy dziesieciozadaniowym configu macierz wychodzi 55-komorkowa z konstrukcji,
wiec `GEN_EXTRA` nie jest potrzebne -- to jest dokladnie ta pulapka, ktora przy T=50 kosztowala
18 GPU-h, i tu znika sama.

### Ostatnia os: alpha, czyli ile glosu ma galaz globalna wewnatrz pudelka

Po domknieciu obecnosci podmiotow zostaly duplikaty przy krawedziach. Wszystkie osi juz
sprawdzone (kappa, bootstrap, feather, regional_steps, skala) albo ich nie ruszaja, albo je
nasilaja. Zostala jedna nietkniete: `alpha`.

Przy alpha=0.1 galaz globalna ma wewnatrz pudelka 10% wagi, a to ONA niesie informacje, ze
w scenie jest jeden pies, a nie dwa -- prompt globalny nie mowi o duplikatach. Podniesienie
powinno tlumic nadmiarowe egzemplarze kosztem czesci tozsamosci.

Uwaga do zaraportowania w pracy, jesli to wejdzie: **alpha=0.1 jest ICH parametrem z rownania
5**, wiec zmiana jest odstepstwem od odtwarzanego protokolu i trzeba ja nazwac wprost. Biegi
3184181 (0.2) i 3184182 (0.3).

## Status nocny, 01:00

Dziewiec zadan biegnie, kolejka pusta. Trzy macierze ablacji (3183942/43/44) ruszyly po
naprawie TIMEOUTu. CIDM-50 blok 11-25 ma godzine do konca, drugi kawalek naszego treningu
cztery godziny. Trzy ewaluacje SDXL koncza sie w ciagu godziny.

Bliizniak `ch-sdxl-noout-eval` biegnie rownolegle na Heliosie (3:12) i Athenie (4:57, godzina
do konca) -- zdejme Heliosowy, gdy Athena skonczy.

### Alpha nie tlumi duplikatow -- zjada tozsamosc

Scena 3.3 przy alpha 0.2 (reszta jak w konfiguracji koncowej): pies i mis w porzadku, ale
prawy region rozpadl sie w szara pierzasta mase. Duplikatow nie ubylo. Czyli podniesienie
wagi galezi globalnej wewnatrz pudelka nie usuwa nadmiarowych egzemplarzy, tylko oslabia
koncept.

**Zostaje ich `alpha=0.1`** -- co jest dobra wiadomoscia dla pracy: o jedno odstepstwo od
odtwarzanego protokolu mniej. Odstepstwa, ktore zostaja i ktore trzeba nazwac: bootstrap
wylaczony (u nich wlaczony, niezapisany w artykule) i nasz grounding wlaczony.

Tym samym **osi bez zmian w kodzie nie zostalo**. Konfiguracja koncowa stoi:
`--scale 0.4 --ground 1 --kappa 2.5 --ground_sched 0.5 --bootstrap 0 --feather 1.5
--regional_steps 35 --alpha 0.1`, a resztkowa wada to duplikaty przy krawedziach na czesci
scen. Dalsze strojenie przestalo sie oplacac.

## Trening p022_v2 SKONCZONY

`ch-p022v2-t30_49-ath` COMPLETED w 2:47:40, log konczy sie na `done task 49:cc101_toy_pikachu1`
i `DONE`. Komplet: 41 checkpointow (task09 skopiowany + 10-49 policzone), `hyper.pt`,
`fresh/`, `forgetting/`, `final/`.

Czyli caly strumien piecdziesieciu zadan na POPRAWIONYCH podpisach jest policzony, w dwoch
kawalkach po 2:24 i 2:48 zamiast jednego osmiogodzinnego -- i bez liczenia zadan 1-10, bo to
koncepty CIFC, ktorych dane sie nie zmienily.

Puszczone ewaluacje: 3184416 (`all50` przy skalach 0.45 i 0.6) i 3184417 (dryf dW na 50
zadaniach). Po nich sekcja o skalowaniu bedzie miala komplet liczb na nowych danych.

## Dwie awarie ewaluacji, obie natychmiastowe

`ch-p022v2-all50` padl po 16 s, `ch-p022v2-drift` po 1 s. Dwie rozne przyczyny, obie
srodowiskowe, nie obliczeniowe:

1. **`_dw_drift.py` nie istnial na Athenie.** To skrypt poza gitem, wyslany kiedys scp tylko
   na Heliosa. Przy przenoszeniu projektu przeniosłem runnery `sbatch_*` i kod z gita, ale nie
   pomocnicze `_*.py`, ktore zyja poza repo. Wyslany.
2. **`outputs/sweep/p022_v2/config.yaml` nie istnieje.** Config kopiuje do katalogu wyjsciowego
   runner SWEEPU, a p022_v2 trenowalismy przez `sbatch_cl.sh`, ktory tego nie robi. Poprawione
   przez wskazanie `configs/phaseT/T50_v2.yaml` wprost.

To jest ta sama klasa bledu co wczorajszy brak `backgrounds_sdxl`: przeniesienie projektu na
drugi klaster zabiera kod i dane, ale nie to, co powstalo po drodze poza gitem.

## CIDM-50 blok 11-25 skonczony

3:50:30, 25 checkpointow na dysku. Puszczony blok 26-38 (3184652, limit 5 h -- poprzedni
blok pokazal, ze 4 h byloby za ciasne na A100). Bliizniak 22882173 na Heliosie zdjety.

## Noc 19.09, obchod ~02:30-05:00

### Pusty `by_lag`: czemu job "udal sie" i nic nie policzyl

`ch-p022v2-drift` (3184647) skonczyl sie z kodem 0 w 37 s, a `dw_drift50.json` mial `by_lag`
z zerem kluczy. Log mowil dokladnie, co sie stalo, jednym wierszem:

    [drift] brak outputs/sweep/p022_v2/ckpts/hyper_after_task00.pt, przerywam na k=0

Przyczyna nie jest w skrypcie, tylko w tym, ze **strumien byl wznawiany miedzy klastrami**.
p022_v2 wystartowal od zadania 10 z checkpointu przeniesionego na Athene, wiec tam leza
zadania 09-49, a 00-08 zostaly na Heliosie. `_dw_drift.py` potrzebuje dW_j(phi_j) dla kazdego j,
wiec przerwal na pierwszym braku. Zero par, pusty slownik, kod wyjscia 0 -- awaria bez sygnalu.
To jest ta sama klasa bledu, co "brakujace dane, nie blad kodu" z `kd128wd0`.

**Czego nie zrobilem: nie przenosilem checkpointow.** Zmierzylem najpierw: jeden plik 104 MB
szedl z Heliosa 6m43s (260 kB/s). Dziewiec plikow to okolo dwoch godzin w dol i drugie dwie
w gore -- caly ten obchod na transfer.

Zamiast tego: dryf liczy sie z macierzy r x r, a z checkpointu j potrzebny jest **jeden**
koncept, ten wlasnie nauczony. To 1/50 zawartosci. Nowy `scripts/_dw_drift_refs.py` zapisuje
dokladnie te czynniki, `_dw_drift.py --ref_factors` je doklada, a brakujacy checkpoint
**pomija zamiast przerywac** (brak k oznacza tylko brak par konczacych sie na k, nie koniec
analizy). Ekstrakcja na Heliosie: 16 s, **16,3 MB**. Transfer: dwie minuty zamiast czterech
godzin.

**Kontrola wbudowana w zakres.** Checkpoint 09 istnieje po obu stronach, wiec jest wyciagany
mimo ze Athena go ma, a `_dw_drift.py` porownuje wersje z pliku z policzona u siebie i drukuje
wzgledna roznice. Gdyby jakis parametr nie siedzial w checkpointcie i zostawal z losowej
inicjalizacji, przenoszenie referencji miedzy klastrami byloby niewazne -- i widac by to bylo
w tej jednej liczbie. Bez tego porownania caly skrot bylby zalozeniem, nie pomiarem.

Wynik pod `dw_drift50_full.json` (job 3184836), nowa nazwa zamiast nadpisania pustego pliku.

### SDXL: trzy ramiona TIMEOUT na Atenie, ale odpowiedz i tak jest

3182118/3182119/3182120 padly na limicie 6:00 -- limit byl mierzony na GH200, a A100 jest
wolniejszy. Zdazyla sie tylko skala 0.4. Bliizniaki na Heliosie licza pelny przemiat i tam
limit pasuje, wiec **nic nie zglaszam ponownie**; czekam na 22516080/22516082/22517539.

Skala 0.4, srednia po dziesieciu konceptach:

| ramie | CLIP-T | CLIP-I | DINO |
|---|---|---|---|
| **b3000 (nasze)** | **75.54** | **80.13** | **0.593** |
| s1600 | 67.67 | 75.66 | 0.524 |
| noout | 76.03 | 72.15 | 0.430 |
| lr3e4 | 51.56 | 58.20 | 0.057 |

Zadne z trzech nie bije b3000. `noout` jest tu warty zdania w pracy: wyrzucenie `to_out`
podnosi CLIP-T o pol punktu i zabiera **0,16 DINO**, czyli to samo, co na SD-1.5 -- twierdzenie
o czterech koniecznych projekcjach przenosi sie na SDXL, a nie tylko na backbone, na ktorym
je wymyslilismy.

### lr3e4 to nie blad pomiaru, to rozpad treningu

DINO 0,057 przy CLIP-I 0,58: obrazy sa z wlasciwej klasy i nie maja **zadnej** tozsamosci.
Trzy powody, dla ktorych to nie jest usterka sciezki liczenia metryk:

1. te same skrypty daja 0,43 i 0,52 dla pozostalych ramion;
2. rozpad jest rowny po wszystkich dziesieciu konceptach, lacznie z **ostatnio uczonym**
   (ink painting 0,164, drawing 0,192) -- zapominanie wygladaloby inaczej, ostatni byl by caly;
3. log treningu (22516061) pokazuje skoki normy gradientu do 43379, 8,4 i 1,3 w obrebie
   jednego zadania.

Przy lr 3e-4 hipersiec sie rozjezdza. Wartosc dla pracy jest negatywna i taka zostaje: gorna
granica kroku uczenia na SDXL lezy ponizej 3e-4.

### Kompozycja: `b0k4.0` nadal najlepsze, i dlaczego nie wyciagam wniosku z `a0.2`/`a0.3`

Obejrzalem sceny 3.2, 3.3, 3.4, 12.3 z `b0k4.0` oraz 3.3 z `a0.2` i `a0.3`.

`b0k4.0` trzyma poziom poza scena, na ktorej byl strojony: 3.3 ma psa, misia i kota naraz,
ostrych; 3.4 ma wszystkie trzy w spojnej swiatyni; 12.3 ma psa i misia w zaulku; **3.2 ma
pelnego psa zamiast uciętej latajacej glowy**, ktora byla tam poprzednio. To jest ta poprawka,
o ktora chodzilo.

`a0.2` i `a0.3` wygladaja wyraznie gorzej -- w `a0.3` znika pies, w `a0.2` rozpadaja sie dwa
podmioty. **Ale nie wolno z tego wyciagnac wniosku o alfie.** Sprawdzilem ich `run-info.txt`:
roznia sie od `b0k4.0` na TRZECH osiach naraz (kappa 2,5 zamiast 4,0, `regional_steps` 35
zamiast braku, alfa 0,2/0,3 zamiast 0,1). Wiedzac, ze kappa 2,5 juz raz zgubila podmiot,
a 4,0 nie, najprostszym wyjasnieniem jest spadek kappy, nie alfa. Zapisuje to jako biegi
nierozstrzygajace, nie jako wynik o alfie.

Zostalo jedno realne pudlo: **scena 3.2 gubi V7**, psa w lewym dolnym rogu, jedyna z czterema
regionami. Kappa jest jedyna osia, ktora w tej serii przywracala brakujace podmioty, wiec
bracketuje ja w gore przy wszystkim innym bez zmian: `xl_b0k6.0` (3184834) i `xl_b0k8.0`
(3184835), po jedenascie scen.

### Dryf policzony: mechanizm widac wprost, i przezywa poprawke podpisow

Job 3184836, `dw_drift50_full.json`. Kontrola zgodnosci miedzy klastrami wypadla
**1,0e-4** -- referencja dla zadania 09 wyciagnieta na Heliosie i policzona na Atenie to ta
sama wielkosc, wiec skrot z przenoszeniem czynnikow byl uprawniony, a nie tylko wygodny.

Wzgledny dryf dW starego konceptu:

| zadan od wlasnego | 1 | 5 | 10 | 20 | 30 | 40 | 49 |
|---|---|---|---|---|---|---|---|
| dryf | 1,2% | 2,6% | 3,7% | 5,4% | 7,0% | 9,5% | **11,0%** |

Po CALYM pozostalym strumieniu czterdziestu dziewieciu konceptow aktualizacja generowana dla
pierwszego konceptu odplywa o jedenascie procent. Krzywa rosnie podliniowo. To jest dokladnie
to, co regularyzator wyjscia obiecuje, zmierzone bez generowania ani jednego obrazu.

**Porownanie ze strumieniem sprzed poprawki podpisow** (p022, stare captiony cc101):

| lag | 1 | 5 | 10 | 20 | 30 | 40 | 49 |
|---|---|---|---|---|---|---|---|
| v1 | 0,0114 | 0,0270 | 0,0386 | 0,0562 | 0,0716 | 0,0903 | 0,1062 |
| v2 | 0,0115 | 0,0261 | 0,0365 | 0,0537 | 0,0700 | 0,0950 | 0,1101 |

Zgodnosc w granicach 3-5% na kazdym opoznieniu, przy DWOCH niezaleznie wytrenowanych sieciach.
Wazna szczegolnosc, ktora to wzmacnia: dla opoznien >= 9 oba ramiona licza na **identycznym**
zbiorze par (przy lagu L jest ich 50-L, a to nie wiecej niz 41 dostepnych w v2), wiec tam nie
ma nawet roznicy estymatora. Roznia sie tylko lagi 1-8, gdzie v2 ma 41 par zamiast 42-49.

Wniosek praktyczny: **figury dryfu nie trzeba przebudowywac.** Para GS / bez-GS zostaje na
v1, gdzie oba ramiona maja te same podpisy, czyli porownanie jest wewnetrznie spojne.
Do tekstu nadaje sie natomiast jedno zdanie, ze przetrenowany strumien odtwarza te krzywa --
bo to jest odpornosc wyniku o mechanizmie na poprawke zbioru, a nie powtorzenie tego samego
pomiaru. Zdania jeszcze NIE wstawilem: sekcja T = 50 i tak bedzie przepisywana, gdy spadna
liczby z `all50`, i lepiej napisac to razem niz zostawic pol-zaktualizowany paragraf.

### noout: pelny przemiat skal z Heliosa

22517539 COMPLETED w 5:06:47, wiec ramie ma komplet, a nie samo s04:

| skala | CLIP-T | CLIP-I | DINO |
|---|---|---|---|
| 0,40 | 76,02 | 72,07 | **0,4300** |
| 0,30 | 76,74 | 71,16 | 0,4065 |
| 0,25 | 76,99 | 70,66 | 0,3911 |

DINO spada monotonicznie ze skala, wiec najlepszy punkt ramienia to 0,430 -- bierzemy dla
niego najkorzystniejsza skale i nadal jest o **0,163 DINO** ponizej b3000. Przy okazji wyszla
darmowa kontrola miedzy klastrami: Athena dala na s04 0,4302, Helios 0,4300.

### CIDM blok 26-38 nie zmiesci sie w limicie -- policzone z tempa, nie zgadniete

Znaczniki czasu z loga 3184652:

    task 26  02:03:18
    task 27  02:28:44   (25,4 min)
    task 28  02:59:12   (30,5 min)

Okolo 28 min na zadanie, czyli trzynascie zadan to ~6,1 h przy limicie 5 h. Bieg wystartowal
02:03, wiec sciana wypada ~07:03, gdzieś na zadaniu 36-37. Poprzedni blok 11-25 szedl 15,3 min
na zadanie, wiec to jest spowolnienie prawie dwukrotne -- najpewniej dzielenie wezla t0010.

Czego NIE robie: limitu nie da sie podniesc (uzytkownik moze go tylko obnizyc), a anulowania
i zgloszenia od nowa nie robie bez zgody. Runner bierze FIRST i LAST jawnie, nie ma
autowznawiania, wiec nie da sie tez sensownie przygotowac zadania zaleznego z gory --
nie znam punktu zatrzymania na tyle dokladnie, zeby nie ryzykowac nadpisania gotowego zadania.

Plan: bieg dobiega do sciany, a obchod po 07:03 odczytuje ostatnie ukonczone zadanie z loga
i zglasza doklanie reszte. Strata to jeden obchod, czyli pol godziny.

### lr3e4: kontrola, ktora zamyka sprawe

Skoki normy gradientu same w sobie nic nie dowodza -- trzeba je z czyms zestawic. Maksimum
`gnorm` po calym treningu, oba ramiona, ten sam kod i te same dane, rozni je wylacznie krok
uczenia:

| ramie | max gnorm |
|---|---|
| b3000 (lr 1e-4) | **6,03** |
| lr3e4 (lr 3e-4) | **941 077** |

Piec rzedow wielkosci. Przy lr 1e-4 drugi w kolejnosci gnorm to 1,22, czyli rozklad jest ciasny;
przy 3e-4 cztery najwieksze to 941076, 43379, 19043 i 7723, czyli rozjazd wraca wielokrotnie,
a nie jest pojedynczym wyskokiem. To nie jest ramie, ktore wypadlo slabiej -- to jest trening,
ktory sie rozbiegl, i DINO 0,057 jest tego skutkiem, nie przyczyna.

### Kappa nie ma jednego optimum -- ma optimum ZALEZNE OD LICZBY REGIONOW (BLEDNE, patrz sprostowanie nizej)

Bracket 4,0 / 6,0 / 8,0 przy wszystkim innym bez zmian (3184834, 3184835). Spodziewalem sie
prostej krzywej z maksimum. Wyszlo cos ciekawszego i niewygodnego.

**Scena 3.2 (CZTERY regiony).** kappa 4,0 daje trzy podmioty z czterech (brak V7, psa w lewym
dolnym rogu). kappa 6,0 sklejaja oba psy w jedna mase futra i zasypuje podloge strzepami.
kappa 8,0 **przywraca V7** -- corgi z czerwona obroza, ostry -- ale w zamian gubi V1 i zamienia
V3 w zdeformowana morde kota, a tlo w "sciane futra". Dwa podmioty z czterech, czyli gorzej
niz przy 4,0.

**Scena 3.3 (TRZY regiony).** Odwrotnie. kappa 8,0 jest **najlepsze z calej serii**: pies, mis
i kot ostre, znika szary duplikat przy lewej krawedzi, ktory byl przy 4,0, a kot ma
**widoczny medal** na obrozy -- czyli atrybut z promptu, ktory przy 4,0 sie nie wiazal. To jest
dokladnie ta klasa usterki, o ktora byla pretensja ("nie sa ubrani tak jak w prompcie").

Czyli kappa nie jest zepsuta powyzej 4,0 -- ona po prostu **skaluje sie z liczba regionow
w druga strone, niz zakladalem**. Im wiecej pudelek, tym mniej sily na pudelko mozna dac,
zanim sasiadujace ograniczenia zaczna ze soba walczyc. Przy czterech pudelkach, ktore
kafelkuja duza czesc plotna, mocny grounding zmusza koncept do wypelnienia ramki, ktorej
nie umie wypelnic naturalnie -- i stad "sciana futra" zamiast psa na lozku.

Rozklad zestawu: dwie sceny maja cztery regiony (3.2, 12.4), szesc ma trzy, trzy maja dwa.
Czyli kappa 8,0 poprawia wiekszosc, a psuje dwie. Wyboru nie robie z dwoch scen -- czekam
na komplet jedenastu i przegladam calosc, bo to jest dokladnie ta sytuacja, w ktorej wynik
z jednej sceny prowadzi w zla strone (juz raz tak bylo z "kolumnami").

### Sprostowanie: wniosek o kappie byl bledny, bo znowu wyciagnalem go z dwoch scen

Napisalem godzine temu, ze kappa ma optimum zalezne od liczby regionow, bo 8,0 psulo scene 3.2
(cztery regiony) i poprawialo 3.3 (trzy). Doczekalem kompletu jedenastu scen i to sie nie
utrzymuje. Przy kappie 8,0:

* **3.1 ma DWA regiony i traci oba podmioty** -- zostaje sam zamek i platy bialego futra;
* 3.4 (trzy regiony) gubi psa, srodek to szara kurtyna;
* 12.4 (cztery regiony) daje dwa podmioty i dwa "platy skory".

Czyli 3.3 byla wyjatkiem, a nie reprezentantem klasy scen trzyregionowych. Charakterystyczna
awaria przy 8,0 to **"plat futra"**: tekstura konceptu rozmazana tak, zeby wypelnic ramke,
zamiast obiektu w ramce. To jest przegrounding, zwyczajny i niezalezny od liczby pudelek.

Trzeci raz w tej serii zbudowalem wyjasnienie na jednej scenie i trzeci raz komplet je obalil
(wczesniej: "plyta to blad maski", "pasy to szwy"). Regula, ktora z tego zostaje: **przy
kompozycji zaden wniosek nie liczy sie przed obejrzeniem wszystkich jedenastu scen.**

Kappa 4,0 zostaje. Bracket 4/6/8 jest zamkniety: 4,0 jest w maksimum albo bardzo blisko.

### Zwyciezca: bootstrap 2 przy kappie 4,0

Zanim cokolwiek puscilem, sprawdzilem, co juz jest policzone -- i okazalo sie, ze `xl_b2k4`
i `xl_b4k4` (3183237, 3183238) to doklanie bootstrap 2 i 4 przy kappie 4,0, komplet jedenastu
scen, gotowe od godzin. Nie trzeba bylo liczyc niczego, trzeba bylo popatrzec.

`b2k4` jest najlepsza konfiguracja w calej serii:

| scena | regionow | b0k4.0 (dotad najlepsze) | **b2k4** |
|---|---|---|---|
| 3.2 | 4 | 3 z 4 (brak V7) | **4 z 4** |
| 12.4 | 4 | 2 podmioty + plat futra | **4 z 4** |
| 3.3 | 3 | 3 z 3, szary duplikat, brak medalu | **3 z 3, kot ma czerwona obroze ze zlotym medalem** |
| 3.4 | 3 | 3 z 3 | **3 z 3, ostrzejsze, bez szwow** |

Zadnych kolumn, paneli ani szwow na granicach pudelek.

**Czego to uczy o bootstrapie.** Zerowalismy go, bo przy 8 zostawial kolumny, i przyjalem, ze
jest szkodliwy. Nie jest -- wlasciwa wartosc jest **mala, ale niezerowa**. Kolumny bralu sie
z DLUGIEJ fazy twardej maski; dwa kroki wystarcza, zeby kazdy region sie zawiazal, i sa za
krotkie, zeby zostawic widoczna nieciaglosc. Przy okazji to bootstrap odpowiada za wiazanie
atrybutow: kapelusz w 3.1 mielismy przy bootstrapie 4-8, stracilismy przy 0, a medal w 3.3
wraca przy 2.

To zmienia tez zdanie do pracy. Nie jest tak, ze nasz grounding zastepuje ich bootstrap --
jest tak, ze **ich bootstrap przy ich wartosci zostawia slad na obrazie, a w polaczeniu
z naszym groundingiem wystarcza go dwa kroki zamiast pietnastu** (ich domyslna to 15).

Bootstrap 4 tez daje komplet w 3.2, ale dubluje psa na gorze, wiec 2 jest lepsze od 4.

**Co zostalo:** duplikaty przy krawedziach -- w 3.1 pies ma dwie glowy, w 3.3 jest cien
drugiego misia przy prawej krawedzi. Dwie osie po jednej zmianie od `b2k4`: bootstrap 1
(3184839) i feather 2,5 (3184840).

### b2k4, komplet jedenastu scen (ziarno 0) -- tabela, nie wrazenie

Zgodnie z regula, ktora przed chwila zapisalem: zanim nazwe `b2k4` zwyciezca, przegladam
wszystkie jedenascie, nie cztery wygodne.

| scena | regionow | wynik |
|---|---|---|
| 3.2 | 4 | **4/4, czysto** |
| 12.4 | 4 | **4/4** |
| 3.3 | 3 | **3/3 + medal na obrozy**, slaby cien misia przy krawedzi |
| 3.4 | 3 | **3/3, czysto** |
| 12.5 | 3 | **3/3, czysto** -- plecak, mis i kaczka, czyli koncepty NIEZWIERZECE |
| 12.1 | 3 | 3/3, duplikat psa przy lewej krawedzi |
| 3.5 | 3 | 2/3 + artefakt "rozdartego papieru" |
| 12.2 | 2 | **2/2, czysto** |
| 12.3 | 2 | **2/2, czysto** |
| 3.1 | 2 | 2/2, pies ma dwie glowy |
| 12.6 | 2 | 2/2, pies ma dwie glowy |

Podsumowanie: **10 z 11 scen ma komplet podmiotow**, 5 jest calkowicie czystych, i -- co jest
tu najwazniejsze -- **zadna nie ma kolumn, paneli ani szwow na granicach pudelek**. Ta klasa
usterki, ktora zzerala poprzednie dwa dni, zniknela.

Warta odnotowania jest 12.5: plecak, mis i gumowa kaczka. Trzy koncepty niezwierzece,
wszystkie trzy poprawne. Dotad caly debugging kręcil sie wokol psow i kotow, wiec latwo bylo
przeoczyc, ze reszta klas tez musi dzialac.

Dominujaca resztka to **duplikaty przy krawedziach** (3.1, 12.1, 12.6 i cien w 3.3) oraz jeden
nieudany region w 3.5. Oba zgloszone ramiona (bootstrap 1, feather 2,5) celuja doklanie w to.

## Trzy ramiona T = 10: nowe liczby, i falszywy alarm po drodze

`t10_clipkey`, `t10_learnv`, `t10_noreg` (3183942-44) COMPLETED, po 3:19 kazde, pelne macierze
55-komorkowe przy skali 0,45 (sprawdzone: w logu sa wszystkie checkpointy after_task00..09).

| ramie | CLIP-T | CLIP-I | DINO | zapominanie DINO |
|---|---|---|---|---|
| naglowek (nasze, 3 ziarna) | -- | -- | 0,621 | 0,0060 +- 0,0031 |
| nogs (bez Grama-Schmidta) | -- | -- | 0,632 | 0,0023 |
| sem128 (klucz z obrazow) | 74,24 | 79,23 | 0,6071 | 0,0239 |
| **clipkey (klucz z promptu kanonicznego)** | **76,06** | **79,01** | **0,6146** | **0,0009** |
| **learnv (uczone V_t)** | **76,67** | **78,71** | **0,6153** | **0,0024** |
| **noreg (beta = 0)** | **76,88** | **78,39** | **0,5892** | **0,0213** |

Dwa wnioski, oba nowe:

1. **`noreg` to bezposredni dowod, ze regularyzator dziala.** Bez niego zapominanie rosnie
   z 0,0060 do 0,0213, czyli 3,5 raza, a tozsamosc spada z 0,621 do 0,589. Dotad mielismy na to
   wylacznie porownanie z bazami (LwF, C-LoRA, EWC), czyli z innymi metodami; teraz mamy
   ablacje wlasnej metody przy wszystkim innym bez zmian.
2. **Forma klucza nie ma znaczenia, jego tresc ma.** Klucz syntetyczny (naglowek), kanoniczny
   tekst (`clipkey`) i uczony wektor (`learnv`) daja 0,621 / 0,615 / 0,615 i zapominanie
   0,0060 / 0,0009 / 0,0024 -- nierozroznialne. Dopiero zwiazanie klucza z WYGLADEM konceptu
   (`sem128`, srednie CLIP-owe osadzenie obrazkow) kosztuje: 0,607 i zapominanie 0,0239.
   To wzmacnia zdanie, ktore juz jest w pracy, dwoma dodatkowymi ramionami zamiast jednego.

### Falszywy alarm, ktory warto opisac, zeby go nie powtorzyc

Zobaczylem, ze `clipkey` daje zapominanie 0,0009, a praca pisze o 0,0239, i przez chwile
wygladalo to na sprzecznosc z opublikowana liczba. Nie jest. Praca opisuje ablacje jako
"mean CLIP **image** embedding of the concept's reference photos", a `t10_clipkey` ma
`key_prompt: canonical`, czyli klucz z **tekstu**. To sa dwa rozne mechanizmy: `key_prompt`
wybiera, jaki TEKST idzie na klucz (`canonical` / `index` / `identifier`), a osadzenia
obrazkowe wchodza zupelnie inna droga, przez blok `sem_dim` (`train_cl.py`, komentarz
"semantic block = mean CLIP IMAGE embedding"). Ramie z pracy to `p022_sem128`, nie `clipkey`.

Sprawdzilem to do konca zamiast poprzestac na prawdopodobnym wyjasnieniu: `p022_sem128`
ma zapominanie DINO **0,023893**, czyli 0,0239 co do cyfry.

Przy okazji wyszla usterka warta zapisania: w `cifc_metrics.json` tego ramienia pole
**`average_final` jest NaN**, mimo ze macierz 55 komorek jest kompletna i poprawna. Liczby
z pracy policzylem z ostatniego wiersza macierzy: CLIP-T 0,7424, CLIP-I 0,7923, DINO 0,6071 --
czyli 74,2 i 0,607, zgadza sie z tekstem. Wniosek: **praca jest poprawna, zepsute jest jedno
pole w jsonie.** Gdyby ktos kiedys czytal `average_final` zamiast macierzy, dostalby NaN
i uznal wynik za brakujacy.

## Koniec oceniania okiem: scorer kompozycji

`b1k4` i `b2k4f25` (3184839, 3184840) COMPLETED. Obejrzane:

* **feather 2,5 pogarsza** -- w 3.1 pies dalej ma dwie glowy, a kot rozmywa sie w klebek;
  w 12.6 pysk psa jest zdeformowany. Ta os jest zamknieta.
* **bootstrap 1 jest niejednoznaczny**: w 3.1 usuwa zdublowana glowe psa (czysto, jeden pies,
  jeden kot), w 3.2 trzyma 4 z 4 -- ale w 12.4 spada do 2 z 4, a w 12.6 dubluje kota.

I tu przestaje zgadywac. Roznice miedzy `b1k4` a `b2k4` ida w obie strony zaleznie od sceny,
a ogladalem **po jednym ziarnie na scene**, podczas gdy kazda ma dwa. Ocena konfiguracji
z jednego ziarna to doklanie ten sam blad, co ocena z jednej sceny -- ktory popelnilem juz
trzy razy tej nocy. Roznice sa teraz mniejsze niz rozrzut miedzy ziarnami, wiec oko nie
rozstrzygnie, ile bym nie patrzyl.

### `scripts/_compose_score.py`

Dla kazdego regionu wycinamy z obrazu jego wlasne pudelko (z manifestu) i liczymy DINO do zdjec
referencyjnych TEGO konceptu. Wszystkie trzy tryby awarii, ktore widzielismy, obnizaja te
liczbe kazdy z osobna: brak podmiotu (w ramce tlo), duplikat sasiada (nie ten koncept),
"plat futra" (tekstura zamiast obiektu).

Kontrola wbudowana: dla kazdego wycinka liczymy TEZ DINO do konceptow pozostalych regionow tej
sceny. Jesli wlasny nie jest najwyzszy, to nie jest slaba jakosc, tylko **podmiot wyladowal
w cudzej ramce** -- inna usterka, inna poprawka. Bez tej kolumny obie wygladaja tak samo.

**Wybor modelu byl pulapka i pierwszy bieg na niej padl** (3184841, FAILED w 12 s).
Napisalem scorer na `src.eval.DinoScorer`, ktory bierze dinov2; na Atenie nie ma go w cache'u,
a joby leca z `HF_HUB_OFFLINE=1`, wiec `LocalEntryNotFoundError`. Wlasciwy model to
`vit_small_patch16_224.dino` z `cifc_metrics`, z preprocessingiem skopiowanym z ich
`evaluate.py` -- i to nie jest tylko kwestia cache'u: **na nim liczone sa wszystkie DINO
w pracy**, wiec wynik kompozycji musi byc w tej samej skali, zeby dalo sie go z czymkolwiek
zestawic. Przepiete, zgloszone ponownie jako 3184842, siedem konfiguracji naraz.

## Pomiar kompozycji: prawie wszystko, co dzis mowilem okiem, bylo szumem

Scorer (3184842) przeszedl przez siedem konfiguracji, po 62 regiony kazda (11 scen x 2 ziarna
x regiony). Roznice liczone **parami** -- ten sam region, ta sama scena, to samo ziarno --
wiec rozrzut miedzy scenami sie skraca i widac wylacznie efekt konfiguracji.

| konfiguracja | srednie DINO | roznica do b2k4 | se | t | przegranych |
|---|---|---|---|---|---|
| b0k4.0 | 0,5840 | -0,0022 | 0,0178 | -0,13 | 8/62 |
| b1k4 | 0,5881 | +0,0018 | 0,0144 | +0,12 | 12/62 |
| b2k4 | 0,5863 | -- | -- | -- | 9/62 |
| b4k4 | 0,5951 | +0,0088 | 0,0188 | +0,47 | 9/62 |
| b2k4f25 | 0,6028 | +0,0165 | 0,0100 | +1,66 | 11/62 |
| **b0k6,0** | 0,3905 | **-0,1958** | 0,0337 | **-5,81** | 15/62 |
| **b0k8,0** | 0,3423 | **-0,2440** | 0,0410 | **-5,96** | 17/62 |

**Bootstrap 0, 1, 2 i 4 oraz feather 1,5 i 2,5 sa nierozroznialne.** Wszystkie piec siedzi
w przedziale 0,584-0,603 przy bledzie standardowym roznicy rzedu 0,01-0,02, czyli |t| <= 1,7.
Jedyna os z realnym efektem to kappa, i to szescioma sigmami w dol przy 6,0 i 8,0.

To obala **moj wlasny wniosek sprzed godziny**, ze "bootstrap 2 rozwiazuje problem". Widzialem
prawdziwa rzecz -- w scenie 3.2 przy bootstrapie 0 psa naprawde nie ma, a przy 2 jest -- ale
to, co `b0k4.0` traci tam, odrabia gdzie indziej (12.6: 0,7061 wobec 0,6250). W sumie po
zestawie wychodzi zero. Tak samo "feather 2,5 pogarsza": ma NAJWYZSZA srednia z calej siodemki.

Sprawdzilem tez hipoteze wezsza, ta, od ktorej zaczalem -- ze bootstrap pomaga wlasnie przy
czterech regionach:

| | sceny 4-regionowe (16 regionow) | pozostale (46) |
|---|---|---|
| b0k4.0 | 0,5662 | 0,5903 |
| b2k4 | 0,6102 (**+0,044 +- 0,040**) | 0,5779 (-0,012) |
| b4k4 | 0,5261 (-0,040) | 0,6191 (+0,029) |

Kierunek dla bootstrapu 2 jest zgodny z hipoteza, ale to 1,1 sigma, a bootstrap 4 idzie
w DRUGA strone -- czyli albo efekt jest niemonotoniczny, albo go nie ma. Przy dwoch scenach
czteroregionowych i dwoch ziarnach nie da sie tego rozstrzygnac. Zglosilem doklanie ten
kontrast z czterema razami wiecej ziaren: `xl_b0k4_s8` (3184844) i `xl_b2k4_s8` (3184845),
sceny 3.2 i 12.4, po osiem ziaren, czyli 64 regiony na konfiguracje zamiast 16.

### Liczba, ktora moze byc wynikiem do pracy

Srednie DINO wycinka regionu wynosi **0,586**, a nasze SDXL-owe DINO dla POJEDYNCZEGO konceptu
na pelnym obrazie to **0,593** (b3000, skala 0,4). Czyli podmiot w swojej ramce w zlozonej
scenie pasuje do referencji praktycznie tak samo dobrze, jak podmiot wygenerowany sam.
Zastrzezenie, ktore trzeba sprawdzic przed wpisaniem tego do pracy: wycinek jest skalowany
do 224 z mniejszego obszaru niz caly obraz, wiec efektywna rozdzielczosc podmiotu nie jest
identyczna. Ale rzad wielkosci jest wlasciwy i to jest mocniejsze niz jakakolwiek figura.

### Czego to uczy o calej nocy

Cztery razy z rzedu zbudowalem wniosek na tym, co widac na jednym lub dwoch obrazach, i cztery
razy pomiar albo komplet scen go obalil. Kompozycja ma duzy rozrzut miedzy scenami i miedzy
ziarnami, wiec oko widzi realne roznice pojedynczych obrazow i uogolnia je na konfiguracje --
a tam ich nie ma. Scorer powinien byl powstac wczoraj, przed pierwszym strojeniem.

## Rozstrzygniecie: bootstrap dziala, ale tylko przy wielu regionach i tylko na AWARIE

Osiem ziaren na scenach 3.2 i 12.4 (3184844, 3184845), po 64 regiony na konfiguracje --
cztery razy wiecej danych niz w poprzednim tescie, na dokladnie tym kontrascie.

Sama srednia dalej nie wystarcza: **+0,0511 +- 0,0264, t = 1,93**, a `b2k4` wygrywa tylko
w 33 regionach na 64, czyli jak rzut moneta. Gdybym poprzestal na sredniej, wyszlo by
"na granicy, nie wiadomo". Ale rozklad jest **dwumodalny** i to on niesie wynik:

| | ponizej 0,40 (awaria) | mediana udanych | kwartyl dolny |
|---|---|---|---|
| bootstrap 0 | **19/64 (30%)** | 0,682 | 0,347 |
| bootstrap 2 | **9/64 (14%)** | 0,653 | 0,507 |

Regiony sa sparowane (ta sama scena, ziarno i ramka), wiec test to McNemar, nie chi-kwadrat
na tabelce brzegowej:

    oba dobre 41 | oba padaja 5 | tylko bootstrap 0 pada 14 | tylko bootstrap 2 pada 4
    chi2 = 4,50 (poprawka Yatesa), p = 0,034

**Bootstrap 2 polowi odsetek awarii regionu, placac za to odrobina ostrosci wsrod udanych**
(0,682 -> 0,653). Mediana calosci sie nie rusza (0,637 vs 0,625). To nie jest metoda lepsza
wszedzie po trochu -- to jest metoda, ktora rzadziej gubi podmiot.

### Jak to sie godzi z pomiarem sprzed godziny

Godzine temu napisalem, ze bootstrap 0-4 jest nierozroznialny, i to dalej jest prawda --
**na pelnym zestawie**. Sceny czteroregionowe to tylko 2 z 11, a efekt siedzi wylacznie w nich,
wiec w sredniej po calosci tonie. Moje pierwotne wrazenie ("bootstrap 2 naprawia 3.2") tez bylo
trafne, tylko z zlego powodu: widzialem POJEDYNCZA awarie i uznalem to za przesuniecie jakosci,
a to jest zmiana czestosci awarii.

Trzy odczyty tego samego zjawiska -- okiem, srednia po calosci, i McNemar na podzbiorze --
daly trzy rozne odpowiedzi, i dopiero trzeci jest tym, co mozna napisac w pracy.

**Konfiguracja koncowa:** `--scale 0.4 --ground 1 --kappa 4.0 --ground_sched 0.5
--bootstrap 2 --feather 1.5` (alfa 0,1, ich wartosc).

## CIDM bench10-gen: ich kod nie dziala na wersji, ktora sami przypinaja

22581532 doczekal sie startu i padl po 22 s:

    TypeError: AttnProcessor2_0.__call__() got an unexpected keyword argument 'lora_id'

Przyczyna jest w ICH kodzie i jest jednoznaczna. `revise_edlora_unet_fusionattention_forward`
(`inference.py:123`) podmienia procesor wylacznie tam, gdzie w nazwie warstwy jest `attn2`:

    if layer.__class__.__name__ == 'Attention' and 'attn2' in name:

a `pipeline_edlora.py:296` wola UNet z `cross_attention_kwargs={'lora_id': id}`. W diffusers
0.20.0 -- czyli w wersji z ich wlasnego pinu -- `BasicTransformerBlock.forward` rozpakowuje te
kwargs TAKZE do `self.attn1`, gdzie siedzi jeszcze fabryczny `AttnProcessor2_0`.

Zdjecie tego argumentu w attn1 jest poprawne, a nie obejsciem: attn1 to self-attencja i w ich
schemacie nie ma na niej zadnej LoRA (`inference.py:157` zaklada `MultiLoRALinearLayer` tylko
na attn2), wiec `lora_id` jest tam informacja bez odbiorcy i fabryczny procesor liczy to samo
z nim i bez niego. Wolamy ich kod z `--method ours`, czyli `lora_num = 1`.

Naprawione `scripts/_cidm_attn1_shim.py` -- lata po NASZEJ stronie, na fabrycznej klasie
diffusers, tego samego rodzaju co zaslepki `_tkinter` i `IPython` w `sbatch_cidm_venv.sh`.
Ich zrodla zostaja nietkniete, bo o to chodzi w tym wierszu tabeli: ich metoda ma byc
policzona ich kodem. Zgloszone ponownie jako 23136039.

Warte zapamietania przy pisaniu o ich pracy: ich wydany kod nie ma potoku SDXL, a teraz okazuje
sie, ze sciezka generacji nie przechodzi nawet na przypietej przez nich wersji diffusers.

## Kontrola jednokonceptowa: ile kosztuje samo skladanie

Zostalo jedno zastrzezenie przy liczbie "DINO wycinka regionu 0,586 wobec 0,593 dla pojedynczego
konceptu": te dwie liczby pochodza z roznych ukladow. Wycinek regionu ma inna rozdzielczosc
natywna niz caly obraz, a przyciecie usuwa tlo, ktorego zdjecia referencyjne i tak nie maja.
**Dwa skazenia o przeciwnych znakach**, wiec z porownania nic pewnego nie wynika.

Wlasciwy uklad odniesienia juz istnieje w skrypcie: `--solo 1` renderuje kazdy region OSOBNO,
z ta sama ramka, promptem, skala i ziarnem. Wtedy rozdzielczosc, wielkosc pudelka i przyciecie
sa identyczne po obu stronach, a jedyna roznica jest to, **ile konceptow dzieli plotno**.
To jest dokladnie pytanie, ktore chcemy zadac.

Bieg 3184852: sceny 3.2 i 12.4 (te czteroregionowe), osiem ziaren, bootstrap 2 i kappa 4,0,
czyli konfiguracja koncowa. Scorer rozpoznaje uklad `solo_<V>/<i>.png` sam i tnie tylko ramke
tego jednego regionu, ktory na obrazie jest.

Przy okazji scorer przestal miec dwie definicje miary: obie sciezki, kompozycyjna i kontrolna,
skladaja wiersz przez jedna funkcje `score_one`.

## CIDM 26-38: prognoza sprzed trzech godzin byla pesymistyczna

Pisalem, ze blok nie zmiesci sie w limicie i utnie sie koło zadania 36. Tempo wzroslo:

    task 31  03:56:44
    task 32  04:18:38   (22 min)
    task 33  04:37:30   (19 min)
    task 34  04:57:43   (20 min)
    task 35  05:20:38   (23 min)

Okolo 21 min na zadanie zamiast 28 z pierwszych dwoch. Przy tym tempie zadanie 38 konczy sie
~06:45, a sciana jest 07:03 -- **zmiesci sie**. Pierwsze dwa zadania bloku byly wolniejsze
i ekstrapolowalem z nich, co bylo za male probka; wezel t0010 najwyrazniej sie odkorkowal.

Zglosilem juz blok 39-50 jako **zalezny** (`afterok:3184652`, job 3184853, limit 6 h). Jesli
26-38 skonczy sie poprawnie, nastepny startuje sam, bez czekania na obchod; jesli padnie,
zalezny nie ruszy i obsluze to recznie. Bliizniak na Heliosie (22882177) i tak nigdy nie
wystartuje, bo wisi na zaleznosci od anulowanego 22882176.

## Rano 19.09: tozsamosc konceptow, czyli zarzut powazniejszy niz kompozycja

Zarzut: sceny 3.2 i 3.3 wygladaja zle, 12.5 co najwyzej znosnie, **a obiekty w 12.5 nie
wygladaja jak referencyjne**. Sprawdzone na zdjeciach referencyjnych CIFC:

| koncept | referencja | nasze |
|---|---|---|
| duck_toy | **jaskrawo zolta** gumowa kaczka, pomaranczowy dziob | **bezowo-kremowa**, dziob sie zgadza |
| teddybear | maly, wytarty, plaski mis, bezowy z kremowym pyskiem | pluszowy, okragly, rozowawy, inne proporcje pyska |
| backpack | malinowy plecak z przypinkami | malinowy, ksztalt zblizony -- **trafiony najlepiej** |

Przy okazji poprawka do tego, co sam napisalem godzine wczesniej: mowilem, ze referencyjny
plecak jest czerwony (`eval_prefix: red` w configu), a na zdjeciu jest malinowy -- czyli akurat
ten kolor oddajemy dobrze, a pomylka byla moja.

**Wniosek, ktory wykracza poza kompozycje.** DINO 0,586 odpowiada "wlasciwa klasa, z grubsza
wlasciwy wyglad, **niewlasciwe szczegoly**". To nie jest wada skladania scen: pojedynczy koncept
na pelnym obrazie ma 0,593, czyli praktycznie tyle samo. To jest **sufit metody**. Przed
jakimkolwiek twierdzeniem o tozsamosci w pracy trzeba zestawic te 0,59 z sufitem zbioru
referencyjnego (`scripts/_ref_ceiling.py` -- policzone, ale nie zestawione z tym).

**Blad procesu po mojej stronie, wart zapisania.** Przeszedlem na scorer, bo cztery razy z rzedu
zle odczytalem pojedynczy obraz. Ale scorer mierzy WYLACZNIE "czy wlasciwy podmiot jest
w swojej ramce" -- nie widzi prostokatnych nakladek, wycietych krawedzi, psa bez tulowia ani
zlego koloru kaczki. Zamienilem ocene zaszumiona, lecz trafna, na precyzyjna, lecz nietrafna,
i na tej podstawie oglosilem konfiguracje zwycieska. Metryka byla dobra na pytanie, ktore
scigalem (brakujace podmioty), i zla na pytanie, ktore naprawde trzeba bylo zadac.

Kompozycja wstrzymana na prosbe.

## Anulowane i wpisane do pracy

Anulowane na Heliosie (6 zadan): `ch-sdxl-kd128wd0-eval` (22842707, wisial z
DependencyNeverSatisfied) oraz ramiona T = 50 `p022-noreg`, `p022-clipkey`, `p022-learnv`
wraz z dwiema zaleznymi macierzami `p022-clipkey-matrix` i `p022-learnv-matrix` -- bez swoich
ramion i tak nigdy by nie ruszyly. Wersje T = 10 tych ramion sa policzone i to z nich biora sie
liczby.

Wpisane do `main.tex` (4 zmiany, render lokalny przeszedl, **tresc dalej konczy sie na
stronie 9**, bibliografia na 10-11):

1. **Ramie noreg** w sekcji o zapominaniu: beta = 0 podnosi zapominanie do 0,0213 i obniza
   tozsamosc z 0,621 do 0,589. Dopisane zdanie, ktore z tego wynika, a ktorego wczesniej nie
   mielismy: to ramie **dalej zapomina piec do dziewieciu razy mniej niz bazy**, wiec czesc
   stabilnosci daje sama architektura, a nie kara.
2. **noout na SDXL** w akapicie o czterech projekcjach: DINO 0,593 -> 0,430, IA 80,1 -> 72,1.
   Z uczciwym odnotowaniem, ze CLIP-T jako jedyny nie idzie w te strone (75,5 -> 76,0), co jest
   dokladnie tym, jak powinno wygladac wyciecie projekcji piszacej w strumien rezydualny.
3. `\todo` tego akapitu mowilo "ewaluacja SDXL jest w kolejce" -- zastapione faktycznym
   zakresem przemiatu skal.
4. **Dryf** w dodatku: przetrenowany strumien odtwarza krzywa z dokladnoscia 3-5% na kazdym
   opoznieniu, a dla lagow >= 9 oba biegi usredniaja po IDENTYCZNYM zbiorze par.

### Pulapka, ktora omal nie zepsula pracy po cichu

Patch do `main.tex` pisalem heredokiem. **Powloka zjada jeden backslash**, wiec `\\beta`
docieralo do Pythona jako `\beta`, a Python czyta `\b` jako znak backspace -- do pliku
poszlo by `$<BS>eta = 0$` zamiast `$\beta = 0$`. Tak samo `\\todo` stawalo sie tabulatorem
(i na tym asercja pekla, co uratowalo zapis). Reguła na przyszlosc: **pliki z LaTeX-em pisac
narzedziem Write, nigdy heredokiem**, a wzorce trzymac w surowych literalach.

## Tozsamosc: wycofuje wlasny wniosek sprzed godziny

Napisalem, ze DINO 0,586 dla wycinka regionu jest praktycznie rowne 0,593 dla pojedynczego
konceptu, wiec skladanie nic nie kosztuje, a 0,59 to sufit metody. **To bylo bledne**, bo te
dwie liczby sa mierzone przy ROZNYCH promptach:

* ewaluacja jednokonceptowa dostaje `eval_prefix` -- prompt brzmi "yellow rubber V2 duck toy"
  (`gen_cifc.py:269` i komentarz w linii 60: `"<eval_prefix> V<k> <class>"`);
* RTP z ich figury 12 brzmi samo "V2 duck toy", bez atrybutu.

Czyli **kompozycja pracuje w trudniejszym warunku niz nasza wlasna ewaluacja**, a ja zestawilem
je tak, jakby byly tym samym. Porownanie bylo niewazne.

Poprawka do tego, co sam wczesniej zapisalem o attr_strip: `attr_strip` wycina atrybut
z captionow TRENINGOWYCH (`data.py:195`), ale `eval_prefix` doklada go z powrotem do promptu
GENERACYJNEGO. Nie usunelismy go z ewaluacji.

### Co pokazuja obrazy pojedynczych konceptow

| | nasze (eval, s04, po 10 zadaniach) | ich (ich kod, ich model, wizualizacja treningowa) |
|---|---|---|
| kaczka | **jaskrawo zolta, gladka guma, czerwony dziob -- poprawna** | bardzo dobra, zolta, na asfalcie |
| mis | kremowy pysk, kremowe lapy, wytarte futro -- **struktura z referencji** | **zepsuty: dwa nosy, zdublowana twarz** |

Wniosek, ktory to zmienia: **tozsamosc nie jest sufitem metody.** Pojedynczo generujemy
poprawnie, a w niektorych przypadkach lepiej niz ich kod na tym samym benchmarku. Tozsamosc
gubi sie DOPIERO w kompozycji.

Zastrzezenie do ich strony tabeli: ich wizualizacje sa z checkpointu tuz po nauczeniu danego
konceptu (task_2 dla kaczki, task_5 dla misia), wiec nie mialy okazji zapomniec; nasze sa po
wszystkich dziesieciu zadaniach. To gra na ich korzysc, a mis i tak wyszedl im zepsuty.

### Roznica konstrukcyjna, ktora trzeba nazwac w pracy

Ich identyfikator `<duck1><duck2><duck3>` uczyl sie na captionach ZAWIERAJACYCH "yellow rubber",
wiec atrybut mogl zostac wchloniety przez same tokeny. Nasz adapter uczyl sie na captionach
z wycietym atrybutem, wiec kolor musi niesc adapter -- i w kompozycji, gdzie prompt go nie
podaje, okazuje sie, ze nie niesie. **Nasz uklad jest w tym miejscu trudniejszy niz ich**,
i to jest roznica protokolu, a nie implementacji.

### Dwa testy, ktore to rozstrzygaja (NIE zgloszone, kompozycja wstrzymana)

1. Nasza kompozycja z atrybutem w promptcie regionu ("yellow rubber duck toy"). Kaczka zolknie
   -> kolor jest w promptcie, nie w adapterze. Nie zolknie -> adapter jest zepsuty, osobny problem.
2. **Ich** model, ich `inference.py`, prompt BEZ "yellow rubber". Jesli ich kaczka tez zblednie,
   przewaga na ich figurach jest przewaga protokolu, nie metody. Ten test jest wazniejszy dla
   pracy i nie wymaga kompozycji.

## Ile kosztuje samo skladanie -- zmierzone, i jest to duzo

Kontrola `--solo` (3184852, punktacja 3184854): te same regiony, te same ramki, te same prompty
i te same ziarna, tylko kazdy koncept renderowany OSOBNO zamiast razem. Sceny 3.2 i 12.4, osiem
ziaren, 64 sparowane regiony.

| | srednie DINO wycinka | awarie (<0,40) |
|---|---|---|
| **osobno** | **0,6883** | **0/64** |
| zlozone | 0,5917 | 9/64 (14%) |

Koszt skladania: **-0,0966 +- 0,0191, t = -5,06**. Gorszych po zlozeniu jest **52 regiony z 64**.
Awarie: McNemar daje 9 przypadkow "pada tylko po zlozeniu" i **zero** w druga strone, p = 0,0077.

**Renderowany osobno nie zawodzi ani jeden region z szescdziesieciu czterech.** Cala utrata
tozsamosci powstaje przy skladaniu.

### To obala moje wczesniejsze zdanie i tlumaczy, skad sie wzielo

Pisalem "0,586 wobec 0,593, wiec skladanie nic nie kosztuje". Teraz widac, ze porownanie bylo
zle na DWA sposoby naraz, a nie na jeden:

1. rozne prompty (opisane wyzej: `eval_prefix` kontra RTP bez atrybutu);
2. **wycinek zawyza**. Wlasciwym punktem odniesienia dla wycinka ze zlozonej sceny jest wycinek
   ze sceny JEDNOKONCEPTOWEJ, czyli 0,688 -- a nie DINO calego obrazu z ewaluacji, czyli 0,593.
   Przyciecie usuwa tlo, ktorego zdjecia referencyjne nie maja, wiec podnosi miare o okolo 0,09.
   Zestawiajac wycinek z pelnym obrazem porownalem dwie rzeczy w roznych skalach i wyszlo mi
   zero tam, gdzie jest -0,097.

Podejrzewalem to skazenie wczesniej i nazwalem je ("dwa skazenia o przeciwnych znakach"), ale
mimo to zostawilem wniosek oparty na tym porownaniu zamiast wstrzymac sie do kontroli. Kontrola
byla juz wtedy zaprojektowana i kosztowala dwadziescia minut.

### Co z tego wynika

Zarzut byl trafny i teraz ma liczbe. Tozsamosc **nie** jest sufitem metody -- pojedynczo
wszystkie 64 regiony wychodza poprawnie. Problem jest w mechanizmie skladania i ma rozmiar
0,097 DINO oraz 14% regionow przewroconych.

To jest tez wynik wart pracy, i to mocniejszy niz cokolwiek, co mielismy o kompozycji: mamy
kontrole, ktora izoluje efekt skladania od wszystkiego innego (rozdzielczosc, wielkosc ramki,
przyciecie, prompt, ziarno sa identyczne po obu stronach).

## Alpha: mechanizm potwierdzony, 40% luki domkniete

Hipoteza z odczytu kodu: w `sampling.py` scalanie daje wewnatrz pudelka
`alpha*eps_global + (1-alpha)*eps_r`, wiec przy alpha 0,1 **dziesiec procent kazdego kroku
wewnatrz ramki pochodzi z galezi globalnej**, ktorej prompt to samo ITP, bez zadnego konceptu.
Ta galaz aktywnie chce, zeby podmiotu tam nie bylo, i ciagnie w te strone przez wszystkie
piecdziesiat krokow. Sciezka solo tego czlonu nie ma w ogole -- i to jedyna strukturalna
roznica miedzy sufitem a podloga.

Test: alpha 0,05 i 0,0 przy reszcie bez zmian, sceny 3.2 i 12.4, osiem ziaren, 64 sparowane
regiony (3184855, 3184856; punktacja 3184860).

| | srednie DINO | vs alpha 0,1 | t | lepszych | domyka luki | awarie |
|---|---|---|---|---|---|---|
| sufit (solo) | 0,6883 | -- | -- | -- | -- | 6/64 |
| alpha 0,1 | 0,5917 | -- | -- | -- | -- | 9/64 |
| alpha 0,05 | 0,5786 | -0,0131 +- 0,0214 | -0,61 | 35/64 | -14% | 11/64 |
| **alpha 0,0** | **0,6300** | **+0,0383 +- 0,0200** | +1,91 | **44/64** | **40%** | 6/64 |

Sam test t daje t = 1,91, czyli na granicy -- ale srednia jest rozcienczona przez kilka duzych
wartosci odstajacych w druga strone. **Test znakow na sparowanych regionach daje p = 0,0040**
(44 z 64). Kierunek jest wiec spojny, tylko rozmiar efektu ma ogon. Dla alpha 0,05: 35/64,
p = 0,53, czyli nic -- co przy efekcie o polowe mniejszym od 0,038 i bledzie 0,021 jest
spodziewane, a nie sprzeczne.

Przy alpha 0,0 scena dalej sie trzyma: poza pudelkami `wsum = 0`, wiec czlon tla to nadal
pelne `eps_global`. Znika wylacznie ciagniecie ku "tu nic nie ma" WEWNATRZ ramek.

**Zostaje 60% luki.** To juz nie jest scalanie, tylko sprzezenie przez wspolne plotno: galaz
regionu widzi na obrazie cudze podmioty i odszumia je jak swoje. Na to dziala bootstrap
(0 -> 2 polowilo awarie, p = 0,034) i to jest nastepna os, jesli wracamy do kompozycji.

Uwaga do pracy: alpha 0,1 to ICH wartosc z artykulu. Zejscie do zera trzeba nazwac wprost,
razem z powodem -- i powod jest dobry, bo wynika z rownania, a nie z przemiatu.

## Figura 6: blad rzeczowy w podpisie

`manager.lora_scale` to domyslnie **1.0** (`manager.py:42`), a `train_cl.py` nigdy go nie
ustawia przed zapisem probek `forgetting/` -- `gscale` w `_gen_one` to guidance scale (CFG),
nie sila adaptera. Czyli obrazy w Figurze 6 sa renderowane przy **pelnej sile adaptera**,
a podpis mowi `s_lora = 0.45`. To jest blad rzeczowy i trzeba go poprawic niezaleznie od reszty.

To tez tlumaczy objawy, na ktore zwrocil uwage uzytkownik: nie chodzi tylko o rozjazd na koncu,
ale o **monotoniczny spadek nasycenia i narastanie artefaktow juz od dziesiatego zadania**.
Nasza wlasna Figura 9 mowi, ze przy 1.5 obraz sie rozpada -- 1.0 jest wiec juz w drodze tam,
dwukrotnie powyzej punktu pracy 0,45 i powyzej 0,8, przy ktorym czytamy T = 50 w teaserze.
A poniewaz generowane dW dryfuje wzdluz strumienia (3,7% po dziesieciu zadaniach, 11% po
czterdziestu dziewieciu), przy zablokowanej sile 1.0 obraz odchodzi od rozkladu tym bardziej,
im dluzszy strumien.

Sprawdzane, nie zakladane: bieg 3184859 + 3184861 renderuje koncept 0 z checkpointow po
1/10/20/30/40/50 zadaniach przy skalach 0,45 / 0,8 / 1,0, ten sam prompt i ziarno. Jesli przy
0,45 degradacja znika, figura jest do uratowania i potwierdza to, co juz piszemy w Figurze 4(b).
Jesli nie znika, mamy realne zapominanie w przestrzeni obrazu, **ktorego DINO nie widzi**
(0,0060), i historie o T = 50 trzeba zlagodzic.

Waznym ustaleniem negatywnym jest przy okazji to, ze probki v2 po zadaniu 49 wygladaja **gorzej**
niz obecne w figurze (wyprane, koronkowa faktura), wiec sama podmiana zrodla na przetrenowany
bieg nie jest ratunkiem.

## Teaser na nowym checkpointcie

3184858: pasek teasera (b) z `p022_v2/ckpts/hyper_after_task49.pt`, co piaty koncept
(0, 5, ..., 45), skala 0,8, cztery ziarna. Ten sam tor renderowania co `_teaser_gen.py`
(galaz umiejscowienia zainstalowana przy pelnym kadrze, sampler DPM++, maska tokenow na slowie
klasy) -- kazdy z tych szczegolow byl tam ustalony osobno i zmiana ktoregokolwiek daje inne obrazy.

Nowy `scripts/_fig_rerender.py` laduje model RAZ i chodzi po checkpointach, zadaniach i skalach;
szesnascie wywolan `_teaser_gen.py` to byloby szesnascie ladowan SDXL-a.

Do `outputs/sweep/p022_v2/ckpts/` doszedl `hyper_after_task00.pt`, przeniesiony z Heliosa --
brakowalo go po stronie Atheny, bo v2 wznawial od zadania 10. To jest wlasciwe miejsce tego
pliku (zadania 0-9 v2 to doslownie te z v1), ale odnotowuje, bo to katalog wynikow.

## Figura 6 naprawiona: to nie bylo zapominanie, tylko przesterowany adapter

Przemiat: koncept 0 z checkpointow po 1/10/20/30/40/50 zadaniach, skale 0,45 / 0,80 / 1,00,
ten sam prompt, to samo ziarno (7000). Biegi 3184859 i 3184861.

* **0,45** -- pies nasycony, ostry i spojny przez wszystkie szesc kolumn;
* **0,80** -- koronkowa faktura juz w drugiej kolumnie, dalej coraz gorzej;
* **1,00** -- rozpad, najmocniej na koncu.

Czyli spadek nasycenia i artefakty, ktore uzytkownik zobaczyl na figurze, **nie sa
zapominaniem** -- sa skutkiem czytania modelu przy pelnej sile adaptera. Stara figura pokazywala
dolny wiersz, a podpis twierdzil, ze gorny (`s_lora = 0.45`), bo `train_cl.py` zapisuje probki
`forgetting/` przy domyslnym `manager.lora_scale = 1.0`.

Zrobione: komorki `figures/forget_cells/` podmienione na render przy 0,45 (stare w
`forget_cells_old/`), podpis poprawiony -- zdanie o proweniencji bylo nieprawdziwe po podmianie
("none was generated for this figure"), wiec zastapione opisem faktycznego zrodla. Render
lokalny przeszedl, tresc dalej konczy sie na stronie 9.

### Czego NIE zrobilem i dlaczego

Padla propozycja, zeby w tej figurze dac **malejaca skale wzdluz kolumn**. Odradzilem:
zadaniem tej figury jest wyizolowac wplyw STRUMIENIA na jeden koncept, a skala dobierana per
kolumna odbiera mozliwosc przypisania stalosci sieci -- recenzent przeczyta to jako dobranie
skali, przy ktorej kazda kolumna wyglada dobrze. Do tego kryterium "ktora skala jeszcze dziala"
byloby wzrokowe, czyli nieobronne.

Wersja uczciwa to **skala stala w wierszu, zmienna miedzy wierszami** (0,45 / 0,80 / 1,00 x
szesc checkpointow): wzdluz wiersza widac, co robi strumien, w dol kolumny -- co robi
przesterowanie, i nic nie jest strojone per komorka. Mieszcza sie w obecnym miejscu, bo drugie
ziarno niewiele wnosi.

### Dlug, ktory sam zaciagnalem

Do poprawionego podpisu wpisalem, ze **uzyteczny zakres skali zweza sie wraz z dlugoscia
strumienia** -- na podstawie JEDNEGO konceptu, jednego promptu i dwoch ziaren. To jest ten sam
rodzaj uogolnienia, ktory tej nocy cztery razy upadl po sprawdzeniu. Zamiast zostawic to jako
wrazenie: bieg 3184862 renderuje cztery kolejne koncepty (kaczka, kot, plecak, mis) na pieciu
checkpointach i trzech skalach, a nowy `scripts/_fig_score.py` liczy dla kazdej komorki DINO
do zdjec referencyjnych -- ta sama miara co w calym benchmarku, wiec wynik da sie zestawic
z reszta pracy. Punktacja: 3184863 (koncept 0) i 3184864 (zalezny, cztery pozostale).

Jesli strata wzgledem 0,45 rosnie z numerem checkpointu dla wiekszosci konceptow, zdanie
zostaje i jest osobnym wynikiem. Jesli nie -- zawezam je do tego, co pokazuje figura.

## Skala adaptera wzdluz strumienia: moja hipoteza obalona na czterech z pieciu konceptow

`_fig_score.py` policzyl DINO do referencji dla kazdej komorki (koncept x checkpoint x skala),
ta sama miara co w calym benchmarku. Strata skali 1,00 wzgledem 0,45, od dziesiatego do
piecdziesiatego zadania:

| koncept | po 10 | po 20 | po 30 | po 40 | po 50 |
|---|---|---|---|---|---|
| pies (0) | -0,070 | -0,049 | -0,138 | -0,048 | **-0,161** |
| kaczka (1) | -0,053 | +0,055 | -0,167 | -0,046 | -0,013 |
| **kot (2)** | +0,031 | +0,001 | +0,037 | +0,022 | **+0,095** |
| **plecak (3)** | +0,044 | -0,018 | +0,068 | +0,076 | **+0,061** |
| **mis (4)** | +0,027 | +0,031 | -0,021 | +0,015 | **+0,042** |

Zdanie "uzyteczny zakres skali zweza sie wraz z dlugoscia strumienia" jest prawdziwe dla
**jednego konceptu z pieciu**. Trzy ida dokladnie odwrotnie: przy 0,45 sa niedosterowane
i wola pelna sile, tym bardziej im dluzszy strumien. Zdanie **wycofane z podpisu** Figury 6
i zastapione tym, co widac: ten konkretny koncept traci 0,16 DINO przy pelnej sile, a z czterech
sprawdzonych trzy punktuja tam WYZEJ, wiec `s_lora` wymienia tozsamosc na wiernosc promptowi
per koncept, a jeden punkt pracy jest kompromisem miedzy nimi.

To jest wlasciwy wynik tej kontroli: postawilem zdanie na jednym koncepcie, sam zaplanowalem
sprawdzenie, i sprawdzenie je obalilo, zanim poszlo do recenzenta. Rozrzut samego DINO przy
0,45 jest przy okazji duzy -- pies 0,82, mis 0,73, plecak 0,49 -- wiec srednia po konceptach
kryje bardzo rozne przypadki.

## CIDM 39-50: moj blad przy zgloszeniu, zlapany przez ich bug

Blok 26-38 skonczyl sie poprawnie (4:55:55, mieszczac sie w limicie -- moja pierwsza prognoza
byla pesymistyczna, druga trafna). Zalezny blok 39-50 wystartowal i padl po 21 s:

    TypeError: 'dict_keys' object is not subscriptable
    lib/data/lora_dataset.py:42  instance_prompt_image = replace_mapping.keys()[0]

To jest idiom z Pythona 2 w ICH kodzie, w galezi awaryjnej uruchamianej, gdy dla obrazu
**brakuje pliku z podpisem**. Ale przyczyna jest po naszej stronie i jest moja: logi pokazuja

    26-38: configi ./options/cidm50_v2
    39-50: configi ./options/cidm50

Zgloszajac zadanie zalezne ustawilem `SBATCH_EXTRA`, a **zapomnialem `CIDM_OPTS`**, ktore przy
poprzednich blokach szlo w srodowisku. Blok poszedl configami v1, wskazujacymi na stary katalog
podpisow, ktorego juz nie ma.

Awaria byla szczesliwa: ich martwy kod z Pythona 2 zatrzymal to po dwudziestu jeden sekundach,
zamiast pozwolic trenowac cztery godziny na zlych danych i wpisac to do tabeli. Gdyby captiony
v1 istnialy, bieg przeszedlby cicho.

Poprawione: 3184869 z `CIDM_OPTS=./options/cidm50_v2`, zweryfikowane w naglowku loga.

**Wniosek proceduralny:** `CIDM_OPTS` nie ma wartosci domyslnej wskazujacej na aktualny
strumien -- domyslna jest `./options/cidm50`, czyli v1. Kazde zgloszenie musi ja podac jawnie,
a nagłowek loga (`=== CIDM, configi ...`) trzeba sprawdzac po starcie, nie po awarii.

## lr3e4: pelny przemiat potwierdza rozpad

22516080 COMPLETED. DINO 0,0569 / 0,0353 / 0,0265 przy skalach 0,40 / 0,30 / 0,25 -- rozpad na
kazdej skali, wiec to nie jest kwestia punktu pracy. Athena dala na s04 0,0567, Helios 0,0569:
kolejna darmowa zgodnosc miedzy klastrami.

## all50 nie zmiesci sie w limicie -- policzone z tempa

18 konceptow w 5:29, czyli ~18,3 min na koncept. Zadanie robi 50 konceptow x DWIE skale
(0,45 i 0,6), czyli 100 generacji = okolo 30 h przy limicie 8 h. Do sciany zostaly 2,5 h,
wiec dojdzie do ~26 konceptu PIERWSZEJ skali i sie utnie; skala 0,6 nie ruszy w ogole.

Przyczyna jest w liczbie probek, nie w wolnym wezle: `sbatch_eval50.sh` woła `gen_cifc`
z `--num_samples 50`, czyli 20 promptow x 50 = **1000 obrazow na koncept**, podczas gdy
macierze T = 10 licza `--num_samples 10`, czyli **200**. To nie jest blad -- wiersz koncowy
ma 50 komorek zamiast 55 i stać go na wiecej probek -- ale piec razy wiecej pracy na koncept
przy tym samym limicie sie nie miesci.

`gen_cifc` **nie ma wznawiania** (jedyny `os.path.exists` dotyczy checkpointu, nie wyjscia),
ale ma filtr `--only`, ktory zostawia juz wygenerowane komorki nietkniete. Czyli timeout jest
odwracalny: po scianie odczytuje liste konceptow, ktorych brakuje, i zglaszam je jawnie.

Do decyzji: **czy skala 0,6 jest w ogole potrzebna.** Punkt pracy to 0,45, a krzywa po skali
dla T = 50 jest juz w Figurze 4(b) ze starszego biegu. Rezygnacja z drugiej skali zdejmuje
polowe pozostalej pracy, czyli okolo 15 h GPU.

## s1600: pelny przemiat, i poprawka do mojego uzasadnienia

Zdazyl ze wszystkimi trzema skalami (5:15:22). DINO **0,5236 / 0,4710 / 0,4943** przy
0,40 / 0,30 / 0,25.

Napisalem wczesniej, ze trzecia skala "nie zdazy i nie szkodzi, bo przebieg jest monotoniczny".
Zdazyla, a przebieg **nie jest monotoniczny** -- 0,25 lezy WYZEJ niz 0,30. Wniosek sie nie
zmienia (najlepszy punkt to nadal 0,40), ale uzasadnienie mialem zle i zgadywalem ksztalt
krzywej z dwoch punktow.

Komplet ramion SDXL, kazde przy swojej najlepszej skali:

| ramie | DINO |
|---|---|
| **b3000 (naglowek)** | **0,593** |
| s1600 | 0,524 |
| noout | 0,430 |
| lr3e4 | 0,057 |

Zadne z trzech nie bije naglowka.

## Decyzja: skala 0,6 w all50 ZOSTAJE

Sprawdzilem, co praca z niej robi, zanim ja scialem. Paragraf 5.4 nie traktuje jej jako
dodatku -- podaje przy niej osobny komplet liczb (TA 72,5 / IA 75,3 / DINO 0,553, oraz 0,593
wobec 0,644 dla dziesieciu konceptow), a `\todo` tego akapitu mowi wprost:

    The s = 0.6 row is the one comparison made at the same adapter scale on both sides
    as well as at matched text alignment.

To przy 0,6 stoi porownanie z C-LoRA, czyli jedyne w tym akapicie dopasowane JEDNOCZESNIE
po skali adaptera i po zgodnosci z tekstem -- metodologicznie najmocniejsze, jakie tam mamy.
Wyciecie jej nie oszczedziloby GPU, tylko zabralo argument.

Problem jest wiec obliczeniowy, nie merytoryczny, i rozwiazuje go **zrownolegleniem zamiast
cieciem protokolu**. `--num_samples` zostaje 50, bo obnizenie go do 10 zmienialoby protokol
pomiaru w polowie pracy.

`sbatch_eval50.sh` pomija skale, ktora ma juz `cifc_metrics.json`, wiec skale mozna liczyc
niezaleznie. `gen_cifc --only_concepts` pozwala dzielic po konceptach. Skala 0,6 poszla w trzech
kawalkach po ~17 konceptow (3184910, 3184911, 3184912), po ~5,2 h kazdy, limit 6:30. Metryki
trzeba bedzie policzyc RAZ na koncu, bo kazdy kawalek widzi tylko swoja czesc -- dlatego wolam
`gen_cifc` wprost, a nie przez `sbatch_eval50.sh`, ktory dopinalby metryki po kazdym kawalku.

Reszta skali 0,45 (~24 koncepty) zostanie dogenerowana tym samym sposobem, gdy biezacy job
3184649 dojdzie do sciany -- dopiero wtedy bede wiedzial doklanie, ktorych brakuje.

## kd128wd0: jedyne ramie SDXL, ktore nie przegrywa czysto

3184846 COMPLETED (3:50:42). Skala 0,4: **TA 80,20 | IA 76,35 | DINO 0,5424**.

Zestawienie z naglowkiem b3000 (TA 75,54 | IA 80,13 | DINO 0,593) nie jest jednak rozstrzygniete,
bo to sa **rozne punkty na krzywej kompromisu**, a nie gorszy wynik: przy tej samej nominalnej
skali 0,4 adapter kd128wd0 dziala slabiej (wyzsze TA, nizsze IA), czyli lezy dalej na osi
zgodnosci z tekstem. Praca porownuje "at matched text alignment", a z jednego punktu nie da sie
interpolowac.

Zeby to rozstrzygnac, brakuje drugiej skali dla tego ramienia (np. 0,6), co obwiedzie punkt
pracy z obu stron. Nie zglaszam tego teraz, bo kolejka Atheny ma juz trzy kawalki all50 na
sciezce krytycznej do liczb T = 50, a to ramie jest ablacja poboczna. Do decyzji, gdy all50
zwolni miejsce.

## all50 podzielone na piec kawalkow, komplet w drodze

Bieg 3184649 doszedl do sciany po 8 h z **26 konceptami** skali 0,45 (cifc_dog ...
cc101_actionfigure_2, po kolei). Reszta zgloszona jako dwa kawalki po 12 konceptow:
3185018 (27-38) i 3185019 (39-50), limit 4 h kazdy przy zmierzonych ~15 min na koncept.

Razem plan dla wiersza T = 50:

| kawalek | skala | koncepty | job |
|---|---|---|---|
| (bieg pierwotny) | 0,45 | 1-26 | 3184649, TIMEOUT po 8 h -- wyjscie zostaje |
| D | 0,45 | 27-38 | 3185018 |
| E | 0,45 | 39-50 | 3185019 |
| A | 0,6 | 1-17 | 3184910 |
| B | 0,6 | 18-34 | 3184911 |
| C | 0,6 | 35-50 | 3184912 |

Zgloszenie kawalkow D i E jeszcze przed formalnym koncem 3184649 bylo bezpieczne: pisza do
INNYCH katalogow konceptow, a nowe zadanie startuje minutami, podczas gdy staremu zostawaly
sekundy. Jedyny styk to koncept 27, ktorego stary bieg nie zdazyl domknac -- nowy go nadpisze
w calosci, co jest poprawne.

**Do zrobienia po kawalkach:** `cifc_metrics` trzeba uruchomic RAZ na kazdy katalog skali
(`all50/s045` i `all50/s06`), bo zaden kawalek nie widzi pelnego zestawu i zaden go nie liczy.
Bez tego kroku nie ma `cifc_metrics.json`, czyli nie ma liczb do pracy.

### Tempo: trzeci raz tego dnia pierwszy odstep wprowadzil w blad

Kawalek A po 34 min mial jeden koncept, co dawalo 8,8 h i nie mieszczilo sie w limicie. Po
64 min mial trzy, czyli drugi i trzeci poszly po ~15 min -- pierwszy doliczal ladowanie modelu.
Prognoza spadla z 8,8 h na 4,4 h i miesci sie z zapasem.

To samo bylo dzis z CIDM 26-38 (pierwsze dwa zadania po 28 min, reszta po 21) i z CIDM 39-50
(pierwsze 38,5 min, drugie 27). **Regula: przy tych biegach pierwszy odstep zawiera rozruch
i nie wolno z niego ekstrapolowac** -- potrzebne sa co najmniej dwa ukonczone elementy.

## all50: stan podzialu po pierwszej rundzie

| skala | koncepty | job | wynik |
|---|---|---|---|
| 0,45 | 1-26 | 3184649 | TIMEOUT 8:00, 26 gotowych |
| 0,45 | 27-38 | 3185018 | TIMEOUT 4:00, **11 z 12** -- brak `cc101_toy_gnome` |
| 0,45 | 39-50 | 3185019 | **COMPLETED 3:46:51, 12/12** |
| 0,45 | luka | 3185121 | zgloszony z `afterany:3185018` |
| 0,6 | 1-17 | 3184910 | liczy, 15/17 |
| 0,6 | 18-34 | 3184911 | **COMPLETED 4:50:56, 17/17** |
| 0,6 | 35-50 | 3184912 | **COMPLETED 4:34:20, 16/16** |

Po domknieciu 3184910 i 3185121 oba katalogi skal beda kompletne (50 konceptow kazdy).

Prognozy tempa sprawdzily sie co do konceptu: zapowiadalem, ze D zabraknie ostatniego i tak sie
stalo, a E zdazy i zdazyl. Warunek, ktory to umozliwil, to trzymanie sie reguly, ze **pierwszy
odstep zawiera rozruch** -- przy trzech konceptach kawalki wygladaly na 15 min, przy dziewieciu
na 20-21, i dopiero ta druga liczba byla wlasciwa do prognozy.

**Krok, ktorego nie wolno pominac:** `python -m src.cifc_metrics --config configs/phaseT/T50_v2.yaml
--eval_root outputs/sweep/p022_v2/all50/s045` i to samo dla `s06`. Zaden kawalek nie liczy metryk,
bo zaden nie widzi pelnego zestawu, wiec bez tych dwoch wywolan nie ma `cifc_metrics.json`,
czyli nie ma liczb do paragrafu 5.4.

## Do rozstrzygniecia: baseline'y forgettingu

Pytanie WG: nasze reimplementacje sa niewiarygodne, wiec zawezic porownanie do fine-tuningu,
CIDM i nas.

Stan faktyczny, sprawdzony w `main.tex`:

* §5.3 podaje CZTERY bazy z wartosciami do trzech miejsc (LwF 0,120, C-LoRA 0,148, finetune
  0,184, EWC 0,187) i mowi "between twenty and thirty-one times more" -- czyli nadaje im
  dokladnosc i PORZADEK;
* §6 mowi, ze te same wiersze **nie odtwarzaja opublikowanej tabeli**, ze "no variant recovers
  its ordering", i ze forgetting nalezy czytac jako rzad wielkosci, nie jako reprodukcje;
* CIDM jest z porownania forgettingu nieobecny, bo nikt go nie publikuje, a nasz pomiar ich
  metody to wlasnie `bench10-gen`, zatkany w kolejce Heliosa.

Czyli **§5.3 przeczy §6** i to jest realna niespojnosc niezaleznie od decyzji o zakresie.

Argument za zawezeniem jest przy tym mocniejszy, niz zostal postawiony: nieodtworzone zostaly
metody STROJONE (LwF za mocny, C-LoRA za slaby, EWC ma lambda), a **fine-tuning nie ma zadnego
hiperparametru do przestrojenia**, wiec jako jedyny z czworki nie jest podatny na ten tryb bledu.
To tlumaczy, czemu jego zostawic, a tamte nie -- lepiej niz ogolne "reimplementacje sa
niewiarygodne".

Zastrzezenie: zestaw "finetune + CIDM + my" jest dzis niekompletny, bo czlonu CIDM nie mamy.
Modele bench10 wazza 332 MB, wiec przeniesienie ich na Athene kosztuje ~45 min transferu.

Decyzja WG oczekiwana, nic w `main.tex` nie zmieniam do tego czasu.

## Wiersz T = 50 na poprawionych podpisach -- policzony

`cifc_metrics` na obu katalogach (3185126, 3185128, po ~15-17 min). Piecdziesiat konceptow,
po 1000 obrazow na koncept, model koncowy po wszystkich piecdziesieciu zadaniach.

| skala | | CLIP-T | CLIP-I | DINO |
|---|---|---|---|---|
| 0,45 | **v2 (nowe)** | **74,48** | **73,49** | **0,5244** |
| 0,45 | v1 (w pracy) | 75,1 | 72,0 | 0,502 |
| 0,6 | **v2 (nowe)** | **71,72** | **76,70** | **0,5767** |
| 0,6 | v1 (w pracy) | 72,5 | 75,3 | 0,553 |

Wzorzec jest ten sam na obu skalach: **+0,023 DINO i +1,45 CLIP-I kosztem ~0,7 CLIP-T**.
Przetrenowanie na poprawionych podpisach cc101 kupuje tozsamosc i podobienstwo obrazu, placac
odrobina zgodnosci z tekstem -- czego nalezalo oczekiwac, skoro podpisy przestaly zawierac
cechy wygladu, ktore wczesniej szly przez zamrozone projekcje zamiast przez adapter.

### Konsekwencja, ktorej nie wolno przeoczyc przy wpisywaniu

Paragraf 5.4 porownuje baseline'y **przy zrownanym CLIP-T**, a nasz CLIP-T sie przesunal. Dwa
zdania przestaja byc prawdziwe:

* "C-LoRA lands **within two tenths of a point** of our text alignment (TA 72,3)" -- przy starym
  72,5 roznica byla 0,2, przy nowym 71,72 jest **0,6**. Tego zdania nie da sie zostawic.
* "the closest baseline on text alignment is EWC (TA 75,7)" -- przy starym 75,1 roznica byla 0,6,
  przy nowym 74,48 jest 1,2, wiec "closest" trzeba sprawdzic ponownie wzgledem calej listy.

Marginesy za to **rosna**, wiec zmiana idzie na nasza korzysc:

| wobec | CLIP-I | DINO |
|---|---|---|
| EWC (69,1 / 0,433) | +4,4 zamiast +3 | +0,091 zamiast +0,07 |
| C-LoRA (69,2 / 0,423) | +7,5 zamiast +6 | +0,154 zamiast +0,13 |
| kontrola oracle (0,419) | -- | +0,105 do +0,158 |

Czyli zdanie "the margin therefore runs between three and six points of image alignment, and
between 0,07 and 0,13 DINO" trzeba przeliczyc na **4,4-7,5 punktu** i **0,09-0,15 DINO**.

Nie wpisuje tego do `main.tex`, dopoki nie zapadnie decyzja o zakresie baseline'ow -- bo jesli
LwF, C-LoRA i EWC wypadaja z porownania, to te same zdania i tak sa do przepisania od zera,
a pisanie ich dwa razy nie ma sensu.

## CIDM-50: generacja zgloszona, padla na MOIM bledzie w shimie, poprawiona

Lancuch CIDM-50 skonczyl sie (task_1 ... task_50 w `data/CIFC/output`), Athena stala pusta,
wiec zglosilem generacje i punktacje -- jedyny cel, dla ktorego ten lancuch byl trenowany.

Sprawdzone przed zgloszeniem, bo oba moglyby to wysadzic:

* ich generator domyslnie robi **macierz**, a przy 50 zadaniach to 1275 komorek zamiast 50;
  `sbatch_cidm_gen.sh` zawsze przekazuje `--final_only` (`ks = [a.tasks - 1]`), wiec liczy sam
  wiersz koncowy;
* `_cidm_gen.py` **pomija koncepty z gotowym `prompts.json`**, wiec timeout niczego nie powtarza.

Pierwsze zgloszenie (3185173) padlo po 23 s:

    inference.py: error: unrecognized arguments: .../_cfg_k49.json

**Blad byl moj, w shimie z dzisiejszego rana.** `_cidm_attn1_shim.py` robil
`sys.argv = sys.argv[1:]`, zeby zdjac swoja sciezke. Ale argparse czyta `sys.argv[1:]`, bo
zaklada, ze element zerowy to nazwa programu -- wiec po moim cieciu ich parser gubil PIERWSZA
flage (`--concept_cfg`) i widzial jej wartosc jako argument pozycyjny.

Poprawka: `sys.argv[0] = "inference.py"` zamiast zdejmowania elementu. Shim wyslany na oba
klastry, bo **`bench10-gen` czekajacy w kolejce Heliosa uzywa tego samego pliku** i padlby
identycznie, gdyby dostal wezel przed poprawka. Zglolszone ponownie jako 3185188.

Wart odnotowania wzorzec: ten shim mial dwa bledy z rzedu, oba w warstwie sklejenia, nie
w logice. Pierwszy (heredoc zjadajacy backslash) zlapala asercja, drugi -- ich wlasny parser
po 23 sekundach. Oba tanie, bo padly natychmiast; gdyby ktorykolwiek przechodzil cicho,
kosztowalby godziny.

## CIDM-50 generacja: ICH potok potrzebuje 26 h, lancuch czterech zadan

Po naprawie shimu bieg 3185188 rusza poprawnie, ale jest wolny. Tempo odczytane wprost z ich
paskow postepu, nie oszacowane:

    jedna partia 5 obrazow x 50 krokow odszumiania = 47 s
    koncept = 20 promptow x 2 iteracje = 40 partii = ~31 min
    50 konceptow = ~26 h

przy limicie 8 h. To jest ich `inference.py` na SD-1.5, 512, batch 5 -- nic po naszej stronie
nie jest tu waskim gardlem.

**Czego nie zrobilem: nie obnizylem `per_prompt`.** Zejscie z 10 na 5 skrocilo by to o polowe,
ale `bench10-gen` liczy z 10, a oba wiersze CIDM (T = 10 i T = 50) musza byc na tym samym
protokole, zeby dalo sie je zestawic ze soba i z naszymi.

Zamiast tego lancuch czterech zadan zaleznych (`afterany`): 3185188 -> 3185198 -> 3185199 ->
3185200, po 8 h kazde. Dziala bez zmiany kodu, bo `_cidm_gen.py` **pomija koncepty z gotowym
`prompts.json`** -- kazde kolejne zadanie przeskakuje zrobione i dolicza reszte. `afterany`,
a nie `afterok`, bo poprzednik skonczy sie TIMEOUTEM, nie sukcesem, i `afterok` zatrzymalby
lancuch na pierwszym ogniwie.

Cztery ogniwa po 8 h to 32 h zarezerwowane na 26 h pracy, wiec ostatnie skonczy sie wczesniej
i puentuje calosc scoringiem (`cifc_metrics` jest w tym samym runnerze, po generacji).

## bench10-gen przeniesiony na Athene

Wisial w kolejce Heliosa od rana z `Priority`. Modele spakowane (`tar -czf`, 746 plikow,
343 MB -- tar zamiast `scp -r`, bo 746 osobnych transferow przez to lacze trwaloby dluzej niz
sam wolumen), przeniesione przez dysk lokalny, rozpakowane na Atenie: **746 plikow, zgadza sie
co do pliku, rozmiar archiwum co do bajta**. Bieg 3185279 rusza czysto.

Wersja heliosowa (23136039) anulowana -- "przenies" znaczy przeniesc, nie zdublowac, a modele
sa teraz po obu stronach, wiec powrot jest mozliwy bez transferu.

### Wazne rozroznienie: wiersz koncowy to NIE jest forgetting

`sbatch_cidm_gen.sh` przekazuje `--final_only` bezwarunkowo, wiec 3185279 liczy **10 komorek**
(k = 9, j = 0..9), nie macierz 55-komorkowa. To wystarcza do:

* odtworzenia ich opublikowanego wiersza TA/IA przy T = 10, czyli **walidacji lewego panelu
  Figury 3** -- pytania, czy w ogole wolno nam sie z nimi porownywac;

ale **nie wystarcza do forgettingu**, bo ten jest liczony jako maksimum minus wartosc koncowa
po calej macierzy. Forgetting CIDM wymaga 55 komorek, czyli ~28 h przy ich tempie 31 min na
komorke.

Kolejnosc jest celowa: wiersz koncowy jest piec razy tanszy i odpowiada na wazniejsze pytanie.
Macierz mozna dolozyc potem i **pominie juz policzone komorki** (`prompts.json`), wiec te 5 h
nie przepada.

## Krzywa SDXL i problem, ktory z niej wychodzi

Pelny przemiat `X_sdxl_b3000` po dziesieciu zadaniach:

| skala | TA | IA | DINO |
|---|---|---|---|
| 0,20 | 83,17 | 75,23 | 0,519 |
| 0,25 | 81,43 | 76,58 | 0,534 |
| 0,30 | 78,97 | 78,11 | 0,554 |
| **0,40** | 75,54 | **80,13** | **0,593** |
| 0,50 | 73,67 | 81,29 | 0,627 |

Ich opublikowany wiersz SDXL ma TA 80,0. Interpolacja miedzy 0,25 i 0,30 daje tam
**IA ~77,5 i DINO ~0,545** (skala efektywna ~0,28).

**Problem.** Dodatek `app:sdxl` podaje dzis TA 80,2 / IA 79,3 / DINO 0,597, ale jego `\todo`
mowi, ze to `X_sdxl_ground_800aug`, **nie przepis naglowkowy**. Czyli starsze ramie osiagalo
TA 80 i IA 79,3 JEDNOCZESNIE, a `b3000` przy tym samym TA daje 77,5. Podmiana wiersza na
przepis naglowkowy -- czego ten `\todo` zada dla spojnosci -- **oslabia wynik SDXL o ~1,8
punktu IA**.

**Blokada.** Nigdzie nie mamy zapisanego **ich opublikowanego IA dla SDXL** (ten sam `\todo`
prosi: "fill the published SDXL IA"). Bez tej liczby nie da sie powiedziec, czy przy ich TA
wygrywamy, czy przegrywamy -- wiadomo tylko, ze w ich TA trafiamy.

## Koszt ewaluacji: my kontra oni

Zmierzone na tych samych biegach, SD-1.5, 512, 50 krokow:

* ich potok: 31 min na 200 obrazow = **9,3 s/obraz**
* nasz: 20 min na 1000 obrazow = **1,2 s/obraz**

Okolo osmiu razy. **Zastrzezenie, bez ktorego tej liczby nie wolno wpisac do pracy:** to jest
porownanie "jak uruchomione", nie "jak metoda". My batchujemy po 10, oni po 5, a `_cidm_gen.py`
odpala ich `inference.py` jako NOWY PROCES na kazdy koncept, wiec kazdy z pięćdziesięciu laduje
SD od zera. Czysty pomiar wymagalby zrownania batcha i jednego procesu.

## `outputs/cidm_rerun*` -- NIEAKTUALNE, nie uzywac tych liczb

Szukajac, czy `bench10-gen` nie powtarza zrobionej pracy, znalazlem w
`outputs/cidm_rerun/` komplet metryk CIDM przy T = 10, z macierza wlacznie:

| | TA | IA | DINO | forgetting DINO |
|---|---|---|---|---|
| `cidm_rerun/matrix` | 75,31 | 77,88 | 0,5836 | **0,0174** |
| `cidm_rerun/s05` | 75,98 | 76,77 | 0,5562 | -- |
| `cidm_rerun/s07` | 75,61 | 77,53 | 0,5752 | -- |
| `cidm_rerun/s08` | 75,31 | 77,88 | 0,5836 | -- |
| `cidm_rerun/s10` | 74,74 | 78,61 | 0,6011 | -- |

Przez chwile wygladalo to na dowod, ze biegnacy `bench10-gen` jest zbedny, a zdanie w pracy
o braku pomiaru CIDM -- na nieaktualne. **Jest odwrotnie.** Ta sama liczba 0,0174 jest w tym
raporcie opisana jako pochodzaca ze "starej tabeli, ktorej protokolu nie da sie odtworzyc":

    tamte liczby musialy byc czytane przy bardzo niskiej skali, gdzie baseline prawie nie
    adaptuje i nie ma czego zapominac, ale jego tozsamosc jest daleko ponizej naszej

Zapominanie rosnie monotonicznie ze skala u KAZDEJ metody, wiec odczyt przy jednej skali nie
znaczy nic. `cidm_rerun` jest wiec pulapka, nie wynikiem, a **`bench10-gen` jest jego
zastapieniem, nie powtorka**.

Zastrzezenie do `bench10-gen`, ktore zostaje w mocy: runner przekazuje `--final_only`, wiec da
wiersz koncowy, a nie macierz. Forgetting CIDM na kontrolowanej stopie **nadal nie jest
liczony** i wymaga macierzy 55 komorek przy kilku skalach, zeby dalo sie go odczytac przy
zrownanym TA. To jest osobne, duze zadanie (~28 h na skale).

## Czyszczenie: nie ma czego czyscic

Polecenie "wyczysc nieaktualne wyniki" sprawdzone przed wykonaniem i **nie wykonane celowo**:

* `$SCRATCH` 3,01 TiB z 12 (**25,1%**), 446 tys. plikow z miliona (44,6%) -- zero presji;
* trzy katalogi `cidm_rerun*` wazza **lacznie 120 KB** (same jsony, obrazy dawno wyczyszczone),
  wiec kasowanie oszczedza nic, a niszczy jedyny ocalaly slad pomiaru;
* najwieksze katalogi (`sdxl` 94 G, `phaseA` 66 G, `phaseB` 34 G, `sweep` 26 G) to historyczne
  fazy, ktore nadal stoja za twierdzeniami w tekscie.

Zamiast kasowac -- **oznaczam**. Ten wpis jest tym oznaczeniem: kazdy, kto trafi na
`cidm_rerun`, ma tu powod, dla ktorego tych liczb nie wolno uzyc. Kasowanie jest nieodwracalne,
a zysk zerowy; oznaczenie jest odwracalne i osiaga to samo.

## Grounding: gain jest domyslny, nie nieobecny

Pytanie WG, czy na SD-1.5 uzywamy `ground_gain`. Uzywamy, na obu backbone'ach -- tylko nie ma
go w zadnym configu, bo to **domyslna wartosc w kodzie**:

* `regional.py:500` -- `gain = float(getattr(m, "ground_gain", 1.0))`, brak atrybutu to 1.0;
* `sampling.py:131` -- `base = float(getattr(manager, "ground_gain_base", 1.0))`, potem
  `manager.ground_gain = base if frac < sched else 0.0`.

Flaga `--ground_gain` w `gen_cifc` to nadpisanie NA INFERENCJI, sonda diagnostyczna. To nia
zmierzono na SD-1.5 koszt wylaczenia groundingu: -8,2 IA i -11,1 DINO.

Wniosek dla SDXL sie wzmacnia: sila groundingu i jej harmonogram sa identyczne na obu
backbone'ach **wlasnie dlatego, ze nikt ich nigdzie nie ustawil** -- tak samo jak beta przed
przestrojeniem 7120 -> 3000.

## Teaser z nowego checkpointu: wygenerowany rano, uzyty dopiero teraz

WG zauwazyl, ze teasera z przetrenowanego checkpointu nie zrobilem. **Mial racje.** Bieg 3184858
skonczyl sie poprawnie o poranku (10 konceptow x 4 ziarna z `p022_v2/hyper_after_task49.pt`,
skala 0,8), ale rozmowa przeszla wtedy na tresc podpisu i **nigdy nie sciagnalem ani nie
obejrzalem wyniku**. `figures/teaser.pdf` byl caly czas stary. Obrazy istnialy, robota nie byla
skonczona.

Sciagniete i obejrzane: material jest mocny. Dziesiec konceptow po CALYCH piecdziesieciu
zadaniach -- corgi, figurka anime, wzmacniacz, plusz Meowth, Funko, gitara elektryczna,
kroliczek, sukienka, osoba -- kazdy rozpoznawalny i **stabilny przez cztery ziarna**. Jedynie
zadanie 5 (koncept STYLU) zmienia scene miedzy ziarnami, co jest oczekiwane, bo styl nie
determinuje tresci.

### Blokada nie jest w obrazach, tylko w skladaniu

`teaser.pdf` jest **skladany recznie poza repozytorium**. W repo sa tylko:

* gotowy `figures/teaser.pdf` (+ `.bak-terms`),
* `render/teaser_picks/` z recznie wybranymi kadrami, po jednym na co piaty koncept.

**Nic w obu repozytoriach nie odwoluje sie do `teaser_picks`** -- sprawdzone grepem. Czyli nie
ma skryptu, ktory z tych kadrow buduje figure, a panel (a) (schemat hipersieci) siedzi w tym
samym PDF-ie. Podmiana panelu (b) nie jest wiec wrzuceniem pliku.

Zbudowane teraz: `outputs/teaser_v2_pasek.png` i `.pdf` -- pasek dziesieciu kadrow (ziarno 7000,
512 px, 300 dpi) gotowy do wstawienia, jesli skladanie odbywa sie poza repo. Wybor kadru na
koncept jest decyzja redakcyjna: arkusz wszystkich czterech ziaren jest w
`outputs/teaser_v2_arkusz.png`.

Do rozstrzygniecia: czy istnieje zrodlo skladania `teaser.pdf` (Illustrator, Figma, skrypt poza
repo), czy budujemy figure od nowa skryptem. Druga opcja jest lepsza dlugoterminowo -- figura
bez zrodla w repo jest nieodtwarzalna, a to dokladnie ten rodzaj dlugu, ktory `run-info.txt`
mial eliminowac dla biegow.

## Zrodlo teasera odnalezione -- i lezalo w katalogu, ktory sie kasuje

WG mial racje, ze budowalem te figure. Nie bylo jej w zadnym z repozytoriow, bo powstawala
w **moim katalogu tymczasowym**: `.claude/jobs/6c41d79b/tmp/teaser/`. System opisuje ten
katalog jako "cleaned up when the job is deleted". **Zrodlo Figury 1 istnialo wylacznie tam.**

Znalezione przez zapis sesji (`timeline.jsonl`), nie przez szukanie po plikach -- po nazwach
i po zawartosci w obu repo nie ma nic, bo `teaser.pdf` byl tam tylko kopiowany jako gotowy PDF.

### Jak ta figura naprawde powstaje

1. kandydaci renderowani per koncept do `p022_s06/<zadanie>/<i>.jpg` (wiele ziaren);
2. `swap_*.py` wybiera JEDEN indeks na koncept, skaluje do 256x256 i zapisuje do `small/`;
3. `teaser_body.tex` odwoluje sie do `small/...`;
4. `build_teaser.sh`: `pdflatex teaser_body.tex` -> `crop_teaser.py` (przyciecie do zawartosci)
   -> `cp teaser.pdf` do `figures/teaser.pdf`.

Skrypty `swap_p022.py`, `swap_p031.py`, `swap_s08.py`, `swap_gr.py` to historia podmian kadrow
miedzy biegami -- czyli doklanie ta operacja, ktorej teraz potrzeba dla v2.

### Zabezpieczone

Caly katalog wazy 548 MB, ale to glownie pule kandydatow, odtwarzalne z checkpointow. Minimum
odtwarzalne to **2,0 MB** i zostalo skopiowane do `figures/teaser_src/`: `teaser_body.tex`,
`teaser_src.tex`, `teaser.tex`, `build_teaser.sh`, `crop_teaser.py`, piec skryptow `swap_*`,
`tighten.py`, `equal_heights.py`, `br_box.py`, `fix_a.py`, `dog_colour.py`, `side_by_side.py`
oraz katalogi `small/` (66 wybranych kadrow) i `icons/`.

Nazwa `teaser_src` jest zgodna z konwencja, ktora juz tam dziala: `drift_src`, `embed_src`,
`forget_src`, `ortho_src`.

**Uwaga do przebudowy:** `build_teaser.sh` ma zaszyta sciezke bezwzgledna do katalogu zadania
(`B="C:/Users/w_gro/.claude/jobs/6c41d79b/tmp/teaser"`). Po przeniesieniu trzeba ja zmienic na
sciezke wzgledna wzgledem polozenia skryptu, inaczej po skasowaniu zadania skrypt bedzie
wskazywal w pustke.

### Szersze ustalenie: `continualhyper-paper` NIE jest repozytorium gita

`git rev-parse` zwraca `fatal: not a git repository`. Stad osiem plikow `main.tex.bak-*`
z recznymi znacznikami czasu w nazwie -- praca jest wersjonowana kopiami. To tez powod,
dla ktorego nie dalo sie odzyskac zrodla teasera z historii: historii nie ma.

Dzis wykonalismy na tym pliku szesc zmian, w tym jedna, ktora omal nie zapisala uszkodzonego
LaTeXa (zjedzony backslash w heredocu). Uratowala ja asercja, nie kopia zapasowa.

### Zrodlo teasera zweryfikowane: przebudowa jest piksel w piksel identyczna

Zabezpieczenie plikow to za malo -- trzeba bylo sprawdzic, czy skopiowane minimum faktycznie
sie buduje. Sprawdzone i tak.

Po drodze wyszedl drugi brak: `teaser_body.tex` uzywa `iclr2027_conference.sty` i `fancyhdr.sty`,
ktorych w `teaser_src/` nie ma -- w katalogu tymczasowym lezaly ich kopie. Zamiast duplikowac
pliki stylow w repo, `build_teaser.sh` ustawia teraz `TEXINPUTS` na korzen repo. Duplikat
prędzej czy pozniej rozjechalby sie z oryginalem.

Weryfikacja koncowa, bo sam rozmiar pliku nie wystarcza:

| | wynik |
|---|---|
| rozmiar | 615798 B po obu stronach, identyczny |
| md5 | **rozny** -- PDF osadza znacznik czasu, wiec zgodnosc bajtowa jest niemozliwa |
| przyciecie | 13,99 x 6,46 cm po obu stronach |
| **piksele przy 150 dpi** | **maks. roznica kanalu 0, brak obszaru roznic** |

Czyli odzyskane zrodlo odtwarza Figure 1 doklanie. Porownywanie haszy byloby tu falszywym
testem i daloby falszywy alarm; wlasciwym testem sa piksele.

Poprawione w `teaser_src/`: `build_teaser.sh` i `crop_teaser.py` wyprowadzaja sciezke
z polozenia pliku zamiast miec ja zaszyta.

**Co zostaje do przebudowy teasera na v2:** napisac `swap_v2.py` na wzor istniejacych
`swap_p022.py` / `swap_s08.py` -- wybrac po jednym kadrze z `outputs/teaser_v2/ck49/task*/s080/`
(cztery ziarna do wyboru, arkusz w `outputs/teaser_v2_arkusz.png`), przeskalowac do 256x256,
zapisac do `small/` i przekierowac odwolania w `teaser_body.tex`. Wybor kadru na koncept jest
decyzja redakcyjna, nie mechaniczna -- to jedyny krok w tym potoku, ktorego nie powinienem
podejmowac sam.

## Zawezenie baz wykonane -- i pociagnelo za soba cztery dalsze zmiany

Decyzja WG (powtorzona, bo raz ja zignorowalem trzymajac temat jako otwarty): reimplementacji
nie da sie wiarygodnie odtworzyc, wiec zostaje **fine-tuning i CIDM**. Wykonane w calosci,
razem z konsekwencjami, ktorych pojedyncza zmiana zdania by nie zalatwila.

**1. Paragraf 5.3.** Cztery wartosci z porzadkiem zastapione jedna: fine-tuning $0{,}184$,
trzydziesci jeden razy wiecej. Z uzasadnieniem, ktore jest mocniejsze niz "reimplementacje sa
niewiarygodne": kazda z trzech strojonych ma **wage regularyzacji, ktora MY dobieralismy**,
a fine-tuning takiego pokretla nie ma, wiec jako jedyny nie moze byc artefaktem naszego
strojenia.

**2. Zdanie o `noreg`.** "piec do dziewieciu razy mniej niz bazy" -> "dziewiec razy mniej niz
sekwencyjny fine-tuning".

**3. Paragraf 5.4.** Nowe liczby T = 50 z poprawionych podpisow (TA 74,5 / IA 73,5 / DINO 0,524
przy 0,45 i TA 71,7 / IA 76,7 / DINO 0,577 przy 0,6) oraz dziesiatka CIFC (0,561 i 0,585 zamiast
0,567 i 0,593). Porownania z EWC i C-LoRA usuniete; akapit oparty teraz na **kontroli oracle**
(DINO 0,419), nad ktora jestesmy o 0,105 i 0,158 przy T razy mniejszym magazynie. To jest
uczciwsze porownanie na tej dlugosci: kontrola ogranicza od gory to, co potrafi trening
per koncept.

**4. Prawy panel Figury 3 przebudowany.** `make_forgetting.py` mial liste `ROWS` z czterema
bazami; zostaly dwie. W kodzie zostal komentarz z usunietymi liczbami i powodem, zeby nikt ich
nie przywrocil bezrefleksyjnie. Podpis figury przepisany na liczbe pojedyncza.

**5. Paragraf 6** przepisany tak, zeby nie-odtwarzalnosc byla POWODEM zawezenia, a nie osobnym
zastrzezeniem obok czterech raportowanych liczb.

### Pulapka, w ktora sam wszedlem i z ktorej wyszedlem

Pisząc §6 stwierdzilem, ze ramiona strojone "sa raportowane w dodatku jako kontrola". **Takiego
dodatku nie bylo** -- napisalem twierdzenie bez pokrycia. Zamiast je wycofac, uczynilem je
prawdziwym: nowa podsekcja `app:weakbaselines` podaje EWC 0,1874, C-LoRA 0,1484, LwF 0,1199
z wyjasnieniem, czemu nie wchodza do tekstu glownego, i z argumentem, ktory te liczby
faktycznie niosa -- **kazde z tych ramion, nawet zle dostrojone, siedzi w granicach dwukrotnosci
fine-tuningu i o rzad wielkosci powyzej naszego**, wiec luka nie bierze sie z oslabienia
konkurencji.

Przy okazji `check_draft.py` wylapal skutek uboczny, ktorego nie przewidzialem: po usunieciu
porownania **`kirkpatrick2017ewc` i `li2018lwf` stały sie cytowaniami nieuzywanymi**. Praca
o uczeniu ciagłym bez cytowania EWC i LwF wyglada zle; nowa podsekcja je przywraca.

Stan po zmianach: **tresc dalej 9 stron**, bibliografia 10-12 (urosla o strone, bo wrocily dwa
cytowania), calosc 22. `check_draft`: zero nieuzywanych wpisow, jedyny problem to `fig:method`
bez etykiety -- istnial wczesniej i nie pochodzi z tych zmian.

### Co z CIDM

Drugi czlon zawezonego porownania **nadal nie istnieje**. `bench10-gen` da wiersz koncowy przy
T = 10 (walidacja lewego panelu), ale nie forgetting, bo runner wymusza `--final_only`.
CIDM-50 generuje sie (~24 h) i da wiersz T = 50. Forgetting CIDM wymaga macierzy 55 komorek
przy kilku skalach i jest niepoliczony.

## CIDM zmierzony naszym torem: opublikowany wiersz ODTWORZONY

`bench10-gen` (3185279) COMPLETED po 2:30:05. Ich kod, ich modele, nasze metryki, dziesiec
konceptow benchmarku:

    Average (final): CLIP-T 0.7602 | CLIP-I 0.7713 | DINO 0.5767

Zestawienie z ich **opublikowanym** wierszem SD-1.5 (TA 74,8 / IA 78,0, zaszyty w
`make_tradeoff.py` jako `PUBLISHED`):

| | TA | IA |
|---|---|---|
| ich publikacja | 74,8 | 78,0 |
| nasz pomiar ich metody | **76,02** | **77,13** |
| roznica | **+1,2** | **-0,87** |

**To rozstrzyga pytanie, ktore wisialo od rana: wolno nam sie z nimi porownywac.** Odtwarzamy
ich wiersz w granicach ~1 punktu na obu osiach, uruchamiajac ICH wydany kod i punktujac NASZYM
potokiem. Dla porownania, nasze reimplementacje pozostalych baz mialy blad 1,5 TA i 2,1 IA
i odwracaly porzadek -- czyli **ich metode odtwarzamy LEPIEJ niz wlasne reimplementacje ich
konkurentow**, co jest dodatkowym argumentem za zawezeniem, ktore wlasnie wprowadzilismy.

### Drugi wynik: porownanie przy zrownanym TA, na naszej stopie

Praca twierdzi dotad, na podstawie ICH opublikowanych liczb, ze przy ich zgodnosci z tekstem
nasze IA jest o **1,9 punktu** wyzsze. Teraz mamy to niezaleznie, z wlasnego pomiaru:

**POPRAWKA.** Napisalem tu najpierw przewage +1,9 IA i +0,037 DINO, zestawiajac nasz punkt
przy TA 75,59 z ich pomiarem przy TA 76,02 -- czyli BEZ interpolacji, porownujac dwa rozne
punkty zgodnosci z tekstem. To jest ten sam blad, ktory wylapalem rano przy kompozycji
(wycinek kontra pelny obraz) i powtorzylem go tutaj.

Poprawnie, interpolujac nasza krzywa (s 0,45: TA 75,59 / IA 79,07 / DINO 0,617; s 0,30:
TA 77,47 / IA 76,97 / DINO 0,573) do ICH zmierzonego TA 76,02:

| przy TA = 76,02 | IA | DINO |
|---|---|---|
| CIDM (nasz pomiar ich kodu) | 77,13 | 0,5767 |
| nasze, interpolowane | **78,59** | **0,607** |
| przewaga | **+1,5** | **+0,030** |

Czyli wobec WLASNEGO pomiaru ich metody przewaga jest **mniejsza** niz wobec ich opublikowanego
wiersza (+1,9 przy TA 74,8). Powod jest prosty: mierzymy ich TA o punkt wyzej, niz sami podaja,
a wyzsze TA na naszej krzywej oznacza nizsze IA. Obie liczby sa prawdziwe, tylko odnosza sie
do roznych punktow -- i w pracy podaje obie, bo wybieranie korzystniejszej byloby naciaganiem.

**DINO jest calkowicie nowe** -- oni go nie publikuja, wiec 0,5767 jest pierwsza liczba
tozsamosci dla ich metody na tym benchmarku, jaka ktokolwiek ma.

### Co to zmienia w tekscie

1. Zdanie o "1.9 points higher" mozna oprzec na dwoch niezaleznych pomiarach zamiast na jednym
   odczycie z ich tabeli.
2. CIDM wchodzi wreszcie do porownania jako **drugi czlon zawezonego zestawu** (fine-tuning
   + CIDM + my) na osi TA/IA/DINO.
3. Forgetting CIDM nadal NIE istnieje: runner wymusza `--final_only`, wiec to jest wiersz
   koncowy, nie macierz 55 komorek.
4. Akapit w §6 mowiacy, ze metoda benchmarku jest "absent for want of a measurement on this
   footing", jest teraz **nieaktualny dla TA/IA/DINO**, a pozostaje prawdziwy tylko dla
   forgettingu. Do przepisania.

## Miara artefaktow: trzy podejscia, trzy porazki i jeden wazny wniosek

Zadanie: zmierzyc czwarte kryterium ("brak szwow i artefaktow na granicach pudelek"), bo
`_compose_score.py` mierzy tylko pierwsze (czy podmiot jest w ramce), a ja piec razy w tej
sesji nazwalem "czysta" scene, ktora byla rozsypana.

Walidacja ustalona z gory: miara musi dac NISKO na `xl_b3000_boot/b4` (obejrzane, dobre --
oba podmioty, czarodziejski kapelusz, spojna scena) i WYSOKO na `a0.2` (panele) oraz `b0k8.0`
(platy futra).

| wersja | co mierzy | b4 (dobre) | b2k4 | a0.2 (zle) | b0k8.0 (zle) | werdykt |
|---|---|---|---|---|---|---|
| 1 | moc gradientu na obrysie / otoczenie | **1,81** | 1,42 | 1,32 | 1,32 | **ODWROCONA** |
| 2 | udzial boku z podwyzszonym gradientem prostopadlym | **0,504** | 0,312 | 0,299 | 0,307 | **ODWROCONA** |
| 3 | czy maksimum gradientu lezy NA linii pudelka | 1,193 | 1,187 | 1,103 | 1,203 | **PLASKA** |

Wersje 1 i 2 sa odwrocone z tego samego powodu: **ostra krawedz na granicy pudelka jest objawem
DOBREGO groundingu** -- poprawnie uformowany podmiot konczy sie na krawedzi ramki -- a zle
obrazy sa rozmyta papka o niskim gradiencie wszedzie. Obie mierzyly jakosc, nie usterke.

Wersja 3 mialaby to omijac, bo testuje nie sile krawedzi, tylko czy jej grzbiet pokrywa sie
z linia pudelka. Wyszla plaska na poziomie przypadku we wszystkich czterech konfiguracjach.

### Wniosek wazniejszy od samej miary

Plaska wersja 3 znaczy, ze **maksimum gradientu przy krawedzi pudelka nie lezy systematycznie
na tej krawedzi w ZADNEJ konfiguracji**. Czyli artefakty, ktore widac na obrazach, nie sa
zwiazane z granicami regionow.

Wracajac do mojego wlasnego opisu sceny 12.4: pofaldowany sufit, blat pod niemozliwym katem,
plywajaca mata, **ukosny** szew, jasny prostokat NIE pokrywajacy sie z ramka. Zadne z tego nie
jest wyrownane do pudelka. To jest **globalna niespojnosc sceny** -- geometria, ktora sie nie
trzyma -- a nie szew na styku regionow.

**Konsekwencja dla strojenia.** Jesli usterki sa globalne, to kappa, bootstrap i alpha ich nie
naprawia, bo wszystkie trzy dzialaja na granicach i wnetrzach pudelek. Caly dzisiejszy przemiat
adresowal inny problem niz ten, ktory widac na obrazach. To tlumaczy, czemu DINO roslo, awarie
regionow spadaly, a obrazy dalej wygladaly zle: poprawialismy obecnosc podmiotow, a psuta jest
scena.

### Nastepne podejscie

Sonda CLIP-owa na realizm: podobienstwo obrazu do "a photograph of a coherent scene" kontra
"a distorted collage with warped geometry". Semantyczna, wiec moze zlapac to, czego gradienty
nie widza. Ta sama walidacja: musi odroznic b4 od a0.2 i b0k8.0.

### Rzecz, ktora trzeba bylo zauwazyc wczesniej

`xl_b3000_boot/b4` -- konfiguracja, ktora WG wskazal jako dobra -- rozni sie od mojego
"najlepszego" `b2k4` trzema rzeczami:

| | b4 (dobre) | b2k4 |
|---|---|---|
| kappa | **1,0** | 4,0 |
| ground_sched | **1,0** (cale odszumianie) | 0,5 (polowa krokow) |
| bootstrap | 4 | 2 |

Caly moj przemiat kappy (2,5 / 4,0 / 6,0 / 8,0) szedl przy `ground_sched 0,5`. Dobry obraz
powstal w PRZECIWNYM rogu: slaby grounding, ale przez wszystkie kroki. **Interakcji tych dwoch
parametrow nigdy nie sprawdzilem**, a "optimum kappa 4,0" jest optimum tylko przy zamrozonym
ground_sched 0,5.

# STAN PROJEKTU -- 2026-09-19, noc

Jeden dokument wejsciowy. Wpisy powyzej sa przyrostowe i jest ich na 300 tysiecy znakow.
Termin **25.09**, czyli szesc dni.

## 1. Praca

Tresc **dokladnie 9 stron** (limit ICLR), bibliografia 10-12, calosc 22, renderuje sie czysto.
**13 `\todo`**. `check_draft`: zero nieuzywanych wpisow bib, jeden problem (`fig:method` bez
etykiety, istnial wczesniej). **Repozytorium pracy NIE jest pod gitem** -- osiem `main.tex.bak-*`
zamiast historii.

### Wpisane dzisiaj

| co | liczba |
|---|---|
| ramie `noreg` (beta = 0) | zapominanie 0,0060 -> 0,0213 |
| `noout` na SDXL | DINO 0,593 -> 0,430 |
| dryf dW, odpornosc na poprawke podpisow | zgodnosc 3-5% na kazdym opoznieniu |
| CIDM zmierzony naszym torem | TA 76,0 / IA 77,1 / DINO 0,577 |
| wiersz T = 50 na poprawionych podpisach | DINO 0,524 (s 0,45) i 0,577 (s 0,6) |
| zawezenie baz do fine-tuningu | §5.3, §5.4, §6, Fig. 3 + nowa podsekcja w dodatku |
| Figura 6 | komorki przy 0,45, podpis poprawiony dwukrotnie |
| teaser | skala usunieta z podpisu; zrodlo odzyskane i zweryfikowane piksel w piksel |

### Zmierzone, NIEwpisane

`clipkey` 0,6146 i `learnv` 0,6153 przy T = 10 (forma klucza obojetna) · pelna krzywa SDXL
i ustalenie, ze `kd128`/`wd0` szkodza przy zrownanym TA · `s1600` 0,524 · `lr3e4` 0,057
(rozbieg treningu, gnorm 941 077 wobec 6,03) · kompozycja -- patrz §4.

## 2. Co sie liczy w nocy

| bieg | co |
|---|---|
| 3185188 -> 98 -> 99 -> 200 | CIDM-50 generacja, 10 z 50 konceptow, ~20 h, lancuch `afterany` |
| 3185917 | kompozycja kappa 1,0 / gs 0,5 |
| 3185913 | kompozycja kappa 1,0 / gs 1,0 |
| 3185914 | kompozycja kappa 4,0 / gs 1,0 |

Helios pusty. Punkt ryzyka: przekazania w lancuchu CIDM -- tam moze cicho stanac.

## 3. Teaser: skala rozstrzygnieta

1600 kandydatow (10 konceptow x 4 skale x 40 ziaren). Srednia najlepszych kadrow: 0,45 -> 0,703;
0,60 -> 0,731; **0,80 -> 0,749**; 1,00 -> 0,725; najlepsza skala PER KONCEPT -> 0,754.

**Strojenie per koncept kupuje 0,005 DINO.** Bierzemy jedna skale **0,80** na caly pasek
i podajemy ja w podpisie. Ubocznie: 0,80 to wartosc z oryginalnego teasera -- wybor byl trafny,
teraz jest zmierzony zamiast przyjety.

## 4. Kompozycja: gdzie naprawde jestesmy

Zmierzone: koszt skladania **-0,097 DINO** i **0 -> 14%** awarii regionu (kontrola solo,
t = -5,06); alpha 0,0 domyka 40% luki (p = 0,004); bootstrap 2 polowi awarie przy czterech
regionach (McNemar p = 0,034).

**Ale tego NIE wolno wpisac do pracy w tej formie.** Wszystkie te liczby mierza obecnosc
podmiotu w ramce, a obrazy psuje co innego. Proba zbudowania miary artefaktow: **trzy wersje,
trzy porazki** (dwie odwrocone, jedna plaska -- szczegoly w rozdziale wyzej). Z tego wyszedl
wniosek wazniejszy od miary:

**artefakty NIE sa zwiazane z granicami pudelek.** Pofaldowany sufit, blat pod niemozliwym
katem, plywajaca mata, UKOSNY szew -- to globalna niespojnosc sceny. A skoro tak, to kappa,
bootstrap i alpha jej nie naprawia, bo wszystkie dzialaja na granicach i wnetrzach pudelek.
To tlumaczy caly dzisiejszy wzorzec: DINO roslo, awarie spadaly, obrazy dalej wygladaly zle.

Obecne w pracy *"we make no claim about multi-concept composition"* jest blizsze prawdzie niz
to, co chcialem wstawic. Propozycje przepisania **wycofalem**.

### Trop na noc

`xl_b3000_boot/b4` (wskazany przez WG jako dobry) rozni sie od mojego "najlepszego" `b2k4`:

| | b4 | b2k4 |
|---|---|---|
| kappa | **1,0** | 4,0 |
| ground_sched | **1,0** (cale odszumianie) | 0,5 |
| bootstrap | 4 | 2 |

Caly przemiat kappy szedl przy zamrozonym `ground_sched 0,5`, wiec "optimum 4,0" jest optimum
tylko w tym przekroju. Trzy biegi w nocy domykaja siatke 2x2.

## 5. Otwarte pytania

1. **SDXL**: ~2 punkty IA ponizej ich wiersza przy spojnym mechanizmie. beta wyczerpana
   (optimum 3000), `kd128`/`wd0` szkodza. Nietkniety trop: sila groundingu, nigdy niestrojona
   pod 280 warstw; sonda `gen_cifc --ground_gain 0` kosztuje jedna ewaluacje.
2. **Forgetting CIDM**: nie istnieje. `bench10-gen` dal wiersz koncowy (`--final_only`),
   macierz 55 komorek to ~28 h. Stary `cidm_rerun` 0,0174 jest NIEUZYTECZNY.
3. **Wiersz SDXL w dodatku**: `ground_800aug` osiaga parytet, ale uzywa regularyzatora na
   czynnikach, czemu przeczy §4. Spojnosc czy lepsza liczba.

## 6. Braki formalne

Lista kontrolna ICLR · zanonimizowany link do kodu · sformulowanie ujawnienia LLM · tabele
per-koncept. Zadna nie wymaga GPU.

## 7. Ryzyka

* **brak gita na pracy** przy siedmiu zmianach w tekscie dzis, w tym jednej, ktora omal nie
  zapisala uszkodzonego LaTeXa; uratowala asercja, nie kopia;
* **zrodla figur w katalogach tymczasowych** -- teaser byl o wlos od utraty, odzyskany
  z `timeline.jsonl`, zabezpieczony w `figures/teaser_src/`. Moga byc inne;
* **`CIDM_OPTS` domyslnie wskazuje v1** -- raz przez to blok poszedl starymi captionami;
* **mierzenie jednego kryterium i raportowanie czterech** -- powtorzone piec razy dzisiaj.

## 8. Sciezka krytyczna

Waskie gardlo nie jest obliczeniowe. Bez GPU: cztery braki formalne, decyzja o wierszu SDXL,
rozstrzygniecie co zrobic z podsekcja o kompozycji. Jedyna pozycja realnie zjadajaca czas to
macierz forgettingu CIDM (~28 h).

## Kompozycja: kappa ma optimum na jednostke powierzchni ramki, nie na kadr

Siatka 2x2 (kappa 1,0/4,0 x ground_sched 0,5/1,0, bootstrap 2, 11 scen) obejrzana przeze mnie
kadr po kadrze, nie scorerem. Wynik obala moja wlasna hipoteze z wieczora, ze roznica miedzy
dobrym `b4` a moim `b2k4` lezy w `ground_sched`.

### Co widac

| konfiguracja | sceny 2-regionowe (3.1, 12.2, 12.3, 12.6) | sceny 3-4-regionowe (3.2, 3.3, 12.4) |
|---|---|---|
| kappa 1,0 | **spojne, na poziomie figur CIDM**: oba podmioty, atrybut z promptu (kapelusz czarodzieja), zgodne cienie i perspektywa | **futro bez formy**: uformowany jest tylko najwiekszy region |
| kappa 4,0 | **rozjezdza**: psy o dwoch glowach, koty rozmyte w plamy | wszystkie podmioty obecne, ale wygladzie wyciete i naklejone |

`ground_sched` okazal sie drugorzedny: przy kappa 4,0 wartosc 1,0 tylko POGARSZA (wiecej
duplikatow glow niz przy 0,5), przy kappa 1,0 obie wartosci daja to samo.

### Dlaczego -- liczba, nie wrazenie

Powierzchnie pudelek w ulamku kadru:

| scena | liczba regionow | powierzchnie |
|---|---|---|
| 3.1 / 12.2 / 12.3 / 12.6 | 2 | 0,199 i 0,243 |
| 3.3 | 3 | 0,129 · 0,172 · 0,139 |
| 3.2 / 12.4 | 4 | 0,100 · 0,156 · 0,099 · 0,126 |

Sceny czteroregionowe maja pudelka o **polowe mniejsze**. I znikaja dokladnie te najmniejsze:
w 3.3 przy kappa 1,0 przezyl **najwiekszy** region (mis, 0,172), a zniknely 0,129 i 0,139;
w 3.2 i 12.4 przezyly 0,156 i 0,126, zniknely oba po 0,099. Zaden wyjatek.

### Czym NAPRAWDE jest ta awaria (sprawdzone w pelnej rozdzielczosci)

Na arkuszu 340 px odczytalem to jako "podmiot nieobecny". W pelnej rozdzielczosci (scena 3.3,
ramki naniesione) widac co innego i to jest wazniejsze: **w kazdej ramce COS jest**. Mis
(najwieksze pudelko, 0,172) jest uformowany czysto. W ramce psa (0,129) jest rozowa gwiazdzista
smuga i rozowy klab futra przy dolnej krawedzi. W ramce kota (0,139) rozmazany rozowo-szary klab
z ledwie zarysowanym pyskiem.

Czyli galaz regionalna DZIALA -- wnosi statystyke koloru i tekstury konceptu -- ale przy zbyt
slabym groundingu na jednostke powierzchni **nie wnosi geometrii**. Awaria to nie "brak
podmiotu", tylko **futro bez formy**. To ta sama wielkosc, ktora skalowanie powierzchnia ma
podniesc, wiec diagnoza sie trzyma, ale opis awarii w poprzednim akapicie byl zbyt zgrubny.

**Wniosek: sila groundingu jest wielkoscia na jednostke powierzchni ramki.** Jedna globalna kappa
nie moze byc dobra dla obu rozmiarow -- wartosc, ktora wystarcza do uformowania podmiotu
w malym pudelku, w duzym maluje nim po kadrze. To tlumaczy, czemu przemiat kappy (2,5-8,0)
usredniony po 11 scenach dawal plaskie optimum 4,0: usredniał dwa przeciwne rezimy.

### Poprawka (commit c6fa1f6)

`--kappa_area_ref/--kappa_area_pow` mnoza kappe per region przez `(ref/area)**pow`; manifest
zapisuje mnozniki. Punkt odniesienia 0,22 to srednia powierzchnia pudelka w scenach
dwuregionowych, czyli w tych, na ktorych kappa 1,0 dziala. Mnozniki: 0,91 (0,243) do 2,22 (0,099)
przy pow 1,0 i 0,95 do 1,49 przy pow 0,5 -- oba przedzialy leza miedzy kappa 1 a kappa 4,
czyli tam, gdzie optimum musi byc.

Biegi: **3186441** (pow 1,0) i **3186447** (pow 0,5), bootstrap 2, gs 1,0, 11 scen.

### Konsekwencja dla pracy, niezaleznie od wyniku tych biegow

Sceny dwuregionowe przy kappa 1,0 sa **dobre** -- 12.3 (pies i mis w japonskiej alejce) trzyma
perspektywe, cienie i tozsamosc obu podmiotow. To jest mocniejsze i uczciwsze twierdzenie niz
obecne w pracy *"we make no claim about multi-concept composition"*: mechanizm sklada dwa
koncepty, a degraduje sie z liczba regionow, i wiadomo dlaczego. Do decyzji z WG -- NIE wpisuje
tego sam.

## Skalowanie kappy powierzchnia: dziala, i obala moj wczorajszy wniosek o artefaktach

Bieg 3186487 (`xl_kA1`, ref 0,22 pow 1,0, mnozniki 0,91-2,23) i 3186490 (`xl_kA05`, pow 0,5,
mnozniki 0,95-1,49), oba bootstrap 2, gs 1,0, kappa bazowa 1,0, dwa ziarna, 11 scen.

### Podmioty sie pojawily

Przy `xl_kA1` scena 3.2 daje 4 z 4, 3.3 daje 3 z 3, 12.4 daje 4 z 4 -- wobec odpowiednio 2, 1 i 2
przy jednej globalnej kappa 1,0. **Kot w scenie 3.3 ma zloty medal**, czyli atrybut z promptu
(`a cat wearing a medal`), ktorego wczesniej nie bylo w ogole. Hipoteza o skalowaniu
powierzchnia jest potwierdzona.

### KOREKTA: artefakty SA zwiazane z granicami pudelek

Wczoraj zapisalem, po trzech nieudanych miarach szwu, ze *"artefakty NIE sa zwiazane z granicami
pudelek"* i ze kappa, bootstrap i alpha nie moga ich naprawic. **W tym rezimie to nieprawda.**
W pelnej rozdzielczosci (nie na arkuszu 340 px) widac:

* scena 3.3 -- twarda pozioma nieciaglosc przez CALY kadr na y ~ 0,488, czyli dokladnie na gornej
  krawedzi wszystkich trzech pudelek; zmiana tla (trybuny -> biala firana) na x ~ 0,712, czyli na
  lewej krawedzi pudelka kota;
* scena 12.4 -- duzy pies ma biale **prostokatne halo** w ksztalcie wlasnego pudelka i wyglada jak
  zdjecie przyklejone do sciany.

Tamten wniosek wyciagnalem z konfiguracji kappa 4,0 / gs 0,5, gdzie obrazy byly rozmyta papka bez
wyraznych krawedzi gdziekolwiek -- i dlatego miara szwu wychodzila plaska. Po podniesieniu
groundingu krawedzie sa i sa wyrownane do ramek. Miara szwu (wersja 3) byla prawdopodobnie
poprawna, tylko mierzona na materiale, w ktorym nie bylo czego mierzyc.

### Wykladnik: dwa ziarna daja przeciwne odpowiedzi

| ziarno | obecnosc podmiotow | wtopienie w scene |
|---|---|---|
| 0 | kA1 lepsze (4/4 vs 2/4 w 3.2) | kA05 lepsze (12.4 bez halo, mis w 12.3 z cieniem) |
| 1 | remis, oba 4/4 w 3.2 i 12.4 | kA05 wyraznie lepsze (kA1 nadmuchuje: w 3.3 leb psa wielkosci misia, uciety ramka) |

Wniosek: **roznica na obecnosci miesci sie w szumie miedzy ziarnami przy n = 2, a na wtopieniu
pow 0,5 wygrywa 2:0.** Nie rozstrzygam tego przy dwoch ziarnach -- decyzja po biegu z n = 4.

### Nastepna os: feather

Halo wokol podmiotu jest objawem za mocnego groundingu **na krawedzi** pudelka, a `--feather`
(nigdy nieuzywany, domyslnie 0) rozmywa maske uzywana WYLACZNIE do scalania predykcji; bootstrap
dostaje maske twarda, wiec nie grozi to rozsypaniem obrazu opisanym w komentarzu przy
`hard_masks`. Cztery biegi: **3186663** (pow 1,0 f2), **3186664** (pow 1,0 f4),
**3186673** (pow 0,5 f2), **3186674** (pow 0,5 f4).

## Feather: pomaga TYLKO przy mocnym groundingu (cztery biegi, ziarno 1)

| baza | f0 | f2 | f4 |
|---|---|---|---|
| pow 1,0 (mnozniki 0,91-2,23) | halo wokol owczarka w 12.4, leb psa wielkosci misia w 3.3 | **najlepsze**: halo wyraznie slabsze, mis w 12.3 dostaje nogi i cien na bruku | halo jeszcze slabsze w 12.4, ale 12.3 znow nadmuchane; w 3.3 zdublowana glowa |
| pow 0,5 (mnozniki 0,95-1,49) | 3 z 3 w 3.3, 4 z 4 w 12.4 | **szkodzi**: pies w 3.3 rozplywa sie w klab futra, lewy pies w 12.4 traci glowe | jak f2, pies w 3.3 to zlepek |

Interpretacja: feather odejmuje wage predykcji regionu przy krawedzi, czyli **efektywnie obniza
grounding**. Przy pow 1,0 jest co zmiekczac; przy pow 0,5 spycha z powrotem w "futro bez formy".
Feather i kappa dzialaja na te sama wielkosc i nie wolno ich stroic osobno.

**Najlepsza baza na teraz: pow 1,0 + feather 2.**

## Nowa os: ostrosc maski GROUNDINGU (`--ground_sharp`, commit 66bd32b)

`--feather` rozmywa maske SCALANIA. Maska **groundingu** to osobna rzecz -- iloczyn sigmoidow
w `manager.geo_inside` o zaszytej ostrosci `sh = 40.0`. Przy mapie 8x8 (komorka 0,125 kadru)
sigmoid(40 * 0,125) = 0,993, czyli maska jest twardym prostokatem i koncept domalowuje sie az do
linii pudelka. To jest bezposrednia przyczyna dwoch usterek, ktore zostaly:

* podmiot wypelnia CALE pudelko i urywa sie na jego krawedzi zamiast byc calym zwierzeciem
  stojacym w scenie (owczarek w 12.4 to popiersie wiszace w powietrzu, bo jego ramka
  0,205-0,583 nie dotyka zadnej powierzchni);
* prostokatne halo dokladnie w ksztalcie ramki.

Domyslna wartosc zostaje 40.0, wiec checkpointy zachowuja sie bitowo tak samo. Biegi:
**3186697** (sh 16) i **3186698** (sh 10), oba na bazie pow 1,0 + f2.

**Ryzyko do sprawdzenia na obrazach, nie w glowie:** sigmoid jest symetryczny, wiec nizsza
ostrosc rozlewa maske takze NA ZEWNATRZ ramki. Przy sh 10 zanik siega ~0,1 kadru w kazda strone.
Jesli koncepty zaczna wyciekac poza swoje pudelka, ta os jest slepa w tej postaci i trzeba by
najpierw sciagnac ramke, potem rozmyc.

## Dlug techniczny znaleziony przy okazji

`src/manager.py` mial w komentarzu w linii 129 bajt 0xb3 (cp1250 "l z kreska") i przez to **nie
byl poprawnym UTF-8** -- rozbil mi skrypt czytajacy plik jako UTF-8. Python go tolerowal, wiec
nigdy nie wyszlo. Naprawione w 66bd32b (bajtowo, zeby nie ruszyc niczego innego); reszta
komentarzy w repo i tak jest bez ogonkow.

## Poziom sklejania: zaden z naszych torow nie odtwarza ich rownan doslownie

**Najpierw korekta wlasnego wpisu sprzed kwadransa.** Napisalem, ze `--mode single` jest wiernym
odtworzeniem CIDM, a `unp1` naszym wariantem. Jest **odwrotnie** co do liczby przebiegow: wpis
z wrzesnia (REPORT linia ~1358) ustala, ze `unp1` (2+U wywolan UNeta) to ICH schemat, a `single`
(2 wywolania niezaleznie od U) to NASZ tor pod teze `O(1)`, ktory kasuje ich argument kosztowy.

Ale przy okazji tej pomylki wyszlo cos, co stoi: **poziom, na ktorym sklejamy, rozni sie w obu
torach od ich tekstu.** Sekcja 4.3, znacznik `[PAPER]`:

> the values of **f**^l inside the bounding boxes are respectively **replaced** with the
> corresponding region features [...] to **all layers** of eps(.)

| | liczba przebiegow | co sie sklada | zgodnosc z ich tekstem |
|---|---|---|---|
| ich tekst | U+1 (zeby miec `f_u^l`) | CECHY, na kazdej warstwie | — |
| nasz `unp1` | U+1 | **predykcje epsilon**, raz na krok | liczba przebiegow tak, poziom NIE |
| nasz `single` | 2 | wyjscie attn2 wewnatrz ramki | poziom blizej, liczba przebiegow NIE |

To jest istotne dla usterki, ktora zostala. W `unp1` wnetrze ramki pochodzi z innej trajektorii
odszumiania niz tlo i sklejamy dwa rozne pola epsilon wzdluz prostokata -- przepis na szew
i prostokatne halo, czyli dokladnie na to, co widze w 3.3 i 12.4. Sklejanie na poziomie CECH,
kilkadziesiat warstw przed wyjsciem, zostawia sieci miejsce, zeby to zintegrowala. Jesli tak
jest, to usterka jest w POZIOMIE sklejania, a nie w sile groundingu -- i zaden przemiat kappy,
feathera ani ostrosci jej nie usunie.

`single` nie testuje tego czysto (zmienia tez liczbe przebiegow), ale jest tanie i rozstrzyga
kierunek. Biegi: **3186705** (`single` + skalowanie powierzchnia pow 1,0) i **3186706**
(`single`, kappa globalna 1,0, kontrola). Jesli szwy znikna, warto zrobic wariant wlasciwy:
U+1 przebiegow, ale sklejanie na cechach.

Wymagalo poprawki: `kappa_mul` wchodzil przez `manager.ground_gain` ustawiany w petli regionow,
a ta petla istnieje tylko w `compose_sample_regions`. W `single` wszystkie regiony ida jednym
przebiegiem, wiec `_gsa` dostaje teraz region i mnozy sam (commit e0e97b9).

## Korekta wlasnej oceny: ogladalem metode na trzech najtrudniejszych scenach z jedenastu

Trzy rzeczy wyszly w tej rundzie i wszystkie trzy zmieniaja obraz na korzysc metody.

### 1. Ostrosc maski groundingu to slepa os (biegi 3186697, 3186698)

`sh 16` wobec `sh 40` przy tej samej reszcie: w 3.3 **gubi psa i kota** (zostaje sam mis na
rozmazanym tle), w 12.3 lekko pomaga, w 12.4 i 3.2 bez zmian. Czyli obniza efektywny grounding
dokladnie tak jak feather i jak nizsza kappa. **Trzy osie -- wielkosc kappy, feather, ostrosc --
ruszaja te sama skalarna wielkosc i zadna nie usuwa usterki brzegowej.** To jest wynik
negatywny, ale czysty: usterka nie jest parametrem sily.

### 2. "Halo" to bylo zle nazwane -- to podmiot uciety ramka

Z naniesionymi ramkami, w pelnej rozdzielczosci, scena 12.4: biala piers owczarka **konczy sie
prosta linia na dolnej krawedzi jego pudelka**, uszy sa uciete na gornej. To nie jest poswiata
wokol ramki, tylko zwierze przyciete do niej. Rownoczesnie podmioty w pudelkach dotykajacych
podlogi (szczeniak, kociak, szary kot) **wychodza poza swoje ramki** i wygladaja naturalnie.

### 3. Dlaczego akurat tam: to jedyne wiszace pudelko w zbiorze

| sceny | ramki konczace sie wyzej niz y = 0,90 |
|---|---|
| **3.2, 12.4** | V1 na **y = 0,57/0,58** -- w polowie kadru, nic pod spodem |
| 3.1, 12.2, 12.3, 12.6, 3.5, 12.5 | y = 0,85-0,88, czyli praktycznie przy podlodze |
| 3.3, 3.4, 12.1 | brak |

A RTP tego regionu brzmi `V1 dog sitting on a bed`, podczas gdy ITP to samo `a bedroom.` --
i globalna galaz nie stawia lozka w tym miejscu. Pies nie ma na czym usiasc, wiec wypelnia
ramke i konczy sie na jej krawedzi. **To rozjazd ramki ze scena, nie usterka mechanizmu.**

### 4. Pozostale osiem scen (kA1f2, ziarno 1) -- ogladniete po raz pierwszy

| scena | ocena |
|---|---|
| 3.5 ksiezyc | **bardzo dobra** -- kotek, corgi, pies, trzy cale zwierzeta na skale, spojne swiatlo |
| 12.5 laka | **bardzo dobra** -- plecak, mis z kokarda, gumowa kaczka, trzy cale obiekty na trawie |
| 12.1 ulica JP | dobra -- shiba, corgi, kot na bruku, tlo z szyldami spojne |
| 3.4 palac | dobra -- mis, pies na czerwonym dywanie, kociak; pies nieco za duzy |
| 12.2 bar | dobra -- dwa psy, spojna perspektywa i odbicia |
| 12.6 zamek | dobra -- corgi i kot, spojne tlo |
| 12.3 ulica | dobra -- pies i mis, mis z cieniem na bruku |
| 3.1 zamek | dobra -- pies i kot w kapeluszu czarodzieja |

**Musze to powiedziec wprost: przez ostatnie godziny oceniałem metode na 3.2, 12.4 i 3.3, czyli
na dwoch scenach z wiszaca ramka i jednej z trzema ramkami scisnietymi w jednym rzedzie.**
Na pozostalych osmiu `kA1f2` daje komplet calych podmiotow w spojnej scenie. To jest ten sam
blad metodologiczny co z DINO -- patrzenie na waski wycinek i uogolnianie -- tylko popelniony
wzrokowo zamiast metryka.

### Co z tego wynika dla nastepnych biegow

Nie stroje juz sily groundingu. Zostaja dwie rzeczy: poziom sklejania (`single`, biegi 3186705
i 3186706) oraz porzadny bieg n = 4 na `kA1f2` przez wszystkie 11 scen, zeby ocena nie stala
na jednym ziarnie.

## `mode=single`: hipoteza obalona, i to potwierdza sierpniowy werdykt

Biegi 3186705 (`single` + skalowanie powierzchnia) i 3186706 (`single`, kappa globalna), ziarno 1:

| scena | `unp1` (kA1f2) | `single` + skalowanie | `single` kontrola |
|---|---|---|---|
| 3.5 ksiezyc | 3 z 3 | **1 z 3** | 1 z 3 |
| 12.5 laka | 3 z 3 | **1 z 3** | 1 z 3 |
| 12.4 sypialnia | 4 z 4 | 2 z 4 | **1 z 4** |
| 3.3 stadion | 3 z 3 | **1 z 3** | 1 z 3 |

I rzecz istotna: **sceny w `single` sa najczystsze z calej nocy** -- zero szwow, spojna
geometria, ladne swiatlo, mis w 3.3 siedzi na murawie bez zadnej nieciaglosci. Tylko jest w nich
jeden podmiot zamiast trzech.

To jest dokladnie sierpniowy werdykt zapisany wyzej w tym pliku ("regionalna uwaga trasuje tresc,
ale nie wymusza liczby podmiotow"; `region_rewrite` redukowal dwa podmioty do jednego) --
tyle ze teraz z galezia groundingu, ktorej w sierpniu nie bylo, i ze skalowana kappa. Galaz
groundingu tego nie ratuje: `sgl1` i `sgl0` roznia sie o jeden podmiot w jednej scenie.

**Wniosek: szew w `unp1` jest cena za to, ze kazdy region faktycznie zawiera swoj koncept.**
U+1 osobnych przebiegow wymusza obecnosc podmiotu; jeden przebieg z podmieniana uwaga nie.
Tej usterki nie da sie usunac przez zmiane poziomu sklejania w te strone. Zamykam ten kierunek.

## Stan kompozycji na koniec nocy

Najlepsza konfiguracja: **`unp1`, bootstrap 2, gs 1,0, kappa bazowa 1,0 skalowana powierzchnia
(ref 0,22, pow 1,0), feather 2, skala LoRA 0,4**. Osiem scen z jedenastu ma komplet calych
podmiotow w spojnej scenie; trzy pozostale to dwie sceny z wiszaca ramka (rozjazd RTP z ITP)
i jedna z trzema ramkami w jednym rzedzie.

Zamkniete kierunki (wszystkie sprawdzone na obrazach, nie w glowie): wielkosc kappy, feather,
ostrosc maski groundingu -- jedna wielkosc, zadna nie usuwa usterki brzegowej; poziom sklejania
przez `single` -- kosztuje podmioty.

Biegi rozstrzygajace wykladnik przy czterech ziarnach: **3186708** (pow 1,0 + f2) i **3186709**
(pow 0,5), oba n = 4, wszystkie 11 scen.

## Atrybuty z promptu: renderuja sie tylko te NOSZONE, nie te bedace osobnym obiektem

Kryterium akceptacji ma cztery czesci i trzecia -- atrybuty z promptu -- nie byla dotad
sprawdzona systematycznie. Szesc regionow w zbiorze ma atrybut mozliwy do zweryfikowania
wzrokowo. Sprawdzone w pelnej rozdzielczosci na `kA1f2`, ziarno 1:

| scena | RTP | jest? |
|---|---|---|
| 3.1 | `cat wearing wizard hat and wizard cloak` | **TAK** -- fioletowy kapelusz |
| 3.3 | `cat wearing a medal` | **TAK** -- zloty medal na szyi |
| 12.3 | `teddy bear riding a bike on the street` | NIE -- mis stoi na bruku, roweru nie ma |
| 12.2 | `dog drinking a cup of beer` | NIE -- dwa psy, zadnego kufla |
| 3.4 | `dog wearing a suit, sitting on a throne` | NIE -- czerwony dywan, ale ani garnituru, ani tronu |
| 12.6 | `cat wearing wizard hat` | **ZALEZY OD ZIARNA** -- ziarno 0 ma fioletowy kapelusz i obroze, ziarno 1 nie ma nic |

Trzy braki sprawdzone na OBU ziarnach (nie tylko na jednym, bo 12.6 pokazalo, ze to ma
znaczenie): roweru, kufla ani tronu nie ma przy zadnym. Sam garnitur tez nie -- w 3.4 pies siedzi
na czerwonej poduszce, ktora mozna czytac jako podest, ale tronem nie jest.

**Wzorzec jest ostry: renderuja sie atrybuty NOSZONE na zwierzeciu i male (kapelusz, medal),
a nie renderuja sie OSOBNE OBIEKTY, z ktorymi zwierze wchodzi w interakcje (rower, kufel, tron,
garnitur).**

### Hipoteza mechanistyczna

`ground_sched_frac = 1.0` trzyma grounding przez CALE odszumianie. Adapter konceptu jest
trenowany na samym zwierzeciu, wiec przez wszystkie 50 krokow wstrzyk dopycha do ramki cechy
konceptu i nadpisuje wszystko, co nim nie jest. Region nigdy nie dostaje krokow, w ktorych
moglby zbudowac rower. Kapelusz i medal przechodza, bo leza NA zwierzeciu i nie koliduja
z jego sylwetka; rower wymaga wlasnej struktury w tym samym pudelku.

To tlumaczy tez, czemu przemiat `ground_sched` w siatce 2x2 wyszedl "drugorzedny": mierzylem
nim OBECNOSC podmiotow, a on rzadzi czyms innym -- tym, ile krokow zostaje na wszystko poza
podmiotem. Kolejny raz ten sam blad: jedno kryterium mierzone, cztery raportowane.

Biegi: **3186714** (gs 0,5) i **3186715** (gs 0,7), oba na `kA1f2`.

Przewidywanie zapisane PRZED obejrzeniem wynikow, zeby nie dopasowac go potem: przy gs 0,5
spodziewam sie roweru i kufla albo ich zarysow, kosztem nieco slabszej tozsamosci zwierzat,
i braku zmiany w liczbie obecnych podmiotow (bo o niej rozstrzygaja pierwsze kroki, ktore
grounding i tak obejmuje).

## Wykladnik rozstrzygniety na czterech ziarnach: pow 1,0 (biegi 3186708, 3186709)

Licznik kompletu podmiotow w 12 komorkach (sceny 3.2, 3.3, 12.4 -- jedyne wieloregionowe, wiec
jedyne rozstrzygajace -- razy cztery ziarna):

| konfiguracja | komplet podmiotow | gdzie brakuje |
|---|---|---|
| **pow 1,0 + feather 2** | **12 z 12** | — |
| pow 0,5 | 7 z 12 | 3.2/z0 (2 z 4), 3.3/z0 (2 z 3), **3.3/z2 (1 z 3)**, 12.4/z0 (3 z 4), 12.4/z3 (3 z 4) |

**Korekta wlasnego wniosku.** Kilka godzin temu zapisalem, ze "roznica na obecnosci miesci sie
w szumie miedzy ziarnami, a na wtopieniu pow 0,5 wygrywa 2:0". Pierwsza polowa tego zdania byla
falszywa i wynikala z ogladania dwoch ziaren. Przy czterech pow 0,5 gubi podmiot w pieciu
komorkach na dwanascie -- to nie jest szum, to systematyczna slabosc. Druga polowa sie broni:
tam, gdzie pow 0,5 zbuduje podmiot, wtopienie jest czystsze. Ale obecnosc podmiotu jest
warunkiem koniecznym, a wtopienie stopniowalne, wiec wybor jest jednoznaczny.

Ubocznie: **ziarno 0 jest systematycznie najgorsze dla pow 0,5** (braki w wszystkich trzech
scenach), co jest dodatkowym argumentem, zeby nie oceniac konfiguracji na jednym ziarnie.

### Konfiguracja przyjeta

`unp1`, bootstrap 2, `ground_sched` 1,0, kappa bazowa 1,0 skalowana powierzchnia
(`--kappa_area_ref 0.22 --kappa_area_pow 1.0`), `--feather 2.0`, skala LoRA 0,4.

Utrzymujace sie usterki, obie zdiagnozowane i obie NIE bedace parametrem sily:
* podmiot uciety dolna krawedzia ramki w 3.2 i 12.4 -- jedyne dwa wiszace pudelka w zbiorze,
  z RTP `dog sitting on a bed` przy ITP `a bedroom.`, ktore lozka w tym miejscu nie stawia;
* scisk w 3.3, gdzie trzy ramki leza w jednym pasie y 0,49-0,97.

## `ground_sched` nie odpowiada za atrybuty -- moje przewidywanie bylo BLEDNE

Biegi 3186714 (gs 0,5) i 3186715 (gs 0,7) wobec gs 1,0, sceny 12.3 (rower), 12.2 (kufel),
3.4 (garnitur + tron), oba ziarna.

Zapisalem przed obejrzeniem: *"przy gs 0,5 spodziewam sie roweru i kufla albo ich zarysow,
kosztem nieco slabszej tozsamosci zwierzat"*. **Nic z tego sie nie stalo.** Przy zadnym
z trzech poziomow nie ma ani roweru, ani kufla, ani tronu. Przy ziarnie 1 trzy wiersze sa
niemal nieodroznialne. Jedyna roznica to strata: przy gs 0,5 i ziarnie 0 znika drugi pies
w 12.2.

**Wniosek mocniejszy niz sam test: `ground_sched` w tej konfiguracji nie robi prawie nic.**
Obraz rozstrzyga sie na wczesnych krokach, wiec grounding w polowie drugiej jest juz nieistotny
i jego wylaczenie niczego nie uwalnia. To zamyka czwarta os po kappa, featherze i ostrosci --
i tlumaczy, czemu siatka 2x2 uznala gs za "drugorzedny": on naprawde jest drugorzedny, tylko
wtedy wyciagnalem z tego zly wniosek (ze roznice robi kappa w interakcji z nim).

### Gdzie wiec lezy przyczyna

Nie w sile groundingu i nie w harmonogramie. Zostaja dwie mozliwosci:

1. **pudelko jest za male na atrybut** -- kapelusz i medal leza NA zwierzeciu i mieszcza sie
   w tej samej ramce; rower, kufel i tron potrzebuja wlasnego miejsca, ktorego w ramce nie ma;
2. **wiernosc tekstowi** -- adapter konceptu przy skali 0,4 plus grounding dominuja nad
   tokenami atrybutu.

Druga ma tanie pokretlo: CFG. Biegi **3186728** (cfg 10) i **3186729** (cfg 13) na przyjetej
konfiguracji.

**Uwaga proceduralna do zapamietania:** CIDM uzywa cfg 7,5 (`[CODE]`, z ich `inference.py`).
Jesli wyzsze cfg naprawi atrybuty, to figura porownawcza zrobiona przy cfg 10 odbiega od ich
protokolu i trzeba to albo ujawnic w podpisie, albo zostawic 7,5. **To jest decyzja WG,
nie moja** -- ja tylko zmierze, czy cfg w ogole cokolwiek zmienia.

## Uciety podmiot: moje wyjasnienie bylo BLISKIE, ale mechanizm jest inny

Test falsyfikujacy na wlasnym wyjasnieniu ("pudelko wisi w powietrzu, wiec pies wypelnia ramke
i konczy sie na jej krawedzi"). Scena 12.4, ta sama konfiguracja, dwa ziarna, ramka V1
naniesiona:

* **ziarno 3** -- globalna galaz postawila biale poslanie pod ramka. Pies **NIE jest uciety**:
  siedzi caly, jego tulow schodzi PONIZEJ linii pudelka na posciel. Obraz jest dobry.
* **ziarno 1** -- lozka nie ma, ramka wisi nad firana. Pies uciety w pasie, z oderwana lapa
  wystajaca z lewej.

**Korekta.** Pudelko NIE przycina podmiotu -- podmioty swobodnie z niego wychodza, ziarno 3 to
pokazuje wprost. Mechanizm jest taki: predykcja regionu konczy sie na ramce, a ponizej pracuje
juz sama galaz globalna (poza pudelkami `eps = eps_global`, bo `wsum = 0`). Ta galaz nie wie nic
o adapterze, ale widzi CZESCIOWO UFORMOWANEGO psa w latencie powyzej i **kontynuuje tulow, jesli
kontekst sceny na to pozwala**. Posciel pozwala; firana nie, wiec ciało urywa sie na granicy.

To domyka wszystkie dzisiejsze wyniki negatywne w jedna spojna calosc: kappa, feather, ostrosc
maski i harmonogram dzialaja WEWNATRZ pudelka, a ta usterka rozstrzyga sie POZA nim, w galezi
globalnej. Zaden z tych czterech parametrow nie mogl jej ruszyc i zaden nie ruszyl.

**Konsekwencja praktyczna:** usterka jest stochastyczna po ziarnie i zalezy od tego, czy ITP
`a bedroom.` trafi lozkiem w to miejsce. Bogatsze ITP podnioslby trafialnosc, ale ITP jest ICH
-- pochodzi z warstwy tekstowej ich figury -- wiec zmiana to odejscie od protokolu. **Do decyzji
WG.** Dotyczy dwoch scen z jedenastu (3.2 i 12.4, oba maja RTP `dog sitting on a bed`).

## Pelna ocena przyjetej konfiguracji: 11 scen x 4 ziarna, obejrzane kadr po kadrze

Konfiguracja: `unp1`, bootstrap 2, gs 1,0, kappa 1,0 skalowana powierzchnia (ref 0,22, pow 1,0),
feather 2, skala 0,4, cfg 7,5. Bieg 3186708.

| scena | reg. | podmioty (4 ziarna) | scena | atrybut |
|---|---|---|---|---|
| 3.1 zamek | 2 | 4/4 komplet | spojna | kapelusz czarodzieja **1 z 4** |
| 3.2 sypialnia | 4 | 4/4 komplet | pies uciety, gdy brak lozka | — |
| 3.3 stadion | 3 | 4/4 komplet | **scisk**, podmioty w jednym pasie | medal obecny |
| 3.4 palac | 3 | 4/4 komplet | spojna | tron ~1 z 4, **garnitur 0 z 4** |
| 3.5 ksiezyc | 3 | 3-4/4 komplet | spojna, dobre swiatlo | — |
| 12.1 ulica JP | 3 | 4/4 komplet | spojna | — |
| 12.2 bar | 2 | 4/4 komplet | spojna, odbicia w podlodze | **kufel 0 z 4** |
| 12.3 alejka | 2 | 4/4 komplet | spojna, cienie na bruku | **rower 0 z 4** |
| 12.5 laka | 3 | 3-4/4 komplet | spojna (z2 renderuje sie jako obraz w ramie) | — |
| 12.4 sypialnia | 4 | 4/4 komplet | pies uciety, gdy brak lozka | — |
| 12.6 zamek | 2 | 4/4 komplet | spojna | kapelusz **2 z 4** |

### Werdykt

**Obecnosc podmiotow: rozwiazana.** We wszystkich 11 scenach i 4 ziarnach komplet konceptow
pojawia sie niemal zawsze. To jest zmiana wobec stanu sprzed nocy, gdzie sceny 3- i 4-regionowe
dawaly 1-2 podmioty z 3-4.

**Spojnosc sceny: dobra w 9 z 11.** Zostaja 3.2 i 12.4 (uciety pies przy braku lozka, przyczyna
w galezi globalnej, nie w regionie) oraz 3.3 (trzy ramki w jednym pasie y, podmioty wychodza
scisniete).

**Atrybuty: to jest luka.** Wzorzec potwierdzony na czterech ziarnach:
* **noszone** (kapelusz, medal) -- pojawiaja sie, ale w mniejszosci ziaren (1-2 z 4);
* **rekwizyty do interakcji** (rower, kufel, tron, garnitur) -- **nie pojawiaja sie nigdy**,
  0 z 4 w kazdej scenie.

To jest jedyne kryterium z czterech, ktore nie jest spelnione. Biegi cfg 10 i 13 (3186728,
3186729) sprawdzaja, czy to kwestia wiernosci tekstowi.

## CFG: pomaga atrybutom NOSZONYM, nie rekwizytom (biegi 3186728, 3186729)

cfg 7,5 (ich wartosc, `[CODE]`) wobec 10 i 13, przyjeta konfiguracja, oba ziarna.

| atrybut | rodzaj | cfg 7,5 | cfg 10 | cfg 13 |
|---|---|---|---|---|
| 3.1 kapelusz czarodzieja (z1) | noszony | brak | **granatowy stozek na glowie kota** | **wyrazny kapelusz z rondem + peleryna** |
| 12.6 kapelusz (z1) | noszony | brak | maly ciemny stozek | **czytelny niebieski kapelusz** |
| 12.3 rower (z1) | rekwizyt | brak | ciemny ksztalt za misiem, **niejednoznaczny** | ciemny obiekt, dalej nieczytelny |
| 12.2 kufel (z0) | rekwizyt | brak | brak | brak |
| 3.4 garnitur (z0) | rekwizyt | brak | brak | brak |

**Podzial noszone/rekwizyt utrzymuje sie i teraz ma wyjasnienie.** Atrybut noszony to cecha
POWIERZCHNI podmiotu -- konkuruje z adapterem o te same piksele, wiec wystarczy wzmocnic sygnal
tekstowy, zeby wygral. Rekwizyt to osobny obiekt, ktory musi zajac wlasne miejsce w ramce
juz zajetej przez podmiot; wiekszy CFG go nie tworzy, bo problem nie jest w sile sygnalu
tekstowego, tylko w tym, ze region nie ma dla niego przestrzeni.

**Koszt cfg 13 jest widoczny gołym okiem:** nasycenie i "plastikowy" wyglad, typowa reakcja
SDXL. cfg 10 jest kompromisem bez tego artefaktu.

**Zastrzezenie proceduralne, powtarzam bo wazne:** CIDM generuje przy cfg 7,5. Figura
porownawcza przy cfg 10 odbiega od ich protokolu i wymagalaby ujawnienia w podpisie.
**Decyzja WG.** Ja mierze tylko, o ile to zmienia trafialnosc.

Dwie sceny przy jednym ziarnie to za malo na twierdzenie, wiec bieg potwierdzajacy:
**3186736**, cfg 10, n = 4, wszystkie 11 scen.

## Rekwizyty: test na sile adaptera (biegi 3186743, 3186744)

Hipoteza z poprzedniego wpisu ma dwie galezie i chcialem je rozdzielic renderem solo na PELNYM
kadrze. **Nie da sie tego zrobic flaga `--solo`** -- ona renderuje region osobno, ale nadal
w jego ramce (`compose_sample_single` z jednym regionem), wiec testuje izolacje od innych
konceptow, a nie ograniczenie przestrzenia. Zapisuje, zeby nie probowac tego drugi raz.

Zostaje druga galaz, ktora ma pokretlo zgodne z ich protokolem: **sila adaptera**. Przy skali 0,4
adapter moze dominowac nad tokenami rekwizytu -- ta sama zaleznosc, ktora widac bylo przy
teaserze (skala kupuje tozsamosc kosztem wiernosci promptowi). Biegi: **3186743** (skala 0,25)
i **3186744** (skala 0,55, kontrola w druga strone).

Przewidywanie zapisane PRZED obejrzeniem: jesli przyczyna jest w sile adaptera, przy 0,25
powinien pojawic sie rower albo kufel, a tozsamosc zwierzat wyraznie oslabnac; przy 0,55
rekwizytow ma byc tyle samo co teraz (zero) lub mniej. Jesli przy 0,25 rekwizytow nadal nie ma,
to przyczyna jest w braku miejsca w ramce i jedynym wyjsciem byloby powiekszenie pudelek,
czyli odejscie od ich protokolu -- **decyzja WG, nie moja**.

## Rekwizyty: przyczyna to SILA ADAPTERA, nie brak miejsca (bieg 3186743)

Przewidywanie zapisane przed obejrzeniem sprawdzilo sie. Skala 0,25 wobec 0,4, ta sama reszta,
ziarno 1, w pelnej rozdzielczosci:

| scena | atrybut | skala 0,4 | skala 0,25 |
|---|---|---|---|
| 12.3 | rower | brak | **kompletny rower**: szprychy, rama, kierownica z koszykiem kwiatow |
| 3.1 | peleryna czarodzieja | brak | **granatowa peleryna z kapturem i zlota klamra** |
| 12.2 | kufel | brak | butelka, puszka i zielone naczynie przy psie |
| 3.4 | garnitur | brak | brak (ale sala bardziej tronowa, czerwone draperie) |

**Wniosek: pudelek CIDM ruszac nie trzeba.** Rekwizyt nie powstawal nie dlatego, ze nie ma dla
niego miejsca w ramce, tylko dlatego, ze adapter przy skali 0,4 dominuje nad tokenami promptu.
Przy 0,25 tekst wygrywa i rekwizyt sie pojawia.

### Koszt: tozsamosc. Sprawdzone WZROKOWO wobec referencji, nie metryka

To jest dokladnie ten przypadek, o ktorym WG ostrzegal, ze DINO moze go nie zlapac.

* **pies referencyjny** (`data/CIFC/datasets/images/dog/00.jpg`): puchaty tan-bialy mieszaniec
  o okraglej, lisiej mordce, bialej prędze na pysku, duzych spiczastych uszach. Przy skali 0,4
  zgadza sie. **Przy 0,25 jest to owczarek szetlandzki** -- dluzszy pysk, inne proporcje ciala,
  dluzsza sierc, inna sylwetka.
* **kot referencyjny** (`images/cat/01.jpg`): rudo-bury dlugowlosy kociak, bursztynowo-zielone
  oczy. Przy 0,4 blisko. **Przy 0,25 siwo-bialy, z jaskrawo niebieskimi oczami** -- inny kot.

**Skala 0,25 nie jest wygrana, tylko zamiana: rekwizyty za tozsamosc.** Dwie czesci kryterium
akceptacji stoja tu naprzeciw siebie i nie da sie miec obu przez sama skale.

### Dwie czesciowe naprawy do polaczenia

1. **cfg 10 przy skali 0,4** -- odzyskuje atrybuty NOSZONE (kapelusz w 3.1 i 12.6) bez ruszania
   tozsamosci, ale nie daje rekwizytow;
2. **skala 0,25** -- daje rekwizyty, ale psuje tozsamosc.

Biegi na punkt posredni i na polaczenie: **3186790** (skala 0,32, cfg 7,5) i **3186791**
(skala 0,32 + cfg 10). Do tego pomiar DINO tozsamosci 0,4 wobec 0,25: **3186789** (pierwszy
strzal, 3186788, padl w 9 s -- zapomnialem wymaganego `--config`; poprawione).

## Krzywa zamiany tozsamosc-atrybuty: monotoniczna, bez punktu optymalnego

DINO regionowe (`_compose_score.py`, 62 regiony na konfiguracje, wzorce z `datasets/images/`):

| skala | cfg | DINO wlasne | regiony przegrane z CUDZYM konceptem |
|---|---|---|---|
| 0,55 | 7,5 | **0,7170** | **0 (0%)** |
| 0,40 | 7,5 | 0,6664 | 3 (5%) |
| 0,32 | 10 | 0,6309 | 7 (11%) |
| 0,32 | 7,5 | 0,6192 | 11 (18%) |
| 0,25 | 7,5 | 0,5661 | 12 (19%) |

Atrybuty ida dokladnie w druga strone: rower tylko przy 0,25; kapelusz i peleryna od 0,32;
przy 0,40 i 0,55 nic (poza tym, co daje cfg 10). **Nie ma wartosci posredniej, ktora daje oba.**

### Korekta: "0,32 zachowuje tozsamosc" bylo bledne

Napisalem to po obejrzeniu jednej sceny (pies w 12.3 wygladal poprawnie). Na 62 regionach 0,32
gubi 18%, czyli prawie tyle co 0,25 przy 19% -- a daje mniej rekwizytow. Punktu rownowagi tam
nie ma.

### Czym SA te porazki: myleniem dwoch konceptow tej samej klasy

Rozklad przegranych regionow po konceptach jest jednoznaczny -- **wszystkie** dotycza
`cifc_dog2` (V7) i `cifc_cat2` (V9), nigdy `cifc_dog`, `cifc_cat` ani `cifc_teddybear`.
Przy 0,32 dziewiec z jedenastu to dog2. To nie jest ogolny rozpad tozsamosci, tylko **drugi pies
przegrywajacy z pierwszym psem w tej samej scenie**; obnizanie skali to nasila, bo slabszy
adapter przestaje odrozniac sie od mocniejszego sasiada tej samej klasy.

### Skala NIE jest czescia ich protokolu -- cfg jest

Rzecz, ktora powinienem byl zauwazyc wczesniej. Protokol CIDM to ITP, RTP, pudelka, cfg 7,5,
50 krokow i sampler. **`lora_scale` to parametr NASZEJ hipersieci**, nie ich metody. Strojenie
skali jest wiec uczciwe i nie wymaga zastrzezenia w podpisie; zmiana cfg wymaga.

### Wzrokowo: 0,55 jest lepsze niz 0,4, nie tylko w liczbie

Sprawdzone wobec wzorcow: **kot referencyjny jest rudo-bury**, i przy 0,55 taki wychodzi,
a przy 0,4 siwieje. Pies przy 0,55 ma wyrazniejsza lisia mordke i biala prędze. Komplet
podmiotow utrzymany, sceny bez pogorszenia (12.5: kaczka i plecak wyrazniejsze; 12.4: mniej
poswiaty wokol duzego psa). Moje wczesniejsze "przy 0,4 zgadza sie" bylo zbyt pobłażliwe.

### Rekomendacja (decyzja i tak WG)

**Skala 0,55, cfg 7,5** jako domyslna do figury: zero pomylek konceptow, najwyzsza wiernosc
wzorcom, pelny protokol CIDM. Koszt: atrybuty z promptu sie nie renderuja. Jesli atrybuty maja
byc w figurze, sa dwie drogi o zmierzonym koszcie -- cfg 10 (tylko noszone, wymaga ujawnienia)
albo skala 0,25 (rekwizyty, ale 19% regionow myli koncepty).

Bieg n = 4 na 0,55, wszystkie sceny: **3186800**, zeby rano byly dwa pelne zestawy do wyboru.

## Skala 0,55 na czterech ziarnach: argument, ktorego nie widac w zadnej liczbie

Bieg 3186800, 11 scen x 4 ziarna. Obecnosc podmiotow taka sama jak przy 0,4 (~15 z 16 na
scenach 3.2, 3.3, 12.4, 3.5; jedyny brak to trzeci pies w 3.5 przy ziarnie 2).

**Ale jest roznica jakosciowa, ktora widac dopiero przy czterech ziarnach obok siebie.**
Duzy pies w scenie 3.2:

| ziarno | skala 0,4 | skala 0,55 |
|---|---|---|
| 0 | border collie (czarno-bialy) | corgi referencyjny |
| 1 | owczarek australijski (tricolor) | corgi referencyjny |
| 2 | tan-bialy mieszaniec | corgi referencyjny |
| 3 | szpic | corgi referencyjny |

**Przy 0,4 to MODEL BAZOWY decyduje, jakie to zwierze, a adapter tylko je zabarwia. Przy 0,55
decyduje adapter.** To samo widac w 12.4 i 3.5. Dla pracy, ktorej teza jest zachowanie
tozsamosci konceptu, to jest roznica jakosciowa, nie ilosciowa -- i tlumaczy, czemu DINO rosnie
z 0,666 do 0,717, a liczba regionow mylonych z cudzym konceptem spada z 3 do 0.

Powinienem byl to zobaczyc wczesniej: ogladalem konfiguracje scena po scenie przy JEDNYM
ziarnie, a ta wlasnosc jest widoczna wylacznie w porownaniu ziaren miedzy soba. To trzeci raz
tej nocy, kiedy wniosek z jednego ziarna byl mylacy (poprzednio: wybor wykladnika i "0,32
zachowuje tozsamosc").

# PODSUMOWANIE NOCY 19/20.09 -- KOMPOZYCJA

Punkt wyjscia: sceny 3- i 4-regionowe dawaly 1-2 podmioty z 3-4, reszta byla "futrem bez formy".
Stan koncowy: komplet podmiotow we wszystkich 11 scenach i 4 ziarnach, sceny spojne w 9 z 11.

## Przyjeta konfiguracja

```
--mode unp1 --bootstrap 2 --ground 1 --kappa 1.0 --ground_sched 1.0
--kappa_area_ref 0.22 --kappa_area_pow 1.0 --feather 2.0 --scale 0.55 --alpha 0.1 --cfg 7.5
```

Jedyne dwie zmiany w kodzie: `--kappa_area_ref/--kappa_area_pow` (c6fa1f6, plus e0e97b9 dla
`mode=single`) i `--ground_sharp` (66bd32b, domyslna 40,0 = zachowanie bitowo identyczne).
Reszta to dobor wartosci. **Zadna nie narusza protokolu CIDM** -- skala adaptera jest parametrem
naszej hipersieci, nie ich metody.

## Co rozwiazane

**Skalowanie kappy powierzchnia pudelka.** Jedna globalna kappa nie ma wartosci dobrej dla
wszystkich scen: przy pudelkach 0,20-0,24 kadru dziala 1,0, przy 0,10-0,16 trzeba ~2,2. Znikaly
dokladnie najmniejsze regiony, bez wyjatku. Po poprawce -- komplet podmiotow, 12 z 12 komorek
na scenach wieloregionowych przy czterech ziarnach.

**Skala 0,55 zamiast 0,4.** DINO 0,716 wobec 0,672, regiony mylone z cudzym konceptem 3 wobec 11
(na 124). Ale wazniejszy jest argument, ktorego nie ma w liczbach: przy 0,4 duzy pies w scenie
3.2 byl kolejno border collie, owczarkiem australijskim, mieszancem i szpicem -- inna rasa
w kazdym ziarnie. Przy 0,55 we wszystkich czterech jest ten sam corgi referencyjny. **Przy 0,4
o wygladzie zwierzecia decyduje model bazowy, przy 0,55 adapter.**

## Co pozostaje niespelnione

| kryterium | stan |
|---|---|
| wszystkie podmioty obecne | **spelnione** |
| spojna scena | **spelnione w 9 z 11** (3.2 i 12.4 -- uciety pies; 3.3 -- scisk) |
| atrybuty z promptu | **niespelnione**, w bezposredniej zamianie z tozsamoscia |
| brak szwow na granicach | **niespelnione**, niezaleznie od skali |

**Atrybuty.** Krzywa jest monotoniczna: rower tylko przy skali 0,25 (19% regionow myli
koncepty), kapelusz i peleryna od 0,32 (18%), przy 0,55 nic. Alternatywa: cfg 10 odzyskuje
atrybuty NOSZONE przy zachowanej tozsamosci (kapelusz w 3.1 z 1/4 na 2-3/4, w 12.6 z 2/4 na
3-4/4), ale nie daje rekwizytow i **odbiega od ich protokolu**, wiec wymagaloby ujawnienia
w podpisie. Do decyzji WG.

**Szwy.** Przyczyna ustalona testem falsyfikujacym: predykcja regionu konczy sie na ramce,
a galaz globalna kontynuuje tulow tylko wtedy, gdy kontekst na to pozwala. Gdy pod pudelkiem
jest posciel -- pies jest caly i wychodzi poza ramke; gdy firana -- urywa sie w pasie. To lezy
POZA regionem, wiec zaden parametr groundingu tego nie ruszy.

## Kierunki zamkniete (kazdy sprawdzony na obrazach)

1. **wielkosc kappy, feather, ostrosc maski groundingu** -- trzy pokretla jednej wielkosci,
   zadne nie usuwa usterki brzegowej;
2. **`ground_sched`** -- nie robi prawie nic, obraz rozstrzyga sie na wczesnych krokach;
3. **`mode=single`** -- najczystsze sceny nocy, ale jeden podmiot zamiast trzech; potwierdza
   sierpniowy werdykt, teraz juz z galezia groundingu;
4. **powiekszanie pudelek** -- niepotrzebne, rekwizyty blokuje sila adaptera, nie brak miejsca.

## Bledy metodologiczne, ktore popelnilem tej nocy

Wszystkie tego samego rodzaju i warto je miec spisane:

* **trzy razy wyciagnalem wniosek z jednego lub dwoch ziaren i trzy razy byl bledny** -- wybor
  wykladnika ("roznica w szumie"), "0,32 zachowuje tozsamosc", i najciekawszy: stalosc rasy przy
  0,55 jest widoczna WYLACZNIE w porownaniu ziaren miedzy soba, a ja ogladalem scena po scenie;
* **ocenialem metode na 3 z 11 scen** -- akurat dwoch z wiszaca ramka i jednej ze sciskiem --
  i uogolnialem na calosc;
* **"artefakty nie sa zwiazane z granicami pudelek"** -- wniosek z materialu, w ktorym nie bylo
  zadnych wyraznych krawedzi; po podniesieniu groundingu okazaly sie wyrownane do ramek co do
  piksela;
* **"halo wokol pudelka"** -- to byl podmiot uciety ramka, co widac dopiero z naniesiona ramka
  w pelnej rozdzielczosci.

Wspolny mianownik: **uogolnianie z waskiej probki**. Ten sam blad co z DINO, tylko popelniony
wzrokowo.

## Artefakty do obejrzenia rano

* `outputs/NAJLEPSZE_s055.png` -- po jednym kadrze na scene, ziarno wybrane pomiarem,
  obejrzane przeze mnie;
* `outputs/n4_xl_n4_kA1f2.png` i `outputs/n4_s055.png` -- 4 ziarna x 4 sceny, porownanie skal;
* `outputs/compose_scenes/xl_n4_s055/` i `xl_n4_kA1f2/` -- pelne zestawy 11 scen x 4 ziarna.

## Nietkniety trop: bootstrap w NOWEJ konfiguracji (biegi zglaszane teraz)

Bootstrap maskuje wszystko poza ramka w pierwszych K krokach, zeby podmiot nie mial sie gdzie
uformowac poza pudelkiem. Wartosc 2 dobrano, gdy grounding byl slaby i podmioty w ogole nie
powstawaly ("bootstrap 2 polowi awarie przy czterech regionach", McNemar p = 0,034). Po
skalowaniu kappy powierzchnia grounding jest DUZO mocniejszy -- moze bootstrap nie jest juz
potrzebny, a to wlasnie on wymusza wypelnienie ramki, czyli usterke, ktora zostala.

Przewidywanie zapisane PRZED obejrzeniem: przy bootstrap 0 podmiot powinien przestac konczyc sie
na dolnej krawedzi w 3.2 i 12.4 (bo nic go juz nie zamyka w pudelku), ale moze spasc obecnosc
podmiotow w scenach czteroregionowych -- to byl przeciez powod, dla ktorego bootstrap
wprowadzono. Bootstrap 4 jako kontrola w druga strone: spodziewam sie mocniejszego wypelnienia
ramki i ostrzejszego urwania.

Jesli obecnosc podmiotow sie utrzyma przy bootstrap 0, jest to jedyna zmiana tej nocy, ktora
poprawialaby czwarte kryterium bez zadnego kosztu.

## Bootstrap w nowej konfiguracji: piata zamknieta os (biegi 3186804, 3186805)

Skala 0,55, kappa skalowana powierzchnia, feather 2, ziarno 1. Bootstrap 0 / 2 / 4:

| scena | b0 | b2 | b4 |
|---|---|---|---|
| 12.4 | 4 z 4, corgi uciety ramka | 4 z 4, uciety | 4 z 4, uciety |
| 3.2 | 4 z 4, uciety | 4 z 4, uciety | 4 z 4, uciety |
| 3.3 | 3 z 3, scisk | 3 z 3, scisk | 3 z 3, scisk |
| 12.5 | 3 z 3 | 3 z 3 | 3 z 3 |

**Przewidywanie sie NIE sprawdzilo w obie strony.** Zapisalem, ze bootstrap 0 powinien usunac
urwanie podmiotu na dolnej krawedzi (bo nic go juz nie zamyka w pudelku) kosztem obecnosci
podmiotow w scenach czteroregionowych. Nie stalo sie ani jedno, ani drugie.

### I korekta, ktora wyszla z pomiaru, nie z ogladania

Wiersze wygladaly na arkuszu tak podobnie, ze mialem napisac "bootstrap jest bezczynny". Zmierzylem
roznice pikselowa i **to bylby falsz**: b0 wobec b4 to srednio 22-29 poziomow na kanal przy
maksimum 255, czyli bootstrap realnie zmienia trajektorie probkowania i obrazy sa inne w
szczegolach. Poprawne zdanie brzmi: **bootstrap zmienia probke, ale nie zmienia wyniku na zadnym
z czterech kryteriow.**

Wyjasnienie jest zgodne z reszta nocy: bootstrap mial znaczenie, gdy predykcja regionu byla za
slaba, zeby zlokalizowac podmiot (stad "polowi awarie przy czterech regionach", McNemar
p = 0,034, zmierzone przy jednej globalnej kappie 1,0). Po skalowaniu kappy powierzchnia to
grounding robi lokalizacje, wiec maskowanie tla przez 2 z 50 krokow nie ma juz czego zalatwiac.

### Stan przestrzeni strojenia po stronie regionu: wyczerpana

| os | werdykt |
|---|---|
| wielkosc kappy | **rozwiazala obecnosc podmiotow** przez skalowanie powierzchnia |
| feather | pomaga tylko przy mocnym groundingu; 2 przyjete |
| ostrosc maski groundingu | slepa, gubi podmioty |
| `ground_sched` | nie robi prawie nic |
| bootstrap | zmienia probke, nie zmienia wyniku |
| poziom sklejania (`single`) | czyste sceny, jeden podmiot zamiast trzech |
| skala adaptera | **rozstrzyga tozsamosc**, 0,55 przyjete; w zamianie z atrybutami |

Dwa niespelnione kryteria maja przyczyny POZA ta przestrzenia: atrybuty sa w monotonicznej
zamianie ze skala, a urwanie podmiotu na krawedzi powstaje w galezi globalnej. Dalsze strojenie
po stronie regionu nie ma sensu i **na tym koncze ten kierunek**.

## Weryfikacja wlasnego podsumowania na pelnych 44 komorkach

W podsumowaniu napisalem "komplet podmiotow we wszystkich 11 scenach i 4 ziarnach", ale przy
skali 0,55 obejrzalem n = 4 tylko na czterech scenach, a reszte ocenilem przy dwoch ziarnach.
To ten sam blad, ktory tej nocy popelnilem trzy razy, wiec sprawdzilem pozostale siedem scen
na wszystkich ziarnach (dane juz byly, GPU niepotrzebne).

**Korekta: komplet jest w 42 z 44 komorek, nie we wszystkich.** Braki: trzeci pies w 3.5/z2
i plecak w 12.5/z3. To nadal bardzo dobry wynik, ale zdanie bylo zawyzone.

### Cztery obserwacje, ktorych wczesniej nie mialem

1. **"wizard cloak" NIE jest po prostu nieobecny.** W 3.1 i 12.6, w **3 z 4 ziaren**, za kotem
   renderuje sie **bezksztaltna ciemna elipsa** przypominajaca wlot jaskini. Model odpowiada na
   token peleryny, ale produkuje nieustrukturyzowana mase zamiast ubrania. To precyzyjniejszy
   opis usterki niz "brak atrybutu" i pasuje do podzialu noszone/rekwizyt: peleryna jest duza
   i musi oplywac sylwetke, wiec zachowuje sie jak rekwizyt, a nie jak kapelusz.
2. **12.1/z2: postac ludzka** na ulicy, ktorej nie ma w zadnym prompcie (ani ITP, ani RTP).
3. **12.5/z2: cala scena renderuje sie jako obraz w bialej ramie.**
4. **3.4/z0: duzy pies jest czarno-podpalany, nie corgi.** Czyli stalosc rasy przy 0,55 --
   argument, ktory podalem jako najmocniejszy tej nocy -- jest bardzo dobra, ale **nie
   absolutna**: w 3.2, 12.4 i 3.5 trzyma we wszystkich ziarnach, w 3.4 lamie sie w jednym.

Zadna z tych czterech nie zmienia rekomendacji, ale wszystkie powinny byc w tabeli ograniczen,
jesli kompozycja trafi do pracy.

## CIDM-50: tempo spadlo, dolozone piate ogniwo (3186811)

Rachunek na 06:00: gen2 zrobil koncepty j = 14..19, czyli szesc w 3 h 29 min = **34,8 min na
koncept** (wczesniej 30-32). Zostaje 30 konceptow = 17,4 h. W lancuchu jest 20,5 h (gen2 4h31
+ gen3 8h + gen4 8h), ale kazde przekazanie traci koncept w toku, czyli ~35 min razy dwa.
**Efektywny zapas schodzi do okolo 1,9 h, czyli 11%.**

To za malo jak na noc bez nadzoru, wiec dolozylem **3186811** (`afterany:3185200`), zgloszone
dokladnie ta sama komenda co poprzednie ogniwa (odczytana z `run-info.txt` zadania 3185200).
Jesli okaze sie zbedne, wystartuje, wypisze 50 razy "gotowe, pomijam" i wyjdzie po kilku
minutach -- koszt zerowy, a chroni ~20 h GPU.

Sprawdzone przed zgloszeniem: **`CIDM_OPTS` nie wystepuje w `sbatch_cidm_gen.sh`**, wiec
wrzesniowa wpadka z tym zmienna (blok poszedl starymi captionami) tego skryptu nie dotyczy --
wersje zbioru wybiera argument `configs/phaseT/T50_v2.yaml`, ktory przepisalem jeden do jednego.

## Domkniecie dwoch luznych koncow

### 1. `xl_sh10` -- nigdy nie obejrzany, teraz sprawdzony

Os ostrosci maski groundingu zamknalem po obejrzeniu samego `sh16`; `sh10` (bieg 3186698)
wisial nieobejrzany. Sprawdzony, ziarno 1:

| scena | sh 40 | sh 16 | sh 10 |
|---|---|---|---|
| 3.3 | 3 z 3 | 1 z 3 | 1 z 3, jeszcze bardziej wyprany |
| 12.4 | 4 z 4 | 4 z 4 | **2 z 4** |
| 12.3 | 2 z 2 | 2 z 2 | 2 z 2 |

Degradacja jest **monotoniczna**, wiec zamkniecie osi bylo uzasadnione -- ale opieralo sie na
jednej wartosci, a teraz opiera sie na obu. Gdyby sh10 wypadl lepiej niz sh16, wniosek bylby
zly, a ja bym o tym nie wiedzial.

### 2. Mapa katalogow: `outputs/compose_scenes/KONFIGURACJE.md`

Wygenerowana z manifestow (nie z pamieci): 26 katalogow z tej nocy, kazdy z pelna konfiguracja.
Scratch jest czyszczony po 30 dniach, wiec bez tego odtworzenie "ktory katalog to ktora
konfiguracja" wymagaloby `run-info.txt` z zadan, ktore znikna.

Przy okazji wyszla **luka w proweniencji**: `xl_kA1f2`, `xl_kA1f4`, `xl_kA05f2`, `xl_kA05f4`
byly uruchomione z featherem, ale ich manifesty tego nie zapisuja -- pole doszlo dopiero
w 66bd32b, juz po tych biegach. Wartosci sa pewne (z komend zgloszeniowych) i odnotowane
w KONFIGURACJE.md, ale z samego pliku ich nie odczytasz. Katalogi od `xl_sh16` w dol sa w
porzadku.

## Miara szwu DZIALA -- wczorajszy werdykt o niej byl bledny

Wczoraj odrzucilem `_seam_score.py` w wersji 3 jako "plaska na poziomie przypadku we wszystkich
czterech konfiguracjach". **Walidowalem ja na materiale (kappa 4,0 / gs 0,5), w ktorym nie bylo
zadnych wyraznych krawedzi wyrownanych do pudelek** -- wiec nie bylo czego mierzyc. Puszczona na
dzisiejszym materiale rozdziela sceny wyraznie (mediany per scena, `xl_n4_s055`, 44 obrazy;
poziom przypadku = 1,0):

| scena | mediana | moja ocena wzrokowa |
|---|---|---|
| **12.1** | **1,820** | "bardzo dobra" -- **BLAD, patrz nizej** |
| 3.2 | 1,507 | zla (uciety pies) |
| 12.4 | 1,253 | zla (uciety pies) |
| 12.5 | 1,239 | bardzo dobra |
| 12.3 | 1,199 | bardzo dobra |
| 3.4 | 1,192 | przecietna |
| 3.3 | 1,156 | dobra |
| 12.6 | 1,143 | bardzo dobra |
| 12.2 | 1,104 | dobra |
| 3.5 | 1,077 | bardzo dobra |
| 3.1 | 1,071 | dobra |

Obie sceny, ktore ocenilem jako zle, sa na miejscach 2 i 3. Cztery, ktore nazwalem dobrymi,
zamykaja ranking. Zgodnosc jest dobra z JEDNYM wyjatkiem -- i ten wyjatek to moj blad.

### 12.1: miara miala racje, ja nie

W pelnej rozdzielczosci z naniesionymi ramkami scena 12.1 ma **twarda pozioma nieciaglosc przez
CALY kadr na y = 0,493**, czyli dokladnie na gornej krawedzi wszystkich trzech pudelek. Powyzej
szczegolowa japonska uliczka w stylu ilustracyjnym; ponizej trzy zwierzeta w stylu
fotograficznym na rozmytym tle. Ulica po prostu sie urywa. Skala zwierzat tez nie zgadza sie
z tlem -- leb szczeniaka wypelnia cale lewe pudelko.

Na kafelku 290 px odczytalem to jako "trzy cale zwierzeta na spojnej ulicy". **To czwarty raz
tej nocy, gdy pomylilem sie na pomniejszonym kadrze, i pierwszy, gdy zlapala mnie na tym miara,
a nie ja sam.**

Kontrola z drugiego konca rankingu potwierdza, ze to nie przypadek: 3.5 (1,077) i 3.1 (1,071)
w pelnej rozdzielczosci sa czyste -- podmioty wychodza poza ramki, grunt biegnie bez przerwy
przez caly kadr. Miara slusznie NIE liczy bezksztaltnej czarnej "peleryny" wokol kotka w 3.1,
bo ta nie jest wyrownana do ramki; to nie jest szew i nie jej rzecz go widziec.

### Konsekwencje

1. **Mam ilosciowy uchwyt na czwarte kryterium**, ktore do tej pory opieralo sie wylacznie na
   moim oku. Do wpisania w prace, jesli kompozycja tam trafi.
2. **`outputs/NAJLEPSZE_s055.png` przebudowany.** Poprzednia wersja wybierala ziarno po samym
   DINO i wciagnela 12.1/z0 -- kadr z najgorszym szwem w calym zestawie. Nowy wybor bierze
   najnizszy szew sposrod ziaren mieszczacych sie w 0,03 DINO od najlepszego. Zmienilo sie
   siedem scen z jedenastu; najwazniejsza zmiana to 12.1 z ziarna 0 na 1, gdzie ulica biegnie
   za zwierzetami bez nieciagłosci.
3. Sam `_seam_score.py` nie wymagal zadnej zmiany kodu -- byl dobry od wczoraj, zla byla
   walidacja.

## Miara szwu zastosowana miedzy konfiguracjami: skala NIE zmienia szwow

Skoro miara dziala, sprawdzilem nia twierdzenie, ktore postawilem na podstawie **jednej pary
obrazow**: ze przy skali 0,55 poswiata jest slabsza niz przy 0,4. Ziarna sa sparowane, wiec
test parowany, 44 pary:

| porownanie | mediana roznicy | gorzej / lepiej | test znakow |
|---|---|---|---|
| skala 0,55 wobec 0,40 | -0,029 | 21 / 23 | **p = 0,88** |
| cfg 10 wobec 7,5 | +0,048 | 27 / 17 | p = 0,17 |

**Korekta: skala nie ma zadnego wplywu na szwy.** Moje "przy 0,55 halo jest slabsze" bylo
obserwacja z jednego kadru 12.4 i na 44 parach sie nie broni. cfg 10 idzie w strone gorsza,
ale tez nieistotnie.

To jest zgodne z mechanizmem i wlasciwie potwierdza go po raz kolejny: szew powstaje w galezi
globalnej, a skala dziala na adapter. Gdyby skala go zmieniala, moj model przyczyny bylby zly.

Nie zmienia to rekomendacji 0,55 -- ona stoi na tozsamosci (DINO 0,716 wobec 0,672, 3 wobec 11
regionow mylacych koncept, stalosc rasy miedzy ziarnami), a nie na szwach. Ale jeden z argumentow,
ktore przy niej wymienialem, byl nieprawdziwy i zostaje wycofany.

### Zestawienie zmierzonych kosztow opcji cfg 10 (gdyby WG ja wybral)

| wymiar | efekt |
|---|---|
| atrybuty noszone | **poprawa**: kapelusz w 3.1 z 1/4 na 2-3/4, w 12.6 z 2/4 na 3-4/4 |
| rekwizyty | bez zmian (0/4) |
| szwy | nieistotnie gorzej (p = 0,17), maksimum rosnie z 2,41 do 3,21 |
| protokol | **odejscie od CIDM** (oni uzywaja 7,5), wymaga ujawnienia w podpisie |

## KOREKTA: kapelusz czarodzieja JEST przy skali 0,55 (zgloszone przez WG)

WG zauwazyl, ze na `NAJLEPSZE_s055.png` kot nie ma kapelusza. Sprawdzone w pelnej
rozdzielczosci, wycinki regionu V3 we wszystkich czterech ziarnach:

* **3.1, ziarno 0: fioletowy kapelusz czarodzieja z rondem, wyrazny.** Ziarna 1-3: brak.
* **12.6, ziarno 0: to samo.** Ziarna 1-3: brak (w 2 i 3 ciemna bezksztaltna masa w tle).

**Zdanie z wpisu o krzywej zamiany -- "przy 0,40 i 0,55 nic" -- jest falszywe.** Napisalem je po
obejrzeniu arkuszy 290-330 px. To piaty raz tej nocy, gdy pomylilem sie na pomniejszonym kadrze.

### Drugi blad: automatyczny wybor kadru odrzucil jedyne kadry z atrybutem

`NAJLEPSZE_s055.png` wybieral ziarno po DINO i mierze szwu, wiec dla 3.1 wzial ziarno 3, a dla
12.6 ziarno 1 -- czyli **dokladnie te bez kapelusza**. Kryterium optymalizacji nie zawieralo
trzeciej czesci kryterium akceptacji, ktora sam wczesniej wypisalem.

Koszt przelaczenia na ziarno 0 jest maly:

| scena | DINO | szew |
|---|---|---|
| 3.1: z3 -> z0 | 0,856 -> 0,846 (**-0,010**) | 1,037 -> 1,108 (+0,071) |
| 12.6: z1 -> z0 | 0,839 -> 0,853 (**+0,014**) | 0,970 -> 1,282 (+0,312) |

Przy poziomie przypadku 1,0 i najgorszym w zestawie 2,18 to nadal niskie wartosci szwu.
Nowy arkusz: **`outputs/NAJLEPSZE_s055_v2.png`** (stara nazwa byla zablokowana przez otwarty
podglad).

### Dwie rzeczy do ujawnienia, jesli ten kadr trafi do pracy

1. **Trafialnosc atrybutu to 1 z 4 ziaren.** Wybor ziarna 0 pokazuje, ze mechanizm to potrafi,
   ale bez podania tej liczby jest to cherry-picking.
2. **W 3.1 przy ziarnie 0 jest TRZECIE zwierze** po prawej, choc scena ma dwa regiony --
   dorzucila je galaz globalna poza pudelkami.

# KOREKTA OCENY: kompozycja NIE jest na poziomie figur CIDM

WG obejrzal `NAJLEPSZE_s055_v2.png` i powiedzial, ze kompozycja wyglada "mega srednio",
wskazujac dodatkowego psa za kotem w scenie 3.5. **Ma racje, a moje oceny byly zawyzone.**

## Na czym polegal moj blad oceny

Ocenialem kadry po kryterium "czy wszystkie podmioty sa i czy tlo jest wiarygodne". Przy takim
nastawieniu scena, w ktorej sa trzy zwierzeta i sensowne tlo, dostawala "bardzo dobra". **Nie
polowalem na usterki, tylko potwierdzalem sukces.** Po przejsciu na tryb szukania wad
w pelnej rozdzielczosci obraz jest inny.

## Nowa, nieskatalogowana usterka systemowa: zduplikowane kopie POZA pudelkami

| scena | co jest |
|---|---|
| 3.5 | w szczelinie miedzy pudelkiem kota (konczy sie 0,295) a psa (zaczyna 0,328) -- **rude futro, ucho i fragment pyska dodatkowego psa** |
| 12.5 | na lewo od pudelka plecaka **dwa fragmenty misia**: korpus z biala lapa u gory i drugi przy dolnej krawedzi, oba bez glow |
| 3.1 | przy ziarnie 0 **trzecie zwierze** po prawej, choc scena ma dwa regiony |

**Trzy z jedenastu scen w wyselekcjonowanym arkuszu**, czyli w kadrach, ktore wybralem jako
najlepsze. W calym zestawie moze byc ich wiecej -- nie policzone.

Mechanizm jest ten sam co przy uciętym psie: **poza pudelkami `eps = eps_global`**, a galaz
globalna widzi w latencie czesciowo uformowane zwierze i rozwija je dalej, tyle ze bez adaptera
i bez ramki -- wiec powstaje bezglowa, wtopiona w tlo kopia. Przy uciętym psie ta sama galaz
NIE kontynuowala ciala, bo nie bylo na czym; tutaj kontynuuje tam, gdzie nie powinna.

## Pozostale wady, ktore przepuscilem

* **biale obwodki** wokol podmiotow -- wygladaja jak wycinanka naklejona na tlo (12.3, 3.5);
* **brak dolnych partii ciala** -- pies w 12.3 konczy sie rozmyta masa przy dolnej krawedzi
  pudelka, nie ma lap;
* **niezgodnosc skali** -- mis w 12.3 jest wysokosci witryny sklepowej, zwierzeta w 3.5 sa
  wieksze od ksiezyca w tle;
* **podmioty wisza w powietrzu** -- srodkowy pies w 3.5 nie dotyka niczego lapami;
* 3.5: "powierzchnia ksiezyca" to pomarszczona tkanina i porowata skala, a podmioty stoja na
  bladym owalu przypominajacym balon.

## Uczciwy stan czterech kryteriow

| kryterium | stan |
|---|---|
| wszystkie podmioty obecne | **spelnione** (42 z 44 komorek) |
| atrybuty z promptu | niespelnione; kapelusz 1 z 4 ziaren, rekwizyty tylko przy skali psujacej tozsamosc |
| spojna scena | **NIESPELNIONE** -- wczesniej pisalem "9 z 11", to bylo zawyzone; duplikaty, skala i wiszace podmioty dotykaja wiekszosci scen |
| brak szwow i artefaktow | niespelnione |

**Jedyne w pelni rozwiazane kryterium to obecnosc podmiotow.** Poprzednie podsumowanie nocy
(rozdzial "PODSUMOWANIE NOCY 19/20.09") jest w czesci o spojnosci sceny **zawyzone i nalezy je
czytac przez ten rozdzial.**

## Czego NIE probowalem, a moze dotyczyc duplikatow

* **alpha** (waga galezi globalnej, obecnie 0,1 = ich wartosc) -- wyzsza alpha to wiecej
  globalnej predykcji wszedzie, czyli prawdopodobnie WIECEJ duplikatow; nizsza (0,0) byla
  badana wczoraj pod katem obecnosci podmiotow, nie duplikatow;
* **prompt negatywny** -- obecny to ich lista jakosciowa. Dodanie "duplicate, extra animal"
  byloby najprostsza droga, ale to **zmiana protokolu** i wymaga zgody WG.

Obie sa poza tym, co moge rozstrzygnac sam.

## Hipoteza na duplikaty: to `feather` wylewa predykcje regionu poza pudelko

Pomiar zamiast wrazenia. Scena 3.5: pudelko kota konczy sie na x = 0,295, pudelko psa zaczyna
sie na x = 0,328. **Szczelina ma 0,033 kadru, czyli przy latencie 128x128 okolo 4 piksele.**
`--feather 2.0` to gauss o sigmie 2 w pikselach LATENTU, czyli zasieg okolo 6 pikseli
w kazda strone. **Feather przykrywa cala szczeline predykcja regionu** -- i wlasnie tam siedzi
dodatkowy pies. Fragmenty misia w 12.5 rowniez przylegaja do krawedzi pudelka plecaka.

Rzecz istotna dla protokolu: **`feather` jest NASZYM dodatkiem.** Praca CIDM mowi o masce
binarnej (sekcja 4.3, `[PAPER]`: "the binary region mask, where the values inside the bounding
box are set to 1"). Wylaczenie feathera jest wiec **zblizeniem sie do ich metody, nie odejsciem
od niej** -- w odroznieniu od alfy i promptu negatywnego, ktore sa ich.

Feather przyjalem wczoraj, bo przy skali 0,4 zmniejszal poswiaty wokol podmiotow. **Nigdy nie
sprawdzilem go przy skali 0,55**, a tym bardziej pod katem duplikatow, ktorych wtedy jeszcze
nie zauwazylem.

Przewidywanie zapisane PRZED obejrzeniem wynikow: przy `--feather 0` dodatkowy pies w 3.5
i fragmenty misia w 12.5 powinny zniknac albo wyraznie oslabnac, a wrocic moga ostrzejsze
krawedzie na granicach pudelek (to bylo uzasadnienie feathera). Jesli duplikaty zostana,
hipoteza jest zla i przyczyna lezy w samej galezi globalnej, gdzie nie mam juz czystego
protokolowo pokretla.

Bieg: n = 4, wszystkie sceny, te same ziarna co `xl_n4_s055`, wiec porownanie bedzie parowane.

# TEASER: pokazywal STARY checkpoint, a tekst juz nowy (zgloszone przez WG)

WG: "teaser nie jest aktualny, nigdy nie podmieniles w nim obrazkow". **Racja, i moj wpis
w audycie figur ("teaser aktualna, przebudowana wczoraj z nowego checkpointu") byl falszywy.**

Co sie naprawde stalo wczoraj: odzyskalem ZRODLO teasera z katalogu tymczasowego i sprawdzilem,
ze przebudowa daje wynik piksel w piksel identyczny z istniejacym `teaser.pdf`. To byl dowod,
ze odzysk sie udal -- i oznacza dokladnie tyle, ze w `small/` leza STARE klatki. Nowe rendery
zrobilem osobno i ustalilem pomiarem skale 0,80, ale nigdy ich tam nie wstawilem.

## Co znalazlem przy okazji i jest powazniejsze

* pasek bierze `ref_*` (gora) i **`p022_taskNN_*`** (dol); pliki `gen_*`, ktore tez leza
  w `small/`, pochodza z wczesniejszej wersji i **w ogole nie wchodza do figury**;
* stare klatki sa z przebiegu **`p022`**, a nowe rendery z **`p022_v2`** (`_fig_rerender.py`
  czytal `outputs/sweep/p022_v2/ckpts`, checkpoint 49);
* **§5.4 pracy podaje juz liczby v2** -- DINO 0,524 przy s = 0,45 i 0,577 przy 0,6, czyli wiersz
  na poprawionych podpisach.

**Czyli w pracy tekst byl z jednego przebiegu, a figura z drugiego.** To przesadza o podmianie
i o tym, ze musi objac wszystkie dziesiec klatek.

## Koszt: trzy komorki wygladaja GORZEJ, i to jest uczciwe

Ogladniete wszystkie cztery ziarna na komorke, nie tylko wybrane przez DINO:

| komorka | v1 (stara) | v2/ck49 (nowa) |
|---|---|---|
| task05 styl | plaska ilustracja krajobrazu, na styl | **zaden z czterech kadrow nie odtwarza stylu** |
| task10 figurka | rude wlosy, biala peleryna z plomieniem | **blond wlosy i szary mundurek we wszystkich czterech** |
| task45 osoba | w okularach, jak referencja | **w zadnym kadrze nie ma okularow** |
| task30 gitara | zly korpus, bez bialej maskownicy | **Stratocaster jak w referencji** |
| task40 sukienka | GRANATOWA marynarka zamiast czerwonej | **czerwona, kolory sie zgadzaja** |
| task25 Funko | dwa miecze zamiast rozdzki | **jedna rozdzka, jak w referencji** |
| task20 Meowth | wyblakly, kremowy | **zolty, poprawne znaczenia** |

Sześć komorek lepszych, trzy gorsze. **Trzech gorszych nie wolno podmienic wybiorczo na stare**,
bo podpis mowi *"the final model's generation after all fifty were learned in sequence"* --
wiersz musi pochodzic z JEDNEGO modelu. A oslabienie stylu i utrata atrybutu to dokladnie to,
co mowia liczby v2 (DINO 0,524 wobec 0,617 po dziesieciu zadaniach). Pokazanie ladniejszych
kadrow z v1 obok liczb v2 byloby wprowadzaniem w blad.

## Wykonane

Dziesiec klatek wybranych po DINO (bieg 3186852, cztery ziarna na komorke, skala 0,80),
przyciete do kwadratu i przeskalowane do 256x256, zapisane jako `p022v2_taskNN_<koncept>.jpg`
-- **nowa nazwa celowo**, zeby proweniencja byla widoczna w pliku, a stare klatki zostaly.
`teaser_body.tex`: dziesiec odwolan `p022_task` -> `p022v2_task`. Praca: 22 strony, zero bledow.

## Blad w odzyskanym skrypcie budujacym

`build_teaser.sh` **nie dzialal** na tej maszynie: ustawial `TEXINPUTS="$B/../..;"`, gdzie `$B`
jest sciezka uniksowa Git Basha (`/d/projekty/...`), ktorej MiKTeX nie znajduje -- budowa padala
na `File 'iclr2027_conference.sty' not found`. Naprawione przez `cygpath -w`, z zapasowa galezia
dla systemow bez cygpath. Wniosek na przyszlosc: wczorajsza "weryfikacja przebudowy piksel
w piksel" nie mogla przebiegac ta sciezka, wiec **odzyskane zrodlo bylo zepsute az do dzisiaj**
i nie wiedzialbym o tym, gdyby nie ta podmiana.

Kopia `figures/` sprzed zmian: `figures_backup_20260920_0924.tar.gz`.

## `feather 0` NIE usuwa duplikatow -- hipoteza obalona (bieg 3186815)

Przewidywanie zapisane przed biegiem: przy `--feather 0` dodatkowy pies w 3.5 i fragmenty misia
w 12.5 powinny zniknac albo wyraznie oslabnac. Sprawdzone na sparowanych ziarnach, wycinki
w pelnej rozdzielczosci:

* **3.5, szczelina miedzy pudelkiem kota (0,295) a psa (0,328)** -- przy feather 0 nieco
  czysciej (ciemniejszy pasek zamiast futra), ale amorficzna ruda masa **nadal tam jest**;
* **12.5, na lewo od pudelka plecaka** -- przy feather 0 duplikat jest **WYRAZNIEJSZY**:
  uformowana rozowo-biala glowa misia z oczami i nosem, podczas gdy przy feather 2 byl tylko
  rozmyty fragment z lapa.

**Rachunek, ktory mnie zmylil, byl poprawny, ale nieistotny.** Szczelina rzeczywiscie ma okolo
4 pikseli latentu, a feather sigma 2 siega okolo 6 -- wiec feather ja przykrywa. Tyle ze
duplikaty nie powstaja w tej szczelinie: sa duze, uformowane i siedza gleboko w tle, gdzie
`eps = eps_global` **zawsze**, niezaleznie od feathera. Pomylilem korelacje polozenia
(przy krawedzi) z przyczyna.

To zamyka szosta os i po raz kolejny wskazuje to samo zrodlo: **galaz globalna reagujaca na
czesciowo uformowane podmioty w latencie**. Ta sama przyczyna daje trzy objawy: uciety podmiot
tam, gdzie nie ma na czym kontynuowac; kontynuacje ciala tam, gdzie jest (posciel w 12.4);
i bezglowe kopie tam, gdzie tlo na to pozwala.

Feather zostaje na 2.0, bo bez niego nic sie nie poprawia, a poswiaty przy krawedziach byly
przy nim slabsze.

## `--global_boot`: atak na przyczyne, nie na objaw (commit 51b3565)

Sesc zamknietych osi wskazywalo to samo zrodlo, wiec zamiast siodmego parametru brzegowego --
zmiana celujaca w mechanizm. Trzy dzwignie rozwazone:

| dzwignia | werdykt |
|---|---|
| prompt negatywny ("duplicate, extra animal") | leczenie objawu i zmiana ICH protokolu |
| `alpha` | **pozorna**: poza pudelkami `alpha*eps_g + (1-alpha)*eps_g = eps_g`, wiec wypada z rownania i na duplikaty nie wplywa w ogole |
| maskowanie latentu dla galezi globalnej | celuje w przyczyne, koszt obliczeniowy zerowy |

Wybrane trzecie. Jest to **dokladne odbicie bootstrapu**, ktory juz mamy: dzis galaz regionu
dostaje `inp_r = inp*m + szum*(1-m)` (widzi swoje pudelko, nie widzi reszty); teraz galaz
globalna dostaje przez pierwsze `global_boot` krokow `inp_g = inp*(1-m) + szum*m` -- widzi tlo,
nie widzi podmiotow. Uklad tla rozstrzyga sie bez nich, wiec **nie ma czego duplikowac**,
a po tych krokach wglad wraca i cienie moga sie jeszcze uformowac.

Szum to ten sam "nierozstrzygniety" `x_t` dla `x_0 = 0`, ktorego uzywa bootstrap regionow --
nie staly obraz, bo to juz raz zemscilo sie szara plyta w ksztalcie sumy pudelek (komentarz
w `sampling.py` przy `hard_masks`).

**Wyciek zaadresowany:** przy zamaskowanym latencie predykcja globalna WEWNATRZ pudelek jest
bez sensu, a wchodzilaby tam przez czlon `alpha*eps_global`, czyli 10%. Na czas maskowanych
krokow `alpha` jest zerowana.

**Domyslnie wylaczone** (`global_boot = 0`), sciezka bitowo identyczna -- losowanie szumu siedzi
wewnatrz `if`, wiec nie rusza strumienia RNG i wszystkie dotychczasowe biegi pozostaja
odtwarzalne.

Biegi: **3186855** (2 kroki) i **3186856** (5 krokow), n = 4, skala 0,55, te same ziarna co
`xl_n4_s055`, wiec porownanie bedzie parowane.

**Przewidywanie zapisane przed wynikami.** Duplikaty w 3.5, 12.5 i 3.1 powinny oslabnac albo
zniknac. Ryzyko, ktore uwazam za realne: **wroci ucinanie podmiotu przy krawedzi**, bo
kontynuacja tulowia na posciel w 12.4 pochodzi Z TEGO SAMEGO mechanizmu -- to jest zamiana,
a nie darmowa naprawa. Jesli przy 5 krokach duplikaty znikna, ale wroci ucinanie, wlasciwa
wartoscia bedzie 2 albo cos posredniego.

## Teaser, nizsze skale adaptera: jedna komorka do poprawy, dwie nie (bieg 3186853)

240 kadrow: 10 konceptow x skale 0,30 / 0,45 / 0,60 x 8 ziaren, plus istniejace 4 ziarna przy
0,80. Ogladniete jako `outputs/TEASER_SKALE_task{05,10,45}.png` -- wiersze to skale, kolumny
ziarna, referencja u gory.

| komorka | werdykt |
|---|---|
| **task05 styl** | **POPRAWA przy skali 0,60**: ziarna 04, 05, 06 to plaskie, stylizowane pejzaze o zredukowanej palecie, duzo blizsze referencji niz abstrakcyjna plama przy 0,80. Przy 0,30 i 0,45 styl w ogole nie wchodzi -- model maluje doslownie scene z malarzem przy sztalugach |
| task10 figurka | **zadna skala nie pomaga**: blond wlosy przy wszystkich 28 kadrach, rudych nie ma nigdy. Efekt plomienia pojawia sie TYLKO przy 0,80 (ziarno 02), wiec obecny wybor zostaje najlepszy |
| task45 osoba | **zadna skala nie pomaga**: okularow nie ma w zadnym z 28 kadrow. Przy 0,30 tozsamosc sie rozpada -- ziarno 02 to kobieta, ziarno 04 to zdjecie legitymacyjne |

### Dlaczego tak, i co to mowi o mechanizmie

Podzial jest dokladnie taki sam jak przy rekwizytach w kompozycji. **Okulary i kolor wlosow sa
czescia KONCEPTU**, wiec obnizanie skali adaptera ich nie przywraca -- ono tylko oslabia to,
czego wlasnie brakuje. **Styl to sam koncept**, ale konkuruje z trescia promptu: przy 0,80
adapter dominuje i produkuje abstrakcje, przy 0,60 zostawia miejsce na scene i wychodzi
stylizowany pejzaz.

Utrata okularow i koloru wlosow to **realna degradacja po pieciudziesieciu zadaniach**, a nie
zly dobor kadru. To jest zgodne z liczbami v2 (DINO 0,524 wobec 0,617 po dziesieciu) i nalezy
do ograniczen, nie do rzeczy do naprawienia przed terminem.

### Decyzja do podjecia przez WG

Poprawa w task05 wymaga **innej skali w jednej komorce niz w pozostalych dziewieciu**. To jest
do obrony wylacznie z ujawnieniem w podpisie ("adapter scale chosen per concept"), inaczej jest
cherry-pickingiem. Alternatywa: zostawic 0,80 wszedzie i przyjac slabszy kadr stylu.

## Awaria i naprawa: macierze fine-tuningu szly na zla sciezke checkpointow

Biegi 3186842 i 3186843 padly po ~50 s. Log generacji: `[gen] MISSING
outputs/blvar/finetune_cifc_s2025/bank_after_task09.pt - skipping task 9`, potem dziesiec
pominietych zadan, pusty katalog wyjsciowy i `FileNotFoundError` na zapisie `cifc_metrics.json`.

**Przyczyna: moj blad w argumencie.** `src.train_baselines` zapisuje banki do
`<output_dir>/ckpts/`, a ja podalem `--ckpt_dir outputs/blvar/finetune_cifc_s2025` zamiast
`.../ckpts`. `gen_cifc` szuka `bank_after_taskNN.pt` wprost w podanym katalogu, wiec nie znalazl
zadnego i po cichu pominal wszystkie zadania -- generacja "zakonczyla sie sukcesem" i dopiero
metryki sie wywrocily. Poprawione, zgloszone ponownie jako **3186862** i **3186863**, oba
generuja.

### Rzecz, ktora wyszla przy okazji i jest warta zapamietania

Katalog **oryginalnego** biegu fine-tuningu (ziarno 2024, `outputs/blvar/finetune_cifc/`)
**JUZ NIE ISTNIEJE** -- wyczyszczony ze `$SCRATCH` po trzydziestu dniach. Przetrwala tylko jego
macierz `outputs/matrix/finetune/s08`, czyli wynik 0,1839 uzyty w Figurze 3. To znaczy, ze:

* liczby z tamtego biegu sa **nieodtwarzalne** -- checkpointow nie ma i nie da sie ich przeliczyc
  przy innej skali ani sprawdzic;
* gdyby macierz tez zostala wyczyszczona, slupek fine-tuningu w Figurze 3 zniknalby bez sladu.

To samo dotyczy kazdego starszego biegu, z ktorego zostawilismy tylko wynik. **Warto zrobic
przeglad, ktore liczby w pracy maja jeszcze checkpointy, a ktore juz tylko zapisany wynik** --
bo te drugie nie sa juz weryfikowalne.

## `--global_boot`: WYNIK NEGATYWNY, zmiana odrzucona (biegi 3186855, 3186856)

Przewidywanie bylo polowicznie trafne, ale skutek uboczny okazal sie znacznie gorszy, niz
zakladalem. Sparowane ziarna, ramki naniesione:

| scena | bez (gb 0) | gb 2 | gb 5 |
|---|---|---|---|
| 3.5 | 3 podmioty + dodatkowy pies w szczelinie | **trzecie pudelko PUSTE**, plaski szary prostokat | j.w. |
| 12.5 | plecak, mis, kaczka + fragmenty misia z lewej | **plecak ZNIKNAL**, w ramce szary kamyk | **plecak ZNIKNAL**, plaska zolto-zielona plama; mis zmienil kolor na niebieskoszary |
| 12.4 | 4 podmioty, corgi uciety | corgi to glowa w bialym owalu | corgi w owalu, **szary kot ZNIKNAL** -- 3 z 4 |

**Duplikaty rzeczywiscie znikaja, ale przez usuniecie podmiotow.** To nie jest naprawa.

### Dlaczego -- i dlaczego powinienem byl to przewidziec

Komentarz w `sampling.py` przy `hard_masks` opisuje dokladnie ten tryb awarii dla bootstrapu
regionow: maskowanie zostawia plaska plyte w ksztalcie sumy pudelek, bo **UNet ma globalne pole
recepcyjne**. Zaslaniajac galezi globalnej wnetrza ramek psuje sie jej predykcje WSZEDZIE, nie
tylko w zaslonietym miejscu. Region traci wtedy kontekst, o ktory mogl sie zaczepic, i pudelko
nigdy nie "decyduje sie" na podmiot -- a po zakonczeniu maskowanych krokow galaz globalna widzi
juz plaski obszar i maluje plaskie.

Ten sam komentarz, ktory zacytowalem w uzasadnieniu zmiany jako argument ZA uzyciem
"nierozstrzygnietego" x_t zamiast stalego obrazu, byl jednoczesnie argumentem PRZECIW calemu
pomyslowi. Przeczytalem go jako przepis na szum, a nie jako ostrzezenie o mechanizmie.

### Stan

`global_boot` zostaje w kodzie z **domyslna wartoscia 0**, czyli wylaczony i bitowo neutralny.
Nie usuwam go, bo wynik negatywny jest udokumentowany i ktos moze chciec go powtorzyc, ale
**nie nalezy go wlaczac**.

### Co to znaczy dla duplikatow

Siodma zamknieta os. Duplikaty poza pudelkami sa cena za to, ze galaz globalna widzi cala scene
-- a ona musi ja widziec, zeby podmioty w ogole sie uformowaly i zeby tlo do nich pasowalo.
**Nie da sie tego rozdzielic w tym schemacie scalania.** Pozostaje albo prompt negatywny
(objaw, zmiana protokolu), albo uznanie tego za ograniczenie metody.

## Wyscig miedzy rownoleglymi macierzami CIDM (bieg 3186848, naprawione w 0efb636)

3186848 (macierz CIDM ziarno 1) padl po 12 minutach na:
`FileNotFoundError: .../data/CIFC/_cfg_k0.json` w `os.remove(cfg_path)`.

**Przyczyna: `_cidm_gen.py` nazywal plik tymczasowy `_cfg_k{k}.json`, czyli zaleznie TYLKO od
`k`.** Trzy macierze CIDM (ziarna 0, 1, 2) chodza rownolegle na tym samym katalogu
`data/CIFC`, wiec jedna kasowala config, ktorego druga jeszcze uzywala. Klasyczny wyscig,
niewidoczny dopoki biegi byly pojedyncze -- a stal sie mozliwy dopiero dzis, gdy WG poprosil
o dodatkowe ziarna i trzy biegi ruszyly naraz.

Poprawka: PID w nazwie. **Zagrozone byly takze dwa biegi juz trwajace**, w tym macierz ziarna 0
z ponad trzema godzinami pracy -- one maja stary kod wczytany do procesu, wiec poprawka ich nie
obejmuje i moga jeszcze na siebie trafic. Praca nie przepada (gotowe komorki maja znacznik
`prompts.json` i sa pomijane przy wznowieniu), ale trzeba je bedzie zgłosic ponownie.

### Drugi problem, ktory przy okazji wyszedl

`scripts/_cidm_gen.py` **nie byl w gicie**, mimo ze `.push-local` obejmuje wylacznie
`scripts/sbatch_*.sh`, a wszystkie pozostale `scripts/_*.py` sa sledzone. Na klastrze istniala
kopia nieśledzona, ktora blokowala `git pull --ff-only` po dodaniu pliku do repo
(`error: untracked working tree files would be overwritten by merge`). Przed usunieciem
porownalem obie wersje: **roznily sie wylacznie moja poprawka**, wiec nic nie przepadlo.

Wniosek: warto sprawdzic, czy na klastrze nie ma innych nieśledzonych kopii skryptow, ktore
rozjechaly sie z repo -- bo taka kopia cicho wygrywa z gitem az do pierwszego konfliktu.

## Naprawa: 14 plikow tylko na klastrze, i co to znaczylo dla odtwarzalnosci

Awaria 3186848 odslonila wiekszy problem. `git status` na klastrze pokazal **czternascie
nieśledzonych plikow**, ktore jednoczesnie tam byly i byly uzywane:

* **siedem configow**: `configs/phaseT/T10_v2.yaml`, `T50_v2.yaml`, `configs/phaseX/X_sdxl_b3000.yaml`
  i cztery jego warianty (`kd128wd0`, `lr3e4`, `noout`, `s1600`);
* **siedem skryptow**: `_cidm_attn1_shim.py`, `_compose_score.py`, `_dw_drift.py`,
  `_dw_drift_refs.py`, `_fig_rerender.py`, `_fig_score.py`, `_pick_frames.py`.

`.push-local` obejmuje wylacznie `scripts/sbatch_*.sh`, wiec te pliki nie mialy zadnej
legalnej drogi na klaster -- **configi nie istnialy lokalnie w ogole**, czyli powstaly
bezposrednio tam, wbrew regule "code moves only through git".

**Konsekwencja, ktora liczy sie najbardziej: kazdy dzisiejszy bieg zapisywal w `run-info.txt`
commit, ktory NIE ZAWIERAL uzywanych plikow.** Dotyczy to calej kompozycji
(`X_sdxl_b3000.yaml`), teasera i CIDM-50 (`T50_v2.yaml`) -- wiec wynikow nie dalo sie odtworzyc
z commita, mimo ze launcher go zapisywal.

### Naprawione (commit d501c9d)

Skrypty porownane bajt po bajcie z kopiami klastrowymi -- **identyczne**, wiec commit nie zmienia
tego, co sie liczylo. Configi sciagniete z klastra (LF sprawdzone), wszystkie czternascie dodane
do repo, wypchniete, kopie nieśledzone z klastra usuniete, `pull` odtworzyl wersje z gita.
`git status` na klastrze pokazuje teraz tylko `data` i `wandb`, czyli dowiazania.

**Potwierdzenie, ze dziala: nowe biegi zapisuja `commit: d501c9d` BEZ dopiska `(DIRTY)`** --
pierwszy raz dzisiaj.

### Restart dwoch zagrozonych biegow

3186836 (macierz ziarno 0, 3 h 44 min) i 3186849 (ziarno 2, 51 min) mialy stary kod wczytany do
procesu, wiec poprawka wyscigu ich nie obejmowala. Zatrzymane i zgloszone ponownie jako
**3187134** i **3187135**; praca nie przepadla, bo gotowe komorki maja znacznik `prompts.json`
-- 3187134 pominal 16 komorek od razu po starcie. Trzecia macierz (**3187117**, ziarno 1) biegnie
od 23 minut.

Nazwy configow maja teraz PID (`_cfg_k1_p1746329.json`), wiec kolizja nie moze sie powtorzyc.
Lancuch CIDM-50 (gen3) wciaz uzywa starej nazwy `_cfg_k49.json`, ale jest teraz jej JEDYNYM
uzytkownikiem, wiec jest bezpieczny. W `data/CIFC` zostaly trzy osierocone pliki
(`_cfg_k2.json`, `_cfg_k5.json`, `_cfg_k49.json`) po anulowanych biegach -- nieszkodliwe,
zostawiam bez pytania o zgode na kasowanie.

## Dodatkowe ziarna fine-tuningu: pierwsze liczby i problem z porownywalnoscia

Macierze 55-komorkowe przy skali 0,8 (biegi 3186862, 3186863) skonczyly sie po ~3 h:

| ziarno | TA (CLIP-T) | IA (CLIP-I) | DINO koncowe | zapominanie DINO @ s = 0,8 |
|---|---|---|---|---|
| 2025 | 0,7261 | 0,7546 | 0,4438 | **0,2542** |
| 2026 | 0,7239 | 0,7411 | 0,4646 | **0,2436** |

**Tych liczb NIE wolno wstawic do Figury 3 jako slupkow bledu.** Slupek fine-tuningu w figurze to
**0,1839**, odczytane przy ZROWNANYM TA 75,59, a nie przy jakiejs skali -- komentarz
w `make_forgetting.py` mowi wprost, ze surowe zapominanie idzie 0,17 / 0,25 / 0,37 przy skalach
0,6 / 0,8 / 1,0, wiec sam wybor skali decydowalby o wyniku. Moje 0,2542 i 0,2436 to wartosci
surowe przy 0,8 i sa z nim zgodne (~0,25), ale nieporownywalne.

Oba nowe ziarna maja przy 0,8 TA **72,6 i 72,4**, czyli ponizej 75,59 -- potrzebna jest druga,
nizsza skala do interpolacji. Zgloszone: **3187164** i **3187165**, skala 0,6, ~3 h.

### Trzeci raz ten sam wzorzec: zostaje wynik, znika droga do niego

**`scripts/_forget_matched.py` nie istnieje ani lokalnie, ani na klastrze.** Skrypt, ktory
policzyl 0,1839, zaginal. Do tego macierze ziarna 2024 przy skalach 0,6 i 1,0 zostaly
wyczyszczone ze `$SCRATCH` (zostal tylko `outputs/matrix/finetune/s08`), a katalog checkpointow
tego biegu tez juz nie istnieje. **Per-skalowe wartosci dla ziarna 2024 przetrwaly wylacznie
w komentarzu w `make_forgetting.py`.**

To trzeci przypadek tego samego dzisiaj:
1. teaser -- zrodlo w katalogu tymczasowym, odzyskane w ostatniej chwili i, jak sie okazalo,
   niesprawne;
2. fine-tuning ziarno 2024 -- checkpointy i dwie z trzech macierzy wyczyszczone;
3. `_forget_matched.py` -- skrypt liczacy wartosc w figurze, nie istnieje.

Konsekwencja praktyczna: **procedure trzeba odtworzyc**, bo bez niej trzy ziarna nie beda
policzone tak samo. Sama interpolacja jest prosta (dwa punkty (TA, zapominanie), odczyt przy
TA = 75,59), ale to, ze trzeba ja pisac od nowa, jest objawem, nie szczegolem.

## Slupek fine-tuningu: dwa ziarna policzone, i powod, dla ktorego trzeba trzeciego

Macierze przy skali 0,6 (biegi 3187164, 3187165) domknely interpolacje. Odczyt przy zrownanym
TA 75,59 (`scripts/_forget_matched.py`, odtworzony w ce66d65):

| ziarno | s = 0,8 | s = 0,6 | **przy TA 75,59** |
|---|---|---|---|
| 2025 | TA 72,61 / 0,2542 | TA 75,98 / 0,1743 | **0,1834** |
| 2026 | TA 72,39 / 0,2436 | TA 76,08 / 0,1676 | **0,1777** |

**Srednia 0,1805, sd 0,0040** na dwoch ziarnach. Liczba stojaca dzis w Figurze 3 (0,1839) miesci
sie w tym rozrzucie.

### Czego NIE wolno zrobic: dokleic tych ziaren do istniejacego slupka

Katalog, z ktorego pochodzi 0,1839 (`outputs/matrix/finetune`), ma **przy skali 0,8 TA 76,37**,
podczas gdy oba nowe ziarna maja tam 72,61 i 72,39. **Roznica 3,8 punktu przy tej samej skali to
za duzo na szum ziarna.**

Co wiecej, to sie nie domyka arytmetycznie: skoro tamten bieg ma przy 0,8 juz TA 76,37, czyli
POWYZEJ celu 75,59, to zrownanie wymaga skali WYZSZEJ, a ta daje zapominanie WYZSZE niz 0,2363.
Nie moze wiec dac 0,1839. **Ta liczba nie pochodzi z tego katalogu** -- pochodzi z macierzy,
ktorej juz nie ma.

Do tego: katalog jest z **11-12 sierpnia**, ma tylko jedna skale, checkpointy tamtego biegu sa
wyczyszczone, a w `configs/blvar/` sa CZTERY warianty fine-tuningu (`_cifc`, `_ours`, `_aug`,
`_aug_allattn`) i nie wiadomo, ktorym szedl. **Nie da sie wykazac, ze nowe ziarna to to samo
ramie**, wiec doklejenie ich jako slupkow bledu byloby nieuczciwe.

### Naprawa

Trzecie ziarno odtwarzane **tym samym, udokumentowanym configiem**: `finetune_cifc.yaml` ma
domyslnie `seed: 2024`, wiec wystarczylo go puscic bez zmian. Bieg **3187360** (trening,
~50 min), macierze przy 0,8 i 0,6 doczepione jako **3187361**. Za okolo cztery godziny beda
**trzy ziarna jednego, w pelni opisanego ramienia** -- i wtedy slupek w figurze zastapi sie
srednia z sd, a osierocone 0,1839 wypadnie z pracy razem z liczba, ktorej nie umiemy odtworzyc.

## Teaser: wybory WG i co z nich wynika (20.09, popoludnie)

### Kadry wybrane

| koncept | skala | ziarno | status |
|---|---|---|---|
| task00 pies | **0,45** | z00 | wybrane |
| task25 Funko | **0,60** | z03 | wybrane |
| task30 gitara | **0,60** | z06 | wybrane |
| task40 sukienka | **0,80** | z02 | wybrane |
| task45 osoba | **0,80** | z02 | wybrane, plus wariant z promptem o okularach do obejrzenia |

**Konsekwencja, ktora trzeba bedzie ujawnic:** wybory obejmuja TRZY rozne skale (0,45 / 0,60 /
0,80). Podpis mowi dzis tylko "the final model's generation after all fifty were learned in
sequence" -- jesli rzad ma miec rozne skale per koncept, **musi to paść w podpisie**, inaczej
jest to cherry-picking. To ta sama decyzja, ktora wczesniej dotyczyla samego task05, tylko teraz
dotyczy polowy paska.

### task05: prompt jest zly dla tego konceptu

Prompt z configu to **`"a photo of painting"`**, slowo klasy `painting`. Koncept jest
**STYLEM** (ink painting), a prompt traktuje go jak obiekt -- prosimy o zdjecie obrazu, a nie
o obraz W TYM STYLU. Stad to, co widac: przy niskiej skali doslowna scena z malarzem przy
sztalugach, przy wysokiej abstrakcja. Referencja to plaska ilustracja krajobrazu.

To nie jest wiec wada checkpointu ani skali, tylko **niedopasowanie promptu do rodzaju
konceptu**. Poprawny byloby cos w rodzaju `a painting of a landscape` przy tym samym adapterze.
Do decyzji WG, bo zmiana promptu dla jednego konceptu z dziesieciu to kolejna rzecz do
ujawnienia.

### Checkpoint zweryfikowany

WG pytal, czy rendery ida z wlasciwego checkpointu. **Tak: `outputs/sweep/p022_v2/ckpts`,
checkpoint 49.** `T50_v2.yaml` ma `eval_prefix` i `attr_strip`, czyli maszynerie poprawionych
podpisow, a starego `p022` nie ma juz na scratchu. Brak ognia w task10 przy niskich skalach to
nie zly checkpoint, tylko to, ze przy slabym adapterze pole oddaje model bazowy.

### Dogenerowane (biegi 3187501-3187504, ck49, ziarno startowe 11000)

* **3187501** -- zadania 10, 15, 20, 35 przy skalach 0,6 i 0,8, po 12 ziaren;
* **3187502** -- zadanie 15 przy 0,3 i 0,45 (domyka "wszystkie skale");
* **3187503** -- zadanie 20 przy **1,0 i 1,2** (WG: dopiero od 0,8 zaczyna przypominac Meowtha);
* **3187504** -- zadanie 45 przy 0,8 z promptem **`a photo of person wearing glasses`**
  (flaga `--prompt` dodana w 025536d).

## Dwie nowe proby po pytaniu WG "czy da sie to jakkolwiek podciagnac"

Uczciwa ocena stanu na 20.09 wieczorem: **zaden kadr nie spelnia wszystkich czterech kryteriow.**
Najlepszy obejrzany w pelnej rozdzielczosci to **3.4 / ziarno 2** -- trzy podmioty, tozsamosc
wszystkich trzech poprawna, brak duplikatow, nic nieuciete ramka, spojne tlo palacu. Ale bez
garnituru i tronu z promptu, pies wielkosci podstawy kopuly, i zadnego wspolnego swiatla ani
cieni: widac, ze podmioty sa zlozone, a nie sfotografowane razem. Drugi w kolejnosci, 12.6 / z1,
ma poprawne oba podmioty i spojny zamek, ale bez kapelusza i z psem urwanym w rogu kadru.

**Dwa z czterech kryteriow w najlepszych kadrach.** Obecnosc podmiotow i tozsamosc rozwiazane;
atrybuty i "sfotografowane razem" nie.

### 1. `regional_steps` -- os, ktora przeoczylem

Parametr istnieje od dawna, ale jego docstring opisuje go jako **optymalizacje kosztu**
("kompozycja rozstrzyga sie wczesnie, pozne kroki tylko dopracowuja teksture"), wiec nigdy nie
potraktowalem go jako dzwigni jakosci. A to wlasnie z tego opisu wynika, ze moze pomoc: jesli
galezie regionalne wylacza sie po 35-40 z 50 krokow, **ostatnie kroki przechodza jednym wspolnym
przebiegiem na calym kadrze** -- czyli dokladnie na etapie, na ktorym ujednolica sie swiatlo,
cienie i mikrotekstura.

To NIE jest kolejna odmiana strojenia sily groundingu. Wszystkie siedem zamknietych osi dzialalo
WEWNATRZ pudelek podczas formowania podmiotow. Ta dziala PO uformowaniu i na calym kadrze naraz.

**Ryzyko zapisane przed wynikiem:** w tych ostatnich krokach galaz globalna nie wie nic
o konceptach, wiec tozsamosc moze odplynac. Stad ciecie pozne (40 i 35 z 50), a nie wczesniejsze,
i pomiar DINO przed i po. Biegi **3187562** (rs 40) i **3187563** (rs 35).

### 2. cfg 10 przy skali 0,55

cfg 10 testowalem przy skali 0,4 -- odzyskiwal atrybuty NOSZONE (kapelusz w 3.1 z 1/4 na 2-3/4,
w 12.6 z 2/4 na 3-4/4) bez kosztu w tozsamosci. **Nigdy nie sprawdzilem go przy 0,55**, czyli
przy skali przyjetej, ktora sama tlumi atrybuty mocniej. Bieg **3187564**.

Przewidywanie: kapelusz i medal moga wrocic, rekwizyty (rower, kufel, tron) nie -- one wymagaja
skali 0,25, ktora psuje tozsamosc (19% regionow mylacych koncept wobec 2% przy 0,55).

### Czego celowo NIE probuje

Powiekszania pudelek (przyczyna lezy w sile adaptera, sprawdzone), promptu negatywnego
z "duplicate" (objaw + zmiana ich protokolu), powrotu do zamknietych osi.

# AUDYT PRACY 20.09: kazde twierdzenie liczbowe w tresci glownej

Przeszedlem 28 linii z liczbami w tresci glownej (bez dodatku), sprawdzajac kazda wobec zrodla.

## Zweryfikowane jako poprawne

* **21,9M / 4,26M / 21,3M** parametrow -- zgodne ze stalymi w `make_scaling.py`
  (`OURS_M = 21.9`, `PER_CONCEPT_M = 0.426`);
* **zapominanie 0,0060 +- 0,0031 na trzech ziarnach (0,0094 / 0,0051 / 0,0035)** -- zgodne
  z `make_forgetting.py`;
* **ramie beta = 0**: 0,0213 i spadek tozsamosci 0,621 -> 0,589;
* **tabela umiejscowienia** spojna ze wstepem: 100% kwadrantu i DINO maski 0,599 -> 0,717;
* **+1,9 punktu IA** przy ich opublikowanym TA oraz **+1,5 / +0,030** wobec naszego wlasnego
  pomiaru CIDM -- dwa rozne punkty odniesienia, oba obecne i niesprzeczne.

## Bledy znalezione i naprawione

**1. Wstep byl niespojny z par. 5.4 co do tej samej wielkosci.** Wstep: DINO **0,557** wobec
**0,621**; par. 5.4: **0,561** wobec **0,617** przy `s_lora = 0.45`. Ta sama rzecz -- dziesiec
konceptow benchmarku po pieciudziesieciu zadaniach -- podana dwoma zestawami liczb. Wyrownane
do par. 5.4, bo tam stoi adnotacja o skali; dopisana takze skala we wstepie.

**2. `	odo` o SDXL mowilo ,,SDXL on the p022 recipe is training''** -- ten bieg **skonczyl sie
16.09**. Co wazniejsze, jego liczby zmieniaja sens decyzji: `X_sdxl_p022` przy skali 0,40 daje
**TA 72,1 / IA 76,4 / DINO 0,536**, czyli **osiem punktow TA ponizej** ich opublikowanego wiersza
(80,0). Podmiana wiersza na spojna z par. 4 recepture oznacza wiec **utrate twierdzenia
o parytecie**, a nie kosmetyke, jak sugerowala notatka. Wybor przepisany wprost.

**3. `	odo` o kompozycji opisywalo ,,four scenes, one checkpoint''** -- nieprawda od dzisiaj.
Przepisane na stan faktyczny: 11 scen x 4 ziarna, siedem zamknietych osi, obecnosc i tozsamosc
rozwiazane, atrybuty i spojnosc oswietlenia nie.

## Nieaktualne, ale nie do naprawy dzisiaj (dane w drodze)

* **,,sequential fine-tuning forgets 0,184, thirty-one times more''** -- 0,184 pochodzi
  z osieroconej macierzy sierpniowej. Nowy pomiar udokumentowanego ramienia: **0,1805 +- 0,0040**
  na dwoch ziarnach (0,1834 i 0,1777), trzecie liczy sie teraz. Po nim zmieni sie takze mnoznik
  (0,1805 / 0,0060 = **30**, nie 31) oraz ,,nine times less'' przy ramieniu beta = 0
  (0,1805 / 0,0213 = 8,5).
* **forgetting CIDM** -- trzy macierze 55-komorkowe licza sie na Atenie.

## Decyzja WG odnotowana

Pasek teasera uzywa trzech skal adaptera dobranych per koncept (0,45 / 0,60 / 0,80, mapowanie
w `figures/teaser_src/small/PICKS.txt`). WG: dobor probek to standard w pracach generatywnych,
`	odo` o ujawnieniu usuniete. Zapisuje wlasne zastrzezenie raz i do niego nie wracam: dobor
ZIARNA jest standardem, dobor SKALI ADAPTERA per koncept jest inna osia, bo skala to
hiperparametr metody i po nim biegnie krzywa w Figurze 3.

## Stan formalny po audycie

22 strony, zero bledow LaTeXa, **13 `	odo`**, `check_draft` bez zmian (jeden problem:
`fig:method` bez etykiety, sprzed dzisiaj). Kopie zapasowe: `main.tex.bak-gs-*`,
`main.tex.bak-audyt-*`.

# ZAPOMINANIE CIDM POLICZONE (bieg 3187134, 6 h 48 min, 55 z 55 komorek)

To byla dziura, przez ktora Figura 3 pokazywala tylko fine-tuning, a WG wskazal ja jako
blokujaca porownanie.

```
Average (final): CLIP-T 0.7602 | CLIP-I 0.7713 | DINO 0.5767
Forgetting:      CLIP-I +0.0118 | DINO +0.0228
```

| metoda | zapominanie DINO | TA odczytu | ziarna |
|---|---|---|---|
| **nasza** | **0,0060 +- 0,0031** | 75,59 | 3 |
| **CIDM** | **0,0228** | 76,02 | 1 (dwa dalsze licza sie) |
| fine-tuning | 0,1805 +- 0,0040 | 75,59 | 2 (trzecie liczy sie) |

**Zapominamy 3,8x mniej niz CIDM.**

### Dlaczego to porownanie jest uczciwe bez interpolacji

Nasz odczyt stoi przy TA 75,59, CIDM wypada przy **76,02** -- roznica **0,4 punktu**. Przy
fine-tuningu interpolacja byla konieczna, bo jego TA przy skali 0,8 to 72,6, czyli trzy punkty
nizej. Tutaj punkty pracy same sie spotykaja, wiec liczby sa porownywalne wprost. Warto to
napisac w podpisie figury, bo uprzedza oczywiste pytanie recenzenta.

### Czego brakuje do zamkniecia

* dwa dalsze ziarna CIDM (biegi 3187117 i 3187135, ~7,5 h, okolo polowy komorek) -- dadza slupek
  bledu takze po ich stronie;
* trzecie ziarno fine-tuningu (3187361, 2,5 h z ~3);
* potem: przeliczyc mnozniki w tekscie (0,1805/0,0060 = 30, nie 31; 0,1805/0,0213 = 8,5, nie 9)
  i przepisac `	odo` przy Figurze 3 oraz zdanie w par. 5.3 o CIDM.

### Helios: piec renderow teasera i grid zapominania gotowe

Wszystkie skonczone, najdluzszy 9 min. Bliźniaki na Atenie dokonczyly sie same, gdy zwolnily
sie sloty -- nie bylo czego anulowac. Material na obie figury jakosciowe (grid zapominania
i porownanie z CIDM przy jednym koncepcie) jest kompletny.

## Figura 3 (prawy panel) przepisana: trzy slupki, bez interpolacji

Po pytaniu WG ,,a moze w ogole bez interpolacji?'' -- okazalo sie, ze tak sie da, i jest to
lepsze niz oba warianty, ktore proponowalem.

| slupek | punkt pracy | TA | zapominanie | ziarna |
|---|---|---|---|---|
| fine-tuning | `s_lora = 0,6` | 75,98 / 76,08 | **0,1710 +- 0,0047** | 2 (trzecie w biegu) |
| CIDM | ich `alpha = 0,8` | 76,02 | **0,0228** | 1 (dwa w biegu) |
| nasze | `s_lora = 0,45` | 75,59 | **0,0060 +- 0,0031** | 3 |

Trzy punkty mieszcza sie w **pol punktu TA**, wiec interpolacja jest zbedna. Zyski: kazdy slupek
jest pomiarem; znika asymetria (fine-tuning interpolowany, CIDM nie); **znika zaleznosc od
zaginionego `_forget_matched.py`**, ktorego odczytu 0,1839 i tak nie da sie dzis zweryfikowac.

Nasz punkt lezy przy NAJNIZSZYM TA z trzech, a zapominanie rosnie ze skala adaptera, wiec odczyt
jest dla nas konserwatywny -- to tez wpisane do podpisu.

### Rzecz, ktora WG wychwycil, a ja bym przepuscil

Pytanie ,,czy finetuning ma takie samo IA/TA jak na wykresie po lewej?'' -- **nie ma**.
Opublikowany wiersz fine-tuningu: **TA 70,0 / IA 73,7**. Nasza re-implementacja przy 0,6:
**TA 76,0 / IA 74-76**. **Szesc punktow TA rozjazdu**, a w obu panelach stala ta sama etykieta
,,Finetuning'' i ten sam kolor. Dopisane do podpisu wprost.

### Korekta mojego wlasnego nadmiaru

Powiedzialem, ze mocniejszy baseline jest ,,na pewno na nasza korzysc''. WG zakwestionowal i mial
racje: **sila w TA/IA to inna os niz zapominanie**, wiec sam rozjazd nic o zapominaniu nie mowi.
Wlasciwy, wezszy argument brzmi inaczej i jest sprawdzalny: nasze ramie fine-tuningu trzyma
**wyuczony embedding tokenu na kazdy koncept, ktorego kolejne zadania nie nadpisuja** -- ma wiec
mechanizm SPRZYJAJACY zapamietywaniu i mimo to zapomina 28 razy wiecej. Tak jest teraz napisane
w par. 5.3.

Uboczny koszt, ktory trzeba miec z tylu glowy: skoro baseline ma per-konceptowe tokeny, to ma
magazyn rosnacy z T -- nie wolno wiec sugerowac, ze jest od magazynu wolny, bo rama pracy mowi
,,zastepujemy per-konceptowy magazyn jedna siecia''.

### Jak dziala nasz fine-tuning (sprawdzone w kodzie)

`StaticLoRABank` z `per_task = False`: **jedna wspolna para LoRA na warstwe**, ranga 4, trenowana
po kolei na wszystkich konceptach, bez resetu, `lam = 0`. Te same cztery projekcje co u nas
(`attn2.to_q/to_k/to_v/to_out.0`), ta sama ranga, te same 800 krokow na koncept -- pojemnosc
i budzet zrownane. Plus `learned_tokens` per koncept, ktorych nasz p022 NIE uzywa (u nas
tozsamosc niesie ortogonalizowany task embedding). To najprawdopodobniejsze wyjasnienie szesciu
punktow TA nad opublikowanym wierszem.

### Zmienione liczby w tekscie

* ,,forgets $0.184$, thirty-one times more'' -> **$0.171$, twenty-eight times more**, plus zdanie
  o CIDM (,,close to four times more'');
* ramie beta = 0: ,,nine times less'' -> **eight times less** (0,1710 / 0,0213 = 8,0).

22 strony, zero bledow, `	odo` przy figurze przepisane na stan faktyczny z numerami biegow.

# WERYFIKACJA AKTUALNOSCI WSZYSTKICH WYNIKOW W PRACY (20.09, wieczor)

95 twierdzen liczbowych: **28 w tresci glownej, 67 w dodatku**. Sprawdzone przez ustalenie,
z ktorego biegu pochodza i czy ten bieg jeszcze istnieje.

## Korekta mojego wczesniejszego alarmu

Twierdzilem dzis, ze ,,biegi wiersza glownego zostaly wyczyszczone''. **To bylo za mocne** --
sprawdzalem tylko Atene. Stan faktyczny:

| grupa wynikow | zrodlo | stan |
|---|---|---|
| tab:curve, tradeoff, forgetting (3 ziarna) | `p022`, `p022_s2025`, `p022_s2026` | **Helios, sa** |
| skalowanie T = 50 | `p022` (50 konceptow) | **Helios, jest** |
| ablacje nogs / noout | `p022_nogs`, `p022_noout` | **Helios, sa** |
| SDXL (trzy warianty) | `X_sdxl_ground_800aug`, `X_sdxl_800aug_fix`, `X_sdxl_p022` | **Helios, sa** |
| EWC / LwF / C-LoRA / L2DM / F_base | `outputs/matrix/*` | **Atena, sa** |
| ortogonalnosc (0,056 / 0,040 / 0,27) | `p022/lora_ortho.json` | **Helios, jest** |
| dryf dW | `drift_src`, `p022_v2/dw_drift50.json` | **jest** |

**Naprawde nie istnieja dwie rzeczy**: checkpointy fine-tuningu ziarna 2024 (odtwarzane, bieg
3187360/3187361) i `_forget_matched.py` (odtworzony, a po przejsciu na punkty zmierzone juz
niepotrzebny).

## Znaleziona dziura: tabela ma dosłowne ,,pending''

`tab:attr` (wplyw usuwania atrybutow z podpisow), wiersz ,,cat (fluffy)'', kolumna ziarna B:
**`pending`**. A podpis tabeli twierdzi *,,paired across two training seeds''*. Jedna komorka nie
istnieje, a tabela oglasza sparowanie. **Do naprawy przed wyslaniem**: albo doliczyc ta komorke,
albo przerobic tabele i podpis na jedno ziarno dla tego konceptu.

## Zaktualizowane dzisiaj (bylo nieaktualne)

* slupek fine-tuningu **0,1839 -> 0,1710 +- 0,0047**, i to bez interpolacji;
* **CIDM 0,0228** -- wczesniej nie istnial;
* mnozniki **31x -> 28x** i **9x -> 8x**;
* wstep: DINO **0,557/0,621 -> 0,561/0,617** (byl niespojny z par. 5.4);
* `scaling`: krzywa T = 10 z jednego ziarna **-> trzy ziarna** (byla niespojna z `tradeoff`);
* teaser: klatki z `p022` **-> `p022_v2`** (tekst juz podawal liczby v2);
* ablacja Gram-Schmidta w przestrzeni obrazow **usunieta** (decyzja WG).

## Liczby, ktore jeszcze sie przesuna (biegi w toku)

* fine-tuning: trzecie ziarno (3187361, ~3 h) zmieni srednia i sd;
* CIDM: dwa dalsze ziarna (3187117, 3187135, ~8 h) dodadza slupek bledu;
* po nich: przeliczyc 28x i 8x.

## Znane, nierozwiazane (nie regresje, tylko dlugi)

* wiersz SDXL: `ground_800aug` daje parytet, ale uzywa regularyzatora na czynnikach, czemu
  przeczy par. 4; spojna receptura `X_sdxl_p022` daje TA 72,1, czyli osiem punktow ponizej ich
  wiersza -- **decyzja WG**;
* tabela umiejscowienia i `placement_grid`: jeden checkpoint;
* `fig:method` bez etykiety (sprzed dzisiaj).

## Dwie ostatnie osie kompozycji: obie negatywne (biegi 3187562, 3187563, 3187564)

### `regional_steps` -- zadnej roznicy (osma zamknieta os)

Hipoteza: zatrzymac galezie regionalne po 35-40 z 50 krokow i oddac koncowke jednemu przebiegowi
na calym kadrze, zeby ujednolicic swiatlo i cienie -- czyli to, czego brak sprawia, ze podmioty
wygladaja na doklejone.

Na arkuszu wygladalo na poprawe: w 12.6 ciemny owal za kotem zdawal sie znikac, w 3.5 zwierzeta
byly lepiej osadzone. **W pelnej rozdzielczosci obrazy sa niemal identyczne** -- ciemna masa za
kotkiem nadal tam jest, korona na psie w obu, zamek ten sam; roznica to kontrast. Kolejny raz
pomylka z kafelka.

Wyjasnienie, ktore z tego wynika i jest uczciwsze niz moja hipoteza: docstring parametru mowil
prawde, ze **kompozycja rozstrzyga sie wczesnie**, ale ja wyciagnalem z tego zly wniosek. Pozne
kroki dopracowuja TEKSTURE -- one nie przeoswietlaja sceny. Do 35. kroku obraz jest juz
przesadzony, wiec oddanie koncowki galezi globalnej nie ma czego naprawic.

### cfg 10 przy skali 0,55 -- brak efektu, choc przy 0,4 dzialal

Kapelusz czarodzieja: **1 z 4 ziaren** w 3.1 i 1 z 4 w 12.6 -- dokladnie tyle, ile przy cfg 7,5.
Tymczasem **przy skali 0,4 cfg 10 podnosil trafialnosc z 1/4 na 2-3/4**. Wniosek: przy mocniejszym
adapterze wzmocnienie sygnalu tekstowego nic nie daje, bo to nie sygnal tekstowy jest waskim
gardlem, tylko dominacja adaptera. To domyka pytanie z rana, czy cfg 10 uratuje atrybuty przy
przyjetej skali: **nie**.

### Bilans

Dziewiec sprawdzonych osi, wszystkie zamkniete. Dwa z czterech kryteriow spelnione (obecnosc
podmiotow, tozsamosc), dwa nie (atrybuty, ,,sfotografowane razem''). Przy skali chroniacej
tozsamosc atrybuty nie wracaja ZADNYM ze sprawdzonych sposobow -- ani skala, ani cfg, ani
harmonogramem, ani poziomem sklejania.


## SDXL: co jest wyczerpane, co bylo bledem i co zostalo

### Wycofanie wlasnego argumentu

Twierdzilem, ze skala adaptera to "rezim, ktorego model nie widzial w treningu", bo `lora_scale`
nie wystepuje w `train_cl.py`. To **nie jest** niezgodnosc treningu z testem. Trening przy STALEJ
skali s jest reparametryzacja treningu przy 1,0 -- siec nauczylaby sie delty 1/s razy wiekszej
i wynik bylby ten sam co do gradientu. Zostawaloby tylko sprzezenie z regularyzatorem (kara na
dW) i z weight decay. Jedyna nietrywialna wersja to LOSOWANIE skali co krok, a jej efekt jest
niepewny w obie strony. Wniosek: skala nie jest bledem, tylko rzeczywista interpolacja miedzy
modelem bazowym a zaadaptowanym, i nie tedy droga.

### Nowy fakt: 1600 krokow SZKODZI na SDXL

`X_sdxl_b3000_s1600` byl policzony i nigdy nieodczytany. Wiersz koncowy, ten sam tor ewaluacji:

| skala | arm | clip_t (TA) | clip_i (IA) | DINO |
|---|---|---|---|---|
| 0,25 | b3000 (800 krokow) | **0,8143** | **0,7658** | **0,5336** |
| 0,25 | s1600 | 0,7315 | 0,7572 | 0,4943 |
| 0,30 | b3000 | **0,7897** | **0,7811** | **0,5535** |
| 0,30 | s1600 | 0,6997 | 0,7485 | 0,4710 |
| 0,40 | b3000 | **0,7554** | **0,8013** | **0,5931** |
| 0,40 | s1600 | 0,6767 | 0,7566 | 0,5236 |

Gorzej na **kazdej** metryce przy **kazdej** skali, i to duzo -- 8 punktow TA. Podwojenie
budzetu treningu nie jest niedotrenowaniem, tylko przetrenowaniem na pieciu obrazach koncepta.

**Konsekwencja, ktora latwo przeoczyc:** to jest argument PRZECIW dokladaniu pojemnosci.
Skoro ramie przetrenowuje sie przy 1600 krokach, to `rank` 4 -> 8 albo `head_hidden` 50 -> 150
najpewniej tez przyspieszy przetrenowanie. Nie proponuje retreningu na pojemnosc.

### Stan przemiatu SDXL

| os | zakres | wynik |
|---|---|---|
| beta (kara na dW) | 300 / 1000 / 2000 / 3000 | optimum 3000 |
| liczba krokow | 800 / 1600 | 800, z duzym zapasem |
| lr | 1e-4 / 3e-4 | 3e-4 rozbiega trening (gnorm 941 077) |
| key_dim 128 + wd 0 | -- | szkodzi przy zrownanym TA |
| bez kary na wyjsciu (`noout`) | -- | DINO 0,593 -> 0,430 |

### Co naprawde zostalo: adapter ma DWA pokretla, a krzywa uzywa jednego

`gen_cifc` przyjmuje niezaleznie `--lora_scale` (skaluje dW na `attn2`) i `--ground_gain`
(skaluje addytywny wstrzyk GSA, `gain * tanh(gate) * inside * read`). W generacji
jednokonceptowej nie ma pudelka, wiec `geo_inside` daje pelna ramke i GSA jest globalnym
skladnikiem addytywnym -- ale o **innej formie funkcyjnej** niz dW: dW zmienia odwzorowanie
warstwy, GSA dodaje odczyt stanow ukrytych.

Cala krzywa kompromisu z Figury 3 to przekatna `ground_gain = 1`. Jesli oba tory maja rozne
nachylenia TA/IA, to front Pareto lezy POZA przekatna i nasz obecny punkt pracy jest
suboptymalny bez zadnej winy treningu.

Punkt odniesienia (`X_sdxl_b3000`, gg = 1): s0,25 -> TA 81,4 / IA 76,6; s0,30 -> TA 79,0 /
IA 78,1. Interpolacja na ich TA 80,0 daje IA ~77,5 wobec ich 79,5.

Cztery biegi, **zero treningu**, nic nie nadpisuja (osobne katalogi `eval_gg*`):

| job | ground_gain | skale | katalog |
|---|---|---|---|
| 23282579 | 0 (kolejka wczesniej) | -- | -- |
| 23288087 | 0,5 | 0,30 / 0,40 / 0,50 | `eval_gg05` |
| 23288062 | 1,5 | 0,20 / 0,25 / 0,30 | `eval_gg15` |
| 23287955 | 2,5 | 0,20 / 0,25 / 0,30 | `eval_gg25` |

**Uczciwe zastrzezenie.** To jest dobor punktu pracy na metryce ewaluacyjnej -- ta sama klasa
co istniejacy przemiat skali, ale z jednym stopniem swobody wiecej. Jesli cos z tego wyjdzie,
raportujemy FRONT, nie jeden wybrany punkt. Osobna decyzja (nie moja): czy odczytac wiersz
SD-1.5 przy tym samym dwupokretlowym ustawieniu. To nie zmienia przepisu treningu, ale zmienia
opublikowane liczby.

## Wiersz SDXL rozstrzygniety: raportujemy `X_sdxl_b3000`, i to jako KRZYWA

Decyzja WG z 20.09: nie `ground_800aug` (parytet TA 80,2 / IA 79,3, ale kara na CZYNNIKACH
LoRA, czemu przeczy par.~4), nie `p022` (przeszczep bety 7120, TA 72,1 -- osiem punktow ponizej
ich wiersza), tylko **receptura glowna z beta przestrojona pod SDXL na 3000**. Parytet kupiony
mechanizmem, ktorego praca nie opisuje, jest gorszy niz spojnosc bez parytetu.

Piec zmierzonych skal, jedno ziarno, wiersz koncowy, wszystkie wygenerowane 17.09:

| s_lora | TA | IA | DINO |
|---|---|---|---|
| 0,20 | 83,17 | 75,23 | 0,5187 |
| 0,25 | 81,43 | 76,58 | 0,5336 |
| 0,30 | 78,97 | 78,11 | 0,5535 |
| 0,40 | 75,54 | 80,13 | 0,5931 |
| 0,50 | 73,67 | 81,29 | 0,6267 |

Odczyt przy KAZDYM opublikowanym wierszu SDXL, przy JEGO tekstowym wyrownaniu (liczone
w `figures/make_tradeoff_sdxl.py`, nie na kartce):

| ich wiersz | ich IA | nasze IA przy tym TA | margines |
|---|---|---|---|
| Finetuning | 71,5 | 80,28 | **+8,78** |
| LwF | 76,5 | 78,86 | **+2,36** |
| EWC | 77,6 | 79,21 | **+1,61** |
| L2DM | 77,1 | 78,27 | **+1,17** |
| C-LoRA | 77,8 | 78,80 | **+1,00** |
| CIDM | 79,5 | 77,47 | **-2,03** |

Bijemy piec wierszy na szesc; przegrywamy z ich wlasna metoda o dwa punkty. Dlatego krzywa,
nie wiersz -- przy jednej liczbie werdykt zalezy od tego, ktora skale sie wybierze, a to
wyglada na dobor pod wynik.

### Znacznik odczytu zdjety (zwrocona uwaga WG)

Dorysowalem pusty punkt przy ich TA 80,0 z podpisem 77,5. `make_tradeoff.py` dla SD-1.5 ma tam
sama kropkowana linie. Efekt byl taki, ze na figurze, gdzie PRZEGRYWAMY, liczba byla
podswietlona, a na tej, gdzie wygrywamy -- nie. Zdjete; obie figury buduja sie tak samo,
liczba stoi w tekscie.

### Hipoteza WG: czy dwa enkodery tekstu SDXL to bug?

Sprawdzone, i jedna realna pulapka w tym miejscu ISTNIEJE, ale ten bieg jej nie dotyczy.

`sampling.py` do 2026-09-07 podawal galezi bezwarunkowej sekwencje negatywna z **zerowym**
`text_embeds` -- kombinacja, ktorej SDXL nigdy nie widzial, a blad galezi uncond mnozy sie
w CFG przez (1 - 7,5) = -6,5. Poprawka jest domyslna, stara sciezka zostala pod flaga
`--uncond_legacy_zero`. **Wszystkie piec skal b3000 policzono 17.09, dziesiec dni po poprawce,
bez tej flagi** -- czyli na poprawionym protokole i, co wazniejsze dla figury, na TYM SAMYM
protokole we wszystkich pieciu punktach.

Reszta toru dwuenkoderowego wyglada poprawnie: `hidden_states[-2]` z obu enkoderow zlaczone
po kanalach do 2048, `o2.text_embeds` (1280) jako pooled -- to jest standardowe SDXL.
Maska tokenowa idzie po osi sekwencji (77), wspolnej dla obu tokenizerow, a `identifier: ''`
znaczy, ze nie dokladamy zadnego rzadkiego tokenu, wiec rozjazd slownikow nie ma jak wystapic.

**Kontrargument z danych, mocniejszy niz czytanie kodu:** gdyby tor tekstowy byl zepsuty, TA
bylaby niska WSZEDZIE. Przy s 0,20, gdzie adapter jest prawie wylaczony, mamy TA 83,17 -- to
zdrowa liczba. TA spada gladko 83,2 -> 73,7 wraz z sila adaptera, czyli ma ksztalt kompromisu,
a nie awarii.

Domykam to pomiarem, a nie argumentem: **job 23290925** liczy `s_lora = 0` przy
`ground_gain = 0`, czyli zamrozony SDXL naszym torem ewaluacji. Jesli tor jest zdrowy, TA
wyjdzie na poziomie 83-85. To tez uzyteczna kotwica krzywej: punkt startowy backbone'u.

### Praca

`main.tex`: dodatek `app:sdxl` napisany od nowa, z figura `tradeoff_sdxl.pdf`; beta rozdzielona
per backbone w opisie recepturty (7120 SD-1.5 / 3000 SDXL); wskaznik w par.~4 mowi teraz
o krzywej i o tym, ze przegrywamy z ich metoda; `\todo` DECYZJA usuniete; pozycja "SDXL at the
headline recipe" skreslona z planu pracy. Bez bledow, 23 strony.

### ZNALEZIONE PRZY OKAZJI: tresc glowna jest o STRONE ZA DLUGA

ICLR tnie twardo na dziewieciu stronach. Tresc konczy sie **na koncu strony dziesiatej**, czyli
okolo 56 wierszy za duzo. **To nie jest skutek moich zmian** -- sprawdzone przez kompilacje
kopii sprzed nich: 22 strony, REFERENCES tak samo na 11. Moja jedna strona doszla w dodatku,
ktory sie nie liczy. Notatka w planie pracy ("the body is exactly nine pages") byla NIEPRAWDZIWA
i zostala poprawiona na stan faktyczny wraz z lista kandydatow do ciecia (par. o kompozycji,
par. 5.6 do dodatku, akapit o podstawach porownania w par. 6). Decyzja, co wyciac, nalezy do WG.

## `regional_steps`: pierwszy ruch na osi, ktora byla zablokowana

Przypomnienie stanu: obecnosc podmiotow i tozsamosc byly ROZWIAZANE (42 z 44 komorek,
DINO regionowe 0,716), a nierozwiazane zostalo "sfotografowane razem, a nie sklejone" --
podmioty renderowaly sie we wlasnym stylu i wlasnym swietle, przyklejone do sceny. Wczesniej
ustalilem, ze usterki NIE sa zwiazane z granicami pudelek, tylko sa globalna niespojnoscia
sceny, wiec kappa, feather i bootstrap ich nie naprawia -- wszystkie dzialaja na granicach.

`regional_steps = N` puszcza galaz regionowa tylko przez pierwsze N z 50 krokow odszumiania,
a ogon oddaje galezi globalnej. Hipoteza: tozsamosc ustala sie wczesnie, a styl, swiatlo
i geometria sceny pozno -- wiec oddanie ogona globalnej galezi powinno scalic podmioty ze scena,
nie tracac tozsamosci. To jest pierwsza os, ktora atakuje globalna niespojnosc, a nie granice.

Obejrzane w PELNEJ rozdzielczosci (nie na arkuszu -- na arkuszu mylilem sie piec razy):

| konfiguracja | scena 12.1 (japonska ulica, komiks) |
|---|---|
| `s055` (50 krokow, baza) | podmioty fotorealistyczne, przyklejone do komiksowego tla; swiatlo cieple, ulica chlodna; srodkowy pies stoi na drewnianej platformie znikad |
| `rs25` | **podmioty przerysowane w stylu ulicy** -- plaskie cieniowanie, ta sama kreska; swiatlo wspolne; pies ma kontakt z podlozem |
| `rs15` | **podmioty ZNIKAJA**, zostaje sam krajobraz -- 15 krokow nie wystarczy, zeby sie uformowaly |

Scena 3.4 (zlota sala) potwierdza kierunek i pokazuje koszt:

* `rs25`: znika czarny prostokat za kotem, podloga jest ciagla, pies siedzi na posadzce zamiast
  na ceglanym cokole znikad, halo wokol podmiotow slabsze;
* ale drugi obiekt w ramce misia, ktory juz w bazie mial NIEBIESKIE KOCIE OCZY, przy `rs25`
  rozstrzyga sie w strone kota -- czyli istniejaca dwuznacznosc rozwiazuje sie na niekorzysc;
* `rs35` jest posrodku: prostokat za kotem znika, ale cokol i halo zostaja.

Porzadek na spojnosci: **rs25 > rs35 > baza**. Na duplikatach kierunek wyglada odwrotnie, ale
to sa DWIE komorki i wlasnie dlatego nie wyciagam z tego wniosku.

**Mierze zamiast patrzec dalej:** job 23293436 liczy `_compose_score.py` na piec konfiguracji
(s055 / rs40 / rs35 / rs25 / rs15), 11 scen x 4 ziarna x 3 regiony. Ta miara lapie dokladnie
te os, ktora tu ryzykujemy -- brak podmiotu, duplikat sasiada, plat futra -- razem z kolumna
kontrolna "czy wlasny koncept wygrywa z cudzymi". Stylu ona nie mierzy i nie bedzie; trzy
proby zbudowania miary artefaktow skonczyly sie dwoma odwroconymi i jedna plaska.

## CIDM na 50 konceptach: dwadziescia godzin GPU dalo 10 000 CZARNYCH klatek

Lancuch generacji domkniety: `gen4` COMPLETED (3:47), `gen5` COMPLETED w 3 minuty (wszystko
juz bylo). Metryki policzone same z siebie i wygladaja jak sensacja:
**CLIP-T 0,5447 | CLIP-I 0,5257 | DINO 0,0273**, przy naszym 0,524 DINO na tym samym strumieniu.

To NIE jest zapominanie. **Wszystkie 10 000 plikow ma identyczne 4723 bajty i sa czarne.**
Sprawdzilem, bo DINO 0,027 to nie jest liczba, jaka daje zly obraz -- to liczba, jaka daje
BRAK obrazu. Gdybym tego nie otworzyl, wpisalbym do pracy "CIDM zalamuje sie przy 50
konceptach" -- falszywe i krzywdzace twierdzenie o cudzej metodzie.

Podpis w logu: `pil_utils.py:43: RuntimeWarning: invalid value encountered in cast`, pietnascie
razy na job, czyli przy kazdej partii. `numpy_to_pil` robi `(images*255).round().astype(uint8)`;
na NaN-ach rzutowanie daje zera, czyli czern. Model zwraca NaN.

### Co WYKLUCZYLEM

| hipoteza | werdykt |
|---|---|
| przepelnienie fp16 przy sumowaniu 50 adapterow | **nie** -- ich `torch_dtype=torch.float16` stoi w nawiasach `DPMSolverMultistepScheduler.from_pretrained`, a nie pipeline'u. Scheduler nie ma wag, wiec to no-op i pipeline i tak laduje sie w fp32 |
| wagi fuzji rosna z liczba adapterow | **nie** -- `weights = pow(cos, 4)`, potem `/ norm(p=1)`, czyli sumuja sie do 1 |
| pusty wycinek `task_indexes[j-1]:task_indexes[j]` -> `mean` z pustego -> NaN | **nie** -- zaden z trzech configow nie ma pustego `replace_mapping` |
| pomylony strumien v1/v2 (`task50.json` ma na pozycji 23 "bike", `task50_v2.json` "car") | **nie** -- bieg uzyl `task50_v2.json`, a nasz `configs/phaseT/T50_v2.yaml` ma na 23 `cc101_transport_car5`, class\_word `car`. Para v2 sie zgadza |

**Przyczyna nieustalona.** Zostaje `weights / weights.norm(p=1)` przy 50 zadaniach, ale nie mam
na to dowodu, a zgadywanie juz raz mnie tu kosztowalo dwie hipotezy.

### Co z tym zrobic -- moja rekomendacja: NIC

Praca tej liczby nie potrzebuje. Par. 5.5 porownuje sie na strumieniu 50-konceptowym
z kontrola bez interferencji (jeden adapter na koncept, nie ma czego zapominac) i **jawnie
argumentuje, ze to jest uczciwe porownanie na tej dlugosci**, nie zregularyzowana baza.
CIDM na 50 bylby dodatkiem.

Koszt domkniecia: znalezienie NaN-a wymaga instrumentacji ich pipeline'u (hook na pierwszy
nieskonczony tensor -- da sie bez dotykania ich plikow, opakowaniem w `scripts/`), a potem
**ponownej generacji ~20 h GPU**. Do terminu piec dni, tresc glowna jest o strone za dluga,
Helios ma 85 ze 110 wezlow niereagujacych, a Atena liczy rzeczy Z SCIEZKI KRYTYCZNEJ
(macierze CIDM na dwa ziarna i finetuning s2024 -- slupki bledu do Figury 3). Nie wchodze
w to bez decyzji WG.

**Wazne, zeby nie zginelo:** katalog `outputs/cidm50_v2/` zawiera 10 000 bezuzytecznych plikow
i `cifc_metrics.json` z liczbami, ktore wygladaja na wynik. Kazdy, kto je przeczyta bez tego
wpisu, wyciagnie falszywy wniosek. Nie kasuje niczego w `results/` bez zgody, ale ten plik
metryk trzeba albo usunac, albo oznaczyc.

### Stan klastrow

Helios: **85 ze 110 wezlow nie odpowiada** (`sinfo -R`: "Not responding", "power cycle the node",
"Need to swap GPU2"). W partycji stoi 6 zadan w kolejce -- wszystkie moje -- i 19 chodzi.
Wszystkie moje maja wyliczony start na 00:15-00:41. Nic nie padlo, po prostu nie ma na czym
liczyc. Atena zdrowa, trzy biegi sciezki krytycznej ida.

## Przyczyna czarnych klatek CIDM: okno 77 tokenow CLIP-a

Znalezione. To bug w ICH wydanym kodzie, uruchamianym poza zakresem, dla ktorego powstal,
a nie awaria naszej infrastruktury i nie wlasciwosc metody jako takiej.

### Lancuch

1. `inference.py:164-173` sklada **jeden** prompt ze sklejonych `replace_mapping`
   **wszystkich** konceptow strumienia (po dwa tokeny na koncept) i koduje go CLIP-em.
2. `encode_prompt` wola tokenizer z `padding="max_length", max_length=tokenizer.model_max_length,
   truncation=True`. Dla SD-1.5 to **77**. Nie ma dzielenia na kawalki -- nadmiar jest OBCINANY.
   Wyjscie ma ksztalt `[16, 77, 768]`, wiec `classifier_weights` ma 77 wierszy.
3. `task_indexes` liczy sie z dlugosci **NIEobcietej**: startuje od 1 i rosnie o liczbe tokenow
   konceptu. Dla 50 konceptow po 2 tokeny konczy sie na **101**.
4. `EDLoRA_FusionAttnProcessor.__call__` robi
   `mean(lora_weights[:, task_indexes[j-1]:task_indexes[j]], dim=-1)`. Dla zadan, ktorych
   wycinek zaczyna sie za 77, wycinek jest **PUSTY**, a `torch.mean` z pustego to **NaN**
   (sprawdzone: `float(torch.mean(torch.zeros(1,0), dim=-1)[0])` -> `nan`).
5. Zaraz potem `weights = weights / weights.norm(p=1, dim=-1, keepdim=True)`. Norma z wektora
   zawierajacego NaN jest NaN, wiec **wszystkie 50 wag staje sie NaN**, nie tylko te dwanascie.
6. `sum(weights * lora_outputs)` -> NaN -> wyjscie UNetu NaN -> obraz NaN ->
   `numpy_to_pil` rzutuje NaN na 0 -> **czern**, w 100% przypadkow, na kazdej warstwie
   i w kazdym kroku.

### Arytmetyka progu

| strumien | konceptow | tokenow | `task_indexes` do | pustych wycinkow |
|---|---|---|---|---|
| `task10` | 10 | 23 | 24 | **0** |
| `task50_v2` | 50 | 100 | 101 | **12** (zadania 39-50; zadanie 38 skazone przez EOS) |

**Sufit to 37 konceptow, nie 38.** Okno 77 pozycji miesci BOS, tokeny i EOS, wiec
`1 + 2N + 1 <= 77` daje `N <= 37`. Dokladny obraz przy 50 konceptach:

* koncepty **1-37**: wycinki czyste, po dwa tokeny, pozycje 1-74;
* koncept **38**: wycinek `[75, 77)` -- niepusty, ale drugi token zostal zastapiony przez EOS
  wymuszony obcieciem, wiec waga jest liczona z czegos, co nie jest jego tokenem;
* koncepty **39-50**: wycinki puste -> `mean` = NaN -> po normalizacji L1 **wszystkie 50 wag**
  jest NaN.

(Osobna proba policzenia tego tokenizerem dala te sama liczbe, ale metoda bledna -- nie
rejestrowalem w nim nowych tokenow, wiec `<c01a>` rozpadalo sie na kilka BPE. Nie cytuje jej;
powyzsze opiera sie na arytmetyce okna i na `task_indexes`, ktore policzylem z configu.)

### Co to potwierdza, a czego nie

Potwierdzone na ich wlasnym kodzie, bez GPU: dla 10 konceptow okno CLIP ma 77, `task_indexes`
konczy sie na 21, pustych wycinkow 0, **NaN-ow w wagach fuzji 0**. To zgadza sie z tym, ze
nasz bieg dziesieciokonceptowy jest zdrowy (DINO 0,577).

Wykluczone wczesniej i nadal wykluczone: przepelnienie fp16 (ich `torch_dtype=torch.float16`
jest w nawiasach schedulera, wiec no-op -- pipeline jest fp32), zepsute checkpointy
(sprawdzone wszystkie 50: zero NaN/inf, `max|w|` rowne 0,08-0,09 w kazdym), pusty
`replace_mapping` w configu (nie ma takiego), pomylony strumien v1/v2 (para v2 sie zgadza).

**Czego to NIE mowi:** ze metoda CIDM zalamuje sie przy 50 konceptach. Mowi, ze ich routing
trzyma tokeny wszystkich konceptow naraz w jednym kontekscie CLIP-a, a ten ma 77 pozycji.
To jest ograniczenie implementacji i zarazem realne ograniczenie projektu tego routingu --
ale zeby cokolwiek o metodzie twierdzic, trzeba by ja najpierw naprawic i zmierzyc.

### Naprawa, gdybysmy chcieli wiersz CIDM na 50

Rachunek w procesorze to srednie po wycinkach tokenow, wiec kodowanie promptu **w kawalkach
po 77 tokenow i sklejenie wierszy `classifier_weights`** zachowuje go dokladnie. To ~15 linii
w opakowaniu, bez dotykania ich plikow. Potem **~20 h GPU na ponowna generacje**.

Koszt vs pozytek: par. 5.5 porownuje sie na tym strumieniu z kontrola bez interferencji
i jawnie argumentuje, ze to jest uczciwe porownanie na tej dlugosci. Do terminu piec dni,
tresc jest o strone za dluga. Decyzja WG.

### Czy sufit 77 tokenow dotyka macierzy, ktore licza sie teraz? NIE

Pytanie wroci, wiec na pismie. Biegi `ch-cidmmx-s1/s2` (3187117, 3187135) ida na strumieniu
**dziesieciokonceptowym**: 10 x 2 = 20 tokenow plus BOS i EOS, czyli 22 z 77. Obcięcia nie ma.
Dowod nie z arytmetyki tylko z pomiaru: trzecia taka macierz juz sie policzyla (3187134,
6 h 48 min) i dala DINO 0,577 oraz forgetting 0,0228 -- liczbe, ktora stoi w pracy.

Sufit dotyczy wylacznie strumienia 50-konceptowego. Gdyby robic krzywa do T = 50, punkty
T = 10 / 20 / 30 / 37 zajmuja 22 / 42 / 62 / **76** pozycji, wiec wszystkie sie mieszcza,
a 37 jest ostatnim wlasnie dlatego, ze 76 <= 77.

Stan na 22:5x: s1 na komorce `k=9 j=2` (48 z 55), s2 na `k=9 j=3` (49 z 55). Tempo 13,1 min
na komorke przy dwoch zadaniach na wezle t0015; zostalo po ~1,5 h przy limicie 5 h 29 min.

### Prog zmierzony, nie wyliczony

Sonda na ich kodzie (`nan_probe.py`, bez GPU, ten sam config przyciety do N konceptow, wiec
miedzy wariantami zmienia sie WYLACZNIE ich liczba):

| konceptow | okno CLIP | `task_indexes` do | pustych wycinkow | NaN w wagach fuzji |
|---|---|---|---|---|
| 10 | 77 | 21 | 0 | 0 / 20 |
| 20 | 77 | 41 | 0 | 0 / 40 |
| 30 | 77 | 61 | 0 | 0 / 60 |
| 37 | 77 | 75 | 0 | 0 / 74 |
| **38** | 77 | 77 | 0 | **0 / 76** |
| **39** | 77 | 79 | **1** | **78 / 78** |
| 50 | 77 | 101 | 12 | 100 / 100 |

Dwie rzeczy widac wprost. Klif jest miedzy 38 a 39, nie stopniowy. I **jeden** pusty wycinek
przy N = 39 psuje **wszystkie 78** wag -- bo norma L1 z wektora zawierajacego NaN jest NaN,
wiec po `weights / weights.norm(p=1)` NaN jest wszedzie. To zamyka lancuch: jeden `mean`
z pustego tensora wystarczy, zeby caly obraz wyszedl czarny.

Na krzywej zatrzymujemy sie na **37**, mimo ze 38 nie daje NaN: przy 38 mamy 76 tokenow
konceptow plus BOS, wiec EOS wypada na granicy obcięcia i wycinek ostatniego konceptu liczy
sie czesciowo z EOS. 37 to ostatni punkt w pelni czysty.

## Krzywa retencji do T = 50: co juz jest, czego brakuje

Zbior oceny zamrozony: **dziesiec konceptow CIFC**, ktore sa zadaniami 1-10 strumienia
piecdziesieciokonceptowego (sprawdzone: koncept 10 to juz `cc101_actionfigure_1`). W kazdym
punkcie zmienia sie model, nie zbior -- wiec koszt punktu jest staly, 10 komorek, i znika
confound roznej trudnosci konceptow, ktory na krzywej T = 1..10 daje zabkowanie wieksze niz
pasmo trzech ziaren.

Biegiem naszej strony jest **`outputs/sweep/p022_v2`** (Atena) -- to z niego pochodzi wiersz
T = 50 w pracy (job 3185126, `--eval_root outputs/sweep/p022_v2/all50/s045`).

| co | stan |
|---|---|
| nasze T = 50 | **JUZ JEST**: `all50/s045`, komorki `49,0..9`, TA 78,10 / IA 75,02 / **DINO 0,5610** -- zgadza sie z "0,561 po piecdziesieciu zadaniach" w par. 5.5 |
| nasze checkpointy zadan 9, 19, 29, 36, 49 | wszystkie obecne w `p022_v2/ckpts` |
| nasze T = 10 / 20 / 30 / 37 | do policzenia, 40 komorek, jednym biegiem `gen_cifc --only_tasks 9,19,29,36 --only_concepts <dziesiatka>` -- **bez zmiany kodu**, te flagi juz sa |
| CIDM T = 10 / 20 / 30 / 37 | do policzenia, 40 komorek, ~5 h przy zmierzonych 7,4 min/komorke |
| CIDM T > 37 | **nie istnieje i nie bedzie istniec** -- okno 77 tokenow |

Jedyna zmiana kodu: `--eval_first N` w `scripts/_cidm_gen.py` (ich petla oceny szla zawsze po
wszystkich k+1 konceptach, wiec bez tego punkt T = 37 kosztowalby 37 komorek zamiast 10).
Fuzja adapterow zostaje pelna 1..k+1; ogranicza sie wylacznie zbior mierzony. Domyslnie 0,
czyli sciezka bez flagi jest bit w bit ta sama.

**Uwaga: `cidm_bench10` NIE nadaje sie na punkt T = 10 tej krzywej** -- pochodzi z innego
treningu (`task10.json` wskazuje tokeny `<dog1>`, a biezacy `./output/task_1` ma `<c01a>`),
wiec trzeba go wygenerowac od nowa z checkpointow biegu piecdziesieciokonceptowego.
