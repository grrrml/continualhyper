# ContinualHyper — raport stanu projektu

> Żywy dokument-pamięć projektu: syntetyczny obraz "gdzie jesteśmy i skąd to wiemy".
> Aktualizowany po każdym domknięciu wątku (werdykt, faza, decyzja ramowa).
> Szczegółowy dziennik pomiarów i odrzuceń: `assets/STATUS.md`. Stan na: **2026-09-07**.

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

## 6. W toku / otwarte

**W kolejce (Helios, 2026-09-07) — każde rozstrzyga jedno pytanie:**
- **Po przeglądzie SDXL (commit `1686063`, wysłane 2026-09-07 wieczór):** (a) **22098594**
  `X_sdxl_ground_800aug` eval @s=0.4/0.5/0.6 → `eval10f_uc/` z poprawioną gałęzią uncond — efekt
  samej poprawki (1) wobec istniejącego `eval10f/` (ten sam checkpoint, te same ziarna);
  (b) retrening **22098601** `X_sdxl_base800` (**czy sufit tekstowy istnieje** — baza bez groundingu
  przy 800, z poprawnym mikro-warunkowaniem) + evale 22099013/23/27 (s=0.4/0.5/0.7) i **22098605**
  `X_sdxl_best_nu` (fp32 logity + `paste_no_upscale`) + evale 22099028/30/32 (s=0.4/0.5/0.6),
  łańcuchy na `afterany`; (c) potem instrument umiejscowienia SDXL na nowym `best_nu` i κ=0.5/0.7 —
  anulowane 22094511 / 22087569/70 liczyły stary checkpoint. `X_sdxl_1600` evale 22077405/07
  anulowane (wynik znany, −1.49; 1600 nasyca się).

**Do zrobienia przed wysyłką (ścieżka krytyczna to pisanie, nie kolejka):**
- Drugie ziarno `P_best` + jego pełna macierz forgettingu (tabela forgettingu w pracy jest
  wciąż z `erode`).
- Trzecie ziarno SDXL wybranego przepisu — bez tego żadne zdanie o SDXL nie ma wagi.
- Abstract, Introduction, Related work, Limitations, Conclusion — puste `\todo{}`.
- Transfer w przód — zmierzony, **ani zdania w tekście**.
- `REPORT.md` → `git commit` (nie commitowany od 2026-09-06).

**Następna wersja (nie ta):** sweep T=35 na CustomConcept101 na obu architekturach; bramka
`g + h(ramka)` i kotwica wkładu bezramkowego; skalowanie gałęzi przez `s_lora`; Option C
dwuenkoderowe na SDXL; letterbox z bboxem przez korelację wzorca; L2DM na SDXL (OOM); kompozycja
wielokonceptowa (ITP/RTP); `R_tail` trzecie ziarno na SD-1.5.

**Nie robić (zmierzone albo rozstrzygnięte):** `share_heads` pod groundingiem (−1.6…−1.9 IA,
−16 pp placementu); kotwica na gałąź (zamraża transfer w przód);
`prompt_aug` (−1.40 IA na SDXL); `paste_scale_full_p` (−2.4 IA); `boxonly` (LoRA nie przejmuje
tożsamości: 53.6/55.0 wobec 62.3); 1600 kroków (nasyca się); tokeny groundingu bez klucza
(= GLIGEN bez semantyki, claim (ii) upada); rozdzielanie tożsamości od umiejscowienia przez
wejście gałęzi (wyciek bierze się z sygnału treningowego, nie z wejścia); podnoszenie rangi /
szerokości głowicy per warstwa; gonienie wypełnienia kadru pod metrykę.

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

## 8. Konwencja aktualizacji

Po każdym domkniętym wątku: zaktualizować sekcję wyników lub sagę ramek (tabela wariantów),
przenieść pozycje z "W toku" do właściwych sekcji, datę w nagłówku. Szczegóły i pełne liczby
zawsze w `assets/STATUS.md`; tu tylko synteza, którą da się przeczytać w 5 minut.
