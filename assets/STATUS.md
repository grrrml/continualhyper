# ContinualHyper — status

Stan na 2026-08-07, 12:48. Wszystkie liczby to odczyty **matched-TA** przy TA = 0.748 (TA raportowane
przez CIDM), eval @10 obrazów/prompt, chyba że zaznaczono inaczej. Odniesienie zewnętrzne:
**CIDM = TA 0.748 / IA 0.780**.

---

## 0. DECYZJA O WERSJI GŁÓWNEJ (2026-08-10)

**Wersją główną metody jest `F_base`**: wejście hipersieci = 128-wymiarowy losowy wektor
ortonormalny per task (deterministyczny z seeda), h50, rank 4, attn2 q/k/v/out, maska ON, 21.1 M.
Uzasadnienie: na niej wiszą wszystkie weryfikacje (lora_solo 2 seedy + sweep, ablacje β/h/rank),
jest najprostsza do opisania, a każdy dokładany mechanizm albo nie przekroczył szumu, albo czeka
na potwierdzenie. `R_tail` (+0.011 sparowane, 2 seedy zgodne, brak trzeciego) → sekcja ablacji
jako „obiecujące, niepotwierdzone". Kombinacja `R_tail+nomask` na bazie F — NIEZBADANA.

## 0b. NOWA RAMA I MAPA DROGOWA (decyzja użytkownika, 2026-08-10)

**Setting = ścisłe CL:** model o ograniczonej pamięci, zadania sekwencyjnie, bez rosnącego
składowania per task. `lora_solo` i CIDM → referencje poza settingiem (pamięć O(T)).
**Liga właściwa:** finetune, EWC, LwF, C-LoRA, L2DM.

**FINALNA TABELA (2026-08-12, pełne krzywe zwycięzców λ, matched-TA):**

| metoda | zakres TA | Δ@0.748 | Δ@0.755 | Δ@0.762 |
|---|---|---|---|---|
| LwF λ=10 | [0.746, 0.785] | **+0.070** | +0.065 | +0.056 |
| EWC λ=1e5 | [0.759, 0.783] | — | — | **+0.081** |
| C-LoRA λ=100 | [0.716, 0.761] | **+0.122** | +0.122 | — |
| finetune | [0.661, 0.801] | +0.238 | +0.221 | +0.200 |
| L2DM | [0.711, 0.797] | +0.377 | +0.343 | +0.306 |

(Δ = przewaga F_base w DINO; wszystkie 6–40× ponad próg szumu 0.0091.)

**Forgetting z pełnych macierzy (peak−final, 55 komórek, s08) — KOMPLET:**
F_base **0.0015** | LwF 0.0183 | EWC 0.0363 | C-LoRA 0.0365 | L2DM 0.1591 | finetune 0.2363.
Dominacja na obu osiach naraz: jakość 6–40× nad szumem, forgetting 12–150× mniejszy.

**F_base, 3 seedy @ TA=0.748: DINO 0.6415 ± 0.0012, IA 0.8054 ± 0.0017.** Rozrzut headline'u
jest 8× mniejszy niż kalibracja międzyseedowa mechanizmów (0.0091) — konfiguracja główna jest
wyjątkowo stabilna.

**Forgetting F_base z PEŁNEJ macierzy (peak−final, 55 komórek, s08): DINO 0.0015, CLIP-I 0.0011**
— praktycznie zero, już nie deklaracja tylko pomiar.

**Stara tabela @ s08 (jedna skala, historycznie):**

| metoda | TA | IA | DINO | my@ich TA | przewaga DINO |
|---|---|---|---|---|---|
| **F_base** | 0.736 | 0.814 | 0.652 | — | — |
| LwF λ=10 | 0.768 | 0.780 | 0.563 | ~0.608 | **+0.045** |
| EWC λ=1e5 | 0.772 | 0.759 | 0.532 | ~0.605 | **+0.073** |
| C-LoRA λ=100 | 0.739 | 0.761 | 0.520 | ~0.647 | **+0.127** |
| finetune | 0.764 | 0.709 | 0.423 | ~0.612 | +0.189 |
| L2DM | 0.763 | 0.682 | 0.318 | ~0.613 | +0.295 |

W tej lidze przewaga jest o rząd większa niż wszystko z sekcji 5 (+0.045…+0.30 wobec ±0.01).
Zastrzeżenia: zwycięzcy λ mają 1 skalę (pełne krzywe: joby 2893673-75), macierze forgettingu
niepoliczone, F_base 1 seed.

**KOLUMNA SDXL (2026-08-12, F_sdxl = F_base na SDXL bez zmian hiperparametrow, 87.5M hiper):**
s05: TA 0.822/IA 0.771/DINO 0.570 | s07: 0.795/0.796/0.618 | s08: 0.780/0.806/0.634 |
s10: 0.748/0.824/0.652. Wobec ich publikacji SDXL (TA 0.800/IA 0.795): przy TA 0.800 nasze
IA ≈ 0.791 — PARYTET (−0.004). Przenosnosc: SDXL @TA 0.748 daje IA 0.824 vs 0.806 na SD-1.5.
Hiperparametry przeniosly sie 1:1; "degeneracja" z fresh/final byla artefaktem renderowania 512px
(SDXL poza dystrybucja) — naprawione w trenerze; retrainy lr5e5/st200 = niepotrzebna ablacja
(ckpty sa, evale skasowane).

**Mapa drogowa:** (1) domknąć zwykły CL na SD-1.5 (krzywe ligi, forgetting, seedy F_base) →
(2) port SDXL i zamknięcie CL na obu backbone'ach → (3) dopiero potem kompozycja — MECHANIZM DOCELOWY (decyzja
2026-08-10): **ramka jako wejście hipersieci** — (cx,cy,w,h) → Fourier → MLP → cond_dim
(zero-init); nadzór z augmentacji rozmieszczeniem (znane p, rekonstrukcja karze złe położenie);
domyślna ramka = pełny kadr (protokół jednokonceptowy bez zmian); kotwice na kadrze pełnym.
Delty natywnie stawiają podmiot w ramce — atakuje zmierzony tryb porażki (podmioty wyśrodkowane
sklejają się). GO/NO-GO przed kompozycją: zgodność rozmieszczenia jednokonceptowo, bez masek
runtime (~45 min SD-1.5). Ryzyko: delta jednorodna przestrzennie (GLIGEN wybrał architekturę);
przepis rów. 5 + bootstrap zostaje jako plan B.

## 1. Gdzie jesteśmy wobec CIDM

| konfiguracja | parametry hipersieci | DINO@0.748 | IA@0.748 | vs CIDM |
|---|---|---|---|---|
| a2f (dotychczasowy headline) | 50.2 M | 0.629 | 0.803 | +0.023 IA |
| A_nonorm | 50.2 M | 0.637 | 0.805 | +0.025 IA |
| **F_base** — h50 + klucze ortonormalne 128 | **21.1 M** | **0.640** | **0.806** | **+0.026 IA** |
| B_nomask_nonorm (maska OFF) | 50.2 M | 0.652 | 0.812 | +0.032 IA |

### KOSZT PAMIĘCI — sprostowanie (zmierzone 2026-08-09 na wytrenowanych checkpointach)

Dotychczasowa liczba „0.398 M/zadanie vs ich 0.435 M" dotyczy **wypieczonej** LoRA i **wprowadza
w błąd** — my nie pieczemy, tylko generujemy z hipersieci, więc naszym kosztem jest jej rozmiar.

| | na zadanie | **razem @ T=10** |
|---|---|---|
| `lora_solo` (10 niezależnych) | 0.398 M | **3.98 M** |
| CIDM (LoRA 0.398 + embeddingi 0.025–0.037; text-enc LoRA = 0.000 M, martwy kod POTWIERDZONY) | 0.426 M | **4.26 M** |
| **my** (hipersieć) | ~0 (klucz deterministyczny z seeda) | **21.12 M** |

**Przy 10 zadaniach jesteśmy 5× więksi, nie mniejsi.** Próg opłacalności: **~49 konceptów**.

Poprawna teza to skalowanie, nie rozmiar: oni **O(T)**, my **O(1)**. Ale stała jest duża, a teoria
nasycenia (`obciążenie headu = T·|b|`, sekcja 4) mówi, że pojemność się wyczerpuje — więc
twierdzenie „stała pamięć dla dowolnie wielu zadań" **wymaga sweepu `h` przy T=35** (Tier-1,
niezrobione). Bez tego nie wolno go stawiać.

**Wszystkie trzy metody to dziesięć adapterów.** Różnica jest w tym, jak powstają i jak są wybierane:

| | jak powstają | jak wybierane | razem @T=10 |
|---|---|---|---|
| `lora_solo` | niezależnie | oracle indeks | 3.98 M |
| CIDM | niezależnie + kara ortogonalności | miękki cosinus, **0.613** | 4.26 M |
| my | z jednej hipersieci | token, **0.999** | 21.12 M |

Zweryfikowane +0.028 nad `lora_solo` izoluje **wyłącznie współdzieloną parametryzację** — i kosztuje
5× pamięci przy T=10.

`F_base` to najlepsza konfiguracja **z maską**, przy 2.4× mniejszej hipersieci. Wariant bez maski jest
lepszy o 0.012 DINO, ale maska została utrzymana decyzją projektową ze względu na kompozycję.

---

## 1b. ~~UZASADNIENIE HIPERSIECI~~ → ANALIZA POZA SETTINGIEM (decyzja 2026-08-10)

**DECYZJA: `lora_solo` wypada z narracji jako baseline.** Dziesięć niezależnych treningów
z per-taskowym magazynem i oracle'owym indeksem NIE jest metodą ciągłego uczenia — to rozwiązanie
ignorujące problem CL (brak ograniczenia pamięci, brak współdzielonego modelu). W settingu
„model o ograniczonej pamięci, zadania sekwencyjnie" jest niedopuszczalne. Pomiary poniżej
zostają jako ANALIZA (odpowiadają na pytanie „skąd bierze się jakość"), nie jako uzasadnienie.

Konsekwencje: (a) właściwa liga porównawcza = klasyczne baseline'y CL (finetune, EWC, LwF,
C-LoRA, L2DM — wytrenowane) + CIDM; (b) +0.028 nad niezależnymi adapterami czytamy jako wynik
o priorze wielozadaniowym, nie jako headline; (c) UWAGA recenzencka: pytanie „a czemu nie
10 osobnych LoRA po 0.4M?" i tak padnie — odpowiedzią jest definicja settingu, i trzeba ją
w artykule postawić jawnie, zanim postawi ją recenzent.

Baseline kontrolny `lora_solo`: **dziesięć niezależnie wytrenowanych LoRA, właściwa wybierana
oracle'owym indeksem zadania**. Ten sam rozmiar wdrożenia (0.398 M/zadanie), to samo wymaganie
podania konceptu przy inferencji, zerowe zapominanie z konstrukcji. Różni się od nas **wyłącznie**
brakiem współdzielonej parametryzacji i regularyzacji von Oswalda.

| TA | `F_base` | `lora_solo` | ΔDINO | ΔIA |
|---|---|---|---|---|
| 0.745 | 0.6432 | 0.6218 | +0.0213 | +0.0095 |
| 0.755 | 0.6321 | 0.6033 | +0.0288 | +0.0119 |
| 0.765 | 0.6155 | 0.5848 | +0.0307 | +0.0118 |
| 0.770 | 0.6071 | 0.5756 | +0.0316 | +0.0118 |
| | | **średnio** | **+0.0279** | **+0.0112** |

**Efekt jest 3× większy od rozrzutu międzyseedowego (0.0091)** — jedyna zmierzona w tym projekcie
różnica, która pewnie przekracza próg.

**Mechanizm:** regularyzacja wielozadaniowa w reżimie kilku obrazów. Przy 3–7 zdjęciach na koncept
niezależna LoRA przeucza się na własnym zbiorku; współdzielone heady dają prior z pozostałych zadań.
To jedyna hipoteza, która przeżyła — kompresja, generalizacja i stała pamięć zostały obalone
pomiarami (sekcja 4), a siedem prób mechanistycznych nic nie dało (sekcja 5).

**ZWERYFIKOWANE (2026-08-09) — oba zastrzeżenia zamknięte:**

| konfiguracja baseline'u | zakres TA | średnia przewaga hipersieci |
|---|---|---|
| 800 kr, lr 1e-4, seed 2024 | [0.685, 0.771] | +0.0281 |
| 800 kr, lr 1e-4, **seed 2025** | [0.687, 0.767] | **+0.0361** |
| 400 kroków (parytet z naszą metodą) | [0.694, 0.770] | +0.0351 |
| 200 kroków | [0.710, 0.779] | +0.0323 |
| **lr 5e-5 — NAJLEPSZY baseline** | [0.690, 0.768] | **+0.0258** |

1. **Drugi seed potwierdza:** średnia dwóch seedów **+0.0315**, przedział [+0.0273, +0.0339].
   Bez zmiany znaku — w przeciwieństwie do bramki i klucza semantycznego.
2. **Sweep baseline'u wykonany, hipoteza przeuczenia OBALONA:** krótszy trening baseline'owi
   **szkodzi** (400 kr: +0.035, 200 kr: +0.032 na naszą korzyść). Najlepszą konfiguracją okazał
   się słabszy lr — i nadal przegrywa o **+0.0258 = 2.8× próg szumu**. Baseline dostał swoje
   maksimum z czterech konfiguracji.

**Wniosek: uzasadnienie hipersieci jest zweryfikowane.** Prior wielozadaniowy w reżimie kilku
obrazów to jedyny potwierdzony mechanizm przewagi w całym projekcie.

---

## 1c. UCZCIWOŚĆ PORÓWNANIA Z CIDM — zmierzona asymetria

**Jak CIDM naprawdę wybiera adapter** (`lib/models/edlora.py:222-253`, zweryfikowane w kodzie):
ładuje **wszystkie 10 LoRA** i liczy rozkład po konceptach z samego promptu — cosinus między
zapamiętanymi embeddingami tokenów a stanami ukrytymi promptu, max po pozycjach, średnia po
tokenach konceptu, `pow(·,4)`, normalizacja L1. Wyjście = **ważona suma dziesięciu adapterów**.
Nie potrzebują indeksu zadania; my potrzebujemy.

**Zmierzone wagi routingu** (własne checkpointy CIDM, 10 tasków, bez generowania obrazów):

| | waga właściwego konceptu | wyciek na obce |
|---|---|---|
| średnio po 10 konceptach | **0.613** | **0.387** |
| najlepszy (teddy bear) | 0.807 | 0.193 |
| najgorszy (**cat2**) | 0.399 | 0.601 |
| drugi duplikat klasowy (**dog2**) | 0.547 | 0.453 |

Wyciek jest **rozproszony** (najsilniejszy obcy koncept 0.05–0.10), czyli to rozcieńczenie, nie
pomyłka. Najgorzej routują się **oba duplikaty klasowe** — ta sama zapaść same-class, którą u nas
rozwiązują klucze ortogonalne, u nich uderza w routing.

**Konsekwencja dla uczciwości porównania:**
- **vs `lora_solo` — ODPORNE.** Ten baseline też dostaje oracle'owy indeks, też nie miesza
  adapterów. Więc **+0.026…+0.036 izoluje wkład hipersieci** i jest czyste.
- **vs CIDM — SKAŻONE.** Dostajemy wagę 1.0 z konstrukcji, oni efektywnie 0.61. Część
  raportowanego +0.026 IA to ich rozcieńczenie routingiem, nie nasza metoda.

**Do rozstrzygnięcia generacją:** waga 0.61 nie musi znaczyć 39% straty jakości. Porównanie
CIDM one-hot vs CIDM miękki routing (ich wagi, nasze metryki) daje koszt routingu w metrykach.

---

## 1d. KOMPOZYCJA WIELOKONCEPTOWA W JEDNYM PRZEBIEGU (2026-08-09, jakościowo)

**Jak robi to CIDM** (zweryfikowane w pracy, arXiv 2410.17594 §4.3): „region noise estimation" —
dla każdego z `U` regionów osobna predykcja szumu warunkowana promptem regionu, scalane binarnymi
maskami bboxów: `E* = αE + Σ(1−α)E_u ⊙ m_u`, α = 0.1. Czyli **U+1 przebiegów UNetu na krok**,
bboxy wymagane. Wielokoncept oceniany **wyłącznie jakościowo** (Rys. 3 i 12) — brak jakiejkolwiek
tabeli liczbowej, mimo że „concept neglect" jest ich deklarowanym wkładem. W wydanym kodzie
ścieżki kompozycji nie ma (`RegionT2I_AttnProcessor` istnieje, ale nic go nie importuje).

**Nasz mechanizm — JEDEN przebieg**, koszt niezależny od `U`. Cztery maski, każda naprawia inny
zmierzony tryb porażki:

| maska | naprawia |
|---|---|
| tokenowa na `to_k`/`to_v` | — (mieliśmy wcześniej) |
| **przestrzenna na `to_q`/`to_out`** | koncept zajmował cały kadr (przy s=1.0 kot zamieniał psa w kota) |
| **regionalna uwaga krzyżowa `attn2`** | pozycje regionu patrzyły na cudze tokeny |
| **regionalna samo-uwaga `attn1`** | cechy przeciekały między regionami — dwa psy zlewały się w jednego |
| **maski z map uwagi od kroku 10** | prostokąt przycinał psa: 1/3 sylwetki dostawała sierść kota |

Maski z uwagi: akumulacja map cross-attn per koncept → `argmax` po konceptach (**rozłączność
z konstrukcji**) → próg `tau` odcina tło (żaden adapter nie maluje piasku). Bboxy potrzebne tylko
przez pierwsze ~10 kroków jako prior układu.

**WERDYKT: NIE DZIAŁA WIARYGODNIE.** Z trzech próbek pary pies+pies2 (`R_tail`, s=0.7, maski
z uwagi) jedna jest czysta, druga zawiera **trzy psy zamiast dwóch**. Wcześniejsza wersja tego
wpisu twierdziła, że mechanizm działa — to była ocena na podstawie **wyłącznie `0.png` przy każdym
wariancie**, czyli cherry-picking. Poprawione po obejrzeniu pozostałych próbek.

**Wariant BEZ bboxów: odrzucony pomiarem.** Maski wyprowadzone z uwagi bez priora układu wychodzą
skrajnie niezbalansowane (pokrycia 0.89/0.11, 0.94/0.06, 0.92/0.08) — jeden koncept zagarnia kadr,
drugiego podmiotu nie ma w ogóle. Teza „kompozycja bez wejścia przestrzennego" upada.

**Co zostaje ustalone mimo to:** (a) `F_base` nie potrafi pary same-class, bo bez identyfikatorów
prompt „a dog and a dog" nie rozróżnia konceptów — to uzasadnia kroki (ii)/(iii) lepiej niż ich
+0.011; (b) każda z czterech masek naprawia konkretny, zaobserwowany tryb porażki; (c) mechanizm
wymaga bboxów, czyli **tego samego wejścia co CIDM** — przewagi „mniejszy wkład użytkownika" nie ma.

**WĄTEK ZAMKNIĘTY (2026-08-09 wieczór) — pięć mechanizmów, żaden nie działa wiarygodnie:**

| mechanizm | wynik |
|---|---|
| suma delt + maski tokenowe | koncepty się mieszają / zapadają do jednego |
| jeden przebieg + maski przestrzenne + regionalna uwaga | 1/3 próbek czysta; trzy psy zamiast dwóch w innej |
| maski z map uwagi bez bboxów | pokrycia 0.89/0.11 — drugi podmiot znika |
| U+1 (rekonstrukcja ich eq. 4, z confine) | jeden wyśrodkowany pies; kara tokenowa nie działa,
bo przyczynowość CLIP rozsmarowuje koncept na wszystkie następne tokeny (audyt 2885915) |
| podmiana K/V per region (region_rewrite) ± izolacja attn1 | jeden hybryd; z izolacją attn1 —
artefakty (samo-uwaga jest tym, co skleja obraz; nie da się jej wyciąć) |

**Wniosek mechanistyczny:** warunkowanie regionalne nie wymusza liczby podmiotów — trajektoria
odszumiania jest globalna (wspólny latent + samo-uwaga), model stawia jednego spójnego psa
w poprzek granicy. Ich metoda robi to na SDXL i bez wydanego kodu; nasza rekonstrukcja
z równań pracy NIE reprodukuje kompozycji na SD-1.5. To jest obserwacja do artykułu
(niereprodukowalność), nie mechanizm do użycia.

**KOREKTA po bisekcji (2885991):** artefakty z runów kv2/kv3 to był błąd sondy (pozostawiona
izolacja `attn1`), nie mechanizmu. Wierna rekonstrukcja regionalnej uwagi (`region_rewrite`,
`q` globalne, ciasne ramki, tło z promptu globalnego) działa poprawnie numerycznie — i **redukuje
dwa podmioty do jednego** (sam prompt globalny: dwa psy; z regionami: jeden). Regionalna uwaga
trasuje treść, nie wymusza liczby podmiotów. Scalanie szumu (ich rów. 5) testowane tylko naiwnie,
bez bootstrapu wnętrza ramki (standard z MultiDiffusion, w ich pracy nieopisany).

**FINAŁ WĄTKU — rów. 5 + bootstrap (2026-08-10):** scalanie szumu z bootstrapem wnętrza ramki
(MultiDiffusion; w pracy CIDM NIEOPISANY, a konieczny) daje **2/3 próbek z dwoma podmiotami**
wobec 0/3 naiwnie — diagnoza potwierdzona, implementacja kompletna. Granice na SD-1.5: 1/3 próbek
gubi drugi podmiot, tożsamości niestabilne (dog1 odpływa do białego psa; dog2 zawsze częściowy),
szwy scalania. Przepis zwalidowany, backbone jest wąskim gardłem.

**DECYZJA (2026-08-09): kompozycja ZAPARKOWANA do portu SDXL** (Faza 5 Tier-1 dostaje gotowy,
sprawdzony przepis: rów. 5 + bootstrap 10–20 kroków + ciasne ramki + tło z promptu globalnego). CIDM pokazuje
kompozycję wyłącznie na SDXL; na SD-1.5 nie da się rozdzielić „mechanizm nie działa" od „backbone
nie trzyma układu wielopodmiotowego" — każdy wynik byłby niediagnostyczny i podważalny. Brakujący
test (rów. 5 + bootstrap) ma sens dopiero na SDXL.

**Trwałe ustalenia wątku:** (a) `F_base` nie potrafi pary same-class (prompt nie rozróżnia),
`R_tail` z identyfikatorami potrafi ją wyrazić — identyfikator jest warunkiem zdolności;
(b) regionalna uwaga bez zewnętrznego wymuszenia układu jest na SD-1.5 niedookreślona;
(c) izolacja samo-uwagi niszczy generację — attn1 jest tym, co skleja obraz.

---

## 1e. CIDM ZREPRODUKOWANY — sanity check protokołu ZALICZONY (2026-08-13)

Ich kod, ich komenda (`accelerate launch --num_processes 2`), 5 udokumentowanych łat zgodności
środowiska. **@α=0.8: TA 0.753 / IA 0.779 wobec opublikowanych 0.748 / 0.780** — zgodność
±0.005, kryterium planu (±0.01) spełnione. Diagnoza efektywnego batcha potwierdzona
(v1 batch=1: 0.765/0.762 — niedotrenowany).

**Porównanie flagowe w NASZYM protokole, matched-TA @0.748:**
F_base (3 seedy: IA 0.8054±0.0017, DINO 0.6415±0.0012) vs CIDM v2 (α=1.0, TA 0.747):
**+0.019 IA, +0.040 DINO** — 4.4× próg szumu, na identycznym protokole i metrykach.

**Sweep L2DM (@0.8):** mniej replayu i więcej destylacji pomagają NIEZALEŻNIE:
α=0.1 → DINO 0.480 (+0.16 nad domyślne!), γ=10 → 0.471; β w obie strony ~neutralne/gorsze.
Wiersz L2DM w lidze do podmiany na najlepszy; naturalna runda 2 = kombinacja α0.1+γ10.
Nasz sweep czyni ich L2DM MOCNIEJSZYM niż w ich własnej publikacji na DINO — zarzut chochoła zamknięty.

**SDXL — warianty na przebicie 0.795 IA:** h100/h150 OBALONE (h150 katastrofa: TA 0.57,
DINO 0.21 — nadadaptacja; h50 przenosi się poprawnie, intuicja „szersze warstwy → większe h"
przetestowana i odrzucona). lr5e5 ≈ baza (+0.005 IA @TA~0.805), st200 gorszy. Wyostrzony odczyt
(s06): IA@TA0.800 = 0.792 vs ich 0.795 — **parytet w szumie, bez przebicia**. Pozostałe dźwignie:
nomask (±0.015 wg SD-1.5, DECYZJA UŻYTKOWNIKA) i port R_tail (+0.011 wg SD-1.5).

## 1g. NIEZGODNOŚĆ PROTOKOŁÓW PROMPTÓW — wykryta przez użytkownika (2026-08-14) i naprawiana

Nasz harness dodaje `eval_prefix` („yellow rubber" kaczka, „red" plecak) do promptu generacji
ORAZ kandydata CLIP-T; pipeline CIDM (ich inference.py) — nie. Skutki: (a) grid wizualny CIDM
miał niebieski plecak, bo prompt nigdy nie mówił „red"; (b) **wiersz flagowy my-vs-CIDM mieszał
protokoły** — część +0.019 IA może być prefiksem. Naprawa: `--replace_prompt` z prefiksem
(generacja i tekst CLIP-T stają się identyczne z naszymi); 4 alfy „CIDM-prefixed" w kolejce
(2900514-17). Krzywa natywna zostaje wyłącznie jako sanity check reprodukcji (±0.005 od
publikacji). Po alfach: macierz prefiksowana do grida porównawczego. Wszystkie wiersze
NASZEGO harnessa (F_base, liga CL) są między sobą spójne — problem dotyczył tylko mostka do CIDM.

## 1i. RAMKA JAKO WEJŚCIE HIPERSIECI — NO-GO (2026-08-16, dwunaste odrzucenie)

Mechanizm kroku 3 (Fourier(cx,cy,w,h)→MLP→cond, wklejka + strata maskowana do ramki, modulacja
ZA Gram-Schmidtem — zweryfikowane, że GS jej nie zjada, siła identyczna na task 0 i 9):
**zgodność połówek 41/84 = 48.8% (= przypadek; próg 80%)**, pełnokadrowe DINO 0.613 vs 0.634
bazy (−0.021, poza szumem). `box_emb` trenował się (norma 0.82), ale nauczona modulacja nie
steruje położeniem — potwierdzenie hipotezy GLIGEN: placement wymaga architektury (tokeny ramek
w uwadze), nie delt wag. Werdykt na SD-1.5 uznany za rozstrzygający mechanistycznie (null 49%
vs 80% to własność wyrazistości delty). Kompozycja wraca do PLANU B: rów. 5 + bootstrap na SDXL.
AKTUALIZACJA 2026-08-19: decyzją użytkownika ramki kontynuowane mechanizmem attention-level → 1j.

## 1j. GROUNDING W UWADZE (GLIGEN-style) — WERDYKT WSTRZYMANY (2026-08-19, test A/B w toku)

Po decyzji użytkownika „na pewno robimy ramki" mechanizm przeniesiony z wag do uwagi:
`e = ground_head(klucz ⊕ Fourier(ramka))` (MLP 192→256→768, 246.8k parametrów), per attn2
człon addytywny `out_i += tanh(gate_l)·σ(⟨q_i, K·e⟩·scale)·V·e` z zamrożonymi K/V (pod
`no_lora()` — LoRA zostaje czystym nośnikiem treści) i 16 gate'ami zero-init (start bitowo
= F_base). Razem 21.37M (+1.17%). Trening: wklejka (`box_aug_p` 0.5) + strata maskowana
+ `set_ground(k, ramka)`; ścieżki sampler/eval/sonda wpięte.

**Historia debugowania (obie usterki złapane przez smoke+strażnika, zanim spaliły pełny trening):**
1. shape-bug: `e` [1,768] vs `head_to_batch_dim` (wymaga osi sekwencji) — crash; fix `unsqueeze`,
   test przez PRAWDZIWY blok `Attention` diffusers (lekcja: unit test managera nie wystarczy);
2. **martwy punkt podwójnego zero-initu**: grad(gate) ∝ V·e=0 i grad(head) ∝ tanh(gate)=0 —
   obie strony iloczynu czekają na siebie; gate'y zostały dokładnie 0. Fix w stylu GLIGEN:
   z zera startuje TYLKO gate (head ×0.1) — bitowa równość z bazą zachowana, gradient płynie.

**Pełny trening przeszedł** (gate'y max 0.048 / mediana 0.021, head[-1] norma 4.7), sonda
połówkowa: **41/84 = 48.8%** — poziom losowy. UWAGA: zbieżność z 48.8% wersji wagowej to
przypadek sum (liczniki per koncept różne: 23→24/48, 35→36/72; inne checkpointy).

**TEST A/B ROZSTRZYGNĄŁ (2026-08-19): mechanizm STRZELA, ale pozycja NIE steruje.**
Ten sam seed+checkpoint, ramka lewo/prawo/bez: |L−OFF| = 0.042 (człon aktywny przy
inferencji — okablowanie czyste, procesory zainstalowane, wektor liczony), ale
|L−R| = 0.0054 (8× mniej) — treść `e` jest niemal niezmiennicza na ramkę; wizualnie ten
sam wyśrodkowany pies we wszystkich trzech wariantach. Wniosek: head nauczył się używać
`e` jako globalnego wzmocnienia treści (pomaga rekonstruować wklejkę GDZIEKOLWIEK),
ignorując część Fourier(box). Pozostałe przyczyny (obie mogą współistnieć): pozycja
nie żyje w `q_i` (uwaga SD bez jawnego kodowania pozycji) + sygnał treningowy nie
wymusza ramki (obiekt widoczny w z_t; strata maskowana nie karze malowania poza ramką).
To TRZYNASTE odrzucenie — tym razem z czystą diagnozą mechanistyczną, nie bugiem.

**DOMKNIĘCIE POTRÓJNE (2026-08-19 po południu):** obie hipotezy ratunkowe zmierzone i obalone
na tym samym protokole (sonda 84 próbki + A/B lewo/prawo/bez):

| run | sonda | \|L−R\| | \|L−OFF\| | gate max/med | uwagi |
|---|---|---|---|---|---|
| run 1 (400 kr., p=.5) | 48.8% | 0.0054 | 0.042 | .048/.021 | — |
| long (1200 kr., p=.8, ~5× ramek) | 47.6% | 0.0024 | 0.020 | .065/.043 | parametry UROSŁY, wpływ SPADŁ |
| geo (jawna geometria) | 51.2% | 0.0017 | 0.0072 | .049/.023 | pos/box_proj ZOSTAŁY NA INICIE (norma ~4.6) |

Kluczowe odkrycie: projekcje geometryczne w geo nie dostały żadnego sygnału (normy = init),
a w long parametry rosły, gdy efektywny wpływ członu malał. Wniosek mechanistyczny (3×
zmierzony): **objective wklejka+strata-maskowana-do-ramki nie generuje presji gradientu na
użycie ramki** — z_t zdradza pozycję przy niemal każdym t, strata nie karze treści poza ramką,
a przy najwyższym szumie łatwiejszym wygranym jest treść. Adresowanie podane za darmo leży
nieużyte. To zamyka CAŁĄ rodzinę lekkich bramek na tym objective — niezależnie od architektury
członu.

**DOMKNIĘCIE PIĄTE I SZÓSTE (2026-08-19 wieczór): objective v2 (segmentowana wklejka na
naturalne tła, strata bez maski) + maska analityczna.** SEG/v2 (uczona geometria): sonda 50.0%,
|L−R|=0.0027, pos_proj ruszył o 1.6% po 10 taskach (za mało). AM (maska analityczna, 2 skalary):
sonda 47.6%, ale **|L−R|=0.0288 ≈ |L−OFF|=0.0305 — 17× skok**: zastrzyk jest w pełni zależny
od ramki przy inferencji. Kluczowy pomiar przestrzenny: różnice L↔OFF NIE koncentrują się
w połowie ramki (L zmienia prawą połowę BARDZIEJ niż lewą) — lokalny zastrzyk wczesnie w
trajektorii rozlewa się globalnie (przestawia wariant sceny), nie nukleuje obiektu w ramce.
Diagnoza końcowa rodziny: adresowanie działa (analityczne), treść działa (|L−OFF|), ale
zastrzyk o amplitudzie tanh(gate)·σ·V·e nie jest w stanie ZASIAĆ podmiotu przeciw priorowi
kompozycyjnemu SD — trajektoria traktuje go jak szum warunków, nie jak rozkaz układu.
Sześć wariantów, sześć czystych negatywów: wagi / qKe / +sygnał / uczona geo / objective v2 /
maska analityczna.

**PRZEŁOM — SIÓDMY WARIANT: GO (2026-08-20). GSA hipersieciowa przeszła pre-rejestrowany próg:
sonda 71/84 = 84.5% (próg 80%), |L−R| = 0.179 (66× wersja wagowa), |L−OFF| = 0.219.**
Konstrukcja: 4 tokeny groundingu z hipersieci (klucz ⊕ Fourier(ramka)) + FiLM z klucza
(norma 0→1.09 po treningu — hipersieć realnie moduluje czytanie) + wspólna wąska uwaga
czytająca per attn2 (proj. 64: q z obrazu, k/v z tokenów, out; ~1.2M wspólnych parametrów)
× analityczna maska inside × tanh(gate) + przeważenie t∈[T/2,T) na krokach z wklejką
+ objective v2 (segmentowana wklejka na naturalne tła, strata bez maski). Co było kluczowe
wobec 6 porażek: STRUKTURA zastrzyku (każda pozycja czyta z tokenów wg własnego stanu,
zamiast dostawać skalar·stały wektor) — gate'y zostały małe (med 0.021), a mimo to placement
działa; amplituda nigdy nie była wąskim gardłem, pojemność odczytu była.
Gałka κ na gate'ach przy inferencji (tylko przy ramce): połówki 84.5%→**94.0% @κ=2**, ćwiartki (przypadek 25%) 40.5%→**59.5% @κ=2** (κ=4 artefakty, κ=8 rozpad — odwrócone U). Pozostało do domknięcia werdyktu: kryterium 2 pre-rejestracji — metryki pełnokadrowe @50
(ev_gsa) w szumie względem bazy; potem 2. seed. Baza porównania (P_ground bez GSA @50):
s07 TA 0.761/IA 0.792/DINO 0.631 ≈ F_base (grounding v1 nie kosztował metryk). Drogi pozostałe przy "ramki must-have": (a) pełna warstwa GSA à la GLIGEN
trenowana u nas (skok pojemności zastrzyku; ryzyko danych 4-7 obr./koncept), (b) rów.5+bootstrap
dla pojedynczego konceptu (JEDYNY mechanizm zmierzony jako działający na SD-1.5 w tym projekcie;
zero treningu, 2 przebiegi; hipersieć = tożsamość, protokół = placement). Drogi dalej: (A) zmiana OBJECTIVE — wklejka na naturalne tła + strata BEZ maski
(nadzór "poza ramką = tło" czyni ramkę informacyjnie konieczną przy wysokim szumie; syntetyczne
dane grounded à la GLIGEN z naszych 30 obrazków), (B) gotowe wagi GSA GLIGEN + ReGround,
(C) zamknięcie wątku potrójnym wynikiem negatywnym (materiał analityczny: skala personalizacji
nie uczy placementu przez denoising loss). Diagnostyka wandb dla kolejnych prób gotowa
(diag_bbox z rysowaną ramką, krzywe gate'ów, mapa σ(geo) — smoke zaliczony).

**Trop z literatury (LayoutDiffusion, CVPR 2023, arXiv 2303.17189):** uwaga SD nie ma jawnej
pozycji — `q_i` zna swoje miejsce tylko z przecieków konwolucji. Ich Object-aware Cross-Attention
daje OBU stronom afiniczności jawne embeddingi pozycyjne (patch = mini-obiekt z bboxem z siatki).
Kandydat na poprawkę: `logit_i = ⟨q_i,K·e⟩ + ⟨P·fourier(pos_i), R·fourier(box)⟩` — „gdzie"
z geometrii (dokładne, darmowe), treść nadal z hipersieci. To NIE jest powrót do obalonego
training-free maskowania (tam: ograniczanie istniejącej tożsamości bez treningu; tu: trenowany
content-injection z dokładnym adresem). Do decyzji użytkownika, razem z ew. karą za koncept
poza ramką w stracie. Drugi trop (ReGround, ECCV 2024, **arXiv 2403.13589** — sprostowane 2026-08-31, wcześniej
błędnie „03388”; tytuł: *ReGround: Improving Textual and Spatial Grounding at No Cost*, Lee & Sung): nasza topologia RÓWNOLEGŁA
jest zgodna z ich wnioskiem (szeregowa GSA GLIGEN-a tłumi tekst); plan awaryjny = gotowe
wagi GSA GLIGEN-a (placement wytrenowany na dużych danych) + hipernet podaje treść tokenu;
ostrzeżenie: podbijanie amplitudy groundingu ma znany tryb awarii (zjada TA jak GLIGEN γ=1).

## 1l. POLEROWANIE GSA NA HELIOSIE (2026-08-31) — pełne liczby

Wszystko na `outputs/phaseP/P_ground_gsa/hyper.pt`, ziarna 31337+i, 30 kroków, `lora_scale`
0.7, ćwiartki (ramka = 25% kadru), 84 generacje na punkt. Instrument: `scripts/_ground_iou.py`
(Mask R-CNN R50-FPN-v2 COCO; wybór detekcji przez podobieństwo DINO do referencji, nie przez
score, bo dla kaczki COCO strzela `dining table`/`vase`).

### Numeryka Heliosa wobec Ateny — ZGODNA
| pomiar | Athena | Helios | job |
|---|---|---|---|
| ćwiartki @κ=2/s=0.3 per koncept | 42/83/83/75/42/58/58 | **identyczne** | 21594432 vs 3043021 |
| DINO per koncept @κ=2/s=0.3 | 0.8417…0.4607 | Δ ≤ 0.0015 | jw. |
| sonda połówkowa κ=1 | 71/84 = 84.5% | 73/84 = 86.9% | 21590216 |
| sonda połówkowa κ=2 | 79/84 = 94.0% | 77/84 = 91.7% | 21590217 |
Rozjazd w sondzie połówkowej siedzi prawie cały na kaczce (κ=2: 11/12 vs 8/12) — koncept
o najniższym DINO wobec referencji, więc argmax po połówkach jest tam najchybotliwszy.
GH200 ~3× szybszy od A100 na tym kodzie (84 generacje @30 kroków: 5 min wall).

### Punkty pracy i gałki (RAZEM po 7 konceptach)
| konfiguracja | ćwiartki | IoU | IoU>0.5 | zawarcie | wypełn. | DINO | kolor dRGB | det | job |
|---|---|---|---|---|---|---|---|---|---|
| κ=2, s=0.3 | 63% | 0.375 | 23% | 0.49 | 2.19 | 0.7270 | **0.137** | 84/84 | 21598893 |
| κ=4, s=0.15 | 79% | 0.541 | **62%** | 0.67 | 1.60 | 0.7001 | 0.180 | 84/84 | 21597878 |
| κ=4, s=0.15, `gain_res 64:0` | 74% | 0.495 | 46% | 0.61 | 1.89 | 0.6969 | 0.173 | 84/84 | 21597884 |
| κ=4, s=0.15, confine 3 (span) | 80% | 0.548 | 64% | 0.69 | 1.58 | 0.6967 | 0.178 | 84/84 | 21597878 |
| κ=4, s=0.15, confine 6 (span) | 81% | 0.549 | 64% | 0.69 | 1.64 | 0.6964 | 0.173 | 83/84 | 21597878 |

**Per koncept @κ=2/s=0.3** (stary detektor Faster R-CNN, job 21594432 — dlatego nie miesza się
z tabelą wyżej): dog 0.273/0%/0.31/2.38 · duck 0.482/33%/0.58/1.89 · cat 0.427/33%/0.48/2.13 ·
backpack 0.349/25%/0.46/1.63 · teddy 0.289/8%/0.31/2.52 · dog2 0.403/25%/0.43/2.28 ·
cat2 0.400/17%/0.41/2.45 (IoU / IoU>0.5 / zawarcie / wypełnienie).

### Werdykty
1. **Metryka ćwiartek zawyża placement.** Mierzy argmax podobieństwa po ćwiartkach, czyli GDZIE
   koncept jest najwyrazistszy, a nie czy MIESCI SIE w ramce. @κ=2 kaczka: 83% ćwiartek vs 33%
   IoU>0.5. Do artykułu obie liczby z nazwanym rozróżnieniem.
2. **Harmonogram + κ robią dla zawierania więcej, niż widziała stara metryka**: (κ=2,s=0.3) →
   (κ=4,s=0.15) to IoU>0.5 23% → 62% i wypełnienie 2.19 → 1.60, za −0.027 DINO i +0.043 dRGB.
3. **`gain_res` (κ per rozdzielczość) — ODRZUCONE.** Wyłączenie wstrzyku na mapach 64² zabiera
   16 punktów IoU>0.5, a kolor poprawia o 0.007 (w szumie). Najdrobniejsze warstwy attn2
   współpracują w układzie, a dryf koloru nie pochodzi ze wstrzyku.
4. **`confine` na spanie konceptu — NO-OP.** Kara 0→3→6 daje IoU>0.5 62→64→64% (dwie próbki
   z 84), wypełnienie bez zmian. Zgodne z przyczynowością CLIP-a (audit 2885915): kara na
   pozycjach `dog` zostawia EOS i padding niosące całe zdanie. Test wyjaśnienia: wariant
   `cummax` (kara od pierwszego tokenu konceptu do końca sekwencji), job 21602457.
5. **Prior skali z referencji** (hipoteza, kontrola w skrypcie): porządek wypełnienia idzie po
   sposobie fotografowania konceptu — zbliżenia zwierząt i maskotek 2.1–2.5, obiekty w scenie
   duck 1.89 i backpack 1.63. Przy `box_aug_p 0.5` połowa kroków treningu to gołe zbliżenia bez
   ramki, które uczą „ten koncept wypełnia kadr". Dźwignia: `box_aug_p` 0.8, sam config.
6. **κ-aware trening nie ma sensu jako kalibracja punktu pracy**: uczy się iloczyn
   `gain·tanh(gate)`, więc stałe κ w treningu to reparametryzacja. Sens ma tylko κ **losowane**,
   jako regularyzacja odporności na skalę.

### Przeciek atrybutu w captionach treningowych — potwierdzony
Captiony CIFC: `yellow rubber duck toy sitting on a gravel surface` (4/4),
`red backpack sitting on a rock in the woods` (6/6), `fluffy cat ...` (1/5), reszta bez
przymiotników obiektu. `P_ground_gsa.yaml` nie ustawia `attr_strip` → kolor wchodził promptem;
`token_mask_lora: true` dodatkowo ogranicza deltę do pozycji `class_word`, więc tokeny
`yellow`/`rubber` szły przez zamrożone K/V. `gen_cifc.py` dokłada `eval_prefix`, więc artefakt
był niewidoczny, dopóki sonda nie promptowała goło. Trening bez atrybutów: job 21592961
(`P_ground_gsa_nocap`, seed 2024, sparowany z bazą), ewaluacja 21601163 (`afterok`).
Od `e442b47` każdy trening drukuje captiony per task.

## 1k. SDXL — KOLUMNA DOMKNIĘTA (2026-08-19 rano)

- **Macierz forgettingu kompletna** (rescue): **DINO +0.0012, CLIP-I +0.0019** — ta sama
  historia co SD-1.5 (0.0015). Grid wizualny: `assets/figures/fgt_grid_F_sdxl.jpg` (55/55
  komórek, wiersze stabilne).
- **Headline @50, 3 seedy, s0.7: TA 0.7921±0.0091 | IA 0.7988±0.0050 | DINO 0.6262±0.0076**
  — parytet z opublikowanym CIDM-SDXL (0.795/0.800) na ich protokole.
- **Liga CL @10 (nasz harness), 4/5 metod:** najlepszy baseline LwF s07: TA 0.816/IA 0.750/
  DINO 0.519; EWC 0.483, finetune 0.486, C-LoRA 0.451 — nasza przewaga **+0.115 DINO,
  +0.056 IA** nad najlepszym (jeszcze przed odczytem matched-TA; baseline'y siedzą na wyższym
  TA). **L2DM ZABLOKOWANY: OOM** na A100 40GB (ECD+TAME @1024px) — do decyzji: gradient
  checkpointing / batch 1 + akumulacja / liga 4-metodowa.

## 1h. WIERSZ FLAGOWY ROZSTRZYGNIĘTY (2026-08-15): prefiksy POGARSZAJĄ CIDM

Obawa z 1g (że +0.019 IA to prefiks) odwróciła się: CIDM ewaluowany w naszym protokole promptów
(prefiksy = język ICH captionów treningowych) jest GORSZY niż w swoim natywnym evalu:
@α=0.8: TA 0.743/IA 0.775 (prefiks) vs 0.753/0.779 (natywnie). Ich uczone tokeny najwyraźniej
kolidują z jawnym prefiksem w prompcie.

**FLAGOWE, protokół w pełni dopasowany (matched-TA 0.748, 3 seedy vs CIDM-prefixed):**
**+0.036 IA, +0.065 DINO.** (Wobec natywnego: +0.020/+0.042 — raportujemy oba wiersze:
natywny jako walidację reprodukcji, prefiksowany jako uczciwe porównanie.)

## 1f. NOCNE ROZSTRZYGNIĘCIA (2026-08-13 rano)

**SDXL — próba przebicia ich 0.795 IA: PARYTET, bez przebicia.** F_sdxl 3 seedy @TA 0.800:
**IA 0.7916 ± 0.0022** (−0.003 ≈ 1.5σ od ich liczby). Wszystkie dźwignie zmierzone i odrzucone:
h100/h150 (nadadaptacja), st600 (−0.015), h100_st600 (TA zapadnięte do szpilki 0.005),
lr5e5 (≈baza), st200 (gorzej), **nomask −0.018 — ODWRÓCENIE względem SD-1.5 (+0.019)**:
lekcje maskowania nie przenoszą się między backbone'ami (wynik do artykułu).
Ostatnia niezmierzona dźwignia: port R_tail (ale patrz niżej — jego status osłabł).

**R_tail, trzeci seed sparowany:** @0.762: +0.005 / +0.015 / +0.005 (seedy 2024/25/26) —
**średnia +0.008, znak 3/3 dodatni, PONIŻEJ progu 0.0091**. Etykieta: konsekwentnie dodatni,
niepotwierdzony. Zostaje w ablacjach, nie w metodzie.

**L2DM — finalny wiersz ligi: α=0.1 (DINO 0.480 @0.8, TA 0.787, IA 0.739).** Combo α0.1+γ10
gorsze od singli (0.465) — poprawki się nie sumują. Nasz L2DM na DINO mocniejszy niż w ich
publikacji; na IA wciąż poniżej ich 0.761 (offset protokołu jak wszędzie).

**Ablacja bloków SDXL domknięta:** profil sił delt płaski (1.6×), LOO najsłabszego-największego
bloku (down_blocks.2, ~47M) kosztuje ~−0.009 IA/DINO przy matched-TA — **nie ma darmowego bloku**;
164M hipersieci SDXL jest używane w całości.

## 2. Rozstrzygnięte — odrzucone

Każde z pomiarem, nie z przeczucia.

| kierunek | werdykt |
|---|---|
| współdzielone heady (Faza C) | najlepszy wariant −0.0115 DINO przy 9.18 M (2.1× CIDM) |
| faktoryzacja wyjścia `x_L = U·z` (Faza F) | −0.012 (q=128) … −0.057 (q=32), **bez kolana** przy zmierzonym rzędzie 40 |
| baza `U` losowa zamiast uczonej | −0.064 — baza musi być uczona |
| zamrożona baza wejściowa z korpusu tekstowego | 53–83% błędu rekonstrukcji na promptach held-out |
| parametryzacja w bazie własnej wagi (SVDiff) | 0.26% wyrażalności wobec 0.156% dla macierzy losowej |
| routing treścią per token (przez kontekstowe zanurzenia) | odrzucone sondą bez treningu: 'V1 dog' vs 'V7 dog' cos 0.909 (gorzej niż klucze, które zapadły metodę); w kompozycji drugi span RÓWNOODLEGŁY od obu wzorców (0.778 vs 0.779) i poza dziedziną treningową |
| routing treścią z ORTOGONALNYMI tokenami w słowniku | separacja pojedynczego konceptu rośnie 1.48×→2.94×, ale kompozycja nadal martwa (drugi span: margines +0.008) — przyczynowe mieszanie kontekstu nieusuwalne z jednego przebiegu encodera; kompozycja wymaga kodowania per koncept |

Dwa ostatnie odrzucone **bez trenowania czegokolwiek** — pomiarem strukturalnym.

### Faza K/R — klucze semantyczne i identyfikator w prompcie (2026-08-09)

| wariant | seed 2024 | seed 2025 | **średnia** | werdykt |
|---|---|---|---|---|
| `K_sem64` (klucz: 64 semantyki z obrazów + 128 instancji) | +0.0125 | **−0.0070** | **+0.0028** | **OBALONE drugim seedem** |
| `K_sem64_lv` (klucz semantyczny + uczone `V_t`) | +0.0119 | — | — | jeden seed, po obaleniu bazy bez znaczenia |
| `K_sem128` | +0.0110 | — | — | j.w. |
| `K_latent` (stan generacji: mean/std latentu) | −0.0017 | — | — | martwy, zgodnie z przewidywaniem (latent ≈ krok czasowy) |
| `R_id` (identyfikator „V1" jako string w prompcie) | −0.0161 | **−0.0147** | **−0.0154** | **SZKODZI, potwierdzone dwoma seedami** |
| `R_orth` (`<V1>` jako token słownika, wiersz = klucz ortogonalny) | +0.0019 | — | ~0 | **NEUTRALNE** — i to jest jego wartość |

**Powtórzenie historii bramki.** `K_sem64` wyglądał na pierwszy mechanizm nad progiem szumu
(+0.012 w obu punktach odczytu, dwa warianty zgodne co do kierunku). Drugi seed zmienił znak.
Średnia +0.0028 przy rozrzucie międzyseedowym 0.0091 — **niepotwierdzone**.

**ALE: struktura JEST indukowalna** (pomiar na 16 warstwach, seed 2024):

| | tło (pary losowe) | dog/dog2 | cat/cat2 | **separacja** |
|---|---|---|---|---|
| `F_base` (klucze arbitralne) | +0.0310 | +0.0640 | +0.0476 | **+0.0248** |
| `K_sem64` | +0.0317 | +0.0887 | +0.0466 | **+0.0359** |
| `K_sem128` | +0.0514 | +0.0917 | +0.1340 | **+0.0614** |

Klucze semantyczne **przenoszą strukturę klasową na adaptery** — separacja rośnie 2.5×,
monotonicznie z `sem_dim`. To odpowiada na pytanie z sekcji 4 („brak struktury między
zadaniami"): struktura nie jest niemożliwa, jest **nieindukowana przez funkcję celu**.
Wymuszona z zewnątrz pojawia się — i **nie poprawia metryk**. To jest wynik do artykułu:
kompresja między zadaniami nie jest zablokowana architekturą, tylko nieopłacalna dla straty
rekonstrukcyjnej.

**Krok (i) vs (ii) — identyfikator kosztuje, ortogonalny token nie.** String „V1" tokenizuje się
na śmieciowe podtokeny i kosztuje −0.015 (oba seedy zgodne); jeden czysty token słownika
z wierszem = klucz ortogonalny wraca do parytetu (+0.002). Różnica (ii)−(i) ≈ **+0.018**.
Znaczenie: **interfejs z identyfikatorem w prompcie jest dostępny za darmo** — co odblokowuje
routing per token i kompozycję — ale sam w sobie nie jest zyskiem na metrykach.


## 3. Rozstrzygnięte — przyjęte

| zmiana | efekt |
|---|---|
| `head_hidden` 100 → 50 + klucze ortonormalne 128-d | 50.2 M → **21.1 M**, jakość bez zmian (2 punkty matched-TA) |
| rank 8 → 4 | bez zmian (+0.003 = szum), połowa LoRA na zadanie |
| ~~`nonorm`~~ | **+0.0007 na obecnej bazie — NIEPOTWIERDZONE, patrz niżej** |
| `token_mask` OFF | +0.019 DINO — **niewykorzystane**, maska zostaje ze względu na kompozycję |

---

## 4. Wyniki analityczne (materiał do artykułu)

**Teoria nasycenia.** `rank(X_b − β1ᵀ) ≤ h`, obciążenie headu = `T·|b|`. Zmierzony ogon widma:
6 kubełków @h100 → 3–6% dla największych, rozdzielenie po roli → 0%. Trafnie przewidziała, że
rozdzielenie kubełków bije poszerzanie przy tym samym koszcie (−0.016 vs −0.035) oraz że `code64`
szkodzi. **Nie przewidziała** wyniku przy stałym budżecie 4.6 M — sufit rzędu tłumaczy ~połowę zjawiska.

**Brak struktury między zadaniami**, potwierdzony czterema niezależnymi pomiarami:
- efektywny rząd rośnie o `|b|` na zadanie (97% pełnego) — brak kompresji między zadaniami
- cos delt wag: **+0.011** średnio, `dog/dog2` **+0.017** — pary tej samej klasy **nieodróżnialne** od losowych
- składowa wspólna: 12% energii
- widmo rodziny adapterów prawie płaskie (0.190 … 0.059 przy równomiernym 0.111)

**Encoder CLIP jest przyczynowy** — zanurzenie tokenu konceptu zależy **wyłącznie od prefiksu**:

| | różnica zanurzenia |
|---|---|
| ten sam prefiks, inne zakończenie promptu | **0.0000%** |
| inny prefiks | 28–67% |

Tłumaczy to trzy wcześniejsze pomiary: dlaczego maska kosztuje 0.019 DINO (strona tekstowa
sprowadza się do stałego offsetu), dlaczego modulacja musi czytać `pooled` a nie token konceptu,
i dlaczego captiony treningowe rozpinają tylko 2 wymiary wobec 11 dla promptów ewaluacyjnych.

**Koszt wdrożenia CIDM zweryfikowany w ich kodzie:** LoRA rank 4 na attn2 (identyczny hookpoint co
nasz) + **48 embeddingów per koncept** (3 tokeny × 16 warstw cross-attention) = 0.435 M/zadanie.
Text-encoder LoRA w ich configu to martwy kod — encoder jest zamrożony.

---

## 5. Osie warunkowania — werdykt

Odczyt matched-TA przy TA = 0.770 (`F_base`: DINO 0.6071 / IA 0.7877).

> **UWAGA: ponizsza tabela to WYLACZNIE seed 2024.** Bramka, ktora wyglada tu najlepiej, zostala
> obalona drugim seedem (srednia −0.0019, patrz nizej). `T_time` nie ma drugiego seeda w ogole,
> wiec jego +0.0017 nalezy czytac jako **niepotwierdzone**, nie jako zysk.

| wariant | oś | skal | @0.762 | @0.770 | @0.775 | **szer. TA** |
|---|---|---|---|---|---|---|
| ~~P_gate05~~ | bramka skalarna, 769 par. | 4 | +0.0037 | +0.0029 | +0.0023 | 0.083 |  ← **OBALONE, seed 2025 daje −0.007** |
| T_time | krok czasowy | 4 | +0.0025 | +0.0017 | +0.0012 | 0.074 |  ← **jeden seed, niepotwierdzone** |
| P_gate025 | bramka, słabsza | 4 | +0.0024 | −0.0001 | −0.0016 | 0.072 |
| T_base_s800 | 800 kroków (kontrola) | 4 | −0.0036 | −0.0022 | — | 0.081 |
| T_time_s800 | czas + 800 kroków | 3 | −0.0038 | −0.0043 | — | 0.047 |
| P_split64 | modulacja w bloku rozdzielonym | 4 | −0.0028 | −0.0042 | −0.0050 | 0.073 |
| P_mod1 | pełna modulacja wektorowa, 57.5 k | 4 | −0.0101 | −0.0103 | −0.0104 | 0.079 |
| S_kappa1e2 | skala, kara silna | 4 | −0.0816 | — | — | **0.007** |
| S_kappa1e3 / 1e4 | skala, kara słabsza | 4 | — | — | — | **0.000 / 0.002** |
| T_time_scale | czas + skala | 4 | — | — | — | **0.000** |

Odniesienie `F_base`: szerokość zakresu TA **0.082**.

**Żadna oś nie przesuwa krzywej ponad szum.** Dwa warianty wygladaly dodatnio na seedzie 2024 —
bramka zostala obalona drugim seedem, a czas drugiego seeda nie ma. Po lekcji z bramki (efekt tej
samej wielkosci zmienil znak) `T_time` nalezy traktowac jako niepotwierdzony.

**Modulacja wektorowa szkodzi** — przeuczenie na 52 unikalnych promptach przy 57.5 k parametrów,
potwierdzone podręcznikową sygnaturą: **niższy** loss treningowy, **gorszy** eval. Zamknięcie jej
w bloku ortogonalnym do kluczy zmniejsza szkodę trzykrotnie (−0.010 → −0.003), czyli straty brały
się z psucia kanału tożsamości, nie z samej modulacji.

**Warunkowanie skalą zawiodło w nieoczekiwany sposób** — nie przez gorsze metryki, tylko przez
**unieruchomienie gałki**: szerokość zakresu TA spadła z 0.082 do 0.000–0.004, więc krzywa
kompromisu zapadła się do punktu i matched-TA przestaje być liczalny. `κ = 1e−2` daje najszerszy
zakres (0.004), ale kosztuje +0.104 lossu — użytecznego okna nie ma.

**Dłuższy trening szkodzi** (−0.0022 przy 800 krokach), spójnie z rundą 1, gdzie 600 kroków dawało
−0.010. Warunkowanie czasem zyskuje z budżetu odrobinę więcej niż baza (−0.0195 vs −0.0167 lossu),
ale obie tracą na metrykach.

**BRAMKA OBALONA DRUGIM SEEDEM.** Pełny matched-TA, cztery punkty odczytu:

| TA | seed 2024 | seed 2025 | **średnia** |
|---|---|---|---|
| 0.765 | +0.0034 | −0.0078 | **−0.0022** |
| 0.770 | +0.0029 | −0.0069 | **−0.0020** |
| 0.775 | +0.0024 | −0.0060 | **−0.0018** |
| 0.780 | +0.0019 | −0.0051 | **−0.0016** |

Znak zmienia się w każdym punkcie, średnia ujemna wszędzie. `P_gate075` daje +0.0075…+0.0017, ale
wyłącznie na seedzie 2024 — tym, który sprzyja bramce przypadkiem.

**Bilans wszystkich prób mechanistycznych:**

| próba | wynik |
|---|---|
| współdzielone heady | −0.0115 |
| faktoryzacja wyjścia | −0.012 … −0.057 |
| zamrożona baza tekstowa | odrzucona bez treningu (53–83% błędu) |
| baza własna wagi (SVDiff) | odrzucona bez treningu (0.26% wyrażalności) |
| modulacja per prompt | −0.010 (pełna) / −0.003 (blok rozdzielony) |
| warunkowanie skalą | unieruchomiło gałkę (szerokość TA 0.000–0.007) |
| warunkowanie czasem | +0.002 na jednym seedzie — **niezweryfikowane, po lekcji z bramki domyslnie szum** |
| bramka skalarna | +0.002 / −0.004 — **szum** |

Siedem prób, żadna nie działa. Jedyne, co dało efekt, to **uproszczenia**: `nomask` +0.019,
`nonorm` +0.008, `h50` + klucze ortonormalne (2.4× mniejszy model bez straty jakości).

### Kalibracja szumu — dotyczy wstecznie wszystkich jednoseedowych wyników

Pełny matched-TA dla bramki na dwóch seedach: **+0.0029 (2024)** wobec **−0.0069 (2025)** przy
TA = 0.770. Rozrzut **między seedami** dla tej samej różnicy konfiguracji wynosi więc **0.010** —
trzykrotnie więcej niż 0.003, którego używałem jako progu (szacowanego z wahań wewnątrz runu).

Konsekwencja: każdy jednoseedowy wynik poniżej ~0.010 jest niepotwierdzony.

| zmiana | efekt | status wobec rozrzutu międzyseedowego |
|---|---|---|
| `nomask` | +0.019 | bezpiecznie powyżej |
| `nonorm` | +0.008 | **na granicy — wymaga drugiego seeda** |
| rank 8 → 4 | +0.003 | poniżej progu; wniosek „bez różnicy" utrzymany |
| bramka | +0.003 / −0.007 | **obalona** |
| `nonorm` | +0.0052…+0.0022 (s2024) / ≈0.0000 (s2025), średnia **+0.0019** | **zbędne na obecnej bazie** |

### `nonorm` stał się zbędny po zmianie kluczy

Zweryfikowane parami po seedach na aktualnej bazie (`h50` + klucze ortonormalne 128), **komplet
czterech skal, pięć punktów odczytu**: seed 2024 daje +0.0052…+0.0022, seed 2025 daje ≈0.0000,
średnio **+0.0019** przy rozrzucie międzyseedowym **0.0036**. Efekt jest połową rozrzutu, czyli
nieustalony. Stary pomiar dawał +0.008, ale pochodził z bazy `h100` + klucze CLIP.

Wyjaśnienie jest mechanistyczne i wynika z naszego wcześniejszego pomiaru: `preserve_norm`
przeskalowuje resztę po Gram-Schmidcie do `‖h‖`. Klucze CLIP mają wzajemne cos 0.77, więc GS
zabierał im **ponad połowę normy** i przeskalowanie było istotną interwencją. Klucze ortonormalne są
ortogonalne z konstrukcji, więc GS nie ma czego usuwać, a `preserve_norm` nie ma czego
przeskalowywać. **Dwa „usprawnienia" okazały się tym samym usprawnieniem** — klucze ortonormalne
dostarczają tego, co `nonorm` kompensował.

Rozrzut międzyseedowy wynosi tu 0.0036 wobec 0.0091 dla bramki — co samo w sobie jest korektą:
**poziom szumu nie jest uniwersalną stałą**, zależy od porównania. Bramka wnosiła własną wariancję
(uczona sieć modulująca klucz), `nonorm` jest zmianą deterministyczną. Progu 0.010 nie należy więc
stosować globalnie; trzeba go szacować per porównanie, z pary seedów. `nonorm` zostaje w `F_base`, bo nie szkodzi, ale **nie może
być raportowany jako wkład**.

`P_gate075` daje +0.0055, najwięcej z całej serii — ale pochodzi z seeda 2024, czyli tego, który
sprzyjał bramce przypadkiem. Bez drugiego seeda ta liczba nic nie znaczy.

## 6. Baseline'y

Sweep λ **domknięty** — każda metoda ma maksimum wewnątrz siatki, żadna nie wybrana z brzegu:

| metoda | wybrane λ | TA | IA | DINO |
|---|---|---|---|---|
| EWC | 1e5 (1e6, 1e7 spadają) | 0.7720 | 0.7588 | 0.5323 |
| LwF | 10 | 0.7684 | 0.7798 | 0.5625 |
| C-LoRA | 10 (płasko 0.1–100, spada od 1e3) | 0.7522 | 0.7635 | 0.5178 |

**L2DM jest zepsuty** i nie wolno go tak raportować: przy skali treningowej (s = 1.0) DINO = 0.125,
a krzywa jest odwrócona (tożsamość spada ze wzrostem skali). Przyczyna zdiagnozowana — człon ECD
wynosi 0.0057 przy TAME 0.2375, czyli destylacja jest ~40× słabsza od replayu przy domyślnym
`α = β = γ = 1`. Wymaga sweepu, którego praca L2DM nie specyfikuje.

---

## 7. Otwarte kwestie

**Zarzut egzystencjalny.** Warunkowanie zdegenerowało się do indeksu zadania, adaptery są wzajemnie
ortogonalne, kompresja niemożliwa, wypiekanie bezstratne — metoda jest funkcjonalnie **zbiorem
dziesięciu LoRA wybieranych po indeksie**. Przy dziesięciu zadaniach każda hipersieć musi się tak
zdegenerować: z dziesięciu punktów nie da się nauczyć funkcji, można je tylko zapamiętać.
Brakuje baseline'u „niezależna LoRA per koncept z oracle'owym indeksem" — jedyna niesprawdzona
hipoteza na jego korzyść to **regularyzacja w reżimie 3–7 obrazów na koncept**.

**Kierunek nietknięty przez żadne z pięciu odrzuceń:** warunkowanie semantyczne (nazwa klasy albo
zanurzenie obrazów), klucz dwublokowy, Gram-Schmidt **tylko na bloku instancji**. Wszystkie pomiary
braku struktury dotyczyły modelu, w którym nic tej struktury nie zachęcało. Kryterium zarejestrowane
z góry: separacja cosinusa par tej samej klasy wobec dzisiejszych +0.017 przy tle +0.011.

**Decyzja do podjęcia — maska.** Broniła się dwoma argumentami i oba osłabły: oszczędność 23%
odpadła razem z bazą wejściową, a przyczynowość encodera pokazała, że maskowanie strukturalnie
sprowadza stronę tekstową do stałego offsetu. Zostaje wyłącznie obietnica kompozycji, wciąż
niezmierzona. Koszt: 0.012–0.019 DINO.

---

## 8. Do zrobienia z planu Tier-1

- sweep α/β/γ dla L2DM (bez tego baseline jest chochołem)
- sweep `h` (12/15/20/35) — rozstrzyga, czy zapas zależy od `h/T`, czy od `h − T`; potrzebne do 35 konceptów
- CIDM re-run w naszym harnessie (Faza 3b) — jedyny wiersz apples-to-apples
- 6 metod × 3 seedy @50, krzywe metod
- ~~SDXL 1024~~ → DOMKNIĘTE (1k): macierz + @50×3 seedy + liga 4/5 (L2DM: OOM, do decyzji)
- grounding: test A/B → werdykt → ew. wariant geometryczny LayoutDiffusion (1j)
- benchmark kompozycyjny (protokół CCDM: kadrowanie po bboxach, 20 kombinacji × 10 promptów, 25 próbek)

## 9. Uwagi operacyjne

- konto GPU wybierać przez `scripts/pick_account.sh` (`plgideascvgroup1` wykluczony)
- skrypt ewaluacyjny kończy się `exit 1` przy błędzie — wcześniej połykał crash i raportował sukces
- stuby managera (`_Plain`, `StaticLoRABank`) muszą implementować **pełny** interfejs; lista przez
  `grep -hoE "manager\.[a-zA-Z_]+|hyper\.[a-zA-Z_]+" src/sampling.py src/gen_cifc.py src/injection.py`
- bufory zależne od configu rejestrować **warunkowo** — bezwarunkowy `register_buffer` unieważnia
  wszystkie wcześniejsze checkpointy (ścisły `load_state_dict`)
- smoke musi mieć **włączoną regularyzację**, inaczej nie przechodzi ścieżek kotwic i lookaheadu
