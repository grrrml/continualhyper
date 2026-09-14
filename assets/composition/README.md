# Kompozycja wielokonceptowa — potok porównania z CIDM

Porównanie jakościowe naszej metody z CIDM (Dong i in., arXiv:2410.17594, NeurIPS 2024) na
**ich** scenach: ich prompty ITP/RTP, ich geometria ramek, nasze obrazy.

## Co tu jest

| plik | co to |
|---|---|
| `scenes.json` | 11 scen: ITP, RTP, parowanie koncept→ramka, prompty regionów, indeksy tasków |
| `../cidm_composition_notes.md` | źródło `scenes.json` — skąd każda liczba pochodzi i czego praca **nie** podaje |

Kod: `scripts/_compose_scenes.py` (sterownik), `src/sampling.py::compose_sample_regions`
(ich rów. 4–5 naszymi adapterami), `scripts/_verify_regional.py` (audyt procesora na CPU).

## Skąd się biorą liczby w `scenes.json`

- **Prompty** — przepisane dosłownie z warstwy tekstowej figur w źródle e-print arXiv, razem
  z literówkami autorów (podwójna spacja w 12.2, brak spacji i kropki w 12.5).
- **Ramki** — zmierzone z wektorowych prostokątów panelu „Region Boxes", nie z oka.
  Konwencja `(x0,y0,x1,y1)` w [0,1] od **lewego górnego** rogu — ta sama, co
  `src.regional._region_vec`. `manager.cond_box` używa innej, `(cx,cy,w,h)`; przelicza to
  `src.sampling._cxcywh` i **nigdzie indziej**.
- **Parowanie koncept↔ramka** — po kolorze prostokąta, nie po kolejności w RTP. W scenach
  3.2, 3.4, 3.5 i 12.1 te dwie kolejności się różnią, więc przepisanie promptu „z figury"
  daje złe przypisanie.

Weryfikacja bez GPU: `python scripts/_compose_scenes.py --dry_run 1 --out <kat>` wypisuje
prompty i rysuje `layout.png` dla każdej sceny — nasz odpowiednik ich panelu „Region Boxes",
i zarazem sposób na sprawdzenie parowania okiem wobec `data/CIFC/Figs/multi-concept.png`.

## Mapowanie ich konceptów na nasze

`V<k>` → task `k−1`, bitowo: V1 dog=0, V2 duck toy=1, V3 cat=2, V4 backpack=3, V5 teddy bear=4,
V6 painting=5, V7 dog=6, V8 drawing=7, V9 cat=8, V10 ink painting=9 — dokładnie kolejność
konceptów w `configs/phaseP/P_paper.yaml`, `configs/phaseX/X_sdxl_*.yaml` i
`configs/phaseR/R_tail_*.yaml`. Sterownik **sprawdza** tę zgodność i przerywa przy rozjeździe.

Identyfikator (`<V1>`) bierze się z configu, nie ze specyfikacji scen: `R_tail` uczy się na
`<V1> dog`, `P_paper`/`X_sdxl` na gołym `dog`. Ten sam plik scen obsługuje obie rodziny.
W scenach 3.2, 3.5, 12.1, 12.2 i 12.4 występują pary tej samej klasy (dwa psy, dwa koty) —
przy rów. 4–5 rozróżnia je **klucz adaptera**, nie tekst promptu, więc brak identyfikatorów
na SDXL nie blokuje żadnej sceny.

## Czego praca nie podaje, a my musimy wybrać

Rozdzielczość, sampler, liczba kroków, ziarno i prompt negatywny dla figur kompozycyjnych
**nie są w pracy**. Tekst mówi o ziarnie 0, wydany kod ma na sztywno 2024. Każdy przebieg
zapisuje te wybory w `manifest.json` obok obrazów; to samo trafia do podpisu rysunku.
Domyślne: natywna rozdzielczość backbone'u, DDIM 50 kroków, CFG 7.5, α=0.1, ziarna 4242+i.

Wydany kod CIDM **nie zawiera pipeline'u SDXL** ani ścieżki kompozycji, więc ich figur nie da
się odtworzyć z ich kodu nawet w zasadzie — to jest zdanie do pracy, nie wymówka.

## Otwarte decyzje (dla człowieka)

1. ~~Scena 3.5~~ — **rozstrzygnięta 14.09 obrazem**: w ich własnym panelu „Ours" w prawej
   ramce jest **pies**, gładkowłosy corgi w skafandrze, zgodny z miniaturą V7 i wyraźnie inny
   od dużego, puchatego V1 w środku; szary kot z miniatury V9 nie występuje w kadrze w ogóle.
   Czyli „V9 dog" to błąd etykiety zamiast **V7**, i w RTP, i przy ramce. Domyślny odczyt to
   `v7`; `--scene35 v9|literal` odtwarza figurę tak, jak jest wydrukowana.
2. **Które sceny do rysunku.** 12.4 (cztery koncepty, dwie pary tej samej klasy) jest
   najtrudniejsza i najbardziej informatywna; 3.1 i 12.6 to ta sama scena u nich dwa razy,
   czyli jedyne miejsce, gdzie widać ICH rozrzut między przebiegami.
3. **Skąd panele CIDM.** `data/CIFC/Figs/multi-concept.png` (= Rys. 12) jest na licencji
   Apache-2.0 i to jedyna wersja, do której mamy jawne prawo redystrybucji — ale 1061×1236 px
   na siedem kolumn daje ~150 px na panel, za mało obok naszych 1024. Wektorowa figura z
   e-printu ma pełne bitmapy 1024, lecz arXiv `nonexclusive-distrib/1.0` nie daje nam praw —
   to prośba do autorów, nie decyzja jednostronna.

Wymagana atrybucja przy reprodukcji `Figs/multi-concept.png` — patrz sekcja 6 notatek.
