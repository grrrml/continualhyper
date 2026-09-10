#!/bin/bash
# Pakuje drzewa obrazow z ZAKONCZONYCH faz do jednego .tar na zamiatanie.
#
# Po co: na $SCRATCH wyczerpuje sie limit INODOW (1 000 000), a nie miejsce --
# przy 4.4% zajetosci dysku bylo 97.8% inodow. Jedno zamiatanie ewaluacji to
# 10 000 plikow PNG, wiec kilka nowych przebiegow nie ma sie gdzie zapisac.
# Tar zwija takie drzewo do jednego inodu i nie traci ani jednego obrazu.
#
# Co ZOSTAJE poza tarem, zeby wyniki dalo sie czytac bez rozpakowywania:
#   *.pt (checkpointy), *.json (w tym cifc_metrics.json), *.txt, *.yaml, *.md
#
# Kasujemy DOPIERO po weryfikacji: liczba wpisow w tarze musi sie zgadzac z liczba
# plikow na liscie. Jesli sie nie zgadza, katalog zostaje nietkniety i mowimy o tym.
#
# Uzycie (z katalogu repo):
#   bash scripts/_pack_outputs.sh outputs/phaseP outputs/sdxl
#   DRY=1 bash scripts/_pack_outputs.sh outputs/phaseP     # tylko pokaz, co by zrobil
set -uo pipefail

MIN_FILES=${MIN_FILES:-100}      # ponizej tego tar nie oplaca sie inodowo
total_before=0 total_after=0

for ROOT in "$@"; do
  [ -d "$ROOT" ] || { echo "!! nie ma $ROOT -- pomijam"; continue; }
  for D in "$ROOT"/*/; do
    D="${D%/}"
    [ -d "$D" ] || continue
    TAR="$D.images.tar"
    if [ -e "$TAR" ]; then echo "-- $D: tar juz jest, pomijam"; continue; fi

    LIST=$(mktemp)
    find "$D" -type f \
      ! -name '*.pt' ! -name '*.json' ! -name '*.txt' ! -name '*.yaml' ! -name '*.md' \
      -print > "$LIST"
    n=$(wc -l < "$LIST")
    if [ "$n" -lt "$MIN_FILES" ]; then
      echo "-- $D: $n plikow, ponizej progu $MIN_FILES -- pomijam"
      rm -f "$LIST"; continue
    fi
    total_before=$((total_before + n))

    if [ -n "${DRY:-}" ]; then
      echo "DRY $D: spakowalbym $n plikow -> $TAR"
      rm -f "$LIST"; continue
    fi

    if ! tar -cf "$TAR" -T "$LIST"; then
      echo "!! $D: tar padl -- nic nie kasuje"
      rm -f "$TAR" "$LIST"; continue
    fi
    m=$(tar -tf "$TAR" | wc -l)
    if [ "$m" -ne "$n" ]; then
      echo "!! $D: w tarze $m wpisow, na liscie $n -- NIE kasuje, tar zostaje do obejrzenia"
      rm -f "$LIST"; continue
    fi

    xargs -a "$LIST" -d '\n' rm -f
    find "$D" -mindepth 1 -type d -empty -delete
    rm -f "$LIST"
    total_after=$((total_after + 1))
    echo "OK  $D: $n plikow -> $TAR"
  done
done

echo "== spakowano $total_before plikow do $total_after archiwow"
