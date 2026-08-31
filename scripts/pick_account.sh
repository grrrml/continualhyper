#!/bin/bash
# Wypisuje konto GPU, na ktore nalezy rozliczyc zadanie na TYM klastrze.
#
# Polityka (CLAUDE.md): najpierw wydaj grant, ktory wygasa najszybciej -- godziny
# w wygasajacym grancie przepadaja, w waznym jeszcze pol roku nie.
#
# Stan na 2026-08-31 (hpc-grants):
#   plgroomagine-gpu-a100        athena  active do 2026-09-08  ~37 330 h wolnych
#   plgideascvgroup1-gpu-a100    athena  active do 2027-03-23  ~21 030 h
#   plgideascvgroup1-gpu-gh200   helios  active do 2027-03-23  ~35 450 h
#   plgideascv1cl-*              FINISHED 2026-08-26 -- NIE uzywac
#
# Nie parsujemy hpc-grants: format jest site-specific i sie zmienia (patrz CLAUDE.md).
# Zamiast tego lista jest jawna, a wybor weryfikowany przez sshare.
set -u

TODAY=$(date +%Y-%m-%d)
ROOMAGINE_END=2026-09-08

case "$(hostname -f 2>/dev/null || hostname)" in
  *helios*)
    CANDIDATES='plgideascvgroup1-gpu-gh200' ;;
  *athena*)
    if [[ "$TODAY" < "$ROOMAGINE_END" || "$TODAY" == "$ROOMAGINE_END" ]]; then
      CANDIDATES='plgroomagine-gpu-a100 plgideascvgroup1-gpu-a100'
    else
      CANDIDATES='plgideascvgroup1-gpu-a100'
    fi ;;
  *)
    echo "pick_account: nieznany klaster ($(hostname))" >&2; exit 1 ;;
esac

USABLE=$(sshare -U -u "$USER" --noheader --format=Account%40 2>/dev/null | tr -d ' ')

for acc in $CANDIDATES; do
  if grep -qx "$acc" <<< "$USABLE"; then
    echo "$acc"
    exit 0
  fi
done

echo "pick_account: zaden z kandydatow ($CANDIDATES) nie jest dostepny." >&2
echo "  sshare zwrocil: $(echo "$USABLE" | tr '\n' ' ')" >&2
echo "  sprawdz 'hpc-grants' -- moze grant sie skonczyl i lista wymaga aktualizacji." >&2
exit 1
