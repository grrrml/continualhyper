#!/bin/bash
# Wypisuje konto, na ktore nalezy rozliczyc zadanie na TYM klastrze.
#
# Polityka (CLAUDE.md): najpierw grant wygasajacy najszybciej - jego godziny
# przepadaja, godziny w grancie waznym jeszcze pol roku nie.
#
# Kandydaci i ich daty pochodza z profilu klastra (ACCOUNTS="konto:data ...").
# Wpisy po dacie sa pomijane, wiec lista nie wymaga sprzatania po terminie.
# Wybor jest weryfikowany przez sshare - nie zwracamy konta, ktorego nie mamy.
set -u

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SLURM_DIR/env.sh"

TODAY=$(date +%Y-%m-%d)
USABLE=$(sshare -U -u "$USER" --noheader --format=Account%40 2>/dev/null | tr -d ' ')

SKIPPED=""
for entry in $ACCOUNTS; do
  acc=${entry%%:*}
  exp=${entry##*:}
  if [ "$exp" != "$acc" ] && [ "$TODAY" \> "$exp" ]; then
    SKIPPED="$SKIPPED $acc(wygasl $exp)"
    continue
  fi
  if grep -qx "$acc" <<< "$USABLE"; then
    echo "$acc"
    exit 0
  fi
  SKIPPED="$SKIPPED $acc(brak w sshare)"
done

echo "pick-account: brak dostepnego konta na $CLUSTER." >&2
echo "  odrzucone:$SKIPPED" >&2
echo "  sshare zwrocil: $(echo "$USABLE" | tr '\n' ' ')" >&2
echo "  sprawdz 'hpc-grants' - moze trzeba dopisac konto do slurm/clusters/$CLUSTER.sh" >&2
exit 1
