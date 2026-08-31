#!/bin/bash
# Wypisuje konto GPU o najwyzszym FairShare sposrod DOZWOLONYCH.
# plgideascvgroup1 celowo pominiete -- decyzja uzytkownika (2026-08-06).
sshare -U -u "$USER" --noheader --format=Account%30,FairShare%12 2>/dev/null \
  | awk '$1 ~ /^(plgideascv1cl|plgroomagine)-gpu-a100$/ {print $2, $1}' \
  | sort -rn | head -1 | awk '{print $2}'
