#!/bin/bash -l
# Buduje venv dla TEGO klastra. Uruchamiac przez: bash scripts/run.sh slurm/bootstrap.sbatch
#
# Dlaczego jako job, a nie na login-nodzie:
#  - Helios: login node to x86_64, wezly GPU to aarch64. Venv zbudowany na login-nodzie
#    dostanie kola x86 i padnie w jobie.
#  - Athena: login node nie ma bibliotek dev, wiec cokolwiek sie kompiluje musi isc w jobie.
#
#SBATCH --job-name=bootstrap
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
. slurm/env.sh

echo "=== $(hostname)  arch=$(uname -m)  klaster=$CLUSTER  $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(brak nvidia-smi)"
echo "=== python: $($PYTHON --version 2>&1), $(command -v $PYTHON)"
echo "=== venv docelowy: $VENV"

if [ -d "$VENV" ] && [ -z "${REBUILD:-}" ]; then
  echo "=== venv istnieje. REBUILD=1 zeby przebudowac. Sprawdzam tylko, czy dziala."
else
  [ -d "$VENV" ] && { echo "=== REBUILD: usuwam $VENV"; rm -rf "$VENV"; }
  mkdir -p "$(dirname "$VENV")"
  "$PYTHON" -m venv "$VENV"
fi
. "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip setuptools wheel

REQ=requirements.txt
[ -f "$REQ" ] || { echo "brak $REQ"; exit 1; }

# --- 1) Kola prebuildowane pod ta maszyne, dopasowane do pinow z requirements.txt.
#     Zadnych nazw plikow na sztywno: dla kazdego 'name==version' szukamy koła
#     w $WHEELS. Jesli jest - instalujemy je PRZED reszta, bo inaczej pip sciagnie
#     torcha bez CUDA albo niezgodny torchvision.
SATISFIED=""
if [ -n "${WHEELS:-}" ] && [ -d "$WHEELS" ]; then
  echo
  echo "=============== kola z $WHEELS"
  while IFS= read -r line; do
    case "$line" in ''|'#'*) continue ;; esac
    name=${line%%==*}; ver=${line#*==}
    [ "$name" = "$line" ] && continue          # brak pinu - pomijamy
    # PIN_OVERRIDES: gdy klaster nie ma pinowanej wersji, profil moze wskazac inna.
    for ov in ${PIN_OVERRIDES:-}; do
      ovn=${ov%%==*}
      [ "$(echo "$ovn" | tr 'A-Z_' 'a-z-')" = "$(echo "$name" | tr 'A-Z_' 'a-z-')" ] && ver=${ov#*==}
    done
    lname=$(echo "$name" | tr 'A-Z-' 'a-z_')
    hit=$(ls "$WHEELS"/${lname}-${ver}*aarch64.whl 2>/dev/null | head -1 || true)
    [ -z "$hit" ] && hit=$(ls "$WHEELS"/${lname}-${ver}*.whl 2>/dev/null | head -1 || true)
    if [ -n "$hit" ]; then
      echo "--- $name==$ver  ->  $(basename "$hit")"
      pip install --quiet "$hit"
      SATISFIED="$SATISFIED $name"
    fi
  done < "$REQ"
  [ -z "$SATISFIED" ] && echo "(zadnego pinu nie dopasowano do kol)"
fi

# --- 2) Reszta z requirements.txt: bez tego, co przyszlo z kol, i bez SKIP_PACKAGES.
echo
echo "=============== reszta z $REQ"
FILTERED="$SCRATCH/venvs/req-$PROJECT-$CLUSTER.txt"
: > "$FILTERED"
while IFS= read -r line; do
  case "$line" in ''|'#'*) continue ;; esac
  name=${line%%==*}
  skip=""
  for s in $SATISFIED ${SKIP_PACKAGES:-}; do
    [ "$(echo "$s" | tr 'A-Z_' 'a-z-')" = "$(echo "$name" | tr 'A-Z_' 'a-z-')" ] && skip=1
  done
  [ -n "$skip" ] || echo "$line" >> "$FILTERED"
done < "$REQ"
echo "--- pominiete (kola tej maszyny): $SATISFIED"
echo "--- pominiete (SKIP_PACKAGES profilu): ${SKIP_PACKAGES:-brak}"
echo "--- instaluje $(wc -l < "$FILTERED") pozycji"
pip install -r "$FILTERED"

# --- 3) Weryfikacja: liczy sie tylko to, czy torch widzi GPU i czy kod sie importuje.
echo
echo "=============== WERYFIKACJA"
python - <<'PY'
import sys, platform, importlib
print("python     :", sys.version.split()[0], platform.machine())
import torch
print("torch      :", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu        :", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    x = torch.randn(1024, 1024, device="cuda"); assert float((x@x).sum()) == float((x@x).sum())
    print("matmul     : OK")
for m in ("torchvision", "diffusers", "transformers", "numpy", "timm"):
    try: print(f"{m:11}:", importlib.import_module(m).__version__)
    except Exception as e: print(f"{m:11}: BRAK ({type(e).__name__})")
for m in ("accelerate", "xformers", "bitsandbytes"):
    try: print(f"{m:11}:", importlib.import_module(m).__version__)
    except Exception: print(f"{m:11}: nieobecny")
PY

FREEZE="$SCRATCH/venvs/freeze-$PROJECT-$CLUSTER.txt"
pip freeze > "$FREEZE"
echo
echo "=== pip freeze -> $FREEZE ($(wc -l < "$FREEZE") paczek)"
echo "=== inody venva: $(find "$VENV" -printf . | wc -c)"
echo "=== KONIEC $(date -u +%FT%TZ)"
