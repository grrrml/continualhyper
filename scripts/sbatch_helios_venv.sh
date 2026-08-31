#!/bin/bash -l
# Buduje venv aarch64 dla continualhyper na wezle GPU Heliosa.
# Wsadowo, nie interaktywnie: powtarzalne i zostaje log.
#SBATCH --job-name=ch-venv
#SBATCH --account=plgideascvgroup1-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/net/scratch/hscra/plgrid/plgrrrml/continualhyper/logs/%x-%j.out
#SBATCH --error=/net/scratch/hscra/plgrid/plgrrrml/continualhyper/logs/%x-%j.err

set -euo pipefail

WHEELS=/net/software/aarch64/el8/wheels/ML-bundle/24.06a
VENV="$SCRATCH/venvs/continualhyper-helios"
REPO="$HOME/projekty/continualhyper"

echo "=== wezel: $(hostname), arch: $(uname -m)"
echo "=== data: $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

module load ML-bundle/24.06a
echo "=== python: $(python3.11 --version), $(which python3.11)"

mkdir -p "$(dirname "$VENV")"
if [ -d "$VENV" ]; then
  echo "=== venv juz istnieje, usuwam i buduje od nowa"
  rm -rf "$VENV"
fi
python3.11 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip setuptools wheel
echo "=== pip: $(pip --version)"

# --- 1) koła aarch64 NAJPIERW, zeby pip nie sciagnal x86/CPU-only z PyPI
echo
echo "=============== koła aarch64 z $WHEELS"
for w in \
  torch-2.6.0+cu124.post3-cp311-cp311-linux_aarch64.whl \
  torchvision-0.21.0+cu124torch260-cp311-cp311-linux_aarch64.whl \
  triton-3.1.0-cp311-cp311-linux_aarch64.whl ; do
  if [ -f "$WHEELS/$w" ]; then
    echo "--- $w"
    pip install --quiet "$WHEELS/$w"
  else
    echo "!!! BRAK: $w"
  fi
done

# --- 2) reszta z requirements.txt, bez rodziny torcha (juz zainstalowana)
echo
echo "=============== reszta z requirements.txt"
REQ_HELIOS="$SCRATCH/venvs/req-continualhyper-helios.txt"
grep -viE '^(torch|torchvision|triton|xformers|bitsandbytes)==' "$REPO/requirements.txt" > "$REQ_HELIOS"
echo "--- pominiete (dostarczone jako koła aarch64):"
grep -iE '^(torch|torchvision|triton|xformers|bitsandbytes)==' "$REPO/requirements.txt" | sed 's/^/      /'
echo "--- instaluje:"
cat "$REQ_HELIOS" | sed 's/^/      /'
pip install -r "$REQ_HELIOS"

# --- 3) weryfikacja
echo
echo "=============== WERYFIKACJA"
python - <<'PY'
import sys, platform
print("python    :", sys.version.split()[0], platform.machine())
import torch
print("torch     :", torch.__version__)
print("cuda avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu       :", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    x = torch.randn(2048, 2048, device="cuda")
    print("matmul    :", float((x @ x).sum()) != 0.0)
import torchvision; print("torchvision:", torchvision.__version__)
import diffusers, transformers, accelerate
print("diffusers :", diffusers.__version__)
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
import numpy; print("numpy     :", numpy.__version__)
try:
    import xformers; print("xformers  :", xformers.__version__)
except Exception as e:
    print("xformers  : BRAK/BLAD:", e)
try:
    import bitsandbytes; print("bitsandbytes:", bitsandbytes.__version__)
except Exception as e:
    print("bitsandbytes: BRAK/BLAD:", e)
PY

echo
echo "=============== pip freeze -> plik"
pip freeze > "$SCRATCH/venvs/freeze-continualhyper-helios.txt"
echo "zapisano $SCRATCH/venvs/freeze-continualhyper-helios.txt ($(wc -l < "$SCRATCH/venvs/freeze-continualhyper-helios.txt") paczek)"
echo "=== inody venva: $(find "$VENV" -printf . | wc -c)"
echo "=== KONIEC $(date -u +%FT%TZ)"
