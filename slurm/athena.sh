# Profil klastra: Athena (x86_64 + A100). Same fakty, zadnej logiki.
CLUSTER=athena
PARTITION=plgrid-gpu-a100

# Konta w kolejnosci preferencji: najpierw grant wygasajacy najszybciej, bo jego
# godziny przepadaja. Format: konto:data_wygasniecia. Wpisy po dacie sa pomijane
# automatycznie, wiec lista nie wymaga sprzatania po terminie.
ACCOUNTS="plgroomagine-gpu-a100:2026-09-08 plgideascvgroup1-gpu-a100:2027-03-23"

# Athena: drzewo modulow jest niewspierane - nie ma tu 'module load'.
CLUSTER_MODULES=""

# Login node ma TYLKO Pythona 3.9 (/usr/bin/python3.9); 3.11 nie istnieje w systemie.
# Interpreter dostarcza wiec uv, ktory potrafi go sobie sciagnac. Tak byly zbudowane
# dotychczasowe venvy (analysing-unlearning: cpython-3.10.19 z magazynu uv).
# UWAGA na przedawniona alternatywe: UnHype trzymal python311/ WEWNATRZ katalogu
# projektu na pr2, czyli jego interpreter wygasa razem z grantem 2026-09-08.
VENV_TOOL=uv
PYTHON=3.11

# Magazyn interpreterow i cache uv na $SCRATCH: $HOME ma 100 000 inodow, a jedna
# instalacja CPythona to ~4000 plikow, cache uv dziesiatki tysiecy.
export UV_PYTHON_INSTALL_DIR="$SCRATCH/uv/python"
export UV_CACHE_DIR="$SCRATCH/uv/cache"

VENV="$SCRATCH/venvs/$PROJECT-athena"
HF_CACHE="$SCRATCH/.cache/huggingface"
DATA_ROOT="$SCRATCH/$PROJECT"

# Kola prebuildowane: na Athenie bierzemy z PyPI (x86_64).
WHEELS=""

# Skalowanie zasobow na 1 GPU (wezel 8x A100 = ~1/8 rdzeni i RAM).
CPUS_PER_GPU=16
MEM_PER_GPU=80G

GPU_ONLY=1   # Cyfronet zawiesza konta za powazne obciazenia CPU-only na tej partycji
