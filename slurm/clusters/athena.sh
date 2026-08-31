# Profil klastra: Athena (x86_64 + A100). Same fakty, zadnej logiki.
CLUSTER=athena
PARTITION=plgrid-gpu-a100

# Konta w kolejnosci preferencji: najpierw grant wygasajacy najszybciej, bo jego
# godziny przepadaja. Format: konto:data_wygasniecia. Wpisy po dacie sa pomijane
# automatycznie, wiec lista nie wymaga sprzatania po terminie.
ACCOUNTS="plgideascvgroup1-gpu-a100:2027-03-23"    # plgroomagine to grant magazynowy tego projektu; obliczenia szly na plgideascv1cl (ZAKONCZONY 2026-08-26)

# Athena: drzewo modulow jest niewspierane - nie ma tu 'module load'.
CLUSTER_MODULES=""
PYTHON=python3.11

VENV="$SCRATCH/venvs/$PROJECT-athena"
HF_CACHE="$SCRATCH/.cache/huggingface"
DATA_ROOT="$SCRATCH/$PROJECT"

# Kola prebuildowane: na Athenie bierzemy z PyPI (x86_64).
WHEELS=""

# Skalowanie zasobow na 1 GPU (wezel 8x A100 = ~1/8 rdzeni i RAM).
CPUS_PER_GPU=16
MEM_PER_GPU=80G

GPU_ONLY=1   # Cyfronet zawiesza konta za powazne obciazenia CPU-only na tej partycji
