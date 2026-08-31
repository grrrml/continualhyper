# Profil klastra: Helios. UWAGA na rozjazd architektur:
# login node = x86_64 (AMD EPYC 9654), wezly GPU = aarch64 (Grace) + GH200.
# Dlatego venv MUSI powstawac w jobie, nie na login-nodzie, i Anaconda nie dziala.
CLUSTER=helios
PARTITION=plgrid-gpu-gh200

ACCOUNTS="plgideascvgroup1-gpu-gh200:2027-03-23"

# ML-bundle/24.06a, NIE domyslny 25.04: 25.04 ma 4 kola i nie ma torchvision.
CLUSTER_MODULES="ML-bundle/24.06a"

# Na Heliosie interpreter bierzemy Z MODULU, nie z uv: to ten sam python3.11,
# pod ktory zbudowano kola aarch64 w $WHEELS.
VENV_TOOL=venv
PYTHON=python3.11

VENV="$SCRATCH/venvs/$PROJECT-helios"
HF_CACHE="$SCRATCH/.cache/huggingface"
DATA_ROOT="$SCRATCH/$PROJECT"

# Kola aarch64 zbudowane pod te maszyne. Instalowane PRZED requirements.txt,
# inaczej pip sciaga torcha bez CUDA albo niezgodny torchvision.
WHEELS=/net/software/aarch64/el8/wheels/ML-bundle/24.06a

CPUS_PER_GPU=32
MEM_PER_GPU=120G

# sbatch wymaga '#!/bin/bash -l' - bez -l system modulow nie jest zainicjalizowany.
NEEDS_LOGIN_SHELL=1
GPU_ONLY=1

# Helios: kolo xformers 0.0.28 zbudowane pod torch 2.3.1 - importuje sie, ale
# rozszerzenia C++/CUDA nie laduja sie pod 2.5.1. Guard is_xformers_available()
# sprawdza importowalnosc, nie dzialanie, wiec by przepuscil i enable_xformers_*
# padloby w runtime. Bez pakietu guard zwraca False, a diffusers uzywa
# AttnProcessor2_0 (natywne SDPA torcha) - porownywalnie szybko.
SKIP_PACKAGES="xformers"
