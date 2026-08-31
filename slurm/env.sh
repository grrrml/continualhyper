#!/bin/bash
# Jedyne miejsce, ktore ustala srodowisko uruchomieniowe. Wybiera profil klastra
# po hostname i eksportuje wszystko, czego potrzebuja joby.
#
# Uzycie: source slurm/env.sh   (dziala i na login-nodzie, i w jobie)
#
# Nie hardkodowac tu niczego zaleznego od klastra - to idzie do clusters/<alias>.sh.

# Katalog repo wyliczony ze sciezki tego pliku, zeby nie zalezec od cwd.
SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$SLURM_DIR/.." && pwd)"
export PROJECT="$(basename "$REPO_ROOT")"

: "${SCRATCH:?SCRATCH nie jest ustawione - to nie jest wezel Cyfronetu?}"

case "$(hostname -f 2>/dev/null || hostname)" in
  *helios*) CLUSTER_PROFILE="$SLURM_DIR/clusters/helios.sh" ;;
  *athena*) CLUSTER_PROFILE="$SLURM_DIR/clusters/athena.sh" ;;
  # Wezly obliczeniowe nie zawsze maja alias klastra w hostname; rozpoznajemy
  # po sciezce SCRATCH, ktora jest inna na kazdym klastrze.
  *) case "$SCRATCH" in
       */scratch/hscra/*) CLUSTER_PROFILE="$SLURM_DIR/clusters/helios.sh" ;;
       */tscratch/*)      CLUSTER_PROFILE="$SLURM_DIR/clusters/athena.sh" ;;
       *) echo "env.sh: nie rozpoznaje klastra (hostname=$(hostname), SCRATCH=$SCRATCH)" >&2
          return 1 2>/dev/null || exit 1 ;;
     esac ;;
esac

# shellcheck source=/dev/null
. "$CLUSTER_PROFILE"

export CLUSTER PARTITION ACCOUNTS PYTHON VENV WHEELS DATA_ROOT
export CPUS_PER_GPU MEM_PER_GPU
export HF_HOME="$HF_CACHE"
# Wagi z torch.hub / torchvision (detektory, DINO) ida na $SCRATCH, nie do $HOME:
# $HOME ma 100 000 inodow na wszystko, a domyslny ~/.cache/torch rosnie po cichu.
export TORCH_HOME="${TORCH_HOME:-$SCRATCH/.cache/torch}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg          # wezly nie maja wyswietlacza; plt.show() by cicho gubil figury

if [ -n "${CLUSTER_MODULES:-}" ]; then
  # '1>&2' jest tu istotne: 'module load' sypie kilkadziesiat linii, a env.sh jest
  # sourcowany w podstawieniu polecen (ACCOUNT=$(bash slurm/pick-account.sh)).
  # Gdyby cokolwiek trafilo na stdout, konto zostaloby zasmiecone i zadanie
  # rozliczyloby sie na bzdure - cicho. Lmod pisze na stderr, ale to konwencja,
  # nie gwarancja. Nie da sie tego objac przez $(...), bo 'module' to funkcja
  # shella i musi dzialac w BIEZACYM shellu, inaczej eksporty przepadaja.
  #
  # '|| true' jest potrzebne, a nie lenistwo: na Heliosie 'module load ML-bundle/24.06a'
  # zwraca 1 NA LOGIN-NODZIE, bo podmoduly Clang i NCCL istnieja tylko na wezlach GPU.
  # Bundle laduje sie mimo tego i dziala (potwierdzone: venv zbudowany tym modulem).
  # Bez tego run.sh (ktory ma set -e i uruchamia sie wlasnie na login-nodzie) umieral
  # bez zadnego komunikatu. Wewnatrz joba ten sam load zwraca 0.
  for m in $CLUSTER_MODULES; do
    module load "$m" 1>&2 || echo "env.sh: 'module load $m' zwrocil blad (na login-nodzie normalne - czesciowy load)" >&2
  done
fi

if [ "${1:-}" = "--print" ]; then
  echo "PROJECT=$PROJECT"
  echo "CLUSTER=$CLUSTER   profil=$CLUSTER_PROFILE"
  echo "PARTITION=$PARTITION"
  echo "ACCOUNTS=$ACCOUNTS"
  echo "VENV=$VENV"
  echo "HF_HOME=$HF_HOME"
  echo "DATA_ROOT=$DATA_ROOT"
  echo "WHEELS=${WHEELS:-(brak)}"
  echo "CPUS_PER_GPU=$CPUS_PER_GPU  MEM_PER_GPU=$MEM_PER_GPU"
fi
