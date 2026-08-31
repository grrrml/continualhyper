#!/bin/bash
# Jedyna droga uruchamiania zadan. Dlaczego nie goly sbatch: to miejsce wybiera
# grant, wstawia parametry klastra i zapisuje proweniencje. Bez proweniencji
# wynik z trzech miesiecy wstecz jest nieodtwarzalny.
#
# Uzycie:
#   bash scripts/run.sh jobs/sbatch_foo.sh [argumenty skryptu...]
#   GPUS=2 bash scripts/run.sh jobs/sbatch_foo.sh
#   SBATCH_EXTRA="--time=00:20:00" bash scripts/run.sh jobs/sbatch_foo.sh
#   DRY=1 bash scripts/run.sh jobs/sbatch_foo.sh      # pokaz komende, nie wysylaj
#
# Konto i partycja sa podawane w LINII POLECEN sbatch, bo CLI ma pierwszenstwo
# nad '#SBATCH' w skrypcie (sprawdzone: job 21542893 z martwym grantem w skrypcie
# poszedl na koncie z CLI). Dzieki temu istniejace skrypty z zaszytymi kontami
# dzialaja bez edycji.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"

. slurm/env.sh

JOB="${1:?podaj skrypt sbatch, np. jobs/sbatch_foo.sh}"
shift || true
[ -f "$JOB" ] || { echo "run.sh: nie ma $JOB" >&2; exit 1; }

# --- kod idzie tylko przez gita
if [ -z "${SKIP_PULL:-}" ]; then
  echo "== git pull"
  git pull --quiet --ff-only || echo "   (pull nieudany - lece dalej na tym, co jest)"
fi

COMMIT=$(git rev-parse --short HEAD)
DIRTY=""
if [ -n "$(git status --porcelain -- ':!results' ':!logs' ':!data' 2>/dev/null)" ]; then
  DIRTY=" (DIRTY)"
  echo "== UWAGA: niezacommitowane zmiany w kodzie. Wynik nie bedzie odtwarzalny z commita."
fi

# --- grant
ACCOUNT=$(bash slurm/pick-account.sh)
echo "== klaster   : $CLUSTER"
echo "== partycja  : $PARTITION"
echo "== konto     : $ACCOUNT"
echo "== commit    : $COMMIT$DIRTY"

# --- zasoby: nadpisujemy TYLKO gdy jawnie poproszono. Skrypty maja strojone
#     --mem/--gres i hurtowe nadpisywanie ich zepsulo by wiecej, niz naprawia.
RES=()
if [ -n "${GPUS:-}" ]; then
  RES+=(--gres="gpu:$GPUS")
  RES+=(--cpus-per-task=$(( CPUS_PER_GPU * GPUS )))
  MEMNUM=${MEM_PER_GPU%G}
  RES+=(--mem=$(( MEMNUM * GPUS ))G)
  echo "== zasoby    : ${GPUS}x GPU, $(( CPUS_PER_GPU * GPUS )) cpu, $(( MEMNUM * GPUS ))G"
fi

CMD=(sbatch --account="$ACCOUNT" --partition="$PARTITION" "${RES[@]}")
[ -n "${SBATCH_EXTRA:-}" ] && read -ra EXTRA <<< "$SBATCH_EXTRA" && CMD+=("${EXTRA[@]}")
CMD+=("$JOB" "$@")

if [ -n "${DRY:-}" ]; then
  printf '== DRY, nie wysylam:\n   '; printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

OUT=$("${CMD[@]}")
echo "$OUT"
JOBID=$(echo "$OUT" | grep -oE '[0-9]+$' | tail -1)
[ -n "$JOBID" ] || { echo "run.sh: nie wyciagnalem jobid z '$OUT'" >&2; exit 1; }

# --- proweniencja: jedyna rzecz, ktora pozwala odtworzyc wynik po miesiacach
INFO_DIR="$DATA_ROOT/results/$JOBID"
mkdir -p "$INFO_DIR"
{
  echo "jobid      : $JOBID"
  echo "kiedy      : $(date -u +%FT%TZ)"
  echo "klaster    : $CLUSTER ($(hostname -f 2>/dev/null || hostname))"
  echo "konto      : $ACCOUNT"
  echo "partycja   : $PARTITION"
  echo "projekt    : $PROJECT"
  echo "commit     : $COMMIT$DIRTY"
  echo "branch     : $(git rev-parse --abbrev-ref HEAD)"
  echo "remote     : $(git config --get remote.origin.url)"
  echo "skrypt     : $JOB"
  echo "argumenty  : $*"
  echo "komenda    : $(printf '%q ' "${CMD[@]}")"
  echo "venv       : $VENV"
  echo "HF_HOME    : $HF_HOME"
  echo "DATA_ROOT  : $DATA_ROOT"
  echo
  echo "--- git status (kod, bez results/logs/data)"
  git status --porcelain -- ':!results' ':!logs' ':!data' 2>/dev/null || true
  echo
  echo "--- pip freeze"
  # Nie uruchamiamy tu pipa: na Heliosie venv jest aarch64, a run.sh chodzi na
  # login-nodzie x86_64, wiec 'pip freeze' po prostu sie nie wykona. Bierzemy
  # zrzut zapisany przez bootstrap.sbatch po WLASCIWEJ stronie.
  FREEZE="$SCRATCH/venvs/freeze-$PROJECT-$CLUSTER.txt"
  if [ -f "$FREEZE" ]; then
    echo "(z $FREEZE, $(date -u -r "$FREEZE" +%FT%TZ))"
    cat "$FREEZE"
  elif [ -d "$VENV" ]; then
    echo "(brak $FREEZE - venv istnieje, ale nie zbudowany przez bootstrap.sbatch;"
    echo " uruchom: bash scripts/run.sh slurm/bootstrap.sbatch)"
  else
    echo "(BRAK VENVA $VENV - job padnie; uruchom: bash scripts/run.sh slurm/bootstrap.sbatch)"
  fi
} > "$INFO_DIR/run-info.txt"

echo "== proweniencja: $INFO_DIR/run-info.txt"
echo "== log: ssh $CLUSTER \"tail -f $DATA_ROOT/logs/*-$JOBID.out\""
