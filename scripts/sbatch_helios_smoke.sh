#!/bin/bash -l
# Smoke test continualhyper na Heliosie. Nie przebudowuje venva -- tylko go sprawdza
# i przechodzi sciezke diffusers SD-1.5 -> GPU -> obrazek.
#SBATCH --job-name=ch-smoke
#SBATCH --account=plgideascvgroup1-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/net/scratch/hscra/plgrid/plgrrrml/continualhyper/logs/%x-%j.out
#SBATCH --error=/net/scratch/hscra/plgrid/plgrrrml/continualhyper/logs/%x-%j.err

set -euo pipefail

VENV="$SCRATCH/venvs/continualhyper-helios"
REPO="$HOME/projekty/continualhyper"
export HF_HOME="$SCRATCH/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
OUT="$SCRATCH/continualhyper/results/helios-smoke"
mkdir -p "$OUT" "$HF_HOME"

echo "=== wezel $(hostname)  arch $(uname -m)  $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load ML-bundle/24.06a
source "$VENV/bin/activate"
cd "$REPO"

echo
echo "=== wersje"
python - <<'PY'
import sys, platform, torch, torchvision, diffusers, transformers, timm, numpy
print("python     :", sys.version.split()[0], platform.machine())
print("torch      :", torch.__version__, "| cuda:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))
print("torchvision:", torchvision.__version__)
print("diffusers  :", diffusers.__version__)
print("transformers:", transformers.__version__)
print("timm       :", timm.__version__)
print("numpy      :", numpy.__version__)
PY

echo
echo "=== czy kod projektu sie importuje"
python -c "import sys; sys.path.insert(0,'.'); import src.hyper_head, src.injection, src.manager, src.sampling, src.sd_loader, src.tokens; print('src.* OK')"

echo
echo "=== SMOKE: SD-1.5 -> GPU -> obrazek"
python - <<'PY'
import os, json, time, pathlib, torch
from diffusers import StableDiffusionPipeline
out = pathlib.Path(os.environ["SCRATCH"]) / "continualhyper/results/helios-smoke"
out.mkdir(parents=True, exist_ok=True)

MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
t0 = time.time()
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False).to("cuda")
load_s = round(time.time() - t0, 1)

g = torch.Generator(device="cuda").manual_seed(2025)
t0 = time.time()
img = pipe("a chain saw on a workbench, product photo",
           num_inference_steps=25, guidance_scale=7.5, generator=g).images[0]
gen_s = round(time.time() - t0, 1)
img.save(out / "smoke.png")

info = {
    "host": os.uname().nodename, "arch": os.uname().machine,
    "torch": torch.__version__, "torchvision": __import__("torchvision").__version__,
    "gpu": torch.cuda.get_device_name(0),
    "slurm_job": os.environ.get("SLURM_JOB_ID"),
    "account": os.environ.get("SLURM_JOB_ACCOUNT"),
    "model": MODEL, "seed": 2025, "load_s": load_s, "gen_s": gen_s,
    "uwaga": "torch 2.6.0/torchvision 0.21.0 - o wersje nizej niz pin 2.7.1/0.22.1 z requirements.txt (brak kol aarch64)",
}
(out / "run-info.json").write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2))
PY

echo
ls -la "$OUT"
echo "=== KONIEC $(date -u +%FT%TZ)"
