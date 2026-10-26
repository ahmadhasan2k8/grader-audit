#!/usr/bin/env bash
# Tier-2 (optional): download the benchmark datasets needed to regenerate the model-output logs.
#
# You do NOT need this for Tier-1 reproduction -- that runs on the released logs in data/.
# Requires huggingface-cli (pip install huggingface-hub) and several GB of disk.
mkdir -p datasets

dl() {  # name  hf_repo_id
  echo "== $1 ($2) =="
  huggingface-cli download "$2" --repo-type dataset --local-dir "datasets/$1" \
    || echo "  download failed; see README 'Tier 2' for the canonical source of $1."
}

dl mmmu        MMMU/MMMU
dl omnimedvqa  foreverbeliever/OmniMedVQA
dl pmcvqa      xmcmic/PMC-VQA
dl scienceqa   derek-thomas/ScienceQA
dl pope        lmms-lab/POPE
dl mathvista   AI4Math/MathVista
dl mathvision  MathLLMs/MathVision

echo
echo "MMAD uses the MVTec-AD / VisA images (license-gated); download them from their official sites"
echo "and point MVTEC_ROOT / VISA_ROOT in src/rescore.py at them. See README 'Tier 2'."
