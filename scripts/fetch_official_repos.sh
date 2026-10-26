#!/usr/bin/env bash
# Tier-2 (optional): clone each benchmark's OFFICIAL scoring repository at the commit we audited.
#
# You do NOT need this for Tier-1 reproduction. Every official parse-failure rule is already copied
# VERBATIM into src/rescore_<benchmark>.py, with the exact source file, function, and (where the repo
# is versioned) the pinned commit cited in each script's header. These clones let you diff our
# wrappers against the upstream source.
mkdir -p official_scorers && cd official_scorers || exit 1

clone() {  # name  url  [commit]
  echo "== $1 =="
  if [ ! -d "$1" ]; then git clone --quiet "$2" "$1" || { echo "  clone failed: $2"; return; }; fi
  [ -n "$3" ] && { git -C "$1" checkout --quiet "$3" 2>/dev/null || echo "  (commit $3 not found; default branch)"; }
}

clone MMAD        https://github.com/jam-cc/MMAD.git                   1490f887b9570613565fa6a7b15d5f81aa183c54
clone MMMU        https://github.com/MMMU-Benchmark/MMMU.git           bc168a9119d986d7cdf1e07b1eeb96ed3e8f92fa
clone PMC-VQA     https://github.com/xiaoman-zhang/PMC-VQA.git          1678ec2c97f0b2bf84591ddccd2e84932630e450
clone ScienceQA   https://github.com/lupantech/ScienceQA.git           2cbf8318e07b9ece895bb2ae605e71e38d623264
clone OmniMedVQA  https://github.com/OpenGVLab/Multi-Modality-Arena.git bd2999a3be2bf4539a7e071cf2cfab5822f66417
clone POPE        https://github.com/RUCAIBox/POPE.git
clone MathVista   https://github.com/lupantech/MathVista.git           53d525874bdde205128e6b160b7357a88277d479
clone MathVision  https://github.com/mathllm/MATH-V.git                a1f5cc3add200c0cd080fad463e500f44ef1fb41

echo
echo "Done. The verbatim scoring functions are in ../src/rescore_*.py (see each header for the"
echo "exact upstream file/function/commit). Every repo above is pinned to the exact commit we audited (POPE has no release tags; its"
echo "evaluate.py is quoted verbatim in the paper and supplement)."
