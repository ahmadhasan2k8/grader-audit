# Same Outputs, Different Scores — reproduction package

Code and data for *"Same Outputs, Different Scores: Auditing the Unchecked Graders of Multimodal LLM
Benchmarks,"* Ahmad Jarjis Hasan and Jonathan Lwowski, **WACV 2027** (Evaluations and Datasets Track).
Everything needed to reproduce every number, table, and figure is included here; **Tier-1 reproduction needs no GPU, no model downloads, and no dataset
downloads** — it runs on the released per-item logs.

---

## TL;DR — one command (only Docker required)

```bash
docker build -f Dockerfile.cpu -t grader-audit .   &&   docker run --rm grader-audit
```

This builds the environment, regenerates all results into `results/`, and self-validates, ending in:

```
VALIDATION PASSED: 33/33 checks reproduced.
```

No Docker? Two commands:

```bash
make setup      # pip install matplotlib + scipy  (Tier-1 deps)
make all        # reproduce everything, then validate against the paper
```

---

## What gets reproduced (Tier-1, CPU, ~1 minute)

`make all` runs `reproduce.py` then `validate.py`. The self-test asserts, from the released data:

| Claim (paper) | Reproduced value |
|---|---|
| MMAD scoring-rule swing (Fig. 2) | snap − abstain = **+0.212** (Pixtral), **+0.220** (Llama) |
| PMC-VQA bidirectional (Fig. 3) | **+0.075** (Pixtral) vs **−0.190** (Llama, deflation) |
| MMAD judge-vs-human agreement | Cohen's **κ = 0.956** (n=130) |
| MMAD inter-human agreement | Cohen's **κ = 0.910** (n=40) |
| PMC-VQA judge-vs-human (k≈4) | Cohen's **κ = 0.839** (n=60) |
| PMC-VQA Llama under-credit | **15/48 = 31%** |
| Likelihood confound (OmniMedVQA, Llama) | logit acc **0.403** < snap 0.565 (predicts ``A'' 459/600) |
| Budget robustness (MMAD, 16→64 tokens) | non-compliance/artifact persist: Pixtral 0.27/**+0.183**, Llama 0.33/**+0.171** |
| Unified 8-benchmark table | **40/40 cells** match the released reference |

Figures 2 and 3 are regenerated as `results/fig2_rule_swing.pdf` and `results/fig3_bidirectional.pdf`.

---

## How it works

Each released log record (`data/<benchmark>_<model>.jsonl`) stores, per item, the gold answer and the
verdict of **each scoring rule** (the official rules were executed at inference time and their per-item
output stored). So accuracy under every rule is recomputed here as a plain aggregate over the logs — the
models and the benchmark datasets are not needed. The over/under-credit decomposition and the agreement
κ's are recomputed from the released blinded manifests and the human/judge label sets.

For MMAD, OmniMedVQA, PMC-VQA, and POPE the artifact is the raw snap − abstain gap. For MMMU and
ScienceQA the paper uses the de-circularized `fired_correct / N` definition, and MathVista/MathVision are
reported as defused; `src/unified_artifacts.py` applies the correct per-family definition and reproduces
the released `unified_artifacts.json` exactly.

---

## Verifying we ran the authors' code (optional, CPU + network)

Every scoring decision in this package is the benchmark authors' own, and you can check that without
trusting us:

1. `bash scripts/fetch_official_repos.sh` clones each benchmark's official repository **at the exact
   commit we audited** (all pinned in the script). Diff any `src/rescore_*.py` against the upstream
   file/function cited in its header -- the parse and fallback logic is copied verbatim.
2. `python3 src/run_mmad_official.py --model pixtral-12b` feeds our released responses to MMAD's own
   unmodified `parse_answer` and `caculate_accuracy_mmad` (needs `pip install pandas seaborn`, the
   official repo's dependencies). It prints the official micro accuracy next to the recorded value and
   our wrapper's, from `data/REPRODUCED_NUMBERS.json`: identical to four decimals on four of five
   models, Pixtral within 0.0018 (the first-vs-last tie-break the paper documents in Sec. 4).
3. The per-item logs in `data/` contain the raw responses, so any scorer -- theirs, ours, or your own
   -- can be re-run on identical outputs.

## Tier 2 (optional, GPU) — validate / regenerate the logs from scratch

Only needed to re-run the models themselves. Requires a single 16 GB GPU and the Tier-2 dependencies
(`make tier2`). The released logs are the reproduction unit for the paper's claims (Tier-1), so this is
for transparency, not required to verify any result.

**Validate the inference code (no dataset, no license-gated images):**
```bash
make tier2            # pip install torch/transformers/unsloth/bitsandbytes/...
make smoke            # run a real VLM on a synthetic item; assert a valid record in the released schema
```
`make smoke` loads a small model, runs inference on a generated image + a synthetic yes/no question, and
applies all three scoring rules. We verified it loads `gemma-4-E2B`, generates, and produces a
schema-correct record (`SMOKE PASSED`).

**Regenerate real logs (needs the datasets):**
```bash
make fetch-data       # clone official scorer repos + download benchmark datasets
python src/rescore.py        --model gemma-4-e4b --datasets DS-MVTec --limit 5   # MMAD, sample
python src/rescore_pmcvqa.py --model llama-3.2-11b                               # PMC-VQA
# one src/rescore_<benchmark>.py per benchmark; see each header for usage.
```
We verified `src/rescore.py` regenerates MMAD records in the exact released schema on this hardware.
Note: model generation is version- and hardware-sensitive, so regenerated text may differ slightly from
the released logs across environments — which is precisely why the logs (not a re-run) are the
reproduction unit for the paper.

The five models are Gemma-4-E4B, Qwen2.5-VL-7B, Qwen3-VL-8B, Pixtral-12B, and Llama-3.2-11B-Vision
(all 4-bit, 16 GB). Model loading uses standard `transformers`/`unsloth` (`src/_image_util.py` is the
only helper); no proprietary code.

---

## The official scorers (provenance)

Every parse-failure rule is copied **verbatim** into `src/rescore_<benchmark>.py`, with the upstream
file, function, and (where versioned) the pinned commit in each header. `scripts/fetch_official_repos.sh`
clones the upstream repos so you can diff.

| Benchmark | Rule (function) | Pinned |
|---|---|---|
| MMAD | no-threshold snap (`get_close_matches`, cutoff 0) | — |
| MMMU | `parse_multi_choice_response` → `random.choice` | — |
| OmniMedVQA | no-threshold snap (`SequenceMatcher`) | — |
| PMC-VQA | no-threshold snap (`SequenceMatcher`) | `1678ec2` |
| POPE | default class on parse failure | — |
| ScienceQA | `random.choice` on parse failure | `2cbf831` |

---

## Repository layout

```
reproduce.py        Tier-1 driver: runs every analysis -> results/
validate.py         self-test: asserts reproduced numbers == the paper
Makefile            make docker | setup | all | reproduce | validate | fetch-data | tier2
Dockerfile.cpu      turnkey Tier-1 (only Docker needed)
Dockerfile.gpu      Tier-2 (GPU) image
src/                scorers (rescore_*.py), adjudication tools, analysis, figures
data/               released per-item logs, blinded manifests, and human/judge label sets
scripts/            fetch_official_repos.sh, fetch_datasets.sh  (Tier-2)
results/            regenerated tables and figures land here
```

Key data files: `data/<benchmark>_<model>.jsonl` (per-item logs); `data/adjudication_manifest*.jsonl`
and `data/adjudication_blinded*.jsonl` (blinded adjudication records); `data/human_labels.jsonl` (130,
author), `data/human_labels_annotator2.jsonl` (40, independent annotator naive to the hypothesis),
`data/human_labels_pmcvqa.jsonl` (60), `data/judge_labels*.jsonl` (LLM judge); `data/summary*.jsonl`
and `data/unified_artifacts.json` (released reference numbers).

---

## License

Code: MIT (`LICENSE`). Released labels and derived data: CC-BY-4.0. The benchmark datasets and models
remain under their own licenses; this release contains only model outputs and adjudication labels.

---

## Citation

```bibtex
@inproceedings{hasan2027sameoutputs,
  title     = {Same Outputs, Different Scores: Auditing the Unchecked Graders of
               Multimodal LLM Benchmarks},
  author    = {Hasan, Ahmad Jarjis and Lwowski, Jonathan},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## License

Code MIT, released labels CC BY 4.0. See `LICENSE` and `DATASHEET.md`.
