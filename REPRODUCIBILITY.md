# Reproducibility checklist

Filled against the standard ML reproducibility checklist (WACV encourages its use).

### Models and algorithms
- [x] Clear description of the setting and method — paper Secs. 3–4; the scoring rules are copied
  verbatim into `src/rescore_*.py`.
- [x] Complexity / runtime — Tier-1 reproduction is O(items), CPU, ~1 minute; Tier-2 (inference) is a
  few GPU-hours per model on one 16 GB GPU.
- [x] Downloadable source code with all dependencies — this package; `requirements-cpu.txt` (Tier-1),
  `requirements-gpu.txt` (Tier-2).

### Datasets
- [x] Relevant statistics — per-benchmark item counts are in `data/summary*.jsonl` and printed by
  `reproduce.py`; adjudication label counts: 130 (author) + 40 (independent) MMAD, 60 PMC-VQA.
- [x] Splits / preprocessing — official test splits; the five benchmarks reproduced cleanly, ScienceQA
  treated as code-presence only (near-miss answer format), MathVista/MathVision as defused (documented in paper).
- [x] Excluded data explained — paper Sec. 4 and the taxonomy table.
- [x] Downloadable data — released per-item logs and labels are in `data/`; Tier-2 dataset sources via
  `scripts/fetch_datasets.sh`.
- [x] Annotation process + quality control — verbatim annotator rubric in the paper appendix /
  supplementary; two annotators (one author, one naive to the hypothesis); inter-human κ = 0.910;
  LLM judge certified against the human at κ = 0.956 before scaling.

### Experimental results
- [x] Hyperparameters and how chosen — generation is deterministic (greedy, temperature 0); the LLM
  judge runs at temperature 0; no tuning.
- [x] Number of runs — single deterministic pass per (benchmark, model); no seed variance (stated in
  the Fig. 2 caption). The random-fallback rules are seeded (`random.seed(42)`).
- [x] Definition of the measure — accuracy under each parse-failure rule; artifact = snap − abstain
  (Family A) or fired-correct/N (Family B); over/under-credit defined in the paper.
- [x] Central tendency / variation — exact accuracies on full evaluation sets; agreement reported with
  Cohen's κ; controls report binomial / z-test p-values (`src/k2_control.py`).
- [x] Runtime / compute — Tier-1 ~1 min CPU; Tier-2 single 16 GB consumer GPU.
- [x] Computing infrastructure — five open ≤13B VLMs, 4-bit, one 16 GB GPU.

### Independent reproduction
- [x] Reproducible from this package alone — `make docker` (or `make setup all`) reproduces and
  self-validates every load-bearing number with no external resources. `validate.py` is the automated
  check (33 assertions); it exits non-zero if any number drifts from the paper.
- [x] Inference code validated — `make smoke` runs a real VLM on a synthetic item (GPU, no dataset, no
  license-gated images) and asserts a schema-correct record; verified loading gemma-4-E2B end to end.
