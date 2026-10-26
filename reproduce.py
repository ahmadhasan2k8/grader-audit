"""Tier-1 reproduction driver: no GPU, no dataset download, ~1 minute.

Runs every analysis on the released data (data/) and writes all numbers, tables, and figures to
results/. Reproduces the load-bearing claims of the paper end to end:

  * the per-benchmark artifact magnitudes (the up-to-21-point swing; Fig. 2),
  * the unified artifact table with the paper's per-family definitions,
  * the human-anchored adjudication: judge-vs-human and inter-human agreement (Cohen's kappa),
  * the bidirectional over/under-credit decomposition (Fig. 3, the deflation result),
  * the k=2 latent-knowledge control.

Usage:  python reproduce.py
"""
import subprocess
import sys

STEPS = [
    ("Per-benchmark artifact magnitudes (raw snap - abstain, from logs)", "src/score_from_logs.py"),
    ("Unified artifact table (paper's per-family definitions)", "src/unified_artifacts.py"),
    ("MMAD adjudication: judge-vs-human kappa + over/under-credit", "src/adjudication_analysis.py"),
    ("PMC-VQA adjudication: kappa + bidirectional decomposition", "src/adjudication_analysis_pmcvqa.py"),
    ("k=2 latent-knowledge control", "src/k2_control.py"),
    ("ScienceQA regex-variant table (supplement)", "src/scienceqa_regex_variants.py"),
    ("Reasoning-model tie-break gap + clustered CI", "src/reasoning_clustered_ci.py"),
    ("Figures 2 and 3", "src/make_paper_figs.py"),
]


def main():
    for title, script in STEPS:
        print("\n" + "=" * 74)
        print(title)
        print("=" * 74)
        r = subprocess.run([sys.executable, script])
        if r.returncode != 0:
            sys.exit(f"\nFAILED: {script} (exit {r.returncode})")
    print("\n" + "=" * 74)
    print("Tier-1 reproduction complete. All outputs in results/.")
    print("=" * 74)


if __name__ == "__main__":
    main()
