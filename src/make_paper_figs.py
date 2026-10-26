"""Build the two empirical hero figures for the WACV paper from raw rescore + adjudication data:
  fig2_rule_swing.png      -- MMAD accuracy under snap/abstain/likelihood, 5 models (the money plot)
  fig3_bidirectional.png   -- human-anchored over- vs under-credit, MMAD + PMC-VQA (the bidirectional plot)
Run from the repo root: python data/make_paper_figs.py
"""
from __future__ import annotations
import json, collections
from pathlib import Path

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType, not Type 3 (IEEE-safe)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = Path("data")
OUT = Path("results")
CANON = {"N": "NONE", "M": "MULTIPLE"}
PAPER_MODELS = ["gemma-4-e4b", "qwen2.5-vl-7b", "qwen3-vl-8b", "pixtral-12b", "llama-3.2-11b"]
DISP = {"gemma-4-e4b": "Gemma-4\nE4B", "qwen2.5-vl-7b": "Qwen2.5-VL\n7B",
        "qwen3-vl-8b": "Qwen3-VL\n8B", "pixtral-12b": "Pixtral\n12B", "llama-3.2-11b": "Llama-3.2\n11B"}

C_SNAP, C_ABS, C_LIK = "#d1495b", "#3a7ca5", "#5b8c5a"  # snap=red, abstain=blue, likelihood=green


def load_labels(name):
    p = HERE / name
    out = {}
    if p.exists():
        for l in open(p):
            if l.strip():
                d = json.loads(l)
                out[d["uid"]] = CANON.get(d["label"], d["label"])
    return out


# ---------- FIGURE 2: the rule swing ----------
def fig_rule_swing():
    # summary.jsonl is append-only and holds MULTIPLE datasets (mvtec/visa) per model, so a
    # {model: row} dict silently keeps whichever ran LAST. Filter to the MMAD (mvtec) rows the
    # caption claims, and assert against the per-item logs so this cannot drift again.
    rows = {}
    for l in open(HERE / "summary.jsonl"):
        if not l.strip():
            continue
        d = json.loads(l)
        if d.get("dataset") not in (None, "mvtec") or d.get("n") != 1670 or d.get("suffix"):
            continue
        rows[d["model"]] = d
    for m in PAPER_MODELS:
        assert m in rows, f"no full-set MMAD row for {m}"
        recs = [json.loads(x) for x in open(HERE / f"mmad_{m}.jsonl") if x.strip()]
        truth = sum(r["hard_pred"] == r["gold"] for r in recs) / len(recs)
        assert abs(rows[m]["mmad_fuzzy_acc"] - truth) < 1e-4, (
            f"{m}: summary {rows[m]['mmad_fuzzy_acc']} != per-item log {truth:.4f}")
    snap = [rows[m]["mmad_fuzzy_acc"] for m in PAPER_MODELS]
    abst = [rows[m]["strict_acc"] for m in PAPER_MODELS]
    lik = [rows[m]["soft_logit_acc"] for m in PAPER_MODELS]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = range(len(PAPER_MODELS)); w = 0.26
    ax.bar([i - w for i in x], snap, w, label="No-threshold snap (official)", color=C_SNAP, edgecolor="black", linewidth=0.5)
    ax.bar(list(x), abst, w, label="Abstain (letter-only)", color=C_ABS, edgecolor="black", linewidth=0.5, hatch="///")
    ax.bar([i + w for i in x], lik, w, label="Likelihood (parse-free)", color=C_LIK, edgecolor="black", linewidth=0.5, hatch="xxx")

    # annotate the swing on the two non-compliant models
    for i, m in enumerate(PAPER_MODELS):
        gap = snap[i] - abst[i]
        if gap > 0.02:
            top = max(snap[i], lik[i]) + 0.03
            ax.annotate("", xy=(i - w, snap[i] + 0.012), xytext=(i, abst[i] + 0.012),
                        arrowprops=dict(arrowstyle="<->", color="black", lw=1.1))
            ax.text(i - w / 2, top, f"+{gap*100:.1f} pts\nfrom the rule", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")

    # bracket over the three compliant models
    ax.plot([-0.4, 2.4], [0.90, 0.90], color="gray", lw=1.0)
    ax.text(1.0, 0.915, "compliant models: all rules coincide (gap $=0$)", ha="center", va="bottom",
            fontsize=8.5, color="gray", style="italic")

    ax.set_xticks(list(x)); ax.set_xticklabels([DISP[m] for m in PAPER_MODELS], fontsize=8.5)
    ax.set_ylabel("MMAD accuracy (identical model outputs)", fontsize=9.5)
    ax.set_ylim(0, 1.0); ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    ax.legend(loc="lower left", fontsize=8.3, framealpha=0.95, ncol=1)
    ax.set_title("The scoring rule alone moves the score by up to 22 points",
                 fontsize=10.5, fontweight="bold", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_rule_swing.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig2_rule_swing.png", dpi=200, bbox_inches="tight")
    print("wrote fig2_rule_swing.{pdf,png}", {m: round(snap[i]-abst[i], 3) for i, m in enumerate(PAPER_MODELS)})


# ---------- FIGURE 3: bidirectional over/under-credit ----------
def over_under(manifest, human, judge):
    man = {json.loads(l)["uid"]: json.loads(l) for l in open(HERE / manifest)}
    ref = {**judge, **human}
    res = {}
    for model in ("pixtral-12b", "llama-3.2-11b"):
        items = [m for u, m in man.items()
                 if m["model"] == model and m["uid"] in ref and m.get("_noncompliant", True)]
        n = len(items)
        if not n:
            continue
        over = sum(m["_snap_pred"] == m["_gold"] and ref[m["uid"]] != m["_gold"] for m in items) / n
        under = sum(m["_snap_pred"] != m["_gold"] and ref[m["uid"]] == m["_gold"] for m in items) / n
        res[model] = (over, under)
    return res


def fig_bidirectional():
    # signed accuracy artifact (snap - abstain) on the two HUMAN-ANCHORED benchmarks
    # Fig. 3 shows each official scorer's TOTAL effect (scorer - clean parse). For MMAD that is
    # the fallback alone; for PMC-VQA, whose matcher is unconditional, it is fallback + override.
    # unified_artifacts.json stores only the FALLBACK term, so recompute the total from the logs.
    def total(bench, model, snap, clean):
        recs = [json.loads(x) for x in open(HERE / f"{bench}_{model}.jsonl") if x.strip()]
        n = len(recs)
        return (sum(r[snap] == r["gold"] for r in recs)
                - sum(r[clean] == r["gold"] for r in recs)) / n
    uni = {}
    for m in ("pixtral-12b", "llama-3.2-11b"):
        uni[("mmad", m)] = total("mmad", m, "hard_pred", "strict_pred")
        uni[("pmcvqa", m)] = total("pmcvqa", m, "snap_pred", "clean_pred")
    groups = [("MMAD\nPixtral-12B", uni[("mmad", "pixtral-12b")]),
              ("MMAD\nLlama-3.2-11B", uni[("mmad", "llama-3.2-11b")]),
              ("PMC-VQA\nPixtral-12B", uni[("pmcvqa", "pixtral-12b")]),
              ("PMC-VQA\nLlama-3.2-11B", uni[("pmcvqa", "llama-3.2-11b")])]
    labels = [g[0] for g in groups]
    vals = [g[1] for g in groups]

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    x = list(range(len(groups)))
    colors = [C_SNAP if v >= 0 else C_ABS for v in vals]
    bars = ax.bar(x, vals, 0.58, color=colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        b.set_hatch("" if v >= 0 else "////")
    ax.axhline(0, color="black", lw=0.9)

    for i, v in enumerate(vals):
        ax.text(i, v + (0.011 if v >= 0 else -0.011), f"{'+' if v >= 0 else '−'}{abs(v)*100:.1f} pts",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9.5, fontweight="bold")

    ax.annotate("Same rule, opposite sign. On PMC-VQA the snap\nmarks wrong the wordy-but-correct medical\nanswers Llama actually gave.",
                xy=(2.68, -0.135), xytext=(0.12, -0.085), fontsize=8.8, color=C_ABS, fontweight="bold",
                ha="left", va="top", arrowprops=dict(arrowstyle="->", color=C_ABS, lw=1.4))

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.8)
    ax.set_ylabel("accuracy artifact: snap $-$ abstain\n(percentage points, same outputs)", fontsize=9.5)
    ax.set_ylim(-0.25, 0.27)
    ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2]); ax.set_yticklabels(["−20", "−10", "0", "+10", "+20"])
    ax.set_title("The same no-threshold snap inflates one model and deflates another",
                 fontsize=10.5, fontweight="bold", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_bidirectional.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig3_bidirectional.png", dpi=200, bbox_inches="tight")
    print("wrote fig3_bidirectional.{pdf,png}", dict(zip(labels, vals)))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_rule_swing()
    fig_bidirectional()
