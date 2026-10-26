"""Lead figure: artifact (accuracy a fabricating fallback grants) vs non-compliance (how often the
fallback is invoked), one point per benchmark x model. Source-derived; re-run any time.

Three visual cohorts:
  * FIRING (fabricating scorers): MMAD/OmniMedVQA/PMC-VQA (snap), MMMU/ScienceQA (random), POPE
    (default-class). They lie on a positive trend -> the difficulty/non-compliance scaling law.
  * DEFUSED-BY-DESIGN: MathVision (abstains) -> sits at HIGH non-compliance but ~0 artifact (OFF the
    trend, the contrast point); GMAI-MMBench-style sentinel would land here too.
  * BUFFERED: MathVista has the same no-threshold snap as MMAD, but its GPT/local extraction step
    collapses effective non-compliance ~0.46 -> ~0.03 and the artifact ~0.10 -> ~0.01. Drawn as arrows
    from the (no-buffer counterfactual) firing position down to the (with-buffer actual) position.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path("data")
OFFICIAL = Path("research/official_scorers")
MODELS = ["gemma-4-e4b", "qwen2.5-vl-7b", "qwen3-vl-8b", "pixtral-12b", "llama-3.2-11b"]

STYLE = {  # benchmark -> (color, marker, label)
    "mmad":       ("#d62728", "o", "MMAD (snap, ICLR'25)"),
    "mmmu":       ("#ff7f0e", "s", "MMMU (random, CVPR'24)"),
    "scienceqa":  ("#2ca02c", "^", "ScienceQA (random, NeurIPS'22)"),
    "pmcvqa":     ("#9467bd", "D", "PMC-VQA (snap, Comm.Med.'24)"),
    "omnimedvqa": ("#8c564b", "v", "OmniMedVQA (snap, CVPR'24)"),
    "pope":       ("#7f7f7f", "P", "POPE (default, EMNLP'23)"),
    "mathvision": ("#1f77b4", "X", "MathVision DEFUSED (abstains, NeurIPS'24)"),
    "mathvista":  ("#17becf", "*", "MathVista BUFFERED (extract+snap, ICLR'24)"),
}


def load_unified():
    p = HERE / "unified_artifacts.json"
    return json.loads(p.read_text()) if p.exists() else []


def mathvista_arrows():
    """Per-model: ((raw_noncompl, raw_artifact)  ->  (extract_noncompl, extract_artifact))."""
    raw = {}
    for f in glob.glob(str(HERE / "mathvista_*.jsonl")):
        d = [json.loads(l) for l in open(f) if l.strip()]
        if not d:
            continue
        m = Path(f).stem.replace("mathvista_", "")
        n = len(d)
        noncompl = sum(r["abstain"] for r in d) / n
        artifact = sum(r["fuzzy_ok"] - r["clean_ok"] for r in d) / n
        raw[m] = (noncompl, artifact)
    ext = {}
    sf = HERE / "summary_mathvista_local.jsonl"
    if sf.exists():
        for l in open(sf):
            s = json.loads(l)
            ext[s["test_model"]] = (s["snap_fire_rate"], s["artifact"])
    return [(raw[m], ext[m]) for m in raw if m in ext]


def main():
    rows = load_unified()
    fig, ax = plt.subplots(figsize=(8.2, 6.0))

    # firing-cohort trend line + r
    fire = [r for r in rows if r["cohort"] == "firing"]
    if len(fire) >= 3:
        xs = [r["non_compliance"] for r in fire]
        ys = [r["artifact"] for r in fire]
        mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        a = my - b * mx
        vx = sum((x - mx) ** 2 for x in xs) ** 0.5
        vy = sum((y - my) ** 2 for y in ys) ** 0.5
        r = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (vx * vy)
        xlo, xhi = 0, max(xs) * 1.05 + 0.02
        ax.plot([xlo, xhi], [a + b * xlo, a + b * xhi], "--", color="0.4", lw=1.3, zorder=1,
                label=f"firing trend (r={r:.2f}, n={len(fire)})")

    # scatter all cohorts
    drawn = set()
    for r in rows:
        b = r["benchmark"]
        color, marker, lab = STYLE.get(b, ("k", "o", b))
        ax.scatter(r["non_compliance"], r["artifact"], c=color, marker=marker, s=70,
                   edgecolors="k", linewidths=0.4, alpha=0.9, zorder=3,
                   label=lab if b not in drawn else None)
        drawn.add(b)

    # MathVista buffer arrows: counterfactual (no buffer) -> actual (with buffer)
    arrows = mathvista_arrows()
    for (rx, ry), (ex, ey) in arrows:
        ax.annotate("", xy=(ex, ey), xytext=(rx, ry),
                    arrowprops=dict(arrowstyle="->", color="#17becf", lw=1.2, alpha=0.8), zorder=2)
        ax.scatter(rx, ry, facecolors="none", edgecolors="#17becf", marker="*", s=120,
                   linewidths=1.0, zorder=2)  # hollow = counterfactual (no buffer)

    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    ax.set_xlabel("non-compliance rate  (fraction of outputs where the fallback is invoked)")
    ax.set_ylabel("artifact  (accuracy the fabricating fallback adds vs abstaining)")
    ax.set_title("Verdict-fabricating fallbacks: the artifact scales with non-compliance\n"
                 "(firing cohort on the trend; MathVision abstains -> off-trend; MathVista's buffer drags it down)")
    ax.legend(fontsize=7.2, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = HERE / "fig_artifact_vs_noncompliance.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}  ({len(rows)} cells, {len(arrows)} MathVista buffer arrows)")


if __name__ == "__main__":
    main()
