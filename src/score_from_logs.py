"""Tier-1 reproduction (no GPU, no dataset download): recompute every benchmark's accuracy under
each parse-failure scoring rule directly from the released per-item logs, and report the artifact
(snap minus abstain) for each model. Reproduces the magnitudes in the paper's results section and the
inputs to Fig. 2 / Fig. 3.

Each released log record stores the gold answer and the verdict of each rule on that item (the rules
were executed at inference time and their per-item output stored), so accuracy is recomputed here as a
plain aggregate -- no model and no benchmark dataset are needed.

Run:  python src/score_from_logs.py            # prints the table, writes results/artifact_table.csv
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

LOGS = Path("data")
OUT = Path("results")
MODELS = ["gemma-4-e4b", "qwen2.5-vl-7b", "qwen3-vl-8b", "pixtral-12b", "llama-3.2-11b"]

# For each benchmark and rule: ('pred', field) -> acc = mean(rec[field] == gold);
#                              ('ok',   field) -> acc = mean(rec[field]).
SPECS = {
    "mmad":       {"gold": "gold", "snap": ("pred", "hard_pred"), "abstain": ("pred", "strict_pred"),
                   "likelihood": ("pred", "soft_pred")},
    "mmmu":       {"gold": "gold", "snap": ("pred", "mmmu"), "abstain": ("pred", "strict")},
    "omnimedvqa": {"gold": "gt", "snap": ("ok", "fuzzy_ok"), "abstain": ("ok", "clean_ok")},
    "pmcvqa":     {"gold": "gold", "snap": ("pred", "snap_pred"), "abstain": ("pred", "clean_pred")},
    "pope":       {"gold": "gold", "snap": ("pred", "pope"), "abstain": ("pred", "strict"),
                   "likelihood": ("pred", "logit")},
    "scienceqa":  {"gold": "gold", "snap": ("pred", "sqa_pred"), "abstain": ("pred", "strict")},
    "mathvista":  {"gold": "gt", "snap": ("ok", "fuzzy_ok"), "abstain": ("ok", "clean_ok")},
    "mathvision": {"gold": "gold", "snap": ("ok", "fab_ok"), "abstain": ("ok", "abstain_ok")},
}


def acc(records, gold_field, rule):
    kind, field = rule
    n = len(records)
    if not n:
        return None
    if kind == "ok":
        return sum(bool(r.get(field)) for r in records) / n
    return sum(r.get(field) == r.get(gold_field) for r in records) / n


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for bench, spec in SPECS.items():
        for model in MODELS:
            p = LOGS / f"{bench}_{model}.jsonl"
            if not p.exists():
                continue
            recs = [json.loads(l) for l in open(p) if l.strip()]
            g = spec["gold"]
            snap = acc(recs, g, spec["snap"])
            abst = acc(recs, g, spec["abstain"])
            lik = acc(recs, g, spec["likelihood"]) if "likelihood" in spec else None
            rows.append({"benchmark": bench, "model": model, "n": len(recs),
                         "snap": round(snap, 4), "abstain": round(abst, 4),
                         "likelihood": round(lik, 4) if lik is not None else "",
                         "artifact_snap_minus_abstain": round(snap - abst, 4)})

    with open(OUT / "artifact_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    hdr = f"{'benchmark':12s} {'model':16s} {'n':>5s} {'snap':>7s} {'abstain':>8s} {'likeli':>7s} {'artifact':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        lik = f"{r['likelihood']:.3f}" if r["likelihood"] != "" else "   -  "
        print(f"{r['benchmark']:12s} {r['model']:16s} {r['n']:5d} {r['snap']:7.3f} {r['abstain']:8.3f} "
              f"{lik:>7s} {r['artifact_snap_minus_abstain']:+9.4f}")
    print(f"\nwrote {OUT/'artifact_table.csv'} ({len(rows)} cells)")
    return rows


if __name__ == "__main__":
    main()
