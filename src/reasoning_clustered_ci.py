"""Reasoning-model tie-break gap with its category-clustered CI (Sec. 5.3(v)).

MMAD items are grouped by MVTec category and the first-vs-last tie-break gap varies strongly across
categories, so an item-level interval would be roughly 3x too narrow. This script recomputes, from the
released log (data/mmad_qwen3-vl-8b-thinking_budget1024.jsonl) and the released category mapping
(data/mmad_mvtec_categories.json, item-aligned, join verified against the gold sequence at build time):

  - first-match / last-match accuracy and the gap (the paper's 0.689 / 0.784 / 9.5 points);
  - per-category gaps (the paper's range 0.000 to 0.237 over 15 categories);
  - the seeded cluster-bootstrap 95% CI (the paper's [6.4, 13.1]); seed fixed for determinism.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
PAT = re.compile(r"\b([A-E])\b")


def main():
    recs = [json.loads(l) for l in open(DATA / "mmad_qwen3-vl-8b-thinking_budget1024.jsonl") if l.strip()]
    cats = json.load(open(DATA / "mmad_mvtec_categories.json"))
    assert len(recs) == len(cats) == 1670, (len(recs), len(cats))

    per = {}
    first = last = 0
    for c, r in zip(cats, recs):
        ms = PAT.findall(r["resp"] or "")
        f, l = (ms[0], ms[-1]) if ms else (r["hard_pred"], r["hard_pred"])
        first += f == r["gold"]
        last += l == r["gold"]
        per.setdefault(c, []).append((f == r["gold"], l == r["gold"]))
    n = len(recs)
    gap = (last - first) / n
    catgaps = {c: (sum(b for _, b in v) - sum(a for a, _ in v)) / len(v) for c, v in per.items()}

    ks = sorted(per)
    rng = random.Random(20260810)
    boots = []
    for _ in range(5000):
        samp = [per[rng.choice(ks)] for _ in ks]
        items = [x for g in samp for x in g]
        boots.append(sum(b for _, b in items) / len(items) - sum(a for a, _ in items) / len(items))
    boots.sort()
    lo, hi = boots[125], boots[4875]

    print(f"first-match acc : {first / n:.4f}")
    print(f"last-match  acc : {last / n:.4f}")
    print(f"gap             : {gap:.4f}  ({100 * gap:.1f} points)")
    print(f"categories      : {len(ks)}   per-category gap range: "
          f"{min(catgaps.values()):.3f} .. {max(catgaps.values()):.3f}")
    print(f"cluster bootstrap 95% CI (seed 20260810): [{100 * lo:.1f}, {100 * hi:.1f}] points")
    return gap, catgaps, (lo, hi)


if __name__ == "__main__":
    main()
