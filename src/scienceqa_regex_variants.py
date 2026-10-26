"""ScienceQA regex-variant table (supplement, "One character of regex").

ScienceQA's released scorer extracts with re.compile(r'The answer is ([A-Z]).') and falls back to
random.choice on failure. The models overwhelmingly answer in the near-miss form `The answer is (X).`.
This script recomputes, per model, from the released per-item logs:

  official  -- the stored official predictions (regex + seeded random fallback), as released;
  widened   -- accept both spellings via  (?:[Tt]he answer is:?\\s*\\(?([A-Z])\\)?),  first match,
               ABSTAINING on the residual failures (this is the paper's conservative variant);
  E[random] -- the widened parse with ScienceQA's own random fallback kept instead of abstention,
               reported in EXPECTATION (each residual failure contributes 1/k), i.e. the
               like-for-like comparison at the official fallback. No seeds involved.

The paper reports `widened` (conservative: the pure-regex effect under random fallback is LARGER).
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
WIDE = re.compile(r"[Tt]he answer is:?\s*\(?([A-Z])\)?")


def variants(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n = len(rows)
    official = sum(r["sqa_pred"] == r["gold"] for r in rows) / n
    widened_hits = exp_random = 0.0
    for r in rows:
        m = WIDE.search(r.get("resp", "") or "")
        pred = (ord(m.group(1)) - 65) if m else None
        if pred is not None:
            hit = float(pred == r["gold"])
            widened_hits += hit
            exp_random += hit
        else:
            exp_random += 1.0 / r["k"]          # official random fallback, in expectation
    return official, widened_hits / n, exp_random / n, n


def main():
    print(f"{'model':16s} {'official':>9s} {'widened(abstain)':>17s} {'E[widened+random]':>18s}")
    out = {}
    for f in sorted(glob.glob(str(DATA / "scienceqa_*.jsonl"))):
        m = Path(f).stem.replace("scienceqa_", "")
        off, wid, exp, n = variants(f)
        out[m] = (off, wid, exp)
        print(f"{m:16s} {off:9.4f} {wid:17.4f} {exp:18.4f}   (n={n})")
    return out


if __name__ == "__main__":
    main()
