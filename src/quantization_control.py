"""The 8-bit quantization control (supplement, "An 8-bit quantization control").

The audited runs load models 4-bit (bitsandbytes NF4), the working precision of the 16GB
single-GPU regime the paper scopes to. This control reruns the identical MMAD protocol
(same 1,670 items in the same order, official prompt, greedy decoding, budget 16) changing
only the load precision to 8-bit, for one compliant model and the headline rambler, and
recomputes from the released logs:

  non-compliance -- fraction of responses with no parseable option letter (strict parse fails);
  snap / abstain -- accuracy under the no-threshold closest-option snap / under abstention;
  artifact       -- snap minus abstain, the parse-failure fallback's contribution.

Reading: compliance is a property of the model, not of the compression. The compliant model
is compliant at both precisions (artifact exactly 0); the rambler's non-compliance RISES at
8-bit and the spread widens, with a large share of its 8-bit failures answering in words
(bare "Yes."/"No."), which the letter parser cannot credit.
"""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
PAIRS = [
    ("gemma-4-e4b", "mmad_gemma-4-e4b.jsonl", "mmad_gemma-4-e4b_8bit.jsonl"),
    ("llama-3.2-11b", "mmad_llama-3.2-11b.jsonl", "mmad_llama-3.2-11b_8bit.jsonl"),
]


def stats(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n = len(rows)
    nc = sum(1 for r in rows if r["strict_pred"] is None) / n
    snap = sum(r["hok"] for r in rows) / n
    abst = sum(r["stok"] for r in rows) / n
    return rows, nc, snap, abst


def main():
    print(f"{'model':15s} {'prec':>5s} {'non-comp':>9s} {'snap':>7s} {'abstain':>8s} {'artifact':>9s}")
    out = {}
    for model, f4, f8 in PAIRS:
        r4, nc4, s4, a4 = stats(DATA / f4)
        r8, nc8, s8, a8 = stats(DATA / f8)
        assert len(r4) == len(r8) == 1670, (model, len(r4), len(r8))
        assert all(x["gold"] == y["gold"] for x, y in zip(r4, r8)), f"run-order mismatch: {model}"
        for prec, nc, s, a in [("4-bit", nc4, s4, a4), ("8-bit", nc8, s8, a8)]:
            print(f"{model:15s} {prec:>5s} {nc:9.4f} {s:7.4f} {a:8.4f} {s - a:+9.4f}")
            out[(model, prec)] = (nc, s, a)
        if model == "llama-3.2-11b":
            ncr = [r for r in r8 if r["strict_pred"] is None]
            words = sum(1 for r in ncr if r["resp"].strip().rstrip(".").lower() in ("yes", "no"))
            print(f"\n8-bit non-compliant answering in words (bare Yes/No): {words} / {len(ncr)}")
    return out


if __name__ == "__main__":
    main()
