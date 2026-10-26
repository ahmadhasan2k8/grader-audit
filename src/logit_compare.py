"""Robust-method comparison: for each firing benchmark x model, put three scoring rules side by side
on the SAME model outputs / items:

  buggy_acc   -- the benchmark's official FABRICATING scorer (no-threshold snap / random / default)
  abstain_acc -- the robust ABSTAIN rule (mark parse-failures wrong); = the 'strict/clean' column
  logit_acc   -- the robust PARSE-FREE likelihood scorer (rescore_logit.py); no fallback at all

The story: on COMPLIANT models the three agree; on NON-COMPLIANT ones the buggy rule pulls AHEAD of
abstain (that gap is the fabrication artifact), while logit gives an independent parse-free number that
cannot be inflated by a fallback. inflation = buggy_acc - abstain_acc.

Sources (dedup last-per-model): MMAD <- REPRODUCED_NUMBERS.json (official_fuzzy/official_strict),
MMMU/OmniMedVQA/ScienceQA <- their summary_*.jsonl, logit <- summary_logit.jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path("data")
OFFICIAL = Path("research/official_scorers")
MODELS = ["gemma-4-e4b", "qwen2.5-vl-7b", "qwen3-vl-8b", "pixtral-12b", "llama-3.2-11b"]


def last_per_model(path, key="model"):
    out = {}
    p = Path(path)
    if p.exists():
        for l in open(p):
            if l.strip():
                s = json.loads(l); out[s[key]] = s
    return out


def main():
    buggy = {}    # (bench, model) -> buggy_acc
    abstain = {}  # (bench, model) -> abstain_acc
    # MMAD (official reproduced numbers)
    rn = OFFICIAL / "REPRODUCED_NUMBERS.json"
    if rn.exists():
        for m, d in json.loads(rn.read_text())["MMAD"]["per_model"].items():
            buggy[("mmad", m)] = d["official_fuzzy"]; abstain[("mmad", m)] = d["official_strict"]
    # MMMU (parse+random): abstain holds the OFFICIAL parser fixed = buggy - clean fabrication artifact
    # (fired_correct/N), so inflation == the headline artifact, NOT mmmu_acc - a stricter parser's recall.
    for m, s in last_per_model(HERE / "summary_mmmu.jsonl").items():
        art = s["fired_correct"] / s["n"]
        buggy[("mmmu", m)] = s["mmmu_acc"]; abstain[("mmmu", m)] = round(s["mmmu_acc"] - art, 4)
    # OmniMedVQA
    for m, s in last_per_model(HERE / "summary_omnimedvqa.jsonl").items():
        buggy[("omnimedvqa", m)] = s["fuzzy_acc"]; abstain[("omnimedvqa", m)] = s["clean_acc"]
    # ScienceQA (768-token clean run; parse+random): abstain = buggy - clean fabrication artifact
    for m, s in last_per_model(HERE / "summary_scienceqa.jsonl").items():
        art = s.get("artifact_fabrication", s["fired_correct"] / s["n"])
        buggy[("scienceqa", m)] = s["sqa_acc"]; abstain[("scienceqa", m)] = round(s["sqa_acc"] - art, 4)
    # logit (parse-free robust)
    logit = {}
    for l in (open(HERE / "summary_logit.jsonl") if (HERE / "summary_logit.jsonl").exists() else []):
        if l.strip():
            s = json.loads(l); logit[(s["benchmark"], s["model"])] = s["logit_acc"]

    benches = ["mmad", "mmmu", "omnimedvqa", "scienceqa"]
    print(f"{'benchmark':11s} {'model':14s} {'buggy':>7s} {'abstain':>8s} {'logit':>7s} {'inflation':>10s}")
    rows = []
    for b in benches:
        for m in MODELS:
            if (b, m) not in buggy:
                continue
            bu = buggy[(b, m)]; ab = abstain[(b, m)]; lo = logit.get((b, m))
            infl = bu - ab
            rows.append((b, m, bu, ab, lo, infl))
            los = f"{lo:7.4f}" if lo is not None else "   --  "
            flag = "  <== inflation" if infl >= 0.05 else ""
            print(f"{b:11s} {m:14s} {bu:7.4f} {ab:8.4f} {los} {infl:+10.4f}{flag}")
        print()
    # headline: mean inflation the fabricating rule grants, and whether logit avoids it
    big = [r for r in rows if r[5] >= 0.05]
    print(f"cells with >=5pt fabrication inflation: {len(big)}  "
          f"(these are where the buggy rule beats abstain; logit is the parse-free alternative)")
    if (HERE / "summary_logit.jsonl").exists():
        have = [r for r in rows if r[4] is not None]
        print(f"logit available for {len(have)}/{len(rows)} cells")
    else:
        print("logit data PENDING (summary_logit.jsonl not yet written)")


if __name__ == "__main__":
    main()
