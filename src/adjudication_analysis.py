"""Anchor the scoring rules to what the model ACTUALLY answered (human label if available, else the
blinded Opus judge) and decompose MMAD's snap-vs-abstain gap on non-compliant items into:
  (a) abstain-harshness: the model DID answer (in text), the snap read it, but letter-only abstain
      threw it away  -- this is NOT fabrication, it's our abstain reference being too strict; and
  (b) real fabrication: the model did NOT commit (judge=NONE) yet the snap still scored it correct
      -- pure over-credit.
Also reports judge-vs-human agreement on the overlap, so the Opus labels can be trusted (or not)."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path("data")


_CANON = {"N": "NONE", "M": "MULTIPLE"}   # human tool uses short keys; judge uses long form


def load(name):
    p = HERE / name
    if not p.exists():
        return {}
    out = {}
    for l in open(p):
        if l.strip():
            d = json.loads(l)
            out[d["uid"]] = _CANON.get(d["label"], d["label"])
    return out


def main():
    man = {json.loads(l)["uid"]: json.loads(l) for l in open(HERE / "adjudication_manifest.jsonl")}
    judge, human = load("judge_labels.jsonl"), load("human_labels.jsonl")

    overlap = [u for u in judge if u in human]
    if overlap:
        agree = sum(judge[u] == human[u] for u in overlap)
        print(f"judge-vs-human agreement: {agree}/{len(overlap)} = {agree/len(overlap):.3f}\n")
    ref, refname = (human, "HUMAN") if human else (judge, "OPUS-JUDGE (provisional)")
    print(f"reference = {refname} ({len(ref)} labels)\n")

    nc = [man[u] for u in ref if u in man and man[u]["_noncompliant"]]
    cc = [man[u] for u in ref if u in man and not man[u]["_noncompliant"]]
    if not nc:
        print("no labeled non-compliant items yet -- run adjudicate.py / judge first"); return

    def lab(m):
        return ref[m["uid"]]

    n = len(nc)
    committed = sum(lab(m) not in ("NONE", "MULTIPLE") for m in nc)
    model_gold = sum(lab(m) == m["_gold"] for m in nc)                       # model truly answered gold
    snap_ok = sum(m["_snap_pred"] == m["_gold"] for m in nc)                 # snap scored correct
    snap_legit = sum(m["_snap_pred"] == m["_gold"] and lab(m) == m["_gold"] for m in nc)
    snap_fab = snap_ok - snap_legit                                          # snap right but model didn't answer gold

    print(f"On {n} NON-COMPLIANT MMAD items (snap fired; letter-only abstain = wrong):")
    print(f"  model actually COMMITTED to an option : {committed}/{n} = {committed/n:.2f}   (genuine NONE/MULTIPLE: {n-committed})")
    print(f"  model actually answered the GOLD option: {model_gold}/{n} = {model_gold/n:.2f}")
    print(f"  the SNAP scored these correct          : {snap_ok}/{n} = {snap_ok/n:.2f}")
    print(f"      |- LEGIT  (model truly answered gold in text; snap read it; abstain too harsh): {snap_legit}  ({snap_legit/n:.2f})")
    print(f"      |- FABRIC (model did NOT answer gold; snap guessed right = over-credit)        : {snap_fab}  ({snap_fab/n:.2f})")
    if cc:
        cc_snap = sum(m["_snap_pred"] == m["_gold"] for m in cc)
        print(f"  [compliant controls: {len(cc)}, snap acc {cc_snap/len(cc):.2f} -- snap==abstain here, no gap]")

    # Scale to the benchmark: +0.21 gap = non_compliance_rate x snap_correct_rate, split legit vs fabric.
    print(f"\nDecomposition of MMAD's snap>abstain credit ON the non-compliant set:")
    print(f"  fraction that is ABSTAIN-HARSHNESS (legit text-answers): {snap_legit/snap_ok:.0%}" if snap_ok else "  (no snap-correct)")
    print(f"  fraction that is REAL FABRICATION  (over-credit on NONE): {snap_fab/snap_ok:.0%}" if snap_ok else "")
    print("  (the per-benchmark +0.21 = non_compliance_rate x snap_correct_rate; multiply each share above)")
    if not human:
        print("\nNOTE: provisional Opus-judge gold. Run adjudicate.py to add human labels + validate agreement.")


if __name__ == "__main__":
    main()
