"""PMC-VQA (k=4 medical) adjudication analysis: certify the Opus judge against the human audit
(Cohen's kappa) and report the bidirectional over/under-credit decomposition of the no-threshold snap.
Anchors to the human label where audited, else the provisional Opus judge."""
from __future__ import annotations

import collections
import json
from pathlib import Path

HERE = Path("data")
CANON = {"N": "NONE", "M": "MULTIPLE"}


def load(name):
    p = HERE / name
    if not p.exists():
        return {}
    out = {}
    for l in open(p):
        if l.strip():
            d = json.loads(l)
            out[d["uid"]] = CANON.get(d["label"], d["label"])
    return out


def main():
    man = {json.loads(l)["uid"]: json.loads(l) for l in open(HERE / "adjudication_manifest_pmcvqa.jsonl")}
    human, judge = load("human_labels_pmcvqa.jsonl"), load("judge_labels_pmcvqa.jsonl")

    overlap = [u for u in human if u in judge]
    if overlap:
        po = sum(human[u] == judge[u] for u in overlap) / len(overlap)
        labs = set(human[u] for u in overlap) | set(judge[u] for u in overlap)
        hc = collections.Counter(human[u] for u in overlap)
        jc = collections.Counter(judge[u] for u in overlap)
        N = len(overlap)
        pe = sum((hc[l] / N) * (jc[l] / N) for l in labs)
        kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
        print(f"judge-vs-human (PMC-VQA, k=4 medical): {sum(human[u]==judge[u] for u in overlap)}/{N} "
              f"= {po:.3f}, Cohen's kappa={kappa:.3f}")
        for u in overlap:
            if human[u] != judge[u]:
                print(f"   disagree {u}: human={human[u]} judge={judge[u]}  {man[u]['response'][:55]!r}")
    else:
        print("no human PMC-VQA labels yet -- run the audit (adjudicate.py --manifest ...pmcvqa...).")

    ref = {**judge, **human}  # human overrides judge where audited
    print(f"\nover/under-credit (ref = HUMAN where audited [{len(human)}], else Opus-judge):")
    for model in ("pixtral-12b", "llama-3.2-11b"):
        items = [m for u, m in man.items() if m["model"] == model and m["uid"] in ref]
        n = len(items)
        if not n:
            continue
        over = sum(m["_snap_pred"] == m["_gold"] and ref[m["uid"]] != m["_gold"] for m in items)
        under = sum(m["_snap_pred"] != m["_gold"] and ref[m["uid"]] == m["_gold"] for m in items)
        ansg = sum(ref[m["uid"]] == m["_gold"] for m in items)
        print(f"  {model} (n={n}): model answered gold {ansg}/{n}; "
              f"OVER-credit {over}/{n}={over/n:.2f}  UNDER-credit/deflation {under}/{n}={under/n:.2f}")


if __name__ == "__main__":
    main()
