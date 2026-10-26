"""Cross-benchmark answer-extraction taxonomy. Re-scores the SAME 5-model generations under each
PUBLISHED benchmark's exact released rule, against a parser-free logit reference, to isolate the
scoring artifact and its direction:

  MMAD (ICLR'25)      MCQ regex + get_close_matches(cutoff=0.0)  -> FABRICATE (majority "Yes")
  Anomaly-OV (CVPR'25) 'Yes'/'yes' substring in response          -> SUBSTRING-MISFIRE (collapse to normal)
  M3-AD (preprint)    strict <answer>tag</answer>, else abstain   -> DISCARD (recoverable answer thrown away)

Artifact is reported per model; the signature is artifact~0 on format-compliant models and large
(in the benchmark's characteristic direction) on non-compliant ones.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("data")
MODELS = ["gemma-4-e4b", "qwen2.5-vl-7b", "qwen3-vl-8b", "pixtral-12b", "llama-3.2-11b"]
COMPLIANT = {"gemma-4-e4b", "qwen2.5-vl-7b", "qwen3-vl-8b"}


def jl(name):
    p = OUT / name
    return [json.loads(l) for l in open(p)] if p.exists() else None


def mmad(model, suf):  # released = fuzzy(cutoff=0.0); honest text rule = strict(abstain); + logit ref
    r = jl(f"{model}{suf}.jsonl")
    if not r:
        return None
    n = len(r)
    fuzzy = sum(x["hok"] for x in r) / n
    strict = sum(x.get("stok", x["hok"]) for x in r) / n
    logit = sum(x["sok"] for x in r) / n
    fire = sum(1 for x in r if x.get("strict_pred") is None) / n
    return {"released": fuzzy, "honest": strict, "logit": logit, "artifact": fuzzy - strict, "trigger%": fire}


def anomalyov(model, suf):  # released = 'Yes'/'yes' substring; honest ref = logit
    r = jl(f"yesno_{model}{suf}.jsonl")
    if not r:
        return None
    n = len(r); correct = 0; pred_def = 0; logit_def = 0
    for x in r:
        g = int(x["gold_defect"]); resp = x.get("resp", "")
        p = 1 if ("Yes" in resp or "yes" in resp) else 0
        correct += int(p == g); pred_def += p
        logit_def += int(x["logit"] == 1) if g else 0  # placeholder; recompute below
    sub = correct / n
    logit = sum(x["logit"] for x in r) / n
    # predicted-defect rate under each, to expose the one-sided collapse toward "normal"
    lp = sum(1 for x in r if (1 if ("Yes" in x.get("resp", "") or "yes" in x.get("resp", "")) else 0) == 1) / n
    return {"released": sub, "honest": logit, "logit": logit, "artifact": sub - logit, "pred_def%": lp}


def m3ad(model, suf):  # released = strict tag; honest text rule = lenient(recover); + logit ref
    r = jl(f"m3ad_{model}{suf}.jsonl")
    if not r:
        return None
    n = len(r)
    tag = sum(x["m3"] for x in r) / n
    lenient = sum(x["lenient"] for x in r) / n
    logit = sum(x["logit"] for x in r) / n
    abstain = sum(x["abstain"] for x in r) / n
    return {"released": tag, "honest": lenient, "logit": logit, "artifact": tag - lenient, "trigger%": abstain}


BENCH = [
    ("MMAD (ICLR'25)        FABRICATE", mmad, "trigger%"),
    ("Anomaly-OV (CVPR'25)  MISFIRE  ", anomalyov, "pred_def%"),
    ("M3-AD (preprint)      DISCARD  ", m3ad, "trigger%"),
]


def main():
    for ds, suf, base in [("MVTEC", "", 0.734), ("VISA", "_visa", 0.559)]:
        print("=" * 92)
        print(f"{ds}   (base-rate defect = {base})   released = benchmark's actual rule; honest = text-recoverable / abstain; artifact = released-honest")
        print("=" * 92)
        for label, fn, trig in BENCH:
            print(f"\n  {label}")
            print(f"    {'model':14} {'released':>9} {'honest':>8} {'logit':>7} {'artifact':>9}  {trig:>9}")
            comp = []; noncomp = []
            for m in MODELS:
                d = fn(m, suf)
                if not d:
                    print(f"    {m:14}  (missing)"); continue
                tv = d.get(trig)
                print(f"    {m:14} {d['released']:9.3f} {d['honest']:8.3f} {d['logit']:7.3f} {d['artifact']:+9.3f}  {tv:9.3f}")
                (comp if m in COMPLIANT else noncomp).append(d["artifact"])
            cm = sum(comp) / len(comp) if comp else float("nan")
            nm = sum(noncomp) / len(noncomp) if noncomp else float("nan")
            print(f"    {'-> mean artifact':14} compliant={cm:+.3f}   non-compliant={nm:+.3f}")


if __name__ == "__main__":
    main()
