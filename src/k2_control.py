"""DECISIVE CONTROL for the 'latent knowledge' headline.

Claim: when a model does NOT state an answer (human=NONE), the crude snap's correct credit is often the
model recovering LATENT knowledge -- its own logit prefers the gold option. The threat: on binary MMAD
(k=2) a *random* logit prefers gold 50% of the time, so 'latent' could be chance. Test:

  A. MMAD (k=2): on human-NONE items, is the logit's accuracy significantly > 0.50 (binomial)?
  B. MMMU + OmniMedVQA (k~4): on NON-COMPLIANT items (parser fired / clean-abstained), is the logit's
     accuracy significantly > the per-item 1/k chance baseline (normal approx)? This kills 'binary is easy'.

If the logit beats chance on un-stated answers, the model genuinely had latent knowledge it didn't state,
and 'mostly latent, not fabrication' survives. If it's at chance, the headline collapses (good to know now).
"""
from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import binomtest, norm

HERE = Path("data")
CANON = {"N": "NONE", "M": "MULTIPLE"}


def load_logit(stem, n_expect=None):
    rows = [json.loads(l) for l in open(HERE / f"{stem}.jsonl")]
    return {i: r for i, r in enumerate(rows)}


def control_A():
    man = {json.loads(l)["uid"]: json.loads(l) for l in open(HERE / "adjudication_manifest.jsonl")}
    human = {json.loads(l)["uid"]: CANON.get(json.loads(l)["label"], json.loads(l)["label"])
             for l in open(HERE / "human_labels.jsonl")}
    logit = {m: load_logit(f"logit_mmad_{m}") for m in ("pixtral-12b", "llama-3.2-11b")}
    print("=== CONTROL A: MMAD (k=2). On items the model NEVER stated (human=NONE), is its LOGIT > chance(0.50)? ===")
    for scope in ("pixtral-12b", "llama-3.2-11b", "POOLED"):
        c = n = 0
        for uid, lab in human.items():
            if lab not in ("NONE", "MULTIPLE"):
                continue
            m = man[uid]
            if scope != "POOLED" and m["model"] != scope:
                continue
            lr = logit[m["model"]].get(m["idx"])
            if lr is None or lr["gold_letter"] != m["_gold"]:
                continue
            n += 1; c += lr["ok"]
        if n:
            p = binomtest(c, n, 0.5, alternative="greater").pvalue
            star = "  ***" if p < 0.001 else ("  **" if p < 0.01 else ("  *" if p < 0.05 else "  n.s."))
            print(f"   {scope:14s}: logit picks gold on {c}/{n} = {c/n:.3f} un-stated items  (chance 0.50)  p={p:.3g}{star}")


def control_B():
    print("\n=== CONTROL B: MMMU + OmniMedVQA (k~4). On NON-COMPLIANT items, is logit > per-item 1/k chance? ===")
    MODELS = ["gemma-4-e4b", "qwen2.5-vl-7b", "qwen3-vl-8b", "pixtral-12b", "llama-3.2-11b"]
    specs = [("MMMU", "mmmu", "logit_mmmu", "fired"),
             ("OmniMedVQA", "omnimedvqa", "logit_omnimedvqa", "abstain")]
    for label, parse_stem, logit_stem, nc_key in specs:
        obs = exp = var = n = 0
        for m in MODELS:
            try:
                parse = [json.loads(l) for l in open(HERE / f"{parse_stem}_{m}.jsonl")]
                lg = [json.loads(l) for l in open(HERE / f"{logit_stem}_{m}.jsonl")]
            except FileNotFoundError:
                continue
            for pr, lr in zip(parse, lg):
                if not pr.get(nc_key):                    # only non-compliant items
                    continue
                k = lr.get("k", pr.get("k", 4)) or 4
                p = 1.0 / k
                obs += lr["ok"]; exp += p; var += p * (1 - p); n += 1
        if n:
            acc = obs / n; chance = exp / n
            z = (obs - exp) / (var ** 0.5) if var else 0
            pval = 1 - norm.cdf(z)
            star = "  ***" if pval < 0.001 else ("  **" if pval < 0.01 else ("  *" if pval < 0.05 else "  n.s."))
            print(f"   {label:12s}: logit acc on {n} non-compliant items = {acc:.3f}  (mean 1/k chance {chance:.3f})  z={z:.2f} p={pval:.3g}{star}")
        else:
            print(f"   {label:12s}: no non-compliant items found (data missing?)")


if __name__ == "__main__":
    control_A()
    control_B()
    print("\n*** p<0.001  ** p<0.01  * p<0.05.  If logit beats chance on un-stated/non-compliant items,")
    print("the model genuinely had LATENT knowledge it failed to state -> the 'latent, not fabrication' headline holds.")
