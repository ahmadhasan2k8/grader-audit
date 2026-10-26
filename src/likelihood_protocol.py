"""The likelihood (logit) rule, specified and decomposed -- camera-ready evidence (WACV'27 rev.).

Reviewer QRkQ asked for the likelihood rule's exact operational status: is it gated by parsing (a
hybrid policy) or applied to every item (in which case agreement with the generated letter is not
guaranteed)? And they noted correctly that lower accuracy against gold does not by itself show the
rule mis-recovers a generated answer -- those are different evaluation targets.

The protocol (see rescore_logit.py for the implementation):
  * UNCONDITIONAL. One forward pass per item, applied to EVERY item. Parsing never gates it; there
    is no decoded text and no fallback. It is a standalone rule, not a post-parse-failure hybrid.
  * Conditioning context: image + the same prompt body the text run used (question + lettered
    options + the benchmark's own answer cue), through the model's chat template.
  * Readout position: the logits over the FIRST response token.
  * Tokenization: for each letter, the candidate first-token ids for both "A" and " A", selecting
    the token whose decoded text stripped equals the letter (handles BPE merge-the-space and
    SentencePiece leading-space-as-its-own-token); score = max logit over that letter's id set;
    prediction = argmax over letters.

This script measures the three quantities that pin the rule down empirically:

  (1) AGREEMENT WITH THE GENERATED LETTER on compliant items. If the rule tracked a different target
      everywhere, this would be low. It is 0.968-1.000, so on compliant outputs the likelihood rule
      and the model's own stated answer essentially coincide -- the comparison is apples-to-apples
      there, and the rule is a legitimate parse-free reading of a committed answer.

  (2) ACCURACY DECOMPOSED BY COMPLIANCE. The rule's accuracy deficit is concentrated on the
      non-compliant stratum, i.e. exactly where it stops tracking a stated answer and makes its own
      choice.

  (3) THE OPTION-PRIOR SKEW on the non-compliant stratum, where gold is near-balanced. The rule
      picks the first option well above the gold rate (binomial test against the stratum's own gold
      base rate). This is the direct evidence for "likelihood follows answer-token priors" -- a
      measured skew, not an inference from lower accuracy.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
MODELS = ["gemma-4-e4b", "qwen2.5-vl-7b", "qwen3-vl-8b", "llama-3.2-11b", "pixtral-12b"]

try:
    from scipy.stats import binomtest
except ImportError:  # keep the package runnable without scipy
    binomtest = None


def load(name):
    return [json.loads(l) for l in open(DATA / name) if l.strip()]


def join(model):
    gen, lg = load(f"mmad_{model}.jsonl"), load(f"logit_mmad_{model}.jsonl")
    assert len(gen) == len(lg), f"{model}: length mismatch"
    assert all(g["gold"] == l["gold_letter"] for g, l in zip(gen, lg)), f"{model}: gold mismatch"
    return list(zip(gen, lg))


def main():
    print("(1) Likelihood vs the model's OWN generated letter, compliant items only")
    print(f"    {'model':16s} {'n':>6s} {'agreement':>10s}")
    for m in MODELS:
        pairs = join(m)
        c = [(g, l) for g, l in pairs if g["strict_pred"] is not None]
        agree = sum(1 for g, l in c if g["strict_pred"] == l["pred_letter"]) / len(c)
        print(f"    {m:16s} {len(c):6d} {agree:10.3f}")

    print("\n(2) Likelihood accuracy decomposed by compliance stratum")
    print(f"    {'model':16s} {'compliant':>16s} {'non-compliant':>16s} {'overall':>8s}")
    for m in ["llama-3.2-11b", "pixtral-12b"]:
        pairs = join(m)
        c = [(g, l) for g, l in pairs if g["strict_pred"] is not None]
        n = [(g, l) for g, l in pairs if g["strict_pred"] is None]
        f = lambda s: sum(l["ok"] for _, l in s) / len(s)
        print(f"    {m:16s} {len(c):5d}/{f(c):.3f}   {len(n):8d}/{f(n):.3f}   {f(pairs):8.3f}")

    print("\n(3) Option-prior skew on the non-compliant stratum (gold is near-balanced there)")
    for m in ["llama-3.2-11b", "pixtral-12b"]:
        pairs = join(m)
        n = [(g, l) for g, l in pairs if g["strict_pred"] is None]
        pred = collections.Counter(l["pred_letter"] for _, l in n)
        gold = collections.Counter(g["gold"] for g, _ in n)
        tot = len(n)
        pa, ga = pred["A"] / tot, gold["A"] / tot
        line = (f"    {m:16s} n={tot:4d}  likelihood picks A {pred['A']:4d} ({pa:.3f})  "
                f"gold is A {gold['A']:4d} ({ga:.3f})")
        if binomtest is not None:
            p = binomtest(pred["A"], tot, ga, alternative="greater").pvalue
            line += f"  p={p:.2e}"
        print(line)


if __name__ == "__main__":
    main()
