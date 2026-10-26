"""ScienceQA (NeurIPS'22) released-scorer artifact, quantified on our 5 current VLMs.

Released scorer (ScienceQA models/run_gpt3.py, commit 2cbf831):
  get_gpt3_result() applies   re.compile(r'The answer is ([A-Z]).')  to the RAW model completion
  (output = response["choices"][0]["text"].strip()); if exactly one match -> that letter, else the
  string "FAILED". Then get_pred_idx(prediction, choices, options):
        if prediction in options[:len(choices)]: return options.index(prediction)
        else:                                     return random.choice(range(len(choices)))
  So "FAILED" (and any out-of-range / unparseable letter) is mapped to a RANDOM index -- the model
  scores ~1/k by fabrication instead of abstaining. Accuracy is plain mean(pred_idx == gold_idx)
  (confirmed by tools/evaluate_acc.py: true_false = (label == int(results[index]))).

CRITICAL (the MathVista lesson): the regex runs DIRECTLY on the raw completion text. There is NO
LLM/GPT cleanup or extraction buffer between generation and the parse -- get_gpt3_result both
generates and parses in the same function on `output`. So the fabricating fallback fires on RAW
output. We reproduce that verbatim (seeded) and compare to a strict abstain-on-no-clean-letter rule;
artifact = sqa_acc - strict_acc = the free accuracy the random fallback grants on non-compliant
outputs. Single-image multiple-choice (ScienceQA-IMG) questions only.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "data")
from rescore import MODELS, load_model  # noqa: E402

OUT = Path("data")

# ScienceQA's option letters (run_gpt3.py parse_args default: --options ["A","B","C","D","E"]).
OPTIONS = ["A", "B", "C", "D", "E"]

# ScienceQA's verbatim answer-extraction regex (run_gpt3.py line 53).
_SQA_PATTERN = re.compile(r"The answer is ([A-Z]).")


def sqa_get_pred_idx(prediction_text, choices, options=OPTIONS):
    """ScienceQA's verbatim parse + fabricating fallback. Returns (pred_idx, random_fired).

    Mirrors models/run_gpt3.py exactly:
      1. get_gpt3_result(): pattern.findall(output); if len(res)==1 -> res[0], else "FAILED".
         The regex is applied to the RAW completion text (no cleanup buffer).
      2. get_pred_idx(prediction, choices, options):
            if prediction in options[:len(choices)]: return options.index(prediction)
            else:                                     return random.choice(range(len(choices)))
    """
    res = _SQA_PATTERN.findall(prediction_text)
    if len(res) == 1:
        prediction = res[0]            # 'A', 'B', ...  (a single capital letter)
    else:
        prediction = "FAILED"          # ScienceQA's parse-failure sentinel
    # get_pred_idx: in-range letter -> its index; else fabricate a random index.
    if prediction in options[:len(choices)]:
        return options.index(prediction), False
    else:
        return random.choice(range(len(choices))), True   # <-- the fabricating fallback


def strict_parse(response, choices, letters):
    """Honest alternative: abstain (None) unless the response clearly states exactly one option.

    Mirrors rescore_mmmu.strict_parse's intent: a clean '(A)'-style or ' A '-style single letter,
    OR exactly one option's full text. If 0 or >1 clean signals -> None (abstain -> scored wrong).
    """
    r = " " + response.strip(",.!?;:'") + " "
    # 1) bracketed letters, e.g. "(A)"
    brack = [L for L in letters if f"({L})" in r]
    if len(brack) == 1:
        return letters.index(brack[0])
    if len(brack) > 1:
        return None
    # 2) space-delimited letters, e.g. " A "
    spaced = [L for L in letters if f" {L} " in r]
    if len(spaced) == 1:
        return letters.index(spaced[0])
    if len(spaced) > 1:
        return None
    # 3) exactly one option's full text appears verbatim
    low = response.lower()
    hits = [i for i, c in enumerate(choices) if str(c).strip() and str(c).strip().lower() in low]
    if len(hits) == 1:
        return hits[0]
    return None


def prep(pil, max_size=560):
    img = pil.convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        s = max_size / max(w, h)
        img = img.resize((int(w * s), int(h * s)))
    return img


def run(model_key, limit):
    import torch
    from datasets import load_dataset
    backend, name = MODELS[model_key]
    print(f"Loading {name} ...")
    model, tok = load_model(backend, name)
    device = next(model.parameters()).device

    ds = load_dataset("lmms-lab/ScienceQA-IMG", split="test")
    rows = []
    for r in ds:
        if r.get("image") is None:
            continue                              # image-only subset: keep loadable images
        if not (2 <= len(r["choices"]) <= 5):
            continue
        rows.append(r)
    if limit:
        rows = rows[:limit]
    print(f"{len(rows)} ScienceQA-IMG single-image MCQ questions")

    random.seed(42)                               # official fallback uses random -> seed it
    res = {k: 0 for k in ("sqa", "strict", "abstain", "fired", "fired_correct")}
    n = 0
    t0 = time.time()
    jsonl = OUT / f"scienceqa_{model_key}.jsonl"
    with open(jsonl, "w") as fh:
        for r in rows:
            choices = r["choices"]
            k = len(choices)
            letters = [chr(65 + i) for i in range(k)]
            opt_text = "".join(f"({chr(65 + i)}) {c}\n" for i, c in enumerate(choices))
            hint = (r.get("hint") or "").strip()
            ctx = f"Context: {hint}\n" if hint else ""
            # Elicit ScienceQA's OWN expected output format so the regex parser has a fair chance on
            # compliant answers; the random fallback then fires only on GENUINE non-compliance (not on a
            # prompt/scorer format mismatch). This is the faithful, non-overclaiming setup.
            user = (f"{ctx}{r['question']}\n\n{opt_text}\n"
                    "Answer the question, then end your response with 'The answer is X.' "
                    "where X is the letter of the correct option.")
            img = prep(r["image"])
            chat = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]}]
            pt = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
            inp = tok(text=pt, images=[img], return_tensors="pt", padding=True)
            inp = {k2: (v.to(device) if isinstance(v, torch.Tensor) else v) for k2, v in inp.items()}
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=768, do_sample=False)  # 768 (was 256): ScienceQA's CoT-then-"answer is X" format truncated at 256, inducing spurious parse-failures
            il = inp["input_ids"].shape[-1]
            resp = tok.decode(gen[0][il:], skip_special_tokens=True).strip()
            gold = int(r["answer"])
            pred, fired = sqa_get_pred_idx(resp, choices, OPTIONS)
            strict = strict_parse(resp, choices, letters)
            n += 1
            res["sqa"] += int(pred == gold)
            if fired:
                res["fired"] += 1
                res["fired_correct"] += int(pred == gold)
            if strict is None:
                res["abstain"] += 1
            else:
                res["strict"] += int(strict == gold)
            fh.write(json.dumps({"gold": gold, "sqa_pred": pred, "fired": int(fired),
                                 "strict": strict, "k": k, "resp": resp}) + "\n")
            if n % 100 == 0:
                print(f"  [{n}/{len(rows)}] {round(time.time()-t0)}s")
    summary = {"model": model_key, "benchmark": "scienceqa", "n": n,
               "sqa_acc": round(res["sqa"] / n, 4), "strict_acc": round(res["strict"] / n, 4),
               "abstain_rate": round(res["abstain"] / n, 4),
               "random_fire_rate": round(res["fired"] / n, 4),
               "fired_correct": res["fired_correct"],
               # PRIMARY (clean, consistent w/ MMMU): credit the random fallback adds over ABSTAINING on
               # the IDENTICAL regex parse-failures = fired_correct / N. Holds the official parser fixed.
               "artifact_fabrication": round(res["fired_correct"] / n, 4),
               # SECONDARY (confounded by the lenient strict parser's extra recall; can go negative when
               # the narrow regex misses a clear answer that strict recovers): sqa_acc - strict_acc.
               "artifact_sqa_minus_strict": round((res["sqa"] - res["strict"]) / n, 4),
               "elapsed_min": round((time.time() - t0) / 60, 1)}
    print("SUMMARY", json.dumps(summary))
    with open(OUT / "summary_scienceqa.jsonl", "a") as fh:
        fh.write(json.dumps(summary) + "\n")
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--limit", type=int, default=600)
    a = ap.parse_args()
    run(a.model, a.limit)
