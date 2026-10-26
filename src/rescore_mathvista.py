"""MathVista (ICLR'24 Oral) released-scorer artifact, quantified on our 5 VLMs.

Released scorer (calculate_score.py): get_most_similar maps the model's answer to the choice with the
minimum Levenshtein distance -- NO threshold, so any response is forced to a choice (never abstains).
Hard math/geometry MCQ guarantees our general VLMs ramble (non-compliance), so the no-threshold snap
fires. We compare to a containment reference that abstains when the response states no clear choice.
Artifact = fuzzy_acc - clean_acc.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "data")
from rescore import MODELS, load_model  # noqa: E402

OUT = Path("data")

try:
    from Levenshtein import distance as _lev
except Exception:
    def _lev(a, b):
        a, b = str(a), str(b)
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]; dp[0] = i
            for j in range(1, n + 1):
                cur = dp[j]
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
                prev = cur
        return dp[n]


def get_most_similar(choices, target):
    """MathVista verbatim: choice with min Levenshtein distance, no threshold (always returns one)."""
    return min(range(len(choices)), key=lambda i: _lev(str(choices[i]), str(target)))


def clean_match(resp, choices):
    """Honest reference: the choice whose text the response clearly contains; abstain if 0 or >1."""
    low = resp.lower().strip()
    hits = [i for i, c in enumerate(choices) if str(c).lower() in low]
    if len(hits) == 1:
        return hits[0]
    for i, c in enumerate(choices):
        if low == str(c).lower():
            return i
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
    ds = load_dataset("AI4Math/MathVista", split="testmini")
    rows = [r for r in ds if r.get("question_type") == "multi_choice" and r.get("decoded_image") is not None]
    if limit:
        rows = rows[:limit]
    print(f"{len(rows)} MathVista multi_choice questions")
    res = {k: 0 for k in ("fuzzy", "clean", "abstain", "snapped")}
    n = 0
    t0 = time.time()
    jsonl = OUT / f"mathvista_{model_key}.jsonl"
    with open(jsonl, "w") as fh:
        for r in rows:
            choices = r["choices"]
            gt = r["answer"]
            user = f"{r['question']}\nChoices: {choices}\nAnswer with exactly one of the given choices."
            img = prep(r["decoded_image"])
            chat = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]}]
            pt = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
            inp = tok(text=pt, images=[img], return_tensors="pt", padding=True)
            inp = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=64, do_sample=False)
            il = inp["input_ids"].shape[-1]
            resp = tok.decode(gen[0][il:], skip_special_tokens=True).strip()
            fi = get_most_similar(choices, resp)
            fuzzy_ok = int(str(choices[fi]) == str(gt))
            ci = clean_match(resp, choices)
            clean_ok = int(ci is not None and str(choices[ci]) == str(gt))
            n += 1
            res["fuzzy"] += fuzzy_ok
            if ci is None:
                res["abstain"] += 1
                res["snapped"] += 1
            else:
                res["clean"] += clean_ok
            fh.write(json.dumps({"gt": str(gt)[:40], "fuzzy_ok": fuzzy_ok, "clean_ok": clean_ok,
                                 "abstain": int(ci is None), "k": len(choices), "resp": resp}) + "\n")
            if n % 100 == 0:
                print(f"  [{n}/{len(rows)}] {round(time.time()-t0)}s")
    summary = {"model": model_key, "benchmark": "mathvista", "n": n,
               "fuzzy_acc": round(res["fuzzy"] / n, 4), "clean_acc": round(res["clean"] / n, 4),
               "abstain_rate": round(res["abstain"] / n, 4), "snap_rate": round(res["snapped"] / n, 4),
               "artifact_fuzzy_minus_clean": round((res["fuzzy"] - res["clean"]) / n, 4),
               "elapsed_min": round((time.time() - t0) / 60, 1)}
    print("SUMMARY", json.dumps(summary))
    with open(OUT / "summary_mathvista.jsonl", "a") as fh:
        fh.write(json.dumps(summary) + "\n")
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.model, a.limit)
