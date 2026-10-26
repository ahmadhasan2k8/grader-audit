"""MathVision (NeurIPS'24 D&B) -- the NEGATIVE CONTROL. Hard math like MathVista, so our models ramble
just as much (high non-compliance), BUT its released scorer ABSTAINS on parse-failure (extract the
letter; wrong if none) -- it does NOT fabricate. We run our 5 models and report: (1) non-compliance
rate (expected high, like MathVista); (2) the artifact under MathVision's actual abstaining scorer
(= 0 by design, since abstain IS the clean reference); (3) artifact_avoided = what a fabricating
fallback (random.choice over the letters, MMMU-style) WOULD have added. Same difficulty as MathVista;
only the scorer's choice differs -> the figure's high-difficulty / zero-artifact control point.
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
LETTER = re.compile(r"\b([A-E])\b")


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
    ds = load_dataset("MathLLMs/MathVision", split="test")
    rows = [r for r in ds if r.get("options") and r.get("decoded_image") is not None]
    if limit:
        rows = rows[:limit]
    print(f"{len(rows)} MathVision MCQ questions")
    random.seed(42)
    res = {k: 0 for k in ("abstain_ok", "fab_ok", "noncompliant")}
    n = 0
    t0 = time.time()
    jsonl = OUT / f"mathvision_{model_key}.jsonl"
    with open(jsonl, "w") as fh:
        for r in rows:
            letters = [str(o) for o in r["options"]]  # e.g. ['A','B','C','D','E']
            gold = str(r["answer"])
            user = f"{r['question']}\nAnswer with the letter of the correct option."
            img = prep(r["decoded_image"])
            chat = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]}]
            pt = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
            inp = tok(text=pt, images=[img], return_tensors="pt", padding=True)
            inp = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=256, do_sample=False)
            il = inp["input_ids"].shape[-1]
            resp = tok.decode(gen[0][il:], skip_special_tokens=True).strip()
            m = LETTER.search(resp)
            clean = m.group(1) if (m and m.group(1) in letters) else None
            noncompliant = clean is None
            abstain_ok = int(clean == gold) if clean else 0          # MathVision's actual: abstain -> wrong
            fab = clean if clean else random.choice(letters)         # hypothetical fabricating fallback
            fab_ok = int(fab == gold)
            n += 1
            res["abstain_ok"] += abstain_ok
            res["fab_ok"] += fab_ok
            res["noncompliant"] += int(noncompliant)
            fh.write(json.dumps({"gold": gold, "clean": clean, "abstain_ok": abstain_ok,
                                 "fab_ok": fab_ok, "noncompliant": int(noncompliant), "resp": resp}) + "\n")
            if n % 100 == 0:
                print(f"  [{n}/{len(rows)}] {round(time.time()-t0)}s")
    summary = {"model": model_key, "benchmark": "mathvision", "n": n,
               "abstain_acc": round(res["abstain_ok"] / n, 4),          # MathVision's released scorer (the control)
               "fabricate_acc": round(res["fab_ok"] / n, 4),            # if it had fabricated instead
               "noncompliance_rate": round(res["noncompliant"] / n, 4),
               "artifact_under_official": 0.0,                          # abstain IS the clean reference -> 0 by design
               "artifact_avoided": round((res["fab_ok"] - res["abstain_ok"]) / n, 4),
               "elapsed_min": round((time.time() - t0) / 60, 1)}
    print("SUMMARY", json.dumps(summary))
    with open(OUT / "summary_mathvision.jsonl", "a") as fh:
        fh.write(json.dumps(summary) + "\n")
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args()
    run(a.model, a.limit)
