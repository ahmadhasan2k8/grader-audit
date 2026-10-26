"""POPE (EMNLP'23) released-scorer artifact, quantified on our 5 current VLMs.

Released rule (AoiDragon/POPE evaluate.py:9-21): keep first sentence, split on spaces; if any of the
tokens 'No'/'not'/'no' appear -> 'no', ELSE -> 'yes'. So any hedge/refusal/caption/ambiguous output
silently DEFAULTS TO 'yes' (directional, toward the positive class). We compare against a parser-free
yes/no first-token logit reference and a strict abstain-on-ambiguity rule, and report the default-yes
rate and the inflation. POPE is balanced 50/50, so default-to-yes is a measurable directional bias.
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


def pope_parse(text):
    """MMAD-style verbatim POPE rule. Returns (pred 1=yes/0=no, defaulted_to_yes)."""
    t = text.split(".")[0] if "." in text else text
    t = t.replace(",", "")
    words = t.split(" ")
    if "No" in words or "not" in words or "no" in words:
        return 0, False
    return 1, True  # default-to-yes


def strict_parse(text):
    """Honest alternative: abstain instead of forcing yes on ambiguity."""
    low = text.lower()
    w = low.replace(",", " ").replace(".", " ").split()
    has_no = ("no" in w) or ("not" in w) or ("none" in w) or ("nope" in w) or ("n't" in low) or low.startswith("no")
    has_yes = ("yes" in w) or low.startswith("yes")
    if has_no and not has_yes:
        return 0
    if has_yes and not has_no:
        return 1
    return None  # ambiguous -> abstain


def prep(pil, max_size=560):
    img = pil.convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        s = max_size / max(w, h)
        img = img.resize((int(w * s), int(h * s)))
    return img


def run(model_key, limit, category):
    import torch
    from datasets import load_dataset
    backend, name = MODELS[model_key]
    print(f"Loading {name} ...")
    model, tok = load_model(backend, name)
    enc = getattr(tok, "tokenizer", tok)
    device = next(model.parameters()).device
    yes_id = enc("Yes", add_special_tokens=False).input_ids[0]
    no_id = enc("No", add_special_tokens=False).input_ids[0]
    ds = load_dataset("lmms-lab/POPE", split="test")
    if category:
        ds = ds.filter(lambda r: r["category"] == category)
    ds = ds.select(range(min(limit, len(ds)))) if limit else ds
    print(f"{len(ds)} POPE questions (category={category})")
    res = {k: 0 for k in ("pope", "logit", "strict", "abstain", "default_yes", "gold_yes", "pope_yes", "logit_yes")}
    n = 0
    t0 = time.time()
    jsonl = OUT / f"pope_{model_key}.jsonl"
    with open(jsonl, "w") as fh:
        for r in ds:
            gold = 1 if r["answer"].strip().lower().startswith("yes") else 0
            img = prep(r["image"])
            chat = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": r["question"]}]}]
            pt = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
            inp = tok(text=pt, images=[img], return_tensors="pt", padding=True)
            inp = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=16, do_sample=False,
                                     return_dict_in_generate=True, output_scores=True)
            first = gen.scores[0][0].float()
            logit_pred = 1 if first[yes_id].item() > first[no_id].item() else 0
            il = inp["input_ids"].shape[-1]
            resp = tok.decode(gen.sequences[0][il:], skip_special_tokens=True).strip()
            pope_pred, defaulted = pope_parse(resp)
            strict_pred = strict_parse(resp)
            n += 1
            res["pope"] += int(pope_pred == gold)
            res["logit"] += int(logit_pred == gold)
            res["default_yes"] += int(defaulted)
            res["gold_yes"] += gold
            res["pope_yes"] += pope_pred
            res["logit_yes"] += logit_pred
            if strict_pred is None:
                res["abstain"] += 1
            else:
                res["strict"] += int(strict_pred == gold)
            fh.write(json.dumps({"gold": gold, "pope": pope_pred, "logit": logit_pred,
                                 "strict": strict_pred, "default": int(defaulted), "resp": resp[:80]}) + "\n")
            if n % 300 == 0:
                print(f"  [{n}/{len(ds)}] {round(time.time()-t0)}s")
    summary = {"model": model_key, "benchmark": "pope", "category": category, "n": n,
               "pope_acc": round(res["pope"] / n, 4), "logit_acc": round(res["logit"] / n, 4),
               "strict_acc": round(res["strict"] / n, 4), "abstain_rate": round(res["abstain"] / n, 4),
               "default_yes_rate": round(res["default_yes"] / n, 4),
               "gold_yes_rate": round(res["gold_yes"] / n, 4),
               "pope_yes_rate": round(res["pope_yes"] / n, 4), "logit_yes_rate": round(res["logit_yes"] / n, 4),
               "artifact_pope_minus_logit": round((res["pope"] - res["logit"]) / n, 4),
               "artifact_pope_minus_strict": round((res["pope"] - res["strict"]) / n, 4),
               "elapsed_min": round((time.time() - t0) / 60, 1)}
    print("SUMMARY", json.dumps(summary))
    with open(OUT / "summary_pope.jsonl", "a") as fh:
        fh.write(json.dumps(summary) + "\n")
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--category", default="adversarial")
    a = ap.parse_args()
    run(a.model, a.limit, a.category)
