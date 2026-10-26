"""MMMU (CVPR'24 Oral) released-scorer artifact, quantified on our 5 current VLMs.

Released parse_multi_choice_response (MMMU eval_utils.py): try '(A)'-style, then ' A '-style, then
option-text substring; if NONE match -> pred_index = random.choice(all_choices). So a model that
fails to emit a clean letter still scores ~1/k by random fabrication, instead of abstaining. We
re-implement MMMU's parser verbatim (seeded) and compare to a strict abstain-on-no-letter rule;
artifact = mmmu_acc - strict_acc = the free accuracy the random fallback grants on non-compliant
outputs. Single-image multiple-choice validation questions only.
"""
from __future__ import annotations

import argparse
import ast
import gc
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, "data")
from rescore import MODELS, load_model  # noqa: E402

OUT = Path("data")
SUBJECTS = ["Accounting", "Art", "Biology", "Chemistry", "Computer_Science", "Economics",
            "Geography", "History", "Math", "Physics", "Psychology", "Sociology"]


def parse_multi_choice_response(response, all_choices, index2ans):
    """MMMU's verbatim parser. Returns (pred_letter, random_fired)."""
    for ch in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(ch)
    response = " " + response + " "
    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice); ans_with_brack = True
    if len(candidates) == 0:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index); index_ans = False
    if len(candidates) == 0:
        return random.choice(all_choices), True            # <-- the fabricating fallback
    if len(candidates) > 1:
        starts = []
        for can in candidates:
            if index_ans:
                key = f"({can})" if ans_with_brack else f" {can} "
                starts.append(response.rfind(key))
            else:
                starts.append(response.lower().rfind(index2ans[can].lower()))
        return candidates[max(range(len(starts)), key=lambda i: starts[i])], False
    return candidates[0], False


def strict_parse(response, all_choices):
    """Honest alternative: abstain (None) if no clean letter is emitted."""
    r = " " + response.strip(",.!?;:'") + " "
    for choice in all_choices:
        if f"({choice})" in r:
            return choice
    for choice in all_choices:
        if f" {choice} " in r:
            return choice
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
    rows = []
    for subj in SUBJECTS:
        ds = load_dataset("MMMU/MMMU", subj, split="validation")
        for r in ds:
            if r["question_type"] != "multiple-choice":
                continue
            if r.get("image_2") is not None or r.get("image_1") is None:
                continue  # single-image only
            rows.append(r)
    if limit:
        rows = rows[:limit]
    print(f"{len(rows)} MMMU single-image MCQ questions")
    random.seed(42)
    res = {k: 0 for k in ("mmmu", "strict", "abstain", "fired", "fired_correct")}
    n = 0
    t0 = time.time()
    jsonl = OUT / f"mmmu_{model_key}.jsonl"
    with open(jsonl, "w") as fh:
        for r in rows:
            opts = ast.literal_eval(r["options"]) if isinstance(r["options"], str) else r["options"]
            letters = [chr(65 + i) for i in range(len(opts))]
            index2ans = {chr(65 + i): str(o) for i, o in enumerate(opts)}
            opt_text = "".join(f"({chr(65 + i)}) {o}\n" for i, o in enumerate(opts))
            user = f"{r['question']}\n\n{opt_text}\nAnswer with the option's letter from the given choices directly."
            img = prep(r["image_1"])
            chat = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]}]
            pt = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
            inp = tok(text=pt, images=[img], return_tensors="pt", padding=True)
            inp = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=256, do_sample=False)
            il = inp["input_ids"].shape[-1]
            resp = tok.decode(gen[0][il:], skip_special_tokens=True).strip()
            gold = r["answer"]
            pred, fired = parse_multi_choice_response(resp, letters, index2ans)
            strict = strict_parse(resp, letters)
            n += 1
            res["mmmu"] += int(pred == gold)
            if fired:
                res["fired"] += 1
                res["fired_correct"] += int(pred == gold)
            if strict is None:
                res["abstain"] += 1
            else:
                res["strict"] += int(strict == gold)
            fh.write(json.dumps({"gold": gold, "mmmu": pred, "fired": int(fired),
                                 "strict": strict, "k": len(letters), "resp": resp}) + "\n")
            if n % 100 == 0:
                print(f"  [{n}/{len(rows)}] {round(time.time()-t0)}s")
    summary = {"model": model_key, "benchmark": "mmmu", "n": n,
               "mmmu_acc": round(res["mmmu"] / n, 4), "strict_acc": round(res["strict"] / n, 4),
               "abstain_rate": round(res["abstain"] / n, 4),
               "random_fire_rate": round(res["fired"] / n, 4),
               "fired_correct": res["fired_correct"],
               "artifact_mmmu_minus_strict": round((res["mmmu"] - res["strict"]) / n, 4),
               "elapsed_min": round((time.time() - t0) / 60, 1)}
    print("SUMMARY", json.dumps(summary))
    with open(OUT / "summary_mmmu.jsonl", "a") as fh:
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
