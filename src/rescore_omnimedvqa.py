"""OmniMedVQA (CVPR'24) released-scorer artifact, quantified on our 5 VLMs.

Released scorer (MedicalEval/.../medicalqa.py): find_most_similar_index uses difflib.SequenceMatcher
ratio with NO threshold -> always snaps the model's free-text answer to the lexically-closest option,
then exact-matches that option to gt_answer. Same fuzzy idiom as MMAD, in a medical MCQ benchmark
where answers are full option TEXT (no letters). We compare to a containment reference that ABSTAINS
when the response doesn't clearly contain exactly one option. Artifact = fuzzy_acc - clean_acc =
the free credit the no-threshold snap grants on non-compliant outputs.
"""
from __future__ import annotations

import argparse
import difflib
import gc
import glob
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, "data")
from rescore import MODELS, load_model  # noqa: E402

OUT = Path("data")
ROOT = Path("research/datasets/omnimedvqa")


def str_similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def find_most_similar_index(str_list, target):
    """OmniMedVQA verbatim: closest option by SequenceMatcher ratio, no threshold."""
    bi = None
    hi = 0
    for i, s in enumerate(str_list):
        sim = str_similarity(str(s), str(target))
        if sim > hi:
            bi = i
            hi = sim
    return bi  # None only if every ratio == 0


def clean_match(resp, options):
    """Honest reference: the option whose full text the response clearly contains; abstain if 0 or >1."""
    low = resp.lower().strip()
    hits = [i for i, o in enumerate(options) if str(o).lower() in low]
    if len(hits) == 1:
        return hits[0]
    for i, o in enumerate(options):
        if low == str(o).lower():
            return i
    return None  # abstain (ambiguous / no clear option stated)


def resolve_image(image_path):
    for cand in [ROOT / "OmniMedVQA" / image_path, ROOT / image_path, ROOT / "OmniMedVQA" / "Images" / image_path]:
        if cand.exists():
            return cand
    return None


def prep(pil, max_size=560):
    img = pil.convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        s = max_size / max(w, h)
        img = img.resize((int(w * s), int(h * s)))
    return img


def load_qa():
    files = glob.glob(str(ROOT / "**" / "QA_information" / "Open-access" / "*.json"), recursive=True)
    data = []
    for f in files:
        data += json.load(open(f))
    return data


def run(model_key, limit):
    import torch
    from PIL import Image
    backend, name = MODELS[model_key]
    print(f"Loading {name} ...")
    model, tok = load_model(backend, name)
    device = next(model.parameters()).device
    data = load_qa()
    random.seed(42)
    random.shuffle(data)
    rows = []
    for item in data:
        ip = resolve_image(item["image_path"])
        if ip is None:
            continue
        rows.append((item, ip))
        if limit and len(rows) >= limit:
            break
    print(f"{len(rows)} OmniMedVQA questions")
    res = {k: 0 for k in ("fuzzy", "clean", "abstain", "snapped")}
    n = 0
    t0 = time.time()
    jsonl = OUT / f"omnimedvqa_{model_key}.jsonl"
    with open(jsonl, "w") as fh:
        for item, ip in rows:
            opts = [item["option_A"], item["option_B"]]
            if item.get("option_C"):
                opts.append(item["option_C"])
            if item.get("option_D"):
                opts.append(item["option_D"])
            gt = item["gt_answer"]
            user = (f"{item['question']} Here are {len(opts)} candidate answers:{opts} "
                    f"Only return what you think is the correct answer from the candidate answers, "
                    f"do not return any other irrelevant text!")
            img = prep(Image.open(ip))
            chat = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]}]
            pt = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
            inp = tok(text=pt, images=[img], return_tensors="pt", padding=True)
            inp = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=32, do_sample=False)
            il = inp["input_ids"].shape[-1]
            resp = tok.decode(gen[0][il:], skip_special_tokens=True).strip()
            fi = find_most_similar_index(opts, resp)
            fuzzy_ok = int(fi is not None and str(opts[fi]) == str(gt))
            ci = clean_match(resp, opts)
            clean_ok = int(ci is not None and str(opts[ci]) == str(gt))
            n += 1
            res["fuzzy"] += fuzzy_ok
            if ci is None:
                res["abstain"] += 1
                res["snapped"] += 1  # fuzzy snapped where clean would abstain
            else:
                res["clean"] += clean_ok
            fh.write(json.dumps({"gt": str(gt)[:40], "fuzzy_ok": fuzzy_ok, "clean_ok": clean_ok,
                                 "abstain": int(ci is None), "k": len(opts), "resp": resp[:80]}) + "\n")
            if n % 200 == 0:
                print(f"  [{n}/{len(rows)}] {round(time.time()-t0)}s")
    summary = {"model": model_key, "benchmark": "omnimedvqa", "n": n,
               "fuzzy_acc": round(res["fuzzy"] / n, 4), "clean_acc": round(res["clean"] / n, 4),
               "abstain_rate": round(res["abstain"] / n, 4), "snap_rate": round(res["snapped"] / n, 4),
               "artifact_fuzzy_minus_clean": round((res["fuzzy"] - res["clean"]) / n, 4),
               "elapsed_min": round((time.time() - t0) / 60, 1)}
    print("SUMMARY", json.dumps(summary))
    with open(OUT / "summary_omnimedvqa.jsonl", "a") as fh:
        fh.write(json.dumps(summary) + "\n")
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--limit", type=int, default=1000)
    a = ap.parse_args()
    run(a.model, a.limit)
