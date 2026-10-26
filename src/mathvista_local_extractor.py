"""MathVista, scored LOCALLY end-to-end (no GPT, no API) -- the on-thesis version.

MathVista's official pipeline = LLM-extract the answer from the model's response, then snap (Levenshtein
closest choice, no threshold). The extractor is GPT-4 in their repo. We replace it with the STRONGEST
LOCAL open model (Qwen3-VL-8B) so the whole pipeline runs on a 16GB laptop; the extraction prompt and
the normalize/snap are MathVista's VERBATIM code. We report MathVista's official (snap) accuracy vs an
abstain reference (ignore_empty_extractions / abstain on non-letter), per test model:
   artifact = official_snap_acc - clean_abstain_acc = the no-threshold snap's free credit,
   snap_fire_rate = fraction of items where the extraction was NOT a clean letter (so the snap fired).
Using the strongest local extractor makes the artifact a LOWER bound (a weaker reader -> more snapping).
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "data")
from rescore import MODELS, load_model  # noqa: E402

OUT = Path("data")
TEST_MODELS = ["gemma-4-e4b", "qwen2.5-vl-7b", "qwen3-vl-8b", "pixtral-12b", "llama-3.2-11b"]
EXTRACTOR = "qwen3-vl-8b"

try:
    from Levenshtein import distance as _lev
except Exception:
    def _lev(a, b):
        a, b = str(a), str(b); m, n = len(a), len(b); dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]; dp[0] = i
            for j in range(1, n + 1):
                cur = dp[j]; dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1])); prev = cur
        return dp[n]

# --- MathVista verbatim (prompts/ext_ans.py demo_prompt; calculate_score.py get_most_similar + normalize) ---
DEMO_PROMPT = """Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?\\nChoices:\\n(A) 3/11\\n(B) 8/11\\n(C) 6/11\\n(D) 3/5

Model response: The correct answer is (B) 8/11.

Extracted answer: B"""


def create_test_prompt(query, response):
    return f"{DEMO_PROMPT.strip()}\n\n{query}\n\n{response}\n\nExtracted answer: "


def get_most_similar(prediction, choices):
    distances = [_lev(prediction, c) for c in choices]
    return choices[distances.index(min(distances))]


def normalize_mc(extraction, choices, abstain=False):
    """MathVista's multi_choice normalize. abstain=True -> return None instead of snapping (the honest ref)."""
    extraction = extraction.strip() if isinstance(extraction, str) else str(extraction)
    if abstain and not extraction:
        return None
    letter = re.findall(r"\(([a-zA-Z])\)", extraction)
    if len(letter) > 0:
        extraction = letter[0].upper()
    seq = [chr(ord("A") + i) for i in range(len(choices))]
    if extraction in seq:
        return choices[seq.index(extraction)], False          # clean letter -> map (no snap)
    if abstain:
        return None, None                                     # honest ref: abstain (no clean letter)
    return get_most_similar(extraction, choices), True        # SNAP (no threshold) -- the fabrication


def main():
    import torch
    from datasets import load_dataset
    # MathVista items in the SAME order our rescore used
    ds = load_dataset("AI4Math/MathVista", split="testmini")
    items = [r for r in ds if r.get("question_type") == "multi_choice" and r.get("decoded_image") is not None]
    print(f"{len(items)} MathVista MCQ items")
    backend, name = MODELS[EXTRACTOR]
    print(f"Loading extractor {name} ...")
    model, tok = load_model(backend, name)
    device = next(model.parameters()).device

    def extract(query, response):
        prompt = create_test_prompt(query, response)
        chat = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        pt = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        inp = tok(text=pt, return_tensors="pt")
        inp = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=16, do_sample=False)
        il = inp["input_ids"].shape[-1]
        return tok.decode(gen[0][il:], skip_special_tokens=True).strip().split("\n")[0].strip()

    for tm in TEST_MODELS:
        f = OUT / f"mathvista_{tm}.jsonl"
        if not f.exists():
            print(f"  skip {tm}: no responses"); continue
        rows = [json.loads(l) for l in open(f)]
        if len(rows) != len(items):
            print(f"  WARN {tm}: {len(rows)} rows vs {len(items)} items -- index join unsafe"); continue
        n = snap_ok = clean_ok = clean_abstain = snap_fired = 0
        t0 = time.time()
        for it, row in zip(items, rows):
            resp = row.get("resp", "")
            choices = it["choices"]; gold = str(it["answer"])
            ex = extract(it["query"], resp)
            norm, snapped = normalize_mc(ex, choices, abstain=False)
            snap_ok += int(str(norm) == gold)
            snap_fired += int(bool(snapped))
            cnorm, _ = normalize_mc(ex, choices, abstain=True)
            if cnorm is None:
                clean_abstain += 1
            else:
                clean_ok += int(str(cnorm) == gold)
            n += 1
            if n % 100 == 0:
                print(f"  {tm} [{n}/{len(items)}] {round(time.time()-t0)}s")
        summary = {"test_model": tm, "extractor": EXTRACTOR, "n": n,
                   "official_snap_acc": round(snap_ok / n, 4),
                   "clean_abstain_acc": round(clean_ok / n, 4),
                   "snap_fire_rate": round(snap_fired / n, 4),
                   "abstain_rate": round(clean_abstain / n, 4),
                   "artifact": round((snap_ok - clean_ok) / n, 4)}
        print("SUMMARY", json.dumps(summary))
        with open(OUT / "summary_mathvista_local.jsonl", "a") as fh:
            fh.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
