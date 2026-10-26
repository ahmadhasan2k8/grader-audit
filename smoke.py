"""Self-contained Tier-2 smoke test: validate the inference + scoring code path end to end.

Runs a real VLM on a SYNTHETIC image + a synthetic yes/no question (no benchmark dataset, no
license-gated images), then applies the same snap / abstain / likelihood scoring used to produce the
released logs, and asserts a valid record in the released schema. This confirms the Tier-2 inference
code runs; it does not reproduce paper numbers (that is Tier-1, `python reproduce.py`).

Requires a GPU and the Tier-2 dependencies (`pip install -r requirements-gpu.txt`); the model
auto-downloads from the Hugging Face Hub on first use.

Usage:  python smoke.py [--model gemma-4-e2b]
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-4-e2b", help="any key in rescore.MODELS")
    a = ap.parse_args()

    import torch
    from PIL import Image
    from rescore import load_model, mmad_parse, MODELS, SUFFIX

    backend, name = MODELS[a.model]
    print(f"Loading {name} ...")
    model, tok = load_model(backend, name)
    device = next(model.parameters()).device
    enc = getattr(tok, "tokenizer", tok)

    img = Image.new("RGB", (256, 256), (170, 170, 170))  # synthetic gray test image
    opts = {"A": "Yes", "B": "No"}
    user = f"Question: Is there a visible defect in this image? \nA. Yes\nB. No\n{SUFFIX}"
    chat = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]}]
    prompt = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    inp = tok(text=prompt, images=[img], return_tensors="pt", padding=True)
    inp = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}
    with torch.no_grad():
        gen = model.generate(**inp, max_new_tokens=16, do_sample=False,
                             return_dict_in_generate=True, output_scores=True)

    first = gen.scores[0][0].float()
    soft_pred = max(opts, key=lambda L: first[enc(L, add_special_tokens=False).input_ids[-1]].item())
    il = inp["input_ids"].shape[-1]
    resp = tok.decode(gen.sequences[0][il:], skip_special_tokens=True).strip()
    rec = {
        "hard_pred": mmad_parse(resp, opts),                          # no-threshold snap
        "strict_pred": (re.findall(r"\b([A-E])\b", resp) or [None])[0],  # abstain
        "soft_pred": soft_pred,                                       # likelihood
        "resp": resp[:80],
    }
    print("produced record:", json.dumps(rec))

    ok = rec["hard_pred"] in opts and rec["soft_pred"] in opts and isinstance(rec["resp"], str)
    print("\nSMOKE PASSED: model loads, generates, and all three scoring rules produce a valid record."
          if ok else "\nSMOKE FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
