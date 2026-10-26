"""MMAD re-scoring: does the model ranking change with the SCORING PATH?

On MMAD's Anomaly Detection subtask we score each open MLLM two ways, from the SAME forward pass:
  * HARD  — MMAD's own scorer: regex \\b([A-E])\\b on the generated text (+ fuzzy fallback).
  * SOFT  — read the option-letter logits at the first generated position; argmax over them.

The prompt + the hard parser are copied verbatim from MMAD's released harness, so the only thing
that differs between the two numbers is the scoring path. If models reorder between HARD and SOFT,
a published leaderboard's ranking depends on the scoring choice.

Usage (requires a 16 GB GPU + transformers/unsloth):
    python data/rescore.py --model qwen2.5-vl-7b --datasets DS-MVTec --limit 100
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from difflib import get_close_matches
from pathlib import Path

MMAD_JSON = "data/mmad.json"
MVTEC_ROOT = "research/mvtec_data"
VISA_ROOT = "research/datasets/visa"
SUFFIX = "Answer with the option's letter from the given choices directly."
OUT = Path("data")

MODELS = {  # key: (backend, hf_name)
    "qwen2.5-vl-7b": ("unsloth", "unsloth/Qwen2.5-VL-7B-Instruct"),
    "qwen3-vl-8b": ("unsloth", "unsloth/Qwen3-VL-8B-Instruct"),
    "gemma-4-e4b": ("unsloth", "unsloth/gemma-4-E4B-it"),
    "gemma-4-e4b-8bit": ("unsloth", "unsloth/gemma-4-E4B-it"),  # same weights, 8-bit (RESCORE_8BIT=1) -- near-lossless quantization control
    "gemma-4-e2b": ("unsloth", "unsloth/gemma-4-E2B-it"),       # smaller -> lower-capability point on the swing-vs-capability curve
    "pixtral-12b": ("unsloth", "unsloth/Pixtral-12B-2409"),
    "llama-3.2-11b": ("unsloth", "unsloth/Llama-3.2-11B-Vision-Instruct"),
    # real MMAD-leaderboard entries via transformers-native loading:
    "llava-1.5-7b": ("llava", "llava-hf/llava-1.5-7b-hf"),
    "llava-1.6-7b": ("llava_next", "llava-hf/llava-v1.6-vicuna-7b-hf"),
    "qwen2-vl-7b": ("unsloth", "unsloth/Qwen2-VL-7B-Instruct"),
}


def mmad_parse(response_text, options, index=0):
    """MMAD's released hard parser: \\b([A-E])\\b, else fuzzy-match the option text.

    MMAD's thirteen released harnesses share one answer parser (identical parse_answer
    copies in the GPT-4/Gemini query files, imported by the rest) returning ALL regex
    matches, but they index that match list INCONSISTENTLY: eleven take the last match
    (gpt4o/gpt4v/gpt4o-mini, gemini, qwen, qwen2_5_vl, internvl, minicpm, internlm_xcomposer,
    sphinx, anomalygpt) and two take the first (llava_query.py:108, cambrian_query.py:99).
    The choice is unreported, and on the models MMAD was built to evaluate it is inert: only
    19 of 8,350 stored responses contain more than one option letter, and the snap-vs-abstain
    ARTIFACT this paper measures is identical under both by construction (the two rules share
    the regex parse and separate only where no letter appears); the ACCURACY reading moves by
    +0.0018 for pixtral only, and validate.py pins both. We report index=0; index=-1 reproduces the majority variant. The choice stops being
    inert once a model reasons before answering -- see the paper's reasoning-model result.
    """
    answers = re.compile(r"\b([A-E])\b").findall(response_text)
    if not answers and options is not None:
        cm = get_close_matches(response_text, list(options.values()), n=1, cutoff=0.0)
        if cm:
            for k, v in options.items():
                if v == cm[0]:
                    answers.append(k)
                    break
    return answers[index] if answers else None


def load_ad_questions(datasets, limit=None):
    d = json.load(open(MMAD_JSON))
    out = []
    for key, v in d.items():
        ds = key.split("/")[0]
        if ds not in datasets:
            continue
        if ds == "DS-MVTec":
            cat = key.split("/")[1]
            local = f"{MVTEC_ROOT}/{cat}/" + v["image_path"].replace("image/", "test/", 1)
        elif ds == "VisA":
            cat = key.split("/")[1]
            ip = v["image_path"].replace("test/bad", "Anomaly").replace("test/good", "Normal")
            local = f"{VISA_ROOT}/{cat}/Data/Images/{ip}"
        else:
            continue
        if not os.path.exists(local):
            continue
        for qa in v["conversation"]:
            if qa["type"] == "Anomaly Detection":
                out.append({"image": local, "question": qa["Question"],
                            "options": qa["Options"], "gold": qa["Answer"]})
    return out[:limit] if limit else out


def load_model(backend, model_name):
    import torch
    load8 = os.environ.get("RESCORE_8BIT", "0") == "1"           # 8-bit (near-lossless) quantization control
    load4 = (not load8) and os.environ.get("RESCORE_4BIT", "1") == "1"
    if backend == "unsloth":
        import transformers.modeling_utils
        transformers.modeling_utils.caching_allocator_warmup = lambda *a, **k: None
        from unsloth import FastVisionModel
        m, t = FastVisionModel.from_pretrained(
            model_name=model_name,
            max_seq_length=4096 if "Pixtral" in model_name else 2048,
            load_in_4bit=load4, load_in_8bit=load8,
            dtype=None if (load4 or load8) else torch.bfloat16, device_map="cuda:0")
        FastVisionModel.for_inference(m)
        return m, t
    # transformers-native (real MMAD entries unsloth doesn't cover)
    from transformers import AutoProcessor, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16) if load4 else None
    if backend == "llava":
        from transformers import LlavaForConditionalGeneration as Cls
    elif backend == "llava_next":
        from transformers import LlavaNextForConditionalGeneration as Cls
    else:
        raise ValueError(f"unknown backend {backend}")
    m = Cls.from_pretrained(model_name, quantization_config=bnb, device_map="cuda:0",
                            torch_dtype=torch.bfloat16 if bnb is None else None)
    m.eval()
    t = AutoProcessor.from_pretrained(model_name)
    return m, t


def run(model_key, datasets, limit, max_new_tokens=16):
    import torch
    from _image_util import load_image
    backend, model_name = MODELS[model_key]
    # (model is passed to load_model directly)
    print(f"Loading {model_name} (backend={backend}) ...")
    model, tok = load_model(backend, model_name)
    device = next(model.parameters()).device
    enc = getattr(tok, "tokenizer", tok)  # text tokenizer (processor.__call__ expects images)
    qs = load_ad_questions(datasets, limit)
    print(f"{len(qs)} Anomaly-Detection questions")

    tag = "-".join(d.replace("DS-", "").lower() for d in datasets)  # mvtec / visa
    OUT.mkdir(parents=True, exist_ok=True)
    jsonl = OUT / (f"{model_key}.jsonl" if tag == "mvtec" else f"{model_key}_{tag}.jsonl")
    n = hard_ok = strict_ok = soft_ok = disagree = abstain_n = 0
    t0 = time.time()
    with open(jsonl, "w") as fh:
        for i, q in enumerate(qs):
            img = load_image(Path(q["image"]), max_size=560)
            opts = q["options"]
            options_text = "".join(f"{k}. {v}\n" for k, v in opts.items())
            user_text = f"Question: {q['question']} \n{options_text}{SUFFIX}"
            chat = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": user_text}]}]
            prompt = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
            inp = tok(text=prompt, images=[img], return_tensors="pt", padding=True)
            inp = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                                     return_dict_in_generate=True, output_scores=True)
            first = gen.scores[0][0].float()  # logits at first generated position

            # SOFT: logit of each option letter's token, argmax
            letter_logits = {}
            for L in opts:
                ids = enc(L, add_special_tokens=False).input_ids
                letter_logits[L] = first[ids[-1]].item()
            soft_pred = max(letter_logits, key=letter_logits.get)

            # HARD: MMAD's regex parse of the decoded text
            il = inp["input_ids"].shape[-1]
            resp = tok.decode(gen.sequences[0][il:], skip_special_tokens=True).strip()
            hard_pred = mmad_parse(resp, opts)                       # MMAD: regex + fuzzy fallback
            strict_m = re.compile(r"\b([A-E])\b").findall(resp)
            strict_pred = strict_m[0] if strict_m else None          # strict: regex only, else abstain

            hok = int(hard_pred == q["gold"])
            stok = int(strict_pred == q["gold"])
            sok = int(soft_pred == q["gold"])
            hard_ok += hok; strict_ok += stok; soft_ok += sok; n += 1
            abstain_n += int(strict_pred is None)
            disagree += int(hard_pred != soft_pred)
            fh.write(json.dumps({"gold": q["gold"], "hard_pred": hard_pred,
                                 "strict_pred": strict_pred, "soft_pred": soft_pred,
                                 "hok": hok, "stok": stok, "sok": sok,
                                 "resp": resp[:80]}) + "\n")
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(qs)}] hard_acc={hard_ok/n:.3f} soft_acc={soft_ok/n:.3f} "
                      f"disagree={disagree/n:.3f}")

    summary = {"model": model_key, "dataset": tag, "n": n,
               "mmad_fuzzy_acc": round(hard_ok / n, 4),     # MMAD's scorer
               "strict_acc": round(strict_ok / n, 4),       # regex-only, abstain=wrong
               "soft_logit_acc": round(soft_ok / n, 4),     # option-letter logit (caveat: tokenizer)
               "fuzzy_inflation": round((hard_ok - strict_ok) / n, 4),
               "abstain_rate": round(abstain_n / n, 4),     # fraction that emit no clean letter
               "elapsed_min": round((time.time() - t0) / 60, 1)}
    print("SUMMARY", json.dumps(summary))
    with open(OUT / "summary.jsonl", "a") as fh:
        fh.write(json.dumps(summary) + "\n")
    del model, tok
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--datasets", nargs="+", default=["DS-MVTec"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    a = ap.parse_args()
    run(a.model, a.datasets, a.limit, a.max_new_tokens)
