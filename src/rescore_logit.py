"""Parse-free LIKELIHOOD (logit) re-scorer — the robust MCQ method we recommend the field adopt.

The text-parsing scorers re-implemented in this directory (rescore.py / rescore_mmmu.py /
rescore_scienceqa.py / rescore_omnimedvqa.py) all share a "fabricating fallback": when the model's
free-text output doesn't yield a clean answer, the released harness *snaps* it to an option (regex +
fuzzy match, ' A ' substring, no-threshold SequenceMatcher, or a seeded random.choice). That snap
grants free accuracy on non-compliant outputs and makes leaderboards depend on the SCORING PATH.

This file implements the lm-eval-harness / SEED-Bench style alternative: PARSE-FREE likelihood
scoring. For each MCQ item we present the lettered options + a terse answer cue, do ONE forward pass
(NO generation), read the next-token logits, and for each candidate letter A,B,C,... take the model's
logit for that letter as the FIRST response token. argmax over the k letters is the prediction. There
is no decoded text, hence no parsing and no fallback — the method is structurally immune to the
verdict-fabrication bug. It is also cheaper (1 forward pass per item, no autoregressive decode).

Faithfulness: each benchmark's item filtering, options, gold answer, prompt body and 560px image prep
are replicated EXACTLY from the corresponding text re-scorer, so the logit numbers are measured on the
SAME items as the text run (apples to apples: buggy-snap vs abstain vs logit).

Usage (requires a 16 GB GPU + transformers/unsloth):
    python data/rescore_logit.py --benchmark mmmu --model qwen2.5-vl-7b --limit 100
    python data/rescore_logit.py --benchmark mmad --model qwen2.5-vl-7b
"""
from __future__ import annotations

import argparse
import ast
import gc
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

# Reuse the EXACT model registry + loader the text re-scorers use (treat as read-only).
sys.path.insert(0, "data")
from rescore import MODELS, load_model  # noqa: E402

OUT = Path("data")

# ---- benchmark-specific constants, copied verbatim from the text re-scorers ----
# MMMU (rescore_mmmu.py): 12 of the 30 subjects, single-image multiple-choice validation only.
MMMU_SUBJECTS = ["Accounting", "Art", "Biology", "Chemistry", "Computer_Science", "Economics",
                 "Geography", "History", "Math", "Physics", "Psychology", "Sociology"]
# MMAD (rescore.py): default to the DS-MVTec Anomaly-Detection subtask, matching the existing run.
MMAD_JSON = "data/mmad.json"
MVTEC_ROOT = "research/mvtec_data"
VISA_ROOT = "research/datasets/visa"
MMAD_SUFFIX = "Answer with the option's letter from the given choices directly."
# OmniMedVQA (rescore_omnimedvqa.py)
OMNI_ROOT = Path("research/datasets/omnimedvqa")


# ----------------------------------------------------------------------------- #
#  Image prep — identical 560px longest-edge logic used by every text re-scorer  #
# ----------------------------------------------------------------------------- #
def prep(pil, max_size=560):
    """Replicates rescore_mmmu.prep / rescore_scienceqa.prep / rescore_omnimedvqa.prep."""
    img = pil.convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        s = max_size / max(w, h)
        img = img.resize((int(w * s), int(h * s)))
    return img


# ----------------------------------------------------------------------------- #
#  THE SHARED PARSE-FREE SCORER                                                   #
# ----------------------------------------------------------------------------- #
def _letter_token_ids(tok, letter):
    """Candidate first-token ids for an option letter under this tokenizer.

    Goal: the id(s) the model would emit as its FIRST response token when answering with `letter`.
    Two tokenizer behaviours have to be handled, both observed in this repo's model registry:

      * BPE merge-the-space (Qwen / Llama-3 / Gemma / Pixtral-Mistral): `encode(" A")` -> a SINGLE
        distinct token that decodes to " A" (e.g. Qwen2.5-VL: "A"->32, " A"->362). Either may be the
        first response token depending on whether the chat template's generation prefix ends on
        whitespace, so BOTH are valid candidates.
      * SentencePiece leading-space-as-its-own-token (LLaVA-1.5 / Vicuna / Llama-2): `encode(" A")`
        -> [<whitespace '▁'>, <"A">] (e.g. " A"->[29871, 319]). Here the FIRST token (29871) is a
        bare whitespace marker shared by ALL letters, NOT the letter; the letter is the SECOND token.

    So we cannot blindly take toks[0] for the spaced variant. For each variant we pick the first
    token whose decoded text *stripped* equals the letter (the actual letter token), and only if no
    token decodes to the letter do we fall back to toks[0]. Bare `encode("A")` is always a single
    letter token across these families, so it anchors the mapping. Results are deduped; the scorer
    later takes the max logit over the set, keeping it tokenizer-agnostic.
    """
    enc = getattr(tok, "tokenizer", tok)   # VLM processors (e.g. Gemma4Processor) wrap a .tokenizer
    ids = []
    for variant in (letter, " " + letter):
        toks = enc.encode(variant, add_special_tokens=False)
        if not toks:
            continue
        chosen = None
        for tid in toks:
            try:
                if enc.decode([tid]).strip() == letter:  # the genuine letter token
                    chosen = tid
                    break
            except Exception:
                continue
        ids.append(chosen if chosen is not None else toks[0])
    # dedupe, preserve order
    seen = set()
    out = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def letter_logit_score(model, tok, image, question_text, k, device):
    """Parse-free likelihood score for one MCQ item.

    Builds user content = image + `question_text` (which already contains the lettered options and
    a terse answer cue), runs ONE forward pass (no generation), reads the logits over the FIRST
    response token, and for each letter L in A..chr(64+k) scores it by the max logit over that
    letter's candidate first-token ids. Returns (pred_idx in 0..k-1, unmapped_flag).

    This is the ONLY model call per item — no autoregressive decode, no text, no parsing, no
    fabricating fallback. Structurally immune to the verdict-fabrication bug.

    Args:
        model:         loaded VLM (callable, returns an object with `.logits`).
        tok:           the processor/tokenizer (its __call__ accepts text= and images=; its
                       .encode / .apply_chat_template behave like a tokenizer).
        image:         a single PIL image (already 560px-prepped).
        question_text: the full text block (question + lettered options + answer cue).
        k:             number of options (letters A..chr(64+k)).
        device:        torch device to move tensors onto.

    Returns:
        (pred_idx, unmapped): pred_idx is argmax letter index 0..k-1; unmapped is 1 if any letter
        had NO usable token id (its score was -inf), else 0.
    """
    import torch

    chat = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": question_text}]}]
    pt = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    inp = tok(text=pt, images=[image], return_tensors="pt", padding=True)
    inp = {kk: (v.to(device) if isinstance(v, torch.Tensor) else v) for kk, v in inp.items()}
    with torch.no_grad():
        out = model(**inp)
    if not hasattr(out, "logits"):
        raise RuntimeError(
            "model(**inp) returned an object without `.logits` (got "
            f"{type(out).__name__}). Parse-free logit scoring needs the next-token logit "
            "distribution; this model/backend does not expose it from a plain forward pass."
        )
    logits = out.logits[0, -1, :].float()  # distribution over the FIRST response token

    letters = [chr(65 + i) for i in range(k)]
    scores = []
    unmapped = 0
    for L in letters:
        ids = _letter_token_ids(tok, L)
        if not ids:
            scores.append(float("-inf"))  # no usable id for this letter -> cannot be chosen
            unmapped = 1
            continue
        scores.append(max(logits[i].item() for i in ids))
    pred_idx = max(range(k), key=lambda i: scores[i])
    return pred_idx, unmapped


# ----------------------------------------------------------------------------- #
#  PER-BENCHMARK LOADERS — replicate each text re-scorer's item selection EXACTLY #
#  Each yields dicts: {image (PIL, 560px-prepped), text (q+opts+cue), gold (idx), k}
#  `text` already includes the answer cue matched to that benchmark's text prompt.
# ----------------------------------------------------------------------------- #
def load_mmmu(limit):
    """MMMU single-image MCQ validation items (mirrors rescore_mmmu.run row collection).

    Answer cue: the IDENTICAL trailing instruction the text run used —
    "Answer with the option's letter from the given choices directly." Options are formatted
    "(A) opt\\n..." exactly as in rescore_mmmu (opt_text). gold = letter -> index.
    """
    from datasets import load_dataset
    rows = []
    for subj in MMMU_SUBJECTS:
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
    for r in rows:
        opts = ast.literal_eval(r["options"]) if isinstance(r["options"], str) else r["options"]
        k = len(opts)
        letters = [chr(65 + i) for i in range(k)]
        opt_text = "".join(f"({chr(65 + i)}) {o}\n" for i, o in enumerate(opts))
        # SAME body as rescore_mmmu.run (question + options + the harness instruction).
        text = (f"{r['question']}\n\n{opt_text}\n"
                "Answer with the option's letter from the given choices directly.")
        gold_idx = letters.index(r["answer"])  # r["answer"] is a letter
        yield {"image": prep(r["image_1"]), "text": text, "gold": gold_idx, "k": k}


def load_scienceqa(limit):
    """ScienceQA-IMG single-image MCQ items (mirrors rescore_scienceqa.run row collection).

    Answer cue: the text run elicited ScienceQA's own "The answer is X." format. For parse-free
    scoring there is nothing to parse, so we use a terse direct cue that yields a single leading
    letter ("Answer with the option's letter directly."), keeping the same question + Context +
    "(A) c\\n" option formatting. gold = int(r["answer"]) index.
    """
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/ScienceQA-IMG", split="test")
    rows = []
    for r in ds:
        if r.get("image") is None:
            continue
        if not (2 <= len(r["choices"]) <= 5):
            continue
        rows.append(r)
    if limit:
        rows = rows[:limit]
    print(f"{len(rows)} ScienceQA-IMG single-image MCQ questions")
    for r in rows:
        choices = r["choices"]
        k = len(choices)
        opt_text = "".join(f"({chr(65 + i)}) {c}\n" for i, c in enumerate(choices))
        hint = (r.get("hint") or "").strip()
        ctx = f"Context: {hint}\n" if hint else ""
        text = (f"{ctx}{r['question']}\n\n{opt_text}\n"
                "Answer with the option's letter from the given choices directly.")
        yield {"image": prep(r["image"]), "text": text, "gold": int(r["answer"]), "k": k}


def load_omnimedvqa(limit):
    """OmniMedVQA Open-access items (mirrors rescore_omnimedvqa.run EXACTLY).

    Item keys (verified): question, option_A, option_B, option_C?, option_D?, gt_answer, image_path.
    gt_answer is the full option TEXT (no letters), e.g. "Microscopy". Options list is built the
    same way: [option_A, option_B, (+option_C if truthy), (+option_D if truthy)]. Same
    shuffle(seed=42) + resolve_image + limit selection as the text run.

    The text harness prompt has NO letters (it asks for the option text). To score parse-free we
    must letter the options, so we add "(A) ...\\n" formatting + a letter cue. gold index = the
    position whose option text == gt_answer. NOTE this is the one place where the logit prompt
    necessarily differs from the OmniMedVQA text prompt (the released benchmark is letter-free); the
    ITEMS, options and gold are identical — only the elicitation switches from text to letter, which
    is exactly the parse-free method's premise.
    """
    from PIL import Image

    def resolve_image(image_path):
        for cand in [OMNI_ROOT / "OmniMedVQA" / image_path, OMNI_ROOT / image_path,
                     OMNI_ROOT / "OmniMedVQA" / "Images" / image_path]:
            if cand.exists():
                return cand
        return None

    files = glob.glob(str(OMNI_ROOT / "**" / "QA_information" / "Open-access" / "*.json"),
                      recursive=True)
    data = []
    for f in files:
        data += json.load(open(f))
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
    for item, ip in rows:
        opts = [item["option_A"], item["option_B"]]
        if item.get("option_C"):
            opts.append(item["option_C"])
        if item.get("option_D"):
            opts.append(item["option_D"])
        gt = item["gt_answer"]
        k = len(opts)
        opt_text = "".join(f"({chr(65 + i)}) {o}\n" for i, o in enumerate(opts))
        text = (f"{item['question']}\n\n{opt_text}\n"
                "Answer with the option's letter from the given choices directly.")
        # gold = index whose option text matches gt_answer (string-compare, like the text scorer).
        gold_idx = next((i for i, o in enumerate(opts) if str(o) == str(gt)), None)
        if gold_idx is None:  # gt not among options (shouldn't happen) -> skip, can't score fairly
            continue
        yield {"image": prep(Image.open(ip)), "text": text, "gold": gold_idx, "k": k}


def load_mmad(limit, datasets=("DS-MVTec",)):
    """MMAD DS-MVTec Anomaly-Detection items (mirrors rescore.load_ad_questions EXACTLY).

    Item keys (verified from mmad.json): per file v has image_path + conversation[]; each
    Anomaly-Detection turn has Question (str), Options (dict {'A':'Yes.','B':'No.'}), Answer
    (a letter, e.g. 'A'). Same DS-MVTec local-path resolution + os.path.exists filter + limit.

    Answer cue: the SAME suffix the text run used — MMAD's released instruction
    "Answer with the option's letter from the given choices directly." Body matches rescore.run:
    "Question: {q} \\n{A. opt\\nB. opt\\n}{SUFFIX}". gold = letter -> index.
    """
    from _image_util import load_image

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
    if limit:
        out = out[:limit]
    print(f"{len(out)} Anomaly-Detection questions")
    for q in out:
        opts = q["options"]                       # ordered dict {A: text, B: text, ...}
        keys = list(opts.keys())                  # ['A','B',...]
        k = len(keys)
        # MMAD's own body: "A. opt\nB. opt\n" then the suffix (rescore.run, verbatim).
        options_text = "".join(f"{kk}. {vv}\n" for kk, vv in opts.items())
        text = f"Question: {q['question']} \n{options_text}{MMAD_SUFFIX}"
        gold_idx = keys.index(q["gold"])          # q["gold"] is a letter
        # load_image downsamples to 560px (longest side), matching the inference-time preprocessing.
        img = load_image(Path(q["image"]), max_size=560)
        yield {"image": img, "text": text, "gold": gold_idx, "k": k}


LOADERS = {
    "mmmu": load_mmmu,
    "scienceqa": load_scienceqa,
    "omnimedvqa": load_omnimedvqa,
    "mmad": load_mmad,
}


# ----------------------------------------------------------------------------- #
#  DRIVER                                                                         #
# ----------------------------------------------------------------------------- #
def run(benchmark, model_key, limit):
    import torch

    backend, name = MODELS[model_key]
    # (model is passed to load_model directly)
    print(f"Loading {name} (backend={backend}) for parse-free logit scoring on {benchmark} ...")
    model, tok = load_model(backend, name)
    device = next(model.parameters()).device

    items = LOADERS[benchmark](limit)

    res_ok = 0
    n = 0
    k_sum = 0
    unmapped_sum = 0
    t0 = time.time()
    jsonl = OUT / f"logit_{benchmark}_{model_key}.jsonl"
    with open(jsonl, "w") as fh:
        for it in items:
            k = it["k"]
            gold_idx = it["gold"]
            pred_idx, unmapped = letter_logit_score(
                model, tok, it["image"], it["text"], k, device)
            ok = int(pred_idx == gold_idx)
            res_ok += ok
            n += 1
            k_sum += k
            unmapped_sum += unmapped
            fh.write(json.dumps({
                "gold_idx": gold_idx,       # gold letter index 0..k-1
                "gold_letter": chr(65 + gold_idx),
                "pred": pred_idx,           # predicted letter index 0..k-1
                "pred_letter": chr(65 + pred_idx),
                "ok": ok,
                "k": k,
                "unmapped": unmapped,       # 1 if any letter had no token id
            }) + "\n")
            if n % 100 == 0:
                print(f"  [{n}] logit_acc={res_ok / n:.3f} {round(time.time() - t0)}s")

    summary = {
        "benchmark": benchmark,
        "model": model_key,
        "n": n,
        "logit_acc": round(res_ok / n, 4) if n else 0.0,
        "k_mean": round(k_sum / n, 4) if n else 0.0,
        "unmapped_rate": round(unmapped_sum / n, 4) if n else 0.0,
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    print("SUMMARY", json.dumps(summary))
    with open(OUT / "summary_logit.jsonl", "a") as fh:
        fh.write(json.dumps(summary) + "\n")

    # Cleanup — mirror rescore_mmmu.py's teardown so the GPU is released between runs.
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Parse-free likelihood (logit) MCQ re-scorer.")
    ap.add_argument("--benchmark", required=True, choices=list(LOADERS.keys()))
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--limit", type=int, default=None,
                    help="Pass-through item cap, applied per loader exactly like the text re-scorers.")
    a = ap.parse_args()
    run(a.benchmark, a.model, a.limit)
