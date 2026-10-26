"""PMC-VQA (Communications Medicine 2024, Nature Portfolio; arXiv 2305.10415) released-scorer
artifact, quantified on our 5 VLMs.

Official scorer (PMC-VQA repo src/MedVInT_TD/test.py & src/MedVInT_TE/test.py, pinned commit
1678ec2c97f0b2bf84591ddccd2e84932630e450). Both define, VERBATIM:

    def str_similarity(str1, str2):
        seq = difflib.SequenceMatcher(None, str1, str2)
        return seq.ratio()
    def find_most_similar_index(str_list, target_str):
        most_similar_index = None; highest_similarity = 0
        for i, s in enumerate(str_list):
            similarity = str_similarity(s, target_str)
            if similarity > highest_similarity:
                most_similar_index = i; highest_similarity = similarity
        return most_similar_index

i.e. snap the model output to the closest option by SequenceMatcher.ratio() with NO threshold and NO
abstain branch -- it ALWAYS returns an option index (the lone edge: returns None iff every ratio is
exactly 0.0, which for a non-empty response over option strings essentially never happens). The paper
ENDORSES this in Methods 4.3.6: "For the generative model, we calculate ACC scores by matching the
model's output with the options using difflib.SequenceMatcher and choosing the most similar one ...".
There is NO GPT/LLM/regex extraction buffer in front of the snap (unlike MathVista) -- the snap eats
the model text directly. So for a rambling general VLM the snap fabricates a choice on non-compliant
outputs instead of abstaining.

The two official scripts differ ONLY in what string they feed the snap, and over which candidate list:
  * MedVInT_TD/test.py (the PMC-VQA *choice-task* MCQ scorer, lines 152-156): snaps
    generated_texts[i][-1]  -- the LAST CHARACTER of the generated text -- over the 4 option TEXTS
    [Choice_A..D].  (Its model is constrained to emit a single letter, so the last char is the answer.)
  * MedVInT_TE/test.py (lines 126-135): snaps the full generated_text over the LETTER list
    ['A','B','C','D'].
We reproduce BOTH against a rambling general VLM:
  - official_snap        : full response snapped over the 4 option TEXTS (the protocol Methods 4.3.6
                           describes for the general/unconstrained generative case). PRIMARY.
  - official_snap_lastchar: response[-1] snapped over the 4 option TEXTS (verbatim MedVInT_TD). Reported
                           as a sensitivity.
We compare to an honest abstain reference (clean_match) that abstains when the response states no
clear single choice. Artifact = snap_acc - clean_acc = the free credit the no-threshold snap grants on
non-compliant outputs.

Images: xmcmic/PMC-VQA itself ships ONLY Figure_path strings + images.zip/images_2.zip (~19 GB), no
inline PIL. We instead load images-inline from hamzamooraj99/PMC-VQA-1 (parquet, Image() feature,
CC-BY-SA-4.0, not gated) whose `test` split embeds the SAME 50k rows as xmcmic test.csv, and inner-join
it onto xmcmic test_clean.csv (the paper's verified 2,000-question clean test split) on the unique key
(Figure_path, Question). Verified on a shard: join is 1:1 and gold-label + choice-text agreement = 1.000.
TODO if hamzamooraj99/PMC-VQA-1 is unavailable: download xmcmic/PMC-VQA images.zip, unzip to a local
dir, set PMCVQA_IMG_DIR, and the loader falls back to reading <dir>/<Figure_path>.

Usage (requires a 16 GB GPU + transformers/unsloth):
    python data/rescore_pmcvqa.py --model qwen2.5-vl-7b --limit 100
"""
from __future__ import annotations

import argparse
import difflib
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "data")
from rescore import MODELS, load_model  # noqa: E402

OUT = Path("data")
INLINE_REPO = "hamzamooraj99/PMC-VQA-1"   # parquet w/ inline Image(); test split == xmcmic test.csv (50k)
CLEAN_REPO = "xmcmic/PMC-VQA"             # source of test_clean.csv (the verified 2,000 clean test split)
CLEAN_CSV = "test_clean.csv"
LETTERS = ["A", "B", "C", "D"]


def str_similarity(str1, str2):
    """PMC-VQA verbatim (src/MedVInT_TD/test.py:55-57, src/MedVInT_TE/test.py:50-52)."""
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()


def find_most_similar_index(str_list, target_str):
    """PMC-VQA verbatim (both test.py). Closest by SequenceMatcher.ratio(), NO threshold, NO abstain.
    Returns None ONLY if every ratio is exactly 0.0 (essentially never for a non-empty response)."""
    most_similar_str = None
    most_similar_index = None
    highest_similarity = 0
    for i, s in enumerate(str_list):
        similarity = str_similarity(s, target_str)
        if similarity > highest_similarity:
            most_similar_str = s
            most_similar_index = i
            highest_similarity = similarity
    return most_similar_index


def official_snap_letters(resp):
    """PRIMARY. Verbatim MedVInT_TE/test.py:134: find_most_similar_index(['A','B','C','D'], generated).
    Snap the FULL response over the LETTER list. This is the letter-space scorer -> it is compatible
    with a letter prompt (a compliant 'B' snaps to B; the clean reference also reads 'B', so they AGREE
    on compliant items and differ only on non-compliant ones where the snap fabricates a letter). Returns
    None only if no A/B/C/D char overlaps the response at all (an effective partial abstention edge)."""
    return find_most_similar_index(LETTERS, resp)


def official_snap_text(resp, option_texts):
    """Sensitivity (Methods 4.3.6 read literally: 'matching the model's output with the options' = option
    TEXTS): snap the FULL response over the 4 option TEXTS. NOTE: under a LETTER prompt this is noisy --
    a compliant single letter shares ~no characters with long option texts, so it does NOT cleanly reward
    compliance; reported only for transparency, not as the headline."""
    return find_most_similar_index(option_texts, resp)


def official_snap_lastchar(resp, option_texts):
    """Sensitivity. Verbatim MedVInT_TD/test.py:154-155: snap the LAST CHARACTER of the response over
    the 4 option TEXTS. Degenerate for free-form VLM text (last char is usually punctuation)."""
    pred = resp[-1] if resp else ""
    return find_most_similar_index(option_texts, pred)


def clean_match(resp, option_texts, letters):
    """Honest abstain reference (mirrors rescore_mathvista.clean_match / rescore_mmmu.strict_parse).
    Return the chosen index iff the response clearly contains EXACTLY ONE option's letter ("(A)" / " A "
    style) or EXACTLY ONE option's full text; else None (abstain -> scored wrong)."""
    r = " " + resp.strip(",.!?;:'\"") + " "
    # 1) bracketed letter "(A)"
    hits = [i for i, L in enumerate(letters) if f"({L})" in r]
    if len(hits) == 1:
        return hits[0]
    # 2) isolated letter " A "
    if not hits:
        hits = [i for i, L in enumerate(letters) if f" {L} " in r]
        if len(hits) == 1:
            return hits[0]
    # 3) exactly one option's full text contained in the response
    low = resp.lower().strip()
    thits = [i for i, o in enumerate(option_texts) if str(o).lower().strip() and str(o).lower().strip() in low]
    if len(thits) == 1:
        return thits[0]
    for i, o in enumerate(option_texts):
        if low == str(o).lower().strip():
            return i
    return None  # ambiguous / no clear single choice -> abstain


def strip_choice_text(choice):
    """Choices in the CSV look like ' A: Chromatin condensation ' or ' A:X-ray '. Return the option TEXT
    only (drop the leading ' X:' label) so the snap matches on content, matching how PMC_QA_Dataset
    composes the option strings the scorer compares against."""
    s = str(choice).strip()
    for L in LETTERS:
        for sep in (f"{L}:", f"{L} :", f"{L}."):
            if s.startswith(sep):
                return s[len(sep):].strip()
    return s


def prep(pil, max_size=560):
    img = pil.convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        s = max_size / max(w, h)
        img = img.resize((int(w * s), int(h * s)))
    return img


def load_clean_split():
    """Build the verified 2,000-question clean PMC-VQA test split WITH inline images.

    Strategy: load hamzamooraj99/PMC-VQA-1 `test` (inline Image(); same 50k rows as xmcmic test.csv) and
    inner-join onto xmcmic test_clean.csv on the unique key (Figure_path, Question). Returns a list of
    dicts: {image (PIL), question, choices [4 raw], option_texts [4 stripped], gold_label ('A'..'D')}.

    If PMCVQA_IMG_DIR is set, instead read test_clean.csv directly and open <PMCVQA_IMG_DIR>/<Figure_path>
    (use this if you unzip xmcmic/PMC-VQA images.zip yourself).
    """
    import pandas as pd
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    clean_csv = hf_hub_download(CLEAN_REPO, CLEAN_CSV, repo_type="dataset")
    clean = pd.read_csv(clean_csv)
    clean["_q"] = clean["Question"].map(lambda s: str(s).strip())

    img_dir = os.environ.get("PMCVQA_IMG_DIR")
    rows = []
    if img_dir:
        from PIL import Image
        for _, r in clean.iterrows():
            p = Path(img_dir) / r["Figure_path"]
            if not p.exists():
                continue
            choices = [r["Choice A"], r["Choice B"], r["Choice C"], r["Choice D"]]
            rows.append({
                "image": Image.open(p),
                "question": str(r["Question"]).strip(),
                "choices": choices,
                "option_texts": [strip_choice_text(c) for c in choices],
                "gold_label": str(r["Answer_label"]).strip(),
            })
        return rows

    # Inline path: index the clean rows by (Figure_path, Question), then walk the inline test split once.
    want = {}
    for _, r in clean.iterrows():
        choices = [r["Choice A"], r["Choice B"], r["Choice C"], r["Choice D"]]
        want[(r["Figure_path"], r["_q"])] = {
            "question": str(r["Question"]).strip(),
            "choices": choices,
            "option_texts": [strip_choice_text(c) for c in choices],
            "gold_label": str(r["Answer_label"]).strip(),
        }
    ds = load_dataset(INLINE_REPO, split="test")  # downloads ~5 GB of test parquet on first run
    # Locate the ~2000 clean rows by (Figure_path, Question) over the 50k inline split WITHOUT decoding
    # every image: scan the two string columns only (select_columns), then ds.select() + decode the
    # matches. (Iterating the full Image()-feature split would decode all 50k images -> very slow, x5 runs.)
    key_ds = ds.select_columns(["Figure_path", "Question"])
    idx_for_key = {}
    for i, ex in enumerate(key_ds):
        k = (ex["Figure_path"], str(ex["Question"]).strip())
        if k in want and k not in idx_for_key:
            idx_for_key[k] = i
    keys = list(idx_for_key.keys())
    sel = ds.select([idx_for_key[k] for k in keys])           # ~2000 matched rows; decode only these
    for ex, k in zip(sel, keys):
        rows.append({"image": ex["image"], **want[k]})
    return rows


def run(model_key, limit):
    import torch
    backend, name = MODELS[model_key]
    print(f"Loading {name} ...")
    model, tok = load_model(backend, name)
    device = next(model.parameters()).device

    rows = load_clean_split()
    if limit:
        rows = rows[:limit]
    print(f"{len(rows)} PMC-VQA clean-test questions (with inline images)")

    res = {k: 0 for k in ("snap", "snap_txt", "snap_lc", "clean", "abstain", "snapped")}
    n = 0
    t0 = time.time()
    jsonl = OUT / f"pmcvqa_{model_key}.jsonl"
    with open(jsonl, "w") as fh:
        for r in rows:
            option_texts = r["option_texts"]
            gold = r["gold_label"]                      # 'A'/'B'/'C'/'D'
            gold_idx = LETTERS.index(gold) if gold in LETTERS else None
            opt_block = "".join(f"({L}) {t}\n" for L, t in zip(LETTERS, option_texts))
            user = (f"{r['question']}\n\n{opt_block}\n"
                    f"Answer with the option's letter from the given choices directly.")
            img = prep(r["image"])
            chat = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]}]
            pt = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
            inp = tok(text=pt, images=[img], return_tensors="pt", padding=True)
            inp = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=256, do_sample=False)
            il = inp["input_ids"].shape[-1]
            resp = tok.decode(gen[0][il:], skip_special_tokens=True).strip()

            si_let = official_snap_letters(resp)                # PRIMARY: verbatim MedVInT_TE (over letters)
            snap_ok = int(si_let is not None and si_let == gold_idx)
            si_txt = official_snap_text(resp, option_texts)     # sensitivity: full resp over option texts
            snap_txt_ok = int(si_txt is not None and si_txt == gold_idx)
            sli = official_snap_lastchar(resp, option_texts)    # sensitivity: verbatim MedVInT_TD (last char)
            snap_lc_ok = int(sli is not None and sli == gold_idx)
            ci = clean_match(resp, option_texts, LETTERS)       # honest abstain reference
            clean_ok = int(ci is not None and ci == gold_idx)

            n += 1
            res["snap"] += snap_ok
            res["snap_txt"] += snap_txt_ok
            res["snap_lc"] += snap_lc_ok
            if ci is None:
                res["abstain"] += 1
                res["snapped"] += 1            # the snap is doing the work where clean would abstain
            else:
                res["clean"] += clean_ok
            fh.write(json.dumps({
                "gold": gold,
                "snap_pred": (LETTERS[si_let] if si_let is not None else None),
                "snap_text_pred": (LETTERS[si_txt] if si_txt is not None else None),
                "snap_lastchar_pred": (LETTERS[sli] if sli is not None else None),
                "clean_pred": (LETTERS[ci] if ci is not None else None),
                "abstain": int(ci is None),
                "k": len(option_texts),
                "resp": resp,
            }) + "\n")
            if n % 100 == 0:
                print(f"  [{n}/{len(rows)}] {round(time.time()-t0)}s "
                      f"snap={res['snap']/n:.3f} clean={res['clean']/n:.3f}")

    summary = {
        "model": model_key, "benchmark": "pmcvqa", "n": n,
        "snap_acc": round(res["snap"] / n, 4),                 # PRIMARY: MedVInT_TE letter-snap
        "snap_text_acc": round(res["snap_txt"] / n, 4),        # sensitivity: full resp over option texts
        "snap_lastchar_acc": round(res["snap_lc"] / n, 4),     # sensitivity: MedVInT_TD last char over texts
        "clean_acc": round(res["clean"] / n, 4),
        "abstain_rate": round(res["abstain"] / n, 4),
        "snap_fire_rate": round(res["snapped"] / n, 4),   # fraction where clean abstains so snap decides
        "artifact_snap_minus_clean": round((res["snap"] - res["clean"]) / n, 4),       # PRIMARY artifact
        "artifact_text_minus_clean": round((res["snap_txt"] - res["clean"]) / n, 4),
        "artifact_lastchar_minus_clean": round((res["snap_lc"] - res["clean"]) / n, 4),
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    print("SUMMARY", json.dumps(summary))
    with open(OUT / "summary_pmcvqa.jsonl", "a") as fh:
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
