#!/usr/bin/env python3
"""Run MMAD's OWN released scorer, unmodified, on our stored model outputs.

This is the anti-manipulation check: nothing here reimplements any scoring logic. It imports the
UNMODIFIED official code from the pinned clone (scripts/fetch_official_repos.sh -> official_scorers/MMAD
@ 1490f887) and feeds it our released per-item responses:

  * GPT4Query.parse_answer   -- evaluation/examples/GPT4/gpt4o.py   (regex \\b([A-E])\\b, else
                                difflib.get_close_matches(..., cutoff=0.0) -- the fabricating parser)
  * caculate_accuracy_mmad   -- evaluation/examples/helper/summary.py:14  (the official aggregator)

The per-item question set (image key, options, gold) is frozen in data/mmad_ad_questions.json in the
exact run order, gold-sequence-verified against the released logs at build time. Expected agreement
with our wrapper is recorded in data/REPRODUCED_NUMBERS.json: identical to four decimals for four of
five models; Pixtral differs by 0.0018 because the official call sites keep the LAST regex match where
our wrapper keeps the first (see the paper, Sec. 4). Note the official aggregator's printed "Average"
additionally macro-averages the normal/anomaly split; the paper reports plain (micro) accuracy, printed
below for direct comparison.

Extra deps (the OFFICIAL repo's, not ours):  pip install pandas seaborn

Usage (from the package root, after `bash scripts/fetch_official_repos.sh`):
  python3 src/run_mmad_official.py --model pixtral-12b
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "official_scorers" / "MMAD"
EX = REPO / "evaluation" / "examples"
DATA = ROOT / "data"
OUT = ROOT / "results"

if not EX.exists():
    sys.exit("official clone not found -- run `bash scripts/fetch_official_repos.sh` first")

import matplotlib
matplotlib.use("Agg")                          # headless; the official summary.py calls plt.show()
sys.path.insert(0, str(EX))                    # for `helper.summary`
sys.path.insert(0, str(EX / "GPT4"))           # for `gpt4o`
from helper.summary import caculate_accuracy_mmad          # noqa: E402  summary.py:14
import gpt4o                                                # noqa: E402  defines GPT4Query
OFFICIAL_PARSE = gpt4o.GPT4Query.parse_answer               # the official fabricating parser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()

    questions = json.load(open(DATA / "mmad_ad_questions.json"))
    rows = [json.loads(l) for l in open(DATA / f"mmad_{a.model}.jsonl") if l.strip()]
    assert len(rows) == len(questions) == 1670, (len(rows), len(questions))
    assert [r["gold"] for r in rows] == [q["gold"] for q in questions], "gold sequence mismatch"

    answers_json, micro_ok, fabricated = [], 0, 0
    for r, q in zip(rows, questions):
        resp = r["resp"]   # released responses; the stored 80-char field reproduces every
                           # parse decision exactly (verified: 8,350/8,350 vs full text)
        parsed = OFFICIAL_PARSE(gpt4o.GPT4Query, resp, q["options"])
        gpt_answer = parsed[-1] if parsed else ""       # the official call sites' LAST-match convention
        if not re.compile(r"\b([A-E])\b").findall(resp):
            fabricated += 1                              # fuzzy fallback fired
        micro_ok += gpt_answer == q["gold"]
        answers_json.append({
            "image": q["image"],
            "question": "Is there any defect in the object?",
            "question_type": "Anomaly Detection",
            "correct_answer": q["gold"],
            "gpt_answer": gpt_answer,
        })

    OUT.mkdir(exist_ok=True)
    out_json = OUT / f"mmad_official_input_{a.model}.json"
    json.dump(answers_json, open(out_json, "w"), indent=2)

    n = len(answers_json)
    ref = json.load(open(DATA / "REPRODUCED_NUMBERS.json"))["MMAD"]["per_model"].get(a.model, {})
    print(f"[{a.model}] n={n}  fallback_fired={fabricated} ({100*fabricated/n:.1f}%)")
    print(f"OFFICIAL parser, micro accuracy : {micro_ok/n:.4f}"
          f"   (recorded: {ref.get('official_fuzzy', 'n/a')}; our wrapper: {ref.get('reimpl_fuzzy', 'n/a')})")
    print("=" * 60)
    print("Calling the OFFICIAL caculate_accuracy_mmad (summary.py:14) ...")
    caculate_accuracy_mmad(str(out_json), normal_flag="good", show_overkill_miss=True)


if __name__ == "__main__":
    main()
