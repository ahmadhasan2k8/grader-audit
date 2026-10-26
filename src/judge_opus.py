#!/usr/bin/env python3
"""Blinded Claude-Opus-4.8 judge over the adjudication manifest -- the SAME items the human labels
(adjudicate.py), so we can report judge-vs-human agreement (kappa) and scale the adjudication beyond
the human-audited subset.

The judge sees ONLY the question, options, and model response (never the gold answer or any scoring
rule's verdict). It outputs which option the model COMMITTED to, or NONE / MULTIPLE -- a competent
reader extracting the model's actual answer (conceptually the "extraction buffer" that defuses the
crude snap, per FINDINGS_mathvista_local.md).

Needs the anthropic SDK + ANTHROPIC_API_KEY (the human tool needs neither). Resumable; temperature 0.
    JUDGE_MODEL=claude-opus-4-8 ANTHROPIC_API_KEY=... python data/judge_opus.py
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

HERE = Path("data")
MANIFEST = HERE / "adjudication_manifest.jsonl"
OUT = HERE / "judge_labels.jsonl"
MODEL = os.environ.get("JUDGE_MODEL", "claude-opus-4-8")

PROMPT = """You are auditing whether a model committed to a multiple-choice answer. Read ONLY the model's response below and decide which option the model COMMITTED TO as its final answer.
- If it clearly committed to exactly one option, output that option's LETTER.
- If it never committed to any option (rambled, hedged, refused, ran out, or gave a non-option), output NONE.
- If it committed to two or more contradictory options, output MULTIPLE.
Judge only what the model SAID. Do NOT solve the question yourself; the correct answer is irrelevant.

Question: {q}
Options:
{opts}

Model response:
{resp}

Output exactly one token: a single option letter (A/B/...), or NONE, or MULTIPLE."""


def parse(txt):
    t = txt.strip().upper()
    m = re.match(r"\(?([A-E])\)?$|^(NONE)$|^(MULTIPLE)$", t)
    if m:
        return next(g for g in m.groups() if g)
    m2 = re.search(r"\b([A-E]|NONE|MULTIPLE)\b", t)
    return m2.group(1) if m2 else t[:10]


def main():
    try:
        import anthropic
    except ImportError:
        print("ERROR: `pip install anthropic` first."); return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY to run the Opus judge (the human tool needs no key)."); return
    client = anthropic.Anthropic()
    items = [json.loads(l) for l in open(MANIFEST)]
    done = {json.loads(l)["uid"] for l in open(OUT)} if OUT.exists() else set()
    todo = [r for r in items if r["uid"] not in done]
    print(f"judging {len(todo)}/{len(items)} items with {MODEL} ...")
    for i, r in enumerate(todo):
        opts = "\n".join(f"({o['letter']}) {o['text']}" for o in r["options"])
        try:
            msg = client.messages.create(
                model=MODEL, max_tokens=8, temperature=0,
                messages=[{"role": "user", "content": PROMPT.format(
                    q=r["question"], opts=opts, resp=r["response"])}])
            raw = msg.content[0].text
            label = parse(raw)
        except Exception as e:
            print(f"  {r['uid']}: ERROR {e}; sleep 10"); time.sleep(10); continue
        with open(OUT, "a") as fh:
            fh.write(json.dumps({"uid": r["uid"], "label": label, "raw": raw.strip()}) + "\n")
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(todo)}")
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
