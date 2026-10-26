#!/usr/bin/env python3
"""Human MCQ adjudication tool (you are the annotator).

You read each model RESPONSE, blinded to the gold answer and to every scoring rule's verdict, and
record the single thing that matters: which option did the model actually COMMIT to -- or did it never
commit (NONE)? That label becomes the ground truth every scoring rule is measured against.

Run it from the repo root in your OWN terminal (it's interactive):
    python data/adjudicate.py                 # the real 130-item run
    python data/adjudicate.py --trial         # a 10-item UX trial (separate file)
Resumable: saves after every keystroke; re-run to continue. ~40-60 items/hour is typical.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path("data")


def read_labels(path):
    return {json.loads(l)["uid"]: json.loads(l)["label"]
            for l in open(path) if l.strip()} if path.exists() else {}


def rewrite_labels(done, path):
    with open(path, "w") as fh:
        for uid, lab in done.items():
            fh.write(json.dumps({"uid": uid, "label": lab}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", action="store_true", help="10-item UX trial (separate trial files)")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--labels", default=None)
    a = ap.parse_args()
    if a.trial:
        manifest = HERE / "adjudication_trial.jsonl"
        labels = HERE / "trial_labels.jsonl"
    else:
        manifest = Path(a.manifest) if a.manifest else HERE / "adjudication_manifest.jsonl"
        labels = Path(a.labels) if a.labels else HERE / "human_labels.jsonl"

    items = sorted((json.loads(l) for l in open(manifest)), key=lambda r: r.get("order", 0))
    done = read_labels(labels)
    total = len(items)
    print("\n" + "=" * 72)
    print(f" MCQ ADJUDICATION{'  [TRIAL]' if a.trial else ''}   {len(done)}/{total} done   ->  {labels.name}")
    print(" Read the MODEL RESPONSE. Which option did it COMMIT to as its final answer?")
    print(" Keys:  letter (A/B/C/D/E) = that option   N = none / never committed")
    print("        M = multiple/contradictory   S = skip   U = undo last   Q = save & quit")
    print("=" * 72)

    order = [r for r in items if r["uid"] not in done]
    last_uid = None
    k = 0
    while k < len(order):
        r = order[k]
        letters = [o["letter"] for o in r["options"]]
        n_done = sum(1 for it in items if it["uid"] in done)
        print(f"\n[{n_done + 1}/{total}]  ({r['benchmark']} / {r.get('model','?')})")
        print(f"  Q: {r['question']}")
        for o in r["options"]:
            print(f"      ({o['letter']}) {o['text']}")
        print("\n  MODEL RESPONSE:")
        print(f"      {r['response']!r}")
        try:
            ans = input(f"\n  commit to {('/'.join(letters))} / N / M / S / U / Q ? > ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print("\n  saved, bye."); return
        if ans == "Q":
            print("  saved, bye."); return
        if ans == "U":
            if last_uid and last_uid in done:
                done.pop(last_uid); rewrite_labels(done, labels)
                order.insert(k, next(it for it in items if it["uid"] == last_uid))
                print(f"  undid {last_uid}"); last_uid = None
            else:
                print("  (nothing to undo)")
            continue
        if ans == "S":
            k += 1; continue
        if not (ans == "N" or ans == "M" or ans in letters):
            print(f"  ?? type one of: {', '.join(letters)} / N / M / S / U / Q")
            continue
        done[r["uid"]] = ans
        with open(labels, "a") as fh:
            fh.write(json.dumps({"uid": r["uid"], "label": ans}) + "\n")
        last_uid = r["uid"]; k += 1

    print(f"\n  All {total} items labeled ({labels.name}). "
          + ("Trial done -- run without --trial for the real 130." if a.trial else "Run adjudication_analysis.py next."))


if __name__ == "__main__":
    main()
