"""Build the blinded adjudication manifest: the SAME item list a human (adjudicate.py) and the Opus
judge (judge_opus.py) both label, so we can compute judge-vs-human agreement and anchor every scoring
rule to "what the model actually answered."

v1 = MMAD (our anchor; binary Yes/No; responses are COMPLETE because MMAD generates only 16 tokens by
its own protocol, which is why we reproduce its numbers to 4 decimals). We re-derive the question +
options + gold from mmad.json in the SAME order the rescorer used, JOIN with the saved per-item
responses by index, and VERIFY the join by checking the gold sequence matches exactly (aborts if not).

Each manifest row stores BLINDED display fields (question, options, response) and HIDDEN ground-truth
fields prefixed with "_" (gold, the snap/abstain verdicts, non-compliance) that the labelers never see.
Sampling stratifies toward the non-compliance items (where the rules disagree and adjudication matters)
plus a few compliant controls. Shuffled (seeded) so labels aren't biased by grouping.

Extending later: MMMU/ScienceQA (full responses, re-join options from the HF dataset), OmniMedVQA
(storage-truncated to 80 chars -> needs a full-response re-gen first).
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

HERE = Path("data")
MMAD_JSON = "data/mmad.json"
MVTEC_ROOT = "research/mvtec_data"


def load_mmad_text(datasets=("DS-MVTec",)):
    """Reproduce rescore.load_ad_questions ORDER exactly (incl. the os.path.exists filter), text-only.
    Yields dicts: {question, options {A:..,B:..}, gold (letter)}."""
    d = json.load(open(MMAD_JSON))
    out = []
    for key, v in d.items():
        ds = key.split("/")[0]
        if ds not in datasets:
            continue
        if ds == "DS-MVTec":
            cat = key.split("/")[1]
            local = f"{MVTEC_ROOT}/{cat}/" + v["image_path"].replace("image/", "test/", 1)
        else:
            continue
        if not os.path.exists(local):
            continue
        for qa in v["conversation"]:
            if qa["type"] == "Anomaly Detection":
                out.append({"question": qa["Question"], "options": qa["Options"], "gold": qa["Answer"]})
    return out


def add_mmad(rows_out, model, n_noncompliant, n_compliant, rng):
    jf = HERE / f"{model}.jsonl"
    if not jf.exists():
        print(f"  MMAD {model}: no per-item file {jf} -- skip"); return
    saved = [json.loads(l) for l in open(jf) if l.strip()]
    meta = load_mmad_text()
    if len(saved) != len(meta):
        print(f"  MMAD {model}: LENGTH MISMATCH saved={len(saved)} meta={len(meta)} -- skip (cannot join)"); return
    # VERIFY join order by gold sequence
    mism = sum(s["gold"] != m["gold"] for s, m in zip(saved, meta))
    if mism:
        print(f"  MMAD {model}: GOLD MISMATCH on {mism}/{len(saved)} rows -- join order wrong, ABORT"); return
    joined = []
    for i, (s, m) in enumerate(zip(saved, meta)):
        noncompliant = s.get("strict_pred") is None      # snap fired = model didn't cleanly state a choice
        joined.append({"idx": i, "noncompliant": noncompliant, **s, **m})
    nc = [r for r in joined if r["noncompliant"]]
    cc = [r for r in joined if not r["noncompliant"]]
    rng.shuffle(nc); rng.shuffle(cc)
    pick = nc[:n_noncompliant] + cc[:n_compliant]
    print(f"  MMAD {model}: {len(nc)} non-compliant / {len(cc)} compliant; sampling {len(pick)}")
    for r in pick:
        opts = r["options"]                              # ordered dict {A:'Yes.', B:'No.'}
        rows_out.append({
            "uid": f"mmad-{model}-{r['idx']}",
            "benchmark": "mmad", "model": model, "idx": r["idx"],
            "question": r["question"],
            "options": [{"letter": k, "text": v} for k, v in opts.items()],
            "response": r["resp"],
            # hidden ground truth / rule verdicts (labelers never see these):
            "_gold": r["gold"],
            "_snap_pred": r.get("hard_pred"),
            "_abstain_pred": r.get("strict_pred"),
            "_noncompliant": r["noncompliant"],
        })


def main():
    rng = random.Random(20260624)
    rows = []
    # MMAD anchor: the two non-compliant models carry the effect; sample them heavily + a few controls.
    add_mmad(rows, "pixtral-12b", n_noncompliant=55, n_compliant=10, rng=rng)
    add_mmad(rows, "llama-3.2-11b", n_noncompliant=55, n_compliant=10, rng=rng)
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["order"] = i
    out = HERE / "adjudication_manifest.jsonl"
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    nnc = sum(r["_noncompliant"] for r in rows)
    print(f"\nwrote {out}: {len(rows)} items ({nnc} non-compliant, {len(rows)-nnc} compliant controls)")
    print("blinded display fields: question, options, response  |  hidden: _gold, _snap_pred, _abstain_pred")


if __name__ == "__main__":
    main()
