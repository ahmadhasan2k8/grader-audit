"""Second-anchor adjudication manifest: PMC-VQA (medical, k=4), where the same no-threshold snap
inflates Pixtral but DEFLATES verbose Llama. Re-joins question/options from the clean split (same order
the rescorer used), joins the saved responses by index (gold-verified), samples the non-compliant items
(+ compliant controls), and writes blinded + full manifests for adjudicate.py and the Opus judge."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "data")
from rescore_pmcvqa import load_clean_split, LETTERS  # noqa: E402

HERE = Path("data")


def main():
    rng = random.Random(20260625)
    print("loading PMC-VQA clean split (question/options/gold; one-time image decode)...")
    clean = load_clean_split()
    rows = []
    for model in ("pixtral-12b", "llama-3.2-11b"):
        jf = HERE / f"pmcvqa_{model}.jsonl"
        if not jf.exists():
            print(f"  {model}: no {jf}; skip"); continue
        saved = [json.loads(l) for l in open(jf)]
        n = len(saved)
        meta = clean[:n]
        mism = sum(s["gold"] != m["gold_label"] for s, m in zip(saved, meta))
        if mism:
            print(f"  {model}: GOLD MISMATCH {mism}/{n} -- join order wrong, ABORT"); return
        joined = [{"idx": i, "nc": bool(s.get("abstain")), **s, **m}
                  for i, (s, m) in enumerate(zip(saved, meta))]
        nc = [r for r in joined if r["nc"]]
        cc = [r for r in joined if not r["nc"]]
        rng.shuffle(nc); rng.shuffle(cc)
        pick = nc[:40] + cc[:8]
        print(f"  {model}: {len(nc)} non-compliant / {len(cc)} compliant; sampling {len(pick)}")
        for r in pick:
            opts = r["option_texts"]
            rows.append({
                "uid": f"pmcvqa-{model}-{r['idx']}",
                "benchmark": "pmcvqa", "model": model, "idx": r["idx"],
                "question": r["question"],
                "options": [{"letter": L, "text": t} for L, t in zip(LETTERS, opts)],
                "response": r["resp"],
                "_gold": r["gold"],
                "_snap_pred": r.get("snap_pred"),
                "_noncompliant": r["nc"],
            })
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["order"] = i
    (HERE / "adjudication_manifest_pmcvqa.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    (HERE / "adjudication_blinded_pmcvqa.jsonl").write_text(
        "".join(json.dumps({k: r[k] for k in ("uid", "benchmark", "question", "options", "response")}) + "\n"
                for r in rows))
    nnc = sum(r["_noncompliant"] for r in rows)
    print(f"\nwrote adjudication_manifest_pmcvqa.jsonl + blinded: {len(rows)} items ({nnc} non-compliant)")


if __name__ == "__main__":
    main()
