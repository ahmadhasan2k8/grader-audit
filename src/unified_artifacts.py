"""Single source of truth for the paper's artifact numbers, with CONSISTENT per-family definitions.

Two mechanism families need two artifact definitions (this is the methodological spine of the paper):

  FAMILY A -- UNCONDITIONAL SNAP (MMAD, OmniMedVQA, PMC-VQA): the scorer maps EVERY response to an
    option (no abstention path). It agrees with a clean reference on clearly-stated answers and only
    diverges on rambles, where it fabricates. So:
        artifact = snap_acc - clean_acc        (clean_acc abstains on no-clearly-stated-option)
        non_compliance = abstain_rate of the clean reference (= how often the snap is the decider)

  FAMILY B -- PARSE-THEN-FABRICATE (MMMU, ScienceQA): an explicit parser, then a DISCRETE random
    fallback on parse-failure. The honest artifact holds the official parser FIXED and only swaps the
    fallback (random -> abstain); it must NOT be credited with the parser's own recall (that was the
    +0.06->+0.041 MMMU correction). So:
        artifact = fired_correct / N            (random-fallback correct hits = credit over abstaining)
        non_compliance = random_fire_rate       (= parse-failure rate)

  DEFUSED (MathVista w/ extraction buffer, MathVision abstain-by-design, POPE easy-task): reported with
    artifact ~ 0 and the REASON. These are the credibility backbone -- the bug is conditional, not universal.

Emits unified_artifacts.json = [{benchmark, model, non_compliance, artifact, family, cohort, venue}]
and prints a tidy table + per-benchmark means. Re-run after each new rescore; pending benchmarks are
skipped with a note until their summary/jsonl exists.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

HERE = Path("data")
OFFICIAL = Path("research/official_scorers")
MODELS = ["gemma-4-e4b", "qwen2.5-vl-7b", "qwen3-vl-8b", "pixtral-12b", "llama-3.2-11b"]

VENUE = {
    "mmad": "ICLR'25", "mmmu": "CVPR'24 Oral", "scienceqa": "NeurIPS'22",
    "pmcvqa": "Comm.Med.'24", "omnimedvqa": "CVPR'24", "pope": "EMNLP'23",
    "mathvista": "ICLR'24 Oral", "mathvision": "NeurIPS'24 D&B",
}
rows = []


def add(benchmark, model, non_compliance, artifact, family, cohort):
    rows.append({"benchmark": benchmark, "model": model,
                 "non_compliance": round(float(non_compliance), 4),
                 "artifact": round(float(artifact), 4),
                 "family": family, "cohort": cohort, "venue": VENUE.get(benchmark, "?")})


# ---- FAMILY B: MMMU (clean = fired_correct/N) -------------------------------------------------
for f in sorted(glob.glob(str(HERE / "mmmu_*.jsonl"))):
    data = [json.loads(l) for l in open(f) if l.strip()]
    if not data:
        continue
    m = Path(f).stem.replace("mmmu_", "")
    n = len(data)
    fired = sum(r["fired"] for r in data)
    fired_correct = sum(r["fired"] and (r["mmmu"] == r["gold"]) for r in data)
    add("mmmu", m, fired / n, fired_correct / n, "parse+random", "firing")

# ---- FAMILY B: ScienceQA (clean = fired_correct/N) --------------------------------------------
for f in sorted(glob.glob(str(HERE / "scienceqa_*.jsonl"))):
    data = [json.loads(l) for l in open(f) if l.strip()]
    if not data:
        continue
    m = Path(f).stem.replace("scienceqa_", "")
    n = len(data)
    fired = sum(r["fired"] for r in data)
    fired_correct = sum(r["fired"] and (r["sqa_pred"] == r["gold"]) for r in data)
    add("scienceqa", m, fired / n, fired_correct / n, "parse+random", "firing")

# ---- FAMILY A: OmniMedVQA (snap - clean; non_compl = abstain_rate) ----------------------------
sf = HERE / "summary_omnimedvqa.jsonl"
if sf.exists():
    for l in open(sf):
        s = json.loads(l)
        add("omnimedvqa", s["model"], s["abstain_rate"], s["artifact_fuzzy_minus_clean"], "snap", "firing")

# ---- FAMILY A: PMC-VQA. Its matcher is UNCONDITIONAL -- it runs on every response, not only on parse
# failures -- so snap_acc - clean_acc mixes two mechanisms: (a) the parse-failure FALLBACK, which is what
# this table is about, and (b) the matcher OVERRIDING answers the parser already read cleanly. For llama
# the total (-0.190) is dominated by (b) (-0.210), while the fallback itself is +0.020. Reporting the
# total here would attribute to the fallback an effect it did not cause -- and would print a cell whose
# magnitude exceeds its own firing rate, which is impossible for a fallback. So we report the FALLBACK
# TERM only, and the override is reported separately in the paper (supplementary, "unconditional matcher").
for f in sorted(glob.glob(str(HERE / "pmcvqa_*.jsonl"))):
    data = [json.loads(l) for l in open(f) if l.strip()]
    if not data:
        continue
    m = Path(f).stem.replace("pmcvqa_", "")
    n = len(data)
    fired = [r for r in data if r["abstain"] == 1]        # clean parser failed -> fallback decides
    fallback = (sum(r["snap_pred"] == r["gold"] for r in fired)
                - sum(r["clean_pred"] == r["gold"] for r in fired)) / n
    add("pmcvqa", m, len(fired) / n, fallback, "snap", "firing")

# ---- FAMILY A-ish: POPE (default-to-yes - strict; non_compl = abstain_rate) -------------------
# POPE DOES fabricate (directional default-to-"yes"); it just sits at the LOW-difficulty end of the
# scaling trend (easy yes/no -> ~2% non-compliance -> tiny artifact). So it's a FIRING anchor, not a
# defused-by-design case. (Defused = non-fabricating scorers: MathVista buffer, MathVision abstain.)
sf = HERE / "summary_pope.jsonl"
if sf.exists():
    for l in open(sf):
        s = json.loads(l)
        add("pope", s["model"], s["abstain_rate"], s["artifact_pope_minus_strict"], "default-class", "firing")

# ---- FAMILY A: MMAD (computed from released per-item logs: artifact = snap - abstain) ----------
for f in sorted(glob.glob(str(HERE / "mmad_*.jsonl"))):
    if "budget" in Path(f).name:   # budget-robustness logs are not separate models
        continue
    data = [json.loads(l) for l in open(f) if l.strip()]
    if not data:
        continue
    m = Path(f).stem.replace("mmad_", "")
    n = len(data)
    snap = sum(r["hard_pred"] == r["gold"] for r in data) / n
    abst = sum(r["strict_pred"] == r["gold"] for r in data) / n
    nonc = sum(r["strict_pred"] is None for r in data) / n
    add("mmad", m, nonc, snap - abst, "snap", "firing")

# ---- DEFUSED: MathVista (faithful local-extractor; artifact ~0; non_compl = post-extract fire) -
sf = HERE / "summary_mathvista_local.jsonl"
if sf.exists():
    for l in open(sf):
        s = json.loads(l)
        add("mathvista", s["test_model"], s["snap_fire_rate"], s["artifact"], "snap(+extract buffer)", "defused")

# ---- DEFUSED: MathVision (abstain-by-design; artifact=0; non_compl = noncompliance_rate) -------
sf = HERE / "summary_mathvision.jsonl"
if sf.exists():
    for l in open(sf):
        s = json.loads(l)
        add("mathvision", s["model"], s["noncompliance_rate"], s["artifact_under_official"], "abstain", "defused")


def main():
    # dedupe to the LAST line per (benchmark, model) -- robust to appended re-runs / the n=8 PMC smoke
    global rows
    rows = list({(r["benchmark"], r["model"]): r for r in rows}.values())
    (Path("results") / "unified_artifacts.json").write_text(json.dumps(rows, indent=2))
    order = ["mmad", "mmmu", "scienceqa", "pmcvqa", "omnimedvqa", "pope", "mathvista", "mathvision"]
    print(f"{'benchmark':12s} {'venue':14s} {'cohort':8s} {'family':22s} {'mean_noncompl':>13s} {'mean_artifact':>13s}  n")
    for b in order:
        br = [r for r in rows if r["benchmark"] == b]
        if not br:
            print(f"{b:12s} {VENUE.get(b,'?'):14s} -- PENDING (no data yet)")
            continue
        mnc = sum(r["non_compliance"] for r in br) / len(br)
        mar = sum(r["artifact"] for r in br) / len(br)
        print(f"{b:12s} {br[0]['venue']:14s} {br[0]['cohort']:8s} {br[0]['family']:22s} "
              f"{mnc:13.4f} {mar:+13.4f}  {len(br)}")
    # firing-cohort correlation (artifact vs non-compliance), the difficulty-scaling evidence
    fire = [r for r in rows if r["cohort"] == "firing"]
    if len(fire) >= 3:
        xs = [r["non_compliance"] for r in fire]
        ys = [r["artifact"] for r in fire]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs) ** 0.5
        vy = sum((y - my) ** 2 for y in ys) ** 0.5
        r = cov / (vx * vy) if vx and vy else float("nan")
        print(f"\nfiring-cohort points: {len(fire)}   Pearson r(non_compliance, artifact) = {r:.3f}")
    print(f"\nwrote {HERE/'unified_artifacts.json'} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
