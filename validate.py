"""Self-validation: recompute the paper's load-bearing numbers from the released data and assert they
match the reported values. Exits 0 and prints "VALIDATION PASSED" iff every check passes.

No GPU, no dataset, no network. Run:  python validate.py
"""
from __future__ import annotations
import collections
import json
import sys
from pathlib import Path

DATA = Path("data")
CANON = {"N": "NONE", "M": "MULTIPLE"}
checks = []


def check(name, got, expect, tol=0.005):
    ok = got is not None and abs(got - expect) <= tol
    checks.append((name, got, expect, ok))


def load_labels(name):
    out = {}
    p = DATA / name
    if p.exists():
        for l in open(p):
            if l.strip():
                d = json.loads(l)
                out[d["uid"]] = CANON.get(d["label"], d["label"])
    return out


def kappa(a, b, keys):
    n = len(keys)
    if not n:
        return None
    po = sum(a[k] == b[k] for k in keys) / n
    labs = set(a[k] for k in keys) | set(b[k] for k in keys)
    ca, cb = collections.Counter(a[k] for k in keys), collections.Counter(b[k] for k in keys)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labs)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def mmad_artifact(model):
    recs = [json.loads(l) for l in open(DATA / f"mmad_{model}.jsonl") if l.strip()]
    n = len(recs)
    snap = sum(r["hard_pred"] == r["gold"] for r in recs) / n
    abst = sum(r["strict_pred"] == r["gold"] for r in recs) / n
    return snap - abst


def pmcvqa_artifact(model):
    """TOTAL effect of PMC-VQA's unconditional matcher (fallback + override). Fig. 3."""
    recs = [json.loads(l) for l in open(DATA / f"pmcvqa_{model}.jsonl") if l.strip()]
    n = len(recs)
    snap = sum(r["snap_pred"] == r["gold"] for r in recs) / n
    abst = sum(r["clean_pred"] == r["gold"] for r in recs) / n
    return snap - abst


def pmcvqa_split(model):
    """Split that total into the parse-failure FALLBACK term and the OVERRIDE term.

    PMC-VQA's matcher runs on every response, so unlike MMAD its aggregate swing is not
    attributable to the fallback alone. The unified table reports the fallback term; the
    override is reported separately in the paper.
    """
    recs = [json.loads(l) for l in open(DATA / f"pmcvqa_{model}.jsonl") if l.strip()]
    n = len(recs)
    fired = [r for r in recs if r["abstain"] == 1]
    rest = [r for r in recs if r["abstain"] != 1]
    hit = lambda rs, k: sum(r[k] == r["gold"] for r in rs)
    return ((hit(fired, "snap_pred") - hit(fired, "clean_pred")) / n,
            (hit(rest, "snap_pred") - hit(rest, "clean_pred")) / n)


def reasoning_parser_scope():
    """Sec. 5.3(v): on a fully COMPLIANT reasoning model the parse-failure fallback never fires,
    yet MMAD's two released tie-break conventions disagree by ~9.5 points, because the parser
    also reads the chain of thought. Returns the quantities the paper reports."""
    import re
    pat = re.compile(r"\b([A-E])\b")
    recs = [json.loads(l) for l in
            open(DATA / "mmad_qwen3-vl-8b-thinking_budget1024.jsonl") if l.strip()]
    n = len(recs)
    first = last = 0
    closed = agree = 0
    for r in recs:
        ms = pat.findall(r["resp"] or "")
        f, l = (ms[0], ms[-1]) if ms else (r["hard_pred"], r["hard_pred"])
        first += f == r["gold"]
        last += l == r["gold"]
        if r.get("think_closed"):
            closed += 1
            agree += (l == r["post_think_pred"])
    return {
        "n": n,
        "non_compliance": sum(r["strict_pred"] is None for r in recs) / n,
        "artifact": (sum(r["hard_pred"] == r["gold"] for r in recs)
                     - sum(r["strict_pred"] == r["gold"] for r in recs)) / n,
        "first": first / n, "last": last / n, "gap": (last - first) / n,
        "last_equals_stated": agree / closed,
    }


def mmad_tiebreak_invariance(model):
    """MMAD's 13 released harnesses index their shared parser inconsistently (11 take the LAST
    regex match, 2 take the FIRST). The artifact (snap - abstain) is invariant BY CONSTRUCTION --
    both rules share the regex parse and separate only on responses with no letter at all -- so the
    delta-artifact check below is a structural guard, not empirical evidence. The empirical content
    is the ACCURACY delta: the tie-break moves the accuracy reading itself by +0.0018 for Pixtral
    (0.6557 first-match vs 0.6539 last-match; the paper's "within 0.2 points") and 0 for every
    other model, asserted here. Returns (delta_artifact, n_items_with_multiple_matches)."""
    import re
    pat = re.compile(r"\b([A-E])\b")
    recs = [json.loads(l) for l in open(DATA / f"mmad_{model}.jsonl") if l.strip()]
    n = len(recs)
    af = al = accf = accl = multi = 0
    for r in recs:
        ms = pat.findall(r.get("resp", "") or "")
        if len(ms) > 1:
            multi += 1
        # snap: regex hit, else the fuzzy fallback (which the tie-break cannot affect)
        sf, sl = (ms[0], ms[-1]) if ms else (r["hard_pred"], r["hard_pred"])
        # abstain: regex hit, else no answer
        bf, bl = (ms[0], ms[-1]) if ms else (None, None)
        af += (sf == r["gold"]) - (bf == r["gold"])
        al += (sl == r["gold"]) - (bl == r["gold"])
        accf += sf == r["gold"]
        accl += sl == r["gold"]
    expected_acc_delta = {"pixtral-12b": 0.0018, "llama-3.2-11b": 0.0}
    got = (accf - accl) / n
    exp = expected_acc_delta[model]
    assert abs(got - exp) < 5e-4, (
        f"{model}: first-vs-last tie-break accuracy delta {got:+.4f}, expected {exp:+.4f}")
    return (al - af) / n, multi


def mmad_compliant_disagreements(model):
    """On MMAD the snap and the abstaining rule agree on EVERY compliant item -- which is what
    licenses attributing the entire swing to the parse-failure fallback. Must be 0."""
    recs = [json.loads(l) for l in open(DATA / f"mmad_{model}.jsonl") if l.strip()]
    return sum(r["hard_pred"] != r["strict_pred"] for r in recs if r["strict_pred"] is not None)


def main():
    # 1. headline artifact magnitudes (Fig. 2 / Fig. 3), recomputed from per-item logs
    check("MMAD Pixtral snap-abstain = +0.212", mmad_artifact("pixtral-12b"), 0.212)
    check("MMAD Llama  snap-abstain = +0.220", mmad_artifact("llama-3.2-11b"), 0.220)
    check("PMC-VQA Pixtral total swing = +0.075", pmcvqa_artifact("pixtral-12b"), 0.075)
    check("PMC-VQA Llama  total swing = -0.190", pmcvqa_artifact("llama-3.2-11b"), -0.190)

    # 1b. attribution: MMAD's swing is the fallback alone (0 disagreements on compliant items);
    #     PMC-VQA's is NOT -- its unconditional matcher also overrides cleanly-parsed answers.
    check("MMAD Pixtral: snap==abstain on every compliant item",
          float(mmad_compliant_disagreements("pixtral-12b")), 0.0)
    check("MMAD Llama:   snap==abstain on every compliant item",
          float(mmad_compliant_disagreements("llama-3.2-11b")), 0.0)
    # 1c. MMAD ships two tie-break conventions; neither changes anything we report.
    for _m in ("pixtral-12b", "llama-3.2-11b"):
        _d, _multi = mmad_tiebreak_invariance(_m)
        check(f"MMAD {_m}: first-vs-last tie-break changes artifact by 0", _d, 0.0)

    # 1d. the reasoning-model result (Sec. 5.3(v)): compliant model, fallback silent,
    #     but the parser's tie-break still decides ~9.5 accuracy points.
    rs = reasoning_parser_scope()
    check("Reasoning model: n = 1670", float(rs["n"]), 1670.0, tol=0.5)
    check("Reasoning model: non-compliance = 0", rs["non_compliance"], 0.0)
    check("Reasoning model: parse-failure artifact = 0", rs["artifact"], 0.0)
    check("Reasoning model: first-match acc = 0.689", rs["first"], 0.6892, tol=0.001)
    check("Reasoning model: last-match acc = 0.784", rs["last"], 0.7844, tol=0.001)
    check("Reasoning model: tie-break gap = 9.5 pts", rs["gap"], 0.0952, tol=0.001)
    check("Reasoning model: last-match == stated answer on 100%",
          rs["last_equals_stated"], 1.0)

    # 1e. Sec 5.3(iv)/(v) auxiliaries: the ScienceQA regex-variant headline and the
    #     category structure behind the clustered CI (full recomputation incl. the seeded
    #     bootstrap lives in src/scienceqa_regex_variants.py and src/reasoning_clustered_ci.py).
    import re as _re
    _wide = _re.compile(r"[Tt]he answer is:?\s*\(?([A-Z])\)?")
    _sq = [json.loads(l) for l in open(DATA / "scienceqa_llama-3.2-11b.jsonl") if l.strip()]
    check("ScienceQA Llama official = 0.448",
          sum(r["sqa_pred"] == r["gold"] for r in _sq) / len(_sq), 0.4483, tol=0.001)
    _wid = 0
    for r in _sq:
        m = _wide.search(r.get("resp", "") or "")
        _wid += (m is not None and (ord(m.group(1)) - 65) == r["gold"])
    check("ScienceQA Llama widened-regex (abstain) = 0.743", _wid / len(_sq), 0.7433, tol=0.001)
    _cats = json.load(open(DATA / "mmad_mvtec_categories.json"))
    _th = [json.loads(l) for l in open(DATA / "mmad_qwen3-vl-8b-thinking_budget1024.jsonl") if l.strip()]
    _pat = _re.compile(r"\b([A-E])\b")
    _per = {}
    for c, r in zip(_cats, _th):
        ms = _pat.findall(r["resp"] or "")
        f, l = (ms[0], ms[-1]) if ms else (r["hard_pred"], r["hard_pred"])
        _per.setdefault(c, []).append(((f == r["gold"]), (l == r["gold"])))
    check("Reasoning: 15 MVTec categories in the released map", float(len(_per)), 15.0, tol=0.5)
    _gaps = [(sum(b for _, b in v) - sum(a for a, _ in v)) / len(v) for v in _per.values()]
    check("Reasoning: max per-category tie-break gap = 0.237", max(_gaps), 0.2368, tol=0.001)

    fb_p, ov_p = pmcvqa_split("pixtral-12b")
    fb_l, ov_l = pmcvqa_split("llama-3.2-11b")
    check("PMC-VQA Pixtral fallback term = +0.097", fb_p, 0.0967, tol=0.001)
    check("PMC-VQA Llama   fallback term = +0.020", fb_l, 0.0200, tol=0.001)
    check("PMC-VQA Llama   override term = -0.210", ov_l, -0.2100, tol=0.001)

    # 2. human-anchored adjudication agreement (Cohen's kappa)
    h, j = load_labels("human_labels.jsonl"), load_labels("judge_labels.jsonl")
    check("MMAD judge-vs-human kappa = 0.956", kappa(h, j, [k for k in h if k in j]), 0.956)
    a2 = load_labels("human_labels_annotator2.jsonl")
    check("MMAD inter-human kappa = 0.910", kappa(h, a2, [k for k in a2 if k in h]), 0.910)
    hp, jp = load_labels("human_labels_pmcvqa.jsonl"), load_labels("judge_labels_pmcvqa.jsonl")
    check("PMC-VQA judge-vs-human kappa = 0.839", kappa(hp, jp, [k for k in hp if k in jp]), 0.839)

    # 3. the over/under-credit decomposition (Table: bidirectional mechanism)
    man = {json.loads(l)["uid"]: json.loads(l) for l in open(DATA / "adjudication_manifest_pmcvqa.jsonl")}
    ref = {**jp, **hp}
    llama = [m for u, m in man.items() if m["model"] == "llama-3.2-11b" and m["uid"] in ref]
    under = sum(m["_snap_pred"] != m["_gold"] and ref[m["uid"]] == m["_gold"] for m in llama) / len(llama)
    check("PMC-VQA Llama under-credit = 0.31 (15/48)", under, 0.3125, tol=0.01)

    # 3b. the likelihood reference is itself confounded (OmniMedVQA, Llama): the option-letter logit
    #     collapses onto 'A', giving accuracy below the snap -- the third leg of "no rule is reliable".
    lr = [json.loads(l) for l in open(DATA / "logit_omnimedvqa_llama-3.2-11b.jsonl") if l.strip()]
    nlr = len(lr)
    check("OmniMed Llama likelihood acc = 0.403 (< snap 0.565)", sum(r["ok"] for r in lr) / nlr, 0.403)
    check("OmniMed Llama logit predicts 'A' = 0.765 (459/600)",
          sum(r["pred_letter"] == "A" for r in lr) / nlr, 0.765, tol=0.01)

    # 4. full unified table regenerated == released reference (40 cells)
    rel = {(r["benchmark"], r["model"]): round(r["artifact"], 3)
           for r in json.load(open(DATA / "unified_artifacts.json"))}
    regen = Path("results/unified_artifacts.json")
    if regen.exists():
        new = {(r["benchmark"], r["model"]): round(r["artifact"], 3) for r in json.load(open(regen))}
        diff = sum(1 for k in rel if rel.get(k) != new.get(k))
        check("Unified table: 40/40 cells reproduce", float(len(rel) - diff), float(len(rel)), tol=0.5)

    # 5. budget robustness (Tab. budget): at 4x the token budget, non-compliance + artifact persist
    for model, exp_nc, exp_gap in [("pixtral-12b", 0.267, 0.183), ("llama-3.2-11b", 0.332, 0.171)]:
        p = DATA / f"mmad_{model}_budget64.jsonl"
        if p.exists():
            rr = [json.loads(l) for l in open(p) if l.strip()]
            m = len(rr)
            nc = sum(x["strict_pred"] is None for x in rr) / m
            gap = (sum(x["hard_pred"] == x["gold"] for x in rr) - sum(x["strict_pred"] == x["gold"] for x in rr)) / m
            check(f"{model} @budget64 non-compliance = {exp_nc}", nc, exp_nc, tol=0.01)
            check(f"{model} @budget64 snap-abstain = {exp_gap}", gap, exp_gap, tol=0.01)

    print(f"\n{'CHECK':54s} {'got':>9s} {'expect':>9s}  result")
    print("-" * 86)
    n_ok = 0
    for name, got, expect, ok in checks:
        g = f"{got:9.4f}" if got is not None else "   MISSING"
        print(f"{name:54s} {g} {expect:9.4f}  {'PASS' if ok else 'FAIL <<<'}")
        n_ok += ok
    print("-" * 86)
    allok = n_ok == len(checks)
    print(f"\n{'VALIDATION PASSED' if allok else 'VALIDATION FAILED'}: {n_ok}/{len(checks)} checks reproduced.")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
