"""De-circularize the 'scaling law'. The skeptic is right that artifact <= non_compliance BY
CONSTRUCTION (the fallback can only earn credit on the fraction where it fires), so correlating
artifact with non_compliance is partly an identity. We DECOMPOSE it instead:

    artifact = non_compliance * conditional_fab_acc

where conditional_fab_acc = P(fabricated verdict is correct | the fallback fired). This separates:
  (1) non_compliance  -- how OFTEN the fallback fires (a property of task difficulty x format x model;
                         this is the non-circular axis -- it ranges ~2% (POPE) to ~40% (MMAD/OmniMed
                         on pixtral) and that variation is the real finding), from
  (2) conditional_fab_acc -- how BIASED the fabrication is when it fires (a property of the FLAVOR:
                         random-over-k ~ 1/k; directional snap-to-majority ~ P(majority is correct),
                         which is much larger). This is the SEVERITY multiplier and it is NOT circular
                         -- it's a fixed property of the scorer, independent of how often it fires.

So the honest claims become: (a) directional snap/default flavors are ~2-4x more severe PER fabrication
than random (a flavor finding, not a tautology); (b) non_compliance varies ~20x across difficulty and
does not shrink with model strength on hard tasks; (c) the realized artifact is the product. Reported
with benchmark-clustered means (the 5 per-model cells are NOT independent).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path("data")
rows = json.loads((HERE / "unified_artifacts.json").read_text())

FLAVOR = {"snap": "directional-snap", "default-class": "directional-default",
          "parse+random": "random", "snap(+extract buffer)": "buffered", "abstain": "abstain"}


def main():
    by_bench = {}
    for r in rows:
        by_bench.setdefault(r["benchmark"], []).append(r)

    print(f"{'benchmark':12s} {'flavor':20s} {'cohort':8s} {'mean_noncompl':>12s} {'mean_artifact':>12s} {'cond_fab_acc':>12s}")
    flavor_cond = {}
    for b, cells in by_bench.items():
        fam = cells[0]["family"]
        flavor = FLAVOR.get(fam, fam)
        mnc = sum(c["non_compliance"] for c in cells) / len(cells)
        mar = sum(c["artifact"] for c in cells) / len(cells)
        # conditional fabrication accuracy: artifact/non_compliance, per cell, averaged where it fired
        cfs = [c["artifact"] / c["non_compliance"] for c in cells if c["non_compliance"] > 0.01]
        cfa = sum(cfs) / len(cfs) if cfs else float("nan")
        if cells[0]["cohort"] == "firing" and cfs:
            flavor_cond.setdefault(flavor, []).extend(cfs)
        print(f"{b:12s} {flavor:20s} {cells[0]['cohort']:8s} {mnc:12.4f} {mar:+12.4f} {cfa:12.3f}")

    print("\n--- SEVERITY per flavor (conditional fabrication accuracy = how biased the fallback is when it fires) ---")
    for flavor, vals in sorted(flavor_cond.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        m = sum(vals) / len(vals)
        print(f"  {flavor:22s} cond_fab_acc = {m:.3f}   (n={len(vals)} firing cells)   "
              f"{'<-- directional: snaps toward a majority/default class' if 'directional' in flavor else '<-- ~1/k chance' if flavor=='random' else ''}")

    # non-circular scaling: benchmark-level non_compliance range (the difficulty-driven axis)
    fire_b = [(b, sum(c['non_compliance'] for c in cs) / len(cs))
              for b, cs in by_bench.items() if cs[0]["cohort"] == "firing"]
    fire_b = [x for x in fire_b if x[1] > 0.005]  # drop placeholder/zero-data benchmarks (e.g. PMC smoke)
    fire_b.sort(key=lambda x: x[1])
    if fire_b:
        lo, hi = fire_b[0], fire_b[-1]
        print(f"\nnon_compliance spans {lo[1]:.3f} ({lo[0]}) -> {hi[1]:.3f} ({hi[0]})  "
              f"= {hi[1]/max(lo[1],1e-9):.0f}x across difficulty (the non-circular axis)")
    print("\nNOTE: report artifact-vs-non_compliance as a DECOMPOSITION, not a discovered correlation;")
    print("the genuine findings are (a) flavor severity ordering and (b) non_compliance's difficulty range.")


if __name__ == "__main__":
    main()
