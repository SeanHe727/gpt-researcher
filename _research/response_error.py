"""
Response-fidelity analysis for the perturbation test.

For each metric x fixture we have a trajectory S(f) over f in {0,.25,.5,.75,1.0},
where f is the MANUAL degradation intensity. Two error definitions:

  E_raw(f)  = |(S0 - Sf)/S0            - f|   relative drop vs intensity
                                              (assumes full degradation -> 0)
  E_norm(f) = |(Sf - S0)/(S1 - S0)     - f|   endpoint-normalised response
                                              (floor-agnostic: tests linearity)

For increasing metrics (unsupported_claim_rate) E_raw uses (Sf - S0)/(1 - S0).
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")

from evals.quality_eval import metrics, perturbation as P

FRACTIONS = P.FRACTIONS
FIXTURES = Path("./evals/quality_eval/"
                "fixtures/perturbation_fixtures.json")


def errors(vals, increasing=False):
    """Return (E_raw list, E_norm list) aligned with FRACTIONS."""
    s0, s1 = vals[0], vals[-1]
    e_raw, e_norm = [], []
    for f, sf in zip(FRACTIONS, vals):
        if sf is None or s0 is None:
            e_raw.append(None); e_norm.append(None); continue
        # raw: response measured against the theoretical fully-degraded value
        # (increasing metric -> ceiling 1.0; decreasing metric -> floor 0.0)
        if increasing:
            resp = (sf - s0) / (1.0 - s0) if s0 != 1.0 else None
        else:
            resp = (s0 - sf) / s0 if s0 else None
        e_raw.append(abs(resp - f) if resp is not None else None)
        span = (s1 - s0)
        e_norm.append(abs((sf - s0) / span - f) if span else None)
    return e_raw, e_norm


def show(name, rows, increasing=False):
    """rows: list of (topic, vals)"""
    print(f"\n=== {name} ===")
    print(f"  {'topic':<10}" + "".join(f"{int(f*100):>8}%" for f in FRACTIONS))
    all_raw, all_norm = [], []
    for topic, vals in rows:
        print(f"  {topic:<10}" + "".join(
            f"{v:>9.3f}" if v is not None else f"{'--':>9}" for v in vals))
        er, en = errors(vals, increasing)
        all_raw += [e for e in er if e is not None]
        all_norm += [e for e in en if e is not None]
        print(f"  {'  E_raw':<10}" + "".join(
            f"{e:>9.3f}" if e is not None else f"{'--':>9}" for e in er))
        print(f"  {'  E_norm':<10}" + "".join(
            f"{e:>9.3f}" if e is not None else f"{'--':>9}" for e in en))
    mae_raw = sum(all_raw) / len(all_raw) if all_raw else None
    mae_norm = sum(all_norm) / len(all_norm) if all_norm else None
    print(f"  -> MAE_raw = {mae_raw:.3f}   MAE_norm = {mae_norm:.3f}   (n={len(all_raw)})")
    return {"metric": name, "rows": rows, "mae_raw": mae_raw, "mae_norm": mae_norm,
            "n": len(all_raw), "increasing": increasing}


async def main(with_llm: bool, model: str):
    fixtures = json.loads(FIXTURES.read_text(encoding="utf-8"))
    out = []

    rows = [(fx["topic"], [
        metrics.citation_faithfulness(
            P.degrade_citations(fx["report"], f), fx["sources"], context=fx["context"]
        )["citation_faithfulness"] for f in FRACTIONS]) for fx in fixtures]
    out.append(show("citation_faithfulness", rows))

    rows = [(fx["topic"], [
        metrics.source_diversity(P.degrade_diversity(fx["sources"], f))["diversity_ratio"]
        for f in FRACTIONS]) for fx in fixtures]
    out.append(show("source_diversity", rows))

    if with_llm:
        from langchain_openai import ChatOpenAI
        grader = ChatOpenAI(temperature=0, model_name=model,
                            openai_api_key=os.environ["OPENAI_API_KEY"])
        print(f"\n[grader = {model}]")

        rows = []
        for fx in fixtures:
            vals = []
            for f in FRACTIONS:
                r = await metrics.source_authority(
                    P.degrade_authority(fx["sources"], f), grader_model=grader)
                vals.append(r["avg_authority_score"])
            rows.append((fx["topic"], vals))
        out.append(show("source_authority", rows))

        rows = []
        for fx in fixtures:
            base = await metrics.subtopic_coverage(fx["query"], fx["report"], grader)
            subs = base.get("expected_subtopics", [])
            vals = []
            for f in FRACTIONS:
                r = await metrics.subtopic_coverage(
                    fx["query"], P.degrade_subtopics(fx["report"], f), grader, subtopics=subs)
                vals.append(r.get("subtopic_coverage_rate"))
            rows.append((fx["topic"], vals))
        out.append(show("subtopic_coverage", rows))

        rows = []
        for fx in fixtures:
            base = await metrics.unsupported_claim(fx["report"], fx["context"], grader)
            claims = [c["claim"] for c in base.get("breakdown", [])]
            vals = []
            for f in FRACTIONS:
                r = await metrics.unsupported_claim(
                    fx["report"], fx["context"], grader,
                    claims=P.corrupt_claims(claims, f))
                vals.append(r.get("unsupported_claim_rate"))
            rows.append((fx["topic"], vals))
        out.append(show("unsupported_claim", rows, increasing=True))

    det = [o for o in out if o["metric"] in
           ("citation_faithfulness", "source_diversity")]
    llm = [o for o in out if o not in det]

    def pooled(group, label):
        if not group:
            return
        n = sum(o["n"] for o in group)
        raw = sum(o["mae_raw"] * o["n"] for o in group if o["mae_raw"] is not None) / n
        nrm = sum(o["mae_norm"] * o["n"] for o in group if o["mae_norm"] is not None) / n
        print(f"  {label:<28} points={n:<4} MAE_raw={raw:.3f}  MAE_norm={nrm:.3f}")

    print("\n=== POOLED ===")
    pooled(det, "deterministic metrics")
    pooled(llm, f"LLM metrics ({model})")
    pooled(out, "all metrics")

    dump = {"fractions": FRACTIONS, "grader": model if with_llm else None,
            "metrics": [
                {"metric": o["metric"], "increasing": o["increasing"],
                 "mae_raw": o["mae_raw"], "mae_norm": o["mae_norm"], "n": o["n"],
                 "trajectories": {t: v for t, v in o["rows"]}} for o in out]}
    op = Path(__file__).parent / f"response_error_{model}.json"
    op.write_text(json.dumps(dump, indent=2), encoding="utf-8")
    print(f"\n  saved -> {op}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="also run the 3 LLM metrics")
    ap.add_argument("--model", default="gpt-4-turbo", help="grader model")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.llm and not os.getenv("OPENAI_API_KEY"):
        kp = Path("D:/chenh/api_keys/gpt_api_key.txt")
        if kp.exists():
            os.environ["OPENAI_API_KEY"] = kp.read_text().strip()
    asyncio.run(main(a.llm, a.model))
