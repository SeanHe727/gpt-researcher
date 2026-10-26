"""Run the perturbation test on the rebuilt fixtures with a chosen grader.

Mirrors evals/quality_eval/perturbation.py exactly — same degradation functions,
same fraction ladder, same monotonicity predicate — but selects a subset of
fixtures and additionally reports response error, so one (expensive) pass yields
both numbers: does each metric move the right way, and does it move by the right
amount.

Both metric fixes are live here (references split out of coverage judging,
CONTEXT_CHARS at 50000), and the fixtures now carry their full 31-46k context
rather than a generator-truncated 8000, so nothing is evidence-starved.
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")

from evals.quality_eval import metrics, perturbation as P
from evals.quality_eval.perturbation import _header, _row, _monotone

FR = P.FRACTIONS
FIX = Path("./evals/quality_eval/"
           "fixtures/perturbation_fixtures.json")

# Theoretical fully-degraded value per metric, for the floor-corrected error.
FLOOR = {"citation_faithfulness": 0.0, "source_diversity": None,
         "source_authority": 0.205, "subtopic_coverage": 0.0,
         "unsupported_claim": 1.0}


def errors(name, topic, vals, n_sources, increasing):
    """|observed - expected| in points, expecting linear decay to the true floor."""
    s0 = vals[0]
    floor = FLOOR[name]
    if floor is None:
        floor = 1.0 / max(n_sources, 1)
    return [abs(v - (s0 + f * (floor - s0)))
            for f, v in zip(FR, vals) if v is not None]


async def main(model, topics):
    from langchain_openai import ChatOpenAI
    grader = ChatOpenAI(temperature=0, model_name=model,
                        openai_api_key=os.environ["OPENAI_API_KEY"])
    fixtures = [f for f in json.loads(FIX.read_text(encoding="utf-8"))
                if f["topic"] in topics]
    print(f"grader = {model}   fixtures = {[f['topic'] for f in fixtures]}\n")

    results, err_pool = [], {}

    async def section(title, name, increasing=False):
        print(f"=== {title} ===")
        print(_header())
        return name, increasing

    # --- citation_faithfulness (deterministic) ---
    name, inc = await section("citation_faithfulness  (remove inline body citations)",
                              "citation_faithfulness")
    for fx in fixtures:
        vals = [metrics.citation_faithfulness(
            P.degrade_citations(fx["report"], f), fx["sources"],
            context=fx["context"])["citation_faithfulness"] for f in FR]
        print(_row(fx["topic"], vals)); results.append((name, fx["topic"], vals, inc))
        err_pool.setdefault(name, []).extend(errors(name, fx["topic"], vals, len(fx["sources"]), inc))

    # --- source_diversity (deterministic) ---
    print()
    name, inc = await section("source_diversity  (collapse onto one domain)",
                              "source_diversity")
    for fx in fixtures:
        vals = [metrics.source_diversity(
            P.degrade_diversity(fx["sources"], f))["diversity_ratio"] for f in FR]
        print(_row(fx["topic"], vals)); results.append((name, fx["topic"], vals, inc))
        err_pool.setdefault(name, []).extend(errors(name, fx["topic"], vals, len(fx["sources"]), inc))

    # --- source_authority ---
    print()
    name, inc = await section("source_authority  (swap to low-authority, LLM-scored)",
                              "source_authority")
    for fx in fixtures:
        vals = []
        for f in FR:
            r = await metrics.source_authority(P.degrade_authority(fx["sources"], f),
                                               grader_model=grader)
            vals.append(r["avg_authority_score"])
        print(_row(fx["topic"], vals)); results.append((name, fx["topic"], vals, inc))
        err_pool.setdefault(name, []).extend(errors(name, fx["topic"], vals, len(fx["sources"]), inc))

    # --- subtopic_coverage (fixed subtopic set) ---
    print()
    name, inc = await section("subtopic_coverage  (remove report sections, fixed subtopics)",
                              "subtopic_coverage")
    for fx in fixtures:
        base = await metrics.subtopic_coverage(fx["query"], fx["report"], grader)
        subs = base.get("expected_subtopics", [])
        vals = []
        for f in FR:
            r = await metrics.subtopic_coverage(
                fx["query"], P.degrade_subtopics(fx["report"], f), grader, subtopics=subs)
            vals.append(r.get("subtopic_coverage_rate"))
        print(_row(fx["topic"], vals)); results.append((name, fx["topic"], vals, inc))
        err_pool.setdefault(name, []).extend(errors(name, fx["topic"], vals, len(fx["sources"]), inc))

    # --- unsupported_claim (fixed claim set, expect rise) ---
    print()
    name, inc = await section("unsupported_claim  (corrupt fixed claim set, expect RISE)",
                              "unsupported_claim", increasing=True)
    for fx in fixtures:
        base = await metrics.unsupported_claim(fx["report"], fx["context"], grader)
        claims = [c["claim"] for c in base.get("breakdown", [])]
        vals = []
        for f in FR:
            r = await metrics.unsupported_claim(
                fx["report"], fx["context"], grader, claims=P.corrupt_claims(claims, f))
            vals.append(r.get("unsupported_claim_rate"))
        print(_row(fx["topic"], vals, increasing=True))
        results.append((name, fx["topic"], vals, inc))
        err_pool.setdefault(name, []).extend(errors(name, fx["topic"], vals, len(fx["sources"]), inc))

    # --- verdicts ---
    passed = sum(_monotone(v, inc) for _, _, v, inc in results)
    total = len(results)
    print(f"\n{'='*66}")
    print(f"单调性  {passed}/{total} PASS")
    fails = [(n, t) for n, t, v, i in results if not _monotone(v, i)]
    for n, t in fails:
        print(f"   FAIL  {n} / {t}")

    print(f"\n{'指标':<26}{'响应误差(百分点)':>18}{'n':>5}")
    allerr = []
    for n, e in err_pool.items():
        allerr += e
        print(f"{n:<26}{sum(e)/len(e)*100:>17.1f}%{len(e):>5}")
    print(f"{'合计':<26}{sum(allerr)/len(allerr)*100:>17.1f}%{len(allerr):>5}")

    Path(f"_research/data/perturbation_{model}.json").write_text(
        json.dumps({"grader": model, "fractions": FR,
                    "trajectories": [{"metric": n, "topic": t, "vals": v, "increasing": i}
                                     for n, t, v, i in results],
                    "monotone_pass": passed, "monotone_total": total,
                    "mean_error_pp": sum(allerr)/len(allerr)*100}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--topics", default="medical,legal,ai_tech")
    a = ap.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = Path("D:/chenh/api_keys/gpt_api_key.txt").read_text().strip()
    asyncio.run(main(a.model, set(a.topics.split(","))))
