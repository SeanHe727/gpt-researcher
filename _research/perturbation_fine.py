"""Fine-grained perturbation run: 10 reports x 11 degradation levels.

The 25% ladder was too coarse to say anything about resolution — four steps is
easy to order correctly, so tau near 1.0 only showed the metrics were not
broken. At 10% steps the ladder approaches the metrics' own quantisation limit
(round(N * 0.1) is 1-2 items for every degradation function here), so the
numbers describe how finely each metric can actually discriminate.

Reports two figures per metric, both label-free:
    tau-b          ordering fidelity; ties from dead zones cost points, which
                   the monotonicity predicate in perturbation.py does not catch
    response error magnitude fidelity, |observed - expected| against a linear
                   decay to the metric's TRUE floor (0 is wrong for most of them)

Token usage is accumulated from each response so the run reports its own cost
rather than an estimate.
"""
import asyncio
import json
import math
import os
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, ".")

from evals.quality_eval import metrics, perturbation as P

FR = [round(i / 10, 2) for i in range(11)]          # 0.0 .. 1.0 by 0.10
FIX = Path("./evals/quality_eval/"
           "fixtures/perturbation_fixtures.json")
OUT = Path("_research/data/perturbation_fine.json")

PRICE_IN, PRICE_OUT = 0.20 / 1e6, 1.20 / 1e6        # GPT-5.6 Luna

FLOOR = {"citation_faithfulness": 0.0, "source_diversity": None,
         "source_authority": 0.205, "subtopic_coverage": 0.0,
         "unsupported_claim": 1.0}


class Metered:
    """Wraps a chat model and accumulates token usage across every call."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = self.tin = self.tout = 0

    async def ainvoke(self, *a, **k):
        r = await self.inner.ainvoke(*a, **k)
        u = getattr(r, "usage_metadata", None) or {}
        self.calls += 1
        self.tin += u.get("input_tokens", 0)
        self.tout += u.get("output_tokens", 0)
        return r

    @property
    def cost(self):
        return self.tin * PRICE_IN + self.tout * PRICE_OUT

    def line(self):
        return (f"{self.calls} calls  in={self.tin:,}  out={self.tout:,}  "
                f"${self.cost:.3f}")


def taub(x, y):
    """Kendall tau-b. Ties in y (a metric that cannot separate two damage
    levels) enlarge the denominator, so dead zones cost points."""
    C = D = Ty = 0
    for (x1, y1), (x2, y2) in combinations(zip(x, y), 2):
        if y1 == y2:
            Ty += 1
        elif (x1 - x2) * (y1 - y2) > 0:
            C += 1
        else:
            D += 1
    den = math.sqrt((C + D) * (C + D + Ty))
    return ((C - D) / den if den else 0.0), C, D, Ty


def resp_err(name, vals, n_sources):
    s0 = vals[0]
    floor = FLOOR[name]
    if floor is None:
        floor = 1.0 / max(n_sources, 1)
    return [abs(v - (s0 + f * (floor - s0)))
            for f, v in zip(FR, vals) if v is not None]


def report_row(topic, vals, tb, dead):
    cells = " ".join(f"{v:.2f}" if v is not None else " -- " for v in vals)
    return f"  {topic:<10}{cells}   tau={tb:+.3f}  dead={dead}"


def dead_zone(vals):
    """Largest fraction at which the metric is still at its baseline."""
    out = 0.0
    for f, v in zip(FR[1:], vals[1:]):
        if v is not None and abs(v - vals[0]) < 1e-9:
            out = f
        else:
            break
    return f"{int(out*100)}%"


async def main(model):
    from langchain_openai import ChatOpenAI
    raw = ChatOpenAI(temperature=0, model_name=model,
                     openai_api_key=os.environ["OPENAI_API_KEY"])
    g = Metered(raw)
    fixtures = json.loads(FIX.read_text(encoding="utf-8"))
    print(f"grader = {model}   {len(fixtures)} fixtures x {len(FR)} levels\n", flush=True)

    rows = []

    def emit(name, fx, vals, increasing=False):
        tb, C, D, Ty = taub(FR, vals if increasing else [-v for v in vals])
        err = resp_err(name, vals, len(fx["sources"]))
        rows.append({"metric": name, "topic": fx["topic"], "vals": vals,
                     "increasing": increasing, "tau": tb, "C": C, "D": D,
                     "ties": Ty, "err": sum(err) / len(err)})
        print(report_row(fx["topic"], vals, tb, dead_zone(vals)), flush=True)

    print("=== citation_faithfulness ===", flush=True)
    for fx in fixtures:
        emit("citation_faithfulness", fx, [
            metrics.citation_faithfulness(
                P.degrade_citations(fx["report"], f), fx["sources"],
                context=fx["context"])["citation_faithfulness"] for f in FR])

    print("\n=== source_diversity ===", flush=True)
    for fx in fixtures:
        emit("source_diversity", fx, [
            metrics.source_diversity(
                P.degrade_diversity(fx["sources"], f))["diversity_ratio"]
            for f in FR])

    print(f"\n=== source_authority ===   [{g.line()}]", flush=True)
    for fx in fixtures:
        vals = []
        for f in FR:
            r = await metrics.source_authority(
                P.degrade_authority(fx["sources"], f), grader_model=g)
            vals.append(r["avg_authority_score"])
        emit("source_authority", fx, vals)

    print(f"\n=== subtopic_coverage ===   [{g.line()}]", flush=True)
    for fx in fixtures:
        base = await metrics.subtopic_coverage(fx["query"], fx["report"], g)
        subs = base.get("expected_subtopics", [])
        vals = []
        for f in FR:
            r = await metrics.subtopic_coverage(
                fx["query"], P.degrade_subtopics(fx["report"], f), g, subtopics=subs)
            vals.append(r.get("subtopic_coverage_rate"))
        emit("subtopic_coverage", fx, vals)

    print(f"\n=== unsupported_claim (expect RISE) ===   [{g.line()}]", flush=True)
    for fx in fixtures:
        base = await metrics.unsupported_claim(fx["report"], fx["context"], g)
        claims = [c["claim"] for c in base.get("breakdown", [])]
        vals = []
        for f in FR:
            r = await metrics.unsupported_claim(
                fx["report"], fx["context"], g, claims=P.corrupt_claims(claims, f))
            vals.append(r.get("unsupported_claim_rate"))
        emit("unsupported_claim", fx, vals, increasing=True)
        OUT.write_text(json.dumps({"rows": rows, "fractions": FR, "grader": model,
                                   "usage": {"calls": g.calls, "in": g.tin,
                                             "out": g.tout, "cost": g.cost}},
                                  indent=1), encoding="utf-8")

    # ---- summary ----
    print("\n" + "=" * 72)
    print(f"{'指标':<26}{'tau-b':>9}{'响应误差':>11}{'平均死区':>10}{'n':>5}")
    by = {}
    for r in rows:
        by.setdefault(r["metric"], []).append(r)
    for m, rs in by.items():
        dz = sum(float(dead_zone(r["vals"]).rstrip("%")) for r in rs) / len(rs)
        print(f"{m:<26}{sum(r['tau'] for r in rs)/len(rs):>9.3f}"
              f"{sum(r['err'] for r in rs)/len(rs)*100:>10.1f}%{dz:>9.0f}%{len(rs):>5}")
    print("-" * 72)
    print(f"{'合计':<26}{sum(r['tau'] for r in rows)/len(rows):>9.3f}"
          f"{sum(r['err'] for r in rows)/len(rows)*100:>10.1f}%{'':>10}{len(rows):>5}")
    print(f"\n实测用量: {g.line()}")
    OUT.write_text(json.dumps({"rows": rows, "fractions": FR, "grader": model,
                               "usage": {"calls": g.calls, "in": g.tin,
                                         "out": g.tout, "cost": g.cost}},
                              indent=1), encoding="utf-8")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = Path("D:/chenh/api_keys/gpt_api_key.txt").read_text().strip()
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "gpt-5.6-luna"))
