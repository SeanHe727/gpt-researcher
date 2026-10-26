"""Three ways to express "metric response vs manual degradation intensity" as a
percentage, so we can pick the one that is both intuitive and correct.

  A. user framing / E_raw : |relative_drop - f|,  relative_drop = (S0-Sf)/S0
                            assumes full degradation -> 0.  Intuitive but wrong
                            for any metric with a structural floor.

  B. floor-corrected      : expected Sf = S0 + f*(FLOOR - S0)   (linear to the
                            metric's TRUE fully-degraded value)
                            error = |S_obs - S_exp|, read directly as points.

  C. full-scale normalised: |(Sf-S0)/(S1-S0) - f|, endpoint-normalised
                            (uses the OBSERVED endpoint, so it is self-
                            referential -- can't detect a broken endpoint).

Plus an exact-model check for source_diversity, where the analytically correct
response curve is derivable and we can separate "metric is wrong" from
"my expectation model was wrong".
"""
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, ".")
from evals.quality_eval import perturbation as P

FR = [0.0, 0.25, 0.50, 0.75, 1.0]
FIX = json.loads(Path("./evals/quality_eval/"
                      "fixtures/perturbation_fixtures.json").read_text(encoding="utf-8"))
NSRC = {f["topic"]: len([u for u in f["sources"] if u]) for f in FIX}

# Theoretical fully-degraded value per metric.
# None -> derived per fixture (diversity: 1/N).
FLOOR = {
    "citation_faithfulness": 0.0,     # every body link removed -> nothing cited
    "source_diversity":      None,    # 1/N: all sources collapsed onto one domain
    "source_authority":      0.205,   # measured: what the grader gives blogs/forums
    "subtopic_coverage":     0.0,     # empty report -> nothing covered
    "unsupported_claim":     1.0,     # every claim corrupted -> all unsupported
}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def analyse(path, label):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    print(f"\n{'='*74}\n{label}  (grader = {d['grader']})\n{'='*74}")
    print(f"{'metric':<24}{'A 用户口径':>12}{'B 地板修正':>12}{'C 满量程':>12}")
    tot = {"A": [], "B": [], "C": []}
    per = {}
    for m in d["metrics"]:
        name, inc = m["metric"], m["increasing"]
        ea, eb, ec = [], [], []
        for topic, vals in m["trajectories"].items():
            s0, s1 = vals[0], vals[-1]
            floor = FLOOR[name]
            if floor is None:
                floor = 1.0 / NSRC[topic]
            for f, sf in zip(FR, vals):
                if sf is None:
                    continue
                # A: relative drop vs f (assumes floor 0 / ceiling 1)
                if inc:
                    ra = (sf - s0) / (1.0 - s0) if s0 != 1.0 else None
                else:
                    ra = (s0 - sf) / s0 if s0 else None
                if ra is not None:
                    ea.append(abs(ra - f))
                # B: linear to the TRUE floor, error in absolute points
                eb.append(abs(sf - (s0 + f * (floor - s0))))
                # C: endpoint-normalised
                span = s1 - s0
                if span:
                    ec.append(abs((sf - s0) / span - f))
        per[name] = (mean(ea), mean(eb), mean(ec))
        tot["A"] += ea; tot["B"] += eb; tot["C"] += ec
        print(f"{name:<24}" + "".join(
            f"{v*100:>11.1f}%" if v is not None else f"{'--':>12}"
            for v in per[name]))
    print("-" * 60)
    print(f"{'POOLED':<24}" + "".join(
        f"{mean(tot[k])*100:>11.1f}%" for k in "ABC"))
    return per, tot


def diversity_exact():
    """Analytic expectation for source_diversity, then error against it.

    degrade_diversity replaces the first n = round(N*f) URLs with paths on ONE
    domain. So the exact expected unique-domain count is:
        unique(remaining originals)  +  (1 if n > 0 else 0)
    This is derived from the perturbation definition alone -- independent of the
    metric implementation -- so it is a genuine ground truth.
    """
    print(f"\n{'='*74}\nsource_diversity — 对照【精确解析期望】而非线性假设\n{'='*74}")
    print(f"  {'topic':<9}{'':>6}" + "".join(f"{int(f*100):>9}%" for f in FR))
    allerr = []
    for fx in FIX:
        srcs = [u for u in fx["sources"] if u]
        N = len(srcs)
        doms = [urlparse(u).netloc.lower().removeprefix("www.") for u in srcs]
        obs, exp = [], []
        for f in FR:
            n = round(N * f)
            uniq = len(set(doms[n:])) + (1 if n > 0 else 0)
            exp.append(uniq / N)
            from evals.quality_eval import metrics
            obs.append(metrics.source_diversity(
                P.degrade_diversity(srcs, f))["diversity_ratio"])
        err = [abs(o - e) for o, e in zip(obs, exp)]
        allerr += err
        print(f"  {fx['topic']:<9}{'obs':>6}" + "".join(f"{v:>10.3f}" for v in obs))
        print(f"  {'':<9}{'exp':>6}" + "".join(f"{v:>10.3f}" for v in exp))
        print(f"  {'':<9}{'err':>6}" + "".join(f"{v*100:>9.1f}%" for v in err))
    print(f"\n  -> MAE vs 精确期望 = {mean(allerr)*100:.2f} 个百分点  (n={len(allerr)})")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    analyse("_research/data/response_error_gpt-4-turbo.json", "gpt-4-turbo")
    analyse("_research/data/response_error_gpt-4o-mini.json", "gpt-4o-mini")
    diversity_exact()
