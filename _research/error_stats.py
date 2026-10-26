"""Recompute MAE / MSE / RMSE / percentage-error from the saved trajectories.

Fixes a weighting bug in the first pass: E_norm is undefined when a trajectory is
flat (span = S1 - S0 = 0), so MAE_norm must be weighted by the count of DEFINED
norm points, not by the raw point count.
"""
import json
import sys
from pathlib import Path

FR = [0.0, 0.25, 0.50, 0.75, 1.0]
DETERMINISTIC = {"citation_faithfulness", "source_diversity"}


def resp_pairs(vals, increasing):
    """Yield (f, response_raw, response_norm) per level; None where undefined."""
    s0, s1 = vals[0], vals[-1]
    span = s1 - s0
    for f, sf in zip(FR, vals):
        if sf is None or s0 is None:
            yield f, None, None
            continue
        if increasing:
            r_raw = (sf - s0) / (1.0 - s0) if s0 != 1.0 else None
        else:
            r_raw = (s0 - sf) / s0 if s0 else None
        r_norm = (sf - s0) / span if span else None
        yield f, r_raw, r_norm


def stats(errs):
    n = len(errs)
    if not n:
        return None
    mae = sum(abs(e) for e in errs) / n
    mse = sum(e * e for e in errs) / n
    return {"n": n, "mae": mae, "mse": mse, "rmse": mse ** 0.5,
            "max": max(abs(e) for e in errs)}


def fmt(s):
    if s is None:
        return "   --      --      --      --"
    return (f"{s['mae']:>7.3f} {s['mse']:>7.4f} {s['rmse']:>7.3f} "
            f"{s['max']:>7.3f}  n={s['n']}")


def main(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    print(f"grader = {d['grader']}\n")
    print(f"{'metric':<24}{'MAE':>7}{'MSE':>8}{'RMSE':>8}{'max':>8}")

    buckets = {"det_raw": [], "det_norm": [], "llm_raw": [], "llm_norm": []}
    per_metric = {}
    # relative % error, computed over f > 0 only (|resp-f|/f)
    rel = {}

    for m in d["metrics"]:
        raw_e, norm_e, rel_e = [], [], []
        for topic, vals in m["trajectories"].items():
            for f, r_raw, r_norm in resp_pairs(vals, m["increasing"]):
                if r_raw is not None:
                    raw_e.append(r_raw - f)
                    if f > 0:
                        rel_e.append(abs(r_raw - f) / f)
                if r_norm is not None:
                    norm_e.append(r_norm - f)
        key = "det" if m["metric"] in DETERMINISTIC else "llm"
        buckets[f"{key}_raw"] += raw_e
        buckets[f"{key}_norm"] += norm_e
        per_metric[m["metric"]] = (stats(raw_e), stats(norm_e))
        rel[m["metric"]] = sum(rel_e) / len(rel_e) if rel_e else None
        print(f"{m['metric']:<24}" + fmt(stats(raw_e)))
        print(f"{'  (normalised)':<24}" + fmt(stats(norm_e)))

    print(f"\n{'POOLED':<24}{'MAE':>7}{'MSE':>8}{'RMSE':>8}{'max':>8}")
    for label, k in [("deterministic  raw", "det_raw"),
                     ("deterministic  norm", "det_norm"),
                     ("LLM            raw", "llm_raw"),
                     ("LLM            norm", "llm_norm")]:
        print(f"{label:<24}" + fmt(stats(buckets[k])))
    print(f"{'ALL            raw':<24}" + fmt(stats(buckets['det_raw'] + buckets['llm_raw'])))
    print(f"{'ALL            norm':<24}" + fmt(stats(buckets['det_norm'] + buckets['llm_norm'])))

    print("\nrelative error  |resp-f|/f  averaged over f>0 (blows up at small f):")
    for k, v in rel.items():
        print(f"  {k:<24}{v*100:>7.1f}%" if v is not None else f"  {k:<24}    --")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main(sys.argv[1] if len(sys.argv) > 1
         else "_research/data/response_error_gpt-4o-mini.json")
