"""Score the captured default-config runs with the whole metric suite.

Goes through suite.evaluate() -- the standardised entry point -- so every metric
comes back as a MetricResult carrying its group and the standard concept it
aligns with, and the panel doubles as a demonstration of the suite's own API.

These runs used stock pipeline settings (similarity 0.42, 5 results per query,
bs scraper, curation off) and carry their FULL context, so the numbers are not
subject to the fixture truncation that distorted earlier measurements.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")

CAP = Path("_research/data/pipeline_capture.json")
GRADER = "gpt-4-turbo"


def _keys():
    p = Path("D:/chenh/api_keys/gpt_api_key.txt")
    if not os.getenv("OPENAI_API_KEY") and p.exists():
        os.environ["OPENAI_API_KEY"] = p.read_text().strip()


async def main():
    _keys()
    from langchain_openai import ChatOpenAI
    from evals.quality_eval.base import EvalSample
    from evals.quality_eval.suite import evaluate, default_metrics

    grader = ChatOpenAI(temperature=0, model_name=GRADER,
                        openai_api_key=os.environ["OPENAI_API_KEY"])
    caps = json.loads(CAP.read_text(encoding="utf-8"))

    print(f"grader = {GRADER}   pipeline = stock defaults   researcher LLM = gpt-4o-mini\n")

    per_metric = {}
    details = {}
    for cap in caps:
        sample = EvalSample(query=cap["query"], report=cap["report"],
                            sources=cap["visited_urls"], context=cap["context"])
        results = await evaluate(sample, default_metrics(), grader)
        details[cap["topic"]] = results
        print(f"=== {cap['topic']}  ({len(cap['visited_urls'])} sources, "
              f"context {cap['context_len']:,} chars)")
        for r in results:
            per_metric.setdefault(r.name, {"group": r.group, "aligned": r.aligned_with,
                                           "scores": [], "breakdowns": []})
            per_metric[r.name]["scores"].append(r.score)
            per_metric[r.name]["breakdowns"].append(r.breakdown)
            s = "skipped" if r.skipped else (f"{r.score:.3f}" if r.score is not None else "n/a")
            print(f"    {r.name:<24}{s:>9}   {r.reason}")
        print()

    print("=" * 82)
    print(f"{'指标':<24}{'分组':<18}{'对标':<26}{'均值':>8}")
    print("=" * 82)
    for name, d in per_metric.items():
        vals = [v for v in d["scores"] if v is not None]
        avg = sum(vals) / len(vals) if vals else None
        print(f"{name:<24}{d['group']:<18}{d['aligned'][:24]:<26}"
              + (f"{avg:>8.3f}" if avg is not None else f"{'--':>8}"))

    # Secondary fields worth surfacing alongside the headline scores
    print("\n" + "=" * 82)
    print("辅助字段")
    print("=" * 82)
    for name, d in per_metric.items():
        for key in ("domain_entropy", "unique_domains", "total_sources",
                    "unsupported_claim_rate", "inferred_claim_rate", "total_claims",
                    "citation_coverage", "listed_ref_domains"):
            vals = [b.get(key) for b in d["breakdowns"]
                    if isinstance(b, dict) and isinstance(b.get(key), (int, float))]
            if vals:
                print(f"  {name}.{key:<26}{sum(vals)/len(vals):>10.3f}")

    print("\n" + "=" * 82)
    print("来源权威性明细（三次运行合并，按分数）")
    print("=" * 82)
    auth = per_metric.get("source_authority", {}).get("breakdowns", [])
    merged = {}
    for b in auth:
        for e in (b or {}).get("breakdown", []):
            merged[e["domain"]] = e
    for e in sorted(merged.values(), key=lambda x: -x["score"]):
        print(f"  {e['score']:.2f}  {e['domain']:<40}{e['tier'][:34]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
