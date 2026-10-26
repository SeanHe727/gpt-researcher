"""Rebuild the perturbation fixture set: 10 real reports, full context retained.

The existing three fixtures each carry exactly 8000 characters of context, which
is the generator truncating, not the pipeline's real output — that runs 32-44k.
Since claim scoring is now known to be dominated by how much context it can see,
a truncated fixture bakes the very defect the metric was just fixed for. These
are stored whole.

Ten topics instead of three so a perturbation verdict rests on 50 trajectories
rather than 15.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")

OUT = Path("./evals/quality_eval/"
           "fixtures/perturbation_fixtures.json")

QUERIES = {
    "medical":   "Current treatments and clinical outcomes for Type 2 diabetes",
    "legal":     "Legal implications of AI-generated content and copyright law",
    "history":   "Causes of the fall of the Western Roman Empire",
    "climate":   "Latest scientific findings on climate change and sea level rise",
    "ai_tech":   "Latest developments in large language models in 2024",
    "finance":   "How does the Federal Reserve control inflation through monetary policy",
    "physics":   "Principles of quantum entanglement and quantum computing",
    "chemistry": "Mechanism of nucleophilic substitution reactions in organic chemistry",
    "energy":    "Grid-scale battery storage technologies and their limitations",
    "nutrition": "Evidence on intermittent fasting and metabolic health outcomes",
}


def _keys():
    for n, p in [("OPENAI_API_KEY", "D:/chenh/api_keys/gpt_api_key.txt"),
                 ("TAVILY_API_KEY", "D:/chenh/api_keys/tavily_api_key.txt")]:
        if not os.getenv(n) and Path(p).exists():
            os.environ[n] = Path(p).read_text().strip()


async def build_one(topic, query):
    from gpt_researcher.agent import GPTResearcher
    from evals.quality_eval.pipeline_probe import PipelineProbe

    r = GPTResearcher(query=query, report_type="research_report",
                      report_format="markdown", verbose=False)
    with PipelineProbe() as probe:
        context = await r.conduct_research()
        report = await r.write_report()
    # Source metrics should see the pages that actually loaded, not every URL
    # the retriever proposed.
    sources = probe.trace.scraped_urls or r.get_source_urls()
    return {
        "topic":   topic,
        "query":   query,
        "report":  report,
        "sources": sources,
        "context": str(context),
        "cost":    r.get_costs(),
    }


async def main():
    _keys()
    os.environ["SMART_LLM"] = "openai:gpt-4o"      # match the existing fixtures
    os.environ["FAST_LLM"] = "openai:gpt-4o-mini"
    os.environ["STRATEGIC_LLM"] = "openai:gpt-4o"
    os.environ["SMART_TOKEN_LIMIT"] = "6000"

    out, total = [], 0.0
    for topic, query in QUERIES.items():
        try:
            fx = await build_one(topic, query)
        except Exception as e:
            print(f"  {topic:<10} FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        total += fx.pop("cost", 0.0) or 0.0
        out.append(fx)
        print(f"  {topic:<10} report={len(fx['report']):>6,}  "
              f"context={len(fx['context']):>7,}  sources={len(fx['sources']):>3}  "
              f"(${total:.3f} cumulative)", flush=True)
        OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(out)} fixtures -> {OUT}")
    print(f"file size {OUT.stat().st_size/1000:.0f} KB   research cost ${total:.3f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
