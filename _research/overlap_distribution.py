"""How much of the retrieval budget does the planner spend on ground already covered?

Four queries, five repeats each. Repeating the SAME query separates two things a
single run cannot: how much the planner varies run to run on one question
(planner instability) versus how much it varies between questions (some topics
branching more naturally than others).

Only conduct_research runs — the report is not needed and is the expensive half.
Also records the cosine similarity of each sub-query to the original question, to
test the suspected mechanism: that the waste comes from the planner restating the
question it was given, which the pipeline then searches a second time.
"""
import asyncio
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, ".")

OUT = Path("_research/data/overlap_distribution.json")
REPEATS = 5
QUERIES = {
    "medical": "Current treatments and clinical outcomes for Type 2 diabetes",
    "legal":   "Legal implications of AI-generated content and copyright law",
    "ai_tech": "Latest developments in large language models in 2024",
    "finance": "How does the Federal Reserve control inflation through monetary policy",
}


def _keys():
    for n, p in [("OPENAI_API_KEY", "D:/chenh/api_keys/gpt_api_key.txt"),
                 ("TAVILY_API_KEY", "D:/chenh/api_keys/tavily_api_key.txt")]:
        if not os.getenv(n) and Path(p).exists():
            os.environ[n] = Path(p).read_text().strip()


async def one_run(query, emb):
    import numpy as np
    from evals.quality_eval.pipeline_probe import PipelineProbe
    from gpt_researcher.agent import GPTResearcher

    r = GPTResearcher(query=query, report_type="research_report", verbose=False)
    with PipelineProbe() as probe:
        await r.conduct_research()
    t = probe.trace
    f = t.funnel()

    sims = []
    if t.subqueries:
        v = np.array(await emb.aembed_documents([query] + list(t.subqueries)))
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        sims = [round(float(x), 3) for x in (v[1:] @ v[0])]

    return {
        "subqueries":     t.subqueries,
        "subquery_sims":  sims,
        "max_sim":        max(sims) if sims else None,
        "urls_proposed":  f["urls_proposed"],
        "urls_duplicate": f["urls_duplicate"],
        "search_overlap": f["search_overlap"],
        "scrape_rate":    f["scrape_rate"],
        "dead_searches":  sum(1 for s in t.searches if s["proposed"] and not s["accepted"]),
        "cost":           r.get_costs(),
    }


async def main():
    _keys()
    for v in ("FAST_LLM", "SMART_LLM", "STRATEGIC_LLM"):
        os.environ[v] = "openai:gpt-4o-mini"
    from langchain_openai import OpenAIEmbeddings
    emb = OpenAIEmbeddings(model="text-embedding-3-small",
                           openai_api_key=os.environ["OPENAI_API_KEY"])

    runs = []
    for topic, query in QUERIES.items():
        for i in range(REPEATS):
            try:
                rec = await one_run(query, emb)
            except Exception as e:
                print(f"  {topic} #{i+1} FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            rec.update(topic=topic, query=query, run=i + 1)
            runs.append(rec)
            print(f"  {topic:<9}#{i+1}  overlap={rec['search_overlap']}  "
                  f"dead={rec['dead_searches']}  maxsim={rec['max_sim']}  "
                  f"scrape={rec['scrape_rate']}", flush=True)
            OUT.write_text(json.dumps(runs, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n" + "=" * 68)
    print(f"{'topic':<9}{'n':>3}{'overlap 均值':>13}{'范围':>14}{'标准差':>9}{'空转检索':>9}")
    alls = []
    for topic in QUERIES:
        vals = [r["search_overlap"] for r in runs
                if r["topic"] == topic and r["search_overlap"] is not None]
        if not vals:
            continue
        alls += vals
        dead = sum(r["dead_searches"] for r in runs if r["topic"] == topic)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{topic:<9}{len(vals):>3}{statistics.mean(vals):>13.3f}"
              f"{f'{min(vals):.2f}-{max(vals):.2f}':>14}{sd:>9.3f}{dead:>9}")
    if alls:
        print("-" * 68)
        print(f"{'全部':<9}{len(alls):>3}{statistics.mean(alls):>13.3f}"
              f"{f'{min(alls):.2f}-{max(alls):.2f}':>14}"
              f"{statistics.stdev(alls):>9.3f}"
              f"{sum(r['dead_searches'] for r in runs):>9}")
        print(f"\n总花费 ${sum(r['cost'] for r in runs):.4f}   saved -> {OUT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
