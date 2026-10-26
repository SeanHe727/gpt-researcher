"""Phase 1 — instrument one research run per query and capture every stage.

Hooks three boundaries with the same monkey-patch technique benchmark.py already
uses (test process only, upstream untouched):

    ResearchConductor.plan_research            -> the sub-queries
    BrowserManager.browse_urls                 -> URLs in, scraped pages out
    ContextManager.get_similar_content_by_query-> pages in, compressed context out

Saving the raw scraped pages is the point: every later ablation (similarity
threshold, fast-path branch) re-runs compression on THIS captured data instead
of re-scraping, so the only thing varying is the knob under test.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")

OUT = Path("_research/data/pipeline_capture.json")
QUERIES = [
    ("medical", "Current treatments and clinical outcomes for Type 2 diabetes"),
    ("ai_tech", "Latest developments in large language models in 2024"),
    ("legal",   "Legal implications of AI-generated content and copyright law"),
]


def _load_keys():
    for name, path in [("OPENAI_API_KEY", "D:/chenh/api_keys/gpt_api_key.txt"),
                       ("TAVILY_API_KEY", "D:/chenh/api_keys/tavily_api_key.txt")]:
        if not os.getenv(name):
            p = Path(path)
            if p.exists():
                os.environ[name] = p.read_text().strip()


async def capture_one(topic: str, query: str) -> dict:
    from gpt_researcher.agent import GPTResearcher
    from gpt_researcher.skills.researcher import ResearchConductor
    from gpt_researcher.skills.browser import BrowserManager
    from gpt_researcher.skills.context_manager import ContextManager

    rec = {"topic": topic, "query": query, "subqueries": [],
           "scrape_calls": [], "compress_calls": []}

    orig_plan     = ResearchConductor.plan_research
    orig_browse   = BrowserManager.browse_urls
    orig_compress = ContextManager.get_similar_content_by_query

    async def plan(self, *a, **k):
        out = await orig_plan(self, *a, **k)
        rec["subqueries"] = list(out) if out else []
        return out

    async def browse(self, urls, *a, **k):
        pages = await orig_browse(self, urls, *a, **k)
        rec["scrape_calls"].append({
            "urls_in":   len(urls or []),
            "pages_out": len(pages or []),
            "chars_out": sum(len(str(p.get("raw_content", "") or "")) for p in (pages or [])),
            "urls":      list(urls or []),
        })
        return pages

    async def compress(self, q, pages, *a, **k):
        total_chars = sum(len(str(p.get("raw_content", "") or "")) for p in (pages or []))
        ctx = await orig_compress(self, q, pages, *a, **k)
        # Reproduce the branch predicate from ContextCompressor.async_get_context:
        # fast path when the corpus is small AND fits inside max_results (10 here).
        threshold = int(os.environ.get("COMPRESSION_THRESHOLD", "8000"))
        rec["compress_calls"].append({
            "subquery":     q,
            "pages_in":     len(pages or []),
            "chars_in":     total_chars,
            "chars_out":    len(str(ctx or "")),
            "threshold":    threshold,
            "fast_path":    total_chars < threshold and len(pages or []) <= 10,
            # keep the raw pages so later ablations reuse them instead of re-scraping
            "pages":        [{"url": p.get("url") or p.get("source", ""),
                              "title": p.get("title", ""),
                              "raw_content": str(p.get("raw_content", "") or "")}
                             for p in (pages or [])],
        })
        return ctx

    ResearchConductor.plan_research = plan
    BrowserManager.browse_urls = browse
    ContextManager.get_similar_content_by_query = compress
    try:
        r = GPTResearcher(query=query, report_type="research_report", verbose=False)
        context = await r.conduct_research()
        report = await r.write_report()
        rec["context"]      = str(context)
        rec["context_len"]  = len(str(context))
        rec["report"]       = report
        rec["visited_urls"] = sorted(r.visited_urls)
        rec["cost"]         = r.get_costs()
    finally:
        ResearchConductor.plan_research = orig_plan
        BrowserManager.browse_urls = orig_browse
        ContextManager.get_similar_content_by_query = orig_compress
    return rec


async def main():
    _load_keys()
    for var in ("FAST_LLM", "SMART_LLM", "STRATEGIC_LLM"):
        os.environ[var] = "openai:gpt-4o-mini"

    out = []
    for topic, query in QUERIES:
        print(f"\n=== {topic}: {query}")
        try:
            rec = await capture_one(topic, query)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
        out.append(rec)
        sc = rec["scrape_calls"]
        cc = rec["compress_calls"]
        print(f"  子查询 {len(rec['subqueries'])} 个")
        print(f"  抓取 {sum(c['urls_in'] for c in sc)} URL -> {sum(c['pages_out'] for c in sc)} 页, "
              f"{sum(c['chars_out'] for c in sc):,} 字符")
        print(f"  压缩调用 {len(cc)} 次, 快路径 {sum(c['fast_path'] for c in cc)} 次")
        print(f"  context {rec['context_len']:,} 字符   报告 {len(rec['report']):,} 字符   ${rec['cost']:.4f}")
        OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    print(f"\nsaved {len(out)} captures -> {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
