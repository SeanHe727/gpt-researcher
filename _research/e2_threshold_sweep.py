"""E2 — does the compression layer discard evidence, and is the loss tunable?

The funnel shows compression keeping only 15% of scraped characters. Volume loss
is not automatically harm: dropping irrelevant text is the whole point. The test
is whether the DISCARDED text contained evidence the report actually needed.

Design: the report is held fixed (captured), and so is the claim set extracted
from it. Only the context varies -- rebuilt from the SAME captured pages at
different similarity thresholds. So any change in the unsupported rate is
attributable to the compression layer alone.

The "full" condition concatenates every scraped page with no filtering at all:
the ceiling of what this evidence could ever support.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")

CAP = Path("_research/data/pipeline_capture.json")
THRESHOLDS = [0.30, 0.42, 0.55]      # 0.42 is the shipped default
GRADER = "gpt-4o-mini"               # within-condition comparison, so cheap is fine
SCORE_CHARS = 200_000                # effectively no truncation on the eval side


def _keys():
    for n, p in [("OPENAI_API_KEY", "D:/chenh/api_keys/gpt_api_key.txt"),
                 ("TAVILY_API_KEY", "D:/chenh/api_keys/tavily_api_key.txt")]:
        if not os.getenv(n) and Path(p).exists():
            os.environ[n] = Path(p).read_text().strip()


async def build_context(cap, threshold, embeddings, prompt_family):
    """Re-run compression over the captured pages at a given threshold."""
    from gpt_researcher.context.compression import ContextCompressor
    parts = []
    for cc in cap["compress_calls"]:
        pages = [{"raw_content": p["raw_content"], "url": p["url"], "title": p["title"]}
                 for p in cc["pages"]]
        if not pages:
            continue
        comp = ContextCompressor(documents=pages, embeddings=embeddings,
                                 similarity_threshold=threshold,
                                 prompt_family=prompt_family)
        parts.append(await comp.async_get_context(query=cc["subquery"], max_results=10))
    return "\n\n".join(parts)


def build_full_context(cap):
    """No filtering at all -- every scraped page, concatenated."""
    seen, parts = set(), []
    for cc in cap["compress_calls"]:
        for p in cc["pages"]:
            if p["url"] in seen:
                continue
            seen.add(p["url"])
            parts.append(f"Source: {p['url']}\nTitle: {p['title']}\n"
                         f"Content: {p['raw_content']}")
    return "\n".join(parts)


async def main():
    _keys()
    from langchain_openai import ChatOpenAI
    from gpt_researcher.memory import Memory
    from gpt_researcher.prompts import get_prompt_family
    from gpt_researcher.config import Config
    from evals.quality_eval import metrics

    grader = ChatOpenAI(temperature=0, model_name=GRADER,
                        openai_api_key=os.environ["OPENAI_API_KEY"])
    embeddings = Memory("openai", "text-embedding-3-small").get_embeddings()
    pf = get_prompt_family("default", Config())

    caps = json.loads(CAP.read_text(encoding="utf-8"))
    conditions = [f"thr={t}" for t in THRESHOLDS] + ["full(无过滤)"]

    print(f"grader = {GRADER}   评分时 context 不截断 ({SCORE_CHARS:,} 上限)\n")
    print(f"{'topic':<9}{'条件':<14}{'context字符':>12}{'unsupported':>13}{'claims':>8}")

    agg = {c: [] for c in conditions}
    size = {c: [] for c in conditions}
    for cap in caps:
        # Extract the claim set ONCE from the fixed report, reuse for every condition.
        base = await metrics.unsupported_claim(
            cap["report"], cap["context"], grader, context_chars=SCORE_CHARS)
        claims = [c["claim"] for c in base.get("breakdown", [])]
        if not claims:
            print(f"{cap['topic']:<9}  claim 抽取失败，跳过")
            continue

        ctxs = {}
        for t in THRESHOLDS:
            ctxs[f"thr={t}"] = await build_context(cap, t, embeddings, pf)
        ctxs["full(无过滤)"] = build_full_context(cap)

        for cond in conditions:
            ctx = ctxs[cond]
            r = await metrics.unsupported_claim(
                cap["report"], ctx, grader, claims=claims, context_chars=SCORE_CHARS)
            agg[cond].append(r["unsupported_claim_rate"])
            size[cond].append(len(ctx))
            print(f"{cap['topic']:<9}{cond:<14}{len(ctx):>12,}"
                  f"{r['unsupported_claim_rate']:>13.3f}{len(claims):>8}")
        print()

    print("=" * 60)
    print(f"{'条件':<14}{'平均context':>13}{'平均unsupported':>17}")
    for cond in conditions:
        if agg[cond]:
            print(f"{cond:<14}{sum(size[cond])/len(size[cond]):>13,.0f}"
                  f"{sum(agg[cond])/len(agg[cond]):>17.3f}")

    Path("_research/data/e2_result.json").write_text(
        json.dumps({"thresholds": THRESHOLDS, "grader": GRADER,
                    "rates": agg, "sizes": size}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
