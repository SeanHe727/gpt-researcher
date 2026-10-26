"""Attribute every unsupported claim to the pipeline layer that failed it.

For each claim the metric scored 0.0 against the pipeline's context, go back to
the RAW scraped corpus -- everything the scraper actually retrieved, before any
compression -- and retrieve the passages most similar to the claim itself (not
to the sub-query the pipeline searched with). Then ask the grader whether those
passages support it, and check whether they had made it into the final context.

    evidence in raw corpus, NOT in context   -> the selection layer dropped it
    evidence in raw corpus, IS in context    -> writer mis-cited, or grader erred
    no evidence in raw corpus                -> retrieval/scrape never found it

That splits the single end-to-end unsupported rate into per-layer buckets.
"""
import asyncio
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")

CAP = Path("_research/data/pipeline_capture.json")
GRADER = "gpt-4-turbo"
CHUNK = 1200          # chars per corpus chunk
TOP_K = 5             # passages retrieved per claim
CTX_CHARS = 200_000   # score against full context, no truncation

_SUPPORT_PROMPT = """
You are a fact-checking assistant.

Decide whether the passages below contain evidence that supports the claim.
Judge only on the passages -- do not use your own knowledge.

Claim: {claim}

Passages:
{passages}

Return ONLY a JSON object:
{{"supported": true, "which": "<short quote from the supporting passage>"}}
or
{{"supported": false, "which": ""}}
""".strip()


def _keys():
    p = Path("D:/chenh/api_keys/gpt_api_key.txt")
    if not os.getenv("OPENAI_API_KEY") and p.exists():
        os.environ["OPENAI_API_KEY"] = p.read_text().strip()


def _norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def chunks_of(cap):
    """Every scraped page, split into fixed-size chunks (deduped by url)."""
    seen, out = set(), []
    for cc in cap["compress_calls"]:
        for p in cc["pages"]:
            if p["url"] in seen:
                continue
            seen.add(p["url"])
            txt = p["raw_content"] or ""
            for i in range(0, len(txt), CHUNK):
                piece = txt[i:i + CHUNK]
                if len(piece) > 200:
                    out.append({"url": p["url"], "text": piece})
    return out


def cosine_top(qvec, mat, k):
    import numpy as np
    q = np.array(qvec, dtype="float32")
    m = np.array(mat, dtype="float32")
    q /= (np.linalg.norm(q) + 1e-9)
    m /= (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    sims = m @ q
    idx = sims.argsort()[::-1][:k]
    return [(int(i), float(sims[i])) for i in idx]


async def main():
    _keys()
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from evals.quality_eval import metrics

    grader = ChatOpenAI(temperature=0, model_name=GRADER,
                        openai_api_key=os.environ["OPENAI_API_KEY"])
    emb = OpenAIEmbeddings(model="text-embedding-3-small",
                           openai_api_key=os.environ["OPENAI_API_KEY"])
    caps = json.loads(CAP.read_text(encoding="utf-8"))

    buckets = {"selection": 0, "writer_or_grader": 0, "retrieval": 0}
    rows = []

    for cap in caps:
        print(f"\n=== {cap['topic']}")
        res = await metrics.unsupported_claim(
            cap["report"], cap["context"], grader, context_chars=CTX_CHARS)
        bad = [c for c in res["breakdown"] if c.get("category") == "unsupported"]
        print(f"  {len(bad)}/{res['total_claims']} 条 claim 被判无据，逐条回溯原始语料")
        if not bad:
            continue

        ch = chunks_of(cap)
        print(f"  原始语料 {len(ch)} 个片段 ({sum(len(c['text']) for c in ch):,} 字符)")
        mat = await emb.aembed_documents([c["text"] for c in ch])
        ctx_norm = _norm(cap["context"])

        for c in bad:
            claim = c["claim"]
            qv = await emb.aembed_query(claim)
            top = cosine_top(qv, mat, TOP_K)
            passages = "\n\n".join(
                f"[{i+1}] (from {ch[j]['url']})\n{ch[j]['text']}"
                for i, (j, _) in enumerate(top))
            try:
                r = await grader.ainvoke(
                    _SUPPORT_PROMPT.format(claim=claim, passages=passages[:12000]))
                verdict = metrics._parse_json(r.content)
            except Exception:
                verdict = {"supported": False, "which": ""}

            in_ctx = any(_norm(ch[j]["text"])[:300] in ctx_norm for j, _ in top)
            if verdict.get("supported"):
                layer = "writer_or_grader" if in_ctx else "selection"
            else:
                layer = "retrieval"
            buckets[layer] += 1
            rows.append({"topic": cap["topic"], "claim": claim, "layer": layer,
                         "raw_supported": bool(verdict.get("supported")),
                         "top_sim": round(top[0][1], 3)})
            print(f"    [{layer:<17}] sim={top[0][1]:.3f}  {claim[:66]}")

    total = sum(buckets.values())
    print("\n" + "=" * 72)
    print("无据 claim 的责任层归因")
    print("=" * 72)
    labels = {
        "selection":        "证据在原始语料、未进 context  -> 选择层(top-10/相似度)",
        "writer_or_grader": "证据在 context 里却判无据     -> 写作层引用错配 / 判分误差",
        "retrieval":        "原始语料中无证据              -> 检索层 or 抓取层",
    }
    for k, lbl in labels.items():
        n = buckets[k]
        print(f"  {lbl:<52}{n:>3} / {total}  ({n/total*100 if total else 0:>4.0f}%)")

    Path("_research/data/attribution.json").write_text(
        json.dumps({"buckets": buckets, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n  saved -> _research/data/attribution.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
