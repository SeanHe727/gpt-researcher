"""Is the scrape failure rate independent of source authority?

The funnel shows 30% of retrieved URLs never become pages. If those failures
were uniform it would just be lossy. But the domains that failed (congress.gov,
pmc.ncbi.nlm.nih.gov, mayo.edu) are exactly the ones the authority metric rates
highest -- authoritative publishers tend to sit behind the strictest bot
defences. That would mean the scraper systematically strips the best sources
the retriever found, which is a pipeline defect, not a retrieval one.

Uses only the deterministic rule table (.gov/.edu/whitelist/wiki), so the split
costs nothing and cannot be argued with on grader-variance grounds.
"""
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, ".")
from evals.quality_eval.metrics import _score_domain

CAP = Path("_research/data/pipeline_capture.json")


def dom(u):
    return urlparse(u).netloc.lower().removeprefix("www.")


def main():
    caps = json.loads(CAP.read_text(encoding="utf-8"))
    rows = []
    for c in caps:
        tried = {u for s in c["scrape_calls"] for u in s["urls"] if u}
        got = {p["url"] for cc in c["compress_calls"] for p in cc["pages"] if p["url"]}
        for u in tried:
            d = dom(u)
            m = _score_domain(d)
            rows.append({"topic": c["topic"], "url": u, "domain": d,
                         "score": m[0] if m else None,
                         "tier": m[1] if m else "unknown(.com/.org)",
                         "ok": u in got})

    hi = [r for r in rows if r["score"] is not None and r["score"] >= 0.9]
    lo = [r for r in rows if not (r["score"] is not None and r["score"] >= 0.9)]

    def rate(g):
        return sum(r["ok"] for r in g) / len(g) * 100 if g else float("nan")

    print("=" * 72)
    print("抓取成功率  按来源权威性分组（规则表判定，确定性）")
    print("=" * 72)
    print(f"{'分组':<28}{'URL数':>7}{'抓成功':>8}{'成功率':>9}{'失败率':>9}")
    print(f"{'高权威 (.gov/.edu/白名单)':<26}{len(hi):>7}{sum(r['ok'] for r in hi):>8}"
          f"{rate(hi):>8.0f}%{100-rate(hi):>8.0f}%")
    print(f"{'其余 (.com/.org/未知)':<27}{len(lo):>7}{sum(r['ok'] for r in lo):>8}"
          f"{rate(lo):>8.0f}%{100-rate(lo):>8.0f}%")
    print(f"{'全部':<28}{len(rows):>7}{sum(r['ok'] for r in rows):>8}"
          f"{rate(rows):>8.0f}%{100-rate(rows):>8.0f}%")

    if hi and lo:
        print(f"\n  高权威源的失败率是其余源的 {(100-rate(hi))/max(100-rate(lo),1e-9):.1f} 倍")

    print("\n" + "=" * 72)
    print("逐条明细（高权威源）")
    print("=" * 72)
    for r in sorted(hi, key=lambda x: (x["ok"], x["domain"])):
        print(f"  {'OK  ' if r['ok'] else 'FAIL'}  {r['domain']:<34}{r['tier']:<16}{r['score']}")

    print("\n  其余源中抓取失败的：")
    for r in sorted([r for r in lo if not r["ok"]], key=lambda x: x["domain"]):
        print(f"        {r['domain']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
