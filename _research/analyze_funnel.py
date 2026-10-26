"""E1 — funnel analysis over the captured runs.

Per stage: how much goes in, how much comes out, what is lost. The interesting
question is not only how much is dropped but WHICH sources are dropped: a stage
that loses 30% uniformly is very different from one that loses 30% concentrated
on the authoritative domains.
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


def tier(d):
    m = _score_domain(d)
    return f"{m[1]} ({m[0]})" if m else "deferred/unknown"


def main():
    caps = json.loads(CAP.read_text(encoding="utf-8"))

    print("=" * 78)
    print("E1  漏斗：每一层的进出与留存率")
    print("=" * 78)
    print(f"{'topic':<9}{'子查询':>7}{'URL':>6}{'抓成功':>8}{'抓取率':>8}"
          f"{'抓取字符':>11}{'context':>10}{'压缩留存':>9}{'报告':>8}")

    tot = {"urls": 0, "pages": 0, "chars_in": 0, "chars_ctx": 0}
    all_failed, all_ok = [], []
    for c in caps:
        urls_in = sum(s["urls_in"] for s in c["scrape_calls"])
        pages   = sum(s["pages_out"] for s in c["scrape_calls"])
        chars   = sum(s["chars_out"] for s in c["scrape_calls"])
        ctx     = c["context_len"]
        tot["urls"] += urls_in; tot["pages"] += pages
        tot["chars_in"] += chars; tot["chars_ctx"] += ctx

        # which URLs survived scraping?
        tried = {u for s in c["scrape_calls"] for u in s["urls"]}
        got   = {p["url"] for cc in c["compress_calls"] for p in cc["pages"] if p["url"]}
        all_ok     += [dom(u) for u in got if u]
        all_failed += [dom(u) for u in tried - got if u]

        print(f"{c['topic']:<9}{len(c['subqueries']):>7}{urls_in:>6}{pages:>8}"
              f"{pages/urls_in*100:>7.0f}%{chars:>11,}{ctx:>10,}"
              f"{ctx/chars*100:>8.0f}%{len(c['report']):>8,}")

    print("-" * 78)
    print(f"{'合计':<9}{'':>7}{tot['urls']:>6}{tot['pages']:>8}"
          f"{tot['pages']/tot['urls']*100:>7.0f}%{tot['chars_in']:>11,}"
          f"{tot['chars_ctx']:>10,}{tot['chars_ctx']/tot['chars_in']*100:>8.0f}%")

    print("\n" + "=" * 78)
    print("抓取失败的都是什么源？（规则表可判定的高权威域名）")
    print("=" * 78)

    def summarise(label, doms):
        from collections import Counter
        cnt = Counter(doms)
        scored = [(d, _score_domain(d)) for d in cnt]
        hi = [(d, m) for d, m in scored if m and m[0] >= 0.9]
        print(f"\n{label}: {len(doms)} 个 URL, {len(cnt)} 个域名")
        print(f"  其中规则表判定为高权威(>=0.9)的域名: {len(hi)}")
        for d, m in sorted(hi, key=lambda x: -x[1][0]):
            print(f"    {d:<38} {m[1]:<16} {m[0]}")

    summarise("抓取成功", all_ok)
    summarise("抓取失败", all_failed)

    print("\n" + "=" * 78)
    print("E3  快路径分叉：实际触发了吗？")
    print("=" * 78)
    print(f"{'topic':<9}{'子查询语料(字符)':>18}{'阈值':>8}{'走快路径?':>11}")
    for c in caps:
        for cc in c["compress_calls"]:
            print(f"{c['topic']:<9}{cc['chars_in']:>18,}{cc['threshold']:>8,}"
                  f"{('是' if cc['fast_path'] else '否'):>11}")
    allc = [cc for c in caps for cc in c["compress_calls"]]
    fast = sum(cc["fast_path"] for cc in allc)
    mn = min(cc["chars_in"] for cc in allc)
    print(f"\n  {len(allc)} 次压缩调用中, 走快路径 {fast} 次")
    print(f"  最小的一次子查询语料 = {mn:,} 字符, 阈值 = 8,000")
    print(f"  => 实际语料比阈值大 {mn/8000:.0f}-{max(cc['chars_in'] for cc in allc)/8000:.0f} 倍")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
