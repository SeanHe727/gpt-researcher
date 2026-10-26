"""Verify the subtopic_coverage false-positive floor is gone.

Before the fix, degrade_subtopics(report, 1.0) leaves nothing but the
`## References` list -- no prose at all -- yet the grader scored 43-57% of
subtopics as "covered", reading the coverage off reference titles. After the
fix the body is split away from the reference list before judging, so a
references-only report should score ~0.

Also re-checks the 0% (undegraded) level to confirm the fix did not cost real
coverage on an intact report.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
from evals.quality_eval import metrics, perturbation as P

BEFORE = {"medical": (0.857, 0.429), "ai_tech": (1.000, 0.143), "legal": (0.857, 0.571)}
FIX = Path("./evals/quality_eval/"
           "fixtures/perturbation_fixtures.json")


async def main(model):
    from langchain_openai import ChatOpenAI
    grader = ChatOpenAI(temperature=0, model_name=model,
                        openai_api_key=os.environ["OPENAI_API_KEY"])
    fixtures = json.loads(FIX.read_text(encoding="utf-8"))

    print(f"grader = {model}\n")
    print(f"  {'topic':<9}{'0% before':>11}{'0% after':>10}"
          f"{'100% before':>13}{'100% after':>12}")
    floors = []
    for fx in fixtures:
        base = await metrics.subtopic_coverage(fx["query"], fx["report"], grader)
        subs = base.get("expected_subtopics", [])
        intact = await metrics.subtopic_coverage(
            fx["query"], fx["report"], grader, subtopics=subs)
        empty = await metrics.subtopic_coverage(
            fx["query"], P.degrade_subtopics(fx["report"], 1.0), grader, subtopics=subs)
        b0, b1 = BEFORE[fx["topic"]]
        a0 = intact["subtopic_coverage_rate"]
        a1 = empty["subtopic_coverage_rate"]
        floors.append(a1)
        print(f"  {fx['topic']:<9}{b0:>11.3f}{a0:>10.3f}{b1:>13.3f}{a1:>12.3f}")

    mean_floor = sum(floors) / len(floors)
    print(f"\n  假阳性地板: 修复前 0.381 -> 修复后 {mean_floor:.3f}")
    print("  " + ("PASS: 空报告不再被判有覆盖" if mean_floor < 0.10
                  else "FAIL: 地板仍在，需进一步处理"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not os.getenv("OPENAI_API_KEY"):
        kp = Path("D:/chenh/api_keys/gpt_api_key.txt")
        if kp.exists():
            os.environ["OPENAI_API_KEY"] = kp.read_text().strip()
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "gpt-4-turbo"))
