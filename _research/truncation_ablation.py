"""Ablation: how much does the context-truncation limit drive the
unsupported_claim baseline?

unsupported_claim scores each claim against `context[:CONTEXT_CHARS]`. The
fixtures carry 8000 chars of context but the default cut is 4000, so half the
evidence is discarded before scoring. If the baseline unsupported rate falls as
the window grows, the high baseline (0.75-0.88) is an evaluation artefact. If it
stays flat, the agent really is writing claims its own sources do not support --
a pipeline defect rather than a metric one.

Claims are extracted ONCE per fixture and reused at every window, so the only
thing changing is how much evidence the grader can see.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
from evals.quality_eval import metrics

WINDOWS = [1000, 2000, 4000, 8000]
FIX = Path("./evals/quality_eval/"
           "fixtures/perturbation_fixtures.json")


async def main(model: str):
    from langchain_openai import ChatOpenAI
    grader = ChatOpenAI(temperature=0, model_name=model,
                        openai_api_key=os.environ["OPENAI_API_KEY"])
    fixtures = json.loads(FIX.read_text(encoding="utf-8"))

    print(f"grader = {model}   windows = {WINDOWS}\n")
    print(f"  {'topic':<9}{'ctx len':>9}" +
          "".join(f"{w:>9}" for w in WINDOWS) + "   claims")
    rows = {}
    for fx in fixtures:
        ctx = fx["context"]
        ctx = ctx if isinstance(ctx, str) else json.dumps(ctx)
        # Extract the claim set ONCE, then hold it fixed across windows.
        base = await metrics.unsupported_claim(fx["report"], ctx, grader)
        claims = [c["claim"] for c in base.get("breakdown", [])]

        vals = []
        for w in WINDOWS:
            r = await metrics.unsupported_claim(
                fx["report"], ctx, grader, claims=claims, context_chars=w)
            vals.append(r["unsupported_claim_rate"])
        rows[fx["topic"]] = vals
        print(f"  {fx['topic']:<9}{len(ctx):>9}" +
              "".join(f"{v:>9.3f}" for v in vals) + f"   n={len(claims)}")

    print("\n  " + "-" * 56)
    avg = [sum(rows[t][i] for t in rows) / len(rows) for i in range(len(WINDOWS))]
    print(f"  {'MEAN':<9}{'':>9}" + "".join(f"{v:>9.3f}" for v in avg))
    delta = avg[0] - avg[-1]
    print(f"\n  1000 -> 8000 字符，unsupported rate 变化: {delta:+.3f}")
    if delta > 0.10:
        print("  => 截断是主因：证据被切掉导致 claim 被误判为无依据（评估缺陷）")
    elif abs(delta) <= 0.10:
        print("  => 与截断基本无关：高基线源于报告本身或评分标准（需进一步归因）")

    out = Path(__file__).parent / f"truncation_ablation_{model}.json"
    out.write_text(json.dumps({"windows": WINDOWS, "grader": model,
                               "rows": rows, "mean": avg}, indent=2),
                   encoding="utf-8")
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not os.getenv("OPENAI_API_KEY"):
        kp = Path("D:/chenh/api_keys/gpt_api_key.txt")
        if kp.exists():
            os.environ["OPENAI_API_KEY"] = kp.read_text().strip()
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "gpt-4-turbo"))
