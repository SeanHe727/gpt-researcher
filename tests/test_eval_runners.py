"""Offline tests for the quality_eval runners and the pipeline probe.

The runners are exercised end to end with a fake researcher and a fake
grader, so a broken import, a renamed field or a crash in the console
summary shows up here instead of halfway through a paid run. The probe
tests drive it through the library's own observer dispatch, so they fail if
the research_observer hook or the events the probe reads change.
"""

import inspect
import json
import unittest

from evals.quality_eval.pipeline_probe import PipelineProbe, PipelineTrace, sources_to_score

REPORT = """# Report

Intro citing [NIH](https://www.nih.gov/a) and [Nature](https://nature.com/b).

## Section A
Treatment uptake increased by 20% ([Mayo](https://mayoclinic.org/c)).

## References
- [NIH](https://www.nih.gov/a)
- [Nature](https://nature.com/b)
"""
SOURCES = ["https://www.nih.gov/a", "https://nature.com/b", "https://mayoclinic.org/c"]
CONTEXT = "Source: https://www.nih.gov/a\nTreatment uptake increased by 20%."


class _Resp:
    def __init__(self, content):
        self.content = content


class _FakeGrader:
    """Answers each quality_eval prompt with a well-formed reply, or raises."""

    def __init__(self, fail: bool = False):
        self.fail = fail

    async def ainvoke(self, prompt):
        if self.fail:
            raise RuntimeError("grader unavailable")
        if "List the key subtopics" in prompt:
            return _Resp(json.dumps(["treatments", "outcomes"]))
        if "single subtopic" in prompt:
            return _Resp(json.dumps({"covered": True}))
        if "Extract specific, verifiable" in prompt:
            return _Resp(json.dumps(["Uptake increased by 20%"]))
        if "credibility score" in prompt:
            return _Resp(json.dumps([{"claim": "x", "category": "supported", "score": 1.0}]))
        if "Rate the authority" in prompt:
            return _Resp(json.dumps([{"domain": "mayoclinic.org", "score": 0.9, "reason": "clinic"}]))
        raise AssertionError(f"unexpected prompt: {prompt[:80]}")


class _FakeResearcher:
    def __init__(self, **kwargs):
        pass

    async def conduct_research(self):
        return CONTEXT

    async def write_report(self):
        return REPORT

    def get_source_urls(self):
        return SOURCES

    def get_costs(self):
        return 0.01


class TestSourcesToScore(unittest.TestCase):

    def test_scraped_pages_when_probe_saw_the_scrape_stage(self):
        t = PipelineTrace()
        t.record_scrape(SOURCES, [{"url": SOURCES[1], "raw_content": "x"}])
        self.assertEqual(sources_to_score(t, SOURCES), ([SOURCES[1]], "scraped"))

    def test_every_scrape_failing_scores_nothing_rather_than_falling_back(self):
        t = PipelineTrace()
        t.record_scrape(SOURCES, [])
        self.assertEqual(sources_to_score(t, SOURCES), ([], "scraped"))

    def test_falls_back_to_visited_only_when_probe_recorded_nothing(self):
        self.assertEqual(sources_to_score(PipelineTrace(), SOURCES), (SOURCES, "visited"))


class TestProbeObserver(unittest.IsolatedAsyncioTestCase):
    """PipelineProbe records a run from research_observer events."""

    async def _emit(self, probe, events):
        from gpt_researcher.observability import emit_research_event
        for name, fields in events:
            await emit_research_event(probe, name, **fields)

    async def test_funnel_from_observer_events(self):
        probe = PipelineProbe()
        await self._emit(probe, [
            ("query_planned", {"query": "sub 1", "position": 1}),
            ("query_planned", {"query": "sub 2", "position": 2}),
            ("selected_for_fetch", {"query": "sub 1", "url": SOURCES[0]}),
            ("selected_for_fetch", {"query": "sub 1", "url": SOURCES[1]}),
            ("candidate_deduplicated", {"query": "sub 2", "url": SOURCES[0],
                                        "reason_code": "ALREADY_VISITED"}),
            ("selected_for_fetch", {"query": "sub 2", "url": SOURCES[2]}),
            ("fetch_succeeded", {"url": SOURCES[0], "scraper": "bs", "content_length": 10}),
            ("fetch_failed", {"url": SOURCES[1], "scraper": "bs", "content_length": 0,
                              "reason_code": "HTTP_403"}),
            ("content_prefetched", {"query": "sub 2", "provider": "x", "url": SOURCES[2],
                                    "rank": 1, "content_length": 5}),
        ])
        t = probe.trace
        self.assertEqual(t.subqueries, ["sub 1", "sub 2"])
        self.assertEqual(t.scraped_urls, [SOURCES[0], SOURCES[2]])
        self.assertEqual(t.failed_urls, [SOURCES[1]])
        f = t.funnel()
        self.assertEqual((f["urls_proposed"], f["urls_duplicate"]), (4, 1))
        self.assertEqual((f["urls_attempted"], f["urls_scraped"]), (3, 2))
        self.assertIsNone(f["compression_rate"])   # not reported by the hook
        self.assertEqual(sources_to_score(t, SOURCES), ([SOURCES[0], SOURCES[2]], "scraped"))

    def test_researcher_accepts_the_probe_as_observer(self):
        from gpt_researcher.agent import GPTResearcher
        params = inspect.signature(GPTResearcher.__init__).parameters
        self.assertIn("research_observer", params)


class TestRunEvalOffline(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        from evals.quality_eval import run_eval
        self.run_eval = run_eval
        self._orig = run_eval.GPTResearcher
        run_eval.GPTResearcher = _FakeResearcher

    async def asyncTearDown(self):
        self.run_eval.GPTResearcher = self._orig

    async def test_all_metrics_score_with_a_working_grader(self):
        r = await self.run_eval.evaluate_single_query("q?", _FakeGrader(), True, False, True)
        self.assertEqual(r["subtopic_coverage"]["subtopic_coverage_rate"], 1.0)
        self.assertEqual(r["unsupported_claim"]["avg_claim_score"], 1.0)
        self.assertEqual(r["unsupported_claim"]["breakdown"][0]["claim"], "Uptake increased by 20%")
        # The fake researcher never reaches the scrape stage, so the fallback is labelled.
        self.assertEqual(r["sources_basis"], "visited")

    async def test_grader_outage_skips_metrics_instead_of_scoring_zero(self):
        r = await self.run_eval.evaluate_single_query("q?", _FakeGrader(fail=True), True, False, True)
        self.assertEqual(r["subtopic_coverage"]["status"], "skipped")
        self.assertEqual(r["unsupported_claim"]["status"], "skipped")
        self.assertEqual(r["source_authority"]["llm_unscored_domains"], ["mayoclinic.org"])
        # The real cause is reported, not a generic "failed to parse".
        self.assertIn("grader unavailable", r["subtopic_coverage"]["reason"])


class TestJudge(unittest.TestCase):

    def test_env_override_is_read_at_call_time(self):
        import os
        from unittest import mock
        from evals.judge import DEFAULT_JUDGE_MODEL, judge_model_name

        with mock.patch.dict(os.environ, {"EVAL_JUDGE_MODEL": "gpt-4o"}):
            self.assertEqual(judge_model_name(), "gpt-4o")
        with mock.patch.dict(os.environ, {"EVAL_JUDGE_MODEL": ""}):
            self.assertEqual(judge_model_name(), DEFAULT_JUDGE_MODEL)

    def test_reasoning_models_get_no_temperature(self):
        import os
        from unittest import mock
        from evals.judge import make_judge

        # ChatOpenAI refuses to build without a key; no call is made.
        mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}).start()
        self.addCleanup(mock.patch.stopall)
        self.assertEqual(make_judge("gpt-4o").temperature, 0)
        self.assertIsNone(make_judge("gpt-5.4-mini").temperature)
        self.assertIsNone(make_judge("gpt-6-luna").temperature)


class TestBenchmarkOffline(unittest.IsolatedAsyncioTestCase):

    async def test_single_agent_query_evaluates(self):
        from evals.quality_eval import benchmark

        async def fake_single(query, model, probe=None):
            return REPORT, SOURCES, CONTEXT, 0.01

        orig = benchmark._run_single_agent
        benchmark._run_single_agent = fake_single
        try:
            r = await benchmark._evaluate_query(
                "medical", "q?", {"arch": "single", "model": "m"}, _FakeGrader(), True, {})
        finally:
            benchmark._run_single_agent = orig
        self.assertNotIn("error", r)
        self.assertEqual(r["sources_basis"], "visited")
        self.assertEqual(r["subtopic_coverage"]["subtopic_coverage_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
