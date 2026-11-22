"""Tests for the optional research lifecycle observer."""

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpt_researcher.agent import GPTResearcher
from gpt_researcher.observability import emit_research_event
from gpt_researcher.scraper.scraper import Scraper
from gpt_researcher.skills.researcher import ResearchConductor


@pytest.mark.asyncio
async def test_observer_accepts_sync_async_and_object_handlers():
    events = []

    def sync_observer(event, **fields):
        events.append((event, fields))

    await emit_research_event(
        sync_observer, "candidate_discovered", url="https://example.test"
    )

    async def async_observer(event, **fields):
        events.append((event, fields))

    await emit_research_event(
        async_observer, "fetch_succeeded", url="https://example.test"
    )

    class ObjectObserver:
        def on_event(self, event, **fields):
            events.append((event, fields))

    await emit_research_event(ObjectObserver(), "selected_for_fetch", position=1)

    assert [event for event, _ in events] == [
        "candidate_discovered",
        "fetch_succeeded",
        "selected_for_fetch",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("observer", [None, lambda *args, **kwargs: None])
async def test_observer_noop_and_success_return_none(observer):
    assert await emit_research_event(observer, "query_planned", position=1) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_observer_failure_is_suppressed(asynchronous):
    if asynchronous:
        async def broken_observer(*args, **kwargs):
            raise RuntimeError("telemetry unavailable")
    else:
        def broken_observer(*args, **kwargs):
            raise RuntimeError("telemetry unavailable")

    await emit_research_event(
        broken_observer, "fetch_failed", url="https://example.test"
    )


def test_public_observer_parameter_is_appended_without_shifting_existing_parameters():
    parameters = list(inspect.signature(GPTResearcher.__init__).parameters)

    assert parameters[-3:] == ["mcp_strategy", "research_observer", "kwargs"]


@pytest.mark.asyncio
async def test_query_planning_emits_queries_in_order():
    events = []

    async def observer(event, **fields):
        events.append((event, fields))

    researcher = SimpleNamespace(
        retrievers=[],
        websocket=None,
        role=None,
        cfg=SimpleNamespace(),
        parent_query="",
        report_type="research_report",
        add_costs=lambda cost: None,
        kwargs={},
        research_observer=observer,
    )
    conductor = ResearchConductor(researcher)

    with (
        patch(
            "gpt_researcher.skills.researcher.plan_research_outline",
            AsyncMock(return_value=["first query", "second query"]),
        ),
        patch(
            "gpt_researcher.skills.researcher.stream_output",
            AsyncMock(),
        ),
    ):
        outline = await conductor.plan_research("question", search_results=[])

    assert outline == ["first query", "second query"]
    assert events == [
        ("query_planned", {"query": "first query", "position": 1}),
        ("query_planned", {"query": "second query", "position": 2}),
    ]


@pytest.mark.asyncio
async def test_scraper_without_observer_preserves_results():
    scraper = Scraper.__new__(Scraper)
    scraper.urls = ["https://good.test", "https://bad.test"]
    scraper.session = object()
    scraper.scraper = "bs"
    scraper.event_handler = None
    scraper.extract_data_from_url = AsyncMock(
        side_effect=[
            {"url": "https://good.test", "raw_content": "usable", "title": "Good"},
            {"url": "https://bad.test", "raw_content": None, "title": "Bad"},
        ]
    )

    retained = await scraper.run()

    assert retained == [
        {"url": "https://good.test", "raw_content": "usable", "title": "Good"}
    ]


@pytest.mark.asyncio
async def test_scraper_emits_before_failed_rows_are_filtered():
    events = []

    async def observer(event, **fields):
        events.append((event, fields))

    scraper = Scraper.__new__(Scraper)
    scraper.urls = ["https://good.test", "https://bad.test"]
    scraper.session = object()
    scraper.scraper = "bs"
    scraper.event_handler = observer
    scraper.extract_data_from_url = AsyncMock(
        side_effect=[
            {"url": "https://good.test", "raw_content": "usable", "title": "Good"},
            {
                "url": "https://bad.test",
                "raw_content": None,
                "title": "Bad",
                "failure_reason": "BACKEND_FAILURE",
            },
        ]
    )

    retained = await scraper.run()

    assert [item["url"] for item in retained] == ["https://good.test"]
    assert [event for event, _ in events] == ["fetch_succeeded", "fetch_failed"]
    assert "reason_code" not in events[0][1]
    assert events[1][1]["reason_code"] == "BACKEND_FAILURE"
    assert all("raw_content" not in fields for _, fields in events)


@pytest.mark.asyncio
async def test_search_emits_candidates_empty_failure_and_deduplication():
    events = []

    class ResultsRetriever:
        requires_scraping = True

        def __init__(self, query, query_domains=None):
            pass

        def search(self, max_results=None):
            return [
                {"href": "https://one.test"},
                {"href": "https://seen.test"},
            ]

    class EmptyRetriever(ResultsRetriever):
        def search(self, max_results=None):
            return []

    class FailedRetriever(ResultsRetriever):
        def search(self, max_results=None):
            raise OSError("provider unavailable")

    async def observer(event, **fields):
        events.append((event, fields))

    researcher = SimpleNamespace(
        retrievers=[ResultsRetriever, EmptyRetriever, FailedRetriever],
        cfg=SimpleNamespace(max_search_results_per_query=5),
        visited_urls={"https://seen.test"},
        verbose=False,
        websocket=None,
        research_observer=observer,
    )

    urls, _ = await ResearchConductor(researcher)._search_relevant_source_urls(
        "query"
    )

    assert urls == ["https://one.test"]
    candidates = [fields for event, fields in events if event == "candidate_discovered"]
    assert [item["rank"] for item in candidates] == [1, 2]
    assert any(event == "search_empty" for event, _ in events)
    failure = next(fields for event, fields in events if event == "search_failed")
    assert failure["reason_code"] == "OSError"
    selected = next(fields for event, fields in events if event == "selected_for_fetch")
    assert "reason_code" not in selected
    deduplicated = next(
        fields for event, fields in events if event == "candidate_deduplicated"
    )
    assert deduplicated["url"] == "https://seen.test"


def _recording_observer():
    events = []

    async def observer(event, **fields):
        events.append((event, fields))

    return events, observer


@pytest.mark.asyncio
async def test_context_compression_reports_sizes_without_content():
    from gpt_researcher.skills.context_manager import ContextManager

    events, observer = _recording_observer()
    researcher = SimpleNamespace(
        verbose=False,
        websocket=None,
        cfg=SimpleNamespace(),
        prompt_family=None,
        add_costs=lambda cost: None,
        memory=SimpleNamespace(get_embeddings=lambda: None),
        research_observer=observer,
    )
    pages = [{"url": "https://a.test", "raw_content": "x" * 30},
             {"url": "https://b.test", "raw_content": "y" * 20}]

    with patch(
        "gpt_researcher.skills.context_manager.select_context",
        AsyncMock(return_value="selected text"),
    ):
        context = await ContextManager(researcher).get_similar_content_by_query("q", pages)

    assert context == "selected text"
    assert events == [("context_compressed", {
        "query": "q", "pages": 2, "input_length": 50, "content_length": 13,
    })]


@pytest.mark.asyncio
async def test_context_assembled_maps_joined_context_back_to_queries():
    events, observer = _recording_observer()
    researcher = SimpleNamespace(
        retrievers=[],
        cfg=SimpleNamespace(),
        mcp_strategy=None,
        report_type="research_report",
        verbose=False,
        websocket=None,
        research_observer=observer,
    )
    conductor = ResearchConductor(researcher)
    by_query = {"sub 1": "alpha", "sub 2": "", "question": "gamma gamma"}
    conductor._get_initial_search_results = AsyncMock(return_value=[{"href": "https://a.test"}])
    conductor.plan_research = AsyncMock(return_value=["sub 1", "sub 2"])
    conductor._get_context_from_initial_results = AsyncMock(return_value="initial")
    conductor._process_sub_query = AsyncMock(side_effect=lambda q, *a: by_query[q])

    context = await conductor._get_context_by_web_search("question")

    (event, fields), = [e for e in events if e[0] == "context_assembled"]
    assert fields["query"] == "question"
    assert [(p["query"], p["source"]) for p in fields["parts"]] == [
        ("question", "initial_results"), ("sub 1", "sub_query"), ("question", "sub_query"),
    ]
    # The lengths and separator recover each piece of the returned context.
    pieces, start = [], 0
    for part in fields["parts"]:
        pieces.append(context[start:start + part["content_length"]])
        start += part["content_length"] + len(fields["separator"])
    assert pieces == ["initial", "alpha", "gamma gamma"]
    assert all("content" not in p for p in fields["parts"])


@pytest.mark.asyncio
async def test_deep_research_reports_levels_and_shares_the_observer():
    from gpt_researcher.skills.deep_research import DeepResearchSkill

    events, observer = _recording_observer()
    created = []

    class FakeResearcher:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.visited_urls = set()
            self.research_sources = []

        async def conduct_research(self):
            return "context"

        def get_costs(self):
            return 0.0

    skill = DeepResearchSkill.__new__(DeepResearchSkill)
    skill.researcher = SimpleNamespace(
        research_observer=observer, mcp_configs=None, mcp_strategy=None,
        add_costs=lambda cost: None,
    )
    skill.concurrency_limit = 1
    skill.tone = None
    skill.websocket = None
    skill.config_path = None
    skill.headers = {}
    skill.visited_urls = set()
    skill.context = []
    skill.research_sources = []
    skill.generate_search_queries = AsyncMock(
        return_value=[{"query": "branch", "researchGoal": "goal"}])
    skill.process_research_results = AsyncMock(return_value={
        "learnings": ["a learning"], "followUpQuestions": ["next?"], "citations": {}})

    with patch("gpt_researcher.GPTResearcher", FakeResearcher):
        await skill.deep_research("root", breadth=1, depth=1)

    assert created[0]["research_observer"] is observer
    assert events == [
        ("deep_level_planned", {
            "query": "root", "depth": 1, "breadth": 1,
            "queries": [{"query": "branch", "research_goal": "goal"}],
        }),
        ("deep_branch_completed", {
            "query": "branch", "research_goal": "goal",
            "learnings": ["a learning"], "follow_up_questions": ["next?"],
        }),
    ]


@pytest.mark.asyncio
async def test_subtopic_generation_reports_the_subtopics():
    from gpt_researcher.skills.writer import ReportGenerator
    from gpt_researcher.utils.validators import Subtopic, Subtopics

    events, observer = _recording_observer()
    generator = ReportGenerator.__new__(ReportGenerator)
    generator.researcher = SimpleNamespace(
        verbose=False, websocket=None, query="topic", context=[], cfg=None,
        subtopics=[], prompt_family=None, kwargs={}, research_observer=observer,
    )
    result = Subtopics(subtopics=[Subtopic(task="one"), Subtopic(task="two")])

    with patch("gpt_researcher.skills.writer.construct_subtopics",
               AsyncMock(return_value=result)):
        assert await generator.get_subtopics() is result

    assert events == [("subtopics_generated", {"query": "topic", "subtopics": ["one", "two"]})]
