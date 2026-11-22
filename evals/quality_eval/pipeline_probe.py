"""
Pipeline Probe — per-stage instrumentation for the research pipeline.
--------------------------------------------------------------------
Every metric in this suite scores the pipeline's OUTPUT, so a low score says a
report is weak without saying which stage weakened it. This module records what
each stage was handed and what it passed on, turning one end-to-end number into
a funnel that can be read stage by stage.

    plan        query           -> sub-queries
    search      sub-queries     -> URLs
    scrape      URLs            -> pages          (anti-bot defences bite here)
    compress    pages           -> chunks         (similarity filter, then top-N)
    write       context         -> report

The recording logic (`PipelineTrace`) is pure and separately testable.
`PipelineProbe` feeds it from the library's `research_observer` hook
(gpt_researcher/observability.py): it is passed to GPTResearcher like any
other observer, so nothing in the library is patched and a core refactor
that keeps the documented events keeps the probe working.

The observer reports planning, search, de-duplication and fetching. It does
not report the compression step (similarity filter and top-N cut), so the
compression fields of the funnel stay None until the hook covers it.

Usage:
    probe = PipelineProbe()
    researcher = GPTResearcher(query=..., research_observer=probe)
    context = await researcher.conduct_research()
    report  = await researcher.write_report()
    print(probe.trace.funnel())
    sources = probe.trace.scraped_urls      # what the writer could actually read
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Recording  (pure — no imports from gpt_researcher, unit-testable offline)
# ---------------------------------------------------------------------------

@dataclass
class PipelineTrace:
    """What each stage received and produced during one research run."""

    subqueries:   list[str]  = field(default_factory=list)
    searches:     list[dict] = field(default_factory=list)
    scrapes:      list[dict] = field(default_factory=list)
    compressions: list[dict] = field(default_factory=list)
    context_len:  int = 0

    # --- recording -------------------------------------------------------

    def record_search(self, proposed: list[str], accepted: list[str]) -> None:
        """One search's results, before and after de-duplication.

        The pipeline drops URLs it has already seen before anything is scraped,
        so every later stage sees disjoint sets by construction — measuring
        overlap downstream can only ever return zero. This is the one place the
        redundancy is still visible: a sub-query whose results were all seen
        already spent a search slot and returned nothing new.
        """
        self.searches.append({
            "proposed": len(proposed or []),
            "accepted": len(accepted or []),
            "duplicate": len(proposed or []) - len(accepted or []),
        })

    def record_scrape(self, urls_in: list[str], pages_out: list[dict]) -> None:
        """One scrape batch: which URLs were attempted, which came back."""
        got = [p.get("url") or p.get("source") or "" for p in pages_out]
        self.scrapes.append({
            "urls_in":   list(urls_in or []),
            "urls_out":  [u for u in got if u],
            "chars_out": sum(len(str(p.get("raw_content", "") or "")) for p in pages_out or []),
        })

    def record_compression(self, subquery: str, pages_in: list[dict],
                           chars_out: int, *, docs_after_filter: int | None = None,
                           top_n: int | None = None) -> None:
        """One compression call: the corpus in, the context out.

        `docs_after_filter` / `top_n` describe the second, less visible cut:
        the similarity filter decides which chunks qualify, then only the first
        `top_n` of them survive. When more chunks qualify than top_n admits, the
        count cap is what is actually binding, not the similarity threshold.
        """
        chars_in = sum(len(str(p.get("raw_content", "") or "")) for p in pages_in or [])
        threshold = int(os.environ.get("COMPRESSION_THRESHOLD", "8000"))
        self.compressions.append({
            "subquery":          subquery,
            "pages_in":          len(pages_in or []),
            "chars_in":          chars_in,
            "chars_out":         chars_out,
            "docs_after_filter": docs_after_filter,
            "top_n":             top_n,
            # Mirrors ContextCompressor.async_get_context's fast-path predicate.
            "fast_path":         chars_in < threshold and len(pages_in or []) <= 10,
        })

    # --- derived ---------------------------------------------------------

    @property
    def attempted_urls(self) -> list[str]:
        return [u for s in self.scrapes for u in s["urls_in"]]

    @property
    def scraped_urls(self) -> list[str]:
        """URLs that actually produced content.

        This is the honest denominator for source-quality metrics. `visited_urls`
        counts everything the retriever proposed, including pages that never
        loaded, so scoring it credits the pipeline for sources the writer never
        read — and since the strictest bot defences sit in front of the most
        authoritative publishers, that credit is not evenly distributed.
        """
        seen, out = set(), []
        for s in self.scrapes:
            for u in s["urls_out"]:
                if u not in seen:
                    seen.add(u)
                    out.append(u)
        return out

    @property
    def failed_urls(self) -> list[str]:
        got = set(self.scraped_urls)
        seen, out = set(), []
        for u in self.attempted_urls:
            if u not in got and u not in seen:
                seen.add(u)
                out.append(u)
        return out

    def funnel(self) -> dict:
        """Stage-by-stage volumes and retention rates."""
        attempted = len(self.attempted_urls)
        scraped   = len(self.scraped_urls)
        proposed  = sum(s["proposed"] for s in self.searches)
        duplicate = sum(s["duplicate"] for s in self.searches)
        chars_in  = sum(c["chars_in"] for c in self.compressions)
        chars_out = sum(c["chars_out"] for c in self.compressions)
        capped    = [c for c in self.compressions
                     if c.get("docs_after_filter") is not None
                     and c.get("top_n") is not None
                     and c["docs_after_filter"] > c["top_n"]]
        return {
            "subqueries":        len(self.subqueries),
            "urls_proposed":     proposed,
            "urls_duplicate":    duplicate,
            # Share of search results already seen by an earlier query — how
            # much of the retrieval budget the planner spent on ground it had
            # already covered.
            "search_overlap":    round(duplicate / proposed, 3) if proposed else None,
            "urls_attempted":    attempted,
            "urls_scraped":      scraped,
            "scrape_rate":       round(scraped / attempted, 3) if attempted else None,
            "scraped_chars":     chars_in,
            "context_chars":     chars_out,
            "compression_rate":  round(chars_out / chars_in, 3) if chars_in else None,
            "compress_calls":    len(self.compressions),
            "fast_path_calls":   sum(c["fast_path"] for c in self.compressions),
            # How often the top-N cap, not the similarity threshold, did the cutting
            "top_n_capped_calls": len(capped),
        }


def sources_to_score(trace: PipelineTrace, visited: list[str]) -> tuple[list[str], str]:
    """The URLs source-quality metrics should score, and which set they are.

    Returns (urls, basis). basis is "scraped" when the probe saw the scrape
    stage: the pages that loaded, which may be none if every scrape failed.
    It is "visited" only when the probe recorded no scrape at all, meaning
    its hooks did not fire, and the caller falls back to everything the
    retriever proposed. That fallback overstates authority (see
    `scraped_urls`), so it is labelled in the log rather than silent.
    """
    if trace.scrapes:
        return trace.scraped_urls, "scraped"
    if visited:
        print("      [WARN] pipeline probe recorded no scrapes; scoring all visited "
              "URLs, including pages that may never have loaded")
    return list(visited), "visited"


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------

class PipelineProbe:
    """A `research_observer` that records one run into a PipelineTrace.

    Pass it as `GPTResearcher(..., research_observer=probe)`. One probe may be
    shared by several researchers (the multi-agent workflow runs one per
    section); their events land in the same trace.
    """

    def __init__(self) -> None:
        self.trace = PipelineTrace()
        # Per search query: URLs offered for fetching, and how many were new.
        self._searches: dict[str, dict] = {}
        # Per fetch batch the observer gives no grouping, so fetches are kept
        # as one running batch and flushed into the trace when read.
        self._attempted: list[str] = []
        self._loaded: list[dict] = []

    def on_event(self, event: str, **f) -> None:
        if event == "query_planned":
            self.trace.subqueries.append(f.get("query", ""))
        elif event in ("selected_for_fetch", "candidate_deduplicated"):
            s = self._searches.setdefault(f.get("query", ""), {"proposed": [], "accepted": []})
            s["proposed"].append(f.get("url", ""))
            if event == "selected_for_fetch":
                s["accepted"].append(f.get("url", ""))
            self._sync()
        elif event in ("fetch_succeeded", "fetch_failed"):
            url = f.get("url") or ""
            if url:
                self._attempted.append(url)
            if event == "fetch_succeeded":
                self._loaded.append({"url": url, "content_length": f.get("content_length", 0)})
            self._sync()
        elif event == "content_prefetched":
            # The retriever supplied the page itself: it loaded without a scrape.
            url = f.get("url") or ""
            self._attempted.append(url)
            self._loaded.append({"url": url, "content_length": f.get("content_length", 0)})
            self._sync()

    def _sync(self) -> None:
        self.trace.searches = []
        for s in self._searches.values():
            self.trace.record_search(s["proposed"], s["accepted"])
        self.trace.scrapes = []
        if self._attempted or self._loaded:
            self.trace.scrapes.append({
                "urls_in": list(self._attempted),
                "urls_out": [p["url"] for p in self._loaded if p["url"]],
                "chars_out": sum(int(p["content_length"] or 0) for p in self._loaded),
            })
