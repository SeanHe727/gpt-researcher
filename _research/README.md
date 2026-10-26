# Research working material — not for upstream

Scratch scripts and measurement data behind the `quality_eval` fixes. This
directory exists so the work survives moving between machines; it is **not**
part of any pull request. Keep it off branches you intend to propose upstream.

## Running

Scripts resolve paths relative to the repository root, so run them from there:

```bash
python _research/perturbation_fine.py gpt-5.6-luna
```

Credentials come from `OPENAI_API_KEY` / `TAVILY_API_KEY` if set, otherwise from
the local key files named in each script — set the environment variables on a
new machine and the file paths are ignored.

## What produced which number

| Script | Answers | Output |
|---|---|---|
| `perturbation_fine.py` | The headline pair: Kendall tau-b and response error, 10 reports x 11 levels | `data/perturbation_fine.json` |
| `perturbation_luna.py` | Same, 3 reports x 5 levels (the coarse ladder, kept for contrast) | `data/perturbation_gpt-5.6-luna.json` |
| `response_error.py`, `error_stats.py`, `pct_framings.py` | Response error under three framings; MAE/MSE/RMSE; why the floor-corrected one is the honest reading | `data/response_error_*.json` |
| `truncation_ablation.py` | Whether the unsupported-claim signal was the 4000-char cut (it was) | `data/truncation_ablation_*.json` |
| `verify_subtopic_fix.py` | False-positive floor before/after the references fix: 0.381 -> 0.000 | — |
| `build_fixtures.py` | Rebuilds the 10 fixtures at full context | writes the fixtures file |
| `capture_pipeline.py` | One instrumented run per query, saving every stage | `data/pipeline_capture.json` |
| `analyze_funnel.py` | Per-stage retention; where material is lost | — |
| `scrape_bias.py` | Scrape failure split by source authority (71% vs 22%) | — |
| `e2_threshold_sweep.py` | Whether the similarity threshold binds (it does not at its default) | `data/e2_result.json` |
| `attribute_unsupported.py` | Which layer each unsupported claim belongs to | `data/attribution.json` |
| `overlap_distribution.py` | Retrieval overlap across 20 runs (mean 30.5%) | `data/overlap_distribution.json` |
| `full_panel.py` | All five metrics on stock-config runs | — |

`data/pipeline_capture.json` holds the raw scraped pages from three
instrumented runs. Every offline ablation replays against it rather than
re-scraping, which is why the threshold sweep and the attribution pass cost
cents instead of dollars. Keep it.

`data/eval_findings.html` is the source of the findings report published at
https://claude.ai/code/artifact/21a06b3c-7d0e-4c89-940f-7a217194b2f7

## Findings status against upstream main

Re-checked after v3.7.0 replaced the embeddings compressor with
`select_context` (Jev / BM25 / embeddings):

- **Still valid** — scraper failing 71% of high-authority sources vs 22%
  elsewhere; the hardcoded `max_results=10`, which now binds across all three
  filter modes; the fast path that never fires; 30.5% retrieval overlap; and the
  gap between URLs visited and pages actually read, which #2123 widened by
  recording prefetched URLs too.
- **Needs re-scoping** — "SIMILARITY_THRESHOLD is inert at its default" only
  holds in embeddings mode, and embeddings is no longer the default.

All five probe hooks still resolve against the new upstream.
