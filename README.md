# GPT Researcher Evaluation Infrastructure

This repository is an evaluation-focused fork of [assafelovic/gpt-researcher](https://github.com/assafelovic/gpt-researcher).

GPT Researcher is an open-source autonomous research agent for web and local research. It gathers sources, synthesizes context, and generates long-form reports with citations. This fork keeps the original project as the research backbone and focuses on building evaluation infrastructure around the reports produced by the agent.

<div align="center">
<img align="center" height="600" src="https://github.com/assafelovic/gpt-researcher/assets/13554167/4ac896fd-63ab-4b77-9688-ff62aafcc527">
</div>

The original upstream README is preserved at [README-upstream.md](README-upstream.md).

## My Work

**My work is contributed to eval infrastructure: factual evaluation, report-quality metrics, standardized metric execution, perturbation validation, unit tests, and lightweight benchmarking.**

### 1. Simple Factual Evaluation

Based on the OpenAI SimpleQA evaluation idea, `simple_evals` measures factual accuracy with ground-truth answers.

- Evaluates factual correctness for short-form QA tasks.
- Tracks accuracy, F1, answer rate, cost, latency, and source coverage.
- Writes structured JSONL logs so later eval runs can be compared consistently.
- Provides the foundation for repeatable factual evaluation before moving into broader report-quality evaluation.

### 2. Standardized Evaluation Framework

The quality-eval layer follows the structure of DeepEval / RAGAS-style metric systems, while adapting it to autonomous research agents.

- Uses schema-like abstractions: `EvalSample`, `BaseMetric`, and `MetricResult`.
- Standardizes metric outputs with `score`, `group`, `aligned_with`, `reason`, `breakdown`, and `skipped`.
- Keeps all primary scores as **higher is better**, making aggregation and comparison easier.
- Separates raw metric logic from metric wrappers, so metrics can be unit-tested directly and also run through a consistent suite interface.

### 3. Report-Quality Metrics

The metrics are organized into four groups so the evaluation is easy to understand at both a high level and a technical level.

- **Faithfulness**
  - `unsupported_claim`: extracts report claims and classifies each one as `supported`, `inferred`, or `unsupported` against source context.
  - `hallucination`: optional binary check that compares report content against scraped source text.
- **Answer Relevancy**
  - `subtopic_coverage`: uses an LLM judge to check whether the report covers the expected subtopics for the query.
- **Source Quality**
  - `source_diversity`: measures domain variety using unique-domain ratio and Shannon entropy.
  - `source_authority`: scores source reliability with high-confidence rules and optional LLM scoring for unknown domains.
- **Citation**
  - `citation_faithfulness`: checks whether sources listed in `## References` are actually cited in the report body, catching decorative or unused references.

### 4. Unit Tests for Metrics

`tests/test_quality_metrics.py` validates the metric layer with focused tests.

- Covers empty inputs, boundary cases, citation parsing, source-diversity math, and source-authority rule ordering.
- Mocks LLM responses for subtopic coverage and unsupported-claim evaluation.
- Tests bad JSON, LLM failure fallback, skipped metrics, and standardized metric wrapper behavior.
- Includes regression coverage for domain normalization, including the `www.` prefix handling bug.

### 5. Perturbation Testing for Metric Reliability

`evals/quality_eval/perturbation.py` validates whether metrics move in the expected direction when report quality is deliberately degraded.

- Removes inline citations to verify `citation_faithfulness` drops.
- Replaces authoritative sources with weaker sources to verify `source_authority` drops.
- Collapses sources onto a single domain to verify `source_diversity` drops.
- Removes report sections to verify `subtopic_coverage` drops.
- Corrupts factual claims to verify unsupported-claim detection becomes stricter.

**This checks metric reliability, not just implementation correctness: if the input gets worse, the corresponding metric should respond predictably.**

### 6. Lightweight Benchmarking

`evals/quality_eval/benchmark.py` compares different researcher configurations across a fixed topic set.

- Compares single-agent and multi-agent research flows.
- Compares model choices such as `gpt-4o` and `gpt-4o-mini`.
- Saves structured logs for later trend analysis.
- Supports summary replay and run-to-run comparison.

## File Structure

| Path | Purpose |
| --- | --- |
| `README.md` | Main README for this evaluation-focused fork. |
| `README-upstream.md` | Preserved original GPT Researcher README for upstream reference and future PR work. |
| `evals/README.md` | Overview of the three evaluation systems: simple evals, quality evals, and hallucination evals. |
| `evals/simple_evals/simpleqa_eval.py` | SimpleQA-style grading logic for factual correctness. |
| `evals/simple_evals/run_eval.py` | Runs factual QA evaluation and records accuracy, F1, cost, latency, answer rate, and source coverage. |
| `evals/simple_evals/example_output.json` | Example structured output schema for SimpleQA-style logs. |
| `evals/simple_evals/logs/` | Versioned evaluation histories in text and JSONL formats. |
| `evals/simple_evals/problems/` | Ground-truth factual QA dataset for the simple evaluation runner. |
| `evals/quality_eval/README.md` | Detailed documentation for the report-quality evaluation framework. |
| `evals/quality_eval/base.py` | Evaluation abstractions: `EvalSample`, `BaseMetric`, `MetricResult`, and metric groups. |
| `evals/quality_eval/suite.py` | Standardized metric wrappers and the `evaluate()` entry point. |
| `evals/quality_eval/metrics.py` | Core implementations for citation, source quality, subtopic coverage, and unsupported-claim metrics. |
| `evals/quality_eval/run_eval.py` | Runs GPT Researcher reports through the quality metric suite and writes structured logs. |
| `evals/quality_eval/benchmark.py` | Compares different agent/model configurations across a fixed benchmark topic set. |
| `evals/quality_eval/perturbation.py` | Degrades reports and sources to test whether metrics respond in the expected direction. |
| `evals/quality_eval/fixtures/` | Stored report fixtures used by perturbation reliability checks. |
| `evals/quality_eval/logs/` | Structured quality-eval logs for experiment comparison and trend analysis. |
| `evals/quality_eval/requirements.txt` | Additional dependencies for quality evaluation. |
| `evals/hallucination_eval/evaluate.py` | Hallucination evaluator used by the optional hallucination metric. |
| `evals/hallucination_eval/inputs/` | Query inputs used by hallucination and quality-eval runners. |
| `tests/test_quality_metrics.py` | Unit tests for quality metrics, metric wrappers, error handling, and regression cases. |

## Common Commands

```bash
# Run SimpleQA-style factual evaluation
python -m evals.simple_evals.run_eval --num_examples 10

# Run zero-cost report-quality metrics
python -m evals.quality_eval.run_eval --num_examples 10 --no-subtopic --no-hallucination --no-unsupported-claim

# Run full report-quality evaluation
python -m evals.quality_eval.run_eval --num_examples 5

# Run perturbation reliability checks
python -m evals.quality_eval.perturbation

# Run unit tests
python -m pytest tests/test_quality_metrics.py
```
