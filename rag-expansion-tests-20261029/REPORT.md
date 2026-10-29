# Lopaca query-expansion evaluation — October 29, 2026

## Conclusion

Expansion substantially improved retrieval for paraphrased questions in this test,
at a median cost of **3.23 seconds per question**. It also substantially increased
irrelevant context when the requested answer was absent. It is useful, but the
current implementation is not reliably neutral when it cannot help.

**No application code, settings, real chat data, or real documents were changed.**
SHA-256 checks confirmed all six application Python files remained unchanged.
Only this dedicated test directory was added. Preparation, testing, and analysis
took approximately 14 minutes, within the requested 45-minute limit.

## Method

- Server: `http://127.0.0.1:9931`, reporting
  `models/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf`.
- Used the real `LlamaClient.expand_query` and `rag.retrieve` implementations.
  Expansion was enabled by passing the same callback the controller passes when
  the setting is on. The saved defaults were not edited.
- The client overrides launch defaults for expansion: temperature 0, at most 96
  generated tokens, `enable_thinking=False`, and a ten-second network timeout.
  These requests test the client's short expansion path rather than reproduce
  the main chat's reasoning configuration. No end-to-end chat answers were scored.
- Fixed synthetic corpus: **82 files, 405 indexed chunks**, including technical
  notes, service manuals, Python functions, ambiguous terms, repetitive archived
  distractors, and copies of Lopaca's six Python files.
- **120 questions: 96 answerable and 24 missing-answer questions.** Corpus,
  questions, and expected facts were fixed before the main model requests.
  Request order used a fixed random seed. No production algorithm tuning occurred.
- Compared plain SQLite FTS5 BM25 with the same chunk index and field weights,
  current Lopaca with expansion off, and current Lopaca with expansion on.
  All methods were limited to five passages and the same final context budget.
- Scored passages after `format_context`, not before context-budget truncation.
  A fact counted only if its expected text occurred in the expected source.
  “Complete” requires every expected fact; “any hit” requires at least one.
- Repeated a predetermined two questions from each of the twelve categories:
  **24 additional model requests**. These were selected by ID, not by outcome.

## Matched results

| Measure | Plain BM25 | Expansion off | Expansion on |
|---|---:|---:|---:|
| Complete evidence, 96 answerable | 69/96 (71.9%) | 65/96 (67.7%) | **89/96 (92.7%)** |
| At least one expected fact | 77/96 | 65/96 | **89/96** |
| First passage contains an expected fact | 67/96 | 65/96 | **81/96** |
| Missing-answer questions receiving context, lower is better | 12/24 | **2/24** | 15/24 |
| Median retrieval time, including expansion when enabled | 0.36 ms | 2.26 ms | 3,238 ms |

Expansion rescued **24** answerable questions and lost **zero** complete
retrievals relative to expansion off: **25 percentage points** more complete
retrieval. It improved first-passage hits for 16 questions and worsened none.
However, it introduced irrelevant context for **13 additional missing-answer
questions**. Irrelevant context is not the same as a hallucinated final answer;
final-answer behavior was not tested.

The new set also demonstrates that expansion-off Lopaca is **not universally
better than plain BM25**: it retrieved fewer complete answers here while being
much more selective about returning context. The earlier favorable test results
should not be generalized to every corpus or query mix.

### By question type

| Question type | Count | BM25 complete | Off complete | On complete |
|---|---:|---:|---:|---:|
| Recreated earlier vocabulary-gap examples | 8 | 2 | 1 | **7** |
| New vocabulary-gap questions | 16 | 3 | 0 | **10** |
| Direct technical terminology | 24 | 24 | 24 | 24 |
| Exact code identifiers | 12 | 12 | 12 | 12 |
| Explicit source, direct wording | 8 | 8 | 8 | 8 |
| Explicit source, paraphrased wording | 8 | 8 | 0 | **8** |
| Multiple facts across sections | 8 | 0 | 8 | 8 |
| Ambiguity and quoted phrases | 8 | 8 | 8 | 8 |
| Document follow-ups | 4 | 4 | 4 | 4 |

Examples of vocabulary bridges included password guessing → account lockout,
unreachable storage → garbage collection, and sharing memory until mutation →
copy-on-write. Source-specific “activity trail” questions found “audit records”
after expansion. Direct terminology and exact identifiers gained no additional
complete answers, but still incurred the model request latency.

### Missing answers

| Missing-answer type | Count | BM25 returned context | Off | On |
|---|---:|---:|---:|---:|
| Known service, undocumented subscription price | 8 | 8 | 0 | 8 |
| Explicit nonexistent source file | 8 | 0 | 0 | 0 |
| Topic or requested fact not documented | 8 | 4 | 2 | 7 |

## Why expansion sometimes failed or added noise

The saved failure diagnostics distinguish several causes:

1. **Repeated original terms count as expansion evidence.** For a Cobalt price
   question, the model repeats “Cobalt” in its suggested terms. That rare project
   name lets the Cobalt manual pass the relevance gate even though it contains no
   price. All eight undocumented service-price questions exhibited this problem.
2. **Inferred source hints can become incorrect hard filters.** Words such as
   “storage,” “request,” “time,” and even “other” restricted retrieval to unrelated
   filename/title matches. This excluded compaction, backoff, pagination, and
   collaborative-editing documents despite useful expanded terms. Explicit
   filenames are a stronger constraint than these inferred hints.
3. **Ranking can bury correct expanded matches.** The sandboxing and fair-queuing
   documents entered the candidate pool but lost to passages with stronger
   original-word overlap. Expansion cannot help if the final selection drops its
   useful candidates.
4. **Expansion can choose the wrong interpretation.** The consistent-hashing
   question produced migration/replication terms rather than a useful bridge to
   consistent hashing. No relevant candidate entered the pool.

### Small offline diagnostic — not applied to Lopaca

Using the already recorded model outputs, I removed terms already present in the
original query before passing expansion to the ranker. This required no further
model calls and made no application edits.

| Measure | Current expansion | Novel terms only, diagnostic |
|---|---:|---:|
| Complete evidence | 89/96 | 89/96 |
| First-passage hits | 81/96 | 81/96 |
| Missing-answer questions receiving context | 15/24 | **7/24** |

This isolates a concrete small improvement worth considering next. It does not
solve semantic drift or incorrect inferred source restrictions. It was designed
after inspecting failures, so it is exploratory evidence rather than an untouched
validation result. No indexing-time synonym generation was added or tested.

## Timing, stability, and fallbacks

- Expansion-only latency: minimum **2.08 s**, median **3.23 s**, 95th percentile
  **4.09 s**, maximum **4.74 s**; approximately 397 seconds total for the main run.
- **0/120** expansion errors/timeouts and **0/120** empty expansions.
- All **24/24 repeats** produced identical expansion text, passages, and outcomes.
  This demonstrates stability on this subset, not all models or server loads.
- Controlled timeout exceptions and empty callbacks returned exactly the same
  results as expansion off for one question in every category: **24/24 checks**.
  These were injected failures, not observed real ten-second network timeouts.

## Comparison with the earlier report

The original temporary fixtures and raw expansions did not survive the reboot.
Your saved `REPORT.md`, copied here as `previous-report.md`, records earlier
expansion results of **3/8** for the older pipeline and **5/8** after replay through
the revised ranker. Today's recreated eight-example category scored **7/8** with
fresh expansion and **1/8** without it.

Those numbers are directionally encouraging, but **are not an exact rerun**:
the corpus, some reconstructed wording, distractors, and current model differ.
The trustworthy causal comparison is today's matched off/on run. Differences
cannot be attributed specifically to model size or quantization.

## Limits and practical interpretation

These are synthetic retrieval tests, with repeated service templates and related
question pairs. Their proportions deliberately stress paraphrases and missing
answers; they do not estimate performance over your normal chat workload. The
scores measure expected evidence retrieval, not the correctness of generated
answers. No embedding system was measured, so they do not establish equivalence
to vector-based RAG.

Expansion is most useful when the question and source use different terminology.
It currently adds little retrieval benefit to already precise technical queries,
and needs better relevance gating before it can be described as generally safe
from additional irrelevant context. The repeated-term filter and softer inferred
source hints are more concrete next steps than adding another model feature.

## Saved artifacts

- `queries.json`, `corpus/documents/`, `corpus/documents.sqlite`: fixed fixtures.
- `results.jsonl`: every question, expansion, timing, returned passage, and score.
- `summary.json`: category and overall matched results.
- `repeats.jsonl`: the 24 fresh repeat requests and their results.
- `ablations.json`, `diagnostics.json`: offline novel-term comparison, failed-query
  candidate/source diagnostics, fallbacks, stability, and unchanged-file hashes.
- `manifest.json`, `app-snapshot/`: defaults and exact tested application source.
- `props.json`, `v1-models.json`, `health.json`: server metadata.
- `benchmark.py`, `analyze.py`: standard-library-only test harness and analysis.

The main benchmark script resumes recorded IDs instead of reissuing completed
requests. The test folder can be deleted without affecting Lopaca or real chats.
