# Applied retrieval improvements — October 29, 2026

## Changes applied

Only the application file `rag.py` changed: **543 → 545 lines**. No dependencies,
models, settings defaults, index schema, UI, history, or command-approval code
changed. Existing document indexes remain usable without rebuilding them.

1. **Removed guessed source restrictions.** Generic filename/title words such as
   “request,” “time,” and “other” no longer exclude unrelated filenames from the
   search. BM25 still weights source names and sections. Literal filenames remain
   an explicit restriction, and requested nonexistent filenames still abstain.
2. **Honor explicit filenames as exact identifiers.** A requested document can be
   retrieved even when the question paraphrases its contents. Previously filenames
   were excluded from identifier evidence and some valid requests returned nothing.
3. **Added original-query evidence from document bodies.** Two original terms,
   including a corpus-rare term, can support a wordy question whose overall query
   coverage is low. This uses FTS5's existing tokenizer and stemming, not a new
   synonym list or model. Filename/metadata matches alone do not satisfy this check.
4. **Expansion uses new vocabulary.** Terms copied verbatim from the original query
   no longer count again as expansion evidence.
5. **Require corroboration for expansion matches.** Expanded evidence must cover
   enough of the expanded query and include a rare term, supported by another
   expanded term or by that rare term naming the document section.
6. **Score expanded coverage separately.** Useful expanded matches can compete with
   original-word matches instead of being buried under incidental lexical overlap.

All database connections remain local to their retrieval call. The worker-thread
boundary, cancellation progress handler, incremental indexing, final context
budget, and temporary context injection were preserved.

## Test design

- Replayed the prior **120-question benchmark** against its frozen index and
  recorded expansions; no new model calls were needed for these comparisons.
- Created a **45-question development validation set** with new documents and
  real expansions from the running Qwen3.8-27B Q4 server. One regression in this
  set informed the body-evidence improvement, so this is development evidence.
- After selecting the final candidate, ran **21 additional fixed questions** with
  fresh expansions. Their outcomes were not used to modify the candidate.
- Before and after use identical corpus contents, queries, expansion outputs,
  settings, result limits, and context budgets within each comparison.
- In total: **186 questions evaluated**, including **66 new expansion requests**.
  No fresh expansion request failed. The new validation corpus includes the old
  corpus as distractors, so the suites are not independent samples.

## Complete evidence retrieval

“Complete” requires all expected facts to appear in the correct source excerpts
after final context-budget formatting. This is retrieval quality, not final model
answer accuracy.

| Suite | Answerable | Off before | Off applied | On before | On applied |
|---|---:|---:|---:|---:|---:|
| Prior benchmark | 96 | 65 | **74** | 89 | **95** |
| Development validation | 30 | 16 | **26** | 29 | **30** |
| Final fixed questions | 14 | 6 | **11** | 13 | **14** |

No previously complete retrieval became incomplete in the final implementation
on these three suites. Individual passage ordering and irrelevant context can
still regress; this is not a claim of universal improvement.

Plain BM25 complete-retrieval scores on the same suites were 69/96, 29/30, and
13/14. Expansion-off retrieval therefore remains less complete than BM25 on the
two newer sets, while being more selective about returning context. Expansion-on
retrieval recovered more complete evidence than BM25 in all three sets here.

## Returning context when the requested answer is absent

Lower is better under this strict benchmark measure. A topical or explicitly
requested document is still counted if it lacks the expected answer.

| Suite | Missing-answer questions | Off before | Off applied | On before | On applied |
|---|---:|---:|---:|---:|---:|
| Prior benchmark | 24 | **2** | 3 | 15 | **6** |
| Development validation | 15 | **2** | 5 | 9 | **7** |
| Final fixed questions | 7 | **0** | 3 | 5 | **4** |

The expansion-off tradeoff is intentional and material: the improved recall is
less conservative. All three extra cases in the development validation were
explicitly requested existing documents without the requested fact. In the final
set, one was an explicitly named file and two were documents with strong topical
body matches but no licensing/test-coverage fact. They were not unrelated sources,
but can still distract the model.

The application already instructs the model to say when the requested fact is
absent from current excerpts. Final-answer adherence to that instruction was not
measured here. Retrieval cannot reliably determine arbitrary fact absence from
lexical matches alone; the remaining cases are recorded rather than hidden.

Expansion-on reduced missing-answer context in all three suites while improving
complete retrieval. It still does not eliminate unsupported context.

## Performance and lifecycle

Warm `_rank` timing, 360 samples per condition on the frozen original corpus,
excluding indexing/file scanning and model requests:

| Mode | Median before | Median applied | P95 before | P95 applied |
|---|---:|---:|---:|---:|
| Expansion off | 0.98 ms | 1.30 ms | 3.88 ms | 5.44 ms |
| Expansion on, recorded terms | 1.83 ms | 5.85 ms | 10.54 ms | 13.58 ms |

Ranking costs a few more milliseconds because it checks body evidence and searches
globally rather than sometimes narrowing to a guessed source. The model request
itself is unchanged; previous live expansion latency was approximately 3.23 seconds
median. These timings are local synthetic measurements, not production latency
guarantees.

All **14 checks** passed against the applied application module:

- Initial fact retrieval and freshness after a same-size edit.
- Unchanged documents not reread; only edited document reread.
- Deleted sources absent from both results and index.
- Binary data not indexed.
- Cancellation and safely quoted FTS input.
- Timeout and empty-expansion fallback to ordinary retrieval.
- Context character budget and nonexistent requested-file abstention.
- Eight concurrent calls using independent worker connections.

Ruff lint and formatting checks passed for all six application Python files.
Hashes verified that the other five application files were unchanged.

## Interpretation and artifacts

These synthetic tests contain related questions and shared distractors. Only the
last 21 questions were withheld from candidate tuning, and they reuse validation
documents. No embedding system or final generated answers were evaluated. The
results support these targeted improvements, not embedding equivalence or a
general accuracy percentage. No indexing-time synonym/question generation was
added.

Artifacts in this directory:

- `before.py`: exact pre-change `rag.py`.
- `candidate_final.py`: exact applied `rag.py`.
- `comparison.json`, `details.json`: before/applied replay of the prior benchmark.
- `validation-queries.json`, `validation-results.jsonl`: first fresh development
  run; its initial candidate results are retained as originally recorded.
- `validation-final.json`, `final-summary.json`: final applied-code comparisons
  using both fresh sets and the same saved expansions.
- `holdout-queries.json`, `holdout-results.jsonl`: final fixed-question run.
- `validation/documents/`, `validation/documents.sqlite`: new fixture corpus/index.
- `checks.json`: lifecycle results and repeated timing measurements.
- Other candidate files record discarded alternatives, including a blanket
  restriction removal and stricter expansion gates that lost useful answers.

Only this test directory and `rag.py` were edited. Real chats and documents were
not used or modified. Testing and implementation took approximately 22 minutes.
