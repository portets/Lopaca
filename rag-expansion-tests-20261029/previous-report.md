# Lopaca lexical RAG improvement report

Only rag.py was changed. No external dependencies, embeddings, models, index schema changes, UI changes, or modifications to real chat documents were introduced. Existing indexes can be reused. Expansion remains optional and disabled by default.

## Changes

- Replace redundant automatic adjacent-phrase and metadata ranking votes with base BM25, explicitly quoted phrases, and optional expanded vocabulary.
- Use corpus term frequencies, original-query coverage, exact technical identifiers, and filename/title hints to rank and reject weak matches.
- Treat recognized technical identifiers as stronger evidence than inferred filename hints.
- Recover bounded whole-document context for small documents, or whole-section/function context when it fits. Avoid duplicate overlapping excerpts and prefer complementary evidence.
- Preserve useful expanded vocabulary during result selection.

## Retrieval evaluation (expansion disabled)

“Complete” means all expected source facts appeared among returned passages, not that the model generated a correct answer. Plain BM25 uses the same index and field weights.

| Suite | Plain BM25 complete | Previous Lopaca complete | Revised complete | BM25 unsupported contexts | Previous unsupported | Revised unsupported |
|---|---:|---:|---:|---:|---:|---:|
| Original benchmark: 140 answerable, 38 missing-answer | 117/140 | 117/140 | 131/140 | 35/38 | 31/38 | 19/38 |
| Additional diagnostic: 41 answerable, 21 missing-answer | 40/41 | 35/41 | 40/41 | 19/21 | 7/21 | 8/21 |
| Additional validation: 25 answerable, 12 missing-answer | 25/25 | 25/25 | 25/25 | 11/12 | 10/12 | 5/12 |

Original-suite top-one retrieval improved from 116/140 to 129/140. Median ranking time fell from about 4.19 ms to 1.82 ms; plain BM25 was about 0.21 ms. Multi-section complete retrieval improved from 0/14 to 14/14. Bounded parent context alone did not improve complete retrieval over BM25 on these suites.

The fixtures are synthetic and contain related question variants. Failures in both new sets informed development, so they are diagnostic/validation results, not untouched holdouts or a general accuracy estimate. Missing-answer detection remains imperfect; the new system returned one more unsupported context than the old one on the diagnostic suite. Retrieval accuracy is not answer accuracy, and no embedding system was evaluated.

## Other checks

All passed: current content after same-size edits, deleted-source removal from results and index, unchanged files not reread, only changed file reread, cancellation, context budget, and worker-thread use. Ruff checks and formatting passed for all six Python files.

Warm retrieval over the prior 500-document (~7 MB, 14,000 chunk) corpus took about 31 ms and found the expected evidence. This is one sample, not a latency distribution. Initial indexing and model answer generation were not timed again.

Replaying eight previously recorded query-expansion outputs recovered 5/8 expected passages, versus 3/8 in the previous benchmark. No fresh llama-server requests were made. Without expansion, arbitrary vocabulary-gap questions remain a limitation.

## Larger option (not implemented)

Use the existing model at indexing time to generate search-only synonyms and likely questions for each document/section. This could bridge vocabulary gaps without embeddings or per-query model latency, but increases indexing time and needs strict separation of generated search hints from authoritative source text. It should be evaluated separately before adoption.

## Artifacts

comparison.json: original and additional diagnostic results and retrieved passages.
validation.json: additional validation results and passages.
smoke-results.json: lifecycle checks and recorded expansion replay.
original.py: pre-change rag.py snapshot.
