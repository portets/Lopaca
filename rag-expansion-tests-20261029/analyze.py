"""Offline expansion diagnostics using the recorded requests, with no app edits."""

import hashlib
import json
import sqlite3
import statistics
from collections import defaultdict
from contextlib import closing

from benchmark import APP, DOCUMENTS, ROOT, SETTINGS, grade, rag


def analyze():
    rows = [json.loads(line) for line in (ROOT / "results.jsonl").read_text().splitlines()]
    queries = {item["id"]: item for item in json.loads((ROOT / "queries.json").read_text())}
    ablations = []
    failure_diagnostics = []
    fallback_checks = []
    with closing(sqlite3.connect(DOCUMENTS.parent / "documents.sqlite")) as connection:
        for row in rows:
            query = queries[row["id"]]
            effective = rag._retrieval_query(query["query"], query["previous"])
            original_terms = set(rag._terms(effective))
            # Diagnostic only: prevent copied query terms from being counted as new hints.
            novel = " ".join(
                term for term in rag._terms(row["expansion"]["text"]) if term not in original_terms
            )
            result = grade(query, rag._rank(connection, effective, novel, SETTINGS))
            ablations.append(
                {
                    "id": row["id"],
                    "category": row["category"],
                    "novel_expansion": novel,
                    "result": result,
                }
            )
            if row["facts"] and not row["on"]["complete"]:
                terms = rag._terms(effective)
                extra = rag._terms(row["expansion"]["text"])
                total = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
                frequencies = {
                    term: connection.execute(
                        "SELECT count(*) FROM chunks WHERE chunks MATCH ?", (rag._quote(term),)
                    ).fetchone()[0]
                    for term in dict.fromkeys([*terms, *extra])
                }
                paths = [item[0] for item in connection.execute("SELECT path FROM files")]
                sources, anchors = rag._source_scope(
                    connection, paths, terms, frequencies, total, effective
                )
                pool = []
                if sources is not None:
                    for words in (terms, extra):
                        pool.extend(
                            rag._search(
                                connection,
                                " OR ".join(map(rag._quote, words)),
                                SETTINGS.rag_candidates,
                                sources,
                            )
                        )
                expected = {source for source, _needle in row["facts"]}
                failure_diagnostics.append(
                    {
                        "id": row["id"],
                        "query": row["query"],
                        "expansion": row["expansion"]["text"],
                        "inferred_scope": sources,
                        "anchor_terms": sorted(anchors),
                        "expected_sources": sorted(expected),
                        "expected_sources_excluded_by_scope": sorted(expected - set(sources))
                        if sources
                        else [],
                        "expected_sources_in_candidate_pool": sorted(
                            expected & {item[1] for item in pool}
                        ),
                    }
                )

    def timeout(_text):
        raise TimeoutError("Deliberate benchmark expansion failure")

    for category in sorted({row["category"] for row in rows}):
        row = next(row for row in rows if row["category"] == category)
        query = queries[row["id"]]
        off = rag.retrieve(query["query"], DOCUMENTS, SETTINGS, previous_queries=query["previous"])
        timed_out = rag.retrieve(
            query["query"], DOCUMENTS, SETTINGS, expand=timeout, previous_queries=query["previous"]
        )
        empty = rag.retrieve(
            query["query"],
            DOCUMENTS,
            SETTINGS,
            expand=lambda _query: "",
            previous_queries=query["previous"],
        )
        fallback_checks.append(
            {
                "id": query["id"],
                "timeout_equals_off": off == timed_out,
                "empty_equals_off": off == empty,
            }
        )
    summary = {}
    by_category = defaultdict(list)
    for item in ablations:
        by_category[item["category"]].append(item)
        by_category["all"].append(item)
    for category, group in by_category.items():
        positives = [item for item in group if queries[item["id"]]["facts"]]
        negatives = [item for item in group if not queries[item["id"]]["facts"]]
        summary[category] = {
            "answerable": len(positives),
            "complete": sum(item["result"]["complete"] for item in positives),
            "top1": sum(item["result"]["top1"] for item in positives),
            "missing_answer": len(negatives),
            "false_context": sum(item["result"]["false_context"] for item in negatives),
        }
    comparisons = []
    repeat_path = ROOT / "repeats.jsonl"
    if repeat_path.exists():
        originals = {row["id"]: row for row in rows}
        for line in repeat_path.read_text().splitlines():
            repeat = json.loads(line)
            original = originals[repeat["id"]]
            comparisons.append(
                {
                    "id": repeat["id"],
                    "identical_text": original["expansion"]["text"] == repeat.get("expansion"),
                    "same_passages": original["on"]["passages"]
                    == repeat.get("result", {}).get("passages"),
                    "same_complete": original["on"]["complete"]
                    == repeat.get("result", {}).get("complete"),
                    "same_false_context": original["on"]["false_context"]
                    == repeat.get("result", {}).get("false_context"),
                    "error": repeat.get("error"),
                }
            )
    hashes = json.loads((ROOT / "manifest.json").read_text())["app_sha256"]
    unchanged = {
        name: hashlib.sha256((APP / name).read_bytes()).hexdigest() == digest
        for name, digest in hashes.items()
    }
    latencies = sorted(row["expansion"]["seconds"] for row in rows)
    diagnostics = {
        "novel_terms_ablation": summary,
        "failure_diagnostics": failure_diagnostics,
        "fallback_checks": fallback_checks,
        "repeat_comparisons": comparisons,
        "app_files_unchanged": unchanged,
        "expansion_latency_seconds": {
            "min": min(latencies),
            "median": statistics.median(latencies),
            "p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
            "max": max(latencies),
            "total": sum(latencies),
        },
    }
    (ROOT / "ablations.json").write_text(json.dumps(ablations, indent=2))
    (ROOT / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    analyze()
