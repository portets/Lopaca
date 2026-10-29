"""Compare temporary retrieval candidates against the frozen expansion benchmark."""

import importlib.util
import json
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from benchmark import APP, DOCUMENTS, SETTINGS, grade


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidates():
    original = (ROOT.parent / "app-snapshot" / "rag.py").read_text()
    (ROOT / "before.py").write_text(original)
    deduplicated = original.replace(
        "terms, extra = _terms(query), _terms(expansion)",
        "terms = _terms(query)\n    extra = [term for term in _terms(expansion) if term not in terms]",
    )
    start = deduplicated.index("    # A named symbol")
    end = deduplicated.index("\n\ndef _document_context", start)
    explicit_only = deduplicated[:start] + "    return (), set()\n" + deduplicated[end:]
    filename_evidence = explicit_only.replace(
        "set(terms)\n            & {name for path in explicit for name in (path.casefold(), Path(path).name.casefold())}",
        "set()",
    )
    versions = {"before": original, "deduplicated": deduplicated,
                "explicit_only": explicit_only, "filename_evidence": filename_evidence}
    soft_hints = deduplicated.replace(
        "return (tuple(sorted(sources)), set(hints)) if len(sources) <= 64 else ((), set())",
        "return (), set(hints)",
    )
    named_files = soft_hints.replace(
        '    filenames = {name for path in paths for name in (path.casefold(), Path(path).name.casefold())}\n',
        '',
    ).replace(' and term.casefold() not in filenames', '')
    versions.update({"soft_hints": soft_hints, "named_files": named_files})
    for name, source in list(versions.items())[2:]:
        for threshold in (0.4, 0.5):
            balanced = source.replace(
                "best_rank = max(ranks.values())",
                "expansion_weight = sum(weights[term] for term in extra)\n    best_rank = max(ranks.values())",
            ).replace(
                "        expansion_hit = any(",
                "        expansion_coverage = sum(weights[term] for term in matches[rowid] & set(extra)) / max(1, expansion_weight)\n        expansion_hit = expansion_coverage >= 0.2 and any(",
            ).replace(
                "score = coverage + 0.15", "score = max(coverage, expansion_coverage) + 0.15"
            ).replace("source_coverage < 0.5", f"source_coverage < {threshold}")
            versions[f"{name}_balanced_{threshold}"] = balanced
    for name, source in versions.items():
        path = ROOT / f"{name}.py"
        path.write_text(source)
        yield name, load(path)


def evaluate(versions):
    queries = {q["id"]: q for q in json.loads((ROOT.parent / "queries.json").read_text())}
    recorded = [json.loads(line) for line in (ROOT.parent / "results.jsonl").read_text().splitlines()]
    summary, details = {}, []
    with closing(sqlite3.connect(DOCUMENTS.parent / "documents.sqlite")) as connection:
        for name, module in versions:
            result = {}
            for mode in ("off", "on"):
                counts = {"complete": 0, "positive": 0, "top1": 0, "false_context": 0,
                          "negative": 0, "regressions": [], "rescues": [], "elapsed_ms": 0}
                start = time.perf_counter()
                for row in recorded:
                    query = queries[row["id"]]
                    effective = module._retrieval_query(query["query"], query["previous"])
                    found = module._rank(connection, effective, row["expansion"]["text"] if mode == "on" else "", SETTINGS)
                    outcome = grade(query, found)
                    counts["positive"] += bool(query["facts"])
                    counts["negative"] += not query["facts"]
                    counts["complete"] += bool(outcome["complete"])
                    counts["top1"] += bool(outcome["top1"])
                    counts["false_context"] += bool(outcome["false_context"])
                    if row[mode]["complete"] and not outcome["complete"]:
                        counts["regressions"].append(row["id"])
                    if outcome["complete"] and not row[mode]["complete"]:
                        counts["rescues"].append(row["id"])
                    details.append({"variant": name, "mode": mode, "id": row["id"], "result": outcome})
                counts["elapsed_ms"] = (time.perf_counter() - start) * 1000
                result[mode] = counts
            summary[name] = result
            print(name, json.dumps(result), flush=True)
    (ROOT / "comparison.json").write_text(json.dumps(summary, indent=2))
    (ROOT / "details.json").write_text(json.dumps(details, indent=2))


if __name__ == "__main__":
    evaluate(candidates())
