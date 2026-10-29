"""Index lifecycle, worker use, fallback, and retrieval timing checks on scratch data."""

import json
import sqlite3
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

from compare import APP, DOCUMENTS, ROOT, SETTINGS, load


def run():
    module = load(APP / "rag.py")
    folder = ROOT / "lifecycle/documents"
    folder.mkdir(parents=True, exist_ok=True)
    document = folder / "dock.md"
    document.write_text("# Dock\n\nThe reconnect delay is 17 seconds.\n")
    query = "In dock.md, what is the reconnect delay?"
    checks = {}
    found = module.retrieve(query, folder, SETTINGS)
    checks["initial_fact"] = any("17 seconds" in body for _, body in found)
    reads = []
    reader = module.read_document

    def read(path):
        reads.append(path.name)
        return reader(path)

    module.read_document = read
    module.retrieve(query, folder, SETTINGS)
    checks["unchanged_files_not_reread"] = not reads
    document.write_text(document.read_text().replace("17 seconds", "29 seconds"))
    found = module.retrieve(query, folder, SETTINGS)
    checks["same_size_edit_fresh"] = any("29 seconds" in body for _, body in found)
    checks["only_changed_document_read"] = reads == ["dock.md"]
    module.read_document = reader
    document.unlink()
    checks["deleted_source_not_returned"] = not module.retrieve(query, folder, SETTINGS)
    with closing(sqlite3.connect(folder.parent / "documents.sqlite")) as connection:
        checks["deleted_source_not_indexed"] = (
            connection.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
        )
    document.write_text("# Dock\n\nThe reconnect delay is 29 seconds.\n")
    (folder / "binary.dat").write_bytes(b"\x00\x01\x02not a text document")
    module.retrieve(query, folder, SETTINGS)
    with closing(sqlite3.connect(folder.parent / "documents.sqlite")) as connection:
        checks["binary_not_indexed"] = (
            connection.execute("SELECT count(*) FROM chunks WHERE source='binary.dat'").fetchone()[
                0
            ]
            == 0
        )
    try:
        module.retrieve(query, folder, SETTINGS, stopped=lambda: True)
    except InterruptedError, sqlite3.OperationalError:
        checks["cancellation"] = True
    else:
        checks["cancellation"] = False

    def timeout(_query):
        raise TimeoutError("Controlled failure")

    off = module.retrieve(query, folder, SETTINGS)
    checks["timeout_fallback"] = module.retrieve(query, folder, SETTINGS, expand=timeout) == off
    checks["empty_expansion_fallback"] = (
        module.retrieve(query, folder, SETTINGS, expand=lambda _: "") == off
    )
    checks["context_budget"] = len(module.format_context(off * 8, 100)) <= 100
    checks["unknown_requested_file_abstains"] = not module.retrieve(
        "In nonexistent.md, what is the reconnect delay?", folder, SETTINGS
    )
    module.retrieve('"; DROP TABLE chunks; --', folder, SETTINGS)
    checks["fts_query_quoted_safely"] = module.retrieve(query, folder, SETTINGS) == off
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(lambda _: module.retrieve(query, folder, SETTINGS), range(8)))
    checks["independent_worker_connections"] = all(result == off for result in results)

    records = [
        json.loads(line) for line in (ROOT.parent / "results.jsonl").read_text().splitlines()
    ]
    queries = {q["id"]: q for q in json.loads((ROOT.parent / "queries.json").read_text())}
    timings = {}
    for name, implementation in (("before", load(ROOT / "before.py")), ("after", module)):
        for mode in ("off", "on"):
            samples = []
            with closing(sqlite3.connect(DOCUMENTS.parent / "documents.sqlite")) as connection:
                for _ in range(3):
                    for row in records:
                        query = queries[row["id"]]
                        effective = implementation._retrieval_query(
                            query["query"], query["previous"]
                        )
                        start = time.perf_counter()
                        implementation._rank(
                            connection,
                            effective,
                            row["expansion"]["text"] if mode == "on" else "",
                            SETTINGS,
                        )
                        samples.append((time.perf_counter() - start) * 1000)
            samples.sort()
            timings[f"{name}_{mode}"] = {
                "median_ms": statistics.median(samples),
                "p95_ms": samples[int(len(samples) * 0.95)],
                "samples": len(samples),
            }
    result = {"checks": checks, "rank_timings": timings}
    (ROOT / "checks.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    assert all(checks.values()), checks


if __name__ == "__main__":
    run()
