"""Final fixed-query check after candidate selection; no tuning on its outcomes."""

import json
import random
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from compare import APP, SETTINGS, grade, load

sys.path.insert(0, str(APP))
from main import LlamaClient

CASES = [
    (
        "Which process admission primitive lets only three workers in together?",
        [("concurrency/admission.md", "three permits")],
    ),
    (
        "Does our admission control use a counting semaphore, and what is the permit ceiling?",
        [("concurrency/admission.md", "three permits")],
    ),
    (
        "For the resume-state mechanism, give the interval between snapshots of computation.",
        [("recovery/state.md", "181 iterations")],
    ),
    (
        "Find the prescribed codec for analysis where we access just a column of values.",
        [("analytics/layout.md", "ZSTD-9")],
    ),
    (
        "Where is the queued-event pattern that couples committed business data with a notification, and what is its drain size?",
        [("messaging/persistence.md", "27 records")],
    ),
    (
        "What lifetime is used for cached failed DNS resolutions?",
        [("dns/absence.md", "94 seconds")],
    ),
    (
        "Which method merges offline edits without central coordination, and what initial merge clock does it use?",
        [("collaboration/merge.md", "812 ticks")],
    ),
    (
        "With due-date-based scheduling, how far ahead are jobs planned?",
        [("calendar/dispatch.md", "39 minutes")],
    ),
    (
        "For adhesion-driven fluid transport in tubes, state the documented tube size.",
        [("biology/transport.md", "0.38 millimeters")],
    ),
    (
        "According to messaging/persistence.md, how many pending event rows fit in a dispatch pass?",
        [("messaging/persistence.md", "27 records")],
    ),
    (
        "In manuals/mistral.md, give shipment batch timing and financial-record lifetime.",
        [("manuals/mistral.md", "212 milliseconds"), ("manuals/mistral.md", "522 days")],
    ),
    (
        "What is returned by reserve_invoice_credit in implementation/handler_1.py?",
        [("implementation/handler_1.py", "copper-934")],
    ),
    ("What is returned by renewConnectionLease?", [("implementation/handler_0.py", "copper-933")]),
    (
        "What is returned by release_snapshot_pages_obsolete?",
        [("implementation/handler_2.py", "not-current")],
    ),
    ("What is the daily electrical consumption of the semaphore implementation?", []),
    ("What is the maintainers' email address for Arcadia?", []),
    ("What unit test coverage percentage is required for the transactional outbox?", []),
    ("From documents/nonexistent-network.md, give the packet limit.", []),
    ("Which commercial license do we purchase for the negative caching subsystem?", []),
    ("What is the earthquake rating of the Turquoise viewing dome?", []),
    ("In analytics/layout.md, what is the annual carbon tax rate?", []),
]


def run():
    queries = [
        {"id": f"h{i:03}", "query": text, "facts": facts, "previous": []}
        for i, (text, facts) in enumerate(CASES, 1)
    ]
    (ROOT / "holdout-queries.json").write_text(json.dumps(queries, indent=2))
    random.Random(20261031).shuffle(queries)
    before, after = load(ROOT / "before.py"), load(ROOT / "candidate_final.py")
    client = LlamaClient(SETTINGS)
    for query in queries:
        row = dict(query)
        start = time.perf_counter()
        try:
            expansion = client.expand_query(query["query"])
            row["expansion"] = expansion
        except Exception as exc:
            expansion = ""
            row["expansion_error"] = str(exc)
        row["expansion_seconds"] = time.perf_counter() - start
        with closing(sqlite3.connect(ROOT / "validation/documents.sqlite")) as connection:
            for label, module in (("before", before), ("after", after)):
                for mode in ("off", "on"):
                    row[f"{label}_{mode}"] = grade(
                        query,
                        module._rank(
                            connection, query["query"], expansion if mode == "on" else "", SETTINGS
                        ),
                    )
        with (ROOT / "holdout-results.jsonl").open("a") as output:
            output.write(json.dumps(row) + "\n")
        print(
            query["id"],
            "before",
            row["before_off"]["complete"],
            row["before_on"]["complete"],
            "after",
            row["after_off"]["complete"],
            row["after_on"]["complete"],
            flush=True,
        )


if __name__ == "__main__":
    run()
