"""Fresh retrieval validation with recorded real-server expansions and fixed fixtures."""

import json
import random
import shutil
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from compare import APP, DOCUMENTS, SETTINGS, grade, load

sys.path.insert(0, str(APP))
from main import LlamaClient

FOLDER = ROOT / "validation" / "documents"


def prepare():
    if not FOLDER.exists():
        shutil.copytree(DOCUMENTS, FOLDER)
    queries = []

    def add(category, text, facts=()):
        queries.append(
            {
                "id": f"v{len(queries) + 1:03}",
                "category": category,
                "query": text,
                "facts": list(facts),
                "previous": [],
            }
        )

    entries = [
        (
            "concurrency/admission.md",
            "Semaphore",
            "A semaphore bounds concurrent access to a critical section. The admission limit is three permits.",
            "How do we limit how many jobs can enter the protected section at once?",
            "What admission limit does the semaphore enforce?",
            "three permits",
        ),
        (
            "recovery/state.md",
            "Checkpointing",
            "Checkpointing persists resumable execution state. A checkpoint is written every 181 iterations.",
            "How can unfinished calculations resume where they left off after a crash?",
            "How frequently does checkpointing persist state?",
            "181 iterations",
        ),
        (
            "messaging/persistence.md",
            "Transactional outbox",
            "A transactional outbox atomically persists business changes and pending notifications. Each drain handles 27 records.",
            "How do we avoid saving a database change but losing the notification we needed to send?",
            "How many records does the transactional outbox drain handle?",
            "27 records",
        ),
        (
            "dns/absence.md",
            "Negative caching",
            "Negative caching retains unsuccessful DNS lookups. Negative entries expire after 94 seconds.",
            "Can we avoid repeatedly asking the name server about the same nonexistent name?",
            "When do negative caching entries expire?",
            "94 seconds",
        ),
        (
            "analytics/layout.md",
            "Columnar layout",
            "Columnar layout stores values by attribute instead of by record. The compression codec is ZSTD-9.",
            "How can a query read one attribute across all records without loading every field?",
            "Which compression codec does the columnar layout use?",
            "ZSTD-9",
        ),
        (
            "collaboration/merge.md",
            "Conflict-free replicated data types",
            "Conflict-free replicated data types converge without centralized coordination. The merge clock starts at 812 ticks.",
            "How can disconnected editors merge their changes automatically when they reconnect?",
            "What is the initial merge clock for conflict-free replicated data types?",
            "812 ticks",
        ),
        (
            "calendar/dispatch.md",
            "Deadline scheduling",
            "Deadline scheduling prioritizes work by its required completion time. The scheduling horizon is 39 minutes.",
            "How do we prioritize whichever job must finish soonest?",
            "What horizon does deadline scheduling use?",
            "39 minutes",
        ),
        (
            "biology/transport.md",
            "Capillary action",
            "Capillary action moves liquid along narrow channels through adhesion and surface tension. The reference tube radius is 0.38 millimeters.",
            "Why can liquid climb inside a narrow tube without a pump?",
            "What reference tube radius is specified for capillary action?",
            "0.38 millimeters",
        ),
    ]
    for name, title, body, gap, direct, needle in entries:
        path = FOLDER / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {title}\n\n{body}\n")
        add("vocabulary_gap", gap, [(name, needle)])
        add("direct", direct, [(name, needle)])
        add("explicit_file_paraphrase", f"In {name}, {gap[0].lower() + gap[1:]}", [(name, needle)])
    for index, project in enumerate(("Arcadia", "Mistral", "Citrine")):
        name = f"manuals/{project.casefold()}.md"
        path = FOLDER / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"# {project} pipeline\n\n## Delivery\n"
            f"The shipment batching interval is {211 + index} milliseconds.\n\n"
            f"## Ledger\nFinancial records remain available for {521 + index} days.\n"
        )
        add(
            "named_entity",
            f"Give the {project} shipment batching interval and financial record retention.",
            [(name, f"{211 + index} milliseconds"), (name, f"{521 + index} days")],
        )
        add("missing_entity_fact", f"What is the monthly membership fee in dollars for {project}?")
        add("missing_explicit_fact", f"In {name}, what is the employee cafeteria capacity?")
        add(
            "missing_file",
            f"In {project.casefold()}-personnel.md, what is the employee cafeteria capacity?",
        )
    for index, symbol in enumerate(
        ("renewConnectionLease", "reserve_invoice_credit", "release_snapshot_pages")
    ):
        name = f"implementation/handler_{index}.py"
        path = FOLDER / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f'def {symbol}(state):\n    """Return the operation marker."""\n'
            f'    return "copper-{933 + index}"\n\n'
            f'def {symbol}_obsolete(state):\n    return "not-current"\n'
        )
        add("identifier", f"What marker does {symbol} return?", [(name, f"copper-{933 + index}")])
    for text in (
        "What is the annual geothermal output of the Zinnia research station?",
        "What safety certification applies to the Solace diving helmet?",
        "What is the recommended watering schedule for rare alpine orchids?",
        "How many wheels does our lunar excavator have?",
        "What is the exchange rate for copper crowns to platinum marks?",
        "What Unicode normalization form is mandatory for archival filenames?",
    ):
        add("missing_topic", text)
    # Generic headings are deliberate distractors, not candidate-specific exclusions.
    for word in ("Time", "Request", "Other", "Storage"):
        (FOLDER / f"{word.casefold()}-notes.md").write_text(
            f"# {word} notes\n\nThis is an obsolete index of abandoned ideas. Reference QX-17.\n"
        )
    (ROOT / "validation-queries.json").write_text(json.dumps(queries, indent=2))
    return queries


def run():
    before, after = load(ROOT / "before.py"), load(ROOT / "rare_heading.py")
    queries = prepare()
    with closing(sqlite3.connect(FOLDER.parent / "documents.sqlite")) as connection:
        before._index(connection, FOLDER, SETTINGS, lambda: False)
    random.Random(20261030).shuffle(queries)
    client = LlamaClient(SETTINGS)
    path = ROOT / "validation-results.jsonl"
    done = (
        {json.loads(line)["id"] for line in path.read_text().splitlines()}
        if path.exists()
        else set()
    )
    for query in queries:
        if query["id"] in done:
            continue
        row = dict(query)
        start = time.perf_counter()
        try:
            expansion = client.expand_query(query["query"])
            row["expansion"] = expansion
            row["expansion_seconds"] = time.perf_counter() - start
        except Exception as exc:
            expansion = ""
            row["expansion_error"] = str(exc)
        with closing(sqlite3.connect(FOLDER.parent / "documents.sqlite")) as connection:
            for label, module in (("before", before), ("after", after)):
                for mode in ("off", "on"):
                    row[f"{label}_{mode}"] = grade(
                        query,
                        module._rank(
                            connection, query["query"], expansion if mode == "on" else "", SETTINGS
                        ),
                    )
            expression = " OR ".join(before._quote(term) for term in before._terms(query["query"]))
            rows = before._search(connection, expression, SETTINGS.rag_results)
            row["bm25"] = grade(
                query,
                [(f"documents/{item[1]}, lines {item[4]}-{item[5]}", item[3]) for item in rows],
            )
        with path.open("a") as output:
            output.write(json.dumps(row) + "\n")
        done.add(query["id"])
        print(
            f"{len(done)}/{len(queries)} {query['id']} {query['category']} "
            f"before={row['before_off']['complete']}/{row['before_on']['complete']} "
            f"after={row['after_off']['complete']}/{row['after_on']['complete']}",
            flush=True,
        )


if __name__ == "__main__":
    run()
