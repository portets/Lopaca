"""Reproducible lexical RAG evaluation, isolated from real Lopaca chat data."""

import argparse
import hashlib
import json
import random
import shutil
import sqlite3
import statistics
import sys
import time
import urllib.request
from collections import defaultdict
from contextlib import closing
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT.parent
sys.path.insert(0, str(APP))

import rag
from main import LlamaClient
from settings import Settings

SETTINGS = replace(Settings(), server_url="http://127.0.0.1:9931/v1")
DOCUMENTS = ROOT / "corpus" / "documents"

# Each row supplies source, concept, authoritative evidence, vocabulary-gap question,
# and a question with the source terminology. These are fixed before model requests.
TOPICS = [
    (
        "credential-defence.md",
        "Credential throttling",
        "Credential throttling denies authentication after repeated invalid credentials. The lockout threshold is seven attempts.",
        "How do we stop somebody guessing lots of passwords?",
        "What is the credential throttling lockout threshold?",
        "seven attempts",
    ),
    (
        "transport-pressure.md",
        "Backpressure",
        "Backpressure suspends producers when downstream capacity is exhausted. The high-water mark is 640 items.",
        "What prevents a fast sender from overwhelming a slow receiver?",
        "What is the backpressure high-water mark?",
        "640 items",
    ),
    (
        "storage-reclamation.md",
        "Garbage collection",
        "Garbage collection reclaims unreachable objects. Each sweep releases at most 23 files.",
        "How do we reclaim space occupied by things nothing points to anymore?",
        "How many files can garbage collection release per sweep?",
        "23 files",
    ),
    (
        "identity-renewal.md",
        "Credential rotation",
        "Credential rotation revokes compromised secrets and issues replacements. The replacement deadline is 19 hours.",
        "What should we do after somebody leaks our secret?",
        "What is the credential rotation replacement deadline?",
        "19 hours",
    ),
    (
        "replica-quorum.md",
        "Consensus quorum",
        "Consensus quorum tolerates unavailable replicas while preserving agreement. The quorum requires five acknowledgments.",
        "How many machines must agree if some servers disappear?",
        "How many acknowledgments does the consensus quorum require?",
        "five acknowledgments",
    ),
    (
        "work-coalescing.md",
        "Debouncing",
        "Debouncing combines a burst of input events into a single invocation after a quiet interval. The quiet interval is 340 milliseconds.",
        "Can repeated clicks trigger just one action when somebody finally stops?",
        "What quiet interval does debouncing use?",
        "340 milliseconds",
    ),
    (
        "request-idempotency.md",
        "Idempotency",
        "Idempotency prevents duplicate side effects during replayed transactions. The deduplication horizon is 37 minutes.",
        "How can resending a payment avoid charging somebody twice?",
        "What is the idempotency deduplication horizon?",
        "37 minutes",
    ),
    (
        "schedule-fairness.md",
        "Starvation prevention",
        "Weighted fair queuing bounds starvation across competing tenants. The aging coefficient is 0.173.",
        "How can smaller customers get a turn when large customers never stop asking?",
        "What aging coefficient does weighted fair queuing use for starvation prevention?",
        "0.173",
    ),
    (
        "failure-gate.md",
        "Circuit breaker",
        "A circuit breaker halts outbound calls after consecutive failures. The open-state cooldown is 46 seconds.",
        "How can we stop hammering a broken service and let it recover?",
        "What is the circuit breaker open-state cooldown?",
        "46 seconds",
    ),
    (
        "late-work.md",
        "Lazy evaluation",
        "Lazy evaluation defers computation until its result is demanded. The pending-thunk ceiling is 83 entries.",
        "Can we avoid calculating something until somebody actually needs its result?",
        "What is the lazy evaluation pending-thunk ceiling?",
        "83 entries",
    ),
    (
        "snapshot.md",
        "Copy-on-write",
        "Copy-on-write shares immutable pages until mutation requires a private allocation. The private allocation granularity is 96 KiB.",
        "How can a duplicate share memory until somebody changes it?",
        "What allocation granularity does copy-on-write use?",
        "96 KiB",
    ),
    (
        "batch-merge.md",
        "Compaction",
        "Compaction merges sorted runs and drops obsolete tombstones. The merge fan-in is 13 runs.",
        "How are deleted records finally removed from the sorted storage files?",
        "What merge fan-in does compaction use?",
        "13 runs",
    ),
    (
        "clock.md",
        "Logical clock",
        "A Lamport logical clock orders distributed events without synchronized wall time. Its counter starts at 271.",
        "How do we establish what happened first on machines whose watches disagree?",
        "What is the Lamport logical clock initial counter?",
        "271",
    ),
    (
        "password-storage.md",
        "Password hashing",
        "Argon2id password hashing derives a verifier using a unique salt. The memory cost is 72 MiB.",
        "How can we store sign-in checks without keeping readable passwords?",
        "What memory cost does Argon2id password hashing use?",
        "72 MiB",
    ),
    (
        "expiry.md",
        "Time-to-live",
        "Time-to-live expires cached entries after a bounded lifetime. The cache TTL is 117 seconds.",
        "How long can a saved answer stay around before we must fetch a fresh one?",
        "What is the cache time-to-live TTL?",
        "117 seconds",
    ),
    (
        "busy-wait.md",
        "Exponential backoff",
        "Exponential backoff increases retry intervals after transient failures. The retry ceiling is 53 seconds.",
        "Can we wait longer between attempts whenever the request keeps failing?",
        "What retry ceiling does exponential backoff use?",
        "53 seconds",
    ),
    (
        "undo.md",
        "Write-ahead logging",
        "Write-ahead logging persists intent before mutating durable data. The journal sync interval is 61 milliseconds.",
        "How can we recover database changes after a sudden power loss?",
        "What journal sync interval does write-ahead logging use?",
        "61 milliseconds",
    ),
    (
        "owner.md",
        "Lease",
        "A distributed lease grants temporary exclusive ownership. The lease renewal interval is 41 seconds.",
        "How can only one worker own a job until its claim expires?",
        "What renewal interval does the distributed lease use?",
        "41 seconds",
    ),
    (
        "dedupe.md",
        "Content-addressed storage",
        "Content-addressed storage identifies blobs by a digest of their bytes. The blob digest is BLAKE2b-256.",
        "How do we avoid keeping two copies of identical file contents?",
        "What blob digest does content-addressed storage use?",
        "BLAKE2b-256",
    ),
    (
        "partial-read.md",
        "Bloom filter",
        "A Bloom filter rules out missing keys with no false negatives. The configured false-positive target is 0.0047.",
        "Can we quickly tell that an item definitely isn't present without loading the whole database?",
        "What false-positive target does the Bloom filter use?",
        "0.0047",
    ),
    (
        "chunks.md",
        "Pagination",
        "Cursor pagination traverses stable result windows without offset drift. The page size is 73 records.",
        "How do we walk through a long changing result list a little at a time?",
        "What page size does cursor pagination use?",
        "73 records",
    ),
    (
        "isolation.md",
        "Sandboxing",
        "Sandboxing confines untrusted processes with restricted capabilities. The process quota is 29 children.",
        "How do we keep an unknown program from accessing the rest of the machine?",
        "What process quota does sandboxing enforce?",
        "29 children",
    ),
    (
        "merge-changes.md",
        "Operational transformation",
        "Operational transformation rebases concurrent editing operations. The retained operation window is 251 edits.",
        "How can two people type in the same shared document without losing each other's changes?",
        "What operation window does operational transformation retain?",
        "251 edits",
    ),
    (
        "path-choice.md",
        "Consistent hashing",
        "Consistent hashing minimizes key redistribution when membership changes. The ring allocates 113 virtual nodes per member.",
        "How can adding a server avoid moving nearly every saved item to another machine?",
        "How many virtual nodes does consistent hashing allocate per member?",
        "113 virtual nodes",
    ),
]


def write_document(name, text):
    path = DOCUMENTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def prepare():
    DOCUMENTS.mkdir(parents=True, exist_ok=True)
    snapshots = ROOT / "app-snapshot"
    snapshots.mkdir(exist_ok=True)
    hashes = {}
    for name in ("main.py", "ui.py", "storage.py", "settings.py", "rag.py", "agentic.py"):
        shutil.copyfile(APP / name, snapshots / name)
        hashes[name] = hashlib.sha256((APP / name).read_bytes()).hexdigest()
    queries = []

    def add(category, query, facts=(), previous=()):
        queries.append(
            {
                "id": f"q{len(queries) + 1:03}",
                "category": category,
                "query": query,
                "facts": list(facts),
                "previous": list(previous),
            }
        )

    for index, (source, title, evidence, gap, direct, needle) in enumerate(TOPICS):
        write_document(source, f"# {title}\n\n{evidence}\n")
        add(
            "recreated_vocabulary_gap" if index < 8 else "new_vocabulary_gap",
            gap,
            [(source, needle)],
        )
        add("direct_terminology", direct, [(source, needle)])

    projects = ("Harbor", "Meadow", "Cobalt", "Juniper", "Orchid", "Summit", "Linden", "Kestrel")
    for index, project in enumerate(projects):
        source = f"services/{project.casefold()}-guide.md"
        delay, retention = 29 + index * 3, 47 + index * 7
        policy = f"Lantern-{index + 1}"
        write_document(
            source,
            f"# {project} service\n\n## Connection recovery\n"
            f"A dropped connection waits {delay} seconds before reconnecting. "
            f"The recovery policy is named {policy}.\n\n## Audit records\n"
            f"Audit records are retained for {retention} days.\n\n## Deployment\n"
            "Deployments use a rolling update with health probes.\n",
        )
        add(
            "source_scoped",
            f"In {project.casefold()}-guide.md, what is the connection recovery delay?",
            [(source, f"{delay} seconds")],
        )
        add(
            "source_scoped_paraphrase",
            f"According to {project.casefold()}-guide.md, how long do we keep the activity trail?",
            [(source, f"{retention} days")],
        )
        add(
            "multiple_facts",
            f"Give the {project} connection recovery policy and audit retention.",
            [(source, policy), (source, f"{retention} days")],
        )
        add(
            "missing_fact",
            f"What is the monthly subscription price for {project}? We need the price in dollars.",
        )
        add(
            "missing_file", f"In {project.casefold()}-billing.md, what is the invoice grace period?"
        )
        if index < 4:
            add(
                "followup",
                "What is the current value in that document?",
                [(source, f"{delay} seconds")],
                [f"In {project.casefold()}-guide.md, what is the connection recovery delay?"],
            )

    symbols = (
        "readSessionToken",
        "reset_connection_cache",
        "loadAuditPolicy",
        "flush_pending_jobs",
        "rebuild_search_index",
        "scheduleRetry",
        "setConnectionLimit",
        "commit_batch",
        "fetchDocumentVersion",
        "parse_config_value",
        "closeIdleSessions",
        "write_snapshot",
    )
    for index, symbol in enumerate(symbols):
        source = f"code/module_{index:02}.py"
        needle = f"violet-{index + 71}"
        write_document(
            source,
            f'"""Internal service helpers."""\n\n'
            f'def {symbol}(state):\n    """Return the configured marker."""\n'
            f'    return "{needle}"\n\n'
            f'def {symbol}_old(state):\n    return "obsolete-{index}"\n',
        )
        add("exact_identifier", f"What marker does `{symbol}` return?", [(source, needle)])

    # Exact phrases and overloaded words stress drift toward plausible but wrong topics.
    ambiguities = [
        (
            "animals/python.md",
            "Python habitat",
            "The python enclosure temperature is 31 Celsius.",
            "What temperature is specified for the python enclosure?",
            "31 Celsius",
        ),
        (
            "languages/python.md",
            "Python runtime",
            "The Python runtime uses a worker pool of 17 processes.",
            "How many processes does the Python runtime worker pool use?",
            "17 processes",
        ),
        (
            "music/keys.md",
            "Piano keys",
            "The piano keyboard has 88 keys.",
            "How many keys are on the piano keyboard?",
            "88 keys",
        ),
        (
            "security/keys.md",
            "Signing keys",
            "Signing keys are renewed every 67 days.",
            "How often are signing keys renewed?",
            "67 days",
        ),
        (
            "quote-policy.md",
            "Quoted policy",
            'The policy "Silver Window" requires a 137-second pause.',
            'What pause does the "Silver Window" policy require?',
            "137-second",
        ),
        (
            "quotes-other.md",
            "Other policies",
            "The Silver Door policy requires an 11-second pause. The Amber Window policy requires a 19-second pause.",
            'What pause does the "Amber Window" policy require?',
            "19-second",
        ),
        (
            "geography/java.md",
            "Java island",
            "The Java field station identifier is ISLAND-61.",
            "What is the Java island field station identifier?",
            "ISLAND-61",
        ),
        (
            "languages/java.md",
            "Java language",
            "The Java language compiler profile is JVM-24.",
            "What compiler profile does the Java language use?",
            "JVM-24",
        ),
    ]
    for source, title, evidence, query, needle in ambiguities:
        write_document(source, f"# {title}\n\n{evidence}\n")
        add("ambiguity_and_phrases", query, [(source, needle)])

    for query in (
        "What is the warranty period of the Lunar Kettle appliance?",
        "What is the recommended insulin dosage for patient ZYX-149?",
        "What launch date was approved for the underwater tram network?",
        "What is the admission fee for the Nebula Observatory?",
        "What is the checksum for release v99.73.15?",
        "Which Unicode normalization form does our text database enforce?",
        "Which TLS cipher suite is mandated by the regional payment gateway?",
        "What is the maximum sodium content of the campus cafeteria soup?",
    ):
        add("missing_topic", query)

    # Repeated plausible vocabulary competes with the authoritative passages.
    for index in range(24):
        text = "Connection recovery policy delay authentication audit retention cache retry. "
        text += "Python request response document file memory password queue storage server.\n"
        write_document(
            f"archive/incident-{index:02}.md", f"# Archived discussion {index}\n\n" + text * 12
        )
    for name in hashes:
        write_document(f"lopaca/{name}", (APP / name).read_text())
    (ROOT / "queries.json").write_text(json.dumps(queries, indent=2))
    with closing(sqlite3.connect(DOCUMENTS.parent / "documents.sqlite")) as connection:
        rag._index(connection, DOCUMENTS, SETTINGS, lambda: False)
        counts = dict(
            connection.execute(
                "SELECT 'files',count(*) FROM files UNION ALL SELECT 'chunks',count(*) FROM chunks"
            )
        )
    manifest = {
        "settings": asdict(SETTINGS),
        "app_sha256": hashes,
        "queries": len(queries),
        "answerable": sum(bool(q["facts"]) for q in queries),
        "index": counts,
        "seed": 20261029,
        "server_url": SETTINGS.server_url,
        "notes": "New fixed synthetic corpus. Eight vocabulary-gap questions recreate prior examples; original temporary corpus was lost in reboot.",
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


def bounded(passages):
    text = rag.format_context(passages, SETTINGS.max_context_chars)
    # Grade exactly the excerpts that survive the application's final context budget.
    return [
        (block.split("]\n", 1)[0], block.split("]\n", 1)[1])
        for block in text.split("[Source: ")[1:]
        if "]\n" in block
    ]


def grade(query, passages):
    passages = bounded(passages)
    facts = query["facts"]
    hits = [
        any(ref.startswith(f"documents/{source},") and needle in body for ref, body in passages)
        for source, needle in facts
    ]
    relevant = [
        any(ref.startswith(f"documents/{source},") and needle in body for source, needle in facts)
        for ref, body in passages
    ]
    return {
        "complete": all(hits) if facts else None,
        "any_hit": any(hits) if facts else None,
        "top1": relevant[0] if facts and relevant else (False if facts else None),
        "recall": sum(hits) / len(hits) if facts else None,
        "mrr": next((1 / (i + 1) for i, hit in enumerate(relevant) if hit), 0) if facts else None,
        "passage_precision": sum(relevant) / len(relevant) if facts and relevant else 0,
        "false_context": bool(passages) if not facts else None,
        "passages": passages,
    }


def baseline(query):
    effective = rag._retrieval_query(query["query"], query["previous"])
    expression = " OR ".join(rag._quote(term) for term in rag._terms(effective))
    with closing(sqlite3.connect(DOCUMENTS.parent / "documents.sqlite")) as connection:
        rows = rag._search(connection, expression, SETTINGS.rag_results)
    return [(f"documents/{row[1]}, lines {row[4]}-{row[5]}", row[3]) for row in rows]


def run():
    queries = json.loads((ROOT / "queries.json").read_text())
    random.Random(20261029).shuffle(queries)
    client = LlamaClient(SETTINGS)
    if not client.connected():
        raise RuntimeError("llama-server is not healthy")
    for endpoint in ("/health", "/props", "/v1/models"):
        try:
            with urllib.request.urlopen("http://127.0.0.1:9931" + endpoint, timeout=5) as response:
                data = json.load(response)
            (ROOT / (endpoint.strip("/").replace("/", "-") + ".json")).write_text(
                json.dumps(data, indent=2)
            )
        except Exception as exc:
            print(f"Metadata {endpoint}: {exc}", flush=True)
    path = ROOT / "results.jsonl"
    done = (
        {item["id"] for item in map(json.loads, path.read_text().splitlines())}
        if path.exists()
        else set()
    )
    deadline = time.monotonic() + 30 * 60
    for query in queries:
        if query["id"] in done:
            continue
        if time.monotonic() >= deadline:
            print("Benchmark time budget reached", flush=True)
            break
        row = {
            "id": query["id"],
            "category": query["category"],
            "query": query["query"],
            "facts": query["facts"],
        }
        for method in ("bm25", "off", "on"):
            expansion = {"text": "", "error": None, "seconds": 0}

            def expand(text):
                start = time.perf_counter()
                try:
                    expansion["text"] = client.expand_query(text)
                    return expansion["text"]
                except Exception as exc:
                    expansion["error"] = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    expansion["seconds"] = time.perf_counter() - start

            start = time.perf_counter()
            try:
                passages = (
                    baseline(query)
                    if method == "bm25"
                    else rag.retrieve(
                        query["query"],
                        DOCUMENTS,
                        SETTINGS,
                        expand=expand if method == "on" else None,
                        previous_queries=query["previous"],
                    )
                )
                result = grade(query, passages)
                result["seconds"] = time.perf_counter() - start
                row[method] = result
            except Exception as exc:
                row[method] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "seconds": time.perf_counter() - start,
                }
            if method == "on":
                row["expansion"] = expansion
        with path.open("a") as output:
            output.write(json.dumps(row) + "\n")
        done.add(query["id"])
        print(
            f"{len(done)}/{len(queries)} {query['id']} {query['category']} "
            f"off={row['off'].get('complete')} on={row['on'].get('complete')} "
            f"expansion={expansion['seconds']:.2f}s error={expansion['error']}",
            flush=True,
        )


def summarize():
    rows = [json.loads(line) for line in (ROOT / "results.jsonl").read_text().splitlines()]
    summary = {}
    categories = ["all", *sorted({row["category"] for row in rows})]
    for category in categories:
        selected = (
            rows if category == "all" else [row for row in rows if row["category"] == category]
        )
        result = {}
        for method in ("bm25", "off", "on"):
            positive = [row[method] for row in selected if row["facts"]]
            negative = [row[method] for row in selected if not row["facts"]]
            times = sorted(row[method]["seconds"] for row in selected)
            result[method] = {
                "answerable": len(positive),
                "complete": sum(bool(item.get("complete")) for item in positive),
                "any_hit": sum(bool(item.get("any_hit")) for item in positive),
                "top1": sum(bool(item.get("top1")) for item in positive),
                "mean_recall": statistics.mean(item.get("recall") or 0 for item in positive)
                if positive
                else None,
                "mrr": statistics.mean(item.get("mrr") or 0 for item in positive)
                if positive
                else None,
                "mean_passage_precision": statistics.mean(
                    item.get("passage_precision") or 0 for item in positive
                )
                if positive
                else None,
                "missing_answer": len(negative),
                "false_context": sum(bool(item.get("false_context")) for item in negative),
                "errors": sum("error" in row[method] for row in selected),
                "median_ms": statistics.median(times) * 1000,
                "p95_ms": times[min(len(times) - 1, int(len(times) * 0.95))] * 1000,
            }
        result["helped_complete"] = [
            row["id"]
            for row in selected
            if row["facts"] and row["on"].get("complete") and not row["off"].get("complete")
        ]
        result["hurt_complete"] = [
            row["id"]
            for row in selected
            if row["facts"] and row["off"].get("complete") and not row["on"].get("complete")
        ]
        result["new_false_context"] = [
            row["id"]
            for row in selected
            if not row["facts"]
            and row["on"].get("false_context")
            and not row["off"].get("false_context")
        ]
        result["expansion_errors"] = sum(bool(row["expansion"]["error"]) for row in selected)
        result["empty_expansions"] = sum(not row["expansion"]["text"] for row in selected)
        summary[category] = result
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def repeat():
    """Repeat a fixed, category-balanced subset without choosing it by outcome."""
    queries = json.loads((ROOT / "queries.json").read_text())
    groups = defaultdict(list)
    for query in queries:
        groups[query["category"]].append(query)
    selected = []
    for category in sorted(groups):
        selected.extend(sorted(groups[category], key=lambda item: item["id"])[:2])
    client = LlamaClient(SETTINGS)
    for query in selected:
        row = {"id": query["id"], "category": query["category"]}
        effective = rag._retrieval_query(query["query"], query["previous"])
        start = time.perf_counter()
        try:
            row["expansion"] = client.expand_query(effective)
            row["expansion_seconds"] = time.perf_counter() - start
            with closing(sqlite3.connect(DOCUMENTS.parent / "documents.sqlite")) as connection:
                row["result"] = grade(
                    query, rag._rank(connection, effective, row["expansion"], SETTINGS)
                )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        with (ROOT / "repeats.jsonl").open("a") as output:
            output.write(json.dumps(row) + "\n")
        print(
            f"Repeat {query['id']} complete={row.get('result', {}).get('complete')} error={row.get('error')}",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "summarize", "repeat"))
    action = parser.parse_args().action
    {"prepare": prepare, "run": run, "summarize": summarize, "repeat": repeat}[action]()
