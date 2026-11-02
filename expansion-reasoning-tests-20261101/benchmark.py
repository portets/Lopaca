"""Scratch-only comparison of expansion with thinking disabled versus low effort."""

import json
import random
import sys
import time
import urllib.request
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import LlamaClient
from rag import retrieve, retrieve_memories
from settings import Settings
from storage import MemoryStore

ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:9931"
settings = replace(Settings(), server_url=URL + "/v1")
client = LlamaClient(settings)
cases = [
    (
        "reference",
        "What is the other number in that file?",
        "harbor_client.py",
        [
            {"role": "user", "content": "Where are the recovery settings?"},
            {
                "role": "assistant",
                "content": "See harbor_client.py for the delay and connection limit.",
            },
        ],
    ),
    (
        "new topic",
        "According to warehouse-notes.md, what is the recovery delay?",
        "warehouse-notes.md",
        [
            {"role": "user", "content": "Explain harbor_client.py."},
            {"role": "assistant", "content": "It contains Harbor recovery settings."},
        ],
    ),
    (
        "API default",
        "What about its default?",
        "river_client.py",
        [
            {"role": "user", "content": "How do I configure retries?"},
            {"role": "assistant", "content": "Use retry_limit in river_client.py."},
        ],
    ),
    ("memory paraphrase", "Suggest a recipe using wild yeast.", 1, []),
    ("memory category", "What programming languages do I use?", 2, []),
    ("unrelated", "Explain pulsars and neutron stars.", None, []),
]
results = []
with TemporaryDirectory() as temporary:
    folder = Path(temporary) / "documents"
    folder.mkdir()
    (folder / "harbor_client.py").write_text(
        "# Harbor recovery\nRECOVERY_DELAY = 29\nCONNECTION_LIMIT = 61\n"
    )
    (folder / "warehouse-notes.md").write_text(
        "# Warehouse\nThe recovery delay is 43 seconds.\nThe policy is Copper Ladder.\n"
    )
    (folder / "river_client.py").write_text("def connect(retry_limit=7):\n    return retry_limit\n")
    store = MemoryStore(Path(temporary) / "memories.sqlite", settings)
    for text, terms in [
        ("Enjoys baking sourdough bread.", "cooking recipes baking fermentation natural leaven"),
        ("Prefers Python.", "programming languages coding software development"),
    ]:
        store.apply(
            [
                {
                    "action": "add",
                    "text": text,
                    "kind": "casual",
                    "search_terms": terms,
                    "evidence": text,
                }
            ],
            text,
            "scratch",
            set(),
        )
    retrieve("recovery", folder, settings)
    jobs = [(index, mode) for index in range(len(cases)) for mode in ("off", "low")]
    random.Random(731).shuffle(jobs)
    for index, mode in jobs:
        label, query, expected, history = cases[index]
        row = {"case": label, "mode": mode, "expected": expected}

        def complete(instruction, content, max_tokens, timeout, mode=mode, row=row):
            body = {
                "model": settings.model,
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": content},
                ],
                "stream": False,
                "temperature": 0,
                "max_tokens": 96 if mode == "off" else 608,
                "chat_template_kwargs": {"enable_thinking": False}
                if mode == "off"
                else {"reasoning_effort": "low"},
            }
            if mode == "low":
                body["reasoning_effort"] = "low"
            request = urllib.request.Request(
                URL + "/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            start = time.monotonic()
            with urllib.request.urlopen(request, timeout=90) as response:
                data = json.load(response)
            row.update(seconds=round(time.monotonic() - start, 3), response=data)
            return data["choices"][0]["message"].get("content") or ""

        client._complete = complete
        try:
            expansion = client.expand_query(query, history)
            docs = retrieve(
                query,
                folder,
                settings,
                expand=lambda q, expansion=expansion: expansion,
                previous_queries=[m["content"] for m in history if m["role"] == "user"],
                refresh=False,
            )
            memories = retrieve_memories(store, query, expansion)
            row.update(
                expansion=expansion,
                documents=[ref for ref, text in docs],
                memories=[m[0] for m in memories],
            )
            row["correct"] = (
                any(f"documents/{expected}," in ref for ref, text in docs)
                if isinstance(expected, str)
                else expected in row["memories"]
                if expected is not None
                else not docs and not memories
            )
        except Exception as exc:
            row.update(error=str(exc), correct=False)
        results.append(row)
        (ROOT / "results.json").write_text(json.dumps(results, indent=2))
        print(json.dumps({k: v for k, v in row.items() if k != "response"}), flush=True)
