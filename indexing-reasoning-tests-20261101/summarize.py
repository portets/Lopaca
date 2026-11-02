"""Summarize timing, structure and held-out retrieval without another model call."""

import json
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parent
rows = json.loads((root / "results.json").read_text())
summary = []
for mode in ("off", "low"):
    runs = [row for row in rows if row["mode"] == mode]
    if not runs:
        continue
    questions = sum(3 * len(row.get("questions", {})) for row in runs)
    lookup = [score for row in runs for score in row.get("retrieval", [])]

    def hits(field, first, lookup=lookup):
        return sum(
            any(
                ref.startswith("documents/" + score["source"] + ",") for ref in score[field][:first]
            )
            for score in lookup
        )

    summary.append(
        {
            "mode": mode,
            "requests": len(runs),
            "valid": sum(row["valid"] for row in runs),
            "mean_seconds": mean(row["seconds"] for row in runs),
            "questions": questions,
            "base_top1": hits("base", 1),
            "hinted_top1": hits("hinted", 1),
            "base_top5": hits("base", 5),
            "hinted_top5": hits("hinted", 5),
            "retrieval_cases": len(lookup),
        }
    )
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
for row in rows:
    print(row["mode"], row["batch"], row["seconds"], row["valid"])
    for key, questions in row.get("questions", {}).items():
        print(key, questions)
