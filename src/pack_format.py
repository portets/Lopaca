"""Pack publishing contract and compatibility checks shared with installation tools."""

import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

APP_VERSION = "0.1.0"
INDEX_FORMAT = 1
CATALOG_FORMAT = 1


def version_tuple(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d+\.\d+\.\d+", value):
        raise ValueError(f"Expected a version such as 0.1.0, got {value!r}.")
    return tuple(map(int, value.split(".")))


def compatibility_error(manifest):
    if manifest.get("indexed_format_version") != INDEX_FORMAT:
        return "Unsupported indexed pack format."
    try:
        current = version_tuple(APP_VERSION)
        if current < version_tuple(manifest["min_lopaca_version"]):
            return "Requires a newer Lopaca version."
        maximum = manifest.get("max_lopaca_version")
        if maximum and current > version_tuple(maximum):
            return "This pack supports only older Lopaca versions."
    except (KeyError, ValueError) as exc:
        return f"Invalid pack compatibility metadata: {exc}"
    return None


def publishing_metadata(manifest):
    result = dict(manifest, indexed_format_version=INDEX_FORMAT, min_lopaca_version=APP_VERSION)
    result.setdefault("release_sequence", 1)
    return result


def read_manifest(db):
    row = db.execute("SELECT value FROM index_state WHERE key='pack_manifest'").fetchone()
    if row:
        return json.loads(row[0])
    rows = db.execute("SELECT manifest FROM packs").fetchall()
    if len(rows) != 1:
        raise ValueError("Merged indexes require explicit pack_manifest metadata.")
    return dict(json.loads(rows[0][0]), contains=[])


def check_index(db):
    """Check the indexed storage contract without scanning article bodies."""
    if db.execute("PRAGMA user_version").fetchone()[0] != 2:
        raise ValueError("Unsupported search storage version.")
    required = {
        "prepared": {"start_char", "end_char", "body", "keywords"},
        "archive": {"start_char", "end_char", "body"},
        "table_context": {"source_hash", "body"},
        "articles": {"source", "metadata", "hash"},
        "packs": {"id", "manifest", "enabled"},
        "memberships": {"pack", "source"},
        "pack_aliases": {"pack", "source", "title", "normalized"},
        "chunks": {"source", "body", "keywords"},
    }
    for table, columns in required.items():
        actual = {r[1] for r in db.execute(f"PRAGMA table_info({table})")}
        if not columns <= actual:
            raise ValueError(f"Unsupported {table} storage layout.")


def installed_issues(folder):
    issues = []
    for path in sorted(Path(folder).glob("*.sqlite")):
        try:
            with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
                manifest = read_manifest(db)
                error = compatibility_error(manifest)
                if not error:
                    check_index(db)
            if error:
                issues.append((path, error))
        except (sqlite3.Error, ValueError, TypeError) as exc:
            issues.append((path, str(exc)))
    return issues
