"""Pack formats, compatibility, catalog metadata and verified downloads."""

import hashlib
import json
import re
import shutil
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path
from urllib.request import Request, urlopen

from settings import APP_DIR

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


CATALOG_URL = (
    "https://huggingface.co/datasets/portets/lopaca-wikicat/resolve/main/packs/catalog.json"
)


def validate_catalog(data):
    if data.get("catalog_format_version") != CATALOG_FORMAT:
        raise ValueError("Unsupported knowledge catalog version.")
    releases = data.get("releases")
    if not isinstance(releases, list):
        raise ValueError("Invalid knowledge catalog.")
    seen = set()
    for row in releases:
        if not isinstance(row, dict):
            raise ValueError("Invalid catalog release.")
        for key in ("pack_id", "title", "snapshot_id", "filename", "url"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"Catalog release missing {key}.")
        if not isinstance(row.get("release_sequence"), int) or row["release_sequence"] < 1:
            raise ValueError("Invalid catalog release number.")
        key = row["pack_id"], row["release_sequence"]
        if key in seen:
            raise ValueError("Duplicate catalog release.")
        seen.add(key)
        if not re.fullmatch(r"[a-f0-9]{64}", row.get("sha256", "")):
            raise ValueError("Invalid catalog checksum.")
        if not isinstance(row.get("size_bytes"), int) or row["size_bytes"] < 1:
            raise ValueError("Invalid catalog file size.")
        if not row["url"].startswith("https://huggingface.co/datasets/"):
            raise ValueError("Unsupported catalog download URL.")
        if not isinstance(row.get("hidden", False), bool):
            raise ValueError("Invalid hidden-pack flag.")
        dependencies = row.get("requires", [])
        if not isinstance(dependencies, list) or any(
            not isinstance(key, str) or not key or key == row["pack_id"] for key in dependencies
        ):
            raise ValueError("Invalid pack dependencies.")
        version_tuple(row.get("min_lopaca_version"))
        if row.get("max_lopaca_version"):
            version_tuple(row["max_lopaca_version"])
    return releases


def load_catalog(*, refresh=False, folder=None):
    path = (folder or APP_DIR / "knowledge") / "catalog.json"
    if refresh:
        # Bound metadata size and network wait; the caller runs this off the GTK thread.
        with urlopen(CATALOG_URL, timeout=10) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError("Knowledge catalog is too large.")
        data = json.loads(raw)
        validate_catalog(data)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"catalog-{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(raw)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if not path.exists():
        return {}
    releases = validate_catalog(json.loads(path.read_text()))
    selected = {}
    for row in sorted(releases, key=lambda r: r["release_sequence"]):
        old = selected.get(row["pack_id"])
        # Prefer the newest compatible release, even if a newer incompatible one exists.
        if old is None or not compatibility_error(row) or compatibility_error(old):
            selected[row["pack_id"]] = row
    return selected


def download_pack(release, path, progress=None):
    """Stream a resumable download; verify bytes and identity before installation."""
    validate_catalog({"catalog_format_version": CATALOG_FORMAT, "releases": [release]})
    if error := compatibility_error(release):
        raise ValueError(error)
    total = release["size_bytes"]
    offset = path.stat().st_size if path.exists() else 0
    if offset > total:
        path.unlink()
        offset = 0
    if shutil.disk_usage(path.parent).free < total - offset + 64 * 1024**2:
        raise OSError("Not enough free space to download this knowledge pack.")
    digest = hashlib.sha256()
    if offset:
        with path.open("rb") as stream:
            while block := stream.read(4 * 1024**2):
                digest.update(block)
    if offset < total:
        request = Request(release["url"], headers={"Range": f"bytes={offset}-"} if offset else {})
        with urlopen(request, timeout=60) as response:
            if response.status == 206:
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an unexpected download range.")
            elif response.status == 200:
                offset = 0
                digest = hashlib.sha256()
                if (
                    shutil.disk_usage(path.parent).free
                    + (path.stat().st_size if path.exists() else 0)
                    < total + 64 * 1024**2
                ):
                    raise OSError("Not enough free space to restart this download.")
            else:
                raise ValueError("Unexpected download response.")
            with path.open("ab" if offset else "wb") as stream:
                while block := response.read(1024**2):
                    offset += len(block)
                    if offset > total:
                        path.unlink(missing_ok=True)
                        raise ValueError("Download exceeds the advertised size.")
                    stream.write(block)
                    digest.update(block)
                    if progress:
                        progress("Downloading", offset, total, release["title"])
    if offset != total:
        raise OSError("Download interrupted; retry to resume.")
    if digest.hexdigest() != release["sha256"]:
        path.unlink(missing_ok=True)
        raise ValueError("Pack checksum does not match the catalog; download discarded.")
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        manifest = read_manifest(db)
        for key in ("pack_id", "title", "release_sequence", "indexed_format_version", "contains"):
            if manifest.get(key, [] if key == "contains" else None) != release.get(
                key, [] if key == "contains" else None
            ):
                path.unlink(missing_ok=True)
                raise ValueError(f"Downloaded pack's {key} does not match the catalog.")
    return path
