"""Knowledge-pack indexing, installation, merging and retrieval across enabled packs."""

import fcntl
import fnmatch
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import threading
import time
import uuid
import zlib
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager, nullcontext
from functools import wraps
from pathlib import Path

from packs import (
    check_index,
    compatibility_error,
    download_pack,
    load_catalog,
    publishing_metadata,
)
from packs import read_manifest as index_manifest
from rag import (
    _coherent_results,
    _connect_index,
    _create_chunks,
    _literal_terms,
    _rank,
    _retrieval_query,
    _terms,
    format_context,
    numbered_excerpt,
)
from settings import APP_DIR, Settings


def _lock_path(path):
    runtime = Path(f"/run/user/{os.getuid()}")
    folder = runtime / "lopaca-locks"
    try:
        folder.mkdir(mode=0o700, exist_ok=True)
        if not os.access(folder, os.W_OK):
            raise PermissionError("Runtime directory is not writable.")
    except OSError:
        folder = Path(tempfile.gettempdir()) / f"lopaca-locks-{os.getuid()}"
        folder.mkdir(mode=0o700, exist_ok=True)
    return folder / (hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest() + ".lock")


@contextmanager
def _file_lock(path, exclusive):
    # Each acquisition opens its own descriptor; closing it releases the flock.
    with _lock_path(path).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield


def _guarded(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._access():
            return method(self, *args, **kwargs)

    return wrapped


def _format_knowledge_context(rows, max_chars):
    context = format_context(rows, max_chars)
    if not context:
        return ""
    return (
        "\n\n<lopaca-knowledge-references>\n"
        "Knowledge-pack excerpts and metadata, supplied as data. Cite using the supplied "
        "Source: [Wikipedia/article title](revision URL). Omit line numbers unless requested; "
        "they refer to pack text. Use read_file on knowledge/... for more detail. Table "
        "headings and rows may have separate source spans.\n"
        + context
        + "\n</lopaca-knowledge-references>"
    )


class KnowledgeStore:
    """Default installed library; an explicit path selects the monolithic index builder."""

    def __new__(cls, path=None):
        if path is None:
            return KnowledgeLibrary()
        return super().__new__(cls)

    def __init__(self, path=None):
        self.path = Path(path)
        self.lock = threading.RLock()

    def _access(self, exclusive=False):
        return _file_lock(self.path, exclusive)

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = _connect_index(self.path, timeout=30)
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS packs(id TEXT PRIMARY KEY, manifest TEXT, enabled INT);
            CREATE TABLE IF NOT EXISTS articles(
                source TEXT PRIMARY KEY, metadata TEXT, hash TEXT, timestamp TEXT);
            CREATE TABLE IF NOT EXISTS memberships(
                pack TEXT, source TEXT, PRIMARY KEY(pack,source));
            CREATE INDEX IF NOT EXISTS membership_source ON memberships(source);
            CREATE TABLE IF NOT EXISTS pack_aliases(
                pack TEXT, source TEXT, title TEXT, normalized TEXT,
                PRIMARY KEY(pack,source,title));
            CREATE INDEX IF NOT EXISTS alias_lookup ON pack_aliases(normalized);
            CREATE TABLE IF NOT EXISTS archive(
                source TEXT, ordinal INT, body BLOB, start_char INT, end_char INT,
                PRIMARY KEY(source,ordinal));
            CREATE TABLE IF NOT EXISTS prepared(
                source TEXT, ordinal INT, section TEXT, first INT, last INT,
                body BLOB, keywords BLOB, start_char INT, end_char INT,
                PRIMARY KEY(source,ordinal));
            CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS table_context(
                pack TEXT, source TEXT, ordinal INT, group_id INT,
                first INT, last INT, source_hash TEXT, body BLOB,
                PRIMARY KEY(pack,source,ordinal,group_id));
            CREATE INDEX IF NOT EXISTS table_context_lines ON table_context(source,first,last);
            CREATE TABLE IF NOT EXISTS index_state(key TEXT PRIMARY KEY,value TEXT);
        """)
        _create_chunks(db)
        return db

    def _sync(self, db):
        # Only sources whose active membership changed touch FTS; disabled data has no postings.
        active = {
            r[0]
            for r in db.execute("""SELECT DISTINCT source FROM memberships
                    JOIN packs ON pack=packs.id WHERE enabled=1""")
        }
        indexed = {r[0] for r in db.execute("SELECT path FROM files")}
        removed = indexed - active
        # For large removals, rebuilding the small retained corpus is cheaper than
        # tokenizing every removed passage to produce FTS deletion postings.
        rebuild = bool(removed) and len(removed) > len(indexed) // 2
        added = active - indexed
        if rebuild:
            db.execute("INSERT INTO chunks(chunks) VALUES ('delete-all')")
        for source in sorted(removed):
            if rebuild:
                db.execute("DELETE FROM passages WHERE source=?", (source,))
            else:
                db.execute("DELETE FROM chunk_content WHERE source=?", (source,))
            db.execute("DELETE FROM files WHERE path=?", (source,))
        first_new = db.execute("SELECT coalesce(max(id),0) FROM passages").fetchone()[0]
        for source in sorted(added):
            db.execute("INSERT INTO files VALUES (?)", (source,))
            # Prepared blobs already use our codec: preserve them without recompression.
            db.execute(
                """INSERT INTO passages(source,section,body,keywords,start_line,end_line)
                SELECT source,section,body,keywords,first,last
                FROM prepared WHERE source=? ORDER BY ordinal""",
                (source,),
            )
        if rebuild:
            db.execute("INSERT INTO chunks(chunks) VALUES ('rebuild')")
        elif added:
            db.execute(
                """INSERT INTO chunks(rowid,source,section,body,keywords,start_line,end_line)
                SELECT id,source,section,unpack_text(body),unpack_text(keywords),
                    start_line,end_line FROM passages WHERE id>? ORDER BY id""",
                (first_new,),
            )
        for (source,) in db.execute("""SELECT source FROM articles WHERE NOT EXISTS
                (SELECT 1 FROM memberships m WHERE m.source=articles.source)""").fetchall():
            for table in ("archive", "prepared", "articles"):
                db.execute(f"DELETE FROM {table} WHERE source=?", (source,))

    @_guarded
    def import_pack(self, path, progress=None):
        def report(stage, current=None, total=None, title=None):
            if progress:
                progress(stage, current, total, title)

        report("Validating")
        with (
            self.lock,
            closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as src,
        ):
            src.execute("PRAGMA query_only=ON")
            manifest = {k: json.loads(v) for k, v in src.execute("SELECT * FROM manifest")}
            if (
                manifest.get("format") != "lopaca-knowledge-pack"
                or manifest.get("format_version") != "draft-3"
                or manifest.get("codec") != "zlib"
            ):
                raise ValueError("Unsupported knowledge pack format or compression.")
            report("Validating", title=manifest.get("title"))
            if (
                src.execute("PRAGMA quick_check").fetchone()[0] != "ok"
                or src.execute("PRAGMA foreign_key_check").fetchone()
            ):
                raise ValueError("Knowledge pack integrity check failed.")
            for field in ("pack_id", "title", "snapshot_id", "source_revision"):
                if not isinstance(manifest.get(field), str) or not manifest[field]:
                    raise ValueError(f"Knowledge pack manifest requires {field}.")
            if manifest.get("table_context_contract") != "source-spans-v1-experimental":
                raise ValueError("This draft-3 pack lacks the supported table-context contract.")
            for table in ("table_context_groups", "table_context_items"):
                if not src.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
                ).fetchone():
                    raise ValueError(f"Knowledge pack is missing {table}.")
            key = manifest["pack_id"]
            with closing(self._connect()) as db, db:
                previous = db.execute("SELECT enabled FROM packs WHERE id=?", (key,)).fetchone()
                db.execute(
                    "INSERT OR REPLACE INTO packs VALUES (?,?,?)",
                    (key, json.dumps(manifest), previous[0] if previous else 1),
                )
                db.execute("DELETE FROM memberships WHERE pack=?", (key,))
                db.execute("DELETE FROM pack_aliases WHERE pack=?", (key,))
                db.execute("DELETE FROM table_context WHERE pack=?", (key,))
                count = 0
                report("Installing", 0, manifest["article_count"])
                for row in src.execute("SELECT * FROM articles"):
                    (
                        wiki,
                        page,
                        revision,
                        timestamp,
                        title,
                        url,
                        revision_url,
                        snapshot,
                        digest,
                        license_id,
                        attribution,
                    ) = row
                    source = f"knowledge/{wiki}/{page}.md"
                    slices = src.execute(
                        "SELECT ordinal,start_char,end_char,body FROM passages "
                        "WHERE wiki=? AND page_id=? ORDER BY ordinal",
                        (wiki, page),
                    ).fetchall()
                    text_parts, position = [], 0
                    for expected, (ordinal, start, end, body) in enumerate(slices):
                        text = zlib.decompress(body).decode("utf-8")
                        if ordinal != expected or start != position or end != start + len(text):
                            raise ValueError("Noncontiguous article source.")
                        text_parts.append(text)
                        position = end
                    text = "".join(text_parts)
                    if hashlib.sha256(text.encode()).hexdigest() != digest:
                        raise ValueError("Article checksum mismatch.")
                    line_starts = [0] + [i + 1 for i, char in enumerate(text) if char == "\n"]
                    for group in src.execute(
                        "SELECT passage_ordinal,group_id,table_start_char,table_end_char,"
                        "target_start_char,target_end_char,context_status "
                        "FROM table_context_groups WHERE wiki=? AND page_id=?",
                        (wiki, page),
                    ):
                        ordinal, group_id, table_start, table_end, start, end, status = group
                        span = src.execute(
                            "SELECT start_char,end_char FROM search_passages "
                            "WHERE wiki=? AND page_id=? AND ordinal=?",
                            (wiki, page, ordinal),
                        ).fetchone()
                        if (
                            status not in {"provided", "partial", "unavailable"}
                            or not 0 <= table_start <= start < end <= table_end <= len(text)
                            or not span
                            or not span[0] <= start < end <= span[1]
                        ):
                            raise ValueError("Invalid table context target.")
                        items = []
                        for role, a, b, first, last, blob in src.execute(
                            "SELECT role,start_char,end_char,start_line,end_line,body "
                            "FROM table_context_items WHERE wiki=? AND page_id=? "
                            "AND passage_ordinal=? AND group_id=? ORDER BY item_ordinal",
                            (wiki, page, ordinal, group_id),
                        ):
                            value = zlib.decompress(blob).decode()
                            if (
                                role not in {"caption", "column_header", "row_header"}
                                or not table_start <= a < b <= table_end
                                or text[a:b] != value
                                or first != bisect_right(line_starts, a)
                                or last != bisect_right(line_starts, b - 1)
                            ):
                                raise ValueError("Invalid table heading source span.")
                            items.append(
                                {
                                    "role": role,
                                    "start_char": a,
                                    "end_char": b,
                                    "start_line": first,
                                    "end_line": last,
                                    "text": value,
                                }
                            )
                        body = json.dumps(
                            {
                                "status": status,
                                "table_start_char": table_start,
                                "table_end_char": table_end,
                                "items": items,
                                "target_start_char": start,
                                "target_end_char": end,
                                "target_start_line": bisect_right(line_starts, start),
                                "target_end_line": bisect_right(line_starts, end - 1),
                            },
                            ensure_ascii=False,
                        )
                        db.execute(
                            "INSERT INTO table_context VALUES (?,?,?,?,?,?,?,?)",
                            (
                                key,
                                source,
                                ordinal,
                                group_id,
                                bisect_right(line_starts, start),
                                bisect_right(line_starts, end - 1),
                                digest,
                                zlib.compress(body.encode(), 1),
                            ),
                        )
                    old = db.execute(
                        "SELECT hash,timestamp,metadata FROM articles WHERE source=?", (source,)
                    ).fetchone()
                    newer = False
                    if old:
                        old_revision = json.loads(old[2])["revision"]
                        # Wikipedia numeric revision IDs resolve equal/low-resolution dates.
                        if revision.isdecimal() and old_revision.isdecimal():
                            newer = int(revision) > int(old_revision)
                            same_revision = revision == old_revision
                        else:
                            newer = timestamp > old[1]
                            same_revision = revision == old_revision and timestamp == old[1]
                        if old[0] != digest and same_revision:
                            raise ValueError("Conflicting article content at the same revision.")
                    if not old or (old[0] != digest and newer):
                        db.execute("DELETE FROM chunk_content WHERE source=?", (source,))
                        db.execute("DELETE FROM files WHERE path=?", (source,))
                        for table in ("archive", "prepared"):
                            db.execute(f"DELETE FROM {table} WHERE source=?", (source,))
                        metadata = {
                            "title": title,
                            "url": url,
                            "revision": revision,
                            "revision_url": revision_url,
                            "snapshot": snapshot,
                            "license": license_id,
                            "attribution": attribution,
                        }
                        db.execute(
                            "INSERT OR REPLACE INTO articles VALUES (?,?,?,?)",
                            (source, json.dumps(metadata), digest, timestamp),
                        )
                        spans = []
                        for n, section, first, last, start, end, body, keywords in src.execute(
                            "SELECT ordinal,section,start_line,end_line,start_char,end_char,"
                            "body,keywords "
                            "FROM search_passages WHERE wiki=? AND page_id=? ORDER BY ordinal",
                            (wiki, page),
                        ):
                            value = zlib.decompress(body).decode()
                            zlib.decompress(keywords).decode()
                            if text[start:end] != value or first < 1 or last < first:
                                raise ValueError("Invalid search passage source span.")
                            if not 0 <= start <= end <= len(text):
                                raise ValueError("Invalid search passage bounds.")
                            spans.append((start, end))
                            db.execute(
                                "INSERT INTO prepared VALUES (?,?,?,?,?,?,?,?,?)",
                                (source, n, section, first, last, body, keywords, start, end),
                            )
                        # Store only source ranges not already present in search passages.
                        covered = 0
                        gaps = []
                        for start, end in sorted(spans):
                            if start > covered:
                                gaps.append((covered, start))
                            covered = max(covered, end)
                        if covered < len(text):
                            gaps.append((covered, len(text)))
                        db.executemany(
                            "INSERT INTO archive VALUES (?,?,?,?,?)",
                            (
                                (source, n, zlib.compress(text[a:b].encode(), 1), a, b)
                                for n, (a, b) in enumerate(gaps)
                            ),
                        )
                    db.execute("INSERT INTO memberships VALUES (?,?)", (key, source))
                    count += 1
                    if count % 256 == 0 or count == manifest["article_count"]:
                        report("Installing", count, manifest["article_count"])
                if count != manifest["article_count"]:
                    raise ValueError("Article count does not match manifest.")
                if db.execute("SELECT count(*) FROM table_context WHERE pack=?", (key,)).fetchone()[
                    0
                ] != manifest.get("table_context_group_count"):
                    raise ValueError("Table context count does not match manifest.")
                if src.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='aliases'"
                ).fetchone():
                    for wiki, page, title in src.execute(
                        "SELECT wiki,page_id,redirect_title FROM aliases"
                    ):
                        source = f"knowledge/{wiki}/{page}.md"
                        if (
                            not isinstance(title, str)
                            or not title.strip()
                            or not db.execute(
                                "SELECT 1 FROM memberships WHERE pack=? AND source=?", (key, source)
                            ).fetchone()
                        ):
                            raise ValueError("Invalid alias or missing target article.")
                        normalized = " ".join(re.findall(r"\w+", title.casefold()))
                        if not normalized:
                            raise ValueError("Alias must contain searchable text.")
                        db.execute(
                            "INSERT INTO pack_aliases VALUES (?,?,?,?)",
                            (key, source, title, normalized),
                        )
                report("Building search index")
                self._sync(db)
                report("Finishing")
        return manifest["title"]

    @staticmethod
    def _aliases(db, query):
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='pack_aliases'").fetchone():
            return []
        words = re.findall(r"\w+", query.casefold())
        phrases = {
            " ".join(words[start:end])
            for start in range(len(words))
            for end in range(start + 1, min(len(words), start + 12) + 1)
        }
        # Compact model identifiers may omit punctuation, e.g. 3sge vs 3s-ge.
        for word in words:
            if (
                4 <= len(word) <= 16
                and any(char.isdigit() for char in word)
                and any(char.isalpha() for char in word)
            ):
                phrases.update(word[:i] + " " + word[i:] for i in range(1, len(word)))
        aliases = []
        # Standalone stopwords cannot reliably identify a redirect (e.g. "The ?").
        phrases = [phrase for phrase in phrases if _terms(phrase)]
        for start in range(0, len(phrases), 400):
            batch = phrases[start : start + 400]
            placeholders = ",".join("?" for _ in batch)
            aliases.extend(
                db.execute(
                    """SELECT DISTINCT a.source, a.title, a.normalized FROM pack_aliases a
                JOIN packs p ON p.id=a.pack AND p.enabled=1
                JOIN memberships m ON m.pack=a.pack AND m.source=a.source
                WHERE a.normalized IN ("""
                    + placeholders
                    + ")",
                    batch,
                ).fetchall()
            )
        return aliases

    @staticmethod
    def _title_matches(db, query, candidates=()):
        # Search the existing FTS heading index, then verify the complete article title.
        # Unlike general query terms, a single-letter word can be significant inside a title.
        words = re.findall(r"\w+", query.casefold())[:48]

        def title_key(text):
            return " ".join(
                word[:-1]
                if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is"))
                else word
                for word in re.findall(r"\w+", text.casefold())
            )

        found = {}
        for length in range(min(8, len(words)), 1, -1):
            for start in range(len(words) - length + 1):
                phrase = words[start : start + length]
                if not _terms(phrase[0]) or not _terms(phrase[-1]):
                    continue
                normalized = " ".join(phrase)
                if any(" " + normalized + " " in " " + row[2] + " " for row in found.values()):
                    continue
                rows = db.execute(
                    "SELECT source,section FROM chunks WHERE chunks MATCH ? LIMIT 64",
                    ('section:"' + normalized + '"',),
                )
                matches = {}
                for source, section in rows:
                    heading = section.split(" > ", 1)[0]
                    if title_key(heading) != title_key(normalized):
                        continue
                    record = db.execute(
                        "SELECT metadata FROM articles WHERE source=?", (source,)
                    ).fetchone()
                    if record:
                        title = json.loads(record[0])["title"]
                        if title_key(title) == title_key(normalized):
                            matches[source] = (source, title, normalized)
                found.update(matches)
        if not found:
            for source in dict.fromkeys(row[1] for row in candidates):
                row = db.execute(
                    "SELECT metadata FROM articles WHERE source=?", (source,)
                ).fetchone()
                title = json.loads(row[0])["title"]
                parts = re.findall(r"\w+", title.casefold())
                compact = "".join(parts)
                if len(parts) >= 2 and compact in words:
                    found[source] = (source, title, compact)
        if not found:
            # A shortened name may omit a manufacturer's prefix and lack a redirect.
            # Use the rarest substantive word, require a short title ending in it,
            # and keep ambiguous candidates rather than guessing an identity.
            terms = [term for term in _terms(query) if term.isalpha() and len(term) >= 4]
            frequencies = {
                term: db.execute(
                    "SELECT count(*) FROM chunks WHERE chunks MATCH ?", ('"' + term + '"',)
                ).fetchone()[0]
                for term in terms
            }
            for term in sorted(terms, key=frequencies.get)[:3]:
                sources = db.execute(
                    "SELECT DISTINCT source FROM chunks WHERE chunks MATCH ? LIMIT 12",
                    ('section:"' + term + '"',),
                ).fetchall()
                for (source,) in sources:
                    row = db.execute(
                        "SELECT metadata FROM articles WHERE source=?", (source,)
                    ).fetchone()
                    title = json.loads(row[0])["title"]
                    parts = re.findall(r"\w+", title.casefold())
                    if 2 <= len(parts) <= 4 and parts[-1] == term:
                        other_terms = [word for word in _terms(query) if word != term]
                        if other_terms:
                            expression = (
                                "body:("
                                + " OR ".join('"' + word + '"' for word in other_terms)
                                + ")"
                            )
                            evidence = db.execute(
                                "SELECT body FROM chunks WHERE chunks MATCH ? AND rowid IN "
                                "(SELECT id FROM passages WHERE source=?) LIMIT 12",
                                (expression, source),
                            ).fetchall()
                            if not any(
                                set(other_terms) & _literal_terms(body) for (body,) in evidence
                            ):
                                continue
                        found[source] = (source, title, term)
                if found:
                    break
            if found:
                shortest = min(len(re.findall(r"\w+", row[1])) for row in found.values())
                found = {
                    source: row
                    for source, row in found.items()
                    if len(re.findall(r"\w+", row[1])) == shortest
                }
        return list(found.values())

    @staticmethod
    def _table_context(db, source, first, last):
        """Headings have their own citations; never merge them into the row's source span."""
        return [
            json.loads(zlib.decompress(body).decode())
            for (body,) in db.execute(
                "SELECT DISTINCT c.body FROM table_context c JOIN packs p ON p.id=c.pack "
                "JOIN articles a ON a.source=c.source AND a.hash=c.source_hash "
                "WHERE p.enabled=1 AND c.source=? AND c.first<=? AND c.last>=? "
                "ORDER BY c.ordinal,c.group_id",
                (source, last, first),
            )
        ]

    @_guarded
    def search_files(self, query, pattern="knowledge/**", case_sensitive=False):
        if not isinstance(query, str) or not query.strip() or len(query) > 128:
            raise ValueError("Use a knowledge search query of 1-128 characters.")
        if not isinstance(pattern, str) or len(pattern) > 256 or ".." in pattern.split("/"):
            raise ValueError("Invalid knowledge search pattern.")
        if type(case_sensitive) is not bool:
            raise ValueError("case_sensitive must be boolean.")
        if not self.path.exists():
            return {"matches": [], "truncated": False}
        with closing(_connect_index(self.path, timeout=5)) as db:
            deadline = time.monotonic() + 4
            db.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
            try:
                db.execute("BEGIN")
                aliases = self._aliases(db, query)
                rows = _coherent_results(
                    query,
                    _rank(db, query, "", Settings(), split_compounds=True),
                    lambda row: row[2] + "\n" + row[3],
                )
                matches = []
                for source, title, _ in aliases:
                    if fnmatch.fnmatchcase(source, pattern) and (
                        not case_sensitive or query in title
                    ):
                        matches.append({"path": source, "alias": title})
                for _, source, section, body, first, last in rows:
                    if fnmatch.fnmatchcase(source, pattern) and (
                        not case_sensitive or query in body
                    ):
                        matches.append(
                            {
                                "path": source,
                                "line": first,
                                "end_line": last,
                                "section": section,
                                "text": body[:1000],
                                "table_context": self._table_context(db, source, first, last),
                            }
                        )
                return {
                    "matches": matches[:20],
                    "truncated": len(matches) > 20,
                    "hint": "Use read_file for more article lines. Search uses lexical ranking.",
                }
            except sqlite3.OperationalError as exc:
                if "interrupted" not in str(exc):
                    raise
                return {
                    "matches": [],
                    "truncated": True,
                    "hint": "Search timed out; try a more specific query.",
                }

    @_guarded
    def context(
        self, query, settings, expand=None, previous_queries=(), memory_rows=(), _rows=False
    ):
        if not self.path.exists():
            return [] if _rows else ""
        query = _retrieval_query(query, previous_queries)
        with closing(_connect_index(self.path, timeout=30)) as db:
            if not db.execute("SELECT 1 FROM files LIMIT 1").fetchone():
                return [] if _rows else ""
            expansion = expand(query) if expand else ""
            db.execute("BEGIN")
            ranked = _rank(
                db, query, expansion, settings, require_identifiers=True, split_compounds=True
            )
            # Only query-specific casual memories become search clues; always-present core
            # memories would otherwise pull personal-topic articles into unrelated answers.
            memory_query = "\n".join(text for _, text, core in memory_rows if not core)
            if memory_query:
                memory_ranked = _rank(
                    db, memory_query, "", settings, require_identifiers=True, split_compounds=True
                )
                scores, candidates = {}, {}
                for weight, results in ((1.0, ranked), (1.0, memory_ranked)):
                    for rank, row in enumerate(results, 1):
                        scores[row[0]] = scores.get(row[0], 0.0) + weight / (60 + rank)
                        candidates[row[0]] = row
                ranked = sorted(candidates.values(), key=lambda row: scores[row[0]], reverse=True)[
                    : settings.rag_results
                ]
            matched_aliases = {}
            # Match complete redirect titles in the original question.
            aliases = self._aliases(db, query)
            # A short redirect naming only one part of a compound query is not a
            # reliable subject anchor (e.g. a company inside a specific product query).
            query_terms = set(_terms(query))
            aliases = [
                row
                for row in aliases
                if len(row[2].split()) >= 2
                or len(set(_terms(row[1])) & query_terms) >= 0.6 * len(query_terms)
            ]
            redirect_keys = {(source, title) for source, title, _ in aliases}
            title_matches = self._title_matches(db, query, ranked)
            matched_titles = {}
            matches = aliases + title_matches
            if matches:
                # Prefer specific names over contained aliases, while keeping independently
                # named subjects (a longer title must not hide a second, shorter title).
                aliases = [
                    row
                    for row in matches
                    if not any(
                        row[2] != other[2] and " " + row[2] + " " in " " + other[2] + " "
                        for other in matches
                    )
                ]
                alias_rows = []
                extra_alias_rows = []
                supplements = []
                for source, title, _ in aliases:
                    if (source, title) in redirect_keys:
                        matched_aliases.setdefault(source, set()).add(title)
                    for item in title_matches:
                        if item[0] == source and item[2] != " ".join(
                            re.findall(r"\w+", title.casefold())
                        ):
                            matched_titles.setdefault(source, set()).add(item[2])
                    metadata = json.loads(
                        db.execute(
                            "SELECT metadata FROM articles WHERE source=?", (source,)
                        ).fetchone()[0]
                    )
                    candidates = _rank(
                        db,
                        query + " " + title + " " + metadata["title"],
                        expansion,
                        settings,
                    )
                    matching = [r for r in candidates if r[1] == source]
                    row = matching[0] if matching else None
                    if row is None:
                        row = db.execute(
                            "SELECT rowid,source,section,body,start_line,end_line "
                            "FROM chunk_content WHERE source=? "
                            "ORDER BY start_line,rowid LIMIT 1",
                            (source,),
                        ).fetchone()
                    if row and row[0] not in {r[0] for r in alias_rows}:
                        alias_rows.append(row)
                    if row and " > " in row[2]:
                        parent = row[2].rsplit(" > ", 1)[0]
                        overview = db.execute(
                            "SELECT rowid,source,section,body,start_line,end_line "
                            "FROM chunk_content "
                            "WHERE source=? AND section=? ORDER BY start_line,rowid LIMIT 2",
                            (source, parent),
                        ).fetchall()
                        for context_row in overview:
                            if (
                                context_row[0] not in {r[0] for r in alias_rows}
                                and re.sub(r"(?m)^\s*#{1,6}\s+[^\n]*", "", context_row[3]).strip()
                            ):
                                alias_rows.append(context_row)
                    # A redirect may name a section with several useful subsections.
                    # Include distinct child sections before repeating similar excerpts.
                    # The article root alone must not pull in every unrelated subsection.
                    section_names = db.execute(
                        "SELECT DISTINCT section FROM passages WHERE source=? "
                        "AND section!='' ORDER BY start_line",
                        (source,),
                    ).fetchall()
                    alias_key = re.sub(r"\W+", "", title.casefold())
                    for (section_name,) in section_names:
                        parts = section_name.split(" > ")
                        if len(parts) < 2 or not any(
                            re.sub(r"\W+", "", part.casefold()) == alias_key for part in parts[1:-1]
                        ):
                            continue
                        child = db.execute(
                            "SELECT rowid,source,section,body,start_line,end_line "
                            "FROM chunk_content WHERE source=? AND section=? "
                            "ORDER BY start_line,rowid LIMIT 1",
                            (source, section_name),
                        ).fetchone()
                        if child:
                            extra_alias_rows.append(child)
                    supplements.extend(matching[1:])
                # Give ambiguous targets one result each before additional passages.
                seen = {r[0] for r in alias_rows}
                # Keep memory-informed results visible when a generic redirect matches.
                remaining = (
                    memory_ranked + extra_alias_rows + supplements + ranked
                    if memory_query
                    else extra_alias_rows + supplements + ranked
                )
                original = {re.sub(r"\W+", "", term) for term in query_terms}
                identifiers = {
                    re.sub(r"\W+", "", term)
                    for term in query_terms
                    if any(c.isalpha() for c in term) and any(c.isdigit() for c in term)
                }
                covered = set().union(
                    *(_literal_terms(row[2] + "\n" + row[3]) for row in alias_rows)
                )
                child_ids = {row[0] for row in extra_alias_rows}
                named_sources = {row[1] for row in alias_rows}
                for row in remaining:
                    if row[0] in seen:
                        continue
                    literals = _literal_terms(row[2] + "\n" + row[3])
                    novel = (original & literals) - covered
                    if row[0] not in child_ids and not (
                        novel
                        and (row[1] in named_sources or novel & identifiers or len(novel) >= 2)
                    ):
                        continue
                    alias_rows.append(row)
                    seen.add(row[0])
                    covered.update(literals)
                ranked = alias_rows[: settings.rag_results]
            if not matches:
                ranked = _coherent_results(query, ranked, lambda row: row[2] + "\n" + row[3])
            requested_ids = [
                term
                for term in _terms(query)
                if (any(c.isalpha() for c in term) and any(c.isdigit() for c in term))
                or "_" in term
            ]
            rows = []
            outlined_sources = set()
            for _, source, section, body, first, last in ranked:
                metadata = json.loads(
                    db.execute(
                        "SELECT metadata FROM articles WHERE source=?", (source,)
                    ).fetchone()[0]
                )
                reference = (
                    f"Knowledge pack: {source}, {metadata['title']}, "
                    f"prepared-text lines {first}-{last}, "
                    f"revision {metadata['revision']}, {metadata['revision_url']}; "
                    f"citation: Source: [Wikipedia/{metadata['title']}]({metadata['revision_url']})"
                )
                if section:
                    reference += f", section: {section}"
                literals = _literal_terms(section + "\n" + body) if requested_ids else set()
                unmatched = [
                    term for term in requested_ids if re.sub(r"\W+", "", term) not in literals
                ]
                if unmatched:
                    body = (
                        "Retrieval scope, not source text: this excerpt does not identify "
                        + ", ".join(unmatched)
                        + ". It may cover another part of the request; do not infer a "
                        "relationship to those identifiers.\n\n" + body
                    )
                outline = []
                if source not in outlined_sources and source in {item[0] for item in title_matches}:
                    outline = [
                        {"section": name.split(" > ")[-1], "line": line}
                        for name, line in db.execute(
                            "SELECT section,min(start_line) FROM passages WHERE source=? "
                            "GROUP BY section ORDER BY min(start_line)",
                            (source,),
                        )
                        if name.count(" > ") == 1
                    ][:24]
                if outline:
                    outlined_sources.add(source)
                    body = (
                        "Article section navigation (not a complete factual excerpt): "
                        + json.dumps(outline, ensure_ascii=False)
                        + "\n\n"
                        + body
                    )
                if source in matched_titles:
                    names = json.dumps(sorted(matched_titles[source]), ensure_ascii=False)
                    body = (
                        f"Pack title metadata: {names} → {metadata['title']}. "
                        "This is a title-word match, not an independent redirect.\n\n" + body
                    )
                if source in matched_aliases:
                    # Aliases are pack metadata, not words from the cited source lines.
                    titles = json.dumps(sorted(matched_aliases[source]), ensure_ascii=False)
                    body = (
                        f"Pack redirect metadata: {titles} → {metadata['title']}. "
                        "This mapping is not part of the article excerpt.\n\n" + body
                    )
                for group in self._table_context(db, source, first, last):
                    body += (
                        f"\n\nTable context for prepared-text lines "
                        f"{group['target_start_line']}-{group['target_end_line']} "
                        f"({group['status']}; relationships may be incomplete):"
                    )
                    for item in group["items"]:
                        body += (
                            f"\n{item['role']} [Source: {source}, prepared-text lines "
                            f"{item['start_line']}-{item['end_line']}]:\n{item['text']}"
                        )
                if _rows:
                    # Keep article identity and real aliases separate from rendered excerpts.
                    key = (source, first, last, str(metadata["revision"]))
                    names = (
                        metadata["title"],
                        *matched_aliases.get(source, ()),
                        *matched_titles.get(source, ()),
                    )
                    rows.append((key, names, reference, body))
                else:
                    rows.append((reference, body))
            if _rows:
                return rows
            return _format_knowledge_context(rows, settings.max_context_chars)

    @_guarded
    def read_file(self, path, start_line=1, line_count=100):
        if (
            type(start_line) is not int
            or type(line_count) is not int
            or min(start_line, line_count) < 1
        ):
            raise ValueError("Line numbers and counts must be positive integers.")
        with closing(_connect_index(self.path, timeout=30)) as db:
            if not db.execute("SELECT 1 FROM files WHERE path=?", (path,)).fetchone():
                raise ValueError("Article is not in an enabled knowledge pack.")
            db.execute("BEGIN")
            pieces, position = [], 0
            for start, end, blob in db.execute(
                "SELECT start_char,end_char,body FROM prepared WHERE source=? "
                "UNION ALL SELECT start_char,end_char,body FROM archive WHERE source=? "
                "ORDER BY 1,2",
                (path, path),
            ):
                if start > position:
                    raise ValueError("Missing source text.")
                if end > position:
                    pieces.append(zlib.decompress(blob).decode()[position - start :])
                    position = end
            text = "".join(pieces)
            expected = db.execute("SELECT hash FROM articles WHERE source=?", (path,)).fetchone()[0]
            if hashlib.sha256(text.encode()).hexdigest() != expected:
                raise ValueError("Reconstructed article checksum mismatch.")
            lines = text.splitlines()
            if start_line > len(lines) + 1:
                raise ValueError(f"Article has {len(lines)} lines.")
            result = numbered_excerpt(path, lines, start_line, line_count, text_key="content")
            result["table_context"] = self._table_context(db, path, start_line, result["end_line"])
            return result


_LIBRARY_LOCK = threading.RLock()


def _finish_index(path):
    # Published indexes are read-only; rollback journaling avoids persistent WAL sidecars.
    with closing(sqlite3.connect(path, timeout=30)) as db:
        if db.execute("PRAGMA journal_mode=DELETE").fetchone()[0] != "delete":
            raise RuntimeError("Could not finalize the knowledge index.")


def _index_name(manifest):
    category = re.findall(r"[a-z0-9]+", manifest["title"].casefold())
    source = manifest.get("source_dataset", "")
    if (
        "wikipedia" in (manifest["pack_id"] + " " + source).casefold()
        or "wikimedia" in source.casefold()
    ):
        source = "wiki"
    else:
        source = re.sub(r"[^a-z0-9]+", "-", source.casefold()).strip("-") or "knowledge"
    return f"{category[0] if category else 'pack'}-{source}-indexed.sqlite"


def set_index_metadata(path, pack_id, title, contains=()):
    """Write publishing metadata for an index created by Lopaca."""
    if not pack_id or not title or pack_id in contains or any(not key for key in contains):
        raise ValueError("Use nonempty metadata and constituent IDs different from the parent.")
    with closing(_connect_index(path)) as db, db:
        rows = db.execute("SELECT id,manifest FROM packs").fetchall()
        manifests = [json.loads(row[1]) for row in rows]
        manifest = manifests[0]
        if len(manifests) > 1:
            # Full constituent manifests remain in packs; don't mislabel the aggregate
            # with one category's counts, description or publishing identity.
            manifest = {
                k: v
                for k, v in manifest.items()
                if all(other.get(k) == v for other in manifests[1:])
                and not k.endswith("_count")
                and k not in {"pack_id", "title", "description", "contains", "compatibility_error"}
            }
            manifest["description"] = "Combined installed Wikipedia packs."
            manifest["release_sequence"] = max(m.get("release_sequence", 1) for m in manifests)
        manifest = publishing_metadata(manifest)
        manifest.update(pack_id=pack_id, title=title, contains=list(contains))
        manifest["article_count"] = db.execute("SELECT count(*) FROM articles").fetchone()[0]
        db.execute(
            "INSERT OR REPLACE INTO index_state VALUES ('pack_manifest',?)", (json.dumps(manifest),)
        )


class KnowledgeLibrary:
    def __init__(self, folder=None):
        self.folder = Path(folder) if folder else APP_DIR / "knowledge"
        self.lock = _LIBRARY_LOCK

    def _operation(self, *, exclusive=False):
        self.folder.mkdir(parents=True, exist_ok=True)
        return _file_lock(self.folder / ".operations", exclusive)

    def _catalog(self):
        try:
            return load_catalog(folder=self.folder)
        except OSError, ValueError:
            return {}

    def _entries(self):
        entries = []
        catalog = self._catalog()
        state_path = self.folder / "packs.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        for path in sorted(self.folder.glob("*.sqlite")):
            try:
                with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
                    manifest = index_manifest(db)
                    policy = catalog.get(manifest["pack_id"], {})
                    manifest.update(
                        hidden=policy.get("hidden", False), requires=policy.get("requires", [])
                    )
                    manifest["compatibility_error"] = compatibility_error(manifest)
            except sqlite3.OperationalError:
                if not path.exists():
                    continue
                raise
            entries.append((path, manifest, state.get(manifest["pack_id"], True)))
        ids = [m["pack_id"] for _, m, _ in entries]
        if len(ids) != len(set(ids)):
            raise ValueError("Multiple installed files have the same pack ID.")
        suppressed = self._suppressed(entries)
        needed = {
            key
            for _, m, on in entries
            if on and m["pack_id"] not in suppressed and not m.get("compatibility_error")
            for key in m.get("requires", [])
        }
        # Hidden dependencies follow their consumers rather than a separate user switch.
        return [(p, m, m["pack_id"] in needed if m.get("hidden") else on) for p, m, on in entries]

    @staticmethod
    def _suppressed(entries):
        manifests = {m["pack_id"]: m for _, m, _ in entries}
        suppressed = {}
        for _, parent, enabled in entries:
            if not enabled or parent.get("compatibility_error"):
                continue
            pending = list(parent.get("contains", []))
            seen = {parent["pack_id"]}
            while pending:
                key = pending.pop()
                if key in seen:
                    continue
                seen.add(key)
                suppressed.setdefault(key, parent["title"])
                pending.extend(manifests.get(key, {}).get("contains", []))
        return suppressed

    def list_packs(self):
        entries = self._entries()
        suppressed = self._suppressed(entries)
        result = []
        for path, manifest, enabled in entries:
            parent = suppressed.get(manifest["pack_id"])
            manifest = dict(manifest, unused_by=parent, size_bytes=path.stat().st_size)
            result.append(
                (
                    manifest["pack_id"],
                    manifest,
                    enabled and not parent and not manifest.get("compatibility_error"),
                )
            )
        return result

    def _active(self):
        entries = self._entries()
        contained = self._suppressed(entries)
        return [
            KnowledgeStore(path)
            for path, m, on in entries
            if on and not m.get("compatibility_error") and m["pack_id"] not in contained
        ]

    def import_pack(self, path, progress=None, *, _consume=False):
        def report(stage):
            if progress:
                progress(stage, None, None, None)

        with self._operation():
            source = Path(path).resolve()
            temporary = self.folder / f"import-{uuid.uuid4().hex}.building"
            try:
                with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as db:
                    indexed = db.execute(
                        "SELECT 1 FROM sqlite_master WHERE name='prepared'"
                    ).fetchone()
                if indexed:
                    report("Validating")
                    with closing(_connect_index(source)) as db:
                        manifest = index_manifest(db)
                        check_index(db)
                        if error := compatibility_error(manifest):
                            raise ValueError(error)
                        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                            raise ValueError("Invalid indexed pack.")
                        if db.execute("SELECT 1 FROM packs WHERE enabled=0 LIMIT 1").fetchone():
                            raise ValueError(
                                "Re-enable all constituents before exporting an index."
                            )
                    report("Installing")
                    if _consume:
                        if source.parent != self.folder.resolve():
                            raise ValueError(
                                "Downloaded index must be staged in the knowledge folder."
                            )
                        source.replace(temporary)
                    else:
                        with (
                            closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as db,
                            closing(sqlite3.connect(temporary)) as dest,
                        ):
                            db.backup(dest)
                else:
                    index = KnowledgeStore(temporary)
                    index.import_pack(source, progress)
                    with closing(_connect_index(temporary)) as db, db:
                        manifest = publishing_metadata(index_manifest(db))
                        db.execute(
                            "INSERT OR REPLACE INTO index_state VALUES ('pack_manifest',?)",
                            (json.dumps(manifest),),
                        )
                _finish_index(temporary)
                key = manifest["pack_id"]
                with self.lock:
                    entries = self._entries()
                    old = next((p for p, m, _ in entries if m["pack_id"] == key), None)
                    destination = self.folder / _index_name(manifest)
                    if any(p == destination and m["pack_id"] != key for p, m, _ in entries):
                        stem = destination.stem
                        destination = destination.with_name(
                            f"{stem}-{hashlib.sha256(key.encode()).hexdigest()[:8]}.sqlite"
                        )
                    if destination.exists() and destination != old:
                        raise ValueError("Knowledge pack filename is already in use.")
                    if old:
                        with KnowledgeStore(old)._access(exclusive=True):
                            temporary.replace(destination)
                            if old != destination:
                                old.unlink()
                    else:
                        temporary.replace(destination)
                return manifest["title"]
            finally:
                for suffix in ("", "-wal", "-shm"):
                    Path(str(temporary) + suffix).unlink(missing_ok=True)
                _lock_path(temporary).unlink(missing_ok=True)

    def download_pack(self, release, progress=None, *, _dependency=False):
        with self._operation():
            staged = self.folder.resolve() / (release["sha256"] + ".download")
            with KnowledgeStore(staged)._access(exclusive=True):
                if _dependency and any(
                    m["pack_id"] == release["pack_id"]
                    and m.get("release_sequence", 0) >= release["release_sequence"]
                    for _, m, _ in self._entries()
                ):
                    return release["title"]
                download_pack(release, staged, progress)
                title = self.import_pack(staged, progress, _consume=True)
        if not _dependency:
            self.ensure_dependencies(progress)
        return title

    def missing_dependencies(self):
        catalog = self._catalog()
        installed = {m["pack_id"]: m for _, m, _ in self._entries()}
        return [
            key
            for key in {k for m in installed.values() for k in m.get("requires", [])}
            if key not in catalog
            or (
                key not in installed
                or installed[key].get("release_sequence", 0) < catalog[key]["release_sequence"]
            )
        ]

    def ensure_dependencies(self, progress=None):
        catalog = self._catalog()
        for key in sorted(self.missing_dependencies()):
            release = catalog.get(key)
            if release is None:
                raise ValueError(f"Shared knowledge data {key} is missing from the catalog.")
            if progress:
                progress("Downloading shared Wikipedia data", None, None, None)
            try:

                def shared_progress(stage, current, total, title):
                    if progress:
                        progress(f"Shared data — {stage}", current, total, None)

                self.download_pack(release, shared_progress, _dependency=True)
            except Exception as exc:
                raise RuntimeError(
                    "Your requested pack is installed, but shared data could not be installed: "
                    f"{exc}"
                ) from exc

    def change(self, key, enabled=None):
        # Toggles only write preferences, so they need not wait for a long merge.
        with self._operation() if enabled is None else nullcontext(), self.lock:
            entry = next((p for p, m, _ in self._entries() if m["pack_id"] == key), None)
            if entry is None:
                raise ValueError("Pack is no longer installed.")
            if enabled:
                with closing(sqlite3.connect(entry.as_uri() + "?mode=ro", uri=True)) as db:
                    if error := compatibility_error(index_manifest(db)):
                        raise ValueError(error)
            if enabled is None:
                with KnowledgeStore(entry)._access(exclusive=True):
                    entry.unlink()
            state_path = self.folder / "packs.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if enabled is None:
                state.pop(key, None)
            else:
                state[key] = bool(enabled)
            temporary = state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state))
            temporary.replace(state_path)
            if enabled is None:
                remaining = self._entries()
                needed = {k for _, m, _ in remaining for k in m.get("requires", [])}
                for path, manifest, _ in remaining:
                    if manifest.get("hidden") and manifest["pack_id"] not in needed:
                        with KnowledgeStore(path)._access(exclusive=True):
                            path.unlink()

    def context(self, query, settings, expand=None, previous_queries=(), memory_rows=()):
        with self.lock:
            stores = self._active()
            if not stores:
                return ""
            query = _retrieval_query(query, previous_queries)
            # Resolve expansion once before worker threads, preserving the shared request cache.
            expansion = expand(query) if expand else ""

            def retrieve(store):
                return store.context(
                    query,
                    settings,
                    expand=lambda _: expansion,
                    memory_rows=memory_rows,
                    _rows=True,
                )

            with ThreadPoolExecutor(max_workers=min(4, len(stores))) as pool:
                results = list(pool.map(retrieve, stores))
            scores, rows = {}, {}
            memory_clues = [text for _, text, core in memory_rows if not core]
            clues = "\n".join([query, *memory_clues])
            query_key = " " + " ".join(re.findall(r"\w+", clues.casefold())) + " "

            def strength(ranked):
                return max(
                    (
                        len(key)
                        for _, names, _, _ in ranked
                        for name in names
                        if (key := " ".join(re.findall(r"\w+", name.casefold())))
                        and " " + key + " " in query_key
                    ),
                    default=0,
                )

            strengths = [strength(ranked) for ranked in results]
            strongest = max(strengths, default=0)
            for ranked, anchor in zip(results, strengths, strict=True):
                weight = max(0.15, anchor / strongest) if strongest else 1.0
                for rank, (key, _, reference, body) in enumerate(ranked, 1):
                    scores[key] = scores.get(key, 0) + weight / (60 + rank)
                    rows.setdefault(key, (reference, body))
            # Only narrow a direct name lookup. Search stop words include clues such
            # as "how many", which distinguish descriptive questions from name lookups.
            subject = re.sub(
                r"^(?:(?:who|what) (?:is|was|are|were)|(?:please )?tell me about)\s+",
                "",
                query.strip().casefold(),
            )
            subject = re.sub(r"^(?:a|an|the)\s+", "", subject)
            name_key = re.findall(r"\w+", subject)
            if strongest and len(_terms(subject)) >= 2 and not memory_clues:
                named_sources = {
                    key[0]
                    for ranked in results
                    for key, names, _, _ in ranked
                    if any(re.findall(r"\w+", name.casefold()) == name_key for name in names)
                }
                if named_sources:
                    rows = {key: row for key, row in rows.items() if key[0] in named_sources}
            newest = {}
            for source, _, _, revision in rows:
                if revision.isdecimal():
                    newest[source] = max(newest.get(source, 0), int(revision))
            eligible = [
                key for key in rows if not key[3].isdecimal() or int(key[3]) == newest[key[0]]
            ]
            if not strongest:
                eligible = _coherent_results(query, eligible, lambda key: "\n".join(rows[key]))
            chosen = sorted(eligible, key=scores.get, reverse=True)[: settings.rag_results]
            return _format_knowledge_context(
                [rows[key] for key in chosen], settings.max_context_chars
            )

    def read_file(self, path, start_line=1, line_count=100):
        with self.lock:
            candidates = []
            for store in self._active():
                with closing(_connect_index(store.path)) as db:
                    found = db.execute(
                        "SELECT metadata,timestamp FROM articles JOIN files ON path=source "
                        "WHERE path=?",
                        (path,),
                    ).fetchone()
                if found:
                    revision = json.loads(found[0])["revision"]
                    candidates.append(
                        ((int(revision) if revision.isdecimal() else 0, found[1]), store)
                    )
            if candidates:
                store = max(candidates, key=lambda item: item[0])[1]
                return store.read_file(path, start_line, line_count)
            raise ValueError("Article is not in an enabled knowledge pack.")

    def search_files(self, query, pattern="knowledge/**", case_sensitive=False):
        with self.lock:
            stores = self._active()
            with ThreadPoolExecutor(max_workers=max(1, min(4, len(stores)))) as pool:
                results = list(
                    pool.map(lambda s: s.search_files(query, pattern, case_sensitive), stores)
                )
            matches, seen = [], set()
            for rank in range(20):
                for result in results:
                    if rank < len(result["matches"]):
                        row = result["matches"][rank]
                        key = json.dumps(row, sort_keys=True)
                        if key not in seen:
                            matches.append(row)
                            seen.add(key)
            return {
                "matches": matches[:20],
                "truncated": len(matches) > 20 or any(r["truncated"] for r in results),
                "hint": "Use read_file for more detail.",
            }

    def merge_all(self, progress=None, catalog=None):
        with self._operation(exclusive=True):
            entries = [
                entry for entry in self._entries() if not entry[1].get("compatibility_error")
            ]
            if len(entries) < 2:
                raise ValueError("Install at least two packs to merge.")
            keys = sorted(
                {
                    key
                    for _, manifest, _ in entries
                    for key in (manifest.get("contains") or [manifest["pack_id"]])
                }
            )
            if catalog is None:
                try:
                    catalog = list(load_catalog().values())
                except OSError, ValueError:
                    catalog = []
            parent = next(
                (
                    m
                    for m in catalog
                    if set(m.get("contains", [])) == set(keys)
                    and not compatibility_error(m)
                    and all(
                        m.get("snapshot_id") == item.get("snapshot_id")
                        and m.get("source_revision") == item.get("source_revision")
                        for _, item, _ in entries
                    )
                ),
                None,
            )
            pack_id = (
                parent["pack_id"]
                if parent
                else ("merged-" + hashlib.sha256(str(keys).encode()).hexdigest()[:12])
            )
            title = parent["title"] if parent else "Merged Knowledge Packs"
            if any(
                "wikipedia" not in m["pack_id"].casefold()
                and "wikimedia" not in m.get("source_dataset", "").casefold()
                for _, m, _ in entries
            ):
                raise ValueError("All Wikipedia merges require Wikipedia packs only.")
            snapshots = {(m.get("snapshot_id"), m.get("source_revision")) for _, m, _ in entries}
            if len(snapshots) != 1:
                raise ValueError("Align Wikipedia snapshots before merging.")
            target = self.folder / "merge.building"
            target.unlink(missing_ok=True)
            index = KnowledgeStore(target)
            try:
                for n, (path, _, _) in enumerate(entries):
                    if progress:
                        progress("Merging", n, len(entries), title)
                    with closing(index._connect()) as db:
                        db.execute("ATTACH DATABASE ? AS incoming", (str(path),))
                        if db.execute(
                            "SELECT 1 FROM incoming.articles AS src JOIN main.articles AS dst "
                            "USING(source) WHERE src.hash IS NOT dst.hash LIMIT 1"
                        ).fetchone():
                            raise ValueError(
                                "Conflicting article versions; align snapshots before merging."
                            )
                        with db:
                            for table in (
                                "packs",
                                "articles",
                                "memberships",
                                "pack_aliases",
                                "archive",
                                "prepared",
                                "table_context",
                            ):
                                db.execute(
                                    f"INSERT OR IGNORE INTO {table} SELECT * FROM incoming.{table}"
                                )
                with closing(index._connect()) as db, db:
                    db.execute("UPDATE packs SET enabled=1")
                    index._sync(db)
                set_index_metadata(target, pack_id, title, keys)
                with closing(_connect_index(target)) as db, db:
                    manifest = index_manifest(db)
                    manifest["constituent_titles"] = {
                        key: json.loads(raw)["title"]
                        for key, raw in db.execute("SELECT id,manifest FROM packs")
                    }
                    for _, item, _ in entries:
                        manifest["constituent_titles"].update(item.get("constituent_titles", {}))
                        if not item.get("contains"):
                            manifest["constituent_titles"][item["pack_id"]] = item["title"]
                    manifest["constituent_titles"] = {
                        k: manifest["constituent_titles"].get(k, k) for k in keys
                    }
                    if parent:
                        # Local output has its own checksum; only copy publishing identity.
                        for field in ("release_sequence", "description"):
                            if field in parent:
                                manifest[field] = parent[field]
                    db.execute(
                        "UPDATE index_state SET value=? WHERE key='pack_manifest'",
                        (json.dumps(manifest),),
                    )
                _finish_index(target)
                with closing(_connect_index(target)) as db:
                    check_index(db)
                    if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                        raise ValueError(
                            "Merged index failed validation; original packs were kept."
                        )
                destination = self.folder / (
                    _index_name(manifest) if parent else f"{pack_id}-indexed.sqlite"
                )
                with self.lock:
                    # Publish before removing anything: interruption can leave duplicates,
                    # but never removes the only validated copy of an article.
                    existing = next(
                        (p for p, m, _ in self._entries() if m["pack_id"] == pack_id), None
                    )
                    if destination.exists() and destination != existing:
                        raise ValueError("Merged pack filename is already in use.")
                    target.replace(destination)
                    state_path = self.folder / "packs.json"
                    state = json.loads(state_path.read_text()) if state_path.exists() else {}
                    state[pack_id] = True
                    for _, item, _ in entries:
                        if item["pack_id"] != pack_id:
                            state.pop(item["pack_id"], None)
                    temporary = state_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(state))
                    temporary.replace(state_path)
                    for path, _, _ in entries:
                        if path != destination:
                            with KnowledgeStore(path)._access(exclusive=True):
                                path.unlink()
                return title
            finally:
                for suffix in ("", "-wal", "-shm"):
                    Path(str(target) + suffix).unlink(missing_ok=True)
                _lock_path(target).unlink(missing_ok=True)
