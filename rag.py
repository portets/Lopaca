"""Incremental local FTS5 retrieval; call retrieve from a worker thread."""

import ast
import codecs
import hashlib
import logging
import math
import queue
import re
import sqlite3
import threading
import time
from collections import defaultdict
from contextlib import closing, suppress
from pathlib import Path


class DocumentIndexer:
    """Serialize background indexing, with at most one queued check per folder."""

    def __init__(self, settings, generate_questions, model_lock, on_processing):
        self.settings = settings
        self.enabled = settings.rag_model_indexing
        self._generate_questions = generate_questions
        self._model_lock = model_lock
        self._on_processing = on_processing
        self._retry_after = 0
        self._hint_version = (
            "questions-v1:"
            + hashlib.sha256(f"{settings.server_url}:{settings.model}".encode()).hexdigest()
        )
        self._queue = queue.Queue()
        self._pending = {}
        self._active = None
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def schedule(self, folder):
        with self._lock:
            if folder in self._pending:
                return
            ready = threading.Event()
            self._pending[folder] = ready
            self._queue.put((folder, ready))

    def wait(self, folder, stopped):
        """Wait for this chat's already-scheduled pass, without starting another scan."""
        with self._lock:
            ready = self._pending.get(folder)
            if ready is None and self._active and self._active[0] == folder:
                ready = self._active[1]
        while ready is not None and not ready.wait(0.1):
            if stopped():
                break

    def _run(self):
        while True:
            folder, ready = self._queue.get()
            # Allow a follow-up check if files change during this indexing pass.
            with self._lock:
                self._pending.pop(folder)
                self._active = folder, ready
            try:
                if folder.is_dir():
                    with closing(
                        sqlite3.connect(folder.parent / "documents.sqlite", timeout=5)
                    ) as connection:
                        changed = _index(connection, folder, self.settings, lambda: False)
                        version = self._hint_version if self.enabled else "off"
                        state = connection.execute(
                            "SELECT value FROM index_state WHERE key = 'model_indexing'"
                        ).fetchone()
                        if changed or (
                            state != (version,)
                            and (not self.enabled or time.monotonic() >= self._retry_after)
                        ):
                            self._update_hints(connection, folder)
            except Exception as exc:
                self._retry_after = time.monotonic() + self.settings.rag_indexing_retry_interval
                logging.warning("Could not index documents in %s: %s", folder, exc)
            finally:
                with self._lock:
                    self._active = None
                    ready.set()
                self._queue.task_done()

    def _update_hints(self, connection, folder):
        """Apply cached search text; generate missing questions outside write transactions."""
        enabled = self.enabled
        version = self._hint_version if enabled else "off"
        cached = dict(connection.execute("SELECT key, questions FROM document_hints"))
        stamps = {
            name: (mtime, ctime, size)
            for name, mtime, ctime, size in connection.execute("SELECT * FROM files")
        }
        rows = connection.execute(
            "SELECT rowid, source, section, body, keywords FROM chunks"
        ).fetchall()
        jobs, keys = [], set()
        with connection:
            for rowid, source, section, body, keywords in rows:
                key = hashlib.sha256(
                    f"{self._hint_version}\0{source}\0{section}\0{body}".encode()
                ).hexdigest()
                keys.add(key)
                search_text = _keywords(source, section, body)
                if enabled and key in cached:
                    search_text += " " + cached[key]
                elif enabled and any(
                    line.strip() and not line.lstrip().startswith("#") for line in body.splitlines()
                ):
                    jobs.append((rowid, source, section, body, key, search_text))
                if search_text != keywords:
                    connection.execute(
                        "UPDATE chunks SET keywords = ? WHERE rowid = ?", (search_text, rowid)
                    )
            for key in cached.keys() - keys:
                connection.execute("DELETE FROM document_hints WHERE key = ?", (key,))
            connection.execute(
                "INSERT OR REPLACE INTO index_state VALUES ('model_indexing', ?)",
                (version + ":pending" if jobs else version,),
            )
        if not jobs or time.monotonic() < self._retry_after:
            return
        # Chat requests wait for indexing before acquiring this same lock.
        with self._model_lock:
            if not self.enabled:
                return
            self._on_processing(True)
            written = 0
            try:
                for job, questions in self._question_batches(jobs, folder, stamps):
                    if not self.enabled:
                        return
                    rowid, source, section, body, key, search_text = job
                    if not _unchanged(folder / source, stamps[source]):
                        continue
                    with connection:
                        # Row IDs can be reused if another process reindexes this chat.
                        result = connection.execute(
                            "UPDATE chunks SET keywords = ? "
                            "WHERE rowid = ? AND source = ? AND section = ? AND body = ?",
                            (search_text + " " + questions, rowid, source, section, body),
                        )
                        if result.rowcount:
                            connection.execute(
                                "INSERT OR REPLACE INTO document_hints VALUES (?, ?)",
                                (key, questions),
                            )
                            written += 1
                if written == len(jobs):
                    with connection:
                        connection.execute(
                            "INSERT OR REPLACE INTO index_state VALUES ('model_indexing', ?)",
                            (version,),
                        )
            finally:
                self._on_processing(False)
                # Model work can take minutes; refresh edits/deletions made during it
                # before releasing a waiting chat request to read source passages.
                _index(connection, folder, self.settings, lambda: False)

    def _question_batches(self, jobs, folder, stamps):
        """Bound request sizes without splitting or merging the indexed passages."""
        batch, size = [], 0
        for job in jobs:
            _, source, section, body, _, _ = job
            if not _unchanged(folder / source, stamps[source]):
                continue
            length = len(source) + len(section) + len(body)
            if batch and (
                len(batch) >= max(1, self.settings.rag_indexing_batch_size)
                or size + length > max(1, self.settings.rag_indexing_batch_chars)
            ):
                yield from self._generate_batch(batch)
                batch, size = [], 0
            batch.append(job)
            size += length
        if batch:
            yield from self._generate_batch(batch)

    def _generate_batch(self, batch):
        if not self.enabled:
            return
        passages = [
            {"id": rowid, "source": source, "section": section, "body": body}
            for rowid, source, section, body, _key, _search_text in batch
        ]
        try:
            questions = self._generate_questions(passages)
        except ValueError:
            # Smaller requests help models that struggle to keep several IDs straight.
            if len(batch) == 1:
                raise
            middle = len(batch) // 2
            yield from self._generate_batch(batch[:middle])
            yield from self._generate_batch(batch[middle:])
        else:
            for job in batch:
                yield job, questions[job[0]]


def _unchanged(path, stamp):
    try:
        stat = path.stat()
        return not path.is_symlink() and (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size) == stamp
    except OSError:
        return False


def _keywords(source, section, body):
    """Include split camelCase/snake_case terms alongside the original source text."""
    words = re.sub(r"([a-z\d])([A-Z])", r"\1 \2", f"{source} {section} {body}")
    return " ".join(re.findall(r"[^\W_]+", words))


def read_document(path):
    """Read common text encodings, skipping likely binary and unsupported documents."""
    try:
        with path.open("rb") as source:
            sample = source.read(8192)
            if sample.startswith(b"%PDF-"):
                return None
            if sample.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
                encodings = ("utf-32",)
            elif sample.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
                encodings = ("utf-16",)
            else:
                controls = sum(byte < 32 and byte not in b"\t\n\r\v\f\b\x1b" for byte in sample)
                if b"\0" in sample or controls > len(sample) * 0.05:
                    return None
                encodings = ("utf-8-sig", "cp1252", "latin-1")
            data = sample + source.read()
    except OSError:
        return None
    for encoding in encodings:
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        sample_text = text[:8192]
        controls = sum(
            not (char.isprintable() or char.isspace() or char in "\b\x1b") for char in sample_text
        )
        if "\0" in text or controls > len(sample_text) * 0.05:
            return None
        return text.replace("\r\n", "\n").replace("\r", "\n")
    return None


# Remove conversational filler before constructing literal, safely quoted FTS expressions.
_STOP_WORDS = frozenset(
    "a an and are as at be been but by can could "  # noqa: SIM905 -- Compact stop-word list.
    "did do does explain for from had has have "
    "hello help hi how i if in into is it its me my of on or our please should show so some "
    "tell than thank thanks that the their them there these they this those to use using was "
    "we were what when where which who why will with would you your about any also "
    "work works according give find current latest document documents file files "
    "long many much period value values before after called named".split()
)
_TERM = re.compile(r"(?:--?)?\w+(?:[./:+#-]+\w+)*[+#]*", re.UNICODE)
_CODE_HEADING = re.compile(
    r"^(\s*)(?:(?:export|default|pub(?:\([^)]*\))?|public|private|protected|static|async)\s+)*"
    r"(?:def|function|fn|func|class|struct|interface|enum)\s+([\w.]+)"
)
_CODE_SUFFIXES = tuple(
    ".py .pyi .js .jsx .mjs .cjs .ts .tsx .c .h .cc .cpp .hpp "  # noqa: SIM905 -- Compact syntax-family list.
    ".rs .go .java .cs .sh .bash .zsh".split()
)


def _comment_only(body, source):
    """Recognize standalone comments without treating C directives as comments."""
    if source.lower().endswith((".py", ".pyi", ".sh", ".bash", ".zsh")):
        prefix = "#"
    else:
        prefix = "//"
        body = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
    return all(not line.strip() or line.lstrip().startswith(prefix) for line in body.splitlines())


def _terms(query):
    return list(
        dict.fromkeys(
            term
            for term in _TERM.findall(query.casefold())
            if term not in _STOP_WORDS and (len(term) > 1 or term.isdigit())
        )
    )[:24]


def _quote(text):
    return '"' + text.replace('"', '""') + '"'


def _sections(text, filename):
    """Find Markdown sections and code definitions with standard-library tools."""
    lines = text.splitlines(keepends=True)
    starts = {0: ""}
    if filename.lower().endswith((".py", ".pyi")):
        try:
            tree = ast.parse(text)
        except SyntaxError, ValueError, RecursionError:
            tree = None

        def visit(node, parents=()):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                parents = (*parents, node.name)
                start = min([node.lineno, *(item.lineno for item in node.decorator_list)]) - 1
                starts[start] = " > ".join(parents)
                if node.end_lineno < len(lines):
                    starts.setdefault(node.end_lineno, " > ".join(parents[:-1]))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return  # Keep nested callbacks with the function that uses them.
            for child in ast.iter_child_nodes(node):
                visit(child, parents)

        if tree is not None:
            visit(tree)
    headings, code_scope, fence = [], [], ""
    markdown = filename.lower().endswith((".md", ".markdown", ".mdown"))
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        marker = re.match(r"(`{3,}|~{3,})", stripped)
        if marker:
            if not fence:
                fence = marker[1]
            elif stripped.startswith(fence):
                fence = ""
            continue
        if fence:
            continue
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line) if markdown else None
        setext = (
            markdown
            and index + 1 < len(lines)
            and line.strip()
            and re.fullmatch(r"\s*(?:=+|-+)\s*", lines[index + 1])
        )
        if heading or setext:
            level = len(heading[1]) if heading else (1 if "=" in lines[index + 1] else 2)
            title = heading[2] if heading else line.strip()
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, title))
            starts[index] = " > ".join(title for _, title in headings)
        elif not markdown and not filename.lower().endswith((".py", ".pyi")):
            definition = _CODE_HEADING.match(line)
            if definition:
                indent = len(definition[1].expandtabs(4))
                while code_scope and code_scope[-1][0] >= indent:
                    code_scope.pop()
                if filename.lower().endswith(_CODE_SUFFIXES) and any(
                    not container for _, _, container in code_scope
                ):
                    continue
                container = bool(re.search(r"\b(?:class|struct|interface|enum)\b", definition[0]))
                code_scope.append((indent, definition[2], container))
                starts[index] = " > ".join(name for _, name, _ in code_scope)
    boundaries = sorted(starts)
    for index, start in enumerate(boundaries):
        end = boundaries[index + 1] if index + 1 < len(boundaries) else len(lines)
        if end > start:
            yield starts[start], [(number + 1, lines[number]) for number in range(start, end)]


def _chunks(text, filename, size, overlap):
    """Prefer paragraph and section boundaries; split oversized blocks by line."""
    size = max(256, size)
    overlap = max(0, min(overlap, size // 3))
    for section, section_lines in _sections(text, filename):
        # Bound even a minified file's lines, retaining its original line numbers.
        lines = [
            (number, line[offset : offset + size])
            for number, line in section_lines
            for offset in range(0, len(line), size)
        ]
        start = 0
        while start < len(lines):
            end, length, paragraph_end = start, 0, None
            while end < len(lines) and length + len(lines[end][1]) <= size:
                length += len(lines[end][1])
                end += 1
                if not lines[end - 1][1].strip() and length >= size // 2:
                    paragraph_end = end
            if end < len(lines) and paragraph_end is not None:
                end = paragraph_end
            end = max(start + 1, end)
            body = "".join(line for _, line in lines[start:end])
            if body.strip():
                yield section, lines[start][0], lines[end - 1][0], body
            if end == len(lines):
                break
            next_start, length = end, 0
            while next_start > start + 1 and length + len(lines[next_start - 1][1]) <= overlap:
                next_start -= 1
                length += len(lines[next_start][1])
            start = next_start


def _index(connection, folder, settings, stopped):
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY, mtime_ns INTEGER, ctime_ns INTEGER, size INTEGER
        );
        CREATE TABLE IF NOT EXISTS index_state (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS document_hints (key TEXT PRIMARY KEY, questions TEXT);
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
            source, section, body, keywords, start_line UNINDEXED, end_line UNINDEXED,
            tokenize='porter unicode61'
        );
    """)
    signature = f"4:{settings.chunk_chars}:{settings.chunk_overlap}"
    state = connection.execute("SELECT value FROM index_state WHERE key = 'chunking'").fetchone()
    changed = state != (signature,)
    if state != (signature,):
        with connection:
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM files")
            connection.execute("DELETE FROM index_state WHERE key = 'model_indexing'")
            connection.execute(
                "INSERT OR REPLACE INTO index_state VALUES ('chunking', ?)", (signature,)
            )
    cached = {
        path: (mtime, ctime, size)
        for path, mtime, ctime, size in connection.execute(
            "SELECT path, mtime_ns, ctime_ns, size FROM files"
        )
    }
    seen = set()
    for path in folder.rglob("*"):
        if stopped():
            raise InterruptedError("Document indexing stopped")
        if path.is_symlink() or not path.is_file():
            continue
        name = path.relative_to(folder).as_posix()
        seen.add(name)
        try:
            stat = path.stat()
            stamp = stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size
            if cached.get(name) == stamp:
                continue
            text = read_document(path)
            after = path.stat()
            if stamp != (after.st_mtime_ns, after.st_ctime_ns, after.st_size):
                continue  # Retry a file being edited on the next request.
        except OSError:
            continue
        with connection:
            changed = True
            connection.execute("DELETE FROM index_state WHERE key = 'model_indexing'")
            connection.execute("DELETE FROM chunks WHERE source = ?", (name,))
            if text:
                for section, first, last, body in _chunks(
                    text, name, settings.chunk_chars, settings.chunk_overlap
                ):
                    if stopped():
                        raise InterruptedError("Document indexing stopped")
                    connection.execute(
                        "INSERT INTO chunks "
                        "(source, section, body, keywords, start_line, end_line) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (name, section, body, _keywords(name, section, body), first, last),
                    )
            connection.execute("INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?)", (name, *stamp))
    with connection:
        for name in cached.keys() - seen:
            changed = True
            connection.execute("DELETE FROM index_state WHERE key = 'model_indexing'")
            connection.execute("DELETE FROM chunks WHERE source = ?", (name,))
            connection.execute("DELETE FROM files WHERE path = ?", (name,))
    return changed


def _search(connection, expression, limit, sources=()):
    if not expression:
        return []
    scope = " AND source IN (" + ",".join("?" for _ in sources) + ")" if sources else ""
    return connection.execute(
        "SELECT rowid, source, section, body, start_line, end_line FROM chunks "
        "WHERE chunks MATCH ?" + scope + " ORDER BY bm25(chunks, 4.0, 2.0, 1.0, 0.5) LIMIT ?",
        (expression, *sources, limit),
    ).fetchall()


def _source_scope(_connection, paths, terms, _frequencies, _total, query):
    """Restrict sources only when the query names an actual filename."""
    explicit = {
        path for path in paths if path.casefold() in terms or Path(path).name.casefold() in terms
    }
    if explicit:
        return (
            tuple(sorted(explicit)) if len(explicit) <= 64 else (),
            set(terms)
            & {name for path in explicit for name in (path.casefold(), Path(path).name.casefold())},
        )
    suffixes = {Path(path).suffix.casefold() for path in paths} - {""}
    if any(
        Path(term).suffix in suffixes
        and re.search(
            r"\b(?:in|from|according to)\s+[`\"]?" + re.escape(term), query, re.IGNORECASE
        )
        for term in terms
    ):
        return None, set()  # A specifically requested local filename does not exist.
    return (), set()


def _document_context(connection, row, limit):
    """Keep small source documents together instead of losing facts across tiny sections."""
    source = row[1]
    parts = connection.execute(
        "SELECT rowid,source,section,body,start_line,end_line FROM chunks "
        "WHERE source=? ORDER BY start_line,rowid LIMIT 17",
        (source,),
    ).fetchall()
    if not (1 < len(parts) < 17 and sum(len(part[3]) + 2 for part in parts) <= limit):
        parts = (
            connection.execute(
                "SELECT rowid,source,section,body,start_line,end_line FROM chunks "
                "WHERE source=? AND section=? ORDER BY start_line,rowid LIMIT 17",
                (source, row[2]),
            ).fetchall()
            if row[2]
            else []
        )
    if not source.lower().endswith(_CODE_SUFFIXES) and not (
        1 < len(parts) < 17 and sum(len(part[3]) + 2 for part in parts) <= limit
    ):
        parts = connection.execute(
            "SELECT rowid,source,section,body,start_line,end_line FROM chunks "
            "WHERE source=? AND section=? ORDER BY abs(start_line-?),abs(rowid-?) LIMIT 3",
            (source, row[2], row[4], row[0]),
        ).fetchall()
        while len(parts) > 1 and sum(len(part[3]) + 2 for part in parts) > limit:
            parts.pop()  # Farthest neighbor; keep the matching chunk.
        parts.sort(key=lambda part: (part[4], part[0]))
    if 1 < len(parts) < 17 and sum(len(part[3]) + 2 for part in parts) <= limit:
        lines = {}
        for part in parts:
            # Wrapped/minified lines cannot be reconstructed by line number.
            if len(part[3].splitlines()) != part[5] - part[4] + 1:
                return None
            lines.update(enumerate(part[3].splitlines(), part[4]))
        first, last = min(lines), max(lines)
        body = "\n".join(lines.get(number, "") for number in range(first, last + 1))
        return (
            row[0],
            source,
            "",
            body,
            min(part[4] for part in parts),
            max(part[5] for part in parts),
        )
    return None


def _rank(connection, query, expansion, settings):
    terms = _terms(query)
    extra = [term for term in _terms(expansion) if term not in terms]
    expanded_terms = set(extra)
    if not terms:
        return []
    total = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
    if not total:
        return []
    hint_state = connection.execute(
        "SELECT value FROM index_state WHERE key = 'model_indexing'"
    ).fetchone()
    evidence_field = "{body keywords}:" if hint_state and hint_state != ("off",) else "body:"
    frequencies = {
        term: connection.execute(
            "SELECT count(*) FROM chunks WHERE chunks MATCH ?",
            (_quote(term),),
        ).fetchone()[0]
        for term in (*terms, *extra)
    }
    weights = {
        term: math.log1p((total + 0.5) / (frequencies[term] + 0.5)) if frequencies[term] else 1.0
        for term in frequencies
    }
    paths = [row[0] for row in connection.execute("SELECT path FROM files")]
    sources, anchors = _source_scope(connection, paths, terms, frequencies, total, query)
    if sources is None:
        return []
    phrases = [
        next(part for part in pair if part).strip()[:160]
        for pair in re.findall(r'`([^`\n]+)`|"([^"\n]+)"', query)
    ]
    phrases = list(dict.fromkeys(phrase for phrase in phrases if _terms(phrase)))[:8]
    # Fuse the original query, explicit phrases, and optional new vocabulary.
    expressions = [
        (1.0, " OR ".join(_quote(term) for term in terms)),
        (0.4, " OR ".join(_quote(phrase) for phrase in phrases)),
        (0.35, " OR ".join(_quote(term) for term in extra)),
    ]
    candidates, ranks = {}, defaultdict(float)
    for weight, expression in expressions:
        for rank, row in enumerate(
            _search(connection, expression, max(1, settings.rag_candidates), sources), 1
        ):
            candidates[row[0]] = row
            ranks[row[0]] += weight / (60 + rank)
    if not candidates:
        return []
    matches, file_matches, body_matches = defaultdict(set), defaultdict(set), defaultdict(set)
    rowids = ",".join("?" for _ in candidates)
    for term in weights:
        for (rowid,) in connection.execute(
            f"SELECT rowid FROM chunks WHERE chunks MATCH ? AND rowid IN ({rowids})",
            (_quote(term), *candidates),
        ):
            matches[rowid].add(term)
            file_matches[candidates[rowid][1]].add(term)
        if term in terms and frequencies[term]:
            for (rowid,) in connection.execute(
                f"SELECT rowid FROM chunks WHERE chunks MATCH ? AND rowid IN ({rowids})",
                (evidence_field + _quote(term), *candidates),
            ):
                body_matches[rowid].add(term)
    original = set(terms)
    content_terms = original - anchors or original
    denominator = sum(weights[term] for term in content_terms)
    technical = {
        term.casefold()
        for term in _TERM.findall(query)
        if re.search(r"[_./:+#\d-]|[a-z][A-Z]", term)
    }
    # A resolved, existing filename is evidence even when a follow-up has no search terms.
    technical.update(
        set(extra) & {name.casefold() for path in paths for name in (path, Path(path).name)}
    )
    expansion_weight = sum(weights[term] for term in extra)
    best_rank = max(ranks.values())
    scores, covered_terms = {}, {}
    for rowid, source, section, body, _first, _last in candidates.values():
        found = matches[rowid] & content_terms
        coverage = sum(weights[term] for term in found) / denominator
        source_coverage = (
            sum(weights[term] for term in file_matches[source] & content_terms) / denominator
        )
        haystack = f"{source}\n{section}\n{body}".casefold()
        identifiers = sum(
            bool(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", haystack))
            for term in technical
        )
        expanded_found = matches[rowid] & expanded_terms
        expansion_coverage = sum(weights[term] for term in expanded_found) / max(
            1, expansion_weight
        )
        rare = {term for term in expanded_found if frequencies[term] <= max(1, total // 20)}
        # Require corroboration, or a rare term naming the section itself.
        expansion_hit = (
            expansion_coverage >= 0.2
            and bool(rare)
            and (len(expanded_found) >= 2 or bool(rare & set(_terms(section))))
        )
        # Two body/search-hint terms, including a rare one, can support a paraphrase.
        body_found = body_matches[rowid] & content_terms
        original_hit = len(body_found) >= 2 and any(
            frequencies[term] <= max(1, total // 20) for term in body_found
        )
        if source_coverage < 0.5 and not original_hit and not identifiers and not expansion_hit:
            continue
        if len(terms) == 1 and technical and not identifiers and not expansion_hit:
            continue
        if coverage < 0.15 and not original_hit and not identifiers and not expansion_hit:
            continue
        tokens = _TERM.findall(body.casefold())
        diversity = len(set(tokens)) / max(1, len(tokens))
        score = (
            max(coverage, expansion_coverage)
            + 0.15 * ranks[rowid] / best_rank
            + 0.25 * bool(identifiers)
        )
        score *= 0.5 + 0.5 * diversity
        if source.lower().endswith(_CODE_SUFFIXES) and _comment_only(body, source):
            score *= 0.25
        scores[rowid] = score
        # Expanded vocabulary must also count as new evidence during result selection.
        covered_terms[rowid] = found | (expanded_found if expansion_hit else set())
    selected, used, spans = [], set(), defaultdict(list)
    contexts = {}
    context_limit = min(settings.chunk_chars * 2, settings.max_context_chars // 2)
    while scores and len(selected) < settings.rag_results:
        rowid = max(
            scores,
            key=lambda item: (
                scores[item]
                + 0.5 * sum(weights[term] for term in covered_terms[item] - used) / denominator,
                -item,
            ),
        )
        score = scores.pop(rowid)
        if selected and not (covered_terms[rowid] - used) and score < 0.65:
            continue
        source = candidates[rowid][1]
        is_code = source.lower().endswith(_CODE_SUFFIXES)
        context_key = (source, candidates[rowid][2], None if is_code else rowid)
        if context_key not in contexts:
            limit = (
                min(context_limit * 2, settings.max_context_chars // 2)
                if is_code
                else context_limit
            )
            contexts[context_key] = _document_context(connection, candidates[rowid], limit)
        row = contexts[context_key] or candidates[rowid]
        _, source, section, body, first, last = row
        if any(
            max(0, min(last, end) - max(first, start) + 1) > 0.6 * max(1, last - first + 1)
            for start, end in spans[source]
        ):
            continue
        reference = f"documents/{source}, lines {first}-{last}"
        if section:
            reference += f", section: {section}"
        selected.append((reference, body))
        used.update(
            file_matches[source] & content_terms if contexts[context_key] else covered_terms[rowid]
        )
        spans[source].append((first, last))
    return selected


def _retrieval_query(query, previous_queries):
    """Resolve document follow-ups without pulling old topics into unrelated questions."""
    reference = re.compile(
        r"\b(?:that|this) (?:file|document|value|number|function|class|setting)\b"
        r"|\bthe (?:other|same) (?:one|file|document|value|number)\b"
        r"|\b(?:what|how) about (?:its|their|that|those)\b"
        r"|\b(?:read|check|scan) the (?:file|document) again\b"
        r"|^(?:please )?(?:check again|and now|(?:what is )?the current value)[?.!\s]*$",
        re.IGNORECASE,
    )

    def follows_previous(text):
        # Explicit filenames stand on their own; quoted phrases are not references.
        if re.search(r"\b[\w-]+\.[a-zA-Z][\w]*\b", text):
            return False
        return reference.search(re.sub(r'`[^`]*`|"[^"]*"', "", text).strip()) is not None

    if not follows_previous(query):
        return query
    queries = [query]
    # Follow a short chain of references, stopping at the nearest standalone question.
    for previous in reversed(previous_queries[-3:]):
        if not _terms(previous):
            continue
        queries.insert(0, previous)
        if not follows_previous(previous):
            break
    return "\n".join(queries)


def retrieve(
    query, folder, settings, expand=None, stopped=lambda: False, previous_queries=(), refresh=True
):
    """Retrieve passages; background-indexed callers can skip the folder scan."""
    query = _retrieval_query(query, previous_queries)
    if (
        not folder.is_dir()
        or not _terms(query)
        or settings.rag_results <= 0
        or settings.max_context_chars <= 0
    ):
        return []
    with closing(sqlite3.connect(folder.parent / "documents.sqlite", timeout=5)) as connection:
        connection.set_progress_handler(lambda: int(stopped()), 1000)
        if refresh:
            _index(connection, folder, settings, stopped)
        elif not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunks'"
        ).fetchone():
            return []  # The background worker has not prepared this chat's first index yet.
        if stopped():
            return []
        expansion = ""
        if expand is not None and connection.execute("SELECT rowid FROM chunks LIMIT 1").fetchone():
            # Expansion is optional; keep the original query if the request fails.
            with suppress(Exception):
                expansion = expand(query)[:1024]
        if stopped():
            return []
        # Keep term counts, candidate rows, and excerpts in the same read snapshot.
        # Begin after expansion so its model request cannot hold up indexing commits.
        connection.execute("BEGIN")
        return _rank(connection, query, expansion, settings)


def format_context(retrieved, max_chars):
    """Format source excerpts within the document-context character budget."""
    excerpts, remaining = [], max(0, max_chars)
    for source_name, text in retrieved:
        span = re.fullmatch(r"(.+), lines (\d+)-(\d+)(.*)", source_name)
        if (
            span
            and not span[1].lower().endswith(_CODE_SUFFIXES)
            and len(text.splitlines()) == int(span[3]) - int(span[2]) + 1
        ):
            # Give each prose paragraph an exact citation, including surrounding context.
            paragraphs = []
            for match in re.finditer(r"\S[^\n]*(?:\n(?!\s*\n)[^\n]+)*", text):
                first = int(span[2]) + text.count("\n", 0, match.start())
                last = first + match[0].count("\n")
                paragraphs.append(f"[Source: {span[1]}, lines {first}-{last}{span[4]}]\n{match[0]}")
            text = "\n\n".join(paragraphs)
            source_name = None
        header = f"[Source: {source_name}]\n" if source_name else ""
        separator = "\n\n---\n\n" if excerpts else ""
        available = remaining - len(separator) - len(header)
        if available <= 0:
            break
        excerpt = separator + header + text[:available]
        excerpts.append(excerpt)
        remaining -= len(excerpt)
    return "".join(excerpts)
