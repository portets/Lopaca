"""Incremental local FTS5 retrieval; call retrieve from a worker thread."""

import ast
import codecs
import re
import sqlite3
from collections import defaultdict
from contextlib import closing, suppress


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
    "work works".split()
)
_TERM = re.compile(r"(?:--?)?\w+(?:[./:+#-]+\w+)*[+#]*", re.UNICODE)
_CODE_HEADING = re.compile(
    r"^(\s*)(?:(?:async\s+)?(?:def|function|fn|func)|class|struct|interface|enum)\s+([\w.]+)"
)


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
    if filename.lower().endswith(".py"):
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
        elif not markdown and not filename.lower().endswith(".py"):
            definition = _CODE_HEADING.match(line)
            if definition:
                indent = len(definition[1].expandtabs(4))
                while code_scope and code_scope[-1][0] >= indent:
                    code_scope.pop()
                code_scope.append((indent, definition[2]))
                starts[index] = " > ".join(name for _, name in code_scope)
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
            body = "".join(line for _, line in lines[start:end]).strip()
            if body:
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
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
            source, section, body, keywords, start_line UNINDEXED, end_line UNINDEXED,
            tokenize='porter unicode61'
        );
    """)
    signature = f"1:{settings.chunk_chars}:{settings.chunk_overlap}"
    state = connection.execute("SELECT value FROM index_state WHERE key = 'chunking'").fetchone()
    if state != (signature,):
        with connection:
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM files")
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
            connection.execute("DELETE FROM chunks WHERE source = ?", (name,))
            if text:
                for section, first, last, body in _chunks(
                    text, name, settings.chunk_chars, settings.chunk_overlap
                ):
                    if stopped():
                        raise InterruptedError("Document indexing stopped")
                    # Add split camelCase/snake_case terms alongside the original source text.
                    words = re.sub(r"([a-z\d])([A-Z])", r"\1 \2", f"{name} {section} {body}")
                    keywords = " ".join(re.findall(r"[^\W_]+", words))
                    connection.execute(
                        "INSERT INTO chunks "
                        "(source, section, body, keywords, start_line, end_line) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (name, section, body, keywords, first, last),
                    )
            connection.execute("INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?)", (name, *stamp))
    with connection:
        for name in cached.keys() - seen:
            connection.execute("DELETE FROM chunks WHERE source = ?", (name,))
            connection.execute("DELETE FROM files WHERE path = ?", (name,))


def _search(connection, expression, limit):
    if not expression:
        return []
    return connection.execute(
        "SELECT rowid, source, section, body, start_line, end_line FROM chunks "
        "WHERE chunks MATCH ? ORDER BY bm25(chunks, 4.0, 2.0, 1.0, 0.5) LIMIT ?",
        (expression, limit),
    ).fetchall()


def _rank(connection, query, expansion, settings):
    terms, extra = _terms(query), _terms(expansion)
    if not terms:
        return []
    quoted = [_quote(term) for term in terms]
    phrases = list(dict.fromkeys(re.findall(r'`([^`\n]+)`|"([^"\n]+)"', query)))
    phrases = [next(part for part in pair if part).strip()[:160] for pair in phrases]
    phrases += [" ".join(terms[index : index + 2]) for index in range(len(terms) - 1)]
    phrases = [phrase for phrase in phrases if _terms(phrase)][:12]
    expressions = [
        (1.0, " OR ".join(quoted)),
        (1.0, " AND ".join(quoted) if len(terms) > 1 else ""),
        (1.0, " OR ".join(_quote(phrase) for phrase in phrases)),
        (0.8, "{source section} : (" + " OR ".join(quoted) + ")"),
        (0.5, " OR ".join(_quote(term) for term in extra)),
    ]
    candidates, scores = {}, defaultdict(float)
    for weight, expression in expressions:
        for rank, row in enumerate(
            _search(connection, expression, max(1, settings.rag_candidates)), 1
        ):
            candidates[row[0]] = row
            scores[row[0]] += weight / (60 + rank)
    if not candidates:
        return []
    # Measure original-query coverage with the same tokenizer/stemmer as the index.
    coverage, expanded_coverage = defaultdict(int), defaultdict(int)
    rowids = ",".join("?" for _ in candidates)
    for words, counts in ((terms, coverage), (extra, expanded_coverage)):
        for term in words:
            for (rowid,) in connection.execute(
                f"SELECT rowid FROM chunks WHERE chunks MATCH ? AND rowid IN ({rowids})",
                (_quote(term), *candidates),
            ):
                counts[rowid] += 1
    exact = []
    technical = {
        term.casefold()
        for term in _TERM.findall(query)
        if re.search(r"[_./:+#\d-]|[a-z][A-Z]", term)
    }
    for rowid, source, section, body, _first, _last in candidates.values():
        haystack = f"{source}\n{section}\n{body}".casefold()
        identifiers = sum(
            bool(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", haystack))
            for term in technical
        )
        phrase_hits = sum(phrase.casefold() in haystack for phrase in phrases)
        if identifiers or phrase_hits:
            exact.append((identifiers, phrase_hits, coverage[rowid], rowid))
    for rank, match in enumerate(sorted(exact, reverse=True), 1):
        scores[match[-1]] += 1 / (60 + rank)
    exact_ids = {row[-1] for row in exact}
    minimum = min(2, len(terms))
    selected, spans = [], defaultdict(list)
    for rowid in sorted(scores, key=lambda item: (-scores[item], item)):
        # RRF ranks candidates; it cannot decide whether a weak match is relevant.
        if coverage[rowid] < minimum and rowid not in exact_ids and expanded_coverage[rowid] < 2:
            continue
        if len(terms) == 1 and technical and rowid not in exact_ids:
            continue  # Tokenizer punctuation splitting alone isn't an identifier match.
        _, source, section, body, first, last = candidates[rowid]
        # Don't spend the prompt budget on almost identical overlapping chunks.
        if any(
            max(0, min(last, end) - max(first, start) + 1) > 0.6 * max(1, last - first + 1)
            for start, end in spans[source]
        ):
            continue
        reference = f"documents/{source}, lines {first}-{last}"
        if section:
            reference += f", section: {section}"
        selected.append((reference, body))
        spans[source].append((first, last))
        if len(selected) >= settings.rag_results:
            break
    return selected


def retrieve(query, folder, settings, expand=None, stopped=lambda: False):
    """Update changed files and retrieve passages using a thread-local SQLite connection."""
    if (
        not folder.is_dir()
        or not _terms(query)
        or settings.rag_results <= 0
        or settings.max_context_chars <= 0
    ):
        return []
    with closing(sqlite3.connect(folder.parent / "documents.sqlite", timeout=5)) as connection:
        connection.set_progress_handler(lambda: int(stopped()), 1000)
        _index(connection, folder, settings, stopped)
        if stopped():
            return []
        expansion = ""
        if expand is not None and connection.execute("SELECT rowid FROM chunks LIMIT 1").fetchone():
            # Expansion is optional; keep the original query if the request fails.
            with suppress(Exception):
                expansion = expand(query)[:1024]
        if stopped():
            return []
        return _rank(connection, query, expansion, settings)


def format_context(retrieved, max_chars):
    """Format source excerpts within the document-context character budget."""
    excerpts, remaining = [], max(0, max_chars)
    for source_name, text in retrieved:
        header = f"[Source: {source_name}]\n"
        separator = "\n\n---\n\n" if excerpts else ""
        available = remaining - len(separator) - len(header)
        if available <= 0:
            break
        excerpt = separator + header + text[:available]
        excerpts.append(excerpt)
        remaining -= len(excerpt)
    return "".join(excerpts)
