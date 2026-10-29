"""Incremental local FTS5 retrieval; call retrieve from a worker thread."""

import ast
import codecs
import math
import re
import sqlite3
from collections import defaultdict
from contextlib import closing, suppress
from pathlib import Path


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


def _search(connection, expression, limit, sources=()):
    if not expression:
        return []
    scope = " AND source IN (" + ",".join("?" for _ in sources) + ")" if sources else ""
    return connection.execute(
        "SELECT rowid, source, section, body, start_line, end_line FROM chunks "
        "WHERE chunks MATCH ?" + scope + " ORDER BY bm25(chunks, 4.0, 2.0, 1.0, 0.5) LIMIT ?",
        (expression, *sources, limit),
    ).fetchall()


def _source_scope(connection, paths, terms, frequencies, total, query):
    """Prefer explicit filenames, then distinctive filename prefixes such as project names."""
    explicit = {
        path for path in paths if path.casefold() in terms or Path(path).name.casefold() in terms
    }
    if explicit:
        return (
            tuple(sorted(explicit)) if len(explicit) <= 64 else (),
            set(),
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
    if 1 < len(parts) < 17 and sum(len(part[3]) + 2 for part in parts) <= limit:
        body, previous_end = "", 0
        for part in parts:
            text = part[3]
            overlap = 0
            if part[4] <= previous_end:
                overlap = next(
                    (
                        length
                        for length in range(min(len(body), len(text)), 15, -1)
                        if body.endswith(text[:length])
                    ),
                    0,
                )
            separator = "" if overlap or part[4] == previous_end else "\n\n"
            body += (separator if body else "") + text[overlap:]
            previous_end = part[5]
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
    if not terms:
        return []
    total = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
    if not total:
        return []
    frequencies = {
        term: connection.execute(
            "SELECT count(*) FROM chunks WHERE chunks MATCH ?",
            (_quote(term),),
        ).fetchone()[0]
        for term in dict.fromkeys([*terms, *extra])
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
    # Fuse genuinely different queries, not a collection of correlated automatic bigrams.
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
    matches, file_matches = defaultdict(set), defaultdict(set)
    rowids = ",".join("?" for _ in candidates)
    for term in weights:
        for (rowid,) in connection.execute(
            f"SELECT rowid FROM chunks WHERE chunks MATCH ? AND rowid IN ({rowids})",
            (_quote(term), *candidates),
        ):
            matches[rowid].add(term)
            file_matches[candidates[rowid][1]].add(term)
    original = set(terms)
    content_terms = original - anchors or original
    denominator = sum(weights[term] for term in content_terms)
    filenames = {name for path in paths for name in (path.casefold(), Path(path).name.casefold())}
    technical = {
        term.casefold()
        for term in _TERM.findall(query)
        if re.search(r"[_./:+#\d-]|[a-z][A-Z]", term) and term.casefold() not in filenames
    }
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
        expansion_hit = any(
            frequencies[term] <= max(1, total // 20) for term in matches[rowid] & set(extra)
        )
        if source_coverage < 0.5 and not identifiers and not expansion_hit:
            continue
        if len(terms) == 1 and technical and not identifiers and not expansion_hit:
            continue
        if coverage < 0.15 and not identifiers and not expansion_hit:
            continue
        tokens = _TERM.findall(body.casefold())
        diversity = len(set(tokens)) / max(1, len(tokens))
        score = coverage + 0.15 * ranks[rowid] / best_rank + 0.25 * bool(identifiers)
        score *= 0.5 + 0.5 * diversity
        scores[rowid] = score
        # Expanded vocabulary must also count as new evidence during result selection.
        covered_terms[rowid] = found | (matches[rowid] & set(extra) if expansion_hit else set())
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
        context_key = (source, candidates[rowid][2])
        if context_key not in contexts:
            contexts[context_key] = _document_context(connection, candidates[rowid], context_limit)
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
    reference = re.search(
        r"\b(?:that|this|the) (?:file|document|value)\b|\b(?:check again|current value|and now)\b",
        query,
        re.IGNORECASE,
    )
    generic = {
        "changed",
        "change",
        "updated",
        "update",
        "value",
        "values",
        "file",
        "files",
        "document",
        "documents",
        "check",
        "again",
        "current",
        "latest",
        "now",
        "give",
        "read",
    }
    if reference and not (set(_terms(query)) - generic):
        for previous in reversed(previous_queries):
            if set(_terms(previous)) - generic:
                return previous + "\n" + query
    return query


def retrieve(query, folder, settings, expand=None, stopped=lambda: False, previous_queries=()):
    """Update changed files and retrieve passages using a thread-local SQLite connection."""
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
