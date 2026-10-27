"""Local document retrieval using keyword scoring."""

import re

TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".rst",
    ".py",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".css",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
    ".sh",
    ".java",
    ".rs",
    ".go",
    ".sql",
    ".csv",
    ".log",
    ".xml",
}


def _chunk_documents(folder, size, overlap):
    chunks = []
    for path in folder.rglob("*"):
        if (
            path.name.startswith(".")
            or path.suffix.lower() not in TEXT_SUFFIXES
            or not path.is_file()
        ):
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        source_name = str(path.relative_to(folder))
        for start in range(0, len(content), max(1, size - overlap)):
            text = content[start : start + size].strip()
            if text:
                chunks.append((source_name, text))
    return chunks


def _rank_chunks(query, chunks, limit=5):
    """Rank chunks by length-adjusted keyword counts."""
    terms = [
        (term, 1 + min(len(term), 12) / 12)
        for term in re.findall(r"[\w.-]+", query.lower())
        if len(term) > 2
    ]
    if not terms:
        return []
    scored_chunks = []
    for source_name, text in chunks:
        lower_text = text.lower()
        score = sum(lower_text.count(term) * weight for term, weight in terms)
        if score:
            scored_chunks.append((score / (1 + len(text) / 1800), source_name, text))
    return [
        (source_name, text) for _, source_name, text in sorted(scored_chunks, reverse=True)[:limit]
    ]


def retrieve(query, folder, settings):
    """Return relevant ``(source, text)`` pairs ranked by keyword matches."""
    chunks = _chunk_documents(folder, settings.chunk_chars, settings.chunk_overlap)
    return _rank_chunks(query, chunks)


def format_context(retrieved, max_chars):
    """Format source excerpts within the document-context character budget."""
    excerpts, total_chars = [], 0
    for source_name, text in retrieved:
        excerpt = f"[Source: {source_name}]\n{text}"
        if total_chars + len(excerpt) > max_chars:
            break
        excerpts.append(excerpt)
        total_chars += len(excerpt)
    return "\n\n---\n\n".join(excerpts)
