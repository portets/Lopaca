"""Local document retrieval using keyword scoring."""

import codecs
import re


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


def _chunk_documents(folder, size, overlap):
    chunks = []
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        content = read_document(path)
        if not content:
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
