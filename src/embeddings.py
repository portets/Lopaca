"""Local embedding inference and persistent retrieval vectors."""

import base64
import hashlib
import json
import logging
import math
import re
import shlex
import socket
import sqlite3
import struct
import subprocess
import threading
import time
import zlib
from contextlib import closing, suppress
from http.client import HTTPConnection, HTTPException
from pathlib import Path

import thinking_classifier
from model_server import ModelServer
from server_builds import embedding_launch_parameters
from settings import EMBEDDING_QUERY_INSTRUCTION

_QUERY = f"Instruct: {EMBEDDING_QUERY_INSTRUCTION}\nQuery: "
_THINKING_TIMEOUT = 0.3
_THINKING_TIMEOUT_LIMIT = 2
_THINKING_COOLDOWN = 60.0
# Keep small passages together without making a reply compete with a long batch.
_BACKGROUND_BATCH_CHARS = 480
_PASSAGE_CACHE_SIZE = 512
_QUERY_CACHE_SIZE = 16
_THREAD_FLAGS = {"--threads", "--threads-batch", "-t", "-tb"}


def _vector_options(args):
    """Thread counts change throughput without changing the embedding model."""
    kept = []
    values = iter(args)
    for argument in values:
        flag, inline, _ = argument.partition("=")
        if flag in _THREAD_FLAGS:
            if not inline:
                next(values, None)
        else:
            kept.append(argument)
    return kept


class RecallEmbeddings:
    """Keep one local embedding session and discard it when its configuration changes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._session = None
        self._closed = False

    def configure(self, config):
        with self._lock:
            if self._closed:
                return None
            if self._session and self._session.config != config:
                self._session.close()
                self._session = None
            if self._session is None and config:
                self._session = _EmbeddingSession(config)
            if self._session and self._session.available:
                return self._session
        return None

    def schedule(self, path, table="passages"):
        with self._lock:
            if self._session and self._session.available:
                self._session.schedule(path, table)

    def close(self):
        with self._lock:
            self._closed = True
            if self._session:
                self._session.close()
                self._session = None


class _EmbeddingSession:
    def __init__(self, config):
        self.config = config
        self._stopped = threading.Event()
        self._background_allowed = threading.Event()
        self._background_allowed.set()
        self._lock = threading.Lock()
        self._inference = threading.Lock()
        self._process = self._worker = None
        self._port = None
        self._jobs = {}
        self._knowledge_pending = {}
        self._knowledge_vectors = {}
        self._passage_vectors = {}
        self._retry_at = {}
        self._vector_model = None
        self._compatible_vector_models = set()
        self._query_vectors = {}
        self.classifier_ready = False
        self._thinking_timeouts = 0
        self._thinking_retry_at = 0.0

    def close(self):
        with self._lock:
            if self._stopped.is_set():
                return
            self._stopped.set()
            self._background_allowed.set()
            process = self._process
            if process is not None:
                with suppress(ProcessLookupError):
                    process.terminate()
                threading.Thread(
                    target=ModelServer._stop_process, args=(process,), daemon=True
                ).start()

    @property
    def available(self):
        return not self._stopped.is_set()

    def pause_background(self):
        with self._lock:
            if self.available:
                self._background_allowed.clear()

    def resume_background(self):
        self._background_allowed.set()

    def thinking_vector(self, query):
        self.start()
        return self.query_vector(query, for_thinking=True) if self.classifier_ready else None

    def start(self):
        with self._lock:
            if (
                self.available
                and self._worker is None
                and (
                    self._jobs
                    or self._knowledge_pending
                    or self._process is None
                    or self._process.poll() is not None
                )
            ):
                self._worker = threading.Thread(target=self._prepare, daemon=True)
                self._worker.start()

    def schedule(self, path, table):
        job = (path, table)
        with self._lock:
            retry_at = self._retry_at.get(job, 0)
            if not self.available or time.monotonic() < retry_at:
                return
            self._retry_at.pop(job, None)
            self._jobs[job] = None
        self.start()

    def warm_knowledge(self, texts):
        with self._lock:
            for text in texts:
                if text not in self._knowledge_vectors:
                    self._knowledge_pending[text] = None
            while len(self._knowledge_pending) > 32:
                del self._knowledge_pending[next(iter(self._knowledge_pending))]
        self.start()

    def denoise_knowledge(self, query, rows, deferred):
        """Filter cached shortlist passages while keeping exact source matches available."""
        if not rows or not self.available:
            return rows
        normalized = " " + " ".join(re.findall(r"\w+", query.casefold())) + " "
        exact = _exact_patterns(query)
        protected_sources = set()
        choices = []
        with self._lock:
            for key, names, reference, body in rows:
                text = names[0] + "\n" + body[:1200]
                vector = self._knowledge_vectors.pop(text, None)
                if vector is None:
                    deferred.append(text)
                else:
                    self._knowledge_vectors[text] = vector
                anchor = key[0] not in protected_sources and any(
                    (name := " ".join(re.findall(r"\w+", value.casefold())))
                    and f" {name} " in normalized
                    for value in names
                )
                if anchor:
                    protected_sources.add(key[0])
                preserve = anchor or any(pattern.search(body) for pattern in exact)
                choices.append((vector, preserve, (key, names, reference, body)))
        if not any(vector is not None and not preserve for vector, preserve, _ in choices):
            return rows
        query_vector = self.query_vector(query)
        if query_vector is None:
            return rows
        return [
            row
            for vector, preserve, row in choices
            if (
                preserve
                or vector is None
                or len(vector) != len(query_vector)
                or math.sumprod(query_vector, vector) >= 0.55
            )
        ]

    def _start(self):
        self.classifier_ready = False
        self._vector_model = None
        self._compatible_vector_models.clear()
        self._passage_vectors.clear()
        with self._lock:
            self._query_vectors.clear()
        executable, filename, parameters = self.config
        executable = Path(executable).expanduser().resolve()
        model = Path(filename).expanduser().resolve()
        extra = shlex.split(parameters or embedding_launch_parameters())
        reserved = {"--host", "--port", "--model", "-m", "--hf-repo", "-hf", "--model-url", "-mu"}
        if any(option.split("=", 1)[0] in reserved for option in extra):
            raise ValueError("The embedding server address and model path are managed by Lopaca.")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self._port = probe.getsockname()[1]
        args = [
            str(executable),
            "--model",
            str(model),
            "--embedding",
            "--pooling",
            "last",
            "--device",
            "none",
            "--n-gpu-layers",
            "0",
            "--poll",
            "0",
            *extra,
            "--host",
            "127.0.0.1",
            "--port",
            str(self._port),
        ]
        with self._lock:
            if self._stopped.is_set():
                raise InterruptedError
            self._process = subprocess.Popen(
                args,
                cwd=executable.parent,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        for _ in range(80):
            if self._stopped.wait(0.125):
                raise InterruptedError
            if self._process.poll() is not None:
                break
            with suppress(OSError, HTTPException, ValueError):
                if self._request("/health", timeout=0.2).get("status") == "ok":
                    break
        else:
            raise RuntimeError("The local embedding server did not become ready.")
        if self._process.poll() is not None:
            raise RuntimeError("The local embedding server exited before it became ready.")
        self.classifier_ready = self._supports_thinking_classifier(model, extra)
        stat = model.stat()
        # Include configuration that can change vector geometry, even for a custom GGUF.
        identity = (
            "passage-v1",
            str(model),
            stat.st_ino,
            stat.st_dev,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            str(executable),
        )

        def fingerprint(options):
            return hashlib.sha256(repr((*identity, options)).encode()).hexdigest()

        vector_options = _vector_options(extra)
        self._vector_model = fingerprint(vector_options)
        self._compatible_vector_models = {self._vector_model, fingerprint(extra)}
        if not parameters:
            # Older versions stored auto-detected thread counts in the fingerprint.
            # Adopt those vectors without making existing indexes run a full backfill.
            self._compatible_vector_models.update(
                fingerprint([*vector_options, "--threads", str(n), "--threads-batch", str(n)])
                for n in range(1, 11)
            )

    @staticmethod
    def _supports_thinking_classifier(model, args):
        if EMBEDDING_QUERY_INSTRUCTION != thinking_classifier.QUERY_INSTRUCTION:
            return False
        for index, arg in enumerate(args):
            flag, equals, value = arg.partition("=")
            if flag == "--override-kv":
                return False
            if flag == "--pooling":
                pooling = [value] if equals else args[index + 1 : index + 2]
                if pooling != ["last"]:
                    return False
        with suppress(OSError):
            if model.stat().st_size == thinking_classifier.MODEL_SIZE:
                with model.open("rb") as file:
                    return (
                        hashlib.file_digest(file, "sha256").hexdigest()
                        == thinking_classifier.MODEL_SHA256
                    )
        return False

    def _request(self, endpoint, payload=None, *, timeout):
        if self._stopped.is_set():
            raise InterruptedError
        client = HTTPConnection("127.0.0.1", self._port, timeout=timeout)
        try:
            if payload is None:
                client.request("GET", endpoint)
            else:
                client.request(
                    "POST", endpoint, json.dumps(payload), {"Content-Type": "application/json"}
                )
            response = client.getresponse()
            if response.status != 200:
                raise OSError(f"Embedding server returned HTTP {response.status}")
            return json.loads(response.read())
        finally:
            client.close()

    def _embed(self, texts, *, timeout):
        result = self._request(
            "/v1/embeddings", {"input": texts, "encoding_format": "base64"}, timeout=timeout
        )
        vectors = []
        for entry in sorted(result["data"], key=lambda item: item["index"]):
            raw = base64.b64decode(entry["embedding"], validate=True)
            size = min(len(raw) // 4, 256)
            vector = struct.unpack_from(f"<{size}f", raw)
            norm = math.sqrt(math.sumprod(vector, vector))
            if not norm or not math.isfinite(norm):
                raise ValueError("Invalid embedding vector.")
            vectors.append(tuple(value / norm for value in vector))
        if len(vectors) != len(texts):
            raise ValueError("The embedding server returned the wrong number of vectors.")
        return vectors

    def _prepare(self):
        try:
            if self._process is None or self._process.poll() is not None:
                self._start()
            while self.available:
                with self._lock:
                    if self._knowledge_pending:
                        knowledge = list(self._knowledge_pending)[:4]
                        for text in knowledge:
                            del self._knowledge_pending[text]
                        job = None
                    elif self._jobs:
                        knowledge = ()
                        job = next(iter(self._jobs))
                        del self._jobs[job]
                    else:
                        self._worker = None
                        return
                if knowledge:
                    self._embed_knowledge(knowledge)
                    continue
                path, table = job
                try:
                    if path.is_file() and self._index_batch(path, table):
                        with self._lock:
                            self._jobs[job] = None
                except (
                    OSError,
                    HTTPException,
                    sqlite3.Error,
                    zlib.error,
                    KeyError,
                    TypeError,
                    ValueError,
                ):
                    if self.available and path.is_file():
                        with self._lock:
                            self._retry_at[job] = time.monotonic() + 30
                            self._jobs.pop(job, None)
                        logging.debug("Could not index embeddings for %s", path, exc_info=True)
        except OSError, HTTPException, KeyError, TypeError, ValueError, RuntimeError:
            if self.available:
                logging.warning(
                    "Local embeddings unavailable. Using keyword recall.", exc_info=True
                )
                self.close()

    def _embed_knowledge(self, texts):
        try:
            vectors = self._embed_passages(texts)
            if vectors is None:
                return
        except OSError, HTTPException, KeyError, TypeError, ValueError:
            logging.debug("Could not score knowledge excerpts", exc_info=True)
            return
        with self._lock:
            self._knowledge_vectors.update(zip(texts, vectors, strict=True))
            while len(self._knowledge_vectors) > 128:
                del self._knowledge_vectors[next(iter(self._knowledge_vectors))]

    def _embed_passages(self, texts):
        keys = [hashlib.sha256(text.encode()).digest() for text in texts]
        vectors, missing = {}, {}
        for key, text in zip(keys, texts, strict=True):
            if key in self._passage_vectors:
                vectors[key] = self._passage_vectors.pop(key)
                self._passage_vectors[key] = vectors[key]
            else:
                missing[key] = text

        def batches():
            batch, length = {}, 0
            for key, text in missing.items():
                if batch and length + len(text) > _BACKGROUND_BATCH_CHARS:
                    yield batch
                    batch, length = {}, 0
                batch[key] = text
                length += len(text)
            if batch:
                yield batch

        for batch in batches():
            # Give interactive requests first use of the CPU between embedding batches.
            while self.available:
                self._background_allowed.wait()
                with self._inference:
                    if not self.available:
                        return None
                    if not self._background_allowed.is_set():
                        continue
                    generated = self._embed(list(batch.values()), timeout=10)
                    break
            else:
                return None
            found = dict(zip(batch, generated, strict=True))
            vectors.update(found)
            self._passage_vectors.update(found)
            while len(self._passage_vectors) > _PASSAGE_CACHE_SIZE:
                del self._passage_vectors[next(iter(self._passage_vectors))]
        return [vectors[key] for key in keys]

    def _index_batch(self, path, table):
        memory = table == "memories"
        column = "text" if memory else "body"
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True)) as db:
            state = _vector_model(db)
            if state is None:
                if not db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
                ).fetchone():
                    return False  # A newly created chat may not have run its lexical indexer.
                active = "WHERE NOT core" if memory else ""
                if not db.execute(f"SELECT 1 FROM {table} {active} LIMIT 1").fetchone():
                    return False
            if state != self._vector_model:
                with db:
                    db.execute("BEGIN IMMEDIATE")
                    if not self.available:
                        return False
                    state = _vector_model(db)
                    if state in self._compatible_vector_models:
                        db.execute("UPDATE rag_vector_model SET model=?", (self._vector_model,))
                    elif state != self._vector_model:
                        _initialize_vectors(db, table, column, self._vector_model, memory)
            batch = db.execute(
                f"SELECT v.id, s.{column} FROM rag_vectors v JOIN {table} s ON s.id=v.id "
                "WHERE v.vector IS NULL ORDER BY v.id LIMIT 4"
            ).fetchall()
        if not batch or not self.available:
            return False
        texts = [value if memory else zlib.decompress(value).decode() for _, value in batch]
        vectors = self._embed_passages(texts)
        if vectors is None:
            return False
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if not self.available or _vector_model(db) != self._vector_model:
                return False
            # Check the original data again after inference. A replaced row may reuse an ID.
            db.executemany(
                f"UPDATE rag_vectors SET vector=? WHERE id=? AND vector IS NULL AND EXISTS "
                f"(SELECT 1 FROM {table} s WHERE s.id=rag_vectors.id AND s.{column}=?)",
                [
                    (struct.pack(f"<{len(vector)}f", *vector), row_id, value)
                    for (row_id, value), vector in zip(batch, vectors, strict=True)
                ],
            )
        return True

    def rank(self, db, query, rows, *, scope, limit, excluded=(), stopped=lambda: False):
        """Preserve keyword order, filter weak indexed hits, and add one semantic match."""
        model = self._vector_model
        if not model or not self.available or stopped():
            return rows
        if _vector_model(db) not in self._compatible_vector_models:
            return rows
        if not db.execute("SELECT 1 FROM rag_vectors WHERE vector IS NOT NULL LIMIT 1").fetchone():
            return rows
        vector = self.query_vector(query)
        if vector is None or stopped():
            return rows
        return _rank_vectors(db, query, vector, rows, scope, limit, excluded, stopped)

    def query_vector(self, query, *, for_thinking=False):
        if self._stopped.is_set() or not query.strip():
            return None
        query = _QUERY + query.strip()[-1500:]
        with self._lock:
            cached = self._query_vectors.pop(query, None)
            if cached is not None:
                self._query_vectors[query] = cached
                return cached
        if not self._inference.acquire(blocking=False):
            return None
        try:
            if for_thinking and time.monotonic() < self._thinking_retry_at:
                return None
            (vector,) = self._embed([query], timeout=_THINKING_TIMEOUT if for_thinking else 0.2)
            with self._lock:
                self._query_vectors[query] = vector
                if len(self._query_vectors) > _QUERY_CACHE_SIZE:
                    del self._query_vectors[next(iter(self._query_vectors))]
            if for_thinking:
                self._thinking_timeouts = 0
                self._thinking_retry_at = 0.0
            return vector
        except TimeoutError:
            if for_thinking:
                self._thinking_timeouts += 1
                # A slow retry renews the cooldown until one request succeeds.
                if self._thinking_timeouts >= _THINKING_TIMEOUT_LIMIT:
                    self._thinking_retry_at = time.monotonic() + _THINKING_COOLDOWN
            return None
        except OSError, HTTPException, KeyError, TypeError, ValueError:
            return None
        finally:
            self._inference.release()


# Filter conservatively. Related but non-answering passages can still have high similarity.
# Adding a new source beside credible lexical hits requires stronger evidence.
_RECALL_CUTOFFS = {
    "documents": (0.55, 0.55, 0.61),
    "memories": (0.45, 0.56, 0.62),
    "history": (0.55, 0.54, 0.63),
}


def _vector_model(db):
    if db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='rag_vector_model'"
    ).fetchone():
        row = db.execute("SELECT model FROM rag_vector_model WHERE id=1").fetchone()
        return row[0] if row else None
    return None


def _initialize_vectors(db, table, column, model, memory):
    # Only the three local RAG indexes call this. Knowledge packs keep their existing schema.
    db.execute("CREATE TABLE IF NOT EXISTS rag_vector_model (id INTEGER PRIMARY KEY, model TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS rag_vectors (id INTEGER PRIMARY KEY, vector BLOB)")
    db.execute(
        "CREATE INDEX IF NOT EXISTS rag_vectors_pending ON rag_vectors(id) WHERE vector IS NULL"
    )
    condition = "WHERE NOT NEW.core" if memory else ""
    insert = f"INSERT OR REPLACE INTO rag_vectors(id) SELECT NEW.id {condition};"
    db.execute(
        f"CREATE TRIGGER IF NOT EXISTS rag_vector_insert AFTER INSERT ON {table} BEGIN {insert} END"
    )
    db.execute(
        f"CREATE TRIGGER IF NOT EXISTS rag_vector_delete AFTER DELETE ON {table} "
        "BEGIN DELETE FROM rag_vectors WHERE id=OLD.id; END"
    )
    changed = f"OLD.{column} IS NOT NEW.{column}"
    if memory:
        changed += " OR OLD.core IS NOT NEW.core"
    db.execute(
        f"CREATE TRIGGER IF NOT EXISTS rag_vector_update AFTER UPDATE ON {table} "
        f"WHEN {changed} BEGIN DELETE FROM rag_vectors WHERE id=OLD.id; {insert} END"
    )
    db.execute("DELETE FROM rag_vectors")
    condition = "WHERE NOT core" if memory else ""
    db.execute(f"INSERT INTO rag_vectors(id) SELECT id FROM {table} {condition}")
    db.execute("INSERT OR REPLACE INTO rag_vector_model VALUES (1, ?)", (model,))


def _exact_patterns(query):
    query = query[-1500:]
    literals = set(re.findall(r'`([^`]+)`|"([^"\n]+)"', query))
    exact = {a or b for a, b in literals}
    exact.update(
        word
        for word in re.findall(r"[\w./:+#-]+", query)
        if (re.search(r"[a-zA-Z]", word) and re.search(r"\d|[_./:+#]|[a-z][A-Z]", word.strip(".?")))
    )
    return [
        re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
        for value in exact
        if (term := value.rstrip("."))
    ]


def _rank_vectors(db, query, vector, rows, scope, limit, excluded, stopped):
    memory = scope == "memories"
    filter_at, add_at, add_with_keywords_at = _RECALL_CUTOFFS[scope]
    unpack = struct.Struct(f"<{len(vector)}f")
    exact = _exact_patterns(query)
    ranked = {row[0]: row for row in rows}
    protected = {
        row[0]
        for row in rows
        if any(pattern.search(row[1] if memory else "\n".join(row[1:4])) for pattern in exact)
    }
    scores = {}
    best = None
    matched_sources = {row[1] for row in rows} if scope == "history" else set()
    join = "" if memory else "JOIN passages p ON p.id=v.id"
    columns = "v.id, v.vector" if memory else "v.id, v.vector, p.source"
    for index, (row_id, blob, *source) in enumerate(
        db.execute(f"SELECT {columns} FROM rag_vectors v {join} WHERE v.vector IS NOT NULL")
    ):
        if index % 256 == 0 and stopped():
            return rows
        if len(blob) != unpack.size or (source and source[0] in excluded):
            continue
        score = math.sumprod(vector, unpack.unpack(blob))
        if row_id in ranked:
            scores[row_id] = score
        elif (not source or source[0] not in matched_sources) and (best is None or score > best[0]):
            best = score, row_id
    kept = [row for row in rows if row[0] in protected or scores.get(row[0], 1.0) >= filter_at]
    if scope == "documents" and len(kept) > 1 and not protected:
        winner = max(kept, key=lambda row: scores.get(row[0], -1.0))
        first_score = scores.get(kept[0][0])
        if first_score is not None and scores.get(winner[0], -1.0) - first_score >= 0.07:
            kept.remove(winner)
            kept.insert(0, winner)
    add_threshold = max(filter_at, add_with_keywords_at if kept else add_at)
    if best is None or best[0] < add_threshold:
        return kept
    if len(kept) >= limit:
        row_id = kept[-1][0]
        if scope != "documents" or row_id in protected or scores.get(row_id, 1.0) >= best[0]:
            return kept
    row = db.execute(
        "SELECT id,text,core FROM memories WHERE id=? AND NOT core"
        if memory
        else "SELECT rowid,source,section,body,start_line,end_line "
        "FROM chunk_content WHERE rowid=?",
        (best[1],),
    ).fetchone()
    if not row:
        return kept
    if not memory and any(
        row[1] == hit[1] and row[4] <= hit[5] and row[5] >= hit[4] for hit in kept
    ):
        return kept
    return [*kept[: limit - 1], row]
