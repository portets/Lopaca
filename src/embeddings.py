"""Local embedding inference and bounded in-memory recall."""

import base64
import hashlib
import json
import logging
import math
import shlex
import socket
import struct
import subprocess
import threading
import time
from contextlib import suppress
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


class RecallEmbeddings:
    """Keep one local embedding session and discard it when its configuration changes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._session = None
        self._closed = False

    def prepare(self, executable, filename, parameters, texts, *, group="memories", defer=None):
        with self._lock:
            config = (executable, filename, parameters) if executable and filename else None
            session = self._session_for(config, create=bool(texts))
            if session:
                if defer is None:
                    session.warm(group, texts)
                else:
                    defer(lambda: session.warm(group, texts))
                return session.match
        return None

    def _session_for(self, config, *, create):
        if self._closed:
            return None
        if self._session and self._session.config != config:
            self._session.close()
            self._session = None
        if self._session is None and config and create:
            self._session = _EmbeddingSession(config)
        return self._session

    def thinking_vector(self, config, query):
        with self._lock:
            session = self._session_for(config, create=True)
        if session:
            session.warm("auto", (), start=True)
            if session.classifier_ready:
                return session.query_vector(query, for_thinking=True)
        return None

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
        self._lock = threading.Lock()
        self._inference = threading.Lock()
        self._process = self._worker = None
        self._port = None
        self._vectors = {}
        self._groups = {}
        self._last_query = None
        self._failed = False
        self.classifier_ready = False
        self._thinking_timeouts = 0
        self._thinking_retry_at = 0.0

    def close(self):
        with self._lock:
            if self._stopped.is_set():
                return
            self._stopped.set()
            process = self._process
            if process is not None:
                with suppress(ProcessLookupError):
                    process.terminate()
                threading.Thread(
                    target=ModelServer._stop_process, args=(process,), daemon=True
                ).start()

    def warm(self, group, texts, *, start=False):
        with self._lock:
            if self._failed or self._stopped.is_set():
                return
            self._groups[group] = set(texts)
            wanted = set().union(*self._groups.values())
            if not self._worker and self._process is not None and self._process.poll() is not None:
                self._process = None
            self._vectors = {text: v for text, v in self._vectors.items() if text in wanted}
            if not self._worker and (
                wanted.difference(self._vectors) or ((wanted or start) and not self._process)
            ):
                self._worker = threading.Thread(target=self._prepare, daemon=True)
                self._worker.start()

    def _start(self):
        self.classifier_ready = False
        self._last_query = None
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
            if self._process is None:
                self._start()
            while not self._stopped.is_set():
                with self._lock:
                    batch = next(
                        (
                            sorted(missing)[:8]
                            for texts in self._groups.values()
                            if (missing := texts.difference(self._vectors))
                        ),
                        None,
                    )
                    if batch is None:
                        self._worker = None
                        return
                with self._inference:
                    vectors = self._embed(batch, timeout=10)
                with self._lock:
                    wanted = set().union(*self._groups.values())
                    self._vectors.update(
                        (text, vector)
                        for text, vector in zip(batch, vectors, strict=True)
                        if text in wanted
                    )
        except OSError, HTTPException, KeyError, TypeError, ValueError, RuntimeError:
            if not self._stopped.is_set():
                logging.warning(
                    "Local embeddings unavailable. Using keyword recall.", exc_info=True
                )
                with self._lock:
                    self._failed = True
                self.close()

    def match(self, query, choices, *, minimum, rerank_margin=None):
        if self._stopped.is_set() or not query.strip():
            return None
        with self._lock:
            candidates = [
                (key, self._vectors[text]) for key, text in choices if text in self._vectors
            ]
        if not candidates:
            return None
        if rerank_margin is not None and (len(candidates) < 2 or candidates[0][0] != choices[0][0]):
            return None  # Compare against the keyword winner only when its vector is ready.
        vector = self.query_vector(query)
        if vector is None:
            return None
        scores = [(key, math.sumprod(vector, value)) for key, value in candidates]
        key, score = max(scores, key=lambda item: item[1])
        if rerank_margin is not None:
            minimum = max(minimum, scores[0][1] + rerank_margin)
        return key if score >= minimum else None

    def query_vector(self, query, *, for_thinking=False):
        if self._stopped.is_set() or not query.strip():
            return None
        query = _QUERY + query.strip()[-1500:]
        cached = self._last_query
        if cached is not None and cached[0] == query:
            return cached[1]
        if not self._inference.acquire(blocking=False):
            return None
        try:
            if for_thinking and time.monotonic() < self._thinking_retry_at:
                return None
            (vector,) = self._embed([query], timeout=_THINKING_TIMEOUT if for_thinking else 0.2)
            self._last_query = query, vector
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
