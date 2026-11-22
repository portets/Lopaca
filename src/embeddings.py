"""Local embedding inference and bounded in-memory recall."""

import base64
import json
import logging
import math
import shlex
import socket
import struct
import subprocess
import threading
from contextlib import suppress
from http.client import HTTPConnection, HTTPException
from pathlib import Path

from model_server import ModelServer
from server_builds import embedding_launch_parameters
from settings import EMBEDDING_QUERY_INSTRUCTION

_QUERY = f"Instruct: {EMBEDDING_QUERY_INSTRUCTION}\nQuery: "


class RecallEmbeddings:
    """Keep one local embedding session and discard it when its configuration changes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._session = None
        self._closed = False

    def prepare(self, executable, filename, parameters, texts, *, group="memories"):
        with self._lock:
            if self._closed:
                return None
            config = (executable, filename, parameters) if executable and filename else None
            session = self._session
            if session and session.config != config:
                session.close()
                self._session = session = None
            if config and (texts or session):
                if session is None:
                    self._session = session = _EmbeddingSession(config)
                session.warm(group, texts)
                return session.match
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

    def warm(self, group, texts):
        with self._lock:
            if self._failed or self._stopped.is_set():
                return
            self._groups[group] = set(texts)
            wanted = set().union(*self._groups.values())
            if not self._worker and self._process is not None and self._process.poll() is not None:
                self._process = None
            self._vectors = {text: v for text, v in self._vectors.items() if text in wanted}
            if not self._worker and (
                wanted.difference(self._vectors) or (wanted and not self._process)
            ):
                self._worker = threading.Thread(target=self._prepare, daemon=True)
                self._worker.start()

    def _start(self):
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
                    return
        raise RuntimeError("The local embedding server did not become ready.")

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

    def match(self, query, choices, *, minimum):
        if self._stopped.is_set() or not query.strip():
            return None
        with self._lock:
            candidates = [
                (key, self._vectors[text]) for key, text in choices if text in self._vectors
            ]
        if not candidates:
            return None
        query = _QUERY + query.strip()[-1500:]
        cached = self._last_query
        if cached is not None and cached[0] == query:
            vector = cached[1]
        else:
            if not self._inference.acquire(blocking=False):
                return None
            try:
                (vector,) = self._embed([query], timeout=0.2)
                self._last_query = query, vector
            except OSError, HTTPException, KeyError, TypeError, ValueError:
                return None
            finally:
                self._inference.release()
        key, value = max(candidates, key=lambda item: math.sumprod(vector, item[1]))
        return key if math.sumprod(vector, value) >= minimum else None
