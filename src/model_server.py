"""Launch and supervise a local llama-server without owning external servers."""

import json
import os
import shlex
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit

# Blank values inherit llama-server defaults, rather than model-specific presets.
SERVER_FIELDS = (
    ("ctx-size", "Context size", "Tokens; 0 uses the model default"),
    ("n-gpu-layers", "GPU layers", "Number of layers, auto or all"),
    ("threads", "CPU threads", "Generation threads"),
    ("threads-batch", "Prefill threads", "Prompt-processing threads"),
    ("batch-size", "Batch size", "Logical prompt batch size"),
    ("ubatch-size", "Micro batch size", "Physical prompt batch size"),
    ("parallel", "Parallel slots", "Maximum simultaneous requests"),
    ("reasoning-budget", "Reasoning budget", "Server default thinking token budget"),
    ("reasoning-effort", "Reasoning effort", "Server default; requires model support"),
    ("temp", "Temperature", "Sampling temperature"),
    ("top-p", "Top P", "Nucleus sampling threshold"),
    ("top-k", "Top K", "Candidate token count"),
    ("min-p", "Min P", "Minimum relative token probability"),
    ("presence-penalty", "Presence penalty", "Penalty for previously used tokens"),
    ("repeat-penalty", "Repeat penalty", "Repetition penalty"),
)


def launch_command(config, server_url):
    """Build an argv list; additional arguments never run through a shell."""
    executable = Path(config.get("executable", "")).expanduser()
    model = Path(config.get("model", "")).expanduser()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError(f"Choose an executable llama-server file.\nPath: {executable}")
    if not model.is_file():
        raise ValueError(f"The model file does not exist or is not a file.\nPath: {model}")
    address = urlsplit(server_url)
    if address.hostname not in ("localhost", "127.0.0.1"):
        raise ValueError("Automatic launching requires a local server address.")
    extra = shlex.split(config.get("extra", ""))
    # The client and every auxiliary request share this endpoint and model path.
    reserved = {"--host", "--port", "--model", "-m"}
    if any(arg.split("=", 1)[0] in reserved for arg in extra):
        raise ValueError("Use the model field for --model. Host and port are managed by Lopaca.")
    args = [str(executable.resolve()), "--model", str(model.resolve())]
    for key, _title, _hint in SERVER_FIELDS:
        value = str(config.get(key, "")).strip()
        if value:
            args.extend(("--" + key, value))
    for key in ("flash-attn", "fit"):
        if config.get(key, False):
            args.extend(("--" + key, "on"))
    return args + extra + ["--host", "127.0.0.1", "--port", str(address.port or 80)]


class ModelServer:
    """One asynchronous connection attempt, with bounded diagnostics and owned cleanup."""

    def __init__(self, server_url, on_state, on_error):
        self.url = server_url.rstrip("/").removesuffix("/v1")
        self.on_state, self.on_error = on_state, on_error
        self.process = None
        self.busy = False
        self.status = ""
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._output = deque(maxlen=60)
        self._config = None
        self._expected_exits = set()

    def _state(self, text):
        self.status = text
        self.on_state(text, self.busy)

    def connect(self, config, *, startup=False):
        with self._lock:
            if self.busy or self._closed.is_set():
                return
            self.busy = True
        threading.Thread(target=self._connect, args=(dict(config), startup), daemon=True).start()

    def _health(self):
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=2) as response:
                return json.load(response).get("status") == "ok"
        except urllib.error.HTTPError as exc:
            if exc.code == 503:
                return False  # A server loading a model is already listening.
            raise

    def _connect(self, config, startup):
        args = None
        reader = None
        try:
            if self.process is not None and self.process.poll() is None and config != self._config:
                # Validate first: an incomplete edit must not stop a working model.
                launch_command(config, self.url)
                self._state("Restarting llama-server…")
                self._expected_exits.add(self.process)
                self._stop_process(self.process)
                self.process = None
            self._state("Checking for an existing server…")
            try:
                if self._health():
                    self._state("Connected to an existing server")
                    return
            except urllib.error.URLError:
                pass
            address = urlsplit(self.url)
            with socket.socket() as probe:
                occupied = probe.connect_ex((address.hostname, address.port or 80)) == 0
            if occupied:
                self._state("Waiting for the existing server…")
            else:
                if startup and not config.get("executable"):
                    self._state("No server configured")
                    return
                args = launch_command(config, self.url)
                self._output.clear()
                self._state("Starting llama-server…")
                with self._lock:
                    if self._closed.is_set():
                        return
                    self.process = subprocess.Popen(
                        args,
                        cwd=str(Path(args[0]).parent),
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )
                    self._config = config
                reader = threading.Thread(
                    target=self._read_output, args=(self.process,), daemon=True
                )
                reader.start()
            deadline = time.monotonic() + 300
            while not self._closed.wait(0.25):
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError(f"llama-server exited with code {self.process.returncode}.")
                try:
                    if self._health():
                        self._state("Connected")
                        if args:
                            threading.Thread(
                                target=self._watch_exit, args=(self.process, reader), daemon=True
                            ).start()
                        return
                except urllib.error.URLError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("The server did not become ready within 5 minutes.")
        except Exception as exc:
            if reader and self.process.poll() is not None:
                reader.join(timeout=1)
            self._state("Could not connect")
            details = str(exc)
            if args:
                details += "\n\nCommand:\n" + shlex.join(args)
            output = "\n".join(self._output)
            if output:
                details += "\n\nServer output:\n" + output[-12000:]
            if not self._closed.is_set():
                self.on_error(details)
        finally:
            self.busy = False
            self.on_state(self.status, False)

    def _read_output(self, process):
        with process.stdout:
            for line in iter(process.stdout.readline, b""):
                self._output.append(line.decode("utf-8", errors="replace").rstrip()[-2000:])

    def _watch_exit(self, process, reader):
        code = process.wait()
        reader.join(timeout=1)
        if not self._closed.is_set() and process not in self._expected_exits:
            self._state("Server stopped")
            self.on_error(
                f"llama-server exited with code {code}.\n\n" + "\n".join(self._output)[-12000:]
            )
        self._expected_exits.discard(process)

    @staticmethod
    def _stop_process(process):
        with suppress(ProcessLookupError):
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                process.kill()
            process.wait()

    def close(self):
        """Only stop the child we launched; never terminate an external server."""
        self._closed.set()
        with self._lock:
            process = self.process
            if process is not None and process.poll() is None:
                threading.Thread(target=self._stop_process, args=(process,), daemon=True).start()
