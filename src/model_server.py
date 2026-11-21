"""Launch and supervise a local llama-server without owning external servers."""

import json
import os
import platform
import re
import shlex
import socket
import subprocess
import tarfile
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

from web import download_verified_file, matches_download

_SERVER_RELEASE = "b12359"
_SERVER_ARCHIVE = f"llama-{_SERVER_RELEASE}-bin-ubuntu-x64.tar.gz"
_SERVER_URL = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/{_SERVER_RELEASE}/{_SERVER_ARCHIVE}"
)
_SERVER_SIZE = 23_311_084
_SERVER_SHA256 = "0bc1f091e8916102b6d8f2cf944233a620e90a0f856c72d0ce41150c65c2e72e"
_SERVER_LIBRARY = re.compile(
    r"(?:libggml(?:-base|-cpu-[\w-]+)?|libllama(?:-server-impl|-common)?|libmtmd)"
    r"\.so(?:\.\d+)*"
)
_MODEL_NAME = "Qwen3.8-27B-UD-Q4_K_M.gguf"
# Same file as the main-branch download, pinned to the published checksum.
_MODEL_URL = (
    "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/"
    f"4af51b3542f775b0a8e4b310bfb3f17e01c306c9/{_MODEL_NAME}?download=true"
)
_MODEL_SIZE = 16_564_259_744
_MODEL_SHA256 = "1bce504dfecd84153666928a41d8a66706f0158d3a317bc5196b72819d7dc529"
DOWNLOADING_MODEL = "Downloading model"

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


def _extract_default_server(archive_path, staging):
    root = f"llama-{_SERVER_RELEASE}"
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {}
        for member in archive:
            parent, _, name = member.name.rpartition("/")
            if parent != root or not (name == "llama-server" or _SERVER_LIBRARY.fullmatch(name)):
                continue
            if name in members or not (member.isfile() or member.issym()):
                raise ValueError(f"Invalid release member: {member.name}")
            members[name] = member
        if not members.get("llama-server") or not members["llama-server"].isfile():
            raise ValueError("The downloaded release is missing llama-server.")
        for member in members.values():
            if member.issym() and member.linkname not in members:
                raise ValueError(f"Invalid release library link: {member.name}")
        archive.extractall(staging, members=members.values(), filter="data")
    return staging / root


def install_default_model(folder, stopped):
    """Verify the pinned release and model before publishing their configured paths."""
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
        raise ValueError("The automatic setup uses the Ubuntu x64 build of llama-server.")
    server = folder / "server"
    executable = server / "llama-server"
    if not executable.is_file():
        server.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=server.parent, prefix=".server-install-") as temporary:
            staging = Path(temporary)
            archive = staging / _SERVER_ARCHIVE
            download_verified_file(
                _SERVER_URL, archive, size=_SERVER_SIZE, sha256=_SERVER_SHA256, stopped=stopped
            )
            _extract_default_server(archive, staging).rename(server)

    model = folder / "models" / _MODEL_NAME
    model.parent.mkdir(parents=True, exist_ok=True)
    expected = {"size": _MODEL_SIZE, "sha256": _MODEL_SHA256, "stopped": stopped}
    if model.exists():
        if not matches_download(model, **expected):
            raise ValueError(f"The file at {model} differs from the default model.")
    else:
        partial = model.with_suffix(".gguf.download")
        download_verified_file(_MODEL_URL, partial, **expected)
        partial.rename(model)
    return {"executable": str(executable), "model": str(model)}


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

    def connect(self, config, *, startup=False, prepare=None):
        with self._lock:
            if self.busy or self._closed.is_set():
                return
            self.busy = True
        threading.Thread(
            target=self._connect, args=(dict(config), startup, prepare), daemon=True
        ).start()

    def _health(self):
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=2) as response:
                return json.load(response).get("status") == "ok"
        except urllib.error.HTTPError as exc:
            if exc.code == 503:
                return False  # A server loading a model is already listening.
            raise

    def _port_occupied(self):
        address = urlsplit(self.url)
        with socket.socket() as probe:
            return probe.connect_ex((address.hostname, address.port or 80)) == 0

    def _connect(self, config, startup, prepare=None):
        args = None
        reader = None
        try:
            if (
                prepare is None
                and self.process is not None
                and self.process.poll() is None
                and config != self._config
            ):
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
            occupied = self._port_occupied()
            if prepare is not None and not occupied:
                self._state(DOWNLOADING_MODEL)
                config.update(prepare(self._closed.is_set))
                if self._closed.is_set():
                    return
                occupied = self._port_occupied()
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
                if args is not None and self.process.poll() is not None:
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
