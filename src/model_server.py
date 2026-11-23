"""Launch and supervise a local llama-server without owning external servers."""

import json
import logging
import os
import re
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

from settings import (
    MODEL_RECOMMENDATIONS,
    MODEL_SAMPLING_DEFAULTS,
    SERVER_LAUNCH_DEFAULTS,
    SERVER_SAMPLING_DEFAULTS,
)

DOWNLOADING_MODEL = "Downloading model"
_GPU_LAYERS = re.compile(r"\boffloaded (\d+)/\d+ layers to GPU\b")

SERVER_FIELDS = (
    ("port", "Server port", "Local port from 1 to 65535; default 9931"),
    ("ctx-size", "Context size", "Tokens; 0 uses the model default"),
    ("n-gpu-layers", "GPU layers", "Number of layers, auto or all"),
    ("threads", "CPU threads", "Generation threads; -1 uses automatic detection"),
    ("threads-batch", "Prefill threads", "Prompt-processing threads; -1 follows CPU threads"),
    ("batch-size", "Batch size", "Logical prompt batch size"),
    ("ubatch-size", "Micro batch size", "Physical prompt batch size"),
    ("parallel", "Parallel slots", "Maximum simultaneous requests; -1 uses automatic detection"),
    ("cache-ram", "Prompt cache RAM", "MiB; 0 disables the cache and -1 removes its limit"),
    ("flash-attn", "Flash Attention", "on, off or auto"),
    ("fit", "Fit to device memory", "on or off"),
    ("fit-ctx", "Minimum fitted context", "Minimum context size in tokens when fitting"),
    ("fit-target", "Fit memory margin", "MiB to leave free per device; use commas for each device"),
    ("reasoning", "Reasoning", "on, off or auto"),
    ("reasoning-budget", "Reasoning budget", "Server default thinking token budget"),
    ("reasoning-effort", "Reasoning effort", "Server default; requires model support"),
    ("spec-type", "Speculative decoding", "none, draft-mtp or another supported mode"),
    ("spec-draft-n-max", "Maximum drafted tokens", "Tokens predicted per speculative step"),
)
# Choices from the bundled llama-server release. The server still accepts custom
# template levels and comma-separated speculative modes from existing settings.
SERVER_FIELD_CHOICES = {
    "flash-attn": ("auto", "on", "off"),
    "fit": ("on", "off"),
    "reasoning": ("auto", "on", "off"),
    "reasoning-effort": ("default", "minimal", "low", "medium", "high", "xhigh", "max"),
    "spec-type": (
        "none", "draft-mtp", "draft-simple", "draft-eagle3", "draft-dflash", "draft-dspark",
        "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache",
    ),
}
SAMPLING_FIELDS = (
    ("temp", "Temperature", "Sampling temperature"),
    ("top-p", "Top P", "Nucleus sampling threshold"),
    ("top-k", "Top K", "Candidate token count"),
    ("min-p", "Min P", "Minimum relative token probability"),
    ("presence-penalty", "Presence penalty", "Penalty for previously used tokens"),
    ("repeat-penalty", "Repeat penalty", "Repetition penalty"),
)

_SAMPLING_FLAGS = {
    "temperature": "temp", "top_p": "top-p", "top_k": "top-k", "min_p": "min-p",
    "presence_penalty": "presence-penalty", "repetition_penalty": "repeat-penalty",
}


def server_parameters(config):
    """Resolve defaults, the hardware tier, then saved settings for UI and argv."""
    model = Path(config.get("model") or "").name.casefold()
    sampling = next(
        (values for prefix, values in MODEL_SAMPLING_DEFAULTS.items() if model.startswith(prefix)),
        {},
    )
    parameters = dict(SERVER_LAUNCH_DEFAULTS)
    parameters["port"] = 9931
    sampling = SERVER_SAMPLING_DEFAULTS | sampling
    parameters.update(
        (_SAMPLING_FLAGS[name], value) for name, value in sampling.items()
    )
    if rule := MODEL_RECOMMENDATIONS.get(config.get("recommendation")):
        parameters.update(rule.launch_parameters)
    for key in parameters.keys() & config.keys():
        value = config[key]
        # Older settings stored the Flash Attention and Fit checkboxes as booleans.
        if key in ("flash-attn", "fit") and isinstance(value, bool):
            value = "on" if value else ""
        if value is not None and str(value).strip():
            parameters[key] = value
    return parameters


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
    family = config.get("vision_family")
    vision = config.get("mmproj")
    vision_off = config.get("load_vision") is False or config.get("vision_fallback")
    if family or vision_off:
        # Keep vision unloaded for text-only chats, including manual launch flags.
        source_flags = {"-mm", "--mmproj", "-mmu", "--mmproj-url"}
        toggle_flags = {"--mmproj-auto", "--no-mmproj", "--no-mmproj-auto"}
        remaining = []
        extra_args = iter(extra)
        for arg in extra_args:
            flag, equals, _ = arg.partition("=")
            if flag in source_flags:
                if not equals:
                    next(extra_args, None)
            elif flag not in toggle_flags:
                remaining.append(arg)
        extra = remaining
        if vision_off or family in config.get("vision_disabled", ()) or not vision:
            args.append("--no-mmproj")
            vision = None
    if vision:
        projector = Path(vision).expanduser()
        if not projector.is_file():
            raise ValueError(f"The vision file does not exist or is not a file.\nPath: {projector}")
        args.extend(("--mmproj", str(projector.resolve())))
    for key, value in server_parameters(config).items():
        if key != "port":
            args.extend(("--" + key, str(value).strip()))
    devices = config.get("devices")
    if (
        devices
        and not os.environ.get("LLAMA_ARG_DEVICE")
        and not any(arg.split("=", 1)[0] in ("-dev", "--device") for arg in extra)
    ):
        args.extend(("--device", ",".join(devices)))
    return args + extra + ["--host", "127.0.0.1", "--port", str(address.port or 80)]


def _configured_gpu_offload(config, args):
    """Use the launch options when llama-server omits offload details from its log."""
    devices = os.environ.get("LLAMA_ARG_DEVICE", "")
    layers = os.environ.get("LLAMA_ARG_N_GPU_LAYERS", "")
    options = iter(args[1:])
    for argument in options:
        flag, equals, value = argument.partition("=")
        if flag in ("--device", "-dev"):
            devices = value if equals else next(options, "")
        elif flag in ("--n-gpu-layers", "--gpu-layers", "-ngl"):
            layers = value if equals else next(options, "")
    if layers.isdecimal() and int(layers) == 0:
        return False
    if devices:
        return any(device.strip().casefold().startswith(("cuda", "vulkan"))
                   for device in devices.split(","))
    backend = config.get("backend") or ""
    return backend.startswith("cuda") or backend == "vulkan"


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
        self._gpu_offload = None
        self._configured_gpu = False
        self._expected_exits = set()

    def _state(self, text):
        self.status = text
        self.on_state(text, self.busy)

    @property
    def running_config(self):
        process = self.process
        return self._config if process is not None and process.poll() is None else None

    @property
    def gpu_offloaded(self):
        if self.running_config is None:
            return False
        return self._configured_gpu if self._gpu_offload is None else self._gpu_offload

    def connect(
        self,
        config,
        *,
        startup=False,
        prepare=None,
        prepare_config=None,
        on_config=None,
        on_finished=None,
    ):
        with self._lock:
            if self.busy or self._closed.is_set():
                return False
            self.busy = True
        threading.Thread(
            target=self._connect,
            args=(dict(config), startup, prepare, prepare_config, on_config, on_finished),
            daemon=True,
        ).start()
        return True

    def disconnect(self, finished):
        """Unload the model served by our child process."""
        with self._lock:
            if self.busy or self._closed.is_set():
                return
            self.busy = True

        def unload():
            try:
                process = self.process
                if process is not None and process.poll() is None:
                    self._state("Stopping llama-server…")
                    self._expected_exits.add(process)
                    self._stop_process(process)
                self.process = self._config = None
                self._state("Model unloaded")
            except OSError as exc:
                if not self._closed.is_set():
                    self.on_error(str(exc))
            finally:
                self.busy = False
                self.on_state(self.status, False)
            if self.process is None and not self._closed.is_set():
                finished()

        threading.Thread(target=unload, daemon=True).start()

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

    def _connect(self, config, startup, prepare, prepare_config, on_config, on_finished):
        args = None
        try:
            prepared = False
            replacing = prepare is not None and self.running_config is not None
            if prepare is None and self.running_config is not None:
                if prepare_config:
                    self._state("Preparing model configuration…")
                    config = prepare_config(config, self._closed.is_set)
                    prepared = True
                if self._closed.is_set():
                    return
                if config != self._config:
                    # Validate first: an incomplete edit must not stop a working model.
                    launch_command(config, self.url)
                    self._restart_process()
            if not replacing:
                self._state("Checking for an existing server…")
                try:
                    if self._health():
                        self._state("Connected to an existing server")
                        return
                except urllib.error.URLError:
                    pass
                if self._port_occupied():
                    self._wait_for_existing_server()
                    return
            if startup and not (config.get("executable") and config.get("model")):
                self._state("No server configured")
                return

            if prepare:
                self._state(DOWNLOADING_MODEL)
            candidates = prepare(self._closed.is_set) if prepare else (None,)
            last_error = None
            for installed in candidates:
                if self._closed.is_set():
                    return
                if not (replacing and self.running_config is not None) and self._port_occupied():
                    self._wait_for_existing_server()
                    return
                selected = {**config, **(installed or {})}
                if prepare_config and not prepared:
                    selected = prepare_config(selected, self._closed.is_set)
                if self._closed.is_set():
                    return
                args = launch_command(selected, self.url)
                if replacing and self.running_config is not None:
                    self._restart_process()
                replacing = False
                try:
                    try:
                        started = self._launch(args, selected)
                    except (OSError, RuntimeError):
                        if not (selected.get("load_vision") and selected.get("vision_inferred")):
                            raise
                        selected["vision_fallback"] = True
                        args = launch_command(selected, self.url)
                        started = self._launch(args, selected)
                        if started:
                            self.on_error(
                                "The inferred vision file could not be loaded. The model is "
                                "running without vision. You can select a different file in Vision."
                            )
                    if not started:
                        return
                except (OSError, RuntimeError) as exc:
                    if not installed or not self._backend_failed(installed["backend"], exc):
                        raise
                    last_error = exc
                    logging.warning(
                        "Trying another backend after %s failed: %s", installed["backend"], exc
                    )
                    self._state(DOWNLOADING_MODEL)
                    continue
                if on_config:
                    on_config(selected)
                self._state("Connected")
                return
            if last_error and not self._closed.is_set():
                raise last_error
        except Exception as exc:
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
            if on_finished:
                on_finished()

    def _restart_process(self):
        self._state("Restarting llama-server…")
        self._expected_exits.add(self.process)
        self._stop_process(self.process)
        self.process = None

    def _wait_for_existing_server(self):
        self._state("Waiting for the existing server…")
        if self._wait_for_health():
            self._state("Connected to an existing server")

    def _wait_for_health(self, process=None):
        deadline = time.monotonic() + 300
        while not self._closed.wait(0.25):
            if process is not None and process.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {process.returncode}.")
            try:
                if self._health():
                    return True
            except urllib.error.URLError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("The server did not become ready within 5 minutes.")
        return False

    def _launch(self, args, config):
        self._output.clear()
        self._state("Starting llama-server…")
        with self._lock:
            if self._closed.is_set():
                return False
            process = self.process = subprocess.Popen(
                args,
                cwd=str(Path(args[0]).parent),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=(
                    {
                        key: value
                        for key, value in os.environ.items()
                        if key not in {"LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL"}
                    }
                    if (
                        config.get("vision_family") or config.get("load_vision") is False
                        or config.get("vision_fallback")
                    )
                    else None
                ),
            )
            self._config = config
            self._gpu_offload = None
            self._configured_gpu = _configured_gpu_offload(config, args)
        reader = threading.Thread(target=self._read_output, args=(process,), daemon=True)
        reader.start()
        try:
            if not self._wait_for_health(process):
                return False
        except OSError, RuntimeError:
            if process.poll() is None:
                self._stop_process(process)
            reader.join(timeout=1)
            self.process = None
            raise
        threading.Thread(target=self._watch_exit, args=(process, reader), daemon=True).start()
        return True

    def _backend_failed(self, backend, error):
        if backend == "cpu":
            return False
        # Invalid model files and unrelated launch flags should surface directly.
        output = (str(error) + "\n" + "\n".join(self._output)).casefold()
        if any(
            marker in output
            for marker in (
                "cuda error",
                "cuda failure",
                "cudacheck",
                "cublas_status",
                "cublas failed",
                "libcuda.so",
                "libcublas.so",
                "libcublaslt.so",
                "libcudart.so",
                "cuda driver version is insufficient",
                "no kernel image",
                "unsupported toolchain",
                "vk_error",
                "vk::",
                "vkerror",
                "device lost",
                "out of device memory",
                "libvulkan",
                "failed to allocate cuda",
                "failed to allocate vulkan",
                "no devices found",
            )
        ):
            return True
        prefix = "ggml_vulkan" if backend == "vulkan" else "ggml_cuda"
        return any(
            line.startswith(prefix)
            and any(word in line for word in ("fatal", "failed", "out of memory"))
            for line in output.splitlines()
        )

    def _read_output(self, process):
        with process.stdout:
            for line in iter(process.stdout.readline, b""):
                text = line.decode("utf-8", errors="replace").rstrip()[-2000:]
                self._output.append(text)
                # The first report is for the main model. A GPU-capable build can offload zero.
                if self._gpu_offload is None and (match := _GPU_LAYERS.search(text)):
                    with self._lock:
                        if process is self.process:
                            self._gpu_offload = int(match[1]) > 0

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
