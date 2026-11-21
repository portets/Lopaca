"""Detect usable hardware and install pinned llama-server release assets."""

import ctypes as c
import logging
import platform
import re
import struct
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

from web import DownloadChecksumError, download_verified_file, matches_download

_RELEASE = "b12359"
_RELEASE_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{_RELEASE}/"
# Sizes and SHA-256 digests from the upstream release asset metadata.
_ASSETS = {
    ("x64", "cpu"): (23311084, "0bc1f091e8916102b6d8f2cf944233a620e90a0f856c72d0ce41150c65c2e72e"),
    ("arm64", "cpu"): (
        18949862,
        "dc7aa17a67884619b7223870aca9fd6addbc01717713b04a714add1c31137359",
    ),
    ("x64", "vulkan"): (
        36153377,
        "19b60f00395a3f4954e46a5037695bc1d843404fc9d3124f82a612ecd8fbe9f0",
    ),
    ("arm64", "vulkan"): (
        29435221,
        "3f8ae812a58ac92e5a596ca9fbc988f52854d147c5a7fe1264dddc90c1d085b4",
    ),
    ("x64", "cuda-13.5"): (
        178419847,
        "9440f417615bed0e191acef03bff0249117d2be74335358265a6d28475a9baf0",
    ),
    ("arm64", "cuda-13.5"): (
        173529798,
        "e12fd3ca9a599349cac5ec767825e17f8cab9dc2dab8d8e5b85d50ce21ef6d7d",
    ),
    ("x64", "cuda-12.8"): (
        199223370,
        "50c937bec5bbd9f74c99138034f7403283ee21f1943f2429d7b1e4c079eafc11",
    ),
}
_CUDA_RUNTIMES = {
    ("x64", "cuda-13.5"): (
        444078822,
        "976a9146bb8a487a6ea701224d7fb14b05e494ab3c8f5c1799ffc4e5805d640c",
    ),
    ("arm64", "cuda-13.5"): (
        559874231,
        "998f62318999710d398dad40821d1d037731f360791e41628d9dfcde207909d1",
    ),
    ("x64", "cuda-12.8"): (
        594371571,
        "75a12d227f44a391608992329241765343bd143ee9445b90dc62e154ee726e5d",
    ),
}
_SERVER_LIBRARY = re.compile(
    r"(?:libggml(?:-base|-cpu(?:-[\w-]+)?|-cuda|-vulkan)?"
    r"|libllama(?:-server-impl|-common)?|libmtmd)\.so(?:\.\d+)*"
)
_DEVICE_LINE = re.compile(
    r"^\s*((?:CUDA|Vulkan)\d+): (.+) \(\d+ MiB, \d+ MiB free\)\s*$", re.MULTILINE
)

_MODEL_NAME = "Qwen3.8-27B-UD-Q4_K_M.gguf"
_MODEL_URL = (
    "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/"
    f"4af51b3542f775b0a8e4b310bfb3f17e01c306c9/{_MODEL_NAME}?download=true"
)
_MODEL_SIZE = 16_564_259_744
_MODEL_SHA256 = "1bce504dfecd84153666928a41d8a66706f0158d3a317bc5196b72819d7dc529"

BACKEND_LABELS = {
    "cuda-13.5": "CUDA 13",
    "cuda-12.8": "CUDA 12",
    "vulkan": "Vulkan",
    "cpu": "CPU",
}


def _cuda_devices():
    """Query the installed NVIDIA driver without requiring CUDA Toolkit or nvidia-smi."""
    driver = c.CDLL("libcuda.so.1")
    version, count = c.c_int(), c.c_int()
    if driver.cuInit(0) or driver.cuDriverGetVersion(c.byref(version)):
        return 0, []
    if driver.cuDeviceGetCount(c.byref(count)):
        return 0, []
    devices = []
    for index in range(count.value):
        device, major, minor = c.c_int(), c.c_int(), c.c_int()
        name = c.create_string_buffer(256)
        if (
            driver.cuDeviceGet(c.byref(device), index)
            or driver.cuDeviceGetName(name, len(name), device)
            or driver.cuDeviceGetAttribute(c.byref(major), 75, device)
            or driver.cuDeviceGetAttribute(c.byref(minor), 76, device)
        ):
            continue
        devices.append((name.value.decode(errors="replace"), 10 * major.value + minor.value))
    return version.value, devices


class _VulkanInstanceInfo(c.Structure):
    _fields_ = [
        ("s_type", c.c_uint32),
        ("next", c.c_void_p),
        ("flags", c.c_uint32),
        ("application", c.c_void_p),
        ("layer_count", c.c_uint32),
        ("layers", c.c_void_p),
        ("extension_count", c.c_uint32),
        ("extensions", c.c_void_p),
    ]


def _discrete_vulkan_names():
    """Use the system Vulkan loader so enumeration works without vulkaninfo."""
    loader = c.CDLL("libvulkan.so.1")
    info = _VulkanInstanceInfo(s_type=1)  # VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
    instance = c.c_void_p()
    if loader.vkCreateInstance(c.byref(info), None, c.byref(instance)):
        return set()
    try:
        count = c.c_uint32()
        if loader.vkEnumeratePhysicalDevices(instance, c.byref(count), None):
            return set()
        devices = (c.c_void_p * count.value)()
        if loader.vkEnumeratePhysicalDevices(instance, c.byref(count), devices):
            return set()
        discrete, other = set(), set()
        for handle in devices[: count.value]:
            # VkPhysicalDeviceProperties starts with five 32-bit fields and deviceName[256].
            # Reserve aligned space for the trailing UUID, limits and sparse properties.
            properties = (c.c_uint64 * 512)()
            loader.vkGetPhysicalDeviceProperties(c.c_void_p(handle), c.byref(properties))
            *_, kind, raw_name = struct.unpack_from("=5I256s", properties)
            name = raw_name.split(b"\0", 1)[0].decode(errors="replace")
            (discrete if kind == 2 else other).add(name)  # VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
        return discrete - other
    finally:
        loader.vkDestroyInstance(instance, None)


def _candidates(arch):
    version, cuda = 0, []
    try:
        version, cuda = _cuda_devices()
    except (OSError, AttributeError) as exc:
        logging.debug("CUDA probe unavailable: %s", exc)
    # Prefer a driver that advertises the actual toolkit version. Some release kernels
    # ship as PTX and require a driver capable of compiling that toolkit's PTX.
    for backend, minimum, low, high in (
        ("cuda-13.5", 13050, 75, 999),
        ("cuda-12.8", 12080, 50, 120),
    ):
        names = {name for name, capability in cuda if low <= capability <= high}
        if (arch, backend) in _ASSETS and version >= minimum and names:
            yield backend, names
    try:
        if names := _discrete_vulkan_names():
            yield "vulkan", names
    except (OSError, AttributeError) as exc:
        logging.debug("Vulkan probe unavailable: %s", exc)
    yield "cpu", set()


def _extract(archive_path, root, destination, allowed, required):
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {}
        for member in archive:
            parent, _, name = member.name.rpartition("/")
            if parent != root or not allowed(name):
                continue
            if name in members or not (member.isfile() or member.issym()):
                raise ValueError(f"Invalid release member: {member.name}")
            members[name] = member
        if not required <= members.keys() or any(not members[name].isfile() for name in required):
            raise ValueError(f"The release archive is missing required files: {archive_path.name}")
        for member in members.values():
            if member.issym() and member.linkname not in members:
                raise ValueError(f"Invalid release library link: {member.name}")
        archive.extractall(destination, members=members.values(), filter="data")
    return destination / root


def _fetch_asset(root, asset, staging, stopped, on_progress):
    archive = staging / f"{root}.tar.gz"
    size, sha256 = asset
    download_verified_file(
        _RELEASE_URL + archive.name,
        archive,
        size=size,
        sha256=sha256,
        stopped=stopped,
        on_progress=on_progress,
    )
    return archive


def _install_server(server, arch, backend, stopped, on_progress=None):
    key = arch, backend
    executable = _server_path(server, arch, backend)
    target = executable.parent
    if executable.is_file():
        return executable
    suffix = arch if backend == "cpu" else f"{backend}-{arch}"
    archive_root = f"llama-{_RELEASE}-bin-ubuntu-{suffix}"
    server.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=server, prefix=".install-") as temporary:
        staging = Path(temporary)
        server_asset = _ASSETS[key]
        runtime_asset = _CUDA_RUNTIMES.get(key)
        total = server_asset[0] + (runtime_asset[0] if runtime_asset else 0)

        def report(current, _total):
            if on_progress:
                on_progress(current, total)

        archive = _fetch_asset(archive_root, server_asset, staging, stopped, report)
        extracted = _extract(
            archive,
            f"llama-{_RELEASE}",
            staging,
            lambda name: name == "llama-server" or _SERVER_LIBRARY.fullmatch(name),
            {"llama-server"},
        )
        archive.unlink()
        if runtime_asset:
            root = f"cudart-{archive_root}"
            archive = _fetch_asset(
                root,
                runtime_asset,
                staging,
                stopped,
                lambda current, size: report(server_asset[0] + current, size),
            )
            major = backend.removeprefix("cuda-").split(".", 1)[0]
            libraries = {f"lib{name}.so.{major}" for name in ("cudart", "cublas", "cublasLt")}
            runtime = _extract(archive, root, staging, libraries.__contains__, libraries)
            for library in libraries:
                (runtime / library).rename(extracted / library)
        extracted.rename(target)
    return executable


def _server_info(executable, option):
    result = subprocess.run(
        [str(executable), option],
        cwd=executable.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=20,
        text=True,
        errors="replace",
    )
    if result.returncode:
        raise RuntimeError(result.stdout.strip()[-2000:] or f"llama-server {option} failed.")
    return result.stdout


def _verify_devices(executable, backend, names):
    output = _server_info(executable, "--list-devices")
    if "Available devices:" not in output:
        raise RuntimeError(output.strip()[-2000:] or "Device enumeration failed.")
    if backend == "cpu":
        return []
    prefix = "CUDA" if backend.startswith("cuda") else "Vulkan"
    devices = [
        identifier
        for identifier, description in _DEVICE_LINE.findall(output)
        if identifier.startswith(prefix) and description.strip() in names
    ]
    if not devices:
        raise RuntimeError(
            f"No compatible {BACKEND_LABELS[backend]} device reported by llama-server."
        )
    return devices


def _server_path(server, arch, backend):
    return server / f"{_RELEASE}-{arch}-{backend}" / "llama-server"


def _server_arch():
    if platform.system() != "Linux":
        return None
    return {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(
        platform.machine().lower()
    )


def cached_server_config(folder):
    """Recognize a bundled server published before model setup completed."""
    arch = _server_arch()
    for backend in BACKEND_LABELS:
        if (arch, backend) in _ASSETS:
            executable = _server_path(folder / "server", arch, backend)
            if executable.is_file():
                return {"executable": str(executable), "backend": backend, "version": _RELEASE}
    return None


def default_model_path(folder):
    return folder / "models" / _MODEL_NAME


def default_model_download_path(folder):
    return default_model_path(folder).with_suffix(".gguf.download")


def _install_model(folder, stopped, on_progress):
    model = default_model_path(folder)
    model.parent.mkdir(parents=True, exist_ok=True)
    expected = {"size": _MODEL_SIZE, "sha256": _MODEL_SHA256, "stopped": stopped}
    if model.exists():
        if not matches_download(model, **expected):
            raise ValueError(f"The file at {model} differs from the default model.")
    else:
        partial = default_model_download_path(folder)
        resuming = partial.is_file()
        report = _report_progress(on_progress, "model")
        try:
            download_verified_file(_MODEL_URL, partial, on_progress=report, **expected)
        except DownloadChecksumError:
            if not resuming:
                raise
            download_verified_file(
                _MODEL_URL,
                partial,
                on_progress=_report_progress(on_progress, "model"),
                **expected,
            )
        partial.rename(model)
    return model


def _report_progress(callback, stage, server=None):
    last = -1

    def report(current, total):
        nonlocal last
        percent = current * 100 // total
        if last != percent:
            last = percent
            callback(stage, server, percent)

    report(0, 1)
    return report


def _verified_servers(folder, arch, stopped, on_progress):
    verified = False
    failures = []
    for backend, names in _candidates(arch):
        if stopped():
            return
        try:
            executable = _install_server(
                folder / "server", arch, backend, stopped, _report_progress(on_progress, "server")
            )
            devices = _verify_devices(executable, backend, names)
            version = re.search(
                r"^version: (.+)", _server_info(executable, "--version"), re.MULTILINE
            )
        except InterruptedError:
            raise
        except (
            OSError,
            ValueError,
            RuntimeError,
            tarfile.TarError,
            subprocess.TimeoutExpired,
        ) as exc:
            failures.append(f"{BACKEND_LABELS[backend]}: {exc}")
            logging.warning("Could not prepare %s llama-server: %s", backend, exc)
            continue
        server = {
            "executable": str(executable),
            "backend": backend,
            "devices": devices,
            "version": version[1].strip() if version else "",
        }
        verified = True
        on_progress("server", server, None)
        yield server
    if not verified and not stopped():
        raise RuntimeError(
            "No packaged llama-server build could be prepared.\n\n" + "\n".join(failures)
        )


def default_model_configs(folder, stopped, on_progress=lambda *_: None):
    """Download the model alongside server setup and share it across backend fallbacks."""
    arch = _server_arch()
    if arch is None:
        raise ValueError("Automatic setup requires Linux x64 or arm64.")

    cancel_model = Event()

    def prepare_model():
        model = _install_model(folder, lambda: stopped() or cancel_model.is_set(), on_progress)
        if not stopped():
            on_progress("model", None, None)
        return model

    servers = _verified_servers(folder, arch, stopped, on_progress)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-download") as downloads:
        model_download = downloads.submit(prepare_model)
        try:
            first_server = next(servers, None)
            if first_server is None:
                return
            model = model_download.result()
        finally:
            cancel_model.set()
    for server in chain((first_server,), servers):
        if stopped():
            return
        yield {**server, "model": str(model)}
