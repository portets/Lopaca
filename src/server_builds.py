"""Detect usable hardware and install pinned llama-server release assets."""

import ctypes as c
import logging
import os
import platform
import re
import shlex
import struct
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock

from settings import (
    EMBEDDING_DOWNLOAD,
    EMBEDDING_LEGACY_Q8,
    MODEL_DOWNLOADS,
    MODEL_FREE_VRAM_DISCOUNT_GIB,
    MODEL_RECOMMENDATIONS,
    MODEL_TOTAL_RAM_DISCOUNT_GIB,
    MODEL_TOTAL_VRAM_DISCOUNT_GIB,
    VISION_DOWNLOADS,
)
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
    r"^\s*((?:CUDA|Vulkan)\d+): (.+) \((?:(\d+) MiB, )?(\d+) MiB free\)\s*$", re.MULTILINE
)

BACKEND_LABELS = {
    "cuda-13.5": "CUDA 13",
    "cuda-12.8": "CUDA 12",
    "vulkan": "Vulkan",
    "cpu": "CPU",
}
_VISION_DOWNLOAD_LOCK = Lock()


@dataclass(frozen=True)
class GpuMemory:
    compute: int = 0
    total_mib: int | None = None
    free_mib: int | None = None

    def qualifies(self, vram_gib, compute):
        memory = self.total_mib
        discount = MODEL_TOTAL_VRAM_DISCOUNT_GIB
        if memory is None:
            memory = self.free_mib
            discount = MODEL_FREE_VRAM_DISCOUNT_GIB
        required = max(0, vram_gib - discount) * 1024
        return self.compute >= compute and memory is not None and memory >= required


def recommendation_for(backend, devices):
    kind = "cuda" if backend.startswith("cuda") else backend
    ram = total_ram_gib()
    return next(
        key
        for key, rule in MODEL_RECOMMENDATIONS.items()
        if ram >= max(0, rule.ram_gib - MODEL_TOTAL_RAM_DISCOUNT_GIB)
        and (not rule.backends or kind in rule.backends)
        and (
            not (rule.vram_gib or rule.cuda_compute)
            or any(device.qualifies(rule.vram_gib, rule.cuda_compute) for device in devices)
        )
    )


def total_ram_gib():
    with suppress(AttributeError, OSError, ValueError):
        return max(0, os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // (1 << 30))
    return 0


def _nvidia_board_memory(driver, device):
    """CUDA can report less than the board total because of driver reservations."""
    try:
        nvml = c.CDLL("libnvidia-ml.so.1")
        bus = c.create_string_buffer(32)
        if driver.cuDeviceGetPCIBusId(bus, len(bus), device) or nvml.nvmlInit_v2():
            return None
        try:
            handle = c.c_void_p()
            memory = (c.c_ulonglong * 3)()  # nvmlMemory_t: total, free, used in bytes
            if not nvml.nvmlDeviceGetHandleByPciBusId_v2(bus, c.byref(handle)) and not (
                nvml.nvmlDeviceGetMemoryInfo(handle, c.byref(memory))
            ):
                return memory[0] // 1024**2 or None
        finally:
            nvml.nvmlShutdown()
    except OSError, AttributeError:
        pass
    return None


def _cuda_devices(*, memory=True):
    """Query the installed NVIDIA driver without requiring CUDA Toolkit or nvidia-smi."""
    driver = c.CDLL("libcuda.so.1")
    version, count = c.c_int(), c.c_int()
    if driver.cuInit(0) or driver.cuDriverGetVersion(c.byref(version)):
        return 0, {}
    if driver.cuDeviceGetCount(c.byref(count)):
        return 0, {}
    devices = {}
    for index in range(count.value):
        device, major, minor = c.c_int(), c.c_int(), c.c_int()
        name = c.create_string_buffer(256)
        if (
            driver.cuDeviceGet(c.byref(device), index)
            or driver.cuDeviceGetName(name, len(name), device)
            or driver.cuDeviceGetAttribute(c.byref(major), 75, device)
            or driver.cuDeviceGetAttribute(c.byref(minor), 76, device)
        ):
            if not memory:
                return version.value, {}
            continue
        name = name.value.decode(errors="replace")
        compute = 10 * major.value + minor.value
        if not memory:
            devices[name] = GpuMemory(compute=compute)
            continue
        total = c.c_size_t()
        mib = None
        if hasattr(driver, "cuDeviceTotalMem_v2") and not driver.cuDeviceTotalMem_v2(
            c.byref(total), device
        ):
            mib = total.value // 1024**2 or None
        board_mib = _nvidia_board_memory(driver, device)
        # A large discrepancy can indicate a partition or a capped CUDA device.
        if board_mib and (mib is None or 0 <= board_mib - mib <= 512):
            mib = board_mib
        devices[name] = GpuMemory(compute=compute, total_mib=mib)
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


class _VulkanMemoryHeap(c.Structure):
    _fields_ = [("size", c.c_uint64), ("flags", c.c_uint32)]


class _VulkanMemoryProperties(c.Structure):
    _fields_ = [
        ("type_count", c.c_uint32),
        ("types", c.c_uint32 * 64),
        ("heap_count", c.c_uint32),
        ("heaps", _VulkanMemoryHeap * 16),
    ]


def _discrete_vulkan_devices():
    """Use the system Vulkan loader so enumeration works without vulkaninfo."""
    loader = c.CDLL("libvulkan.so.1")
    info = _VulkanInstanceInfo(s_type=1)  # VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
    instance = c.c_void_p()
    if loader.vkCreateInstance(c.byref(info), None, c.byref(instance)):
        return {}
    try:
        count = c.c_uint32()
        if loader.vkEnumeratePhysicalDevices(instance, c.byref(count), None):
            return {}
        devices = (c.c_void_p * count.value)()
        if loader.vkEnumeratePhysicalDevices(instance, c.byref(count), devices):
            return {}
        discrete, other = {}, set()
        for handle in devices[: count.value]:
            # VkPhysicalDeviceProperties starts with five 32-bit fields and deviceName[256].
            # Reserve aligned space for the trailing UUID, limits and sparse properties.
            properties = (c.c_uint64 * 512)()
            loader.vkGetPhysicalDeviceProperties(c.c_void_p(handle), c.byref(properties))
            *_, kind, raw_name = struct.unpack_from("=5I256s", properties)
            name = raw_name.split(b"\0", 1)[0].decode(errors="replace")
            if kind == 2:  # VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
                memory = _VulkanMemoryProperties()
                loader.vkGetPhysicalDeviceMemoryProperties(c.c_void_p(handle), c.byref(memory))
                # Vulkan can expose multiple heaps backed by the same VRAM.
                total = max(
                    (heap.size for heap in memory.heaps[: memory.heap_count] if heap.flags & 1),
                    default=0,
                )
                discrete[name] = GpuMemory(total_mib=total // 1024**2 or None)
            else:
                other.add(name)
        return {name: memory for name, memory in discrete.items() if name not in other}
    finally:
        loader.vkDestroyInstance(instance, None)


def _candidates(arch):
    version, cuda = 0, {}
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
        names = {name: memory for name, memory in cuda.items() if low <= memory.compute <= high}
        if (arch, backend) in _ASSETS and version >= minimum and names:
            yield backend, names
    try:
        if names := _discrete_vulkan_devices():
            yield "vulkan", names
    except (OSError, AttributeError) as exc:
        logging.debug("Vulkan probe unavailable: %s", exc)
    yield "cpu", {}


def detect_recommendation(backend=None):
    arch = _server_arch()
    if arch is not None:
        for candidate, devices in _candidates(arch):
            if backend is None or candidate == backend or (
                backend == "cuda" and candidate.startswith("cuda")
            ):
                return recommendation_for(candidate, devices.values())
    return recommendation_for("cpu", ())


def recommendation_for_config(config):
    if (profile := config.get("recommendation")) in MODEL_RECOMMENDATIONS:
        return profile
    backend = config.get("backend")
    if not backend and (executable := config.get("executable")):
        backend = custom_server_backend(executable) or "cpu"
    return detect_recommendation(backend)


def detect_vision_precision(config, *, fallback="FP16"):
    """Use BF16 only for backends with a known native BF16 path."""
    backend = config.get("backend")
    if not backend:
        if executable := config.get("executable"):
            backend = custom_server_backend(executable)
        elif arch := _server_arch():
            backend, _ = next(_candidates(arch))
    if not backend:
        return fallback
    if backend == "cpu":
        with suppress(OSError):
            flags = re.findall(
                r"^(?:flags|Features)\s*:\s*(.+)$",
                Path("/proc/cpuinfo").read_text(),
                re.MULTILINE,
            )
            if flags and all(
                {"avx512_bf16", "amx_bf16", "bf16"} & set(row.split()) for row in flags
            ):
                return "BF16"
    if backend and backend.startswith("cuda"):
        with suppress(OSError, AttributeError, ValueError):
            overrides = {arg.split("=", 1)[0] for arg in shlex.split(config.get("extra", ""))}
            if not overrides & {
                "--no-mmproj-offload",
                "--device",
                "-dev",
                "--mmproj-device",
                "-mmdev",
                "--rpc",
            } and not any(
                os.environ.get(key)
                for key in ("LLAMA_ARG_DEVICE", "LLAMA_ARG_MMPROJ_OFFLOAD", "MTMD_BACKEND_DEVICE")
            ):
                _, devices = _cuda_devices(memory=False)
                if devices and all(gpu.compute >= 80 for gpu in devices.values()):
                    return "BF16"
    return "FP16"


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


def _server_info(executable, option, *, timeout=20):
    result = subprocess.run(
        [str(executable), option],
        cwd=executable.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        text=True,
        errors="replace",
    )
    if result.returncode:
        raise RuntimeError(result.stdout.strip()[-2000:] or f"llama-server {option} failed.")
    return result.stdout


def custom_server_backend(filename):
    try:
        executable = Path(filename).expanduser().resolve()
        output = _server_info(executable, "--list-devices", timeout=3)
    except OSError, RuntimeError, subprocess.TimeoutExpired:
        return None
    _, header, devices = output.partition("Available devices:")
    if not header:
        return None
    names = [identifier for identifier, *_ in _DEVICE_LINE.findall(devices)]
    if names and all(name.startswith("CUDA") for name in names):
        return "cuda"
    if names and all(name.startswith("Vulkan") for name in names):
        return "vulkan"
    if devices.strip() == "(none)":
        return "cpu"
    return None


def _verify_devices(executable, backend, names):
    output = _server_info(executable, "--list-devices")
    if "Available devices:" not in output:
        raise RuntimeError(output.strip()[-2000:] or "Device enumeration failed.")
    if backend == "cpu":
        return [], recommendation_for(backend, ())
    prefix = "CUDA" if backend.startswith("cuda") else "Vulkan"
    devices = {
        identifier: GpuMemory(
            compute=names[description.strip()].compute,
            total_mib=names[description.strip()].total_mib or (int(total) if total else None),
            free_mib=int(free),
        )
        for identifier, description, total, free in _DEVICE_LINE.findall(output)
        if identifier.startswith(prefix) and description.strip() in names
    }
    if not devices:
        raise RuntimeError(
            f"No compatible {BACKEND_LABELS[backend]} device reported by llama-server."
        )
    recommendation = recommendation_for(backend, devices.values())
    rule = MODEL_RECOMMENDATIONS[recommendation]
    if rule.cuda_compute:
        devices = {
            identifier: gpu
            for identifier, gpu in devices.items()
            if gpu.qualifies(rule.vram_gib, rule.cuda_compute)
        }
    return list(devices), recommendation


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


def model_path(folder, key):
    return folder / "models" / MODEL_DOWNLOADS[key].filename


def model_download_path(folder, key):
    return model_path(folder, key).with_suffix(".gguf.download")


def model_vision_family(folder, filename):
    path = Path(filename or ".").expanduser()
    return next(
        (
            asset.vision_family
            for key, asset in MODEL_DOWNLOADS.items()
            if model_path(folder, key) == path
        ),
        "",
    )


def vision_path(folder, family, precision):
    return folder / "models" / f"qwen{family}-mmproj-{precision}.gguf"


def vision_file_precision(folder, family, filename):
    if filename:
        path = Path(filename).expanduser()
        for precision in VISION_DOWNLOADS[family]:
            if path == vision_path(folder, family, precision):
                return precision
    return None


def embedding_path(folder):
    return folder / "models" / EMBEDDING_DOWNLOAD.filename


def legacy_embedding_file(filename):
    with suppress(OSError, RuntimeError):
        path = Path(filename or ".").expanduser()
        return path.name == EMBEDDING_LEGACY_Q8[0] and path.stat().st_size == EMBEDDING_LEGACY_Q8[1]
    return False


def managed_vision_file(folder, filename):
    return any(vision_file_precision(folder, family, filename) for family in VISION_DOWNLOADS)


def install_model(folder, key, stopped, on_progress, *, discard=lambda: False):
    asset = MODEL_DOWNLOADS[key]
    return _install_model_asset(folder, asset, stopped, on_progress, "model", key, discard)


def install_embedding(folder, stopped, on_progress, *, discard=lambda: False):
    return _install_model_asset(
        folder, EMBEDDING_DOWNLOAD, stopped, on_progress, "embedding", None, discard
    )


def _embedding_threads():
    """Size the worker pool from Linux's usable physical cores and SMT siblings."""
    try:
        available = os.sched_getaffinity(0)
    except AttributeError, OSError:
        return max(1, min(24, (os.cpu_count() or 1) - 1))
    try:
        topology = Path("/sys/devices/system/cpu")
        siblings = [
            (topology / f"cpu{cpu}/topology/thread_siblings_list").read_text().strip()
            for cpu in available
        ]
    except OSError:
        siblings = []
    # Linux reports the same sibling list for every logical CPU on a physical core.
    # Only count siblings that the process can actually run on.
    if siblings and all(siblings):
        if len(set(siblings)) < len(available):
            return max(2, min(24, len(available) - 2))
        return max(1, len(available) - 1)
    return max(1, min(24, len(available) - 1))


def embedding_launch_parameters():
    threads = _embedding_threads()
    return (
        "--embedding --pooling last --device none --n-gpu-layers 0 "
        f"--ctx-size 4096 --parallel 1 --threads {threads} --threads-batch {threads}"
    )


def _install_model_asset(folder, asset, stopped, on_progress, stage, item, discard):
    model = folder / "models" / asset.filename
    expected = {"size": asset.size, "sha256": asset.sha256, "stopped": stopped}
    if model.exists():
        if not matches_download(model, **expected):
            raise ValueError(f"The file at {model} differs from the default model.")
    else:
        _download_gguf(model, asset.url, expected, on_progress, stage, item, discard)
    return model


def install_vision(folder, family, precision, stopped, on_progress, *, discard=lambda: False):
    asset = VISION_DOWNLOADS[family][precision]
    path = vision_path(folder, family, precision)
    item = (family, precision)
    # Connecting and downloading another model can request the same projector.
    with _VISION_DOWNLOAD_LOCK:
        if stopped():
            raise InterruptedError("Download cancelled.")
        if not path.is_file() or path.stat().st_size != asset.size:
            expected = {"size": asset.size, "sha256": asset.sha256, "stopped": stopped}
            try:
                _download_gguf(
                    path, asset.url(precision), expected, on_progress, "vision", item, discard
                )
            finally:
                on_progress("vision", item, None)
    return path


def _download_gguf(path, url, expected, on_progress, stage, item, discard):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".gguf.download")
    resuming = partial.is_file()
    on_progress(stage, item, 0)
    report = _report_progress(on_progress, stage, item)
    try:
        try:
            download_verified_file(url, partial, on_progress=report, **expected)
        except DownloadChecksumError:
            if not resuming:
                raise
            download_verified_file(url, partial, on_progress=report, **expected)
        if expected["stopped"]():
            raise InterruptedError("Download cancelled.")
        partial.replace(path)
    finally:
        if discard():
            partial.unlink(missing_ok=True)


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


def _verified_servers(folder, arch, candidates, stopped, on_progress):
    verified = False
    failures = []
    for backend, names in candidates:
        if stopped():
            return
        try:
            executable = _install_server(
                folder / "server", arch, backend, stopped, _report_progress(on_progress, "server")
            )
            devices, recommendation = _verify_devices(executable, backend, names)
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
            "recommendation": recommendation,
            "version": version[1].strip() if version else "",
        }
        verified = True
        on_progress("server", server, None)
        yield server
    if not verified and not stopped():
        raise RuntimeError(
            "No packaged llama-server build could be prepared.\n\n" + "\n".join(failures)
        )


def default_model_configs(
    folder, stopped, on_progress, *, model_key=None, installed=False, discard=lambda: False,
):
    """Start both downloads together when the hardware identifies a model."""
    arch = _server_arch()
    if arch is None:
        raise ValueError("Automatic setup requires Linux x64 or arm64.")
    candidates = tuple(_candidates(arch))
    backend, devices = candidates[0]
    recommendation = recommendation_for(backend, devices.values())
    predicted = model_key or MODEL_RECOMMENDATIONS[recommendation].balanced
    required_compute = MODEL_DOWNLOADS[model_key].minimum_cuda_compute if model_key else 0
    servers = _verified_servers(folder, arch, candidates, stopped, on_progress)
    if required_compute:
        servers = (
            server
            for server in servers
            if MODEL_RECOMMENDATIONS[server["recommendation"]].cuda_compute >= required_compute
        )
    cancel_model = Event()

    def download(key, cancelled):
        try:
            return install_model(folder, key, cancelled, on_progress, discard=discard)
        finally:
            on_progress("model", key, None)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-download") as downloads:
        pending = None
        # If only the server can report VRAM, wait for its probe before transferring a model.
        memory_known = all(gpu.total_mib is not None for gpu in devices.values())
        if (
            not installed
            and (model_key or memory_known)
            and MODEL_RECOMMENDATIONS[recommendation].cuda_compute >= required_compute
        ):
            pending = downloads.submit(
                download, predicted, lambda: stopped() or cancel_model.is_set()
            )
        try:
            first = next(servers, None)
            if stopped():
                return
            if first is None:
                raise RuntimeError("This model requires a compatible Blackwell or newer CUDA GPU.")
            key = model_key or MODEL_RECOMMENDATIONS[first["recommendation"]].balanced
            if pending is not None and key == predicted:
                model = pending.result()
            else:
                cancel_model.set()
                if pending is not None:
                    with suppress(InterruptedError, OSError, ValueError):
                        pending.result()
                    on_progress("model", predicted, None)
                model = model_path(folder, key) if installed else download(key, stopped)
        finally:
            cancel_model.set()

    if stopped():
        return
    yield {**first, "model": str(model)}
    for server in servers:
        if stopped():
            return
        fallback = model_key or MODEL_RECOMMENDATIONS[server["recommendation"]].balanced
        if fallback != key:
            key = fallback
            model = download(key, stopped)
        yield {**server, "model": str(model)}
