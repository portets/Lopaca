"""Install optional libraries in a local environment for Lopaca."""

import json
import logging
import os
import site
import subprocess
import sys
import sysconfig
from contextlib import suppress
from http.client import HTTPException
from importlib import invalidate_caches
from pathlib import Path
from shutil import which
from tarfile import TarError
from tempfile import TemporaryDirectory

from rag import _OPTIONAL_COMPRESSION_LIBRARIES, DOCUMENT_LIBRARIES
from server_builds import _extract, _server_arch
from web import download_verified_file

_UV_VERSION = "0.10.12"
_UV_ASSETS = {
    "x64": ("x86_64", 22848740, "ec72570c9d1f33021aa80b176d7baba390de2cfeb1abcbefca346d563bf17484"),
    "arm64": (
        "aarch64",
        21389445,
        "0ed7d20f49f6b9b60d45fdfcac28f3ac01a671a6ef08672401ed2833423fea2a",
    ),
}
_OPTIONAL_REQUIREMENTS = (
    dict(DOCUMENT_LIBRARIES.values())
    | dict(_OPTIONAL_COMPRESSION_LIBRARIES.values())
    | {
        "py7zr": "py7zr>=1.1.4",
        "hf_xet": "hf_xet>=1.7",
        "huggingface_hub": "huggingface_hub>=1.0",
        "ocrmypdf": "ocrmypdf>=17",
        "pypdfium2": "pypdfium2",
        "numpy": "numpy",
    }
)
_READY_VERSION = "3"
_PROBE = r"""
import json
import re
from importlib import import_module
from importlib.metadata import version
import sys

missing = []
for name, requirement in json.loads(sys.argv[1]).items():
    try:
        if name == "zstandard":
            try:
                from compression import zstd
            except ImportError:
                pass
            else:
                continue
        import_module(name)
        package, _, minimum = requirement.partition(">=")
        if minimum:
            target = tuple(map(int, minimum.split(".")))
            match = re.match(r"^\d+(?:\.\d+)*", version(package))
            actual = tuple(map(int, match[0].split("."))) if match else ()
            if (actual + (0,) * len(target))[:len(target)] < target:
                missing.append(name)
    except Exception:
        missing.append(name)
print(json.dumps(missing))
"""


def optional_python(folder):
    environment = folder / ".venv"
    python = environment / "bin" / "python"
    if (environment / ".lopaca-ready").is_file() and os.access(python, os.X_OK):
        return python
    return None


def optional_features_ready(folder):
    if optional_python(folder):
        with suppress(OSError):
            return (folder / ".venv" / ".lopaca-ready").read_text() == _READY_VERSION
    return False


def launch_with_optional_python(folder):
    python = optional_python(folder)
    if python and Path(sys.prefix).resolve() != python.parent.parent.resolve():
        os.execv(python, [str(python), *sys.argv])


def activate_optional_packages(folder):
    """Expose packages from a venv built with the running interpreter."""
    python = optional_python(folder)
    if not python or python.resolve() != Path(sys.executable).resolve():
        return False

    prefix = str(python.parent.parent)
    scheme = sysconfig.get_paths("venv", vars={"base": prefix, "platbase": prefix})
    library_paths = dict.fromkeys((scheme["purelib"], scheme["platlib"]))
    directories = [path for path in library_paths if Path(path).is_dir()]
    system_sites = {*site.getsitepackages(), site.getusersitepackages()}
    for directory in reversed(directories):
        if directory in sys.path:
            continue
        position = next(
            (index for index, path in enumerate(sys.path) if path in system_sites), len(sys.path)
        )
        site.addsitedir(directory)
        sys.path.remove(directory)
        sys.path.insert(position, directory)
        system_sites.add(directory)
    invalidate_caches()
    return bool(directories)


def _install_uv(folder, stopped):
    arch = _server_arch()
    if arch not in _UV_ASSETS:
        raise ValueError("Optional feature setup requires Linux x64 or arm64.")
    architecture, size, digest = _UV_ASSETS[arch]
    root = f"uv-{architecture}-unknown-linux-gnu"
    tools = folder / "tools"
    uv = tools / "uv"
    if uv.is_file():
        return uv
    tools.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=tools, prefix=".uv-") as staging:
        destination = Path(staging)
        archive = destination / f"{root}.tar.gz"
        download_verified_file(
            f"https://github.com/astral-sh/uv/releases/download/{_UV_VERSION}/{archive.name}",
            archive,
            size=size,
            sha256=digest,
            stopped=stopped,
        )
        extracted = _extract(archive, root, destination, {"uv"}.__contains__, {"uv"}) / "uv"
        extracted.chmod(0o755)
        extracted.replace(uv)
    return uv


def _run(command):
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=600,
    )
    if result.returncode:
        raise RuntimeError(result.stdout.strip()[-3000:] or f"{command[0]} failed.")
    return result.stdout


def install_optional_features(folder, stopped, progress):
    if optional_features_ready(folder):
        return
    if stopped():
        return
    environment = folder / ".venv"
    python = environment / "bin" / "python"
    system_python = str(Path(sys.executable).resolve())
    cached_uv = folder / "tools" / "uv"
    uv = str(cached_uv) if os.access(cached_uv, os.X_OK) else which("uv")
    try:
        if not uv:
            progress("Downloading")
            uv = str(_install_uv(folder, stopped))
        if stopped():
            return
        progress("Installing")
        _run(
            [
                uv,
                "venv",
                "--no-python-downloads",
                "--allow-existing",
                "--python",
                system_python,
                "--system-site-packages",
                str(environment),
            ]
        )
    except (OSError, ValueError, RuntimeError, HTTPException, TarError, subprocess.TimeoutExpired):
        if stopped():
            return
        logging.debug("uv setup failed. Using system Python and pip.", exc_info=True)
        progress("Installing")
        _run([system_python, "-m", "venv", "--system-site-packages", str(environment)])
        installer = [str(python), "-m", "pip", "install"]
    else:
        installer = [uv, "pip", "install", "--python", str(python)]
    if stopped():
        return
    probe = [str(python), "-c", _PROBE, json.dumps(_OPTIONAL_REQUIREMENTS)]
    missing = json.loads(_run(probe).splitlines()[-1])
    if missing:
        requirements = [_OPTIONAL_REQUIREMENTS[name] for name in missing]
        # Try wheels first, with an exception for pure-Python odfpy.
        wheel_installer = [*installer, "--only-binary=:all:", "--no-binary=odfpy"]
        try:
            _run([*wheel_installer, *requirements])
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            logging.debug("Optional package batch failed: %s", exc)
            retries = (wheel_installer, installer) if which("gcc") else (wheel_installer,)
            for name in json.loads(_run(probe).splitlines()[-1]):
                requirement = _OPTIONAL_REQUIREMENTS[name]
                for command in retries:
                    if stopped():
                        return
                    try:
                        _run([*command, requirement])
                    except (RuntimeError, subprocess.TimeoutExpired) as exc:
                        logging.debug("Optional install failed for %s: %s", requirement, exc)
                    else:
                        break
        missing = json.loads(_run(probe).splitlines()[-1])
    if stopped():
        return
    _run(
        [
            str(python),
            "-c",
            "import gi; gi.require_version('Gtk', '4.0'); gi.require_version('Adw', '1'); "
            "from gi.repository import Gtk, Adw",
        ]
    )
    if stopped():
        return
    # Allow the app and the downloader to use a working partial installation.
    if len(missing) < len(_OPTIONAL_REQUIREMENTS):
        (environment / ".lopaca-ready").write_text("partial" if missing else _READY_VERSION)
    if missing:
        unavailable = ", ".join(_OPTIONAL_REQUIREMENTS[name] for name in missing)
        raise RuntimeError(
            f"Not yet available for Python {sys.version_info.major}.{sys.version_info.minor}: "
            f"{unavailable}. Other optional features installed. Install GCC and retry, or retry "
            "in Settings later."
        )
