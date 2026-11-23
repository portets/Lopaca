"""Install optional libraries in a local environment for Lopaca."""

import json
import os
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
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
        import_module(name)
        package, _, minimum = requirement.partition(">=")
        if minimum:
            target = tuple(map(int, minimum.split(".")))
            match = re.match(r"^\d+(?:\.\d+)*", version(package))
            actual = tuple(map(int, match[0].split("."))) if match else ()
            if (actual + (0,) * len(target))[:len(target)] < target:
                missing.append(name)
    except (ImportError, OSError):
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
    progress("Installing")
    try:
        _run([system_python, "-m", "venv", "--system-site-packages", str(environment)])
    except RuntimeError as exc:
        error = str(exc).lower()
        if not any(
            message in error
            for message in (
                "no module named venv",
                "no module named ensurepip",
                "ensurepip is not available",
            )
        ):
            raise
        progress("Downloading")
        uv = _install_uv(folder, stopped)
        if stopped():
            return
        progress("Installing")
        _run(
            [
                str(uv),
                "venv",
                "--no-python-downloads",
                "--allow-existing",
                "--python",
                system_python,
                "--system-site-packages",
                str(environment),
            ]
        )
        installer = [str(uv), "pip", "install", "--python", str(python)]
    else:
        installer = [str(python), "-m", "pip", "install"]
    if stopped():
        return
    probe = [str(python), "-c", _PROBE, json.dumps(_OPTIONAL_REQUIREMENTS)]
    missing = json.loads(_run(probe).splitlines()[-1])
    if missing:
        _run([*installer, *(_OPTIONAL_REQUIREMENTS[name] for name in missing)])
        if unavailable := json.loads(_run(probe).splitlines()[-1]):
            raise RuntimeError("Could not load optional libraries: " + ", ".join(unavailable))
    _run(
        [
            str(python),
            "-c",
            "import gi; gi.require_version('Gtk', '4.0'); gi.require_version('Adw', '1'); "
            "from gi.repository import Gtk, Adw",
        ]
    )
    if not stopped():
        (environment / ".lopaca-ready").write_text(_READY_VERSION)
