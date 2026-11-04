"""Read-only runtime information and approval-gated command helpers."""

import ast
import io
import json
import locale
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from rag import read_document_stream

_COMMAND_BLOCK_PATTERN = re.compile(r"```(?:bash|sh|shell)\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


def _tool(name, description, properties=None, required=()):
    parameters = {"type": "object", "properties": properties or {}, "required": list(required)}
    if properties is not None:
        parameters["additionalProperties"] = False
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


# These tools run in Lopaca via handle_tool_call, not llama-server's built-in tool API.
GET_OSINFO_TOOL = _tool(
    "get_osinfo",
    "Get host OS, distribution, architecture and desktop details when platform "
    "information matters. Does not read user files.",
)
GET_DATETIME_TOOL = _tool(
    "get_datetime",
    "Returns the current local date and time in ISO 8601 format, "
    "with the time zone name, weekday and preferred clock_format (12h or 24h). "
    "Use clock_format when presenting times unless the user requests otherwise.",
)
RUNTIME_TOOLS = [GET_OSINFO_TOOL, GET_DATETIME_TOOL]


def get_osinfo():
    """Read platform details without running commands or exposing personal paths."""
    info = {
        "os": platform.system(),
        "release": platform.release(),
        "architecture": platform.machine(),
    }
    if info["os"] == "Linux":
        with suppress(OSError):
            info["distribution"] = platform.freedesktop_os_release().get("PRETTY_NAME", "Linux")
        if desktop := os.environ.get("XDG_CURRENT_DESKTOP"):
            info["desktop"] = desktop
    return info


@cache
def _desktop_clock_settings():
    """Use GNOME's preference when available; keep the runtime helpers usable without GI."""
    desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").casefold()
    if desktop and "gnome" not in desktop and "unity" not in desktop:
        return None
    try:
        from gi.repository import Gio

        source = Gio.SettingsSchemaSource.get_default()
        schema = source.lookup("org.gnome.desktop.interface", True) if source else None
        if schema and schema.has_key("clock-format"):
            return Gio.Settings.new_full(schema, None, None)
    except ImportError, RuntimeError:
        pass
    return None


def get_clock_format():
    """Read the current desktop preference, falling back to the process's time locale."""
    settings = _desktop_clock_settings()
    if settings is not None and (value := settings.get_string("clock-format")) in ("12h", "24h"):
        return value
    try:
        pattern = locale.nl_langinfo(locale.T_FMT)
    except AttributeError, ValueError:
        return "24h"
    return "12h" if re.search(r"%(?:E|O)?[Ilrp]", pattern) else "24h"


def get_datetime():
    """Read the host clock; use an IANA zone when available, otherwise the local zone name."""
    now = datetime.now().astimezone()
    zone = os.environ.get("TZ", "").removeprefix(":")
    if "TZ" not in os.environ:
        with suppress(OSError, ValueError):
            zone = str(Path("/etc/localtime").resolve().relative_to("/usr/share/zoneinfo"))
    if zone.startswith("/usr/share/zoneinfo/"):
        zone = zone.removeprefix("/usr/share/zoneinfo/")
    if zone:
        try:
            now = now.astimezone(ZoneInfo(zone))
        except ZoneInfoNotFoundError, ValueError:
            zone = ""
    return {
        "datetime": now.isoformat(timespec="seconds"),
        "timezone": zone or now.tzname(),
        "weekday": now.strftime("%A"),
        "clock_format": get_clock_format(),
    }


FILE_TOOLS = [
    _tool(
        "read_file",
        "Read source lines in the current chat's documents folder. Paths are "
        "relative to documents/: use py65/memory.py, not documents/py65/memory.py. "
        "Returns numbered lines and truncation/continuation information. No writes.",
        {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "minimum": 1},
            "line_count": {"type": "integer", "minimum": 1, "maximum": 160},
        },
        ["path"],
    ),
    _tool(
        "search_files",
        "Find files or literal text within the current chat's documents folder. "
        "pattern is an optional relative file glob: **/*.py searches all subfolders, "
        "folder/** lists files recursively under folder; folder/*.py searches only that level. "
        "An empty query lists paths. "
        "Otherwise searches filenames and text, returning numbered matches. No shell or regex. "
        "Read a known path directly instead of searching again.",
        {
            "query": {"type": "string"},
            "pattern": {"type": "string"},
            "case_sensitive": {"type": "boolean"},
        },
        [],
    ),
]


class DocumentFiles:
    """Bounded, read-only access beneath one request's documents folder (Linux)."""

    def __init__(self, root):
        self.root = Path(root).absolute()

    def _text(self, name):
        path = PurePosixPath(name)
        if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("Use a relative path inside documents/ without '..'.")
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in path.parts[:-1]:
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            source = os.open(path.parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
            with os.fdopen(source, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("Only regular text files are supported.")
                data = stream.read(2_097_153)
            if len(data) > 2_097_152:
                raise ValueError("File exceeds the 2 MiB limit.")
            result = read_document_stream(io.BytesIO(data))
            if result is None:
                raise ValueError("Unsupported/binary document.")
            return result
        finally:
            os.close(fd)

    def read_file(self, path, start_line=1, line_count=100):
        if (
            type(start_line) is not int
            or type(line_count) is not int
            or start_line < 1
            or line_count < 1
        ):
            raise ValueError("Line numbers/counts must be positive integers.")
        lines = self._text(path).splitlines()
        if start_line > len(lines) + 1:
            raise ValueError(f"File has {len(lines)} lines.")
        selected, chars = [], 0
        for number in range(start_line, min(len(lines) + 1, start_line + min(line_count, 160))):
            line = f"{number}: {lines[number - 1]}"
            if chars + len(line) > 10000:
                if not selected:
                    raise ValueError("Source line exceeds the 10,000-character output limit.")
                break
            selected.append(line)
            chars += len(line)
        end = start_line + len(selected) - 1
        return {
            "path": path,
            "start_line": start_line,
            "end_line": end,
            "total_lines": len(lines),
            "text": "\n".join(selected),
            "truncated": end < len(lines),
            "next_line": end + 1 if end < len(lines) else None,
        }

    def search_files(self, query="", pattern="**/*", case_sensitive=False):
        if (
            not isinstance(query, str)
            or not isinstance(pattern, str)
            or len(query) > 128
            or len(pattern) > 256
        ):
            raise ValueError("Use a literal query of at most 128 characters.")
        if PurePosixPath(pattern).is_absolute() or ".." in PurePosixPath(pattern).parts:
            raise ValueError("Search patterns must stay within documents/.")
        if type(case_sensitive) is not bool:
            raise ValueError("case_sensitive must be boolean.")
        matches, scanned, size = [], 0, 0
        deadline = time.monotonic() + 2
        needle = query if case_sensitive else query.casefold()
        entries = (
            os.walk(self.root, followlinks=False)
            if any(char in pattern for char in "*?[")
            else [(self.root, [], [pattern])]
        )
        for directory, folders, files in entries:
            if time.monotonic() > deadline:
                return {"matches": matches[:20], "truncated": True, "files_scanned": scanned}
            folders[:] = sorted(
                folder for folder in folders if not (Path(directory) / folder).is_symlink()
            )
            for filename in sorted(files):
                relative = (Path(directory) / filename).relative_to(self.root).as_posix()
                scanned += 1
                if (
                    scanned > 2000
                    or size > 16_777_216
                    or time.monotonic() > deadline
                    or len(matches) >= 20
                ):
                    return {"matches": matches[:20], "truncated": True, "files_scanned": scanned}
                if not PurePosixPath(relative).full_match(pattern):
                    continue
                try:
                    text = self._text(relative)
                except OSError, ValueError:
                    continue
                size += len(text)
                if not query or needle in (relative if case_sensitive else relative.casefold()):
                    matches.append({"path": relative})
                if query:
                    for number, line in enumerate(text.splitlines(), 1):
                        if needle in (line if case_sensitive else line.casefold()):
                            matches.append({"path": relative, "line": number, "text": line[:240]})
                            if len(matches) >= 20:
                                break
        result = {
            "matches": matches[:20],
            "truncated": len(matches) >= 20,
            "files_scanned": scanned,
        }
        if not matches:
            result["hint"] = (
                "No matches for this query/pattern combination. "
                "Patterns match files, not directories. "
                "Use **/* to discover paths, or folder/** to include a folder's subdirectories. "
                "For text searches, try a shorter literal query or a broader pattern."
            )
        return result


def handle_tool_call(call, files=None):
    """Dispatch only explicitly listed read-only tools without command approval."""
    function = call["function"]
    name = function["name"]
    handlers = {"get_osinfo": get_osinfo, "get_datetime": get_datetime}
    if files is not None:
        handlers.update(read_file=files.read_file, search_files=files.search_files)
    try:
        if name not in handlers:
            raise ValueError(f"Unknown or unavailable tool: {name}.")
        arguments = json.loads(function["arguments"] or "{}")
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object.")
        return json.dumps(handlers[name](**arguments))
    except (OSError, ValueError, TypeError) as exc:
        return json.dumps({"error": str(exc)})


CODING_INSTRUCTION = (
    " When a request would benefit from terminal commands, explain their purpose and put each "
    "complete command in a separate fenced bash block. Keep multiline commands, including "
    "heredocs, together. Lopaca turns these blocks into review buttons and executes commands "
    "only after the user approves them. Prefer non-destructive commands. You can help create "
    "or edit files through approved commands. Never claim a command has run until Lopaca "
    "reports its result. Commands may launch terminal or GUI programs. When providing a command "
    "to create a program, also provide a separate command to run it. Use get_osinfo when platform "
    "details matter, rather than guessing the OS."
)

_RISK_RULES = tuple(
    (level, reason, re.compile(pattern, re.IGNORECASE | re.MULTILINE))
    for level, reason, pattern in (
        (
            "danger",
            "Deletes files or changes disks",
            r"(?<![\w.])(?:rm|rmdir|unlink|shred|wipe|"
            r"srm|dd|mkfs(?:\.\w+)?|mkswap|wipefs|blkdiscard|fdisk|sfdisk|parted|sgdisk|truncate)\b"
            r"[^\n]*",
        ),
        (
            "danger",
            "Discards version-controlled work",
            r"\bgit\s+(?:reset\b[^\n]*--hard|"
            r"clean\b[^\n]*-[\w]*f|checkout\s+--|restore\b)[^\n]*",
        ),
        (
            "danger",
            "Deletes files during search or synchronization",
            r"\b(?:find\b[^\n]*-delete|rsync\b[^\n]*--delete[\w-]*)[^\n]*",
        ),
        (
            "danger",
            "Downloads and executes code",
            r"\b(?:curl|wget)\b[^\n]*\|\s*"
            r"(?:sudo\s+)?(?:sh|bash|python[\d.]*|perl|ruby)\b[^\n]*",
        ),
        (
            "danger",
            "Grants broad or elevated file permissions",
            r"\bchmod\b[^\n]*"
            r"(?:\b777\b|\b[2467][0-7]{3}\b|[ugo]*\+[^\s]*s)[^\n]*",
        ),
        (
            "danger",
            "Possible fork bomb",
            r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*;?\s*\}"
            r"\s*;\s*:",
        ),
        (
            "danger",
            "Writes system or security-sensitive files",
            r"(?:>{1,2}\s*|\bof=)"
            r"['\"]?(?:/(?:etc|boot|dev|proc|sys|usr|root)/|[^\s]*?/\.(?:ssh|gnupg|aws)/)[^\n]*",
        ),
        (
            "danger",
            "Destructive operations in embedded code",
            r"\b(?:shutil\.rmtree|"
            r"os\.(?:remove|unlink|rmdir|removedirs)|fs\.(?:rm|rmdir|unlink)(?:Sync)?|"
            r"Deno\.remove(?:Sync)?|std::filesystem::remove(?:_all)?)\s*\([^\n]*",
        ),
        (
            "warning",
            "Requests elevated permissions",
            r"\b(?:sudo|doas|pkexec|su|setcap|capsh)\b"
            r"|--(?:privileged|cap-add)\b",
        ),
        (
            "warning",
            "Changes permissions, packages, services, or running processes",
            r"\b"
            r"(?:chmod|chown|chgrp|mount|umount|systemctl|service|reboot|shutdown|poweroff|halt|"
            r"kill|killall|pkill|crontab|iptables|nft|ufw|passwd|usermod|useradd|userdel|"
            r"apt(?:-get)?|dpkg|dnf|yum|pacman)\b[^\n]*"
            r"|\b(?:snap|flatpak|pip[\d.]*|npm|cargo)\s+"
            r"(?:install|uninstall|remove|update|upgrade|publish)\b[^\n]*",
        ),
        (
            "warning",
            "Accesses the network",
            r"\b(?:curl|wget)\b[^\n]*",
        ),
        (
            "warning",
            "May overwrite files",
            r"(?<![\w.])(?:cp|mv|rsync|install|tee|tar|unzip)\b[^\n]*",
        ),
        (
            "warning",
            "Writes locations normally requiring elevated permissions",
            r"\b(?:touch|mkdir|cp|mv|rsync|install|tee)\b[^\n]*"
            r"/(?:etc|boot|dev|proc|sys|usr|root)(?:/|\b)[^\n]*",
        ),
    )
)
_HEREDOC_PATTERN = re.compile(
    r"<<-?\s*['\"]?(?P<tag>[A-Za-z_]\w*)['\"]?[^\n]*\n"
    r"(?P<body>.*?)^\t*(?P=tag)\s*$",
    re.MULTILINE | re.DOTALL,
)
_PROTECTED_PATH = re.compile(
    r"^/(?:etc|boot|dev|proc|sys|usr|root)(?:/|$)|/\.(?:ssh|gnupg|aws)(?:/|$)|"
    r"/(?:\.bashrc|\.profile|\.config/autostart)(?:/|$)"
)
_PYTHON_DELETE = {
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "os.removedirs",
    "shutil.rmtree",
    "pathlib.Path.unlink",
    "pathlib.Path.rmdir",
}
_PYTHON_DYNAMIC = {"eval", "exec", "builtins.eval", "builtins.exec", "pickle.load", "pickle.loads"}
_PYTHON_WRITE = {
    "pathlib.Path.write_text",
    "pathlib.Path.write_bytes",
    "pathlib.Path.rename",
    "pathlib.Path.replace",
    "shutil.move",
    "shutil.copy",
    "shutil.copyfile",
    "shutil.copytree",
    "os.rename",
    "os.replace",
    "os.chmod",
    "os.chown",
}


@dataclass(frozen=True)
class CommandRisk:
    start: int
    end: int
    level: str
    reason: str


def _python_name(node, aliases):
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return f"{_python_name(node.value, aliases)}.{node.attr}"
    if isinstance(node, ast.Call):
        return _python_name(node.func, aliases)
    return ""


def _scan_text(text, python=False):
    risks = [
        CommandRisk(match.start(), match.end(), level, reason)
        for level, reason, pattern in _RISK_RULES
        for match in pattern.finditer(text)
    ]
    try:
        tree = ast.parse(text)
    except SyntaxError, ValueError, RecursionError:
        if python:
            risks.append(CommandRisk(0, len(text), "warning", "Python code could not be parsed"))
        return risks
    aliases = {}
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                root = item.name.split(".")[0]
                aliases[item.asname or root] = item.name if item.asname else root
        elif isinstance(node, ast.ImportFrom):
            for item in node.names:
                aliases[item.asname or item.name] = f"{node.module}.{item.name}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = _python_name(node.value, aliases)
            if isinstance(node.value, (ast.Name, ast.Attribute)) or value == "pathlib.Path":
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        aliases[target.id] = value
        if not isinstance(node, ast.Call):
            continue
        name = _python_name(node.func, aliases)
        strings = [
            item.value
            for item in ast.walk(node)
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        ]
        level, reason = "warning", ""
        if name in _PYTHON_DELETE:
            level, reason = "danger", "Deletes files or directories"
        elif name in _PYTHON_DYNAMIC:
            level, reason = "danger", "Executes dynamic or serialized code"
        elif name in _PYTHON_WRITE or (
            name in {"open", "builtins.open", "io.open", "pathlib.Path.open"}
            and any(
                re.search(r"^[rwax][b+t+]*$", value) and any(char in value for char in "wax+")
                for value in strings
            )
        ):
            reason = "Writes or overwrites files"
            if any(_PROTECTED_PATH.search(value) for value in strings):
                level, reason = "danger", "Writes system or security-sensitive files"
        elif name.startswith(("ctypes.", "marshal.")):
            reason = "Loads native or serialized code"
        elif name.startswith(("requests.", "urllib.request.", "socket.", "http.client.")):
            reason = "Accesses the network"
        if reason:
            start, end = offsets[node.lineno - 1], offsets[node.end_lineno]
            risks.append(CommandRisk(start, end, level, reason))
        if name.startswith(("subprocess.", "os.exec")) or name in {
            "os.system",
            "os.popen",
            "asyncio.create_subprocess_exec",
            "asyncio.create_subprocess_shell",
        }:
            command = " ".join(strings)
            risks.extend(
                CommandRisk(offsets[node.lineno - 1], offsets[node.end_lineno], severity, detail)
                for severity, detail, pattern in _RISK_RULES
                if pattern.search(command)
            )
    return risks


def _scan_command(command, cwd, settings):
    limit = max(1, settings.command_scan_chars)
    text = command[:limit]
    risks = _scan_text(text)
    file_write = parse_file_heredoc(text)
    if file_write:
        target = Path(file_write[0])
        target = target if target.is_absolute() else Path(cwd) / target
        header_end = len(command.split("\n", 1)[0])
        if _PROTECTED_PATH.search(str(target.resolve())):
            risks.append(CommandRisk(0, header_end, "danger", "Writes a system or security file"))
        elif target.exists():
            risks.append(CommandRisk(0, header_end, "warning", "Overwrites an existing file"))
    if len(command) > limit:
        risks.append(CommandRisk(0, len(text), "warning", "Command exceeds the inspection limit"))
    shell_text = text
    for match in _HEREDOC_PATTERN.finditer(text):
        body, offset = match.group("body"), match.start("body")
        risks.extend(
            CommandRisk(offset + risk.start, offset + risk.end, risk.level, risk.reason)
            for risk in _scan_text(body)
        )
        masked = "".join("\n" if char == "\n" else " " for char in body)
        shell_text = shell_text[:offset] + masked + shell_text[match.end("body") :]
    try:
        lexer = shlex.shlex(shell_text, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        risks.append(CommandRisk(0, len(text), "warning", "Shell syntax could not be fully parsed"))
        return sorted(set(risks), key=lambda risk: (risk.start, risk.end, risk.reason))
    inspected = set()
    expects_script = False
    cursor = 0
    for index, token in enumerate(tokens):
        anchor = shell_text.find(token, cursor)
        start, end = (anchor, anchor + len(token)) if anchor >= 0 else (0, len(text))
        if anchor >= 0:
            cursor = end
        if index and tokens[index - 1] in {"-c", "-e"}:
            risks.extend(
                CommandRisk(start, end, risk.level, risk.reason) for risk in _scan_text(token)
            )
            expects_script = False
            continue
        if token in {";", "&&", "||", "|", "&"}:
            expects_script = False
            continue
        if re.fullmatch(
            r"(?:python[\d.]*|pypy[\d.]*|ba?sh|dash|zsh|node|perl|ruby|php|lua)", Path(token).name
        ):
            expects_script = True
            continue
        if token.startswith("-"):
            if token == "-m":
                expects_script = False
            continue
        path = Path(os.path.expanduser(token))  # noqa: PTH111 -- Preserve shell path expansion.
        code_file = expects_script or path.suffix.lower() in {
            ".py",
            ".sh",
            ".bash",
            ".js",
            ".ts",
            ".rb",
            ".pl",
            ".php",
            ".lua",
        }
        expects_script = False
        command_start = index == 0 or tokens[index - 1] in {";", "&&", "||", "|", "&"}
        direct_script = command_start and (
            token.startswith(("./", "../", "~/")) or path.is_absolute()
        )
        if not code_file and not direct_script:
            continue
        if index and tokens[index - 1] in {">", ">>", "<", "<<", "<<-"}:
            continue
        path = path if path.is_absolute() else Path(cwd) / path
        if path in inspected:
            continue
        inspected.add(path)
        if len(inspected) > settings.command_scan_files:
            risks.append(
                CommandRisk(start, end, "warning", "Referenced-file inspection limit reached")
            )
            break
        try:
            if not path.is_file():
                raise ValueError("Source is unavailable or is not a regular file")
            with path.open("rb") as source:
                data = source.read(limit + 1)
            if not code_file and data.startswith(b"\x7fELF"):
                continue
            if len(data) > limit or b"\0" in data:
                raise ValueError("Source is too large or is not plain text")
            source_risks = _scan_text(data.decode("utf-8-sig"), python=path.suffix == ".py")
        except (OSError, ValueError) as exc:
            risks.append(
                CommandRisk(start, end, "warning", f"Could not inspect {path.name}: {exc}")
            )
            continue
        risks.extend(
            CommandRisk(start, end, risk.level, f"{path.name}: {risk.reason}")
            for risk in source_risks
        )
    return sorted(set(risks), key=lambda risk: (risk.start, risk.end, risk.reason))


def scan_command(command, cwd, settings):
    """Heuristic review only; errors and uninspectable code require extra confirmation."""
    try:
        return _scan_command(command, cwd, settings)
    except (OSError, ValueError, RecursionError) as exc:
        return [CommandRisk(0, len(command), "warning", f"Risk scan could not complete: {exc}")]


def extract_commands(text):
    # A fenced block is one approval unit, including heredocs and shell sequences.
    return [
        command
        for match in _COMMAND_BLOCK_PATTERN.finditer(text)
        if (command := match.group(1).strip())
    ]


def parse_file_heredoc(command):
    match = re.fullmatch(
        r"\s*cat\s+>\s*(?P<path>[^\s]+)\s+<<\s*"
        r"['\"]?(?P<tag>[A-Za-z_][A-Za-z0-9_]*)['\"]?\s*\n"
        r"(?P<body>.*?)\n(?P=tag)\s*",
        command,
        re.DOTALL,
    )
    if not match:
        return None
    # Preserve the path's spelling in the preview and leave unknown ~users unresolved.
    return (
        os.path.expanduser(match.group("path")),  # noqa: PTH111
        match.group("body") + "\n",
    )


def command_needs_terminal(args, cwd=None):
    if not args or Path(args[0]).name.lower() not in {"python", "python3", "pypy", "pypy3"}:
        return False
    for arg in args[1:]:
        if arg.startswith("-"):
            continue
        try:
            # Match command preparation: an unresolved ~user may be a literal directory.
            script_path = Path(os.path.expanduser(arg))  # noqa: PTH111
            if not script_path.is_absolute() and cwd is not None:
                script_path = Path(cwd) / script_path
            text = script_path.read_text(encoding="utf-8", errors="replace")
            return bool(
                re.search(r"\binput\s*\(|\bgetpass\s*\(|\bcurses\b|\btty\.setraw\s*\(", text)
            )
        except OSError:
            return False
    return False


def prepare_command(command):
    """Prepare an exact command for execution only after explicit approval."""
    if not command.strip() or "\x00" in command:
        return None, "The command is empty or contains an invalid null character."
    if any(token in command for token in ("\n", "\r", "|", ";", "&", ">", "<", "`", "$")):
        return ["/bin/sh", "-c", command], None
    try:
        args = shlex.split(command, posix=True)
    except ValueError as exc:
        return None, f"Invalid command syntax: {exc}"
    if not args:
        return None, "The command contains no executable."
    # These are command arguments, not necessarily paths: keep "", "./", and repeated slashes.
    return [os.path.expanduser(arg) for arg in args], None  # noqa: PTH111


def run_command(args, cwd):
    """Run a reviewed command and return ``(output, exit_code, note)``."""
    if command_needs_terminal(args, cwd):
        terminal, option = shutil.which("x-terminal-emulator"), "-e"
        if not terminal:
            terminal, option = shutil.which("gnome-terminal"), "--"
        if not terminal:
            raise RuntimeError(
                "Interactive program needs a terminal, but no terminal emulator was found."
            )
        subprocess.Popen(
            [terminal, option, *args],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return "", 0, "Interactive program opened in a terminal window."
    result = subprocess.run(
        args,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        shell=False,
    )
    return (result.stdout + result.stderr).strip(), result.returncode, None


def write_file(path, contents, cwd):
    target = Path(path)
    if not target.is_absolute():
        target = Path(cwd) / target
    target.write_text(contents, encoding="utf-8")
    return target
