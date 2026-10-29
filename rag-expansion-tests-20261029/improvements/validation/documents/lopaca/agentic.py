"""Small, approval-gated helpers for the optional local coding assistant."""

import ast
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

_COMMAND_BLOCK_PATTERN = re.compile(r"```(?:bash|sh|shell)\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)

CODING_INSTRUCTION = (
    " You are in a limited coding helper mode. When a terminal command would help, provide it in "
    "a fenced bash block which will provide the user with a button, one command per line, explain "
    "what it does, and wait for the user to click its approval button. Prefer non-destructive "
    "commands. You may propose terminal commands even though you cannot access the filesystem "
    "yourself: the desktop app will show each command to the user and run it only after approval. "
    "The user is using a GUI and can use CLI or GUI programs. Do not say you cannot create or save "
    "files as a reason to avoid proposing a command. Do not substitute operating-system "
    "instructions or a prose-only description. Never claim a command has run until the application "
    "reports its output. If you provide a command for making a program file you should also "
    "provide a command for running it, also in a fenced bash block which will give the user "
    "another button."
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


def command_needs_terminal(args):
    if not args or Path(args[0]).name.lower() not in {"python", "python3", "pypy", "pypy3"}:
        return False
    for arg in args[1:]:
        if arg.startswith("-"):
            continue
        try:
            # Match command preparation: an unresolved ~user may be a literal directory.
            script_path = Path(os.path.expanduser(arg))  # noqa: PTH111
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
    if command_needs_terminal(args):
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
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=120, shell=False)
    return (result.stdout + result.stderr).strip(), result.returncode, None


def write_file(path, contents, cwd):
    target = Path(path)
    if not target.is_absolute():
        target = Path(cwd) / target
    target.write_text(contents, encoding="utf-8")
    return target
