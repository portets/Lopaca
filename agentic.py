"""Small, approval-gated helpers for the optional local coding assistant."""

import os
import re
import shlex
import shutil
import subprocess
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
