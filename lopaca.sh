#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

show_launch_error() {
    local message="$1"
    printf '%s\n' "$message" >&2
    if [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
        if command -v zenity >/dev/null; then
            zenity --error --no-markup --title='Lopaca could not start' --text="$message" || true
        elif command -v kdialog >/dev/null; then
            kdialog --title 'Lopaca could not start' --error "$message" || true
        elif command -v notify-send >/dev/null; then
            notify-send 'Lopaca could not start' "$message" || true
        fi
    fi
}

if ! command -v python3 >/dev/null; then
    show_launch_error "Lopaca requires Python 3.13 or newer. The python3 command was not found."
    exit 1
fi

# Check from a script that older Python versions can parse before loading Lopaca.
if ! error=$(python3 - 2>&1 <<'PY'
import sys

if sys.version_info < (3, 13):
    sys.exit("Lopaca requires Python 3.13 or newer. Found Python " + sys.version.split()[0] + ".")

try:
    import gi

    gi.require_version("Gtk", "4.0")
    gi.require_version("Adw", "1")
    gi.require_version("Graphene", "1.0")
    gi.require_version("Gsk", "4.0")
    from gi.repository import Adw, Gtk
except (ImportError, OSError, ValueError) as exc:
    sys.exit("Lopaca requires PyGObject and the GTK4 and libadwaita bindings.\n\n" + str(exc))

if Gtk.check_version(4, 10, 0) or (Adw.get_major_version(), Adw.get_minor_version()) < (1, 5):
    sys.exit("Lopaca requires GTK 4.10 or newer and libadwaita 1.5 or newer.")
PY
); then
    show_launch_error "$error"
    exit 1
fi

main="$SCRIPT_DIR/main.py"
if [[ ! -f "$main" ]]; then
    main="$SCRIPT_DIR/src/main.py"
fi
exec python3 "$main" "$@"
