#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$HOME/.local/share/lopaca"
APPLICATIONS_DIR="$HOME/.local/share/applications"

mkdir -p -- "$INSTALL_DIR" "$APPLICATIONS_DIR"
if [[ ! "$SCRIPT_DIR" -ef "$INSTALL_DIR" ]]; then
    cp -- "$SCRIPT_DIR"/*.py "$INSTALL_DIR/"
fi

# Escape special characters for a quoted desktop-entry Exec argument.
LAUNCH_PATH=$(printf '%s' "$INSTALL_DIR/main.py" | sed \
    -e 's/\\/\\\\\\\\/g' -e 's/["$`]/\\\\&/g' -e 's/%/%%/g')

cat > "$APPLICATIONS_DIR/Lopaca.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Lopaca
Comment=Chat with a local language model
Exec=python3 "$LAUNCH_PATH"
Icon=network-server-symbolic
Terminal=false
Categories=Utility;
EOF

printf -v MESSAGE 'Installed Lopaca to %s.\n\nLaunch it from your Applications menu.' "$INSTALL_DIR"
# Detach GUI notifications so the launching terminal can close without closing the dialog.
if [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]] && command -v zenity >/dev/null; then
    setsid --fork zenity --info --no-markup --title='Lopaca Installed' \
        --text="$MESSAGE" </dev/null >/dev/null 2>&1
elif [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]] && command -v kdialog >/dev/null; then
    setsid --fork kdialog --title 'Lopaca Installed' \
        --msgbox "$MESSAGE" </dev/null >/dev/null 2>&1
else
    printf 'Installed Lopaca to %s. Launch it from your Applications menu.\n' "$INSTALL_DIR"
    if [[ -t 0 ]]; then
        read -r -p 'Press Enter to close…' || true
    fi
fi
