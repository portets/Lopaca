#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${HOME:?}/.local/share/lopaca"
LAUNCHER="$HOME/.local/share/applications/app.lopaca.Lopaca.desktop"
GUI=""
if [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
    if command -v zenity >/dev/null; then
        GUI=zenity
    elif command -v kdialog >/dev/null; then
        GUI=kdialog
    fi
fi

# Let the launching terminal close while this process waits for the GUI answer.
if [[ -n "$GUI" && "${1:-}" != --detached ]]; then
    setsid --fork bash "${BASH_SOURCE[0]}" --detached </dev/null >/dev/null 2>&1
    exit 0
fi

QUESTION="Would you like to keep Lopaca chat data and settings?"
KEEP_DATA=true
if [[ "$GUI" == zenity ]]; then
    # Explicit choices preserve No/Yes/Cancel order; closing the dialog also cancels.
    # Zenity extra buttons can print a choice while returning a nonzero status.
    ANSWER=$(zenity --question --no-markup --icon=dialog-question-symbolic \
        --width=440 --no-wrap --title='Uninstall Lopaca' --text="$QUESTION" \
        --switch --extra-button=No --extra-button=Yes --extra-button=Cancel) || true
    case "$ANSWER" in
        Yes) KEEP_DATA=true ;;
        No) KEEP_DATA=false ;;
        *) exit 0 ;;
    esac
elif [[ "$GUI" == kdialog ]]; then
    if kdialog --title 'Uninstall Lopaca' --yesnocancel "$QUESTION"; then
        KEEP_DATA=true
    else
        STATUS=$?
        [[ "$STATUS" == 1 ]] || exit 0
        KEEP_DATA=false
    fi
else
    printf '%s\n\n' "$QUESTION"
    while true; do
        if ! read -r -p 'Keep chat data and settings? [Y/n]: ' ANSWER; then
            exit 0
        fi
        case "$ANSWER" in
            ''|[Yy]|[Yy][Ee][Ss]) break ;;
            [Nn]|[Nn][Oo]) KEEP_DATA=false; break ;;
            *) printf 'Please answer yes or no.\n' ;;
        esac
    done
fi

rm -f -- "$LAUNCHER"
if "$KEEP_DATA"; then
    rm -rf -- "$DATA_DIR/app"
    MESSAGE="Lopaca uninstalled. Your chat data and settings were kept in $DATA_DIR."
else
    rm -rf -- "$DATA_DIR"
    MESSAGE='Lopaca uninstalled. Chat data and settings were deleted.'
fi

if [[ "$GUI" == zenity ]]; then
    zenity --info --no-markup --icon=dialog-information-symbolic --width=440 --no-wrap \
        --title='Lopaca Uninstalled' --text="$MESSAGE"
elif [[ "$GUI" == kdialog ]]; then
    kdialog --title 'Lopaca Uninstalled' --msgbox "$MESSAGE"
else
    printf '%s\n' "$MESSAGE"
    if [[ -t 0 ]]; then
        read -r -p 'Press Enter to close…' || true
    fi
fi
