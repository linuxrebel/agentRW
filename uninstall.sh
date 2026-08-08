#!/usr/bin/env bash
# Remove agentRW from /opt/agentRW and /usr/local/bin/cagent.
set -euo pipefail

PREFIX=/opt/agentRW
LINK=/usr/local/bin/cagent

if [ "$(id -u)" -ne 0 ]; then
    echo "This removes $PREFIX and $LINK, which need root." >&2
    echo "Re-run as:  sudo $0" >&2
    exit 1
fi

if [ ! -d "$PREFIX" ] && [ ! -e "$LINK" ]; then
    echo "agentRW is not installed at $PREFIX."
    exit 0
fi

# Say what goes before it goes. Installed plugins live under tools/ and are
# deleted with everything else — they are not backed up anywhere.
if [ -d "$PREFIX/tools" ]; then
    PLUGINS="$(find "$PREFIX/tools" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sed "s|$PREFIX/tools/||" || true)"
    if [ -n "$PLUGINS" ]; then
        echo "These installed plugins will be deleted:"
        echo "$PLUGINS" | sed 's/^/    /'
        echo
    fi
fi

read -r -p "Remove $PREFIX and $LINK? [y/N] " ans
case "$ans" in
    [yY]|[yY][eE][sS]) ;;
    *) echo "Nothing removed."; exit 0 ;;
esac

# -e not -L: the launcher is a small shell script now, not a symlink, because
# shebang resolution is not dependable on macOS.
if [ -e "$LINK" ]; then
    rm -f "$LINK"
    echo "Removed $LINK"
fi
if [ -d "$PREFIX" ]; then
    rm -rf "$PREFIX"
    echo "Removed $PREFIX"
fi

echo
echo "Left alone, because it is yours and not ours to delete:"
echo "    ~/.config/coding_agent/   saved model and flags"
echo "    any DEBT.md or *.bak files in your projects"
