#!/usr/bin/env bash
# Install agentRW to /opt/agentRW with a symlink at /usr/local/bin/cagent.
set -euo pipefail

PREFIX=/opt/agentRW
LINK=/usr/local/bin/cagent
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$(id -u)" -ne 0 ]; then
    echo "This installs to $PREFIX and $LINK, which need root." >&2
    echo "Re-run as:  sudo $0" >&2
    exit 1
fi

for f in coding_agent.py requirements.txt; do
    [ -f "$HERE/$f" ] || { echo "Missing $f — run this from the unpacked release." >&2; exit 1; }
done

command -v python3 >/dev/null || { echo "python3 not found on PATH." >&2; exit 1; }
python3 - <<'PY' || { echo "agentRW needs Python 3.10 or newer." >&2; exit 1; }
import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)
PY

# Preserve plugins across an upgrade: tools/ is where installed plugins live.
KEEP=""
if [ -d "$PREFIX/tools" ]; then
    KEEP="$(mktemp -d)"
    cp -a "$PREFIX/tools/." "$KEEP/" 2>/dev/null || true
    echo "Keeping existing plugins from $PREFIX/tools"
fi

echo "Installing to $PREFIX"
mkdir -p "$PREFIX"
rm -rf "$PREFIX"/*.py "$PREFIX"/*.md "$PREFIX"/requirements.txt "$PREFIX"/tools
cp -a "$HERE"/coding_agent.py "$HERE"/requirements.txt "$PREFIX"/
for f in README.md PLUGINS.md FUTURES.md LICENSE uninstall.sh; do
    [ -e "$HERE/$f" ] && cp -a "$HERE/$f" "$PREFIX"/
done
[ -d "$HERE/tools" ] && cp -a "$HERE/tools" "$PREFIX"/ || mkdir -p "$PREFIX/tools"

if [ -n "$KEEP" ]; then
    cp -a "$KEEP/." "$PREFIX/tools/" 2>/dev/null || true
    rm -rf "$KEEP"
fi

chmod 755 "$PREFIX/coding_agent.py"
[ -f "$PREFIX/uninstall.sh" ] && chmod 755 "$PREFIX/uninstall.sh"

ln -sfn "$PREFIX/coding_agent.py" "$LINK"
echo "Linked $LINK -> $PREFIX/coding_agent.py"

echo
echo "Installed. Run:  cagent"
echo
echo "Python packages (as your normal user, not root):"
echo "    pip install --user -r $PREFIX/requirements.txt"
echo
echo "pylint and autopep8 are better from your distro:"
echo "    Fedora  sudo dnf install python3-pylint python3-autopep8"
echo "    Debian  sudo apt install pylint python3-autopep8"
echo
echo "Uninstall with:  sudo $PREFIX/uninstall.sh"
