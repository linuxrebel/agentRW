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

# Ollama is what agentRW talks to. Nothing works without it, so stop here
# rather than install something that cannot run. Their installer, not ours.
if ! command -v ollama >/dev/null; then
    echo
    echo "  Ollama was not found on this machine."
    echo
    echo "  agentRW talks to a model through Ollama, so it needs to be installed"
    echo "  first. Get it from:"
    echo
    echo "      https://ollama.com"
    echo
    echo "  Install it their way, then run this again."
    echo
    read -n 1 -s -r -p "  Press any key to exit."
    echo
    exit 1
fi

command -v python3 >/dev/null || { echo "python3 not found on PATH." >&2; exit 1; }
python3 - <<'PY' || { echo "agentRW needs Python 3.9 or newer." >&2; exit 1; }
import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)
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
for f in README.md PLUGINS.md FUTURES.md How-to-create-plugins.docx LICENSE uninstall.sh; do
    [ -e "$HERE/$f" ] && cp -a "$HERE/$f" "$PREFIX"/
done
[ -d "$HERE/tools" ] && cp -a "$HERE/tools" "$PREFIX"/ || mkdir -p "$PREFIX/tools"

if [ -n "$KEEP" ]; then
    cp -a "$KEEP/." "$PREFIX/tools/" 2>/dev/null || true
    rm -rf "$KEEP"
fi

chmod 755 "$PREFIX/coding_agent.py"
[ -f "$PREFIX/uninstall.sh" ] && chmod 755 "$PREFIX/uninstall.sh"

# cagent is a symlink into the installed program. The .py's shebang
# (#!/usr/bin/env python3) selects the interpreter at run time.
PYBIN="$(python3 -c 'import sys; print(sys.executable)')"
# -f replaces any existing file or launcher; -n so an existing link is replaced,
# not followed into. An upgrade over the old launcher-based install thus becomes
# a clean symlink.
ln -sfn "$PREFIX/coding_agent.py" "$LINK"
echo "Symlink $LINK -> $PREFIX/coding_agent.py"

# Check as the invoking user, not root: a --user install lives in their home,
# so importing as root would report a missing package that is actually there.
RUNAS="${SUDO_USER:-$(id -un)}"
if ! sudo -u "$RUNAS" "$PYBIN" -c "import openai" >/dev/null 2>&1; then
    echo
    echo "  ============================================================"
    echo "   Installed, but cagent will NOT start yet."
    echo
    echo "   The 'openai' package is missing. Run as yourself, not root:"
    echo
    echo "       $PYBIN -m pip install --user -r $PREFIX/requirements.txt"
    echo "  ============================================================"
    echo
    exit 0
fi

echo
echo "  ============================================================"
echo "   Install successful.  Run 'cagent' to start the harness."
echo "  ============================================================"
echo
echo "Python packages (as your normal user, not root):"
echo "    pip install --user -r $PREFIX/requirements.txt"
echo
echo "That is the whole list. No plugins ship with the agent — each lives in its"
echo "own repo, and says what it needs itself in /plugins."
echo
echo "Uninstall with:  sudo $PREFIX/uninstall.sh"
