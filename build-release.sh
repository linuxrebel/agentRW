#!/usr/bin/env bash
# Build a release tarball into release/. Not shipped inside the tarball.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

VERSION="${1:-$(git describe --tags --abbrev=0 2>/dev/null || echo v0.0.0)}"
VERSION="${VERSION#v}"
NAME="agentRW-${VERSION}"
OUT="release"
STAGE="$(mktemp -d)"
DEST="$STAGE/$NAME"

mkdir -p "$OUT" "$DEST"

# Everything the app needs to run, and nothing else. No .git, no tests, no
# scratch files, no release/ — a tarball that unpacks into a working install.
cp -a coding_agent.py requirements.txt install.sh uninstall.sh "$DEST"/
# CRLF for the Windows scripts. cmd.exe mostly tolerates LF, but multi-line
# constructs and labels are where it stops tolerating.
for b in install.bat uninstall.bat; do
    sed 's/$/\r/' "$b" > "$DEST/$b"
done
for f in README.md PLUGINS.md FUTURES.md LICENSE; do
    [ -e "$f" ] && cp -a "$f" "$DEST"/
done
if [ -d tools ]; then
    cp -a tools "$DEST"/
    find "$DEST/tools" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
fi
chmod 755 "$DEST/coding_agent.py" "$DEST/install.sh" "$DEST/uninstall.sh"

TARBALL="$OUT/${NAME}.tar.gz"
tar -czf "$TARBALL" -C "$STAGE" "$NAME"
rm -rf "$STAGE"

echo "$TARBALL"
tar -tzf "$TARBALL" | sed 's/^/  /'
echo
if command -v sha256sum >/dev/null; then
    echo "sha256: $(sha256sum "$TARBALL" | cut -d" " -f1)"
else
    echo "sha256: $(shasum -a 256 "$TARBALL" | cut -d" " -f1)"   # macOS
fi
echo
echo "Install (Linux/macOS, system-wide):"
echo "  tar -xzf $TARBALL"
echo "  sudo ./$NAME/install.sh"
echo
echo "Install (Windows, per-user, no admin):"
echo "  tar -xzf $NAME.tar.gz"
echo "  $NAME\\install.bat"
