#!/usr/bin/env bash
# Installs BepInEx (the mod loader) into YOUR OWN copy of Soccer Physics (Linux build), downloading the
# official release from https://github.com/BepInEx/BepInEx (LGPL-2.1). Nothing from the game is shipped
# with this repository: you need your own copy of the game.
#
# Usage: scripts/install_bepinex.sh [path/to/SoccerPhysics]   (default: ./SoccerPhysics)
# Then build and install the SoccerRL plugin: scripts/build_plugin.sh [path/to/SoccerPhysics]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GAME_DIR="$(cd "${1:-$ROOT/SoccerPhysics}" && pwd)"
VERSION=5.4.23.6
URL="https://github.com/BepInEx/BepInEx/releases/download/v$VERSION/BepInEx_linux_x64_$VERSION.zip"

[[ -x "$GAME_DIR/SoccerPhysics.x86_64" ]] || {
  echo "No SoccerPhysics.x86_64 in $GAME_DIR: point this script at your Linux copy of the game." >&2
  exit 1
}
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
echo "Downloading BepInEx $VERSION..."
curl -fsSL -o "$tmp/bepinex.zip" "$URL"
unzip -oq "$tmp/bepinex.zip" -d "$GAME_DIR"
# The launcher script must know which binary to start.
sed -i 's/^executable_name=.*/executable_name="SoccerPhysics.x86_64"/' "$GAME_DIR/run_bepinex.sh"
chmod +x "$GAME_DIR/run_bepinex.sh"
mkdir -p "$GAME_DIR/BepInEx/plugins"
echo "BepInEx $VERSION installed in $GAME_DIR. Next: scripts/build_plugin.sh \"$GAME_DIR\""
