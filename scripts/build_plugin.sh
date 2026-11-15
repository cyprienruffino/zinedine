#!/usr/bin/env bash
# Build the SoccerRL BepInEx plugin and copy it into SoccerPhysics/BepInEx/plugins/SoccerRL/.
# Usage: scripts/build_plugin.sh [path/to/SoccerPhysics]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GAME_DIR="$(cd "${1:-$ROOT/SoccerPhysics}" && pwd)"
PROJ="$ROOT/plugin/SoccerRL/SoccerRL.csproj"

if command -v dotnet >/dev/null 2>&1; then
  dotnet build "$PROJ" -c Release -nologo -p:GameDir="$GAME_DIR"
elif command -v nix >/dev/null 2>&1; then
  nix shell nixpkgs#dotnet-sdk_8 -c dotnet build "$PROJ" -c Release -nologo -p:GameDir="$GAME_DIR"
else
  echo "Need the .NET SDK (6+) or nix to build the plugin." >&2
  exit 1
fi
echo "Installed: $GAME_DIR/BepInEx/plugins/SoccerRL/SoccerRL.dll"
