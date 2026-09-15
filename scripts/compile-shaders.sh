#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v glslangValidator >/dev/null 2>&1; then
  echo "error: glslangValidator not found" >&2
  echo "install glslang-tools or an equivalent package, then rerun this script" >&2
  exit 1
fi

glslangValidator -V "$repo_root/shaders/no_op.comp.glsl" -o "$repo_root/shaders/no_op.spv"
glslangValidator -V "$repo_root/shaders/embedding_lookup.comp.glsl" -o "$repo_root/shaders/embedding_lookup.spv"

echo "compiled shaders/no_op.spv"
echo "compiled shaders/embedding_lookup.spv"
