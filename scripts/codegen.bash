#!/bin/bash
set -euo pipefail

SCHEMA="schema/fxfusion.fbs"
GEN_PY_ROOT="python/gen"
GEN_CXX_ROOT="src/gen"

mkdir -p "$GEN_PY_ROOT"
mkdir -p "$GEN_CXX_ROOT"

flatc --cpp -o "$GEN_CXX_ROOT" "$SCHEMA"
flatc --python -o "$GEN_PY_ROOT" "$SCHEMA"

echo "✅ Done generating FlatBuffers code!"