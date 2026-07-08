#!/bin/bash
set -euo pipefail

SCHEMA="schema/graph.fbs"
GEN_PY_ROOT="py/gen"
GEN_CXX_ROOT="csrc/gen"

mkdir -p "$GEN_PY_ROOT"
mkdir -p "$GEN_CXX_ROOT"

flatc --version
flatc --cpp -o "$GEN_CXX_ROOT" "$SCHEMA"
flatc --python -o "$GEN_PY_ROOT" "$SCHEMA"

echo "✅ Done generating FlatBuffers code!"