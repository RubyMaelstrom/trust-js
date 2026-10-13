#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${WASM_SPEC:-$ROOT/wasm-spec}"
REPO="https://github.com/WebAssembly/spec.git"
# TRust JS's advertised core subset is the WebAssembly 2.0 scalar and control-flow baseline plus
# multi-value, funcref tables, sign-extension, non-trapping conversions, and bulk-memory
# instructions. It does not yet advertise SIMD, externref tables, or the GC/exception/thread
# features folded into 3.0. Pin the matching official release suite and audit newer supported
# proposals separately rather than misreporting unsupported features as regressions.
REV="05ca4182176763112561ae20153975c12bd689e4"

if [[ ! -d "$DEST/.git" ]]; then
  git clone --depth 1 --filter=blob:none --sparse "$REPO" "$DEST"
fi

# The core suite evolves with the standard. Keep audit scores reproducible and update this pin
# deliberately after reviewing the corresponding Core Specification changes.
if ! git -C "$DEST" cat-file -e "$REV^{commit}" 2>/dev/null; then
  git -C "$DEST" fetch --depth 1 origin "$REV"
fi
git -C "$DEST" checkout --detach "$REV"
git -C "$DEST" sparse-checkout set test/core

echo "WebAssembly spec tests ready at $DEST ($(git -C "$DEST" rev-parse HEAD))"
