#!/usr/bin/env bash
#
# Run trust-js's benchmarks. Two suites, both on the std-only harness (no third-party bench crate):
#
#   engine     — running representative JavaScript through Engine::eval, plus startup.
#   internals  — the compilation pipeline stage by stage (lex → parse → encode → decode),
#                via the `bench` feature's trust_js::bench_api.
#   runtime    — the assembled runtime's web surface (channel messaging, broadcast fan-out,
#                error reporting) through Runtime::eval with the loop pumped.
#
# Usage:
#   scripts/run-benches.sh            # both suites
#   scripts/run-benches.sh engine     # just the engine suite
#   scripts/run-benches.sh runtime    # just the runtime suite
#   scripts/run-benches.sh internals  # just the internals suite
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

suite="${1:-all}"

case "$suite" in
  engine)    cargo bench -p trust-js --bench engine ;;
  internals) cargo bench -p trust-js --features bench --bench internals ;;
  runtime)   cargo bench -p trust-js-runtime --bench runtime ;;
  all)
    cargo bench -p trust-js --bench engine
    cargo bench -p trust-js --features bench --bench internals
    cargo bench -p trust-js-runtime --bench runtime
    ;;
  *) echo "unknown suite '$suite' (want: engine | internals | runtime | all)" >&2; exit 1 ;;
esac
