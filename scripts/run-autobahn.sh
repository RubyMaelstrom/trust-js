#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WSTEST="${WSTEST:-wstest}"
SPEC="$ROOT/scripts/autobahn-fuzzingserver.json"
REPORTS="$ROOT/autobahn-reports/clients"

if ! command -v "$WSTEST" >/dev/null 2>&1; then
  echo "Autobahn wstest not found; install Autobahn Testsuite 25.10.1 or set WSTEST=/path/to/wstest" >&2
  exit 1
fi

mkdir -p "$REPORTS"
cd "$ROOT"
"$WSTEST" -m fuzzingserver -s "$SPEC" -u 0 >"$ROOT/autobahn-reports/server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ready=false
for _ in {1..100}; do
  if (exec 3<>/dev/tcp/127.0.0.1/9001) 2>/dev/null; then
    exec 3>&- 3<&-
    ready=true
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  sleep 0.1
done
if [[ "$ready" != true ]]; then
  echo "Autobahn server did not become ready; see autobahn-reports/server.log" >&2
  exit 1
fi

OUTPUT="$(cargo run -q -p trust-js-cli -- scripts/autobahn-testee.js)"
echo "$OUTPUT"
if [[ "$OUTPUT" != *"AUTOBAHN COMPLETE 247"* ]]; then
  echo "Autobahn client did not complete the expected 247 applicable cases" >&2
  exit 1
fi

if rg -q '"behavior(Close)?": "(FAILED|NON-STRICT)"' "$REPORTS"/trust_js_case_*.json; then
  echo "Autobahn reported a failed or non-strict case" >&2
  rg -l '"behavior(Close)?": "(FAILED|NON-STRICT)"' "$REPORTS"/trust_js_case_*.json >&2
  exit 1
fi

rg -o '"behavior": "[^"]+"' "$REPORTS"/trust_js_case_*.json \
  | sed 's/.*"behavior": "//;s/"$//' \
  | sort \
  | uniq -c
