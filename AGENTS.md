# Working on TRust JS

This file collects the implementation and testing guidance that used to live in
README.md. Keep README.md a short introduction for people discovering the project.

## Project and workspace

TRust JS is a JavaScript engine written in Rust and the JavaScript backend used by
TRust. The workspace also provides a standalone runtime. Keep the engine independent
of a particular host; use the `embed` feature's curated API for host integration.

| Crate | Responsibility |
| --- | --- |
| `trust-js` | Lexer, parser, interpreter, bytecode VM, native JIT, builtins, embedding API |
| `trust-js-host` | Host state, resource tables, extensions, worker and callback primitives |
| `trust-js-timers` | Timers, immediates, microtask integration |
| `trust-js-fs` | Synchronous and asynchronous filesystem operations |
| `trust-js-web` | Web APIs and their JS glue, HTTP, Streams, WebSocket, WebAssembly support |
| `trust-js-node` | Node compatibility, CommonJS resolution, native addons, Bun compatibility APIs |
| `trust-js-tls` | TLS transport using dynamically loaded system OpenSSL |
| `trust-js-runtime` | Event loop, module loading, console/process, assembly of host extensions |
| `trust-js-repl` | Interactive shell with a persistent realm |
| `trust-js-cli` | Runtime command-line entrypoint |
| `trust-js-typescript` | TypeScript parsing and checking |
| `trust-js-wasm` | WebAssembly build of the JavaScript engine |
| `test262-runner` | Test262 conformance harness |
| `trust-js-difftest` | Differential testing across execution tiers |
| `wasm-spec-runner` | Official WebAssembly test harness |

The main layering is engine → host substrate → op crates → runtime → REPL → CLI.
Keep this dependency direction acyclic. Consult the manifests for the complete graph.

The workspace is no longer zero-dependency. The native engine uses `stacker`;
web support uses URL/IDNA and encoding crates; Wasm bindings and test tooling have
their own dependencies. Check Cargo.toml and Cargo.lock before making dependency
claims. System libraries and the vendored Streams polyfill are separate from Cargo
dependencies.

Linux builds link with packed relative relocations (RELR; see `.cargo/config.toml`), so
their binaries need glibc 2.36 or later where they run, and a linker that supports
`-z pack-relative-relocs` (GNU ld 2.38 or later, LLVM lld 15 or later, or mold).

## Performance work

Optimize the established bytecode/native JIT pipeline and shared runtime as one engine.
The optional Cranelift JavaScript backend has been removed; its dated investigation
records are historical evidence, not an active implementation roadmap.

Use release builds and record the machine, revision, tier, flags, and workload.
Follow [benchmarks/README.md](benchmarks/README.md) for comparison and regression
policy. Check correctness as well as throughput, memory, startup, and compilation
costs. Investigation tools and reports may exist locally without being tracked;
do not assume a fresh clone includes them.

The tracked benchmark runners (Octane, Web Tooling, ARES-6) and their
invocation are listed in [DIAGNOSTICS.md](DIAGNOSTICS.md).

## Documentation

Keep useful public setup instructions in READMEs,
diagnostic commands and environment variables in `DIAGNOSTICS.md`, and
third-party provenance beside the vendored source. General Markdown workfiles
and the retired changelog are ignored and kept locally. Do not make builds or
required contributor instructions depend on those untracked documents.