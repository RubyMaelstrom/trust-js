# Offline browser replay gates

These fixtures exercise TRust's production TRust JS page pipeline without a terminal, window, or
network. Every fixture must finish with matching nonempty `data-replay-checksum` and
`data-replay-expected` attributes and without a JavaScript error or panic.

`event-loop.html` is the fast inner-loop gate. `dom-reconcile.html` stresses large DOM replacement,
selectors, mutation delivery, and layout. `speedometer-vue-todomvc.html` runs the active
Speedometer 3.1 Vue TodoMVC workload: Vue 3.2.47 mounts the application, then the replay performs
the official 100-add, 100-complete, and 100-delete interaction shape.

Run them from the TRust JS root with TRust's `trust-browser-replay` example, which prints a JSON
report and exits nonzero if any fixture fails:

```sh
cargo build --release --manifest-path ../TRust/Cargo.toml --example trust-browser-replay
V=browser-replay-fixtures/speedometer-3.1/resources/todomvc/architecture-examples/vue/dist
../TRust/target/release/examples/trust-browser-replay \
  --external js/chunk-vendors.b4ac9361.js=$V/js/chunk-vendors.b4ac9361.js \
  --external js/app.ad36df07.js=$V/js/app.ad36df07.js \
  --sheet css/app.319576e1.css=$V/css/app.319576e1.css \
  benchmarks/browser-replays/*.html
```

The third-party JavaScript and CSS are not checked into TRust JS. `../browser-replay-assets.json`
pins their upstream repository, revision, paths, and SHA-256 digests; place them under the
git-ignored `browser-replay-fixtures/` directory at those paths.
