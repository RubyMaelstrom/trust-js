# Hono on trust-js

A small [Hono](https://hono.dev) app run on the trust-js runtime — over a real HTTP server
(`TRustJS.serve`) or in-process through `app.fetch`.

Hono's core turns a `Request` into a `Response` via `app.fetch(request)`, the same
`(Request) -> Response` contract that `TRustJS.serve` (and Deno/Bun/Workers) drive. trust-js ships the
web platform (`Request`/`Response`/`Headers`/`URL`) and now a WinterCG-style HTTP server on top.

## Run

```sh
npm install
cargo build --release -p trust-js-cli        # from the repo root
TRUST_JS=../../target/release/trust-js-cli

$TRUST_JS serve-http.js        # real HTTP server on http://localhost:3000
$TRUST_JS ant-style.js         # the hono/logger example, faithfully
$TRUST_JS serve.js             # no socket: dispatch a few requests through app.fetch
```

Then, in another terminal:

```sh
curl localhost:3000/
curl localhost:3000/api/hello?name=trust-js
curl -X POST localhost:3000/echo -H 'content-type: application/json' -d '{"hi":"there"}'
```

## Files

- `app.js` — the Hono app (routes, middleware, JSON, path params, custom 404).
- `serve-http.js` — serves `app.js` over HTTP with `TRustJS.serve(app.fetch, { port })`.
- `ant-style.js` — the `import { logger } from 'hono/logger'` + `export default app` example,
  adapted to `TRustJS.serve` and `TRustJS.version`.
- `serve.js` — drives `app.fetch` in-process (no listening socket); handy for testing routing.
- `bench.mjs` — in-process throughput bench (`app.fetch` in a loop; runtime-agnostic).

## `TRustJS.serve`

```js
// (request) => Response | Promise<Response>
const server = TRustJS.serve(handler, { hostname: '127.0.0.1', port: 3000 });

// Also accepted: TRustJS.serve(options, handler), TRustJS.serve({ fetch, port }),
// and TRustJS.serve(app) for any object with a .fetch() method (e.g. a Hono app).
server.shutdown();     // stop accepting; resolves `server.finished`
server.port;           // the bound port (useful with port: 0)
```

This isn't a WinterTC API — the Minimum Common API standardizes fetch/Request/Response/URL but
not a server — so it follows the cross-runtime `serve(handler)` convention. **v1 limitations**
(see `crates/trust-js-web/src/server.rs`): one connection accepted at a time, `Connection: close`
(no keep-alive), buffered request/response bodies (no streaming), http only (no TLS), no HTTP/2.
