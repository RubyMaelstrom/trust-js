# ky HTTP client on trust-js

[ky](https://github.com/sindresorhus/ky) is a tiny HTTP client built on `fetch`; trust-js ships
`fetch` (http only), so it runs unmodified. The demo spins up a local `node:http` server and calls
it with ky — GET, a JSON POST, and a `beforeRequest` hook.

```sh
npm install
../../target/release/trust-js-cli main.mjs
```
