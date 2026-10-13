# Djot parser on trust-js

Parses [Djot](https://djot.net) markup to an AST and renders it to HTML with `@djot/djot`, a
pure-JS parser — no native code. `@djot/djot` is CommonJS, so it's default-imported and
destructured (trust-js exposes a CJS module's `module.exports` as the ESM default).

```sh
npm install
../../target/release/trust-js-cli parse.mjs
```
