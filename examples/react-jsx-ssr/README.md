# React JSX / TSX SSR on trust-js

Server-side render React components with `react-dom/server`'s `renderToString`.

## `.jsx` — runs natively

trust-js's runtime transpiles JSX to `React.createElement` calls itself (a from-scratch JSX
transformer, no build tool), so a `.jsx` file runs directly:

```sh
npm install
../../target/release/trust-js-cli app.jsx
```

```
<div><h1>hello</h1><h1>world 🌏</h1></div>
```

## `.tsx` — via esbuild

This example transpiles a `.tsx` file (with `interface`, `React.FC<…>` annotations) using
esbuild, which itself runs on TRust JS, before executing:

```sh
../../target/release/trust-js-cli render.mjs app.tsx
```

`app.tsx` (a recursive component) renders a nested tree and reports its SSR time.

## How it works

- **`.jsx` natively:** the runtime lowers `<div a={x}>hi</div>` to
  `React.createElement("div", { a: x }, "hi")` in a single scan (`crates/trust-js-runtime/src/jsx.rs`)
  — classic runtime, so the component's own `import React` supplies the factory. The engine's
  parser is untouched (JSX text with apostrophes would break an up-front lexer), and the transform
  runs as a pre-pass on `.jsx` sources.
- **`.tsx` via esbuild:** `render.mjs` runs esbuild's `tsx` loader (which strips types *and* lowers
  JSX). `ESBUILD_WORKER_THREADS=0` selects esbuild's subprocess path for this example.
- **Module resolution:** the transpiled code keeps `import React` / `import { renderToString }` as
  ESM imports; trust-js resolves them from `node_modules` and follows `react-dom/server`'s CommonJS
  re-export chain (`server.js → server.node.js → cjs/*.js`) to expose `renderToString` as a named
  ESM export.

For a pure-`React.createElement`, no-transpile version served over `node:http`, see
`examples/react-ssr`.
