# TRust JS

TRust JS is a JavaScript engine written in Rust, with a runtime growing
around it. It powers the JavaScript in TRust, and you can also take it out for a
spin on its own: run a script, open a REPL, or give it a little web server to look
after.

Building a JavaScript engine means getting acquainted with every strange corner
of JavaScript. Closures, promises, proxies, regular expressions, the surprising
things you can do to an array… TRust JS takes on the whole messy language. That's
part of the fun.

## What's special about it?

The engine handles each stage from reading source code to generating
native machine code. It has three ways to run JavaScript: an interpreter, a
bytecode VM, and a JIT compiler for ARM64 and x86-64. We can run the same program
through all three and check that they agree. Making it faster should still leave
you with the same JavaScript.

It also has a real day job. Being TRust's engine means dealing with the code
websites actually ship, alongside the smaller programs that help us pin down
bugs. There's plenty to explore outside the browser too: modules, async/await,
`Intl`, `Temporal`, and a runtime with filesystem access, timers, web APIs, and
growing Node.js compatibility.

Much of TRust JS is built with Rust's standard library. We use a few dependencies
where they earn their keep, and some runtime features use system libraries.
Keeping the pieces understandable matters to us.

TRust JS is still an ambitious work in progress. Compatibility and performance keep
improving, and there's a lot left to do. If you like poking around language
runtimes—or finding the tiny JavaScript program that makes one fall over—you'll
probably feel at home here.

## Give it a try

With Rust and Cargo installed, build the runtime from the repository root:

```sh
cargo build --release -p trust-js-cli
./target/release/trust-js-cli -e 'console.log("Hello from TRust JS!")'
./target/release/trust-js-cli repl
```

Or run your own script:

```sh
./target/release/trust-js-cli hello.js
```

On Windows, the executable is `target\release\trust-js-cli.exe`.

For something bigger, the [examples](examples/) include a
[Hono web app](examples/hono-app/README.md),
[React server rendering](examples/react-ssr/README.md), and a
[Vite build](examples/vite-app/README.md). Each has its own setup instructions;
they're useful places to start exploring what works.

## Come poke around

The [engine](crates/trust-js/) and [runtime](crates/trust-js-runtime/) are separate, so
you can explore either one or embed the engine in your own project. Build notes,
testing commands, and guidance for working on the code live in [AGENTS.md](AGENTS.md).

Bug reports with small reproducers are especially welcome. JavaScript has an
excellent supply of weird little edge cases.

TRust JS is [MIT licensed](LICENSE).
