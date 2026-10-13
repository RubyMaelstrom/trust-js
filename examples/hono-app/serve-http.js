import app from './app.js';
TRustJS.serve(app.fetch, {
  hostname: '127.0.0.1',
  port: 3000,
  onListen: ({ port }) => console.log(`hono on trust-js: http://localhost:${port}/`),
});
