// A real-world Express app on the trust-js runtime — CommonJS, running unmodified on trust-js's
// node:http (which bridges to TRustJS.serve). Exercises real middleware: morgan (request logging),
// cors (CORS headers), and Express's own JSON body parser and router.
const express = require('express');
const morgan = require('morgan');
const cors = require('cors');

const app = express();

// Third-party middleware, used exactly as in a Node app.
app.use(morgan('tiny'));               // logs each request line to stdout
app.use(cors());                       // adds Access-Control-* headers
app.use(express.json());               // parses application/json request bodies

// A plain text route.
app.get('/', (req, res) => {
  res.send('Hello from Express on trust-js!\n');
});

// JSON with a route param.
app.get('/users/:id', (req, res) => {
  res.json({ id: req.params.id, from: req.ip });
});

// Query string.
app.get('/search', (req, res) => {
  res.json({ q: req.query.q ?? null, page: Number(req.query.page ?? 1) });
});

// POST with a JSON body echoed back (via express.json()).
app.post('/echo', (req, res) => {
  res.status(201).json({ youSent: req.body });
});

// Custom header + status.
app.get('/teapot', (req, res) => {
  res.set('X-Powered-By', 'trust-js').status(418).send("I'm a teapot\n");
});

// 404 handler (Express falls through to this).
app.use((req, res) => {
  res.status(404).json({ error: 'not found', path: req.path });
});

module.exports = app;
