"use strict";
/* Borrowed from https://github.com/ai/nanoid/blob/3.0.2/non-secure/index.js

The MIT License (MIT)

Copyright 2017 Andrey Sitnik <andrey@sitnik.ru>

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. */

// This alphabet uses `A-Za-z0-9_-` symbols.
// The order of characters is optimized for better gzip and brotli compression.
// References to the same file (works both for gzip and brotli):
// `'use`, `andom`, and `rict'`
// References to the brotli default dictionary:
// `-26T`, `1983`, `40px`, `75px`, `bush`, `jack`, `mind`, `very`, and `wolf`
let urlAlphabet = "useandom-26T198340PX75pxJACKVERYMINDBUSHWOLF_GQZbfghjklqvwyzrict";

function nanoid(size = 21) {
    let id = "";
    // A compact alternative for `for (var i = 0; i < step; i++)`.
    let i = size;
    while (i--) {
        // `| 0` is more compact and faster than `Math.floor()`.
        id += urlAlphabet[(Math.random() * 64) | 0];
    }
    return id;
}

const todoReducer = (state, action) => {
    switch (action.type) {
        case "ADD_ITEM":
            return state.concat({ id: nanoid(), title: action.payload.title, completed: false });
        case "UPDATE_ITEM":
            return state.map((todo) => (todo.id === action.payload.id ? { ...todo, title: action.payload.title } : todo));
        case "REMOVE_ITEM":
            return state.filter((todo) => todo.id !== action.payload.id);
        case "TOGGLE_ITEM":
            return state.map((todo) => (todo.id === action.payload.id ? { ...todo, completed: !todo.completed } : todo));
        case "REMOVE_ALL_ITEMS":
            return [];
        case "TOGGLE_ALL":
            return state.map((todo) => (todo.completed !== action.payload.completed ? { ...todo, completed: action.payload.completed } : todo));
        case "REMOVE_COMPLETED_ITEMS":
            return state.filter((todo) => !todo.completed);
    }

    throw Error(`Unknown action: ${action.type}`);
};

// Derived application probe for the unchanged Speedometer React todoReducer.
// The source module is prepended with its one `export ` token removed, and
// an explicit strict directive preserves the original module strictness.
// Deterministic initial state avoids timing nanoid/Math.random and tests reducers.
function reducerBatch(passes) {
    let checksum = 0;
    for (let pass = 0; pass < passes; pass++) {
        let state = [];
        for (let i = 0; i < 100; i++) state.push({id: i, title: 'Todo ' + i, completed: false});
        for (let i = 0; i < 100; i++)
            state = todoReducer(state, {type: 'TOGGLE_ITEM', payload: {id: i}});
        for (let i = 0; i < 100; i++)
            state = todoReducer(state, {type: 'UPDATE_ITEM', payload: {id: i, title: 'changed-' + i}});
        if (state.length !== 100) throw new Error('Wrong updated state length');
        for (let i = 0; i < 100; i++) {
            if (state[i].id !== i || state[i].title !== 'changed-' + i || state[i].completed !== true)
                throw new Error('Wrong individual update at ' + i);
            checksum += state[i].id + 1 + state[i].title.length + 1000;
        }
        state = todoReducer(state, {type: 'REMOVE_ITEM', payload: {id: 7}});
        if (state.length !== 99) throw new Error('Wrong removed state length');
        checksum += state.length;
        state = todoReducer(state, {type: 'TOGGLE_ALL', payload: {completed: false}});
        for (let i = 0; i < state.length; i++)
            if (state[i].completed !== false || state[i].id === 7) throw new Error('Wrong toggle false');
        state = todoReducer(state, {type: 'TOGGLE_ALL', payload: {completed: true}});
        for (let i = 0; i < state.length; i++) {
            if (state[i].completed !== true) throw new Error('Wrong toggle true');
            checksum++;
        }
        state = todoReducer(state, {type: 'REMOVE_COMPLETED_ITEMS'});
        if (state.length !== 0) throw new Error('Completed items survived removal');
    }
    // Per pass: 1+...+100 = 5050; title lengths = 10*9+90*10 = 990;
    // completed flags = 100000; post-removal length and toggled count = 99+99.
    if (checksum !== 106238 * passes) throw new Error('Independent checksum mismatch');
    return checksum;
}
reducerBatch(2);
const reducerStart = Date.now();
const reducerChecksum = reducerBatch(20);
const reducerElapsed = Date.now() - reducerStart;
console.log(JSON.stringify({case: 'react-reducer', passes: 20, elapsed_ms: reducerElapsed,
    checksum: reducerChecksum, expected: 2124760}));
