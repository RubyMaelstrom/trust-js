// RFC 6455 Autobahn fuzzing-client testee. The server selects each case; the client only echoes
// messages with their original text/binary type and advances after the connection closes.
const base = "ws://127.0.0.1:9001";
const agent = "TRust JS";

function socket(path) {
  const ws = new WebSocket(base + path);
  ws.binaryType = "arraybuffer";
  return ws;
}

function runCase(number, count) {
  if (number > count) {
    const report = socket(`/updateReports?agent=${encodeURIComponent(agent)}`);
    report.onopen = () => report.close();
    report.onclose = () => console.log(`AUTOBAHN COMPLETE ${count}`);
    report.onerror = error => console.log(`AUTOBAHN REPORT ERROR ${error}`);
    return;
  }

  const ws = socket(`/runCase?case=${number}&agent=${encodeURIComponent(agent)}`);
  ws.onmessage = event => ws.send(event.data);
  ws.onclose = () => runCase(number + 1, count);
  ws.onerror = () => {};
}

const countSocket = socket("/getCaseCount");
countSocket.onmessage = event => {
  const count = Number(event.data);
  console.log(`AUTOBAHN CASES ${count}`);
  countSocket.onclose = () => runCase(1, count);
  countSocket.close();
};
countSocket.onerror = error => console.log(`AUTOBAHN COUNT ERROR ${error}`);
