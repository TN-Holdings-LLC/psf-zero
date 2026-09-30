"""Test double of the vLLM chat endpoint (sandbox only): scripted replies, including an HTTP 400,
an HTTP 500, a reasoning field, an empty content and a triangle circuit for fill27."""
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
TRI = {"n_qubits": 27, "gates": sum(([{"name": "h", "qubits": [3*k]}, {"name": "cx", "qubits": [3*k, 3*k+1]},
        {"name": "cx", "qubits": [3*k+1, 3*k+2]}, {"name": "cx", "qubits": [3*k, 3*k+2]}, {"name": "cx", "qubits": [3*k, 3*k+2]}]
        for k in range(7)), []) + sum(([{"name": "h", "qubits": [21+2*j]}, {"name": "cx", "qubits": [21+2*j, 22+2*j]}] for j in range(3)), [])}
SCRIPT = ["400", "think... ```json\n{\"n_qubits\": 3, \"gates\": [{\"name\": \"x\", \"qubits\": [0], \"params\": []}], \"note\": \"x\"}\n```",
          "500", "", "```json\n" + json.dumps(TRI) + "\n```"]
state = {"i": 0, "log": []}
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        state["log"].append({k: body[k] for k in body if k != "messages"} | {"n_messages": len(body["messages"])})
        open("fake_server_log.jsonl", "a").write(json.dumps(state["log"][-1]) + "\n")
        item = SCRIPT[min(state["i"], len(SCRIPT) - 1)]; state["i"] += 1
        if item in ("400", "500"):
            self.send_response(int(item)); self.end_headers(); self.wfile.write(b'{"error":"scripted"}'); return
        out = {"choices": [{"message": {"content": item, "reasoning_content": "r" * 123}, "finish_reason": "stop"}],
               "usage": {"completion_tokens": 10}}
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps(out).encode())
HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
