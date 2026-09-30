"""Sandbox test double for tool calling: with tools offered and no tool result yet, the reply is a
simulate() call; after a tool result, the final JSON. Mode 'reject' answers 400 whenever tools are sent."""
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
MODE = sys.argv[2]
W_OK = {"n_qubits": 3, "gates": [{"name": "ry", "qubits": [0], "params": [1.2309594173407747]}, {"name": "x", "qubits": [0]},
        {"name": "cry", "qubits": [0, 1], "params": [1.5707963267948966]}, {"name": "x", "qubits": [0]},
        {"name": "cx", "qubits": [1, 2]}, {"name": "x", "qubits": [0]}, {"name": "cx", "qubits": [0, 2]}, {"name": "x", "qubits": [0]}]}
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        roles = [m["role"] for m in b["messages"]]
        open("fake4_log.jsonl", "a").write(json.dumps({"tools": "tools" in b, "roles": roles, "seed": b["seed"],
            "last": str(b["messages"][-1].get("content"))[:200]}) + "\n")
        if MODE == "reject" and "tools" in b:
            self.send_response(400); self.end_headers(); self.wfile.write(b'{"error":"auto tool choice requires --enable-auto-tool-choice"}'); return
        if "tools" in b and roles[-1] != "tool":
            msg = {"content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "simulate",
                   "arguments": json.dumps({"gates": W_OK["gates"][:3]})}}]}
            fin = "tool_calls"
        else:
            msg, fin = {"content": "```json\n" + json.dumps(W_OK) + "\n```"}, "stop"
        out = {"choices": [{"message": msg, "finish_reason": fin}], "usage": {"completion_tokens": 5}}
        self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(out).encode())
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
