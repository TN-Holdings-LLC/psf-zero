"""Sandbox test double: high-effort requests end at the token limit with no content every other call;
low-effort (salvage) requests return a circuit. Circuits alternate wrong / right for w3."""
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
W_BAD = {"n_qubits": 3, "gates": [{"name": "h", "qubits": [0]}, {"name": "cx", "qubits": [0, 1]}]}
W_OK = {"n_qubits": 3, "gates": [{"name": "ry", "qubits": [0], "params": [1.2309594173407747]}, {"name": "x", "qubits": [0]},
        {"name": "cry", "qubits": [0, 1], "params": [1.5707963267948966]}, {"name": "x", "qubits": [0]},
        {"name": "cx", "qubits": [1, 2]}, {"name": "x", "qubits": [0]}, {"name": "cx", "qubits": [0, 2]}, {"name": "x", "qubits": [0]}]}
st = {"i": 0}
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        open("fake3_log.jsonl", "a").write(json.dumps({"effort": b.get("reasoning_effort"), "max_tokens": b["max_tokens"],
            "seed": b["seed"], "last_user": b["messages"][-1]["content"][-300:]}) + "\n")
        st["i"] += 1
        if b.get("reasoning_effort") == "low":
            content, fin = "ok\n```json\n" + json.dumps(W_BAD if st["i"] < 4 else W_OK) + "\n```", "stop"
        elif st["i"] % 2 == 1:
            content, fin = "", "length"
        else:
            content, fin = "```json\n" + json.dumps(W_BAD) + "\n```", "stop"
        out = {"choices": [{"message": {"content": content, "reasoning_content": "r" * 50}, "finish_reason": fin}], "usage": {"completion_tokens": 7}}
        self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(out).encode())
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
