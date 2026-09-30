"""e2e_vllm_psf_v9.py -- v8 plus three changes found in pilot 2 (exploratory, workplace, 2026-09-30):
  (6) the salvage request uses medium effort and 8,000 tokens (low effort gave 2 correct of 7 in pilot 2);
  (7) the reply budget is set by the caller (the pilot uses 48,000 tokens; W3 still hit 32,000 in 7 of 12 rounds);
  (8) optional simulator tool (--tool-sim): the model may call simulate(circuit) during a round and gets back the
      amplitudes of ITS OWN circuit (never the target, never a fidelity), so it need not simulate by hand in its
      reasoning. At most --max-tool-calls calls per round. If the server rejects tools, they are switched off
      for the rest of the run and this is recorded.
v8 docstring follows.
e2e_vllm_psf_v8.py -- v7 plus two changes found in the W3 pilot (exploratory, workplace, 2026-09-30):
  (4) salvage: when a reply ends at the token limit without a JSON circuit, one short follow-up request
      (reasoning_effort low, --salvage-tokens) asks for the best circuit now, inside the same round;
  (5) memory: the feedback carries the best circuit so far (JSON) and its fidelity, because the history keeps
      only the last two exchanges and the pilot showed answers oscillating and regressing.
v7 docstring follows.
e2e_vllm_psf_v7.py -- v6 plus the three changes named after the 2026-09-30 CUT (workplace, 2026-09-30):
  (1) larger reply budget (set by the caller: --max-tokens), (2) the controlled-phase gate cp is allowed,
  (3) the system prompt asks for SHORT reasoning and always a final JSON (no gate-by-gate state writing).
Everything else is v6. First use: an exploratory pilot on the weak task (w3) before any new pre-registration.

v6 docstring follows.
e2e_vllm_psf_v6.py -- pre-registered (workplace design, 2026-09-30): does a language model served by
vLLM, coupled to PSF-Zero, earn further investment? Go/no-go test on one RunPod B200.

Loop (per task, per run, up to --rounds rounds):
  entry   a task in words; the exact target state is computed here and never shown to the model
  1       the model proposes a circuit as JSON (free text: it may reason first; the last JSON block counts)
  2       JSON -> PennyLane tape (gate names, qubit indices, parameters validated; a few aliases accepted)
  3       logical check on the CPU simulator (lightning.qubit): fidelity with the target. The target of
          every task is a product over qubit groups, so the check simulates each connected component of
          (circuit interaction graph + target groups) separately; a 27-qubit task stays cheap unless the
          circuit really entangles everything
  4       tape -> Qiskit -> PSF-Zero compile_for_hardware (FakeAuckland, layout_search, entangling_basis cx)
          and, for reference, Qiskit transpile optimization_level=3 on the same input, both timed
  5       compiled check: the PSF output is simulated component by component on physical qubits and read
          at the final layout (ancillas projected on |0>)
  6       feedback to the model: fidelities, the wrong groups with their amplitudes next to the target's, and
          for tasks of <= 4 qubits the state after every gate; the temperature rises when the same wrong
          answer or the same error repeats
  exit    per task and run: rounds.jsonl (every reply, feedback and number), result.json, the best circuit
          (JSON, PennyLane ops, OpenQASM 3), and a StatePreparation baseline compiled the same two ways

v6 over v4 (exploratory, 2026-09-29): HTTP 400 (context) trims the history and retries instead of
aborting; reasoning models (gpt-oss) get their own token budget and reasoning effort; --run sets the
request seeds; parameters on parameter-free gates are ignored; per-gate states; the fill27 task (the
27-qubit device completely filled: seven GHZ-3 and three Bell pairs); every valid circuit is compiled by
PSF-Zero and by Qiskit L3; per-round timings (llm_s, compile_s, q3_s, sim_s); CPU simulation by default;
--retime re-measures the fill27 compile times sequentially after all model runs (for G3).

No IBM account and no network access to IBM: device snapshots only. The only network use is the HTTP call
to the local vLLM server (default http://127.0.0.1:8000/v1). No credentials are read or written.

    python -u e2e_vllm_psf_v6.py --model M --run 1 --tasks w3 --out OUT/M/run1 --layout-dir ... --repo ...
    python -u e2e_vllm_psf_v6.py --mock-llm --out mock                  (pipeline test without a model)
    python -u e2e_vllm_psf_v6.py --retime OUT --layout-dir ... --repo ...   (G3 re-timing pass)
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

import numpy as np

SCRIPT_VERSION = "v9 2026-09-30"

GATES_1Q = {"h", "x", "y", "z", "s", "sdg", "t", "tdg"}
GATES_1Q_PARAM = {"rx", "ry", "rz"}
GATES_2Q = {"cx", "cz", "swap"}
GATES_2Q_PARAM = {"cry", "crz", "cp"}
ALL_GATES = GATES_1Q | GATES_1Q_PARAM | GATES_2Q | GATES_2Q_PARAM
ALIASES = {"cnot": "cx", "hadamard": "h", "sdag": "sdg", "tdag": "tdg", "s_dag": "sdg", "t_dag": "tdg",
           "paulix": "x", "pauliy": "y", "pauliz": "z", "cphase": "cp", "cu1": "cp",
           "controlledphaseshift": "cp"}

SYSTEM = (
    "You design quantum circuits. Think briefly: a few short lines are enough; do not simulate every gate in "
    "detail. You MUST always end your reply with the final circuit as ONE JSON object inside a ```json ... ``` "
    "block, even if you are not sure it is right, in this form:\n"
    '{"n_qubits": <int>, "gates": [{"name": "<gate>", "qubits": [<int>, ...], "params": [<float>, ...]}], '
    '"note": "<one short sentence>"}\n'
    "Allowed gates: h, x, y, z, s, sdg, t, tdg (1 qubit, no params); rx, ry, rz (1 qubit, 1 param in radians); "
    "cx, cz, swap (2 qubits; for cx the first qubit is the control); cry, crz, cp (2 qubits, control first, "
    "1 param in radians; cp(phi) multiplies |11> by exp(i*phi); write params as plain numbers, e.g. 0.785398). Qubits are numbered from 0. In a ket "
    "such as |q0 q1 q2>, qubit 0 is the LEFTMOST bit. All qubits start in |0>. Get the state exactly right "
    "first; then use as few two-qubit gates as possible."
)


SALVAGE = ("Your reasoning ran out of its budget before you gave an answer. Do not start over: give your best "
           "circuit now, as ONE JSON object in a ```json ... ``` block, with at most two sentences before it.")


# ---------------------------------------------------------------- tasks (targets never shown to the model)

def t_ghz(n):
    v = np.zeros(2 ** n, complex)
    v[0] = v[-1] = 1 / np.sqrt(2)
    return v


def t_w(n):
    v = np.zeros(2 ** n, complex)
    for i in range(n):
        v[2 ** (n - 1 - i)] = 1 / np.sqrt(n)
    return v


def t_qft_basis(n, x):
    N = 2 ** n
    return np.exp(2j * np.pi * x * np.arange(N) / N) / np.sqrt(N)


BELL = np.array([1, 0, 0, 1], complex) / np.sqrt(2)

# name -> (n_qubits, text, groups); groups = [(wires, target vector with wires[0] as MSB), ...]
TASKS = {
    "ghz5": (5, "Prepare the 5-qubit GHZ state (|00000> + |11111>)/sqrt(2).", [(list(range(5)), t_ghz(5))]),
    "w3": (3, "Prepare the 3-qubit W state (|001> + |010> + |100>)/sqrt(3).", [([0, 1, 2], t_w(3))]),
    "bell3": (6, "Prepare three Bell pairs (|00> + |11>)/sqrt(2): one on qubits (0,1), one on (2,3), one on (4,5).",
              [([2 * k, 2 * k + 1], BELL) for k in range(3)]),
    "qft3": (3, "Start from the basis state |101> (qubit 0 = 1, qubit 1 = 0, qubit 2 = 1) and apply the 3-qubit "
                "quantum Fourier transform, so that the result is sum_k exp(2*pi*i*5*k/8)|k>/sqrt(8), where k is "
                "read with qubit 0 as the most significant bit.", [([0, 1, 2], t_qft_basis(3, 5))]),
    "fill27": (27, "Prepare seven 3-qubit GHZ states (|000> + |111>)/sqrt(2), on qubits (0,1,2), (3,4,5), (6,7,8), "
                   "(9,10,11), (12,13,14), (15,16,17) and (18,19,20), and three Bell pairs (|00> + |11>)/sqrt(2), on "
                   "qubits (21,22), (23,24) and (25,26). The circuit will run on a 27-qubit device whose qubits are "
                   "connected as a heavy-hexagon lattice (every qubit has at most three neighbours, and two-qubit "
                   "gates act only between neighbours), and all 27 qubits are needed, so there is no spare qubit.",
               [(list(range(3 * k, 3 * k + 3)), t_ghz(3)) for k in range(7)]
               + [([21 + 2 * j, 22 + 2 * j], BELL) for j in range(3)]),
}

MOCK = {  # pipeline test only: canned replies; the first w3 and fill27 replies are wrong on purpose
    "ghz5": [{"n_qubits": 5, "gates": [{"name": "h", "qubits": [0]}] +
              [{"name": "cx", "qubits": [i, i + 1]} for i in range(4)], "note": "chain of CNOTs"}],
    "w3": [{"n_qubits": 3, "gates": [{"name": "h", "qubits": [0], "params": [0]}, {"name": "cnot", "qubits": [0, 1]},
                                     {"name": "cx", "qubits": [1, 2]}], "note": "first try"},
           {"n_qubits": 3, "gates": [{"name": "ry", "qubits": [0], "params": ["2*acos(sqrt(2/3))"]},
                                     {"name": "x", "qubits": [0]}, {"name": "cry", "qubits": [0, 1], "params": [np.pi / 2]},
                                     {"name": "cx", "qubits": [1, 2]}, {"name": "cx", "qubits": [0, 2]},
                                     {"name": "x", "qubits": [0]}], "note": "split amplitude, then distribute"}],
    "bell3": [{"n_qubits": 6, "gates": sum(([{"name": "h", "qubits": [2 * k]}, {"name": "cx", "qubits": [2 * k, 2 * k + 1]}]
                                            for k in range(3)), []), "note": "three H+CNOT"}],
    "qft3": [{"n_qubits": 3, "gates": [{"name": "x", "qubits": [0]}, {"name": "x", "qubits": [2]}, {"name": "h", "qubits": [0]},
                       {"name": "h", "qubits": [1]}, {"name": "cp", "qubits": [1, 0], "params": [np.pi / 2]}], "note": "wrong on purpose (H first)"},
             {"n_qubits": 3, "gates": [{"name": "x", "qubits": [0]}, {"name": "x", "qubits": [2]},
                                       {"name": "h", "qubits": [0]}, {"name": "cp", "qubits": [1, 0], "params": [np.pi / 2]},
                                       {"name": "cphase", "qubits": [2, 0], "params": [np.pi / 4]}, {"name": "h", "qubits": [1]},
                                       {"name": "cp", "qubits": [2, 1], "params": [np.pi / 2]}, {"name": "h", "qubits": [2]},
                                       {"name": "swap", "qubits": [0, 2]}], "note": "textbook QFT with cp"},
             {"n_qubits": 3, "gates": [{"name": "x", "qubits": [0]}, {"name": "x", "qubits": [2]},
                                       {"name": "h", "qubits": [0]}, {"name": "crz", "qubits": [1, 0], "params": [np.pi / 2]},
                                       {"name": "crz", "qubits": [2, 0], "params": [np.pi / 4]}, {"name": "h", "qubits": [1]},
                                       {"name": "crz", "qubits": [2, 1], "params": [np.pi / 2]}, {"name": "h", "qubits": [2]},
                                       {"name": "swap", "qubits": [0, 2]}], "note": "textbook QFT"}],
    "fill27": [{"n_qubits": 27, "gates": sum(([{"name": "h", "qubits": [3 * k]}, {"name": "cx", "qubits": [3 * k, 3 * k + 1]},
                                               {"name": "cx", "qubits": [3 * k, 3 * k + 2]}] for k in range(6)), [])
                + [{"name": "h", "qubits": [21 + 2 * j]} for j in range(3)], "note": "first try (incomplete)"},
               {"n_qubits": 27, "gates": sum(([{"name": "h", "qubits": [3 * k]}, {"name": "cx", "qubits": [3 * k, 3 * k + 1]},
                                               {"name": "cx", "qubits": [3 * k + 1, 3 * k + 2]}] for k in range(7)), [])
                + sum(([{"name": "h", "qubits": [21 + 2 * j]}, {"name": "cx", "qubits": [21 + 2 * j, 22 + 2 * j]}]
                       for j in range(3)), []), "note": "paths of CNOTs"}],
}


# ---------------------------------------------------------------- model

class ContextError(Exception):
    pass


TOOL_SPEC = [{"type": "function", "function": {
    "name": "simulate",
    "description": ("Simulate a candidate circuit from |0...0> and return the largest amplitudes of the state it "
                    "produces (qubit 0 is the leftmost bit). Uses the same gate names and conventions as the final "
                    "answer. It does not know the task's target state."),
    "parameters": {"type": "object", "properties": {
        "gates": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "qubits": {"type": "array", "items": {"type": "integer"}},
            "params": {"type": "array", "items": {"type": "number"}}}, "required": ["name", "qubits"]}}},
        "required": ["gates"]}}}]


def _np_gate(name, ps):
    import math
    c, s_ = (math.cos(ps[0] / 2), math.sin(ps[0] / 2)) if ps else (0, 0)
    one = {"h": np.array([[1, 1], [1, -1]]) / math.sqrt(2), "x": np.array([[0, 1], [1, 0]]),
           "y": np.array([[0, -1j], [1j, 0]]), "z": np.diag([1, -1]), "s": np.diag([1, 1j]), "sdg": np.diag([1, -1j]),
           "t": np.diag([1, np.exp(1j * math.pi / 4)]), "tdg": np.diag([1, np.exp(-1j * math.pi / 4)])}
    if name in one:
        return one[name]
    if name in ("rx",):
        return np.array([[c, -1j * s_], [-1j * s_, c]])
    if name in ("ry", "cry"):
        return np.array([[c, -s_], [s_, c]])
    if name in ("rz", "crz"):
        return np.diag([np.exp(-1j * ps[0] / 2), np.exp(1j * ps[0] / 2)])
    if name == "cp":
        return np.diag([1, np.exp(1j * ps[0])])
    return {"cx": np.array([[0, 1], [1, 0]]), "cz": np.diag([1, -1])}[name]


def np_simulate(gates, wires):
    """Plain numpy state vector of the gates on `wires` (sorted), wires[0] = leftmost bit."""
    k = len(wires)
    idx = {w: i for i, w in enumerate(wires)}
    psi = np.zeros(2 ** k, complex)
    psi[0] = 1
    for name, qs, ps in gates:
        q = [idx[x] for x in qs]
        t = psi.reshape((2,) * k)
        if name == "swap":
            t = np.swapaxes(t, q[0], q[1])
        elif len(q) == 1:
            t = np.moveaxis(np.tensordot(_np_gate(name, ps), t, axes=([1], [q[0]])), 0, q[0])
        else:
            t = t.copy()
            sl = [slice(None)] * k
            sl[q[0]] = 1
            sub = t[tuple(sl)]
            tt = q[1] if q[1] < q[0] else q[1] - 1
            t[tuple(sl)] = np.moveaxis(np.tensordot(_np_gate(name, ps), sub, axes=([1], [tt])), 0, tt)
        psi = t.reshape(-1)
    return psi


def sim_tool(arguments, n):
    """The simulate tool: amplitudes of the model's own circuit, per connected component of its gates."""
    try:
        spec = json.loads(arguments) if isinstance(arguments, str) else arguments
        spec = {"n_qubits": n, "gates": spec.get("gates", [])}
        to_tape(spec, n)  # same validation as a final answer
        gates = []
        for g in spec["gates"]:
            name = ALIASES.get(str(g["name"]).lower().strip(), str(g["name"]).lower().strip())
            raw = g.get("params", []) or []
            raw = raw if isinstance(raw, list) else [raw]
            ps = [_safe_eval(x) for x in raw] if name in GATES_1Q_PARAM | GATES_2Q_PARAM else []
            gates.append((name, [int(x) for x in g["qubits"]], ps))
        uf = _UF(range(n))
        for _, qs, _ in gates:
            for a in qs[1:]:
                uf.union(qs[0], a)
        if n <= 10:
            comps = [list(range(n))]
        else:
            comps = [c for c in uf.comps() if any(set(qs) & set(c) for _, qs, _ in gates)]
        out = []
        for c in comps:
            if len(c) > 12:
                out.append(f"qubits {tuple(c)}: {len(c)} qubits entangled together, too large to show")
                continue
            psi = np_simulate([g for g in gates if g[1][0] in c], c)
            out.append((f"qubits {tuple(c)}: " if len(comps) > 1 or n > 10 else "") + ket_str(psi, len(c), k=16))
        return "State of your circuit: " + (" | ".join(out) if out else "all qubits stay |0>") + \
            (" (qubits not listed stay |0>)" if n > 10 else "")
    except Exception as e:  # the tool reports errors instead of failing the round
        return f"The tool could not simulate this circuit: {type(e).__name__}: {e}"


def converse(args, messages, temperature, seed, max_tokens, n, effort=None):
    """One model turn; with --tool-sim the model may call simulate() up to --max-tool-calls times."""
    tools_log, t_all, usage_all, reasoning_all, msgs = [], 0.0, {}, 0, list(messages)
    for k in range(args.max_tool_calls + 1):
        use_tools = args.tool_sim and k < args.max_tool_calls
        try:
            msg, fin, dt, usage = ask_model_raw(args, msgs, temperature, seed + 100 * k, max_tokens, effort,
                                                tools=use_tools)
        except ContextError as e:
            if use_tools and ("tool" in str(e).lower() or "auto" in str(e).lower()):
                args.tool_sim = False  # server does not accept tools: switch them off for this run
                tools_log.append({"disabled": str(e)[:300]})
                continue
            raise
        t_all += dt
        reasoning_all += len(msg.get("reasoning_content") or msg.get("reasoning") or "")
        for key, v in (usage or {}).items():
            if isinstance(v, (int, float)):
                usage_all[key] = usage_all.get(key, 0) + v
        calls = msg.get("tool_calls") or []
        if not calls:
            return (msg.get("content") or ""), reasoning_all, t_all, usage_all, fin, tools_log
        msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for c in calls:
            res = sim_tool(c.get("function", {}).get("arguments", "{}"), n)
            tools_log.append({"arguments": str(c.get("function", {}).get("arguments"))[:1500], "result": res[:1500]})
            msgs.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": res})
    return "", reasoning_all, t_all, usage_all, "tool_limit", tools_log


def ask_model(args, messages, temperature, seed, max_tokens, effort=None):
    msg, fin, dt, usage = ask_model_raw(args, messages, temperature, seed, max_tokens, effort, tools=False)
    return (msg.get("content") or ""), (msg.get("reasoning_content") or msg.get("reasoning") or ""), dt, usage, fin


def ask_model_raw(args, messages, temperature, seed, max_tokens, effort=None, tools=False):
    body = {"model": args.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
            "seed": seed}
    if effort or args.reasoning_effort:
        body["reasoning_effort"] = effort or args.reasoning_effort
    if tools:
        body["tools"] = TOOL_SPEC
        body["tool_choice"] = "auto"
    req = urllib.request.Request(args.api.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            out = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        if e.code == 400:
            raise ContextError(detail) from None
        raise
    return out["choices"][0]["message"], out["choices"][0].get("finish_reason"), time.perf_counter() - t0, \
        out.get("usage", {})


def _safe_eval(expr):
    """Evaluate a numeric expression such as '-math.pi / 6' or '2*acos(sqrt(1/3))'."""
    import ast
    import math
    if isinstance(expr, (int, float)):
        return float(expr)
    names = {"pi": math.pi, "sqrt": math.sqrt, "acos": math.acos, "asin": math.asin, "atan": math.atan,
             "atan2": math.atan2, "cos": math.cos, "sin": math.sin, "e": math.e}
    expr = re.sub(r"\b(math|np|numpy)\.", "", str(expr).strip().strip('"').strip("'")).replace("π", "pi")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in names and not callable(names[node.id]):
            return names[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            a, b = ev(node.left), ev(node.right)
            return {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
                    ast.Div: lambda: a / b, ast.Pow: lambda: a ** b}[type(node.op)]()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and callable(names.get(node.func.id)):
            return names[node.func.id](*[ev(x) for x in node.args])
        raise ValueError(f"unsupported expression {expr!r}")
    return float(ev(ast.parse(expr, mode="eval")))


def extract_json(text):
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    cand = blocks[-1] if blocks else text[text.find("{"): text.rfind("}") + 1]
    if not cand:
        raise ValueError("no JSON object found in the reply")
    cand = re.sub(r"(#|//)[^\n]*", "", cand)  # models add comments inside the JSON
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        def fix(m):  # parameters written as expressions (math.pi/4): evaluate them safely
            items = [x for x in m.group(2).split(",") if x.strip()]
            return m.group(1) + ", ".join(repr(_safe_eval(x)) for x in items) + m.group(3)
        return json.loads(re.sub(r'("params"\s*:\s*\[)([^\]]*)(\])', fix, cand))


def ket_str(v, n, k=8):
    """The largest amplitudes of a state as text, qubit 0 leftmost."""
    idx = [i for i in np.argsort(-np.abs(v)) if abs(v[i]) > 1e-3][:k]
    parts = []
    for i in sorted(idx):
        a = v[i]
        amp = f"{a.real:+.3f}" if abs(a.imag) < 1e-3 else (f"{a.imag:+.3f}i" if abs(a.real) < 1e-3 else f"({a.real:+.3f}{a.imag:+.3f}i)")
        parts.append(f"{amp}|{i:0{n}b}>")
    more = int(np.sum(np.abs(v) > 1e-3)) - len(idx)
    return (" ".join(parts) + (f" (+{more} more)" if more > 0 else "")) or "(zero vector)"


# ---------------------------------------------------------------- circuits

def to_tape(spec, n):
    """Validated JSON -> PennyLane tape. Returns (tape, logical two-qubit count, notes, gate names)."""
    import pennylane as qml
    if not isinstance(spec, dict) or "gates" not in spec:
        raise ValueError("the JSON object needs a 'gates' list")
    if int(spec.get("n_qubits", n)) != n:
        raise ValueError(f"n_qubits must be {n}, got {spec.get('n_qubits')}")
    ops, two_q, notes, names = [], 0, [], []
    for k, g in enumerate(spec["gates"]):
        name = str(g["name"]).lower().strip()
        name = ALIASES.get(name, name)
        qs = [int(q) for q in g["qubits"]]
        raw_p = g.get("params", []) or []
        if not isinstance(raw_p, list):
            raw_p = [raw_p]
        if name not in ALL_GATES:
            raise ValueError(f"gate {k}: unknown gate {name!r}")
        if any(q < 0 or q >= n for q in qs) or len(set(qs)) != len(qs):
            raise ValueError(f"gate {k}: bad qubits {qs}")
        need_q = 1 if name in GATES_1Q | GATES_1Q_PARAM else 2
        need_p = 1 if name in GATES_1Q_PARAM | GATES_2Q_PARAM else 0
        if need_p == 0 and raw_p:
            notes.append(f"gate {k}: params on {name} ignored")
            raw_p = []
        ps = [_safe_eval(p) for p in raw_p]
        if len(qs) != need_q or len(ps) != need_p:
            raise ValueError(f"gate {k}: {name} needs {need_q} qubit(s) and {need_p} param(s)")
        simple = {"h": qml.Hadamard, "x": qml.PauliX, "y": qml.PauliY, "z": qml.PauliZ}
        if name in simple:
            ops.append(simple[name](wires=qs[0]))
        elif name in ("rx", "ry", "rz"):
            ops.append({"rx": qml.RX, "ry": qml.RY, "rz": qml.RZ}[name](ps[0], wires=qs[0]))
        else:
            op = {"s": lambda: qml.S(wires=qs[0]), "sdg": lambda: qml.adjoint(qml.S(wires=qs[0])),
                  "t": lambda: qml.T(wires=qs[0]), "tdg": lambda: qml.adjoint(qml.T(wires=qs[0])),
                  "cx": lambda: qml.CNOT(wires=qs), "cz": lambda: qml.CZ(wires=qs), "swap": lambda: qml.SWAP(wires=qs),
                  "cry": lambda: qml.CRY(ps[0], wires=qs), "crz": lambda: qml.CRZ(ps[0], wires=qs),
                  "cp": lambda: qml.ControlledPhaseShift(ps[0], wires=qs)}[name]()
            # tape_to_qiskit trusts only QubitUnitary for these (Addenda 164-165)
            ops.append(qml.QubitUnitary(qml.matrix(op, wire_order=qs), wires=qs))
        two_q += need_q == 2
        names.append(name)
    if not ops:
        raise ValueError("the circuit has no gates")
    return qml.tape.QuantumTape(ops, [], shots=None), two_q, notes, names


class _UF:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)

    def comps(self):
        out = {}
        for i in self.p:
            out.setdefault(self.find(i), []).append(i)
        return [sorted(v) for v in out.values()]


def target_on(wires, groups):
    """Target state on `wires` (sorted, a union of whole groups), wires[0] as MSB."""
    order, vec = [], np.array([1], complex)
    for gw, gv in groups:
        if set(gw) <= set(wires):
            order += list(gw)
            vec = np.kron(vec, gv)
    if sorted(order) != list(wires):
        raise ValueError("internal: component is not a union of whole groups plus idle wires")
    idle = [w for w in wires if w not in order]
    for w in idle:
        order.append(w)
        vec = np.kron(vec, np.array([1, 0], complex))
    k = len(wires)
    return np.transpose(vec.reshape((2,) * k), [order.index(w) for w in wires]).reshape(-1)


MAX_COMPONENT = 28


def state_of(dev_name, ops, k):
    import pennylane as qml
    dev = qml.device(dev_name, wires=k)
    tape = qml.tape.QuantumTape(list(ops), [qml.state()], shots=None)
    return np.asarray(qml.execute([tape], dev)[0])


def logical_check(dev_name, tape, n, groups):
    """Component-wise fidelity of the tape's state with the product target. Returns
    (fidelity, [(wires, fidelity, psi, target), ...])."""
    import pennylane as qml
    uf = _UF(range(n))
    for op in tape.operations:
        w = list(op.wires)
        for a in w[1:]:
            uf.union(w[0], a)
    for gw, _ in groups:
        for a in gw[1:]:
            uf.union(gw[0], a)
    parts, f_tot = [], 1.0
    for comp in uf.comps():
        if len(comp) > MAX_COMPONENT:
            raise ValueError(f"the circuit entangles {len(comp)} qubits; too many to check")
        idx = {w: i for i, w in enumerate(comp)}
        ops = [qml.map_wires(op, {w: idx[w] for w in op.wires}) for op in tape.operations if op.wires[0] in idx]
        psi = state_of(dev_name, ops, len(comp)) if ops else np.eye(1, 2 ** len(comp), 0, dtype=complex)[0]
        tgt = target_on(comp, groups)
        f = float(abs(np.vdot(tgt, psi)) ** 2)
        f_tot *= f
        parts.append((comp, f, psi, tgt))
    return f_tot, parts


def compiled_check(dev_name, routed, n, groups):
    """Component-wise check of a routed circuit on physical qubits, read at the final layout (ancillas
    projected on |0>). Returns (fidelity, leaked norm, largest component simulated)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator
    from psf_pennylane_gpu_prototype import qiskit_to_tape
    final = list(routed.layout.final_index_layout(filter_ancillas=True))
    insts = [(inst.operation, [routed.find_bit(q).index for q in inst.qubits]) for inst in routed.data
             if inst.operation.name not in ("barrier", "delay", "measure")]
    phys = set(final)
    for _, qs in insts:
        phys |= set(qs)
    uf = _UF(sorted(phys))
    for _, qs in insts:
        for a in qs[1:]:
            uf.union(qs[0], a)
    for gw, _ in groups:
        for a in gw[1:]:
            uf.union(final[gw[0]], final[a])
    inv = {p: i for i, p in enumerate(final)}
    f_tot, norm_tot, kmax = 1.0, 1.0, 0
    for comp in uf.comps():
        k = len(comp)
        if k > MAX_COMPONENT:
            raise ValueError(f"compiled circuit entangles {k} qubits; too many to check")
        kmax = max(kmax, k)
        idx = {p: i for i, p in enumerate(comp)}
        sub = QuantumCircuit(k)
        for op, qs in insts:
            if qs[0] in idx:
                sub.append(UnitaryGate(Operator(op).data) if len(qs) == 2 else op, [idx[q] for q in qs])
        logical = sorted(inv[p] for p in comp if p in inv)
        if sub.size():
            psi = state_of(dev_name, qiskit_to_tape(sub, list(range(k))).operations, k).reshape((2,) * k)
        else:
            psi = np.eye(1, 2 ** k, 0, dtype=complex)[0].reshape((2,) * k)
        pos = [idx[final[i]] for i in logical]
        anc = [i for i in range(k) if i not in pos]
        phi = np.transpose(psi, pos + anc).reshape(2 ** len(pos), 2 ** len(anc))[:, 0]
        norm_tot *= float(np.vdot(phi, phi).real)
        if logical:
            f_tot *= float(abs(np.vdot(target_on(logical, groups), phi)) ** 2)
    return f_tot, 1 - norm_tot, kmax


def per_gate_states(tape, n, names, limit=14):
    """State after each gate (small n only), as text for the feedback."""
    import pennylane as qml
    psi = np.zeros(2 ** n, complex)
    psi[0] = 1
    lines = []
    for k, op in enumerate(tape.operations[:limit]):
        psi = qml.matrix(op, wire_order=list(range(n))) @ psi
        lines.append(f"  after gate {k} ({names[k]} on {list(op.wires)}): {ket_str(psi, n)}")
    if len(tape.operations) > limit:
        lines.append(f"  ... ({len(tape.operations) - limit} more gates)")
    return "\n".join(lines)


def native_of(target):
    return [g for g in target.operation_names if g in ("cz", "ecr", "cx", "rz", "sx", "x", "id", "rzz")]


def compile_psf(pc, qc, target):
    if hasattr(pc, "_CX_CORE_CACHE"):
        pc._CX_CORE_CACHE.clear()  # as in the release gate v4/v5: no warm synthesis cache
    with contextlib.redirect_stdout(io.StringIO()):
        t0 = time.perf_counter()
        out = pc.compile_for_hardware(qc, coupling_map=target.build_coupling_map(), basis_gates=native_of(target),
                                      entangling_basis="cx", layout_search=True, seed_transpiler=0)
    return out, time.perf_counter() - t0


def compile_q3(qc, target):
    from qiskit import transpile
    t0 = time.perf_counter()
    out = transpile(qc, target=target, optimization_level=3, seed_transpiler=0)
    return out, time.perf_counter() - t0


def twoq(c):
    return sum(1 for i in c.data if len(i.qubits) == 2 and i.operation.name != "barrier")


def swaps(c):
    return sum(1 for i in c.data if i.operation.name == "swap")


def baseline_stateprep(n, groups):
    """Qiskit StatePreparation of each group's target (qubit 0 = MSB, so the qubits are passed reversed),
    decomposed to cx + u at level 0, as a circuit the two compilers then see."""
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import StatePreparation
    qc = QuantumCircuit(n)
    for gw, gv in groups:
        qc.append(StatePreparation(gv), list(reversed(gw)))
    return transpile(qc, basis_gates=["cx", "u"], optimization_level=0)


# ---------------------------------------------------------------- the loop

def feedback_wrong(parts, n, fid):
    bad = [p for p in parts if p[1] < fid]
    lines = []
    for comp, f, psi, tgt in bad[:4]:
        if len(comp) <= 6:
            lines.append(f"On qubits {tuple(comp)}: fidelity {f:.6f}\n  your circuit gives (in the order of these qubits): "
                         f"{ket_str(psi, len(comp))}\n  the target is:                                    {ket_str(tgt, len(comp))}")
        else:
            lines.append(f"On qubits {tuple(comp)} (entangled together by your circuit): fidelity {f:.6f}")
    if len(bad) > 4:
        lines.append(f"... and {len(bad) - 4} more wrong groups.")
    ok = len(parts) - len(bad)
    if ok and len(parts) > 1:
        lines.append(f"{ok} other group(s) are already correct.")
    return "\n".join(lines)


def run_task(args, name, pc, target, outdir):
    from psf_pennylane_gpu_prototype import tape_to_qiskit
    n, text, groups = TASKS[name]
    d = os.path.join(outdir, name)
    os.makedirs(d, exist_ok=True)
    log = open(os.path.join(d, "rounds.jsonl"), "w", encoding="utf-8")
    task_msg = {"role": "user", "content": f"Task: {text}\nUse n_qubits = {n}."}
    messages = [{"role": "system", "content": SYSTEM}, task_msg]
    best, rounds, stale = None, [], 0
    temp_now = args.temperature
    for r in range(1, args.rounds + 1):
        rec = dict(task=name, run=args.run, round=r, temperature=temp_now)
        reply = ""
        try:
            t_sim = 0.0
            if args.mock_llm:
                reply = json.dumps(MOCK[name][min(r, len(MOCK[name])) - 1], default=float)
                reasoning, llm_s, usage, fin = "", 0.0, {}, "stop"
            else:
                seed = 1000 * args.run + r
                try:
                    reply, reasoning, llm_s, usage, fin, tlog = converse(args, messages, temp_now, seed, args.max_tokens, n)
                    reasoning = "r" * reasoning  # converse returns a length
                    if tlog:
                        rec["tool_calls"] = tlog
                except ContextError as e:  # context exceeded: keep only the task and the last feedback, retry once
                    rec["http400"] = str(e)[:300]
                    last_fb = messages[-1]["content"] if len(messages) > 2 else ""
                    messages = [messages[0], {"role": "user", "content": task_msg["content"] +
                                              ("\n\nYour previous attempt: " + last_fb if last_fb else "")}]
                    reply, reasoning, llm_s, usage, fin = ask_model(args, messages, temp_now, seed,
                                                                    max(1000, args.max_tokens // 2))
            rec.update(llm_s=round(llm_s, 3), usage=usage, finish_reason=fin, reply=reply[:6000],
                       reasoning_chars=len(reasoning))
            try:
                spec = extract_json(reply)
            except (ValueError, json.JSONDecodeError):
                if args.mock_llm or args.salvage_tokens <= 0 or fin != "length":
                    raise
                # (4) salvage: the reasoning ran out of budget; ask once, briefly, for the best circuit now
                sal_msgs = messages + [{"role": "assistant", "content": reply or "(no answer yet)"},
                                       {"role": "user", "content": SALVAGE}]
                reply2, reas2, s_s, s_usage, s_fin = ask_model(args, sal_msgs, temp_now, seed + 500,
                                                               args.salvage_tokens, effort=args.salvage_effort)
                rec["salvage"] = dict(llm_s=round(s_s, 3), usage=s_usage, finish_reason=s_fin,
                                      reply=reply2[:4000], reasoning_chars=len(reas2))
                rec["llm_s"] = round(llm_s + s_s, 3)
                reply = reply2
                spec = extract_json(reply)
            tape, logical_2q, notes, gnames = to_tape(spec, n)
            t0 = time.perf_counter()
            f_log, parts = logical_check(args.sim, tape, n, groups)
            t_sim += time.perf_counter() - t0
            qc, _ = tape_to_qiskit(tape, wire_order=list(range(n)))
            routed, c_s = compile_psf(pc, qc, target)
            q3, q3_s = compile_q3(qc, target)
            t0 = time.perf_counter()
            f_cmp, leak, kmax = compiled_check(args.sim, routed, n, groups)
            t_sim += time.perf_counter() - t0
            rec.update(spec=spec, notes=notes, logical_2q=logical_2q, fidelity_logical=f_log,
                       compile_s=round(c_s, 4), routed_2q=twoq(routed), routed_swaps=swaps(routed), depth=routed.depth(),
                       q3_s=round(q3_s, 4), q3_2q=twoq(q3), fidelity_compiled=f_cmp, leak=leak,
                       largest_component=kmax, sim_s=round(t_sim, 4))
            fb = (f"Result of your circuit: fidelity with the target state = {f_log:.6f} (before compiling); after "
                  f"compiling for the device: fidelity = {f_cmp:.6f}, two-qubit gates on the device = {twoq(routed)} "
                  f"(you used {logical_2q}), depth = {routed.depth()}.")
            if notes:
                fb += " Note: " + "; ".join(notes[:3]) + "."
            ok = f_cmp >= args.fidelity
            key = (ok, -twoq(routed) if ok else f_cmp)
            if best is None or key > best[0]:
                best = (key, rec, routed, tape, q3)
                stale = 0
            else:
                stale += 1
            if ok:
                fb += (" The state is correct. If you can, give a circuit with fewer two-qubit gates on the device "
                       "that is still exact; otherwise repeat your circuit.")
            else:
                fb += f" The state is NOT correct yet (need fidelity >= {args.fidelity}).\n" + feedback_wrong(parts, n, args.fidelity)
                if n <= 4:
                    fb += "\nState after each of your gates:\n" + per_gate_states(tape, n, gnames)
                fb += "\nFind the error and give a corrected circuit (try a different approach if the same idea failed before)."
            if args.memory and best is not None and best[1] is not rec:
                b = best[1]
                fb += (f"\nFor reference, your best circuit so far (round {b['round']}, fidelity after compiling "
                       f"{b['fidelity_compiled']:.6f}, {b['routed_2q']} two-qubit gates on the device) was:\n"
                       + json.dumps({"gates": b["spec"]["gates"]}, default=float, separators=(",", ":")))
        except (ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError) as e:
            rec.update(error=f"{type(e).__name__}: {e}")
            fb = f"Your reply could not be used: {e}. Give the JSON object in a ```json block, following the format exactly."
            if args.memory and best is not None:
                b = best[1]
                fb += (f"\nFor reference, your best circuit so far (round {b['round']}, fidelity after compiling "
                       f"{b['fidelity_compiled']:.6f}) was:\n"
                       + json.dumps({"gates": b["spec"]["gates"]}, default=float, separators=(",", ":")))
        except ContextError as e:
            rec.update(error=f"HTTP 400 twice: {str(e)[:300]}")
            fb = "Your reply was too long. Keep the reasoning short and give the JSON object."
        except urllib.error.HTTPError as e:
            rec.update(error=f"HTTP {e.code}")
            fb = "The previous request failed on the server. Please give the circuit again."
            time.sleep(5)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            rec.update(error=f"model not reachable: {e}")
            print(f"  {name} run {args.run} round {r}: model not reachable: {e}", flush=True)
            rounds.append(rec)
            log.write(json.dumps(rec, default=str) + "\n")
            break
        rec["feedback"] = fb
        prev = rounds[-1] if rounds else None
        same_wrong = (prev is not None and "fidelity_compiled" in rec and "fidelity_compiled" in prev
                      and rec["fidelity_compiled"] < args.fidelity
                      and abs(rec["fidelity_compiled"] - prev["fidelity_compiled"]) < 1e-9)
        same_error = prev is not None and "error" in rec and rec.get("error") == prev.get("error")
        if same_wrong or same_error:
            temp_now = min(1.0, temp_now + 0.3)  # stuck on the same wrong answer or error: explore more
        rounds.append(rec)
        log.write(json.dumps(rec, default=str) + "\n")
        log.flush()
        print(f"  {name} run {args.run} round {r}: llm {rec.get('llm_s', '-')} s | "
              + (f"error {rec['error']}" if "error" in rec else
                 f"F_log {rec['fidelity_logical']:.6f} F_cmp {rec['fidelity_compiled']:.6f} 2q {rec['logical_2q']}->"
                 f"{rec['routed_2q']} (L3 {rec['q3_2q']}) compile {rec['compile_s']*1000:.1f} ms (L3 {rec['q3_s']*1000:.1f} ms)"),
              flush=True)
        messages = messages + [{"role": "assistant", "content": reply or "(empty reply)"}, {"role": "user", "content": fb}]
        if len(messages) > 6:  # keep the system prompt, the task and the last two exchanges (context limit)
            messages = messages[:2] + messages[-4:]
        if best and best[0][0] and stale >= 2:
            break  # exact, and two further rounds did not improve on the best
    log.close()

    # StatePreparation baseline, compiled the same two ways (for G2)
    base = baseline_stateprep(n, groups)
    b_psf, b_psf_s = compile_psf(pc, base, target)
    b_q3, b_q3_s = compile_q3(base, target)
    b_f, _, _ = compiled_check(args.sim, b_psf, n, groups)
    result = dict(task=name, run=args.run, model=("MOCK" if args.mock_llm else args.model), n_qubits=n,
                  rounds=len(rounds), solved=bool(best and best[0][0]),
                  errors=sum(1 for x in rounds if "error" in x), http400=sum(1 for x in rounds if "http400" in x),
                  llm_s_total=round(sum(x.get("llm_s", 0) for x in rounds), 2),
                  baseline_psf_2q=twoq(b_psf), baseline_q3_2q=twoq(b_q3), baseline_psf_fidelity=b_f,
                  baseline_psf_s=round(b_psf_s, 4), baseline_q3_s=round(b_q3_s, 4))
    if best:
        _, rec, routed, tape, q3 = best
        from qiskit import qasm3
        f3, _, _ = compiled_check(args.sim, q3, n, groups)
        result.update(best_round=rec["round"], fidelity_compiled=rec["fidelity_compiled"], logical_2q=rec["logical_2q"],
                      routed_2q=rec["routed_2q"], routed_swaps=rec["routed_swaps"], depth=rec["depth"],
                      compile_s=rec["compile_s"], q3_s=rec["q3_s"], q3_2q=rec["q3_2q"], q3_fidelity=f3)
        json.dump(rec["spec"], open(os.path.join(d, "best_circuit.json"), "w"), indent=1, default=float)
        open(os.path.join(d, "best_compiled.qasm"), "w").write(qasm3.dumps(routed))
        open(os.path.join(d, "best_pennylane_ops.txt"), "w").write("\n".join(repr(o) for o in tape.operations))
    json.dump(result, open(os.path.join(d, "result.json"), "w"), indent=1)
    return result


# ---------------------------------------------------------------- G3 re-timing pass

def retime(args, pc, target):
    """Every distinct valid fill27 circuit under --retime, compiled sequentially (nothing else running) by
    PSF-Zero and Qiskit L3, --retime-reps times each; writes retime_fill27.csv."""
    import csv
    from psf_pennylane_gpu_prototype import tape_to_qiskit
    n, _, groups = TASKS["fill27"]
    seen, rows = {}, []
    for dpath, _, files in sorted(os.walk(args.retime)):
        if "rounds.jsonl" in files and os.path.basename(dpath) == "fill27":
            for line in open(os.path.join(dpath, "rounds.jsonl"), encoding="utf-8"):
                rec = json.loads(line)
                if "spec" in rec and "error" not in rec:
                    key = hashlib.sha256(json.dumps(rec["spec"]["gates"], sort_keys=True, default=float).encode()).hexdigest()[:16]
                    seen.setdefault(key, (rec["spec"], []))[1].append(os.path.relpath(dpath, args.retime) + f"#r{rec['round']}")
    print(f"retime: {len(seen)} distinct valid fill27 circuits", flush=True)
    for key, (spec, where) in sorted(seen.items()):
        tape, l2q, _, _ = to_tape(spec, n)
        qc, _ = tape_to_qiskit(tape, wire_order=list(range(n)))
        ts_p, ts_q = [], []
        for _ in range(args.retime_reps):
            routed, s = compile_psf(pc, qc, target)
            ts_p.append(s)
            q3, s3 = compile_q3(qc, target)
            ts_q.append(s3)
        f_cmp, _, _ = compiled_check(args.sim, routed, n, groups)
        row = dict(circuit=key, occurrences=len(where), where=";".join(where), logical_2q=l2q,
                   psf_2q=twoq(routed), psf_swaps=swaps(routed), q3_2q=twoq(q3), q3_swaps=swaps(q3),
                   psf_s_median=float(np.median(ts_p)), q3_s_median=float(np.median(ts_q)),
                   psf_s_all=" ".join(f"{x:.4f}" for x in ts_p), q3_s_all=" ".join(f"{x:.4f}" for x in ts_q),
                   fidelity_compiled=f_cmp)
        rows.append(row)
        print(f"  {key}: psf {row['psf_s_median']*1000:.1f} ms {row['psf_2q']} 2q | L3 {row['q3_s_median']*1000:.1f} ms "
              f"{row['q3_2q']} 2q | F {f_cmp:.6f}", flush=True)
    out = os.path.join(args.retime, "retime_fill27.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["circuit"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--run", type=int, default=1, help="run number; request seed = 1000*run + round")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--fidelity", type=float, default=0.9999)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--reasoning-effort", default=None, help="for reasoning models (gpt-oss): low/medium/high")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--fake", default="FakeAuckland")
    ap.add_argument("--sim", default="lightning.qubit")
    ap.add_argument("--layout-dir", default=None)
    ap.add_argument("--repo", default=os.path.expanduser("~/psf-zero"))
    ap.add_argument("--out", default="e2e_out")
    ap.add_argument("--mock-llm", action="store_true")
    ap.add_argument("--salvage-tokens", type=int, default=8000, help="0 disables the salvage request")
    ap.add_argument("--salvage-effort", default="medium")
    ap.add_argument("--tool-sim", action="store_true", help="offer the simulate() tool to the model")
    ap.add_argument("--max-tool-calls", type=int, default=8)
    ap.add_argument("--no-memory", dest="memory", action="store_false")
    ap.add_argument("--retime", default=None, help="re-time every valid fill27 circuit found under this folder")
    ap.add_argument("--retime-reps", type=int, default=3)
    args = ap.parse_args()
    if args.layout_dir:
        sys.path.insert(0, args.layout_dir)
    for p in (args.repo, os.path.join(args.repo, "benchmarks")):
        if p not in sys.path:
            sys.path.insert(1 if args.layout_dir else 0, p)
    import warnings
    import psf_compile as pc
    import psf_smart_layout as sl
    from qiskit_ibm_runtime import fake_provider
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        target = getattr(fake_provider, args.fake)().target
    head = (f"{SCRIPT_VERSION} | model {'MOCK' if args.mock_llm else args.model} | device {args.fake} | sim {args.sim} | "
            f"psf_compile {pc.VERSION} | CORE_VERSION {getattr(pc, 'CORE_VERSION', None)} | layout {sl.LAYOUT_VERSION}")
    print(head, flush=True)
    if args.retime:
        retime(args, pc, target)
        return
    os.makedirs(args.out, exist_ok=True)
    open(os.path.join(args.out, f"header_run{args.run}_{args.tasks.replace(',', '-')}.txt"), "w").write(
        head + "\n" + json.dumps(vars(args)) + "\n")
    for name in args.tasks.split(","):
        print(f"== {name} (run {args.run}): {TASKS[name][1]}", flush=True)
        res = run_task(args, name, pc, target, args.out)
        print(f"  -> {json.dumps(res)}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
