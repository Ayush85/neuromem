"""Real-model test: does memory make a local LLM better at repeated coding work on an unfamiliar SDK?

For each task the LLM gets the API reference (+ notes from memory, if any), writes code, and the code is run
against tests in a subprocess. On failure it gets the error and one retry. After each task it writes a one-line
general lesson, which the memory conditions store. Halfway through, acmekit moves to version 2 (init changes).

Conditions:  none | retrieval (top-3 lessons by embedding similarity) | neuromem (Brain.prepare briefing)
Usage: python realtest/harness.py [model] [n_tasks] [conditions...]
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import importlib                             # noqa: E402
import random                                # noqa: E402

SDK = os.environ.get("REALTEST_SDK", "acmekit")      # acmekit | paykit
API_DOC = importlib.import_module(SDK).API_DOC
_tasks = importlib.import_module("tasks" if SDK == "acmekit" else f"tasks_{SDK}")
PREAMBLE, TASKS = _tasks.PREAMBLE, _tasks.TASKS


def sequence(seed=0, n=24):
    """Passes over the SDK's task kinds, each in a fresh shuffled order."""
    r, out = random.Random(seed), []
    while len(out) < n:
        p = TASKS[:]
        r.shuffle(p)
        out += p
    return out[:n]

from neuromem import Brain                   # noqa: E402
from neuromem.embed import make_embedder     # noqa: E402

OLLAMA = "http://localhost:11434/api/chat"


def llm(model, messages, max_tokens=500):
    body = json.dumps({"model": model, "messages": messages, "stream": False,
                       "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 4096}}).encode()
    req = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["message"]["content"]


def extract_code(text):
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return (max(m, key=len) if m else text).strip()


def run_code(code, check, version):
    src = PREAMBLE + code + "\n\n" + check + "\nprint('PASS')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
    try:
        p = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout=20,
                           env={**os.environ, "PYTHONPATH": HERE, f"{SDK.upper()}_VERSION": str(version)})
        if "PASS" in p.stdout:
            return True, ""
        err = (p.stderr.strip().splitlines() or ["no output"])[-1]
        return False, err[:400]
    except subprocess.TimeoutExpired:
        return False, "timed out"
    finally:
        os.unlink(f.name)


SYSTEM = {
    "acmekit": ("You write Python code that uses our internal SDK `acmekit` (already imported as `acmekit`; datetime, "
                "timedelta and timezone are imported too). Write only the requested function plus any helpers, in "
                "one ```python block, with no explanation."),
    "paykit": ("You write Python code that uses our internal SDK `paykit` (already imported as `paykit`; Decimal is "
               "imported too). Write only the requested function plus any helpers, in one ```python block, with "
               "no explanation."),
}[SDK]


def prompt(task, version, notes):
    p = API_DOC[version] + "\n\n"
    if notes:
        p += (f"IMPORTANT - rules learned from earlier failures with {SDK}. Follow every one of them:\n"
              + notes + "\n\n")
    return p + "Task: " + task


def fmt_notes(rules, examples):
    out = ""
    if rules:
        out += rules + "\n"
    for t, code in examples:
        out += f"\nWorking code from a similar earlier task ({t[:80]}):\n```python\n{code}\n```\n"
    return out.strip()


class RetrievalMemory:
    """Typical vector memory: top-3 lessons and top-2 examples by similarity, no feedback."""
    def __init__(self, embed):
        self.embed, self.items, self.ex = embed, [], []

    def notes(self, task):
        v = self.embed.one(task)
        rules = ""
        if self.items:
            sims = [float(e @ v) for e, _ in self.items]
            rules = "\n".join("- " + self.items[i][1] for i in np.argsort(sims)[::-1][:3])
        exs = []
        if self.ex:
            sims = [float(e @ v) for e, _, _ in self.ex]
            exs = [(self.ex[i][1], self.ex[i][2]) for i in np.argsort(sims)[::-1][:2]]
        self.shown = []
        return fmt_notes(rules, exs)

    def store(self, task, lesson, ok):
        self.items.append((self.embed.one(lesson), lesson))

    def store_example(self, task, code):
        self.ex.append((self.embed.one(task), task, code))

    def outcome(self, ok):
        pass


class NeuroMemory:
    """neuromem: prepare() briefing (relevant + standing rules), examples ranked by strength, and feedback:
    examples that led to success get stronger, ones that led to failure weaker, then retired."""
    def __init__(self, profile):
        self.brain = Brain(":memory:", profile=profile)
        self.clock = 0.0
        self.shown = []

    def notes(self, task):
        rules = self.brain.prepare(task, max_chars=900, now=self.clock)
        exs = self.brain.examples(task, k=2, now=self.clock)
        self.shown = [e[0] for e in exs]
        return fmt_notes(rules, [(t, code) for _, t, code in exs])

    def store(self, task, lesson, ok):
        self.clock += 0.05
        self.brain.perceive(lesson, layer="procedural", importance=0.6, source="lesson", now=self.clock)

    def store_example(self, task, code):
        self.clock += 0.05
        self.brain.remember_example(task, code, now=self.clock)

    def outcome(self, ok):
        if self.shown:
            self.brain.feedback(self.shown, 1 if ok else -1, now=self.clock)


def run(model, n, condition, profile="full", log=None, seed=0):
    mem = {"none": None, "retrieval": RetrievalMemory(make_embedder(profile)),
           "neuromem": NeuroMemory(profile)}[condition]
    rows = []
    for i, (name, task, check) in enumerate(sequence(seed=seed, n=n)):
        version = 1 if i < n // 2 else 2
        notes = mem.notes(task) if mem else ""
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(task, version, notes)}]
        t0 = time.time()
        reply = llm(model, msgs)
        code1 = extract_code(reply)
        ok1, err = run_code(code1, check, version)
        ok2, err2 = ok1, err
        code_ok = code1 if ok1 else None
        if mem is not None:
            mem.outcome(ok1)                                     # feedback on the memories that were shown
        if not ok1:
            msgs += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": f"Running it failed with:\n{err}\nFix the code."}]
            reply = llm(model, msgs)
            code2 = extract_code(reply)
            ok2, err2 = run_code(code2, check, version)
            code_ok = code2 if ok2 else None
        lesson = ""
        if mem is not None and not ok1:
            msgs += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": f"Write the one rule about {SDK} that would have prevented the first "
                                                 "error, as a concrete instruction including the exact code to write "
                                                 f"(not specific to this task). One sentence starting with '{SDK}:'."}]
            lesson = llm(model, msgs, max_tokens=80).strip().splitlines()[0][:300]
            mem.store(task, lesson, ok2)
        if mem is not None and code_ok:
            mem.store_example(task, code_ok)
        rows.append({"seed": seed, "i": i, "task": name, "version": version, "first_try": ok1, "after_retry": ok2,
                     "error": err[:160], "lesson": lesson, "notes": notes, "code": code1, "secs": round(time.time() - t0, 1)})
        line = (f"[{condition}] {i:2} v{version} {name:<18} first={'PASS' if ok1 else 'fail'} "
                f"retry={'PASS' if ok2 else 'fail'}  {err[:70]}")
        print(line, flush=True)
        if log:
            log.write(json.dumps({"condition": condition, **rows[-1]}) + "\n")
            log.flush()
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("conditions", nargs="*", default=["none", "retrieval", "neuromem"])
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("-n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    model, n = a.model, a.n
    out = os.path.join(HERE, f"results_{model.replace(':', '_')}" + ("" if SDK == "acmekit" else f"_{SDK}") + ".jsonl")
    with open(out, "a") as log:
        summary = {}
        for c in a.conditions:
            if c.startswith("reflex"):
                from reflex import run_reflex
                rows = run_reflex(model, n, sequence(seed=a.seed, n=n), log=log, hints=(c == "reflex"), name=c)
            else:
                rows = run(model, n, c, log=log, seed=a.seed)
            h = n // 2
            f = lambda rs: f"{sum(r['first_try'] for r in rs)}/{len(rs)}"
            summary[c] = (f(rows[:h]), f(rows[h:]), f"{sum(r['after_retry'] for r in rows)}/{n}")
        print(f"\nmodel {model} seed {a.seed}: first-try success  v1 half | v2 half | after retry (all)")
        for c, (x, y, d) in summary.items():
            print(f"  {c:<15} {x:>6} | {y:>6} | {d:>6}")
