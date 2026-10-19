"""MCP server: gives an agent (e.g. Claude Code) a brain-style long-term memory over stdio.

Every tool call is appended to a local usage log ($NEUROMEM_USAGE or ~/.neuromem/usage.jsonl) so real use can
be measured later with `neuromem usage-report`. Nothing leaves the machine.
"""
import functools
import json
import os
import time

from mcp.server.fastmcp import FastMCP

from .brain import Brain

INSTRUCTIONS = """neuromem is your long-term memory across sessions and projects. Use it like this:
- At the start of a non-trivial task, call prepare(task) for what worked, what failed and relevant facts.
- When you learn a lasting fact, decision or user preference, call perceive(text).
- When you hit an error, call fixes_for(error) before debugging from scratch.
- After you fix an error and verify the fix, call remember_fix(error, fix_lines).
- After code passes its tests, call remember_example(task, code) for reusable patterns.
- When a recalled memory turned out right or wrong, call feedback(ids, 1 or -1).
Do not store secrets or credentials."""

mcp = FastMCP("neuromem", instructions=INSTRUCTIONS)
_brain = None
PROJECT = os.path.basename(os.getcwd())


def usage_path():
    return os.environ.get("NEUROMEM_USAGE") or os.path.join(os.path.expanduser("~"), ".neuromem", "usage.jsonl")


def logged(fn):
    """Record tool name, project, argument sizes and result size (not the content) for each call."""
    @functools.wraps(fn)
    def wrap(*a, **kw):
        t0 = time.time()
        out = fn(*a, **kw)
        try:
            os.makedirs(os.path.dirname(usage_path()), exist_ok=True)
            n = len(out) if isinstance(out, (list, str)) else (len(out.get("try", [])) if isinstance(out, dict)
                                                                 and "try" in out else 1)
            with open(usage_path(), "a") as f:
                f.write(json.dumps({"t": round(t0, 1), "tool": fn.__name__, "project": PROJECT,
                                    "args": {k: (v if isinstance(v, (int, float, bool)) or v is None else len(str(v)))
                                             for k, v in kw.items()},
                                    "results": n, "ms": round((time.time() - t0) * 1000)}) + "\n")
        except Exception:
            pass
        return out
    return wrap


def brain():
    global _brain
    if _brain is None:
        _brain = Brain()
    return _brain


@mcp.tool()
@logged
def perceive(text: str, importance: float | None = None, layer: str | None = None) -> dict:
    """Store something worth remembering: an event, a fact, a decision, a preference or a how-to.
    layer: episodic | semantic | procedural (auto-detected if omitted). importance 0-1 (auto if omitted).
    Newer facts automatically supersede contradicting older ones."""
    return brain().perceive(text, importance=importance, layer=layer, source=f"mcp:{PROJECT}")


@mcp.tool()
@logged
def recall(cue: str, k: int = 5, include_history: bool = False) -> list[dict]:
    """Recall memories by meaning, keywords or a fragment. Returns direct matches plus related memories
    (why = association | entity). Recalling strengthens what is found."""
    return brain().recall(cue, k=k, include_history=include_history)


@mcp.tool()
@logged
def about(entity: str) -> list[dict]:
    """Everything remembered about a person, project, tool or place."""
    return brain().about(entity)


@mcp.tool()
@logged
def belief_history(memory_id: int) -> list[dict]:
    """How a belief changed over time (older superseded versions first)."""
    return brain().history(memory_id)


@mcp.tool()
@logged
def experience(task: str, action: str, reward: float, note: str | None = None) -> dict:
    """Record an attempt so the memory learns from it: doing `action` for `task` earned `reward`
    (-1 = failed, 0 = no effect, 1 = worked). Call this after every attempt whose outcome you know."""
    return brain().experience(task, action, reward, note=note)


@mcp.tool()
@logged
def advise(task: str) -> dict:
    """Before acting: what has worked ("try") and failed ("avoid") for tasks like this one."""
    return brain().advise(task)


@mcp.tool()
@logged
def prepare(task: str, max_chars: int = 1200) -> str:
    """A compact briefing for a task: what worked, what failed, and relevant facts and how-tos."""
    return brain().prepare(task, max_chars=max_chars)


@mcp.tool()
@logged
def forget(memory_id: int) -> bool:
    """Delete a memory."""
    return brain().forget(memory_id)


@mcp.tool()
@logged
def sleep() -> dict:
    """Consolidate weak episodes into facts, forget what exceeds the budget, prune weak associations."""
    return brain().sleep()


@mcp.tool()
@logged
def memory_stats() -> dict:
    """Counts per layer, entities, links, storage use."""
    return brain().stats()


@mcp.tool()
@logged
def remember_fix(error: str, fix_lines: list[str], verified: bool = True) -> dict:
    """After fixing an error: store the error message and the lines that fixed it (verified = the code then
    worked). Fixes that keep working become habits; fixes that stop working are retired."""
    return brain().remember_fix(error, fix_lines, verified=verified) or {"action": "ignored"}


@mcp.tool()
@logged
def fixes_for(error: str) -> list[dict]:
    """When you hit an error: how this kind of error was fixed before (trusted fixes only)."""
    return [{"id": i, "lines": l} for i, l in brain().fixes_for(error, k=3, trusted_only=True)]


@mcp.tool()
@logged
def remember_example(task: str, example: str) -> dict:
    """Store a worked example (e.g. code that passed its tests) for a task, to reuse on similar tasks."""
    return brain().remember_example(task, example)


@mcp.tool()
@logged
def examples(task: str, k: int = 2) -> list[dict]:
    """Worked examples from similar past tasks, best first."""
    return [{"id": i, "task": t, "example": e} for i, t, e in brain().examples(task, k=k)]


@mcp.tool()
@logged
def feedback(memory_ids: list[int], reward: float) -> dict:
    """After using memories: did they help (reward 1) or mislead (-1)? Misleading memories are weakened and
    retired, so the memory unlearns what stopped being true."""
    brain().feedback(memory_ids, reward)
    return {"ok": True, "updated": len(memory_ids)}


def main():
    mcp.run()


if __name__ == "__main__":
    main()
