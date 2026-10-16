"""MCP server: gives an agent (e.g. Claude Code) a brain-style long-term memory over stdio."""
from mcp.server.fastmcp import FastMCP

from .brain import Brain

mcp = FastMCP("neuromem")
_brain = None


def brain():
    global _brain
    if _brain is None:
        _brain = Brain()
    return _brain


@mcp.tool()
def perceive(text: str, importance: float | None = None, layer: str | None = None) -> dict:
    """Store something worth remembering: an event, a fact, a decision, a preference or a how-to.
    layer: episodic | semantic | procedural (auto-detected if omitted). importance 0-1 (auto if omitted).
    Newer facts automatically supersede contradicting older ones."""
    return brain().perceive(text, importance=importance, layer=layer, source="mcp")


@mcp.tool()
def recall(cue: str, k: int = 5, include_history: bool = False) -> list[dict]:
    """Recall memories by meaning, keywords or a fragment. Returns direct matches plus related memories
    (why = association | entity). Recalling strengthens what is found."""
    return brain().recall(cue, k=k, include_history=include_history)


@mcp.tool()
def about(entity: str) -> list[dict]:
    """Everything remembered about a person, project, tool or place."""
    return brain().about(entity)


@mcp.tool()
def belief_history(memory_id: int) -> list[dict]:
    """How a belief changed over time (older superseded versions first)."""
    return brain().history(memory_id)


@mcp.tool()
def experience(task: str, action: str, reward: float, note: str | None = None) -> dict:
    """Record an attempt so the memory learns from it: doing `action` for `task` earned `reward`
    (-1 = failed, 0 = no effect, 1 = worked). Call this after every attempt whose outcome you know."""
    return brain().experience(task, action, reward, note=note)


@mcp.tool()
def advise(task: str) -> dict:
    """Before acting: what has worked ("try") and failed ("avoid") for tasks like this one."""
    return brain().advise(task)


@mcp.tool()
def prepare(task: str, max_chars: int = 1200) -> str:
    """A compact briefing for a task: what worked, what failed, and relevant facts and how-tos."""
    return brain().prepare(task, max_chars=max_chars)


@mcp.tool()
def forget(memory_id: int) -> bool:
    """Delete a memory."""
    return brain().forget(memory_id)


@mcp.tool()
def sleep() -> dict:
    """Consolidate weak episodes into facts, forget what exceeds the budget, prune weak associations."""
    return brain().sleep()


@mcp.tool()
def memory_stats() -> dict:
    """Counts per layer, entities, links, storage use."""
    return brain().stats()


def main():
    mcp.run()


if __name__ == "__main__":
    main()
