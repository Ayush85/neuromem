"""ClaudeMind with a fake client: request shape, parsing, and fallback to heuristics on failure."""
import json
from types import SimpleNamespace

import pytest

anthropic = pytest.importorskip("anthropic")
from neuromem import Brain, ClaudeMind  # noqa: E402


class FakeMessages:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def create(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        stop, payload = r
        return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=json.dumps(payload))])


def mind_with(*replies):
    msgs = FakeMessages(replies)
    return ClaudeMind(client=SimpleNamespace(beta=SimpleNamespace(messages=msgs))), msgs


def test_extract_uses_structured_output_and_fallbacks():
    m, msgs = mind_with(("end_turn", {"entities": [{"name": "Priya", "kind": "person"}], "when": "",
                                      "layer": "semantic", "importance": 0.7}))
    out = m.extract("Priya prefers async reviews")
    assert out["layer"] == "semantic" and out["entities"][0]["name"] == "Priya"
    call = msgs.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in call["betas"]


def test_refusal_or_error_falls_back_to_heuristics():
    m, _ = mind_with(("refusal", {}), RuntimeError("network down"))
    assert m.extract("Yesterday Bob fixed the login bug")["layer"] == "episodic"
    assert m.contradicts("The limit is 100 per minute", "The limit is 200 per minute") is True


def test_brain_uses_claude_summary_in_sleep():
    m, _ = mind_with(*[("end_turn", {"entities": [], "when": "", "layer": "episodic", "importance": 0.1})] * 3,
                     ("end_turn", {"fact": "Friday deploys tend to fail because of flaky integration tests"}),
                     ("end_turn", {"entities": [], "when": "", "layer": "semantic", "importance": 0.3}))
    b = Brain(":memory:", mind=m, embedder="none")
    for d in range(3):
        b.perceive(f"The Friday deploy failed because of flaky integration tests, week {d + 1}", now=d * 7)
    out = b.sleep(now=90)
    assert len(out["facts_created"]) == 1
    assert b.recall("why do friday deploys fail", k=1, now=91)[0]["text"].startswith("Friday deploys tend to fail")
