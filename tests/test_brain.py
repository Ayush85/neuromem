import json
import subprocess
import sys

import pytest

from neuromem import Brain, HeuristicMind
from neuromem.embed import make_embedder

SEMANTIC = make_embedder("auto") is not None
needs_semantic = pytest.mark.skipif(not SEMANTIC, reason="fastembed not installed")


@pytest.fixture
def brain():
    return Brain(":memory:")


# ---------------------------------------------------------------- attention
def test_trivial_input_is_ignored(brain):
    assert brain.perceive("ok", now=0)["action"] == "ignored"
    assert brain.stats()["memories"] == 0


def test_duplicate_reinforces_instead_of_storing(brain):
    a = brain.perceive("Alice leads the billing migration project", now=0)
    b = brain.perceive("Alice leads the billing migration project", now=1)
    assert a["action"] == "stored" and b["action"] == "reinforced" and b["id"] == a["id"]
    assert brain.stats()["memories"] == 1


def test_emphasis_raises_importance(brain):
    plain = brain.perceive("The office printer is on the second floor", now=0)
    urgent = brain.perceive("Important: the contract deadline is March 3, never miss it!", now=0)
    assert urgent["importance"] > plain["importance"]


# ---------------------------------------------------------------- layers and entities
def test_layers_detected(brain):
    assert brain.perceive("To deploy the api, run make release then tag the commit", now=0)["layer"] == "procedural"
    assert brain.perceive("Yesterday Bob fixed the login bug in production", now=0)["layer"] == "episodic"
    assert brain.perceive("Water boils at one hundred degrees", layer="semantic", now=0)["layer"] == "semantic"


def test_entities_extracted_and_queryable(brain):
    r = brain.perceive("Priya reviewed the Kafka consumer changes for Project Atlas", now=0)
    assert {"priya", "kafka", "project atlas"} <= set(r["entities"])
    brain.perceive("Priya prefers async code reviews over meetings", now=1)
    assert len(brain.about("priya", now=2)) == 2


# ---------------------------------------------------------------- recall
@needs_semantic
def test_recall_by_meaning(brain):
    brain.perceive("Postgres backups run every night and go to the s3 bucket", now=0)
    brain.perceive("The frontend is built with React and bundled by Vite", now=0)
    assert "backups" in brain.recall("how often do we save copies of the database", now=1)[0]["text"]
    assert brain.recall("recipe for chocolate cake", now=1) == []


def test_recall_from_fragment(brain):
    brain.perceive("Rate limiting is enforced by the nginx gateway at one hundred requests per minute", now=0)
    brain.perceive("The frontend is built with React and bundled by Vite", now=0)
    assert "nginx" in brain.recall("nginx rate limiting", now=1)[0]["text"]


def test_entity_in_cue_pulls_related_memories(brain):
    brain.perceive("Marco owns the payments service", now=0)
    brain.perceive("Marco is on vacation until the 20th", now=0)
    brain.perceive("The search index is rebuilt weekly", now=0)
    hits = brain.recall("who owns payments", k=1, associated=2, now=1)
    texts = [h["text"] for h in hits]
    assert texts[0].startswith("Marco owns") and any("vacation" in t for t in texts)
    assert not any("search index" in t for t in texts)


def test_recall_strengthens_memory(brain):
    m = brain.perceive("The staging database password rotates every quarter", now=0)["id"]
    s0 = brain._cache["S"][brain._pos(m)]
    brain.recall("staging database password", now=5)
    assert brain._cache["S"][brain._pos(m)] > s0


@needs_semantic
def test_working_memory_breaks_ambiguity(brain):
    brain.perceive("Mercury is the closest planet to the sun", now=0)
    brain.perceive("Mercury is a toxic liquid metal used in old thermometers", now=0)
    brain.focus("astronomy and the solar system")
    assert "planet" in brain.recall("mercury", k=1, now=1, strengthen=False)[0]["text"]
    brain.working.clear()
    brain.focus("chemistry and hazardous materials")
    assert "metal" in brain.recall("mercury", k=1, now=1, strengthen=False)[0]["text"]


@needs_semantic
def test_working_memory_does_not_override_clear_match(brain):
    brain.perceive("The python release pipeline publishes wheels to PyPI", now=0)
    brain.perceive("The office plants are watered on Fridays", now=0)
    brain.focus("gardening and indoor plants")
    assert "PyPI" in brain.recall("python release wheels", k=1, now=1)[0]["text"]


# ---------------------------------------------------------------- belief updating
def test_new_fact_supersedes_contradicting_old_one(brain):
    old = brain.perceive("The API rate limit is 100 requests per minute", now=0)["id"]
    new = brain.perceive("The API rate limit is 250 requests per minute", now=5)
    assert old in new["superseded"]
    top = brain.recall("API rate limit", k=1, now=6)[0]
    assert "250" in top["text"]
    assert [h["id"] for h in brain.history(new["id"])] == [old, new["id"]]
    assert any(h["id"] == old for h in brain.recall("API rate limit", include_history=True, now=6))


def test_negation_supersedes(brain):
    old = brain.perceive("Dana is on the oncall rotation this month", now=0)["id"]
    r = brain.perceive("Dana is not on the oncall rotation this month", now=1)
    assert old in r["superseded"]


def test_unrelated_facts_do_not_supersede(brain):
    brain.perceive("The API rate limit is 100 requests per minute", now=0)
    r = brain.perceive("The cafeteria opens at 8 in the morning", now=1)
    assert r["superseded"] == []


# ---------------------------------------------------------------- sleep
@needs_semantic
def test_sleep_consolidates_repeated_episodes_into_fact(brain):
    for d in range(3):
        brain.perceive(f"The Friday deploy failed again because of the flaky integration tests (week {d + 1})", now=d * 7)
    brain.perceive("Lunch on Friday was pasta", now=0)
    out = brain.sleep(now=60)
    assert len(out["facts_created"]) == 1
    fact = brain.recall("why do friday deploys fail", k=1, now=61)[0]
    assert fact["layer"] == "semantic" and fact["sources"] == 3
    assert brain.stats()["consolidated"] == 3


def test_sleep_uses_mind_summary():
    class FakeMind(HeuristicMind):
        def summarize(self, texts):
            return f"Deploys to production keep succeeding ({len(texts)} observations)"
    b = Brain(":memory:", mind=FakeMind())
    for i in range(3):
        b.perceive(f"Deploy {i} to production finished without errors", now=0)
    b.sleep(now=100)
    assert b.recall("production deploys succeeding", k=1, now=101)[0]["text"].startswith("Deploys to production keep")


def test_forgetting_respects_budget_importance_and_layer():
    b = Brain(":memory:", embedder="none", budget_bytes=4000)
    keep = b.perceive("Critical: the root password is stored in the vault, never in git", now=0)["id"]
    how = b.perceive("To rotate keys, run make rotate in the infra repo", now=0)["id"]
    for i in range(40):
        b.perceive(f"Random chatter number {i} about lunch weather and traffic {i * 13}", now=i)
    assert b.stats()["bytes"] <= 4000
    ids = {r[0] for r in b.db.execute("SELECT id FROM memories")}
    assert keep in ids and how in ids


def test_persistence(tmp_path):
    p = str(tmp_path / "b.db")
    a = Brain(p)
    mid = a.perceive("Kenji maintains the Terraform modules", now=0)["id"]
    a.close()
    b = Brain(p)
    assert b.recall("Terraform modules maintainer", now=1)[0]["id"] == mid
    assert b.about("kenji", now=1)[0]["id"] == mid


def test_cli(tmp_path):
    db = str(tmp_path / "cli.db")
    run = lambda *a: subprocess.run([sys.executable, "-m", "neuromem.cli", "--db", db, "--json", *a],
                                    capture_output=True, text=True)
    assert json.loads(run("perceive", "Sara owns the mobile app release").stdout)["action"] == "stored"
    hits = json.loads(run("recall", "mobile app release owner").stdout)
    assert "Sara" in hits[0]["text"]
    assert json.loads(run("stats").stdout)["memories"] == 1
