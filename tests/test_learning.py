import numpy as np
import pytest

from neuromem import Brain
from neuromem.brain import dequantize, quantize
from neuromem.embed import make_embedder

needs_semantic = pytest.mark.skipif(make_embedder("auto") is None, reason="no embedder installed")


def test_quantized_embeddings_keep_cosine():
    r = np.random.default_rng(0)
    for dim in (64, 256, 384):
        a, b = r.normal(size=dim), r.normal(size=dim)
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        assert abs(float(dequantize(quantize(a)) @ dequantize(quantize(b))) - float(a @ b)) < 0.02
        assert len(quantize(a)) == dim + 2                      # 4x smaller than float32


@needs_semantic
def test_learns_what_works_and_what_to_avoid():
    b = Brain(":memory:")
    b.experience("the billing pods are OOMKilled", "restart the service", -1, now=0)
    b.experience("the billing pods are OOMKilled", "increase the memory limit", 1, now=0.1)
    b.experience("search crashed with an out of memory error", "increase the memory limit", 1, now=0.2)
    adv = b.advise("the checkout process ran out of memory", now=1)          # new wording, same kind of problem
    assert adv["try"][0]["action"] == "increase the memory limit"
    adv = b.advise("the checkout pods are OOMKilled", now=1)                 # same situation as the failure
    assert any(x["action"] == "restart the service" for x in adv["avoid"])


@needs_semantic
def test_unrelated_task_gets_no_advice():
    b = Brain(":memory:")
    b.experience("the billing pods are OOMKilled", "increase the memory limit", 1, now=0)
    assert b.advise("plan the team offsite in spring", now=1)["try"] == []


@needs_semantic
def test_adapts_when_the_right_action_changes():
    b = Brain(":memory:")
    for i in range(6):
        b.experience("payments returns 401 unauthorized", "rotate the credentials", 1, now=i * 0.1)
    for i in range(4):                                          # the world changed: rotating stops working
        b.experience("payments returns 401 unauthorized", "rotate the credentials", -1, now=1 + i * 0.1)
        b.experience("payments returns 401 unauthorized", "roll back the last release", 1, now=1.05 + i * 0.1)
    adv = b.advise("payments returns 401 unauthorized", now=2)
    assert adv["try"][0]["action"] == "roll back the last release"


@needs_semantic
def test_trials_compress_into_rules():
    b = Brain(":memory:")
    for i in range(30):
        b.experience(f"the disk on host {i % 3} is full", "free up disk space", 1, now=i * 0.01)
    st = b.stats()
    assert st["rules"] <= 3 and st["situations"] <= 3
    rule = b.db.execute("SELECT n FROM memories WHERE layer = 'procedural'").fetchone()[0]
    assert rule > 5                                             # exponentially weighted evidence accumulates


@needs_semantic
def test_learning_survives_tight_budget():
    b = Brain(":memory:", budget_bytes=6000)
    for i in range(80):
        b.experience("the TLS certificate for the api has expired", "renew the TLS certificate", 1, now=i * 0.05)
        b.perceive(f"Unrelated chatter {i} about lunch menus and the weather number {i * 7}", now=i * 0.05)
    assert b.stats()["bytes"] <= 6000
    assert b.advise("the api certificate expired", now=5)["try"][0]["action"] == "renew the TLS certificate"


@needs_semantic
def test_prepare_builds_a_briefing():
    b = Brain(":memory:")
    b.perceive("To renew certificates, run certbot renew on the gateway host", now=0)
    b.experience("the gateway TLS certificate expired", "renew the TLS certificate", 1, now=0)
    text = b.prepare("gateway certificate has expired", now=1)
    assert "Worked before: renew the TLS certificate" in text and "certbot" in text


@needs_semantic
def test_switching_embedder_reembeds(tmp_path):
    p = str(tmp_path / "b.db")
    a = Brain(p, profile="small")
    mid = a.perceive("Postgres backups run every night to the s3 bucket", now=0)["id"]
    a.close()
    b = Brain(p, profile="tiny")
    assert b.recall("database backup schedule", now=1)[0]["id"] == mid


def test_index_ram_is_small():
    b = Brain(":memory:", profile="none")
    for i in range(500):
        b.perceive(f"Note {i}: service {i % 17} handled {i * 3} requests during window {i % 11}", now=i * 0.01)
    assert b.stats()["index_ram_bytes"] < 200 * 500           # well under 200 bytes per memory without embeddings


@needs_semantic
def test_examples_feedback_retires_misleading_example():
    b = Brain(":memory:")
    old = b.remember_example("sum the order totals for a user", "acmekit.init(app='x'); ...v1 code...", now=0)["id"]
    assert b.examples("total of a user's orders", now=1)[0][0] == old
    b.feedback([old], -1, now=2)
    new = b.remember_example("sum the order totals for a user (v2)", "acmekit.init(project='p', token=t); ...", now=2)["id"]
    b.feedback([old], -1, now=3)                                # misled twice, never helped: retired
    ids = [e[0] for e in b.examples("total of a user's orders", now=4)]
    assert old not in ids and ids[0] == new


@needs_semantic
def test_fixes_become_habits_and_stale_habits_are_unlearned_quickly():
    b = Brain(":memory:")
    err = "RuntimeError: sdk is not initialized: call sdk.init(...) once"
    b.remember_fix(err, ["sdk.init(app='one')"], now=0)
    assert b.habits() == []                                     # one success is not a habit yet
    for i in range(8):                                          # same fix, different literal values
        b.remember_fix(err, [f"sdk.init(app='app{i}')"], now=0.1 * (i + 1))
    (hid, lines), = b.habits()
    assert lines == ["sdk.init(app='app7')"]                    # the latest concrete version is kept
    b.feedback([hid], -1, now=2)
    b.feedback([hid], -1, now=2.1)                              # the world changed: two failures in a row
    assert b.habits() == []


@needs_semantic
def test_tentative_fix_is_hidden_until_seen_again():
    b = Brain(":memory:")
    err = "AssertionError: expected the string free, got Result(ok=True)"
    b.remember_fix(err, ["x = sdk.cache.get(k).value"], now=0, verified=False)
    assert b.known_fixes(now=1) == [] and b.fixes_for(err, now=1, trusted_only=True) == []
    assert b.fixes_for(err, now=1)                              # still available to try (and test) automatically
    b.remember_fix(err, ["x = sdk.cache.get(k).value"], now=2, verified=False)     # replicated
    assert b.known_fixes(now=3) and b.fixes_for(err, now=3, trusted_only=True)
