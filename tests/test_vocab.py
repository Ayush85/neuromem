import pytest

from neuromem import Brain
from neuromem.embed import make_embedder

pytest.importorskip("model2vec")
from neuromem.vocab import DomainEmbedder, train_vocab  # noqa: E402

GLOSS = {"OOMKilled": "killed for running out of memory", "CrashLoopBackOff": "keeps crashing and restarting"}
PAIRS = [("the billing pods got OOMKilled", "search was killed because it ran out of memory"),
         ("checkout is in CrashLoopBackOff", "payments keeps crashing and restarting"),
         ("after the deploy media got OOMKilled", "ledger ran out of memory and was killed"),
         ("auth is in CrashLoopBackOff again", "gateway keeps restarting after crashes")]


def test_vocab_teaches_jargon_and_leaves_other_text_alone():
    base = make_embedder("small")
    d = DomainEmbedder(base)
    d.add_terms(GLOSS)
    d.train(PAIRS, epochs=20)
    q, target, other = "reporting got OOMKilled overnight", "the job ran out of memory", "the job keeps crashing"
    assert base.one(q) @ base.one(target) < d.one(q) @ d.one(target)
    assert d.one(q) @ d.one(target) > d.one(q) @ d.one(other)
    assert (base.one("lunch is at noon") == d.one("lunch is at noon")).all()


def test_brain_with_vocab_gives_advice_across_jargon(tmp_path):
    path = str(tmp_path / "vocab.json")
    train_vocab(make_embedder("small"), GLOSS, PAIRS, path, epochs=20)
    b = Brain(":memory:", profile="small", vocab=path)
    b.experience("billing pods got OOMKilled", "increase the memory limit", 1, now=0)
    assert b.advise("search ran out of memory", now=1)["try"][0]["action"] == "increase the memory limit"
    plain = Brain(":memory:", profile="small")
    plain.experience("billing pods got OOMKilled", "increase the memory limit", 1, now=0)
    assert plain.advise("search ran out of memory", now=1)["try"] == []
