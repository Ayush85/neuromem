"""Does the brain design beat a plain vector store (same embeddings, cosine top-k)?

Scenarios (generated, seeded; distractor sentences mixed in):
  updates    a service attribute changes 1-3 times; ask for the current value. Correct = top hit has the latest value.
             Half the updates say so explicitly ("now", "changed to"), half are bare restatements.
  multihop   "P owns S" + "P is on leave until D"; ask "is the S owner available?". Correct = the leave
             memory for the right person is among the results (brain: matches + associations, baseline: top-3).
  unrelated  off-topic questions. Correct = nothing returned (baseline: cosine threshold tuned like the brain's).

Usage: python bench/bench.py [seed ...]
"""
import random
import sys

import numpy as np

sys.path.insert(0, ".")
from neuromem import Brain                       # noqa: E402
from neuromem.embed import make_embedder         # noqa: E402

SERVICES = ["billing", "search", "checkout", "auth", "inventory", "payments", "notifications", "reporting", "gateway",
            "ledger", "shipping", "catalog", "pricing", "analytics", "identity", "media", "export", "scheduler"]
PEOPLE = ["Priya", "Marco", "Dana", "Kenji", "Sara", "Omar", "Lena", "Tariq", "Mei", "Jonas", "Aisha", "Pavel"]
ATTRS = [("port", "The {s} service runs on port {v}", "{s} now runs on port {v}", "what port does the {s} service use",
          lambda r: r.choice([3000, 5432, 6379, 8000, 8080, 8443, 9000, 9090, 9200, 5000])),
         ("timeout", "The {s} request timeout is {v} seconds", "The {s} timeout was changed to {v} seconds",
          "how long is the {s} timeout", lambda r: r.choice([5, 10, 15, 30, 45, 60, 90, 120])),
         ("replicas", "The {s} deployment has {v} replicas", "We scaled {s} to {v} replicas now",
          "how many replicas does {s} have", lambda r: r.choice([2, 3, 4, 6, 8, 12]))]
DISTRACT = ["The team lunch is on Thursday at the Thai place", "Remember to fill in the quarterly survey",
            "The office wifi password is posted in the kitchen", "Our standup moved to 9:45 for the summer",
            "The new hire onboarding doc lives in the wiki", "Coffee machine on floor two is broken again",
            "The design review for the mobile app went well", "We celebrated the product launch with cake",
            "Parking is limited during the building renovation", "The holiday party will be in December"]
OFFTOPIC = ["what is the capital of australia", "how do I bake sourdough bread", "who won the world cup",
            "best hiking trails near the mountains", "how to learn the piano quickly", "what time is sunset today",
            "recipe for vegetable curry", "how tall is mount everest", "tips for a marathon", "how do bees make honey"]


class VectorStore:
    def __init__(self, embed):
        self.embed, self.texts, self.E = embed, [], []

    def perceive(self, text, **_):
        self.texts.append(text)
        self.E.append(self.embed.one(text))

    def recall(self, cue, k=3, min_cos=0.68):
        if not self.texts:
            return []
        cos = np.array(self.E) @ self.embed.one(cue)
        return [self.texts[i] for i in np.argsort(-cos)[:k] if cos[i] >= min_cos]


def scenario(seed):
    r = random.Random(seed)
    events, upd_q, hop_q = [], [], []
    for s in r.sample(SERVICES, 12):
        name, first, update, q, gen = r.choice(ATTRS)
        v = gen(r)
        events.append(first.format(s=s, v=v))
        for _ in range(r.randint(1, 3)):
            nv = gen(r)
            while nv == v:
                nv = gen(r)
            v = nv
            events.append((update if r.random() < 0.5 else first).format(s=s, v=v))
        upd_q.append((q.format(s=s), str(v)))
    for p, s in zip(r.sample(PEOPLE, 6), r.sample(SERVICES, 6)):
        d = f"{r.choice(['March', 'June', 'October'])} {r.randint(2, 28)}"
        events.append(f"{p} owns the {s} service")
        events.append(f"{p} is on leave until {d}")
        hop_q.append((f"is the {s} owner available this week", f"{p} is on leave"))
    return events, upd_q, hop_q


def run(seed, embed):
    events, upd_q, hop_q = scenario(seed)
    r = random.Random(seed + 1)
    stream = []
    order = list(range(len(events)))          # keep each fact's updates in order, interleave with distractors
    for i in order:
        stream.append(events[i])
        if r.random() < 0.5:
            stream.append(r.choice(DISTRACT))
    brain, base = Brain(":memory:", profile="full"), VectorStore(embed)
    for day, text in enumerate(stream):
        brain.perceive(text, now=day * 0.1)
        base.perceive(text)
    now = len(stream) * 0.1 + 1
    res = {}
    ok_b = sum(any(v in h["text"] for h in brain.recall(q, k=1, associated=0, now=now, strengthen=False)[:1])
               for q, v in upd_q)
    ok_v = sum(any(v in t for t in base.recall(q, k=1)) for q, v in upd_q)
    res["updates"] = (ok_b / len(upd_q), ok_v / len(upd_q))
    ok_b = sum(any(need in h["text"] for h in brain.recall(q, k=1, associated=2, now=now, strengthen=False))
               for q, need in hop_q)
    ok_v = sum(any(need in t for t in base.recall(q, k=3)) for q, need in hop_q)
    res["multihop"] = (ok_b / len(hop_q), ok_v / len(hop_q))
    ok_b = sum(not brain.recall(q, k=1, now=now, strengthen=False) for q in OFFTOPIC)
    ok_v = sum(not base.recall(q, k=1) for q in OFFTOPIC)
    res["unrelated"] = (ok_b / len(OFFTOPIC), ok_v / len(OFFTOPIC))
    return res


if __name__ == "__main__":
    embed = make_embedder("full")
    seeds = [int(s) for s in sys.argv[1:]] or [1, 2, 3, 4, 5]
    tot = {}
    for s in seeds:
        for k, v in run(s, embed).items():
            tot.setdefault(k, []).append(v)
    print(f"seeds {seeds}   (accuracy: neuromem vs plain vector store, same embeddings)")
    print(f"{'scenario':<11}{'neuromem':>10}{'vector':>10}")
    for k, vals in tot.items():
        b, v = np.mean(vals, axis=0)
        print(f"{k:<11}{b:10.1%}{v:10.1%}")
