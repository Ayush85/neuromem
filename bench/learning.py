"""Does the agent get better with experience, in little memory, and adapt when the world changes?

An agent handles a stream of incidents. Each incident type has one correct fix (hidden). The agent picks a fix,
sees success or failure (10% of outcomes are flipped: noise), and records the outcome. Incidents are described
in varied words. Halfway through, the correct fix changes for half the incident types (drift).

Agents (all get the same tiny storage budget):
  random      no memory
  retrieval   stores "task -> action: worked/failed" and copies the action of the most similar past success
              (a typical vector-memory agent; oldest entries dropped when the budget is full)
  neuromem    brain.experience() + brain.advise(); sleeps every 25 incidents (turns trials into rules)

Usage: python bench/learning.py [profile] [seeds...]     profile: tiny | small | full (default small)
"""
import random
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from neuromem import Brain                       # noqa: E402
from neuromem.brain import rss_mb                 # noqa: E402
from neuromem.embed import make_embedder         # noqa: E402

ACTIONS = ["restart the service", "increase the memory limit", "rotate the credentials", "free up disk space",
           "renew the TLS certificate", "add a database index", "roll back the last release", "retry the job"]
TYPES = {
    "oom": ["the {s} process was killed for using too much memory", "{s} crashed with an out of memory error",
            "{s} pods keep getting OOMKilled"],
    "auth": ["{s} returns 401 unauthorized for every request", "{s} cannot authenticate to its upstream",
             "login to {s} fails with invalid token"],
    "disk": ["no space left on device on the {s} host", "the {s} volume is 100 percent full",
             "{s} cannot write logs because the disk is full"],
    "tls": ["clients get certificate expired errors from {s}", "{s} TLS handshake fails, cert not valid",
            "browser shows an insecure connection warning for {s}"],
    "slowdb": ["{s} queries take several seconds", "the database is slow when {s} loads its dashboard",
               "{s} times out on a full table scan"],
    "badrelease": ["{s} started throwing errors right after the deploy", "the new {s} version breaks checkout",
                   "error rate for {s} spiked after today's release"],
    "flaky": ["the nightly {s} batch job failed once with a network blip", "{s} job failed intermittently",
              "a transient error made the {s} cron job fail"],
    "hung": ["{s} stopped responding but the host is fine", "{s} is stuck and not processing requests",
             "health checks for {s} are failing, process looks frozen"],
}
SERVICES = ["billing", "search", "checkout", "auth-api", "inventory", "payments", "reporting", "gateway"]
BEFORE = {"oom": 1, "auth": 2, "disk": 3, "tls": 4, "slowdb": 5, "badrelease": 6, "flaky": 7, "hung": 0}
AFTER = dict(BEFORE, oom=6, disk=0, slowdb=7, flaky=0)    # drift: 4 of 8 types change their correct fix


class Retrieval:
    def __init__(self, embed, budget):
        self.embed, self.budget, self.items = embed, budget, []

    def choose(self, task, r):
        if self.items:
            v = self.embed.one(task)
            sims = np.array([m[0] @ v for m in self.items])
            for i in np.argsort(-sims)[:5]:
                if self.items[i][2] > 0:
                    return self.items[i][1]
        return r.choice(ACTIONS)

    def record(self, task, action, reward):
        self.items.append((self.embed.one(task), action, reward, len(task) + len(action) + 80 + self.embed.dim + 2))
        while sum(m[3] for m in self.items) > self.budget:
            self.items.pop(0)


def neuro_choose(brain, task, r, now, explore=0.1):
    adv = brain.advise(task, now=now)
    if adv["try"] and r.random() > explore:
        return adv["try"][0]["action"]
    avoid = {a["action"] for a in adv["avoid"]}
    return r.choice([a for a in ACTIONS if a not in avoid] or ACTIONS)


def run(seed, profile, episodes=600, budget=20_000, noise=0.1, join=1.0):
    r = random.Random(seed)
    embed = make_embedder(profile)
    agents = {"random": None, "retrieval": Retrieval(embed, budget),
              "neuromem": Brain(":memory:", profile=profile, budget_bytes=budget, join=join)}
    rngs = {k: random.Random(seed * 100 + i) for i, k in enumerate(agents)}
    stream = []
    for ep in range(episodes):
        kind = r.choice(list(TYPES))
        stream.append((kind, r.choice(TYPES[kind]).format(s=r.choice(SERVICES)), r.random() < noise))
    wins = {k: [] for k in agents}
    for ep, (kind, task, flip) in enumerate(stream):
        correct = ACTIONS[(BEFORE if ep < episodes // 2 else AFTER)[kind]]
        now = ep / 25
        for name, agent in agents.items():
            rr = rngs[name]
            if name == "random":
                act = rr.choice(ACTIONS)
            elif name == "retrieval":
                act = agent.choose(task, rr)
            else:
                act = neuro_choose(agent, task, rr, now)
            ok = (act == correct) != flip
            wins[name].append(act == correct)
            reward = 1.0 if ok else -1.0
            if name == "retrieval":
                agent.record(task, act, reward)
            elif name == "neuromem":
                agent.experience(task, act, reward, now=now)
                if ep % 25 == 24:
                    agent.sleep(now=now)
    st = agents["neuromem"].stats()
    return wins, st


if __name__ == "__main__":
    profile = sys.argv[1] if len(sys.argv) > 1 else "small"
    seeds = [int(s) for s in sys.argv[2:]] or [1, 2, 3]
    t0 = time.time()
    allw, stats = {}, []
    for s in seeds:
        w, st = run(s, profile)
        stats.append(st)
        for k, v in w.items():
            allw.setdefault(k, []).append(v)
    windows = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 600)]
    print(f"profile={profile} seeds={seeds}  correct fix chosen, % per window of 100 incidents "
          f"(drift at 300; 8 possible fixes; storage budget 20 KB)")
    print(f"{'agent':<10}" + "".join(f"{f'{a}-{b}':>9}" for a, b in windows))
    for k, runs in allw.items():
        arr = np.array(runs, float)
        print(f"{k:<10}" + "".join(f"{arr[:, a:b].mean():9.0%}" for a, b in windows))
    st = stats[-1]
    print(f"\nneuromem end state (last seed): {st['memories']} memories ({st['situations']} situations, {st['rules']} rules, {st['trials']} trials), "
          f"{st['bytes']:,} B stored, index {st['index_ram_bytes']:,} B in RAM, process {rss_mb()} MB, "
          f"{time.time() - t0:.0f}s")
