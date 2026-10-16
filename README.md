# neuromem

Brain-style long-term memory for AI. It decides what is worth keeping, files experiences as episodes, recalls
by meaning and association, updates beliefs when facts change, distils repeated experience into knowledge
while it sleeps, forgets what is no longer used, and learns from experience without retraining the model.
Runs offline in ~40-130 MB of RAM. Design: [DESIGN.md](DESIGN.md).

> **Status: experimental research code.** Main result so far: with a real local LLM (qwen2.5-coder 7B) writing
> code against an unfamiliar SDK that changes halfway through, neuromem's reflex agent keeps **49%** first-try
> success after the change, while a plain retrieval memory drops to **8%** (3 task orders, 48 tasks each).
> Before the change the two are equal. See [Evidence](#evidence).

```python
from neuromem import Brain

b = Brain("brain.db")                          # SQLite on disk
b.perceive("Marco owns the payments service")
b.perceive("Marco is on leave until June 3")
b.perceive("The payments timeout is 30 seconds")
b.perceive("The payments timeout is 45 seconds")   # supersedes the 30-second belief (kept as history)

b.recall("is the payments owner around?")      # Marco owns payments  + (via Marco) the leave note
b.recall("payments timeout")                   # 45 seconds; 30 only with include_history=True
b.about("marco")                               # everything about an entity
b.sleep()                                      # consolidate weak episodes into facts, forget to budget
```

| Mechanism | What it does |
|---|---|
| Attention | ignores trivial input, reinforces exact repeats, scores importance ("critical", "deadline", "!") |
| Layers | episodic (events), semantic (facts), procedural (how-to); facts and how-tos fade 3x slower |
| Recall | semantic embeddings + sparse fragment codes, weighted by memory strength |
| Association | spreading activation over similarity links and shared entities; recalled-together memories get linked |
| Working memory | the last ~7 items re-rank ambiguous recall ("mercury" the planet vs the metal) |
| Belief updating | a newer fact about the same subject supersedes the older one; history is kept |
| Sleep | clusters of weak, similar episodes become one semantic fact (an LLM summary with `--mind claude`) |
| Forgetting | under a byte budget, the weakest, least important, superseded or consolidated memories go first |
| Learning | `experience()` + `advise()`: learns which actions work for which kinds of task, adapts when that changes |
| Reflexes | verified error -> fix pairs; repeated fixes become habits applied automatically; `feedback()` retires memories that mislead |
| Vocabulary | trainable word vectors teach the small embedding models your team's jargon ("OOMKilled") |
| Small RAM | `profile="tiny"` runs in ~100 MB total; int8 vectors; ~0.2-0.6 KB of RAM per memory |

### Learning from experience

```python
b = Brain("agent.db", profile="small")
b.experience("billing pods were killed for using too much memory", "restart the service", -1)
b.experience("billing pods were killed for using too much memory", "increase the memory limit", 1)
b.advise("search ran out of memory")      # try: increase the memory limit
b.prepare("search ran out of memory")     # short briefing to put in an LLM prompt
```

### Examples, reflexes and feedback (for agents)

```python
b.remember_example("sum a user's order totals", working_code)         # verified worked example
b.examples("total spent by a customer")                                 # [(id, task, code)], best first
b.remember_fix(error_message, ["acmekit.init(app='x')"])               # lines that fixed this kind of error
b.fixes_for(new_error)                                                  # remembered fixes for similar errors
b.habits()                                                              # fixes that worked repeatedly
b.feedback([memory_id], -1)                                             # it misled: weaken, then retire
```

### Teaching it your jargon

```python
from neuromem.embed import make_embedder
from neuromem.vocab import train_vocab
train_vocab(make_embedder("small"), {"OOMKilled": "killed for running out of memory"},
            [("billing pods got OOMKilled", "search ran out of memory and was killed")], "vocab.json")
b = Brain("brain.db", profile="small", vocab="vocab.json")
```

## Install and use

Python 3.10+. `profile="auto"` uses the best installed model: `full` (needs the `full` extra, ~280 MB RAM),
otherwise `small` or `tiny` (model2vec, numpy only). Models download once from Hugging Face, then work offline.

```
pip install -e ".[claude,dev]"          # add ,full for the bge-small transformer profile
neuromem --profile small perceive "Priya prefers async code reviews"
neuromem recall "how does Priya like reviews"
neuromem about priya
neuromem experience "disk full on the db host" "free up disk space" 1
neuromem advise "the volume is out of space"
neuromem train-vocab glossary.json --pairs pairs.json --out vocab.json
neuromem --vocab vocab.json recall "pods got OOMKilled"
neuromem sleep
neuromem stats
```

Claude Code: `claude mcp add neuromem -- neuromem-mcp`
(tools: perceive, recall, about, belief_history, experience, advise, prepare, forget, sleep, memory_stats).

**Mind.** Understanding (entities, layer, importance, gists, contradictions) is done by `HeuristicMind`
(offline, default) or `ClaudeMind` (Claude Opus 5.5 with structured outputs, low effort, server-side refusal
fallback; any failure falls back to the heuristics). Enable with `--mind claude` or `NEUROMEM_MIND=claude`
and an `ANTHROPIC_API_KEY` or `ant auth login`. Each perceive then makes one API call.

## Evidence

`python bench/bench.py 6 7 8 9 10 11 12 13 14 15` compares against a plain vector store with the same
embeddings, on generated scenarios (seeds 1-5 were used while fixing bugs; 6-15 are held out):

| Scenario | neuromem | vector store |
|---|---|---|
| Belief updates: current value is the top answer | 80.0% | 44.2% |
| Multi-hop: "is the S owner available?" finds "P is on leave" | 100% | 0.0% |
| Off-topic questions correctly get no answer | 100% | 100% |

**Learning** (`python bench/learning.py`): an agent fixes a stream of incidents (8 kinds, 8 possible fixes,
10% noisy outcomes); halfway through, the right fix changes for 4 kinds. Held-out seeds 11-20 (seeds 101-103
were used for tuning), % of incidents where the right fix was chosen, before | after the change:

| Storage | Model | random | retrieval agent | neuromem |
|---|---|---|---|---|
| 20 KB | tiny | 13 / 12 | 42 / 52 | 40 / 43 |
| 20 KB | small | 13 / 12 | 36 / 49 | 45 / 49 |
| 20 KB | full | 13 / 12 | 32 / 49 | 44 / 50 |
| 200 KB | tiny | 13 / 12 | 39 / 33 | 41 / 39 |
| 200 KB | small | 13 / 12 | 38 / 39 | 45 / 45 |
| 200 KB | full | 13 / 12 | 46 / 39 | 52 / 50 |

The retrieval agent copies the most similar past success. neuromem learns faster in most settings and holds up
better after the change when it has room, but at 20 KB it only matches the baseline after the change, and with
the tiny model it is worse (43% vs 52%).

**Trained jargon vocabulary** (`python bench/vocab.py`): 20 ops terms, 120 training pairs; evaluated on 60
pairs in sentence frames never seen in training. "Finds paraphrase" = the jargon sentence's nearest plain
sentence (among all 20 terms' paraphrases) is the right one.

| Model | finds paraphrase | similarity (0 unrelated, 1 paraphrase) | unrelated text ≥ 0.5 similar |
|---|---|---|---|
| tiny | 45% | 0.36 | 15% |
| tiny + definitions | 92% | 1.27 | 20% |
| tiny + trained | 98% | 1.34 | 12% |
| small | 55% | 0.43 | 2% |
| small + definitions | 100% | 1.33 | 7% |
| small + trained | 100% | 1.41 | 5% |

Most of the gain comes from initialising each term from its definition; training adds a little more. Text
without the terms embeds exactly as before.

**Memory in weights** (`python bench/assoc.py`, experimental): a Willshaw associative network recalls from 40%
fragments worse than explicit storage and needs more RAM at these sizes:

| Neurons | Stored | explicit storage | network | explicit RAM | network RAM |
|---|---|---|---|---|---|
| 2,048 | 250 | 94% | 76% | 23 KB | 1.5 MB |
| 2,048 | 1,153 | 77% | 39% | 105 KB | 1.5 MB |
| 8,192 | 250 | 95% | 86% | 23 KB | 12 MB |
| 8,192 | 1,153 | 79% | 56% | 106 KB | 12 MB |

Explicit sparse codes cost ~100 bytes per memory, so a fixed weight matrix only saves RAM beyond several
thousand memories, and its capacity runs out before that. Kept as an experiment; the Brain does not use it.

**Real model** (`python realtest/harness.py retrieval reflex --seed 1`, needs [Ollama](https://ollama.com) with
`qwen2.5-coder:7b`): a local LLM (qwen2.5-coder 7B via Ollama) writes code for 48 tasks
against `acmekit`, a fictional SDK with hidden quirks (init first, lowercase emails, integer cents, pagination,
timezones, retries, a Result wrapper); code is run against tests. Halfway through, the SDK moves to v2 and
`init()` changes. First-try success, 3 task orders (seeds 0-2):

| Agent | before the SDK change | after the change | total |
|---|---|---|---|
| no memory | 0% | 0% | 0% |
| retrieval memory + worked examples | 46% | 8% | 27% |
| neuromem reflex agent (`realtest/reflex.py`) | 44% | 49% | 46% |

The reflex agent stores verified error -> fix pairs, shows them up front, turns repeated fixes into habits applied
automatically, and gets feedback on every memory it used, so stale examples and habits are retired quickly.
Before the change it is no better than retrieval; after the change it keeps working while retrieval keeps
replaying outdated examples. Habits only form for whole-statement fixes (e.g. an init call); in-line fixes
(`.value`, `email.strip().lower()`) are not learned yet. Results vary a lot between task orders.

`pytest tests`: 36 tests; they pass with every profile (`NEUROMEM_PROFILE=tiny|small|full pytest tests`).

## Limits

- The benchmark scenarios are templated; the heuristic contradiction rule was written for statements of the
  form "X is N" vs "X is M". Real text needs `ClaudeMind`, which has only been tested with a fake client.
- Heuristic entities are capitalised words; lowercase subjects ("payments") are not entities.
- One process at a time; recall scans all memories (fine to ~10^5).
- Jargon the embedding model doesn't know matches poorly unless you train a vocabulary for it.
- The real-model test uses one fictional SDK and one 7B model; results vary a lot between task orders.
- Reflex habits only learn whole-statement fixes, not edits inside a line.
- No real long-term usage data yet.

## Related work

Most of these ideas exist elsewhere: memory for LLM agents with fact extraction and conflict resolution
(mem0), temporal knowledge graphs that invalidate old facts (Zep / Graphiti), tiered working + archival memory
(Letta / MemGPT), episodic / semantic / procedural memory types (LangMem), recency x importance x relevance
recall with reflection (Generative Agents), forgetting curves (MemoryBank), hippocampus-inspired multi-hop
retrieval (HippoRAG), and sparse random codes (FlyHash). neuromem combines them in one small offline package
and measures each mechanism against simple baselines; it has not been compared head-to-head with those systems.
