# neuromem: design

A memory system for AI that stores and recalls information the way a human brain does: it filters what to
keep, files experiences as episodes, distils repeated experience into knowledge while it "sleeps", recalls by
meaning and association, updates beliefs when facts change, and forgets what is no longer used.

## 1. Memory layers

| Layer | Brain analogue | Holds | Lifetime |
|---|---|---|---|
| Working memory | Prefrontal cortex | The last ~7 things perceived or recalled; biases recall toward the current context | In-process only |
| Episodic | Hippocampus | Events as they happened: text, when, source, entities involved | Fades unless recalled; consolidated during sleep |
| Semantic | Neocortex | Facts distilled from episodes ("deploys usually fail on Fridays") | Long-lived (decays 3x slower) |
| Procedural | Basal ganglia | How-to knowledge ("to deploy, run make release") | Long-lived (decays 3x slower) |

All long-term layers live in one SQLite table (`memories.layer`), plus an entity graph and an association graph.

## 2. Write path: `Brain.perceive(text)`

1. **Attend.** Score salience = novelty + information content + emphasis cues ("important", "never",
   "deadline", "!"). Trivial input ("ok thanks") is ignored. An exact repeat reinforces the existing memory;
   similar-but-distinct events are kept as separate episodes so sleep can distil them.
2. **Understand.** A `Mind` extracts entities, time, layer and importance. `HeuristicMind` works offline;
   `ClaudeMind` uses the Claude API for better extraction.
3. **Encode.** Each memory gets three codes: a sparse neural code (fragment matching), a semantic embedding
   (meaning), and its entity mentions (who/what it is about).
4. **Link.** Associate with the most similar existing memories, and with entities via `mentions`.
5. **Update beliefs.** If the new memory contradicts an older one about the same thing ("the API limit is now
   200/min" vs "the API limit is 100/min"), the old one is marked superseded. It is kept as history, not deleted.
6. **Store** with initial strength S = s0 * (1 + novelty + importance) (x3 for semantic/procedural).

## 3. Read path: `Brain.recall(cue)`

1. **Match.** relevance = 0.7 * semantic similarity + 0.3 * fragment overlap, weighted by strength
   R = exp(-(now - last_touched) / S).
2. **Context.** Memories close to working memory get a small boost (context-dependent recall).
3. **Entities.** If the cue names a known entity, memories about that entity are activated.
4. **Spread activation.** Activation flows from the best hits along association links and shared entities
   (two hops, decaying), so recall returns related memories too, marked `via`.
5. **Reconsolidate.** The top hit is strengthened (spacing effect: more if it was nearly forgotten), memories
   recalled together are linked (Hebbian), and the cue enters working memory.

Superseded memories are hidden unless `include_history=True`.

## 4. Sleep: `Brain.sleep()`

1. **Replay and consolidate.** Weak, old episodes are clustered by similarity; each cluster of 2+ is distilled
   by the Mind into one semantic fact (an LLM writes a real summary; offline it keeps the most central
   sentence). Source episodes are kept but marked consolidated and decay faster.
2. **Forget.** If over the byte budget, drop memories with the lowest retention priority
   R * (1 + importance) * layer_weight, where superseded and consolidated memories count less.
3. **Prune.** Association weights decay; very weak links are removed.

## 5. Learning from experience (no retraining)

The model's weights never change; the memory learns.

- `experience(task, action, reward)` files the attempt under a **situation** (a kind of task). A task joins
  the nearest situation if it is close enough, otherwise starts a new one; the situation's embedding is the
  running centroid of its tasks, so it represents the kind of task rather than one wording.
- Each (situation, action) has a **rule** with exponentially-weighted outcome statistics (each new outcome
  discounts older ones by 0.85), so rules track a world that changes. Trials compress into a handful of rules.
- `advise(task)` decides from the nearest situation's own rules; neighbouring situations only inform actions it
  has not tried, at a discount. Absolute similarity thresholds separate task types poorly (the same service
  name makes different problems look alike), so neighbours are weighted relative to the closest one.
- `prepare(task)` turns advice plus relevant facts and how-tos into a short briefing for an LLM prompt.

## 6. Small RAM

| Profile | Embedding model | Process RAM (5,000 memories) |
|---|---|---|
| none | fragment codes only | ~41 MB |
| tiny | potion-base-2M (static, 64 dims) | ~100 MB |
| small | potion-base-8M (static, 256 dims) | ~129 MB |
| full | bge-small-en-v1.5 (transformer, 384 dims) | ~276 MB |

- Embeddings are stored as int8 with a float16 scale (dim + 2 bytes, 4x smaller than float32) and scored in
  chunks, so temporary arrays stay small. Text and metadata stay in SQLite, not RAM.
- The in-RAM index is ~0.2-0.6 KB per memory. No scipy; numpy only.
- Every embedder is calibrated on built-in sentence pairs, because cosine scales differ by model.
- Switching profile re-embeds the stored memories automatically.
- Rules and trial records are stored without their own search index (they are reached through situations).

## 7. Trained domain vocabulary (`neuromem.vocab`)

Static embedding models split jargon ("OOMKilled" -> o, ##om, ##kill, ##ed) into pieces that carry no meaning.
`DomainEmbedder` adds one trainable vector per glossary term: initialised from the embedding of its plain
definition, then trained with a contrastive (InfoNCE) loss on (jargon sentence, plain paraphrase) pairs, in
numpy (Adam). Base vectors are frozen, so text without the terms embeds exactly as before.
`Brain(vocab="vocab.json")` uses it; `neuromem train-vocab glossary.json --pairs pairs.json --out vocab.json`
trains it.

## 8. Memory in weights (`neuromem.assoc`, experimental)

A Willshaw associative network over the sparse codes: storing switches on binary synapses between a memory's
active neurons and from them to a sparse id code; recall completes a fragment (k-winners-take-all iterations)
and decodes the id. RAM is fixed by the network size. Measured result: at the scales tested it loses to
explicit storage on both accuracy and RAM (see README), so the Brain does not use it.

## 9. Interfaces

- Python: `from neuromem import Brain`
- CLI: `neuromem perceive|recall|sleep|entity|stats|forget`
- MCP server: `neuromem-mcp` (for Claude Code and other agents)

## 10. How we know it works

Every mechanism must earn its place in `bench/`: recall accuracy, paraphrase recall, false-positive rate,
belief updating and retention under a budget, compared against a plain vector store baseline. "Brain-like" is
the inspiration; recall quality is the measure.
