"""mem0 (an established LLM-agent memory) as the memory of the same coding agent, for a head-to-head test.

Lessons go through mem0.add(infer=True): mem0's LLM (the same local model) extracts facts and adds, updates or
deletes existing memories - its conflict-resolution feature. Worked examples are stored verbatim (infer=False).
Before each task the agent gets the top memories for the task. Same embedding model as neuromem (bge-small via
fastembed). Telemetry is disabled.
"""
import os
import tempfile

os.environ.setdefault("MEM0_TELEMETRY", "False")


class Mem0Memory:
    def __init__(self, model="qwen2.5-coder:7b"):
        from mem0 import Memory
        self.dir = tempfile.mkdtemp(prefix="mem0_bench_")
        self.m = Memory.from_config({
            "llm": {"provider": "ollama", "config": {"model": model, "temperature": 0,
                                                    "ollama_base_url": "http://localhost:11434"}},
            "embedder": {"provider": "fastembed", "config": {"model": "BAAI/bge-small-en-v1.5"}},
            "vector_store": {"provider": "qdrant", "config": {"path": os.path.join(self.dir, "q"),
                                                              "embedding_model_dims": 384,
                                                              "collection_name": "bench"}},
            "history_db_path": os.path.join(self.dir, "history.db"),
        })
        self.shown = []

    def _search(self, q, k):
        out = self.m.search(q, top_k=k, filters={"agent_id": "bench"})
        return out.get("results", out) if isinstance(out, dict) else out

    def notes(self, task):
        from harness import fmt_notes
        hits = self._search(task, 6)
        rules = [h["memory"] for h in hits if (h.get("metadata") or {}).get("kind") != "example"][:3]
        exs = [((h.get("metadata") or {}).get("task", ""), h["memory"]) for h in hits
               if (h.get("metadata") or {}).get("kind") == "example"][:2]
        return fmt_notes("\n".join("- " + r for r in rules), exs)

    def store(self, task, lesson, ok):
        self.m.add(lesson, agent_id="bench", infer=True)

    def store_example(self, task, code):
        self.m.add(code, agent_id="bench", infer=False, metadata={"kind": "example", "task": task})

    def outcome(self, ok):
        pass
