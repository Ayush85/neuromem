"""Brain: layered long-term memory that learns from experience, built to run in little RAM.

See DESIGN.md. Time is measured in days (float); pass `now=` to simulate, otherwise wall-clock is used.

RAM: the in-memory index holds, per memory, its sparse code (~40 x int16), an int8 embedding (64-384 bytes) and
a few floats. Text and everything else stays in SQLite. Scoring runs in chunks, so temporary arrays stay small.
"""
import json
import os
import re
import sqlite3
import time
from collections import deque

import numpy as np

from .codes import SparseCoder, words
from .embed import make_embedder
from .mind import CHANGE, NEGATION, make_mind

LAYERS = ("episodic", "semantic", "procedural")
LAYER_DECAY = {"episodic": 1.0, "semantic": 3.0, "procedural": 3.0}     # stability multiplier
LAYER_KEEP = {"episodic": 1.0, "semantic": 2.0, "procedural": 2.0}      # forgetting priority multiplier
CHUNK = 8192                                                            # rows per scoring chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    layer             TEXT NOT NULL,
    text              TEXT NOT NULL UNIQUE,
    code              BLOB NOT NULL,
    emb               BLOB,
    created           REAL NOT NULL,
    t                 REAL NOT NULL,
    S                 REAL NOT NULL,
    imp               REAL NOT NULL DEFAULT 0,
    hits              INTEGER NOT NULL DEFAULT 0,
    sources           INTEGER NOT NULL DEFAULT 1,
    when_text         TEXT,
    source            TEXT,
    meta              TEXT,
    superseded_by     INTEGER,
    consolidated_into INTEGER,
    task              TEXT,
    action            TEXT,
    n                 REAL NOT NULL DEFAULT 0,
    reward            REAL NOT NULL DEFAULT 0,
    situation         INTEGER
);
CREATE INDEX IF NOT EXISTS memories_action ON memories (action);
CREATE INDEX IF NOT EXISTS memories_situation ON memories (situation);
CREATE TABLE IF NOT EXISTS entities (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL DEFAULT 'thing'
);
CREATE TABLE IF NOT EXISTS mentions (
    memory_id INTEGER NOT NULL,
    entity_id INTEGER NOT NULL,
    PRIMARY KEY (memory_id, entity_id)
);
CREATE INDEX IF NOT EXISTS mentions_entity ON mentions (entity_id);
CREATE TABLE IF NOT EXISTS links (
    a INTEGER NOT NULL, b INTEGER NOT NULL, w REAL NOT NULL,
    PRIMARY KEY (a, b)
);
CREATE INDEX IF NOT EXISTS links_b ON links (b);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def default_db_path():
    return os.environ.get("NEUROMEM_DB") or os.path.join(os.path.expanduser("~"), ".neuromem", "brain.db")


def _now():
    return time.time() / 86400.0


def quantize(v):
    """Unit vector -> int8 blob with a float16 scale (dim + 2 bytes)."""
    scale = float(np.abs(v).max()) or 1.0
    q = np.round(v / scale * 127).astype(np.int8)
    return np.float16(scale / 127).tobytes() + q.tobytes()


def dequantize(blob):
    return np.frombuffer(blob[2:], np.int8).astype(np.float32) * float(np.frombuffer(blob[:2], np.float16)[0])


def rss_mb():
    try:
        return int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) // 1024
    except Exception:
        return None


class Brain:
    def __init__(self, path=None, budget_bytes=100_000_000, profile=None, embedder=None, mind=None, s0=3.0,
                 working_size=7, min_cluster=2, join=1.0, vocab=None):
        """profile: 'tiny' | 'small' | 'full' | 'none' | 'auto' (default: $NEUROMEM_PROFILE or 'auto').
        embedder: alias of profile, or a custom embedder object.
        vocab: path to trained domain vocabulary vectors (see neuromem.vocab); needs a tiny/small profile."""
        self.path = path or default_db_path()
        if self.path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.db = sqlite3.connect(self.path)
        if self.path != ":memory:":
            self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.db.commit()
        self.budget, self.s0, self.min_cluster, self.join = budget_bytes, s0, min_cluster, join
        self.coder = SparseCoder()
        choice = embedder if embedder is not None else (profile or os.environ.get("NEUROMEM_PROFILE", "auto"))
        self.embed = make_embedder(choice)
        if vocab and self.embed is not None:
            from .vocab import DomainEmbedder
            self.embed = DomainEmbedder.load(self.embed, vocab)
        self.mind = make_mind(mind or os.environ.get("NEUROMEM_MIND", "heuristic"))
        self.min_score = 0.2 if self.embed is not None else 0.35
        self.working = deque(maxlen=working_size)                       # (memory id or None, embedding)
        self._cache = None
        self._reembed_if_needed()
        self.bytes = self._sql_bytes()

    def close(self):
        self.db.close()

    # ================================================================ embeddings
    def _vec(self, text):
        return None if self.embed is None else self.embed.one(text)

    def _sem(self, cos):
        """Map raw cosine to calibrated similarity: 0 = unrelated, 1 = typical paraphrase."""
        return np.clip((cos - self.embed.lo) / (self.embed.hi - self.embed.lo), 0, 1.5)

    def _reembed_if_needed(self):
        name = getattr(self.embed, "name", None) if self.embed is not None else "none"
        row = self.db.execute("SELECT value FROM meta WHERE key = 'embedder'").fetchone()
        if row and row[0] != name:
            rows = self.db.execute("SELECT id, COALESCE(task, text) FROM memories").fetchall()
            for i in range(0, len(rows), 256):
                part = rows[i:i + 256]
                vecs = self.embed([r[1] for r in part]) if self.embed is not None else [None] * len(part)
                self.db.executemany("UPDATE memories SET emb = ? WHERE id = ?",
                                    [(None if v is None else quantize(v), r[0]) for r, v in zip(part, vecs)])
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('embedder', ?)", (name,))
        self.db.commit()

    # ================================================================ index (kept in sync incrementally)
    def _index(self):
        if self._cache is None:
            rows = self.db.execute("SELECT id, code, emb, t, S, imp, layer, superseded_by IS NOT NULL, "
                                   "consolidated_into IS NOT NULL FROM memories ORDER BY id").fetchall()
            codes = [np.frombuffer(r[1], np.int16) for r in rows]
            dim = len(dequantize(rows[0][2])) if rows and rows[0][2] is not None else 0
            E8 = np.zeros((len(rows), dim), np.int8)
            Esc = np.zeros(len(rows), np.float32)
            for i, r in enumerate(rows):
                if r[2] is not None and dim:
                    E8[i] = np.frombuffer(r[2][2:], np.int8)
                    Esc[i] = float(np.frombuffer(r[2][:2], np.float16)[0])
            self._cache = {
                "ids": np.array([r[0] for r in rows], np.int64),
                "codes": codes,
                "indptr": np.concatenate([[0], np.cumsum([len(c) for c in codes])]).astype(np.int64),
                "indices": np.concatenate(codes) if codes else np.zeros(0, np.int16),
                "E8": E8, "Esc": Esc,
                "t": np.array([r[3] for r in rows], float), "S": np.array([r[4] for r in rows], float),
                "imp": np.array([r[5] for r in rows], float),
                "layer": np.array([LAYERS.index(r[6]) for r in rows], np.int8),
                "sup": np.array([bool(r[7]) for r in rows], bool),
                "cons": np.array([bool(r[8]) for r in rows], bool)}
        return self._cache

    def _pos(self, mid):
        hit = np.flatnonzero(self._index()["ids"] == mid)
        return int(hit[0]) if len(hit) else None

    def _cache_add(self, mid, code, blob, t, S, imp, layer):
        c = self._index()
        c["ids"] = np.append(c["ids"], mid)
        c["codes"].append(code)
        c["indices"] = np.concatenate([c["indices"], code])
        c["indptr"] = np.append(c["indptr"], c["indptr"][-1] + len(code))
        if blob is not None:
            if c["E8"].shape[1] == 0:
                c["E8"] = np.zeros((len(c["ids"]) - 1, len(blob) - 2), np.int8)
            c["E8"] = np.vstack([c["E8"], np.frombuffer(blob[2:], np.int8)[None, :]])
            c["Esc"] = np.append(c["Esc"], float(np.frombuffer(blob[:2], np.float16)[0]))
        else:
            c["E8"] = np.vstack([c["E8"], np.zeros((1, c["E8"].shape[1]), np.int8)])
            c["Esc"] = np.append(c["Esc"], 0.0)
        for k, v in (("t", t), ("S", S), ("imp", imp), ("layer", LAYERS.index(layer)), ("sup", False),
                     ("cons", False)):
            c[k] = np.append(c[k], v)

    def _cache_set(self, mid, **vals):
        i = self._pos(mid)
        if i is not None:
            for k, v in vals.items():
                self._cache[k][i] = v

    # ================================================================ scoring
    def _cos(self, v, rows=None):
        """Cosine of v against all (or the given) memories, in chunks so the float copy stays small."""
        c = self._index()
        E8, Esc = (c["E8"], c["Esc"]) if rows is None else (c["E8"][rows], c["Esc"][rows])
        if not E8.shape[1]:
            return np.zeros(len(Esc), np.float32)
        out = np.empty(len(Esc), np.float32)
        for i in range(0, len(Esc), CHUNK):
            out[i:i + CHUNK] = (E8[i:i + CHUNK].astype(np.float32) @ v) * Esc[i:i + CHUNK]
        return out

    def _overlap(self, code):
        """Fragment overlap: how many of the cue's neurons each memory's code contains."""
        c = self._index()
        if not len(c["indices"]):
            return np.zeros(len(c["ids"]), np.float32)
        hit = np.isin(c["indices"], code).astype(np.float32)
        csum = np.concatenate([[0.0], np.cumsum(hit)])
        return (csum[c["indptr"][1:]] - csum[c["indptr"][:-1]]).astype(np.float32)

    def _relevance(self, code, emb):
        """Per-memory relevance in [0, ~1]: 0.7 calibrated semantic + 0.3 fragment overlap."""
        c = self._index()
        n = self._overlap(code)
        sizes = np.diff(c["indptr"]).astype(np.float32)
        lex = np.minimum(1.0, n / max(1, len(code)) + 0.01 * n / np.maximum(sizes, 1))
        if emb is None or not c["E8"].shape[1]:
            return lex, lex
        sem = np.minimum(self._sem(self._cos(emb)), 1.0)
        return 0.7 * sem + 0.3 * lex, sem

    def _strength(self, now):
        c = self._index()
        return np.exp(-(now - c["t"]) / c["S"])

    # ================================================================ write path
    def perceive(self, text, importance=None, layer=None, source=None, when=None, meta=None, now=None, force=False):
        """Take in an experience. Returns {"action": stored|reinforced|ignored, "id", "layer", "entities", "superseded"}."""
        text = " ".join(text.split())
        now = _now() if now is None else now
        if len(words(text)) < 2 and not force:
            return {"action": "ignored", "reason": "no content", "id": None}
        row = self.db.execute("SELECT id FROM memories WHERE text = ?", (text,)).fetchone()
        if row:                                     # the exact same experience again: reinforce, don't duplicate
            return self._reinforce(row[0], now)
        info = self.mind.extract(text)
        layer = layer or info.get("layer") or "episodic"
        if layer != "episodic" and self.embed is not None and len(self._index()["ids"]):
            # a fact or rule restated in other words strengthens the existing memory instead of duplicating it
            c = self._index()
            sem = self._sem(self._cos(self._vec(text)))
            same = (c["layer"] == LAYERS.index(layer)) & ~c["sup"] & (sem >= 1.25)
            if same.any() and not force:
                i = int(np.argmax(np.where(same, sem, -1)))
                old = self.db.execute("SELECT text FROM memories WHERE id = ?", (int(c["ids"][i]),)).fetchone()[0]
                if not self.mind.contradicts(old, text):          # a changed fact must replace, not reinforce
                    return self._reinforce(int(c["ids"][i]), now)
        imp = float(info.get("importance", 0.0) if importance is None else importance)
        return self._store(text, layer=layer, imp=imp, now=now, entities=info.get("entities", []),
                           when=when or info.get("when") or None, source=source, meta=meta)

    def _reinforce(self, mid, now, factor=1.5):
        i = self._pos(mid)
        S = float(self._cache["S"][i]) * factor
        self.db.execute("UPDATE memories SET t = ?, S = ?, hits = hits + 1 WHERE id = ?", (now, S, mid))
        self.db.commit()
        self._cache_set(mid, t=now, S=S)
        return {"action": "reinforced", "id": mid}

    def _store(self, text, layer, imp, now, entities=(), when=None, source=None, meta=None, task=None,
               action=None, n=0.0, reward=0.0, check_beliefs=True, searchable=True):
        """searchable=False stores no code or embedding (e.g. rules, which are reached through their situation)."""
        if layer not in LAYERS:
            layer = "episodic"
        code = self.coder(task or text) if searchable else np.zeros(0, np.int16)
        emb = self._vec(task or text) if searchable else None
        if not searchable:
            check_beliefs = False
        if searchable:
            self.working.append((None, emb))
        c = self._index()
        rel = None
        if len(c["ids"]) and searchable:
            rel, _ = self._relevance(code, emb)
        novelty = 1.0 if rel is None else 1.0 - float(min(1.0, rel.max()))
        S = self.s0 * (1 + novelty + imp) * LAYER_DECAY[layer]
        blob = None if emb is None else quantize(emb)
        cur = self.db.execute(
            "INSERT INTO memories (layer, text, code, emb, created, t, S, imp, when_text, source, meta, task, action, "
            "n, reward) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (layer, text, code.tobytes(), blob, now, now, S, imp, when, source,
             json.dumps(meta) if meta is not None else None, task, action, n, reward))
        mid = cur.lastrowid
        ents = []
        for e in entities:
            name = e["name"].strip().lower()
            if not name:
                continue
            self.db.execute("INSERT INTO entities (name, kind) VALUES (?, ?) ON CONFLICT(name) DO NOTHING",
                            (name, e.get("kind", "thing")))
            eid = self.db.execute("SELECT id FROM entities WHERE name = ?", (name,)).fetchone()[0]
            self.db.execute("INSERT OR IGNORE INTO mentions VALUES (?, ?)", (mid, eid))
            ents.append(name)

        superseded = []
        if rel is not None and searchable:
            for i in np.argsort(-rel)[:5]:
                if rel[i] < 0.3:
                    break
                other = int(c["ids"][i])
                self._link(mid, other, float(min(1.0, rel[i])))
                # events don't contradict each other; facts/states do, and so does any explicit update
                fact_pair = layer != "episodic" and c["layer"][i] != 0
                if (check_beliefs and not c["sup"][i] and rel[i] >= 0.45
                        and (fact_pair or CHANGE.search(text) or NEGATION.search(text))):
                    old = self.db.execute("SELECT text FROM memories WHERE id = ?", (other,)).fetchone()[0]
                    if self.mind.contradicts(old, text):
                        self.db.execute("UPDATE memories SET superseded_by = ? WHERE id = ?", (mid, other))
                        superseded.append(other)
        self.db.commit()
        self._cache_add(mid, code, blob, now, S, imp, layer)
        for o in superseded:
            self._cache_set(o, sup=True)
        if searchable:
            self.working[-1] = (mid, emb)
        self.bytes += self._row_bytes(text, code, blob, task, action)
        if self.bytes > self.budget:
            self._forget_to_budget(now)
        return {"action": "stored", "id": mid, "layer": layer, "importance": imp, "entities": ents,
                "superseded": superseded}

    @staticmethod
    def _row_bytes(text, code, blob, task, action):
        return (len(text.encode()) + len(code.tobytes()) + (len(blob) if blob else 0) + len((task or "").encode())
                + len((action or "").encode()) + 80)

    def _link(self, a, b, w):
        if a == b:
            return
        a, b = min(a, b), max(a, b)
        self.db.execute("INSERT INTO links (a, b, w) VALUES (?,?,?) ON CONFLICT(a,b) DO UPDATE SET "
                        "w = MIN(1.0, MAX(w, excluded.w) + 0.05)", (a, b, w))

    # ================================================================ learning from experience
    # Experiences are grouped into situations (kinds of task). A situation's embedding is the running centroid
    # of the tasks assigned to it, so it comes to represent the kind of task rather than one wording. For each
    # (situation, action) a rule keeps exponentially-weighted outcome statistics: recent outcomes count more,
    # so rules track a changing world. Dozens of trials compress into a handful of rules.

    def experience(self, task, action, reward, note=None, now=None):
        """Record an attempt: doing `action` for `task` earned `reward` (-1 = failed .. 1 = worked)."""
        now = _now() if now is None else now
        reward = float(max(-1.0, min(1.0, reward)))
        task, action = " ".join(task.split()), action.strip()
        sit = self._situation_for(task, now)
        rule = self.db.execute("SELECT id FROM memories WHERE situation = ? AND action = ? AND layer = 'procedural'",
                               (sit, action)).fetchone()
        if rule is None:
            res = self._store(f"situation {sit}: {action} (new)", layer="procedural", imp=0.5, now=now,
                              action=action, n=0.0, reward=0.0, source="learning", searchable=False)
            rule = res["id"]
            self.db.execute("UPDATE memories SET situation = ? WHERE id = ?", (sit, rule))
            self._link(sit, rule, 0.9)
        else:
            rule = rule[0]
        self._update_rule(rule, reward, now)
        verdict = "worked" if reward > 0 else "failed" if reward < 0 else "had no effect"
        text = f"{task} -> {action}: {verdict}" + (f" ({note})" if note else "")
        while self.db.execute("SELECT 1 FROM memories WHERE text = ?", (text,)).fetchone():
            text += " ."                                         # repeated trials are distinct experiences
        out = self._store(text, layer="episodic", imp=0.0, now=now, task=task, action=action, n=1.0, reward=reward,
                          source="experience", searchable=False)     # already counted in its rule
        # the trial is already counted in its rule: mark it consolidated so it is forgotten first under pressure
        self.db.execute("UPDATE memories SET consolidated_into = ?, situation = ? WHERE id = ?", (rule, sit, out["id"]))
        self.db.commit()
        self._cache_set(out["id"], cons=True)
        self._link(rule, out["id"], 0.6)
        return {**out, "situation": sit, "rule": rule}

    def _situations(self):
        return self.db.execute("SELECT id, n FROM memories WHERE source = 'situation'").fetchall()

    def _situation_sims(self, task, rows):
        pos = np.array([self._pos(r[0]) for r in rows])
        v = self._vec(task)
        if v is not None:
            return self._sem(self._cos(v, pos))
        q = set(self.coder(task).tolist())
        c = self._index()
        return np.array([len(q & set(c["codes"][p].tolist())) / max(1, len(q)) for p in pos])

    def _situation_for(self, task, now):
        """Nearest situation if it is close enough (calibrated similarity >= self.join), else a new one."""
        join = self.join
        rows = self._situations()
        if rows:
            sims = self._situation_sims(task, rows)
            i = int(np.argmax(sims))
            if sims[i] >= join:
                sid, n = rows[i]
                v = self._vec(task)
                if v is not None:                               # move the centroid toward this example
                    p = self._pos(sid)
                    old = self._index()["E8"][p].astype(np.float32) * self._index()["Esc"][p]
                    cen = old * n + v
                    cen /= np.linalg.norm(cen) + 1e-9
                    blob = quantize(cen)
                    self.db.execute("UPDATE memories SET emb = ?, n = n + 1, t = ? WHERE id = ?", (blob, now, sid))
                    self._index()["E8"][p] = np.frombuffer(blob[2:], np.int8)
                    self._index()["Esc"][p] = float(np.frombuffer(blob[:2], np.float16)[0])
                else:
                    self.db.execute("UPDATE memories SET n = n + 1, t = ? WHERE id = ?", (now, sid))
                self._cache_set(sid, t=now)
                return sid
        label = f"Situation: {task}"
        while self.db.execute("SELECT 1 FROM memories WHERE text = ?", (label,)).fetchone():
            label += " ."
        res = self._store(label, layer="semantic", imp=0.5, now=now, task=task, n=1.0,
                          check_beliefs=False, source="situation")
        return res["id"]

    def _update_rule(self, mid, reward, now, decay=0.85):
        """Exponentially-weighted evidence: each new outcome discounts the old ones by `decay`."""
        n, rew, sit, action = self.db.execute("SELECT n, reward, situation, action FROM memories WHERE id = ?",
                                              (mid,)).fetchone()
        n, rew = n * decay + 1.0, rew * decay + reward
        rate = (rew / n + 1) / 2
        text = f"situation {sit}: {action} works {rate:.0%} (evidence {n:.1f})"
        while self.db.execute("SELECT 1 FROM memories WHERE text = ? AND id != ?", (text, mid)).fetchone():
            text += " ."
        i = self._pos(mid)
        S = float(self._cache["S"][i]) * 1.1                    # rules in use stay strong
        self.db.execute("UPDATE memories SET n = ?, reward = ?, text = ?, t = ?, S = ? WHERE id = ?",
                        (n, rew, text, now, S, mid))
        self.db.commit()
        self._cache_set(mid, t=now, S=S)

    def advise(self, task, k=3, now=None, situations=3, temperature=0.15):
        """What has worked for tasks like this? Returns {"try": [...], "avoid": [...], "situations": [...]}, each
        item {"action", "value" (-1..1), "evidence"}. Blends the rules of the nearest situations, weighted by how
        close each is relative to the closest (absolute thresholds don't separate task types well) and by
        how much evidence each rule has."""
        now = _now() if now is None else now
        rows = self._situations()
        if not rows:
            return {"try": [], "avoid": [], "situations": []}
        sims = self._situation_sims(task, rows)
        order = np.argsort(-sims)[:situations]
        top = float(sims[order[0]])
        if top < 0.3:
            return {"try": [], "avoid": [], "situations": []}
        # the nearest situation's own evidence decides; neighbours only inform actions it has not tried,
        # at a discount (their rules are about related but different kinds of task)
        agg, own = {}, set()
        for rank, i in enumerate(order):
            sid = rows[i][0]
            w_sit = float(np.exp((sims[i] - top) / temperature)) * (1.0 if rank == 0 else 0.5)
            for act, n, rew in self.db.execute("SELECT action, n, reward FROM memories WHERE situation = ? AND "
                                               "layer = 'procedural'", (sid,)):
                if rank == 0:
                    own.add(act)
                elif act in own:
                    continue
                w = w_sit * min(n, 3.0)
                a = agg.setdefault(act, [0.0, 0.0])
                a[0] += w * (rew / n if n else 0.0)
                a[1] += w
        scored = [{"action": a, "value": round(s_ / wt, 3), "evidence": round(wt, 2)} for a, (s_, wt) in agg.items()
                  if wt > 0.05]
        scored.sort(key=lambda x: (-x["value"], -x["evidence"]))
        return {"try": [x for x in scored if x["value"] > 0][:k], "avoid": [x for x in scored if x["value"] < 0][:k],
                "situations": [int(rows[i][0]) for i in order]}

    def prepare(self, task, max_chars=1200, now=None):
        """A compact briefing for an LLM prompt: what to try, what to avoid, relevant how-tos and facts."""
        adv = self.advise(task, now=now)
        lines = []
        if adv["try"]:
            lines.append("Worked before: " + "; ".join(f"{x['action']} ({x['value']:+.2f})" for x in adv["try"]))
        if adv["avoid"]:
            lines.append("Failed before: " + "; ".join(f"{x['action']} ({x['value']:+.2f})" for x in adv["avoid"]))
        seen = set()
        for h in self.recall(task, k=5, associated=1, now=now, strengthen=False):
            if h["source"] in ("experience", "situation", "learning"):
                continue                                          # already summarised in the advice above
            lines.append(f"- {h['text']}")
            seen.add(h["id"])
        for mid, text in self.standing_rules(now=now):            # habits apply to every task
            if mid not in seen:
                lines.append(f"- {text}")
        out, size = [], 0
        for line in lines:
            if size + len(line) > max_chars:
                break
            out.append(line)
            size += len(line) + 1
        return "\n".join(out)

    def remember_example(self, task, example, now=None):
        """Store a worked example (e.g. code that passed) under its task, so similar tasks can reuse it."""
        now = _now() if now is None else now
        text = " ".join(example.split()) if len(example) < 80 else example.strip()
        row = self.db.execute("SELECT id FROM memories WHERE text = ?", (text,)).fetchone()
        if row:
            return self._reinforce(row[0], now)
        return self._store(text, layer="procedural", imp=0.5, now=now, task=" ".join(task.split()),
                           check_beliefs=False, source="example")

    def examples(self, task, k=2, now=None, min_sim=0.3):
        """Worked examples whose task resembles `task`, best first: similarity weighted by strength, which
        feedback() raises when an example helped and lowers when it misled. Returns [(id, task, example)]."""
        now = _now() if now is None else now
        rows = self.db.execute("SELECT id, task, text FROM memories WHERE source = 'example' AND superseded_by IS NULL"
                               ).fetchall()
        if not rows:
            return []
        pos = np.array([self._pos(r[0]) for r in rows])
        v = self._vec(task)
        if v is not None:
            sim = self._sem(self._cos(v, pos))
        else:
            q = self.coder(task)
            sim = np.array([len(np.intersect1d(q, self._index()["codes"][p])) / max(1, len(q)) for p in pos])
        R = self._strength(now)[pos]
        score = sim * (0.4 + 0.6 * R)
        return [rows[i] for i in np.argsort(-score)[:k] if sim[i] >= min_sim]

    def feedback(self, memory_ids, reward, now=None):
        """Tell the memory how things went after using these memories (reward -1..1). Helpful memories get
        stronger; misleading ones get weaker, and a memory that misled twice without ever helping is retired
        (kept only as history)."""
        now = _now() if now is None else now
        for mid in memory_ids:
            i = self._pos(mid)
            if i is None:
                continue
            row = self.db.execute("SELECT meta FROM memories WHERE id = ?", (mid,)).fetchone()
            meta = json.loads(row[0]) if row and row[0] else {}
            key = "helped" if reward > 0 else "misled"
            for k_ in ("helped", "misled"):                     # recent outcomes count most (decay 0.7)
                meta[k_] = round(meta.get(k_, 0) * 0.7, 4)
            meta[key] = meta.get(key, 0) + 1
            S = float(self._cache["S"][i]) * (1.5 if reward > 0 else 0.4)
            self.db.execute("UPDATE memories SET S = ?, t = ?, meta = ? WHERE id = ?",
                            (S, now if reward > 0 else float(self._cache["t"][i]), json.dumps(meta), mid))
            self._cache_set(mid, S=S, t=now if reward > 0 else float(self._cache["t"][i]))
            if meta.get("misled", 0) >= 1.5 and meta.get("misled", 0) > meta.get("helped", 0):
                self.db.execute("UPDATE memories SET superseded_by = -1 WHERE id = ?", (mid,))
                self._cache_set(mid, sup=True)
        self.db.commit()

    # ---------------------------------------------------------------- reflexes: error -> verified fix -> habit
    @staticmethod
    def error_signature(error):
        """Normalise an error message so the same kind of error matches: literals and numbers become
        placeholders."""
        e = (error or "").strip().splitlines()[-1] if error else ""
        e = re.sub(r"'[^']*'|\"[^\"]*\"", "<s>", e)
        e = re.sub(r"\b\d+(\.\d+)?\b", "<n>", e)
        return " ".join(e.split())[:160]

    @staticmethod
    def canonical(line):
        """A fix line with its literals abstracted, so the same fix written with different values matches."""
        line = re.sub(r"'[^']*'|\"[^\"]*\"", "<s>", line.strip())
        return re.sub(r"\b\d+(\.\d+)?\b", "<n>", " ".join(line.split()))

    def remember_fix(self, error, fix_lines, now=None):
        """Remember that `fix_lines` (verified: the code passed after adding them) fixed `error`. The same fix
        written with different literal values counts as the same memory, so repeated fixes become habits."""
        now = _now() if now is None else now
        sig = self.error_signature(error)
        lines = [l.strip() for l in fix_lines if l.strip()]
        if not sig or not lines:
            return None
        text = f"When the error is '{sig}': add {' ; '.join(self.canonical(l) for l in lines)}"
        row = self.db.execute("SELECT id, meta FROM memories WHERE text = ?", (text,)).fetchone()
        if row:
            meta = json.loads(row[1] or "{}")
            meta["lines"] = lines                                # keep the latest concrete working version
            self.db.execute("UPDATE memories SET meta = ? WHERE id = ?", (json.dumps(meta), row[0]))
            out = self._reinforce(row[0], now)
            self.feedback([row[0]], 1, now=now)
            return out
        return self._store(text, layer="procedural", imp=0.6, now=now, task=sig, check_beliefs=False, source="fix",
                           meta={"lines": lines, "helped": 1})

    def known_fixes(self, k=3, now=None):
        """The strongest verified fixes, for showing up front: [(error signature, lines)]."""
        now = _now() if now is None else now
        rows = self.db.execute("SELECT id, task, meta FROM memories WHERE source = 'fix' AND superseded_by IS NULL"
                               ).fetchall()
        R = self._strength(now)
        def value(r):
            m = json.loads(r[2] or "{}")
            return R[self._pos(r[0])] * (1 + m.get("helped", 0) - m.get("misled", 0))
        return [(r[1], json.loads(r[2])["lines"]) for r in sorted(rows, key=lambda r: -value(r))[:k]
                if value(r) > 0]

    def fixes_for(self, error, k=2, now=None):
        """Remembered fixes for this kind of error, best first: [(id, lines)]. Exact signature first, then
        similar errors."""
        now = _now() if now is None else now
        sig = self.error_signature(error)
        rows = self.db.execute("SELECT id, task, meta FROM memories WHERE source = 'fix' AND superseded_by IS NULL"
                               ).fetchall()
        if not rows or not sig:
            return []
        pos = np.array([self._pos(r[0]) for r in rows])
        v = self._vec(sig)
        sim = self._sem(self._cos(v, pos)) if v is not None else np.zeros(len(rows))
        exact = np.array([r[1] == sig for r in rows], float)
        R = self._strength(now)[pos]
        score = (exact * 2 + sim) * (0.4 + 0.6 * R)
        return [(rows[i][0], json.loads(rows[i][2])["lines"]) for i in np.argsort(-score)[:k]
                if exact[i] or sim[i] >= 1.0]

    def habits(self, min_helped=1.5):
        """Fixes that have worked repeatedly and never been retired: apply them proactively. [(id, lines)]"""
        out = []
        for mid, meta in self.db.execute("SELECT id, meta FROM memories WHERE source = 'fix' AND "
                                         "superseded_by IS NULL").fetchall():
            m = json.loads(meta or "{}")
            if m.get("helped", 0) >= min_helped and m.get("helped", 0) > m.get("misled", 0):
                out.append((mid, m["lines"]))
        return out

    def standing_rules(self, k=5, now=None):
        """The strongest general how-to rules (procedural memories reinforced most), applied to every task the
        way habits are, whether or not the task looks similar to where they were learned."""
        now = _now() if now is None else now
        rows = self.db.execute("SELECT id, text, hits, imp FROM memories WHERE layer = 'procedural' AND action IS NULL "
                               "AND superseded_by IS NULL").fetchall()
        if not rows:
            return []
        R = self._strength(now)
        scored = sorted(rows, key=lambda r: -(R[self._pos(r[0])] * (1 + r[2]) * (1 + r[3])))
        return [(r[0], r[1]) for r in scored[:k]]

    # ================================================================ read path
    def focus(self, text):
        """Put something into working memory without storing it (sets the context for recall)."""
        self.working.append((None, self._vec(text)))

    def recall(self, cue, k=5, associated=2, now=None, include_history=False, strengthen=True):
        """Recall memories for a cue (keywords, a question, a fragment). Returns a list of dicts with
        id, text, layer, score, why ('match' | 'association' | 'entity'), strength, when, superseded_by."""
        now = _now() if now is None else now
        code = self.coder(cue)
        c = self._index()
        if not len(c["ids"]) or not len(code):
            return []
        emb = self._vec(cue)
        rel, _ = self._relevance(code, emb)
        R = self._strength(now)
        base = rel * (0.85 + 0.15 * R)
        score = base.copy()

        # entities named in the cue activate their memories (this can create a match)
        eligible = base >= self.min_score
        ent_ids = self._entities_in(cue)
        if ent_ids:
            mem = {r[0] for r in self.db.execute(
                f"SELECT memory_id FROM mentions WHERE entity_id IN ({','.join('?' * len(ent_ids))})", ent_ids)}
            boost = np.isin(c["ids"], list(mem))
            score = score + 0.15 * boost
            eligible |= boost

        # context: working memory re-ranks candidates (tips near-ties) but cannot create a match
        ctx = [e for _, e in self.working if e is not None]
        if ctx and self.embed is not None and c["E8"].shape[1]:
            v = np.mean(ctx, axis=0)
            v /= np.linalg.norm(v) + 1e-9
            score = score + 0.16 * np.clip(self._sem(self._cos(v)) + 0.2, 0, None)

        if not include_history:
            eligible &= ~c["sup"]                              # superseded beliefs only appear as history
        order = np.argsort(-score)
        direct = [int(i) for i in order if eligible[i]][:k]
        out, seen = [], set()
        for i in direct:
            out.append(self._card(int(c["ids"][i]), float(score[i]), "match", float(R[i])))
            seen.add(int(c["ids"][i]))

        # spreading activation: links and shared entities, two hops, decaying
        if associated and out:
            act = {}
            frontier = [(h["id"], h["score"]) for h in out]
            for hop in range(2):
                nxt = []
                for mid, a in frontier:
                    for nid, w in self._neighbours(mid):
                        v = a * w * 0.6
                        if nid not in seen and v > act.get(nid, (0, ""))[0]:
                            act[nid] = (v, "association")
                            nxt.append((nid, v))
                    for nid in self._co_mentioned(mid):
                        v = a * 0.5
                        if nid not in seen and v > act.get(nid, (0, ""))[0]:
                            act[nid] = (v, "entity")
                            nxt.append((nid, v))
                frontier = nxt
            added = 0
            for nid, (a, why) in sorted(act.items(), key=lambda kv: -kv[1][0]):
                if added >= associated or a < 0.1:
                    break
                i = self._pos(nid)
                if i is None or (c["sup"][i] and not include_history):
                    continue
                out.append(self._card(nid, float(a), why, float(R[i])))
                added += 1

        # reconsolidation: strengthen the top hit (spacing effect), wire together what fired together
        if strengthen and direct:
            i0 = direct[0]
            mid0 = int(c["ids"][i0])
            S = float(c["S"][i0] * (1 + 2.0 * (1 - R[i0])))
            self.db.execute("UPDATE memories SET S = ?, t = ?, hits = hits + 1 WHERE id = ?", (S, now, mid0))
            for h in out[1:len(direct)]:
                self._link(mid0, h["id"], 0.3)
            self.db.commit()
            self._cache_set(mid0, S=S, t=now)
            if self.embed is not None and c["E8"].shape[1]:
                self.working.append((mid0, c["E8"][i0].astype(np.float32) * c["Esc"][i0]))
        return out

    def _card(self, mid, score, why, strength):
        layer, text, when, sources, sup, cons, src, sit = self.db.execute(
            "SELECT layer, text, when_text, sources, superseded_by, consolidated_into, source, situation FROM memories "
            "WHERE id = ?", (mid,)).fetchone()
        if src == "learning" and sit is not None:              # rules are stored compactly; render them readably
            label = self.db.execute("SELECT task FROM memories WHERE id = ?", (sit,)).fetchone()
            if label:
                text = text.replace(f"situation {sit}:", f"For situations like '{label[0]}':", 1)
        return {"id": mid, "text": text, "layer": layer, "score": round(score, 4), "why": why,
                "strength": round(strength, 4), "when": when, "sources": sources, "source": src,
                "superseded_by": sup, "consolidated_into": cons}

    def _neighbours(self, mid):
        return self.db.execute("SELECT b, w FROM links WHERE a = ? UNION ALL SELECT a, w FROM links WHERE b = ?",
                               (mid, mid)).fetchall()

    def _co_mentioned(self, mid, limit=10):
        return [r[0] for r in self.db.execute(
            "SELECT DISTINCT m2.memory_id FROM mentions m1 JOIN mentions m2 ON m1.entity_id = m2.entity_id "
            "WHERE m1.memory_id = ? AND m2.memory_id != ? LIMIT ?", (mid, mid, limit))]

    def _entities_in(self, text):
        low = f" {' '.join(words(text))} "
        return [eid for eid, name in self.db.execute("SELECT id, name FROM entities")
                if f" {' '.join(words(name))} " in low and words(name)]

    # ================================================================ introspection
    def about(self, entity, now=None, include_history=False):
        """Everything the brain holds about an entity, strongest first."""
        now = _now() if now is None else now
        rows = self.db.execute(
            "SELECT m.id FROM memories m JOIN mentions x ON x.memory_id = m.id JOIN entities e ON e.id = x.entity_id "
            "WHERE e.name = ?" + ("" if include_history else " AND m.superseded_by IS NULL"),
            (entity.strip().lower(),)).fetchall()
        R = self._strength(now)
        cards = [self._card(r[0], 1.0, "entity", float(R[self._pos(r[0])])) for r in rows]
        return sorted(cards, key=lambda h: (h["layer"] == "episodic", -h["strength"]))

    def history(self, memory_id):
        """The chain of beliefs that led to this memory (oldest first)."""
        chain, cur = [], memory_id
        while cur is not None:
            prev = self.db.execute("SELECT id FROM memories WHERE superseded_by = ?", (cur,)).fetchone()
            chain.append(cur)
            cur = prev[0] if prev else None
        return [self._card(m, 1.0, "history", 0.0) for m in reversed(chain)]

    def forget(self, memory_id):
        return self._delete([memory_id]) > 0

    def stats(self):
        by_layer = dict(self.db.execute("SELECT layer, COUNT(*) FROM memories GROUP BY layer").fetchall())
        q = lambda s: self.db.execute(s).fetchone()[0]
        c = self._index()
        index_bytes = sum(c[k].nbytes for k in ("ids", "indptr", "indices", "E8", "Esc", "t", "S", "imp", "layer",
                                                "sup", "cons"))
        return {"memories": sum(by_layer.values()), "by_layer": by_layer,
                "superseded": q("SELECT COUNT(*) FROM memories WHERE superseded_by IS NOT NULL"),
                "consolidated": q("SELECT COUNT(*) FROM memories WHERE consolidated_into IS NOT NULL"),
                "trials": q("SELECT COUNT(*) FROM memories WHERE action IS NOT NULL AND layer = 'episodic'"),
                "situations": q("SELECT COUNT(*) FROM memories WHERE source = 'situation'"),
                "rules": q("SELECT COUNT(*) FROM memories WHERE action IS NOT NULL AND layer = 'procedural'"),
                "entities": q("SELECT COUNT(*) FROM entities"), "links": q("SELECT COUNT(*) FROM links"),
                "bytes": self.bytes, "budget": self.budget, "index_ram_bytes": int(index_bytes),
                "process_ram_mb": rss_mb(), "embedder": getattr(self.embed, "name", "none"),
                "mind": self.mind.name, "working_memory": len(self.working), "path": self.path}

    # ================================================================ sleep
    def sleep(self, now=None, max_strength=0.5):
        """Offline maintenance: turn trials into rules, consolidate weak episodes into facts, forget to budget,
        prune weak links."""
        now = _now() if now is None else now
        facts = self._consolidate(now, max_strength)
        dropped = self._forget_to_budget(now)
        self.db.execute("UPDATE links SET w = w * 0.95")
        pruned = self.db.execute("DELETE FROM links WHERE w < 0.05").rowcount
        self.db.commit()
        self.working.clear()
        return {"facts_created": facts, "forgotten": dropped, "links_pruned": pruned}

    def _consolidate(self, now, max_strength):
        c = self._index()
        if not len(c["ids"]):
            return []
        R = self._strength(now)
        trial = np.zeros(len(c["ids"]), bool)
        tids = {r[0] for r in self.db.execute("SELECT id FROM memories WHERE action IS NOT NULL OR situation IS NOT NULL "
                                                  "OR source = 'situation'")}
        if tids:
            trial = np.isin(c["ids"], list(tids))
        pool = np.flatnonzero((R < max_strength) & (c["layer"] == 0) & ~c["cons"] & ~c["sup"] & ~trial)
        if not len(pool):
            return []
        if self.embed is not None and c["E8"].shape[1]:
            E = c["E8"][pool].astype(np.float32) * c["Esc"][pool][:, None]
            sim = self._sem(E @ E.T)
            threshold = 1.0                                       # at least as close as a typical paraphrase
        else:
            sets = [set(c["codes"][p].tolist()) for p in pool]
            sim = np.array([[len(a & b) / max(1, len(a | b)) for b in sets] for a in sets])
            threshold = 0.6
        used, created = set(), []
        for a in range(len(pool)):
            if a in used:
                continue
            members = [b for b in range(len(pool)) if b not in used and sim[a, b] >= threshold]
            if len(members) < self.min_cluster:
                continue
            used.update(members)
            mids = [int(c["ids"][pool[b]]) for b in members]
            texts = [self.db.execute("SELECT text FROM memories WHERE id = ?", (m,)).fetchone()[0] for m in mids]
            fact = self.mind.summarize(texts)
            if fact in texts:                                   # extractive gist: tag it so it is its own memory
                fact = f"{fact} (seen {len(texts)} times)"
            imp = float(max(c["imp"][pool[b]] for b in members))
            res = self.perceive(fact, importance=imp, layer="semantic", source="sleep", now=now, force=True)
            fid = res["id"]
            self.db.execute("UPDATE memories SET sources = ? WHERE id = ?", (len(mids), fid))
            for m in mids:
                self.db.execute("UPDATE memories SET consolidated_into = ?, S = S * 0.5 WHERE id = ?", (fid, m))
                self._link(fid, m, 0.8)
                i = self._pos(m)
                if i is not None:
                    self._cache_set(m, cons=True, S=float(self._cache["S"][i]) * 0.5)
                for (eid,) in self.db.execute("SELECT entity_id FROM mentions WHERE memory_id = ?", (m,)).fetchall():
                    self.db.execute("INSERT OR IGNORE INTO mentions VALUES (?, ?)", (fid, eid))
            self.db.commit()
            created.append(fid)
        return created

    def _forget_to_budget(self, now):
        dropped = 0
        while self.bytes > self.budget:
            c = self._index()
            if not len(c["ids"]):
                break
            keep = np.array([LAYER_KEEP[LAYERS[l]] for l in c["layer"]])
            prio = (self._strength(now) * (1 + c["imp"]) * keep * np.where(c["sup"], 0.3, 1)
                    * np.where(c["cons"], 0.5, 1))
            per = max(1, self.bytes // len(c["ids"]))
            n = max(1, min(len(c["ids"]), int((self.bytes - self.budget) // per) + 1))
            dropped += self._delete([int(c["ids"][i]) for i in np.argsort(prio)[:n]])
        return dropped

    def _delete(self, mids):
        mids = list(mids)
        for m in list(mids):                                     # a forgotten situation takes its rules with it
            mids += [r[0] for r in self.db.execute("SELECT id FROM memories WHERE situation = ? AND layer = "
                                                   "'procedural'", (m,)) if r[0] not in mids]
        n = 0
        for m in mids:
            n += self.db.execute("DELETE FROM memories WHERE id = ?", (m,)).rowcount
            self.db.execute("DELETE FROM links WHERE a = ? OR b = ?", (m, m))
            self.db.execute("DELETE FROM mentions WHERE memory_id = ?", (m,))
            self.db.execute("UPDATE memories SET consolidated_into = NULL WHERE consolidated_into = ?", (m,))
        self.db.execute("DELETE FROM entities WHERE id NOT IN (SELECT entity_id FROM mentions)")
        self.db.commit()
        self._cache = None                                       # flags of other rows may have changed
        self.bytes = self._sql_bytes()
        return n

    def _sql_bytes(self):
        return self.db.execute("SELECT COALESCE(SUM(LENGTH(CAST(text AS BLOB)) + LENGTH(code) + COALESCE(LENGTH(emb), 0)"
                               " + COALESCE(LENGTH(task), 0) + COALESCE(LENGTH(action), 0) + 80), 0) "
                               "FROM memories").fetchone()[0]
