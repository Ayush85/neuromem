"""The Mind understands text: entities, time, layer, importance; writes gists; detects contradictions.

HeuristicMind works offline with rules. ClaudeMind uses the Claude API for real understanding.
Both implement: extract(text) -> dict, summarize(texts) -> str, contradicts(old, new) -> bool.
"""
import json
import re

from .codes import words

EMPHASIS = {"important": 1.0, "critical": 1.0, "remember": 0.8, "deadline": 0.8, "password": 0.8, "urgent": 0.8,
            "never": 0.5, "always": 0.5, "must": 0.5, "decided": 0.6, "decision": 0.6, "prefer": 0.4,
            "prefers": 0.4, "note": 0.3, "warning": 0.6, "secret": 0.6}
PROCEDURAL = re.compile(r"^\s*(how to|to [a-z]+\b.*?,|steps?:|first,|run |install |click |open |deploy |restart |"
                        r"set |add |call |use \S+ to)", re.I)
WHEN = re.compile(r"\b(\d{4}-\d{2}-\d{2}|today|tomorrow|yesterday|tonight|(?:next|last) (?:week|month|year)|"
                  r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
                  r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2})\b", re.I)
NEGATION = re.compile(r"\b(not|no|never|isn't|aren't|doesn't|don't|won't|cannot|can't|no longer)\b", re.I)
CHANGE = re.compile(r"\b(now|no longer|changed|moved|switched|instead|anymore|replaced|updated|renamed)\b", re.I)
CALENDAR = set("""monday tuesday wednesday thursday friday saturday sunday january february march april may june july
august september october november december jan feb mar apr jun jul aug sep sept oct nov dec today tomorrow
yesterday""".split())
EVENT = re.compile(r"\b(was|were|did|had|happened|again|ago|met|went|said|told|saw|got|came|left|[a-z]{3,}ed)\b", re.I)
STATE = re.compile(r"\b(is|are|has|have|uses|runs|owns|lives|works|prefers|likes|costs|means|requires|lives)\b", re.I)
NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
COMMON_START = {"the", "a", "an", "we", "i", "our", "it", "this", "that", "there", "to", "how", "when", "if",
                "run", "use", "set", "add", "open", "first", "remember", "note", "always", "never", "please",
                "deploy", "restart", "install", "click", "call", "check", "make", "yesterday", "today", "tomorrow",
                "last", "next", "after", "before", "then", "also", "but", "and", "so", "for", "in", "on", "at"}


def emphasis(text):
    s = sum(v for w, v in EMPHASIS.items() if re.search(rf"\b{w}\b", text, re.I))
    return min(1.0, s + 0.2 * text.count("!"))


class HeuristicMind:
    name = "heuristic"

    def extract(self, text):
        ents = []
        for m in re.finditer(r"(?:@[\w.-]+|#[\w-]+|\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*|\b[A-Z]{2,}[0-9]*\b)", text):
            name = m.group(0).strip()
            first = name.split()[0].lower()
            if m.start() == 0 or text[:m.start()].rstrip().endswith((".", "!", "?")):
                if first in COMMON_START or first in EMPHASIS:            # sentence-initial ordinary word
                    rest = name.split()[1:]
                    if not rest:
                        continue
                    name = " ".join(rest)
            if name.lower() in CALENDAR:
                continue
            ents.append({"name": name.lstrip("@#"), "kind": "tag" if m.group(0)[0] in "@#" else "thing"})
        seen, uniq = set(), []
        for e in ents:
            if e["name"].lower() not in seen:
                seen.add(e["name"].lower())
                uniq.append(e)
        w = WHEN.search(text)
        if w or EVENT.search(text):
            layer = "episodic"                                    # something that happened
        elif PROCEDURAL.search(text):
            layer = "procedural"
        elif not STATE.search(text):
            layer = "episodic"                                    # something that happened
        else:
            layer = "semantic"                                    # a standing fact or state
        return {"entities": uniq, "when": w.group(0) if w else "", "layer": layer, "importance": emphasis(text)}

    def summarize(self, texts):
        """Most central text: the one sharing the most words with the others."""
        sets = [set(words(t)) for t in texts]
        best = max(range(len(texts)), key=lambda i: sum(len(sets[i] & s) / max(1, len(sets[i] | s)) for s in sets))
        return texts[best]

    def contradicts(self, old, new):
        """True if NEW replaces OLD. Callers only ask about facts/states, or when NEW signals a change."""
        na, nb = set(NUMBER.findall(old)), set(NUMBER.findall(new))
        filler = set(words(" ".join(CHANGE.pattern.split("|")))) | {"now", "longer", "anymore", "currently"}
        wa = set(words(old)) - na - filler
        wb = set(words(new)) - nb - filler
        if not wa or not wb:
            return False
        same_subject = wa <= wb or wb <= wa          # one statement's words contained in the other's
        if not same_subject:
            return False
        if na and nb and na != nb:
            return True                              # same statement, different numbers
        if bool(NEGATION.search(old)) != bool(NEGATION.search(new)):
            return True                              # same statement, one negated
        return bool(CHANGE.search(new)) and not CHANGE.search(old) and wa != wb


class ClaudeMind(HeuristicMind):
    """Uses Claude for extraction, summaries and contradiction checks; falls back to heuristics on any failure."""
    name = "claude"

    def __init__(self, model="claude-opus-5-5", client=None):
        import anthropic
        self.anthropic = anthropic
        self.client = client or anthropic.Anthropic()
        self.model = model

    def _ask(self, prompt, schema):
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=4000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": prompt}],
        )
        if resp.stop_reason != "end_turn":
            raise RuntimeError(f"stop_reason={resp.stop_reason}")
        return json.loads(next(b.text for b in resp.content if b.type == "text"))

    def extract(self, text):
        schema = {"type": "object", "additionalProperties": False,
                  "required": ["entities", "when", "layer", "importance"],
                  "properties": {
                      "entities": {"type": "array", "items": {
                          "type": "object", "additionalProperties": False, "required": ["name", "kind"],
                          "properties": {"name": {"type": "string"},
                                         "kind": {"type": "string", "enum": ["person", "project", "place", "org",
                                                                             "tool", "thing", "tag"]}}}},
                      "when": {"type": "string"},
                      "layer": {"type": "string", "enum": ["episodic", "semantic", "procedural"]},
                      "importance": {"type": "number"}}}
        prompt = ("Analyse this memory for a long-term memory system.\n"
                  "entities: the specific people, projects, places, orgs, tools it is about (canonical names).\n"
                  "when: the time it refers to, verbatim, or empty string.\n"
                  "layer: episodic (an event / something that happened or was said), semantic (a general fact), "
                  "or procedural (how to do something).\n"
                  "importance: 0 (trivia) to 1 (critical to remember: decisions, deadlines, secrets, preferences).\n\n"
                  f"Memory: {text}")
        try:
            out = self._ask(prompt, schema)
            out["importance"] = max(0.0, min(1.0, float(out["importance"])))
            return out
        except Exception:
            return super().extract(text)

    def summarize(self, texts):
        schema = {"type": "object", "additionalProperties": False, "required": ["fact"],
                  "properties": {"fact": {"type": "string"}}}
        listing = "\n".join(f"- {t}" for t in texts)
        try:
            return self._ask("These related memories are being consolidated. Write the single general fact or "
                             "pattern they support, in one sentence, keeping specific names and numbers that "
                             f"recur.\n\n{listing}", schema)["fact"].strip() or super().summarize(texts)
        except Exception:
            return super().summarize(texts)

    def contradicts(self, old, new):
        schema = {"type": "object", "additionalProperties": False, "required": ["contradicts"],
                  "properties": {"contradicts": {"type": "boolean"}}}
        try:
            return bool(self._ask("Does the NEW memory contradict or replace the OLD one, so the OLD one should no "
                                  f"longer be believed?\n\nOLD: {old}\nNEW: {new}", schema)["contradicts"])
        except Exception:
            return super().contradicts(old, new)


def make_mind(kind="auto"):
    """'heuristic', 'claude', or 'auto' (Claude if the SDK and credentials are available)."""
    if not isinstance(kind, str):
        return kind
    if kind == "heuristic":
        return HeuristicMind()
    try:
        return ClaudeMind()
    except Exception:
        if kind == "claude":
            raise
        return HeuristicMind()
