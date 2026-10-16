"""Semantic embedders, from small to full, each auto-calibrated.

  tiny   minishlab/potion-base-2M   static word vectors (model2vec), ~10 MB on disk, numpy only
  small  minishlab/potion-base-8M   static word vectors (model2vec), ~30 MB on disk, numpy only
  full   BAAI/bge-small-en-v1.5     transformer (fastembed / ONNX), ~130 MB on disk, ~240 MB RAM

Cosine scales differ wildly between models (bge: unrelated ~0.5; potion: unrelated ~0.05), so every embedder
is calibrated on built-in sentence pairs: `lo` = where unrelated text tops out, `hi` = typical paraphrase.
"""
import numpy as np

MODELS = {
    "tiny": ("model2vec", "minishlab/potion-base-2M"),
    "small": ("model2vec", "minishlab/potion-base-8M"),
    "full": ("fastembed", "BAAI/bge-small-en-v1.5"),
}

PARAPHRASES = [
    ("how often do we back up the database", "database backups run every night"),
    ("login tokens stop working after fifteen minutes", "JWT tokens expire after a quarter of an hour"),
    ("the website is very slow to load", "page load times are terrible"),
    ("the server ran out of memory and crashed", "the process was killed for exceeding its RAM limit"),
    ("who is responsible for the billing system", "the billing service owner"),
    ("the disk is completely full", "no space left on the device"),
    ("the certificate has expired", "the TLS cert is no longer valid"),
    ("I prefer written feedback to meetings", "send me comments in writing rather than scheduling a call"),
    ("the car needs servicing", "the automobile is due for maintenance"),
    ("the meeting moved to Thursday afternoon", "we rescheduled the meeting to Thursday after lunch"),
    ("rotate the api keys every quarter", "API credentials must be changed every three months"),
    ("the test fails randomly", "this test is flaky"),
]
UNRELATED = [
    ("how often do we back up the database", "the cafeteria serves pasta on fridays"),
    ("login tokens stop working after fifteen minutes", "the giraffe is the tallest animal"),
    ("the website is very slow to load", "my grandmother knits scarves"),
    ("the server ran out of memory and crashed", "we planted tomatoes in the garden"),
    ("who is responsible for the billing system", "the river flooded after the storm"),
    ("the disk is completely full", "she won the chess tournament"),
    ("the certificate has expired", "bake the bread for forty minutes"),
    ("I prefer written feedback to meetings", "the volcano erupted last century"),
    ("the car needs servicing", "the orchestra played a symphony"),
    ("the meeting moved to Thursday afternoon", "penguins live in the southern hemisphere"),
    ("rotate the api keys every quarter", "the museum has a dinosaur skeleton"),
    ("the test fails randomly", "a recipe for lemon cake"),
]

_CACHE = {}


class Embedder:
    def __init__(self, profile):
        backend, model = MODELS[profile]
        self.name = f"{profile}:{model}"
        if backend == "model2vec":
            from model2vec import StaticModel
            m = StaticModel.from_pretrained(model)
            self.static = m
            self._encode = lambda texts: m.encode(texts)
        else:
            from fastembed import TextEmbedding
            m = TextEmbedding(model)
            self._encode = lambda texts: np.array(list(m.embed(texts)))
        self.dim = len(self(["probe"])[0])
        self.lo, self.hi = self._calibrate()

    def __call__(self, texts):
        v = np.asarray(self._encode(list(texts)), np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    def one(self, text):
        return self([text])[0]

    def _calibrate(self):
        def cos(pairs):
            a, b = self([p[0] for p in pairs]), self([p[1] for p in pairs])
            return (a * b).sum(axis=1)
        unrelated, para = cos(UNRELATED), cos(PARAPHRASES)
        lo = float(np.max(unrelated)) + 0.02                     # just above the most similar unrelated pair
        hi = max(float(np.median(para)), lo + 0.1)
        return lo, hi


def make_embedder(profile="auto"):
    """profile: 'tiny' | 'small' | 'full' | 'none' | 'auto' (best installed: full, then small, then tiny)."""
    if profile == "none" or profile is None:
        return None
    if not isinstance(profile, str):
        return profile                                           # a ready Embedder-like object
    order = ["full", "small", "tiny"] if profile == "auto" else [profile]
    for p in order:
        if p not in _CACHE:
            try:
                _CACHE[p] = Embedder(p)
            except Exception:
                if profile != "auto":
                    raise
                continue
        return _CACHE[p]
    return None
