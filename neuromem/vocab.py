"""Domain vocabulary: learned word vectors that teach a static embedding model your team's jargon.

The base model (potion / model2vec) embeds a sentence as the mean of its token vectors. Jargon like "OOMKilled"
splits into meaningless word pieces. A DomainVocab adds one trainable vector per term: when the term appears in
a sentence it contributes that vector instead of its pieces. Base vectors are frozen, so text without the terms
embeds exactly as before.

Training: each vector starts as the embedding of the term's plain definition, then a contrastive (InfoNCE)
objective pulls sentences using the term toward plain paraphrases and away from the rest of the batch.
"""
import json
import re

import numpy as np



class DomainEmbedder:
    """Wraps a model2vec-backed Embedder from neuromem.embed; same interface (call, one, dim, lo, hi, name)."""

    def __init__(self, base, terms=None, weight=2.0):
        self.base = base
        self.static = base.static                              # model2vec StaticModel
        self.E = np.asarray(self.static.embedding, np.float32)
        self.tok = self.static.tokenizer
        self.terms = {}                                        # lowercase term -> vector
        self.weight = weight                                   # a term counts as `weight` tokens
        self.dim, self.lo, self.hi = base.dim, base.lo, base.hi
        self.name = base.name + f"+vocab{len(terms or {})}"
        for t, v in (terms or {}).items():
            self.terms[t.lower()] = np.asarray(v, np.float32)

    # ------------------------------------------------------------------ encoding
    def _term_re(self):
        if getattr(self, "_re_key", None) != len(self.terms):
            alts = sorted(self.terms, key=len, reverse=True)
            self._re = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(re.escape(t) for t in alts) + r")(?![A-Za-z0-9])",
                                  re.I) if alts else None
            self._re_key = len(self.terms)
        return self._re

    def _parts(self, text):
        """(sum of base token vectors, number of base tokens, list of terms present). Terms may be phrases."""
        found = []
        rx = self._term_re()
        if rx is not None:
            found = [m.group(1).lower() for m in rx.finditer(text)]
            text = rx.sub(" ", text)
        s = np.zeros(self.E.shape[1], np.float32)
        n = 0
        ids = self.tok.encode(text, add_special_tokens=False).ids if text.strip() else []
        if ids:
            s = self.E[ids].sum(axis=0)
            n = len(ids)
        return s, n, found

    def raw(self, text):
        s, n, found = self._parts(text)
        for t in found:
            s = s + self.weight * self.terms[t]
        return s

    def __call__(self, texts):
        V = np.array([self.raw(t) for t in texts], np.float32)
        return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)

    def one(self, text):
        return self([text])[0]

    # ------------------------------------------------------------------ training
    def add_terms(self, glossary):
        """glossary: {term: plain definition}. Initialise each term vector from its definition."""
        for term, definition in glossary.items():
            s = self.base.static.encode([definition], normalize=False)[0].astype(np.float32)
            ref = np.linalg.norm(self.E, axis=1).mean()
            self.terms[term.lower()] = s / (np.linalg.norm(s) + 1e-9) * ref

    def train(self, pairs, epochs=60, lr=0.05, tau=0.05, batch=32, seed=0):
        """pairs: list of (sentence using jargon, plain paraphrase). Only term vectors are updated (Adam)."""
        r = np.random.default_rng(seed)
        m = {t: np.zeros_like(v) for t, v in self.terms.items()}
        v2 = {t: np.zeros_like(v) for t, v in self.terms.items()}
        step, history = 0, []
        for ep in range(epochs):
            order = r.permutation(len(pairs))
            tot = 0.0
            for b in range(0, len(order), batch):
                idx = order[b:b + batch]
                if len(idx) < 2:
                    continue
                A = [self._parts(pairs[i][0]) for i in idx]
                P = [self._parts(pairs[i][1]) for i in idx]
                sa = np.array([s + sum(self.weight * self.terms[t] for t in f) for s, _, f in A])
                sp = np.array([s + sum(self.weight * self.terms[t] for t in f) for s, _, f in P])
                na, np_ = np.linalg.norm(sa, axis=1, keepdims=True), np.linalg.norm(sp, axis=1, keepdims=True)
                ea, ep_ = sa / na, sp / np_
                logits = ea @ ep_.T / tau
                logits -= logits.max(axis=1, keepdims=True)
                prob = np.exp(logits)
                prob /= prob.sum(axis=1, keepdims=True)
                k = len(idx)
                tot += float(-np.log(prob[np.arange(k), np.arange(k)] + 1e-12).sum())
                g = prob.copy()
                g[np.arange(k), np.arange(k)] -= 1.0                  # dL/dlogits
                dea = g @ ep_ / tau / k
                dep = g.T @ ea / tau / k
                dsa = (dea - (dea * ea).sum(1, keepdims=True) * ea) / na
                dsp = (dep - (dep * ep_).sum(1, keepdims=True) * ep_) / np_
                grads = {}
                for (s, n, f), d in list(zip(A, dsa)) + list(zip(P, dsp)):
                    for t in f:
                        grads[t] = grads.get(t, 0) + self.weight * d
                step += 1
                for t, gt in grads.items():                          # Adam
                    m[t] = 0.9 * m[t] + 0.1 * gt
                    v2[t] = 0.999 * v2[t] + 0.001 * gt * gt
                    mh, vh = m[t] / (1 - 0.9 ** step), v2[t] / (1 - 0.999 ** step)
                    self.terms[t] -= lr * mh / (np.sqrt(vh) + 1e-8)
            history.append(tot / len(pairs))
        return history

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        json.dump({t: v.round(5).tolist() for t, v in self.terms.items()}, open(path, "w"))

    @classmethod
    def load(cls, base, path):
        return cls(base, json.load(open(path)))


def train_vocab(base, glossary, pairs, out_path, **kw):
    """glossary {term: plain definition}, pairs [(jargon sentence, plain paraphrase)] -> saved vectors."""
    d = DomainEmbedder(base)
    d.add_terms(glossary)
    if pairs:
        d.train(pairs, **kw)
    d.save(out_path)
    return d
