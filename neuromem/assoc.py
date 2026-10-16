"""Memory stored in network weights: a Willshaw associative network over neuromem's sparse codes.

  store(id, code)   switch on synapses between the code's active neurons (auto-association, for pattern
                    completion) and from them to a sparse code for the id (hetero-association, for naming it)
  recall(cue_code)  complete the cue to the full pattern, map it to an id code, decode the id

RAM is fixed by the network size, not by how many memories are stored: N x N + M x N bits. As more patterns are
stored the weight matrices fill up and recall degrades; `fill` reports how full they are (~0.5 is the classic
capacity limit).
"""
import zlib

import numpy as np


def id_code(mid, units, k):
    """Deterministic sparse code for a memory id (k of `units` active)."""
    idx, j = set(), 0
    while len(idx) < k:
        idx.add(zlib.crc32(f"id{mid}#{j}".encode()) % units)
        j += 1
    return np.array(sorted(idx), np.int32)


class WillshawMemory:
    def __init__(self, neurons=2048, id_units=4096, id_k=12):
        self.N, self.M, self.k = neurons, id_units, id_k
        self.W = np.zeros((neurons, neurons), np.bool_)        # auto-association
        self.H = np.zeros((id_units, neurons), np.bool_)       # pattern -> id code
        self.count = 0
        self.total_active = 0

    def ram_bytes(self, packed=True):
        bits = self.N * self.N + self.M * self.N
        return bits // 8 if packed else bits

    def fill(self):
        return float(self.W.mean()), float(self.H.mean())

    def store(self, mid, code):
        code = np.asarray(code, np.int64)
        self.W[np.ix_(code, code)] = True
        self.H[np.ix_(id_code(mid, self.M, self.k), code)] = True
        self.count += 1
        self.total_active += len(code)

    def complete(self, cue, k=None, steps=3):
        """Pattern completion. First step: neurons connected to the most cue neurons. Then iterate, keeping the k
        neurons most strongly driven by the current pattern (k-winners-take-all), which removes neurons that
        belong to other memories sharing some of the cue."""
        cue = np.asarray(cue, np.int64)
        cue = cue[self.W[cue, cue]]                            # ignore cue neurons no stored memory uses
        if not len(cue):
            return cue
        h = self.W[:, cue].sum(axis=1)
        x = np.flatnonzero(h >= h.max())                       # best supported (tolerates stray cue neurons)
        k = k or max(len(cue), self.avg_k())
        for _ in range(steps):
            if len(x) <= k:
                break
            h = self.W[:, x].sum(axis=1).astype(np.float32)
            h[cue] += len(x)                                   # the cue itself stays clamped on
            x = np.argsort(-h)[:k]
        return np.sort(x)

    def avg_k(self):
        return int(round(self.total_active / max(1, self.count)))

    def recall(self, cue, candidates):
        """Best matching id among `candidates` (an iterable of ids), or None. The id is decoded from the graded
        output of the hetero-associative layer; candidates only enumerate valid ids."""
        x = self.complete(cue)
        if not len(x):
            return None, 0.0
        h = self.H[:, x].sum(axis=1).astype(np.float32)
        best, best_s = None, 0.0
        for mid in candidates:
            s = float(h[id_code(mid, self.M, self.k)].sum()) / (self.k * len(x))
            if s > best_s:
                best, best_s = mid, s
        return best, best_s
