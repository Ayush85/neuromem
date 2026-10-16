"""Memory in weights vs explicit storage: recall accuracy and RAM as the number of stored memories grows.

Sentences are encoded with neuromem's sparse coder; cues keep a fraction of each sentence's words.
  explicit   stores every code; recall = memory whose code contains the most cue neurons (RAM grows with n)
  willshaw   stores memories only in binary synapses; recall = pattern completion + id decoding (RAM fixed)
Usage: python bench/assoc.py
"""
import json
import random
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from neuromem.assoc import WillshawMemory    # noqa: E402
from neuromem.codes import SparseCoder        # noqa: E402

D = list(dict.fromkeys(d["text"] for d in json.load(open("../smriti/corpus.json"))))


def cue_of(text, frac, r):
    w = text.split()
    return " ".join(r.sample(w, max(2, int(len(w) * frac))))


def explicit_recall(codes, cue):
    best, bs = None, -1
    for i, c in enumerate(codes):
        s = len(np.intersect1d(cue, c)) + 0.01 * len(np.intersect1d(cue, c)) / len(c)
        if s > bs:
            best, bs = i, s
    return best


if __name__ == "__main__":
    r = random.Random(0)
    texts = D[:]
    r.shuffle(texts)
    print(f"{len(texts)} distinct sentences; cue = 40% of the words; 300 test cues per row")
    print(f"{'neurons':>7} {'stored':>6} {'explicit':>9} {'willshaw':>9} {'W fill':>7} {'explicit RAM':>13} "
          f"{'willshaw RAM':>13}")
    for neurons in (2048, 8192):
        coder = SparseCoder(neurons=neurons)
        codes_all = [coder(t) for t in texts]
        for n in (250, 500, 1000, len(texts)):
            net = WillshawMemory(neurons=neurons, id_units=4096)
            for i in range(n):
                net.store(i, codes_all[i])
            rr = random.Random(1)
            test = rr.sample(range(n), min(300, n))
            ok_e = ok_w = 0
            t0 = time.time()
            for i in test:
                cue = coder(cue_of(texts[i], 0.4, rr))
                ok_e += explicit_recall(codes_all[:n], cue) == i
                ok_w += net.recall(cue, range(n))[0] == i
            ram_e = sum(c.nbytes for c in codes_all[:n])
            print(f"{neurons:>7} {n:>6} {ok_e / len(test):9.1%} {ok_w / len(test):9.1%} {net.fill()[0]:7.2f} "
                  f"{ram_e / 1024:10.0f} KB {net.ram_bytes() / 1024:10.0f} KB", flush=True)
