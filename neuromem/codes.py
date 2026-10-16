"""Sparse neural codes: each content word fires a few fixed neurons out of 2,048 (fly-olfaction style).
A memory's code is the set of neurons its words fire, so a fragment of a memory overlaps its code."""
import re
import zlib

import numpy as np

STOP = set("""a an the of to and or in on for with is are was were be been by as at it its this that from can
if not will into than then so such these those which when where who what how all any each other may also only
one use used using new we you i he she they our your my me us them do does did have has had just about there
here get got some very more most out up down over""".split())

TOKEN = re.compile(r"[a-z0-9][a-z0-9_]*")


def words(text):
    return [w for w in TOKEN.findall(text.lower()) if w not in STOP and not (len(w) == 1 and not w.isdigit())]


class SparseCoder:
    def __init__(self, neurons=2048, per_word=6):
        self.neurons, self.per_word = neurons, per_word

    def __call__(self, text):
        idx = {zlib.crc32(f"{w}#{j}".encode()) % self.neurons for w in set(words(text)) for j in range(self.per_word)}
        return np.array(sorted(idx), np.int16)
