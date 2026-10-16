"""Does training domain vocabulary vectors teach the model team jargon, without breaking anything else?

For each term and each HELD-OUT sentence frame: does a sentence using the jargon find its plain-language
paraphrase (about a different service) among the plain paraphrases of all terms? Also: how similar are they
(calibrated: 0 = unrelated, 1 = typical paraphrase), and does jargon start matching unrelated text?
Usage: python bench/vocab.py
"""
import random
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "bench")
from neuromem.embed import make_embedder    # noqa: E402
from neuromem.vocab import DomainEmbedder   # noqa: E402
from vocab_data import GLOSSARY, SERVICES, TEST_FRAMES, TRAIN_FRAMES, UNRELATED   # noqa: E402


def pairs(frames, r, n_per=6):
    out = []
    for term, (_, jar, plain) in GLOSSARY.items():
        for _ in range(n_per):
            s1, s2 = r.sample(SERVICES, 2)
            out.append((term, r.choice(frames).format(s=s1, x=jar), r.choice(frames).format(s=s2, x=plain)))
    return out


def evaluate(emb, test):
    terms = list(GLOSSARY)
    plain = {t: emb.one(f"the {SERVICES[0]} service {GLOSSARY[t][2]}") for t in terms}
    P = np.array([plain[t] for t in terms])
    U = emb(UNRELATED)
    hit, sims, fp = 0, [], 0
    for term, jar, par in test:
        a = emb.one(jar)
        hit += terms[int(np.argmax(P @ a))] == term
        sims.append((float(a @ emb.one(par)) - emb.lo) / (emb.hi - emb.lo))
        fp += float(((U @ a) - emb.lo).max()) / (emb.hi - emb.lo) >= 0.5      # meaningfully similar
    return hit / len(test), float(np.mean(sims)), fp / len(test)


if __name__ == "__main__":
    r = random.Random(0)
    train = pairs(TRAIN_FRAMES, r)
    test = pairs(TEST_FRAMES, random.Random(1), n_per=3)
    print(f"{len(GLOSSARY)} terms, {len(train)} training pairs, {len(test)} held-out pairs (unseen sentence frames)")
    print(f"{'model':<26}{'finds paraphrase':>17}{'similarity':>12}{'unrelated >= 0.5':>19}")
    for prof in ("tiny", "small"):
        base = make_embedder(prof)
        rows = [("base " + prof, base)]
        d = DomainEmbedder(base)
        d.add_terms({t: g[0] for t, g in GLOSSARY.items()})
        rows.append((prof + " + definitions", d))
        for name, e in rows:
            acc, sim, fp = evaluate(e, test)
            print(f"{name:<26}{acc:17.0%}{sim:12.2f}{fp:19.0%}")
        hist = d.train([(a, b) for _, a, b in train])
        acc, sim, fp = evaluate(d, test)
        print(f"{prof + ' + trained':<26}{acc:17.0%}{sim:12.2f}{fp:19.0%}   (loss {hist[0]:.2f} -> {hist[-1]:.2f})")
        same = all(np.allclose(base.one(u), d.one(u), atol=1e-5) for u in UNRELATED)
        print(f"{'':<26}text without jargon embeds exactly as before: {same}")
