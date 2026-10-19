import argparse
import json
import os
import sys

from .brain import Brain


def show(hits):
    for h in hits:
        tag = {"match": "", "association": " ~linked", "entity": " ~same entity", "history": ""}[h["why"]]
        old = " (superseded)" if h["superseded_by"] else ""
        print(f"[{h['id']}] {h['layer'][:4]} {h['score']:.2f}{tag}{old}  {h['text']}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="neuromem", description="Brain-style long-term memory.")
    p.add_argument("--db", help="database path (default: $NEUROMEM_DB or ~/.neuromem/brain.db)")
    p.add_argument("--mind", default=None, choices=["heuristic", "claude"],
                   help="who understands text (default: $NEUROMEM_MIND or heuristic)")
    p.add_argument("--profile", default=None, choices=["tiny", "small", "full", "none", "auto"],
                   help="embedding model size (default: $NEUROMEM_PROFILE or auto)")
    p.add_argument("--budget", type=int, default=100_000_000, help="storage budget in bytes")
    p.add_argument("--vocab", default=None, help="trained domain vocabulary (from train-vocab); default $NEUROMEM_VOCAB")
    p.add_argument("--json", action="store_true", help="JSON output")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("perceive", help="take in an experience / fact / instruction")
    a.add_argument("text")
    a.add_argument("--importance", type=float)
    a.add_argument("--layer", choices=["episodic", "semantic", "procedural"])
    a.add_argument("--source")
    r = sub.add_parser("recall", help="recall by meaning, keywords or fragment")
    r.add_argument("cue")
    r.add_argument("-k", type=int, default=5)
    r.add_argument("--history", action="store_true", help="include superseded beliefs")
    e = sub.add_parser("about", help="everything known about an entity")
    e.add_argument("entity")
    h = sub.add_parser("history", help="how a belief changed over time")
    h.add_argument("id", type=int)
    f = sub.add_parser("forget")
    f.add_argument("id", type=int)
    x = sub.add_parser("experience", help="record an attempt and its outcome")
    x.add_argument("task")
    x.add_argument("action")
    x.add_argument("reward", type=float, help="-1 failed .. 1 worked")
    x.add_argument("--note")
    v = sub.add_parser("advise", help="what worked / failed for tasks like this")
    v.add_argument("task")
    pr = sub.add_parser("prepare", help="compact briefing for an LLM prompt")
    pr.add_argument("task")
    pr.add_argument("--max-chars", type=int, default=1200)
    tv = sub.add_parser("train-vocab", help="teach the embedding model your team's jargon")
    tv.add_argument("glossary", help='JSON {"term": "plain definition", ...}')
    tv.add_argument("--pairs", help='JSON [["sentence with jargon", "plain paraphrase"], ...]')
    tv.add_argument("--out", required=True)
    sub.add_parser("sleep", help="consolidate, forget to budget, prune links")
    sub.add_parser("stats")
    sub.add_parser("usage-report", help="summarise real use recorded by the MCP server")
    args = p.parse_args(argv)

    if args.cmd == "usage-report":
        from collections import Counter
        from .mcp_server import usage_path
        rows = [json.loads(l) for l in open(usage_path())] if os.path.exists(usage_path()) else []
        by_tool = Counter(r["tool"] for r in rows)
        recalls = [r for r in rows if r["tool"] in ("recall", "examples", "fixes_for", "advise", "prepare")]
        fb = [r for r in rows if r["tool"] == "feedback"]
        out = {"calls": len(rows), "projects": dict(Counter(r["project"] for r in rows)), "by_tool": dict(by_tool),
               "lookups_with_results": f"{sum(r['results'] > 0 for r in recalls)}/{len(recalls)}",
               "feedback_helped": sum(r["args"].get("reward", 0) > 0 for r in fb),
               "feedback_misled": sum(r["args"].get("reward", 0) < 0 for r in fb),
               "days": round((rows[-1]["t"] - rows[0]["t"]) / 86400, 1) if rows else 0}
        print(json.dumps(out, indent=2))
        return 0
    if args.cmd == "train-vocab":
        from .embed import make_embedder
        from .vocab import train_vocab
        pairs = [tuple(x) for x in json.load(open(args.pairs))] if args.pairs else []
        d = train_vocab(make_embedder(args.profile or "small"), json.load(open(args.glossary)), pairs, args.out)
        print(json.dumps({"terms": len(d.terms), "pairs": len(pairs), "saved": args.out}))
        return 0
    b = Brain(args.db, profile=args.profile, mind=args.mind, budget_bytes=args.budget,
              vocab=args.vocab or os.environ.get("NEUROMEM_VOCAB"))
    try:
        if args.cmd == "perceive":
            out = b.perceive(args.text, importance=args.importance, layer=args.layer, source=args.source)
        elif args.cmd == "recall":
            out = b.recall(args.cue, k=args.k, include_history=args.history)
        elif args.cmd == "about":
            out = b.about(args.entity)
        elif args.cmd == "history":
            out = b.history(args.id)
        elif args.cmd == "forget":
            out = {"forgotten": b.forget(args.id)}
        elif args.cmd == "experience":
            out = b.experience(args.task, args.action, args.reward, note=args.note)
        elif args.cmd == "advise":
            out = b.advise(args.task)
        elif args.cmd == "prepare":
            print(b.prepare(args.task, max_chars=args.max_chars))
            return 0
        elif args.cmd == "sleep":
            out = b.sleep()
        else:
            out = b.stats()
        if isinstance(out, list) and not args.json:
            show(out)
        else:
            print(json.dumps(out, indent=2))
        return 1 if out == [] else 0
    finally:
        b.close()


if __name__ == "__main__":
    sys.exit(main())
