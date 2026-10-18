"""Learned rewrite rules for code: generalise an in-line fix from one (failing, passing) pair so it can be
applied to new code.

Two rule kinds, both keyed by the dotted name of a call:
  arg      the i-th argument is wrapped:   users.find(E)  ->  users.find(E.strip().lower())   "{0}.strip().lower()"
  result   the call's result is extended:  cache.get(K)   ->  cache.get(K).value              "{0}.value"
A rule is written as one line, "rewrite <callee> arg<i>|result -> <template>", so it can be stored like any
other fix in neuromem's memory (remember_fix / fixes_for / habits / feedback).
"""
import difflib
import re

CALLEE = re.compile(r"[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+(?=\()")      # dotted call names only (a.b(...))
SUFFIX = re.compile(r"(?:\.[A-Za-z_]\w*(?:\(\))?)+")
RULE = re.compile(r"^rewrite (\S+) (arg(\d+)|result) -> (.+)$")


def calls(line):
    """[(callee, start, end, [arg texts])] for each dotted call in a line; end is just past ')'."""
    out = []
    for m in CALLEE.finditer(line):
        i, depth, args, cur, quote = m.end(), 0, [], "", None
        while i < len(line):
            ch = line[i]
            if quote:
                cur += ch
                if ch == quote:
                    quote = None
            elif ch in "'\"":
                quote = ch
                cur += ch
            elif ch == "(":
                depth += 1
                if depth > 1:
                    cur += ch
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    if cur.strip() or args:
                        args.append(cur.strip())
                    out.append((m.group(0), m.start(), i + 1, args))
                    break
                cur += ch
            elif ch == "," and depth == 1:
                args.append(cur.strip())
                cur = ""
            else:
                cur += ch
            i += 1
    return out


def _pairs(before, after):
    """Changed (old line, new line) pairs between two versions of some code."""
    a = [l.strip() for l in before.splitlines() if l.strip()]
    b = [l.strip() for l in after.splitlines() if l.strip()]
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op != "replace":
            continue
        for x in a[i1:i2]:
            best = max(b[j1:j2], key=lambda y: difflib.SequenceMatcher(None, x, y).ratio())
            out.append((x, best))
    return out


def learn(before, after):
    """Rewrite rules that turn calls in `before` into the corresponding calls in `after`."""
    rules = []
    for old, new in _pairs(before, after):
        new_calls = calls(new)
        for callee, s, e, args in calls(old):
            expr = old[s:e]
            match = [c for c in new_calls if c[0] == callee]
            if not match:
                continue
            _, ns, ne, nargs = match[0]
            if new[ns:ne] == expr:                               # same call: was its result extended?
                tail = SUFFIX.match(new[ne:])
                old_tail = SUFFIX.match(old[e:])
                if tail and not (old_tail and old_tail.group(0) == tail.group(0)):
                    rules.append(f"rewrite {callee} result -> {{0}}{tail.group(0)}")
                continue
            if len(nargs) == len(args):
                for i, (oa, na) in enumerate(zip(args, nargs)):
                    if oa != na and oa and oa in na:
                        rules.append(f"rewrite {callee} arg{i} -> {na.replace(oa, '{0}', 1)}")
    return list(dict.fromkeys(rules))


def is_rule(line):
    return bool(RULE.match(line.strip()))


def apply(code, rule):
    """Apply one rule to every matching call in `code`. Returns (new_code, changed)."""
    m = RULE.match(rule.strip())
    if not m:
        return code, False
    callee, kind, idx, template = m.group(1), m.group(2), m.group(3), m.group(4)
    piece = template.replace("{0}", "")                         # what the rule adds, e.g. ".strip().lower()"
    lines, changed = code.splitlines(), False
    for n, line in enumerate(lines):
        for c, s, e, args in reversed(calls(line)):            # right to left keeps earlier offsets valid
            if c != callee:
                continue
            if kind == "result":
                if line[e:].startswith(template.replace("{0}", "")):
                    continue
                line = line[:e] + template.replace("{0}", "", 1) + line[e:] if template.startswith("{0}") \
                    else line[:s] + template.replace("{0}", line[s:e]) + line[e:]
                changed = True
            else:
                i = int(idx)
                if i >= len(args) or not args[i] or piece in args[i]:
                    continue
                inner = line[s:e]
                new_args = args[:i] + [template.replace("{0}", args[i])] + args[i + 1:]
                line = line[:s] + inner[:inner.index("(") + 1] + ", ".join(new_args) + ")" + line[e:]
                changed = True
        lines[n] = line
    return "\n".join(lines), changed
