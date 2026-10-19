"""Reflex agent: neuromem briefing + examples, plus error->fix reflexes that act on the code directly.

  habits   fixes that worked >= 2 times are applied to every new attempt before it runs (no LLM call)
  reflex   on a remembered kind of error, apply the remembered fix and re-run (no LLM call)
  retry    otherwise ask the LLM, telling it how this kind of error was fixed before
  learn    every failing -> passing pair (or failing -> failing further on: partial progress) stores the error signature, the lines that fixed it, and in-line
           rewrite rules (neuromem.rewrite) such as users.find(E) -> users.find(E.strip().lower())
  unlearn  a habit or fix whose own line causes an error gets negative feedback and is retired
"""
import re

from neuromem import rewrite
from harness import SDK, NeuroMemory, extract_code, fmt_notes, llm, prompt, run_code, SYSTEM

CALL = re.compile(r"^[A-Za-z_][\w.]*\(.*\)$")


def added_lines(before, after):
    b = {l.strip() for l in before.splitlines()}
    out = []
    for l in after.splitlines():
        t = l.strip()
        if (t and t not in b and not t.startswith(("def ", "#", "return", "import", "from ")) and t not in out
                and not t.endswith(":")):                       # control-flow fragments mean nothing out of context
            out.append(t)
    return out[:4]


def callee(line):
    m = rewrite.RULE.match(line.strip())
    return m.group(1) if m else line.split("(")[0]


def apply_lines(code, lines, replace=True):
    """Apply remembered fix lines that are standalone calls (e.g. an init call):
    missing call -> insert at the top of the first function; present but written differently -> replace it
    (only when replace=True). Returns (new_code, applied_lines)."""
    from neuromem import Brain
    applied = []
    for rule in [l for l in lines if rewrite.is_rule(l)]:          # learned in-line rewrites
        code, changed = rewrite.apply(code, rule)
        if changed:
            applied.append(rule)
    src = code.splitlines()
    for fix in [l for l in lines if CALL.match(l) and not rewrite.is_rule(l)]:
        c = callee(fix) + "("
        idx = [i for i, l in enumerate(src) if l.strip().startswith(c)]
        if idx:
            i = idx[0]
            if replace and Brain.canonical(src[i]) != Brain.canonical(fix):
                indent = re.match(r"\s*", src[i]).group(0)
                src[i] = indent + fix
                applied.append(fix)
            continue
        if any(c in l for l in src):
            continue                                            # called inside an expression: leave it alone
        for i, l in enumerate(src):
            if re.match(r"\s*def \w+\(.*\).*:\s*$", l):
                nxt = next((x for x in src[i + 1:] if x.strip()), "    pass")
                src.insert(i + 1, (re.match(r"\s*", nxt).group(0) or "    ") + fix)
                applied.append(fix)
                break
    return "\n".join(src), applied


def blames(error, lines):
    """Did one of these lines cause the error? A statement is blamed when the error names its function; a
    rewrite rule when the error names something the rule added (e.g. "no attribute 'value'")."""
    for l in lines:
        m = rewrite.RULE.match(l.strip())
        if m:
            added = re.findall(r"[A-Za-z_]\w{2,}", m.group(4).replace("{0}", ""))
            if any(f"'{w}'" in error or f"{w}(" in error for w in added):
                return True
        elif callee(l).split(".")[-1] + "(" in error:
            return True
        elif any(f"'{w}'" in error for w in re.findall(r"[A-Za-z_]\w{2,}", l)):
            return True                                         # e.g. NameError: name 'project_id' is not defined
    return False


class ReflexMemory(NeuroMemory):
    pass


def run_reflex(model, n, sequence, log=None, profile="full", hints=True, name="reflex", replicate=False):
    """replicate=True: fixes learned from partial progress stay hidden until seen twice (safer, scored lower
    here: 44% vs 50% first try over 3 seeds, within noise)."""
    import json
    import time
    mem = ReflexMemory(profile)
    b = mem.brain
    rows = []
    for i, (tname, task, check) in enumerate(sequence):
        version = 1 if i < n // 2 else 2
        t0 = time.time()
        notes = mem.notes(task)
        if hints:                                               # verified fixes, shown up front
            kf = b.known_fixes(now=mem.clock)
            if kf:
                notes += "\n\nVerified fixes for errors seen before:\n" + "\n".join(
                    f"- if you get '{sig}': {' ; '.join(lines)}" for sig, lines in kf)
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(task, version, notes)}]
        reply = llm(model, msgs)
        raw = extract_code(reply)
        # habits act before the code ever runs
        habit_ids, code = [], raw
        for hid, lines in b.habits():
            code, applied = apply_lines(code, lines)
            if applied:
                habit_ids.append((hid, applied))
        ok1, err = run_code(code, check, version)
        mem.outcome(ok1)
        for hid, applied in habit_ids:
            if ok1:
                b.feedback([hid], 1, now=mem.clock)
            elif blames(err, applied):
                b.feedback([hid], -1, now=mem.clock)
        stage, final_ok, first_err, fail_code, good_code = "first", ok1, err, code, code if ok1 else None
        progress_code = None
        # reflex: a remembered fix for this kind of error, applied without asking the LLM
        if not final_ok:
            for fid, lines in b.fixes_for(err, now=mem.clock):
                fixed, applied = apply_lines(code, lines)
                if not applied:
                    continue
                ok_r, err_r = run_code(fixed, check, version)
                if ok_r:
                    b.feedback([fid], 1, now=mem.clock)
                    final_ok, stage, good_code = True, "reflex", fixed
                    break
                if blames(err_r, applied):
                    b.feedback([fid], -1, now=mem.clock)
        # retry with the LLM, told how this kind of error was fixed before
        if not final_ok:
            found = b.fixes_for(err, now=mem.clock, trusted_only=True) if hints else []
            extra = ("\nThis kind of error was fixed before by adding: " +
                     " ; ".join(" ; ".join(l) for _, l in found)) if found else ""
            msgs += [{"role": "assistant", "content": f"```python\n{code}\n```"},
                     {"role": "user", "content": f"Running it failed with:\n{err}{extra}\nFix the code."}]
            code2 = extract_code(llm(model, msgs))
            ok2, err2 = run_code(code2, check, version)
            if ok2:
                final_ok, stage, good_code = True, "retry", code2
            elif (b.error_signature(err2) != b.error_signature(err)
                  and not err2.startswith(("NameError", "SyntaxError", "IndentationError"))
                  and not blames(err2, added_lines(code, code2))):
                progress_code = code2                           # got further, and not by breaking the code
        # learn: the error and the lines that turned failing code into passing code
        lesson = ""
        fixed_code = good_code or progress_code                 # success, or partial progress past the error
        if fixed_code and not ok1:
            mem.clock += 0.05
            verified = good_code is not None or not replicate      # optionally treat partial progress as tentative
            b.remember_fix(first_err, added_lines(fail_code, fixed_code), now=mem.clock, verified=verified)
            for rule in rewrite.learn(fail_code, fixed_code):       # each in-line fix is its own memory
                b.remember_fix(first_err, [rule], now=mem.clock, verified=verified)
        if not ok1:
            msgs += [{"role": "user", "content": f"Write the one rule about {SDK} that would have prevented the first "
                                                 "error, as a concrete instruction including the exact code to write "
                                                 f"(not specific to this task). One sentence starting with '{SDK}:'."}]
            lesson = llm(model, msgs, max_tokens=80).strip().splitlines()[0][:300]
            mem.store(task, lesson, final_ok)
        if good_code:
            mem.store_example(task, good_code)
        rows.append({"condition": name, "seed": getattr(sequence, "seed", None), "i": i, "task": tname, "version": version, "first_try": ok1,
                     "after_retry": final_ok, "stage": stage, "error": first_err[:160], "lesson": lesson,
                     "habits": [l for _, l in habit_ids], "notes": notes, "code": raw, "secs": round(time.time() - t0, 1)})
        print(f"[{name}] {i:2} v{version} {tname:<18} first={'PASS' if ok1 else 'fail'} final={'PASS' if final_ok else 'fail'}"
              f" via={stage:<6} habits={len(habit_ids)}  {first_err[:60]}", flush=True)
        if log:
            log.write(json.dumps(rows[-1]) + "\n")
            log.flush()
    return rows
