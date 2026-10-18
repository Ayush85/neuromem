from neuromem.rewrite import apply, calls, learn

BEFORE = """def greeting(email):
    acmekit.init(app='x')
    user = acmekit.users.find(email)
    name = acmekit.cache.get(f"name:{user['id']}")
    return f"Hello, {name}!"
"""
AFTER = """def greeting(email):
    acmekit.init(app='x')
    user = acmekit.users.find(email.strip().lower())
    name = acmekit.cache.get(f"name:{user['id']}").value
    return f"Hello, {name}!"
"""


def test_calls_parses_nested_arguments():
    (c, s, e, args), = [x for x in calls("x = acmekit.billing.charge(u['id'], int(round(d * 100)))") if "charge" in x[0]]
    assert c == "acmekit.billing.charge" and args == ["u['id']", "int(round(d * 100))"]


def test_learns_argument_and_result_rules():
    rules = learn(BEFORE, AFTER)
    assert "rewrite acmekit.users.find arg0 -> {0}.strip().lower()" in rules
    assert "rewrite acmekit.cache.get result -> {0}.value" in rules


def test_rules_generalise_to_new_code_and_are_idempotent():
    rules = learn(BEFORE, AFTER)
    new = "def plan(addr):\n    u = acmekit.users.find(addr)\n    return acmekit.cache.get('plan:' + u['id'])"
    for r in rules:
        new, _ = apply(new, r)
    assert "acmekit.users.find(addr.strip().lower())" in new
    assert "acmekit.cache.get('plan:' + u['id']).value" in new
    again = new
    for r in rules:
        again, changed = apply(again, r)
        assert not changed
    assert again == new


def test_wrapping_template_with_expression():
    rules = learn("acmekit.billing.charge(uid, dollars)", "acmekit.billing.charge(uid, int(round(dollars * 100)))")
    assert rules == ["rewrite acmekit.billing.charge arg1 -> int(round({0} * 100))"]
    assert apply("r = acmekit.billing.charge(u, fee)", rules[0])[0] == "r = acmekit.billing.charge(u, int(round(fee * 100)))"
