"""Coding tasks against acmekit. Each needs init() plus one or two quirks. `check` is test code run after the
solution is defined; it raises AssertionError with a helpful message on failure."""
import random

PREAMBLE = "import acmekit\nfrom datetime import datetime, timedelta, timezone\n"

TASKS = [
    ("total_spent", "Write total_spent(email) that returns the total amount_cents over all orders of the user "
                    "with that email address.",
     "assert total_spent(' Alice@Example.com') == 7500, f'expected 7500, got {total_spent(\" Alice@Example.com\")}'\n"
     "assert total_spent('CHEN@example.com') == 8400"),
    ("count_orders", "Write count_orders(email) returning how many orders the user with this email has placed.",
     "n = count_orders('Alice@example.com'); assert n == 5, f'expected 5 orders, got {n} (check you read every page)'\n"
     "assert count_orders('chen@example.com ') == 7"),
    ("charge_fee", "Write charge_fee(email, dollars) that charges the user a fee of `dollars` dollars "
                   "(dollars can be a float like 2.5) and returns the charge result.",
     "r = charge_fee('BOB@example.com', 2.5); assert acmekit.CHARGES[-1] == ('u2', 250), acmekit.CHARGES"),
    ("schedule_reminder", "Write schedule_reminder(email, days) that schedules an event named "
                          "'reminder:<user id>' for `days` days from now.",
     "schedule_reminder('alice@example.com', 3); name, when = acmekit.EVENTS[-1]; assert name == 'reminder:u1', name\n"
     "assert abs((when - datetime.now(timezone.utc)).total_seconds() - 3 * 86400) < 60"),
    ("fetch_status", "Write fetch_status(url) that returns the response text of acmekit.http.get(url) with "
                     "surrounding whitespace removed.",
     "assert fetch_status('https://status.acme/api') == 'OK https://status.acme/api'"),
    ("display_name", "Write display_name(user_id) that returns the cached display name stored under the cache "
                     "key 'name:<user_id>'.",
     "v = display_name('u2'); assert v == 'Bobby', f'expected the string Bobby, got {v!r}'"),
    ("user_plan", "Write user_plan(email) that returns the cached plan name for the user with this email "
                  "(cache key 'plan:<user id>').",
     "v = user_plan(' bob@EXAMPLE.com'); assert v == 'free', f'expected the string free, got {v!r}'"),
    ("largest_order", "Write largest_order(email) returning the id of the user's most expensive order.",
     "v = largest_order('Chen@example.com'); assert v == 'p6', f'expected p6, got {v!r} (did you see every order?)'"),
    ("bill_orders", "Write bill_orders(email, dollars_per_order) that charges the user dollars_per_order "
                    "(a float) for each order they have, as a single charge, and returns the result.",
     "bill_orders('alice@example.com', 1.5); assert acmekit.CHARGES[-1] == ('u1', 750), acmekit.CHARGES"),
    ("ping_all", "Write ping_all(urls) that fetches each url with acmekit.http.get and returns a list of the "
                 "stripped response texts.",
     "assert ping_all(['https://a.acme', 'https://b.acme']) == ['OK https://a.acme', 'OK https://b.acme']"),
    ("followup", "Write followup(email, hours) that schedules an event 'followup:<user id>' `hours` hours from now "
                 "and returns the schedule result.",
     "followup(' chen@example.com', 5); name, when = acmekit.EVENTS[-1]; assert name == 'followup:u3', name"),
    ("greeting", "Write greeting(email) returning 'Hello, <cached display name>!' using the cache key "
                 "'name:<user id>'.",
     "v = greeting('ALICE@example.com'); assert v == 'Hello, Alice A.!', f'got {v!r}'"),
]


def sequence(seed=0, n=24):
    """Passes over the task kinds, each in a fresh shuffled order."""
    r = random.Random(seed)
    out = []
    while len(out) < n:
        p = TASKS[:]
        r.shuffle(p)
        out += p
    return out[:n]
