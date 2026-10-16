"""Ops / engineering jargon with plain meanings, and sentence frames split into train and held-out sets."""

# term: (plain definition, jargon phrase, plain phrase)
GLOSSARY = {
    "OOMKilled": ("killed for running out of memory", "got OOMKilled", "was killed because it ran out of memory"),
    "CrashLoopBackOff": ("keeps crashing and restarting", "is in CrashLoopBackOff",
                         "keeps crashing and restarting over and over"),
    "p99": ("slowest one percent latency", "p99 jumped to 4s", "the slowest requests now take four seconds"),
    "SLO": ("reliability target", "is breaching its SLO", "is missing its reliability target"),
    "flaky": ("fails randomly sometimes", "has a flaky test", "has a test that fails randomly"),
    "rollback": ("revert to the previous version", "needs a rollback", "needs to go back to the previous version"),
    "hotfix": ("urgent small fix shipped directly to production", "needs a hotfix",
               "needs an urgent small fix in production"),
    "toil": ("repetitive manual operational work", "generates a lot of toil",
             "creates a lot of repetitive manual work"),
    "postmortem": ("incident review written after an outage", "needs a postmortem",
                   "needs a written review of what went wrong in the outage"),
    "canary": ("release to a small fraction of users first", "is going out as a canary",
               "is being released to a small group of users first"),
    "PagerDuty": ("on-call alerting that wakes engineers up", "fired a PagerDuty alert",
                  "woke up the on-call engineer with an alert"),
    "throttled": ("rate limited and slowed down", "is getting throttled", "is being rate limited"),
    "deadlock": ("two operations waiting on each other forever", "hit a deadlock",
                 "froze because two operations were waiting on each other"),
    "N+1": ("one database query per item instead of one query", "has an N+1 query problem",
            "makes a separate database query for every item"),
    "cold start": ("slow first request after being idle", "suffers from cold start",
                   "is slow on the first request after sitting idle"),
    "backpressure": ("consumers telling producers to slow down", "needs backpressure",
                     "needs a way to make producers slow down when consumers fall behind"),
    "502": ("bad gateway error from the proxy", "returns 502s", "returns bad gateway errors"),
    "DLQ": ("dead letter queue for failed messages", "messages pile up in the DLQ",
            "failed messages pile up in the dead letter queue"),
    "drift": ("configuration differs from what is declared", "has config drift",
              "has configuration that no longer matches what was declared"),
    "thundering herd": ("many clients retry at the same moment", "caused a thundering herd",
                        "caused many clients to retry at the same moment"),
}
SERVICES = ["billing", "search", "checkout", "payments", "inventory", "auth", "reporting", "gateway", "media", "ledger"]
TRAIN_FRAMES = ["the {s} service {x}", "after the deploy {s} {x}", "heads up: {s} {x}", "{s} {x} again today",
                "we noticed {s} {x}"]
TEST_FRAMES = ["looks like {s} {x} this morning", "incident: {s} {x}", "why {s} {x}?"]
UNRELATED = ["the team lunch is on thursday", "please submit your expense reports", "the office plants need water",
             "welcome our new designer to the team", "the quarterly all-hands is next week",
             "parking is closed for renovation", "the holiday party venue is booked", "coffee machine is fixed"]
