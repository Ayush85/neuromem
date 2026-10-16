"""acmekit: a fictional internal SDK with the kind of quirks real internal libraries have.
The LLM sees only API_DOC; the quirks are discovered through errors. ACMEKIT_VERSION=2 changes init()."""
import os
from datetime import datetime, timedelta, timezone

VERSION = int(os.environ.get("ACMEKIT_VERSION", "1"))
_state = {"ready": False}


class TransientError(Exception):
    pass


def init(*args, **kw):
    names = ["app"] if VERSION == 1 else ["project", "token"]
    if len(args) > len(names):
        raise TypeError(f"acmekit.init() takes at most {len(names)} positional arguments")
    kw = {**dict(zip(names, args)), **kw}
    if VERSION == 1:
        if set(kw) != {"app"}:
            raise TypeError("acmekit.init() requires exactly one argument: app='<application name>'")
    else:
        if "app" in kw:
            raise TypeError("acmekit 2.x: init(app=...) was removed, use acmekit.init(project='<project id>', "
                            "token=acmekit.auth.token())")
        if set(kw) != {"project", "token"} or kw["token"] != "tok-123":
            raise TypeError("acmekit 2.x: call acmekit.init(project='<project id>', token=acmekit.auth.token())")
    _state["ready"] = True


def _require():
    if not _state["ready"]:
        raise RuntimeError("acmekit is not initialized: call acmekit.init(...) once before using any acmekit module")


class _Auth:
    @staticmethod
    def token():
        return "tok-123"


auth = _Auth()

_USERS = {"alice@example.com": {"id": "u1", "name": "Alice"}, "bob@example.com": {"id": "u2", "name": "Bob"},
          "chen@example.com": {"id": "u3", "name": "Chen"}}
_ORDERS = {"u1": [{"id": f"o{i}", "amount_cents": 1000 + i * 250} for i in range(5)],
           "u2": [{"id": "o9", "amount_cents": 4200}],
           "u3": [{"id": f"p{i}", "amount_cents": 300 * (i + 1)} for i in range(7)]}
CHARGES, EVENTS, _HTTP_SEEN = [], [], set()
_CACHE = {"name:u1": "Alice A.", "name:u2": "Bobby", "name:u3": "Chen C.", "plan:u1": "pro", "plan:u2": "free",
          "plan:u3": "team"}


class users:
    @staticmethod
    def find(email):
        _require()
        if email in _USERS:
            return dict(_USERS[email])
        raise LookupError(f"no user with email {email!r} (emails are stored lowercase and without surrounding spaces)")


class orders:
    @staticmethod
    def list(user_id, page_token=None):
        _require()
        items = _ORDERS.get(user_id, [])
        start = int(page_token or 0)
        nxt = start + 2
        return items[start:nxt], (str(nxt) if nxt < len(items) else None)


class billing:
    @staticmethod
    def charge(user_id, amount):
        _require()
        if not isinstance(amount, int) or isinstance(amount, bool):
            raise TypeError(f"billing.charge: amount must be an int number of cents, got {type(amount).__name__} "
                            f"{amount!r}")
        CHARGES.append((user_id, amount))
        return {"ok": True, "charged_cents": amount}


class events:
    @staticmethod
    def schedule(name, when):
        _require()
        if not isinstance(when, datetime) or when.tzinfo is None:
            raise ValueError("events.schedule: `when` must be a timezone-aware datetime (e.g. datetime.now(timezone.utc))")
        EVENTS.append((name, when))
        return {"scheduled": name}


class http:
    @staticmethod
    def get(url):
        _require()
        if url not in _HTTP_SEEN:
            _HTTP_SEEN.add(url)
            raise TransientError(f"temporary failure fetching {url}; acmekit.http calls must be retried (up to 3 tries)")
        return f"  OK {url}  "


class Result:
    def __init__(self, value):
        self.ok, self.value = value is not None, value

    def __repr__(self):
        return f"Result(ok={self.ok})"


class cache:
    @staticmethod
    def get(key):
        _require()
        return Result(_CACHE.get(key))


API_DOC = {1: """acmekit API reference (version 1):
  acmekit.init(app: str)                    initialize the SDK with your application's name
  acmekit.users.find(email) -> dict         user with keys id, name
  acmekit.orders.list(user_id, page_token=None) -> (orders, next_token)   orders are dicts with id, amount_cents
  acmekit.billing.charge(user_id, amount) -> dict
  acmekit.events.schedule(name, when) -> dict
  acmekit.http.get(url) -> str              may raise acmekit.TransientError
  acmekit.cache.get(key)                    cached value for key""",
           2: """acmekit API reference (version 2):
  acmekit.init(project: str, token: str)    initialize the SDK for your project
  acmekit.auth.token() -> str               access token for init
  acmekit.users.find(email) -> dict         user with keys id, name
  acmekit.orders.list(user_id, page_token=None) -> (orders, next_token)   orders are dicts with id, amount_cents
  acmekit.billing.charge(user_id, amount) -> dict
  acmekit.events.schedule(name, when) -> dict
  acmekit.http.get(url) -> str              may raise acmekit.TransientError
  acmekit.cache.get(key)                    cached value for key"""}
