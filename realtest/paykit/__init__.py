"""paykit: a second fictional SDK, with different kinds of quirks from acmekit.
PAYKIT_VERSION=2 renames configure(key=) to configure(api_key=) and Quote.amount to Quote.value."""
import os
from decimal import Decimal, InvalidOperation

VERSION = int(os.environ.get("PAYKIT_VERSION", "1"))
_state = {"key": None}
POSTS, _SEEN = [], set()


class RateLimited(Exception):
    pass


def env(name):
    return {"PAYKIT_KEY": "pk_test_42"}.get(name, "")


def configure(*args, **kw):
    want = "key" if VERSION == 1 else "api_key"
    if VERSION == 2 and "key" in kw:
        raise TypeError("paykit 2.x: configure(key=...) was renamed, use paykit.configure(api_key=paykit.env('PAYKIT_KEY'))")
    kw = {**dict(zip([want], args)), **kw}
    if set(kw) != {want}:
        raise TypeError(f"paykit.configure() takes one argument: {want}=paykit.env('PAYKIT_KEY')")
    if kw[want] != "pk_test_42":
        raise PermissionError("paykit: invalid key; read it with paykit.env('PAYKIT_KEY')")
    _state["key"] = kw[want]


def _need():
    if not _state["key"]:
        raise RuntimeError("paykit is not configured: call paykit.configure(...) before any other call")


def _money(x):
    if not isinstance(x, str):
        raise TypeError(f"amounts must be decimal strings like '12.50', got {type(x).__name__} {x!r}")
    try:
        return Decimal(x)
    except InvalidOperation:
        raise ValueError(f"not a decimal amount: {x!r}")


_RATES = {"US": "0.07", "DE": "0.19", "NP": "0.13", "GB": "0.20", "JP": "0.10"}
_FX = {("USD", "EUR"): "0.90", ("USD", "NPR"): "133.00", ("EUR", "USD"): "1.11", ("GBP", "USD"): "1.27"}
_CUSTOMERS = {"cus_1": {"id": "cus_1", "name": "Asha", "country": "NP"},
              "cus_2": {"id": "cus_2", "name": "Ben", "country": "US"},
              "cus_3": {"id": "cus_3", "name": "Clara", "country": "DE"}}


class tax:
    @staticmethod
    def rate(country):
        _need()
        if country not in _RATES:
            raise KeyError(f"unknown country {country!r}: use ISO alpha-2 uppercase codes like 'US'")
        return _RATES[country]


class Quote:
    def __init__(self, v):
        if VERSION == 1:
            self.amount = v
        else:
            self.value = v

    def __repr__(self):
        return "Quote(...)"


class fx:
    @staticmethod
    def convert(amount, frm, to):
        _need()
        a = _money(amount)
        r = Decimal(_FX[(frm.upper(), to.upper())])
        return Quote(str((a * r).quantize(Decimal("0.01"))))


class customers:
    @staticmethod
    def get(customer_id):
        _need()
        if not str(customer_id).startswith("cus_"):
            raise ValueError(f"customer ids start with 'cus_', got {customer_id!r}")
        return dict(_CUSTOMERS[customer_id])


class ledger:
    @staticmethod
    def post(customer_id, amount, memo=""):
        _need()
        a = _money(amount)
        if (customer_id, memo) not in _SEEN:
            _SEEN.add((customer_id, memo))
            raise RateLimited("paykit.ledger.post is rate limited: retry the call (up to 3 times)")
        POSTS.append((customer_id, str(a), memo))
        return {"posted": str(a)}


API_DOC = {1: """paykit API reference (version 1):
  paykit.configure(key: str)                 set up the client with your API key
  paykit.env(name: str) -> str               read a configuration value (the key is 'PAYKIT_KEY')
  paykit.tax.rate(country: str) -> str       sales tax rate for a country, e.g. '0.07'
  paykit.fx.convert(amount, frm: str, to: str)   currency conversion quote
  paykit.customers.get(customer_id) -> dict  keys id, name, country
  paykit.ledger.post(customer_id, amount, memo='') -> dict   may raise paykit.RateLimited""",
           2: """paykit API reference (version 2):
  paykit.configure(api_key: str)             set up the client with your API key
  paykit.env(name: str) -> str               read a configuration value (the key is 'PAYKIT_KEY')
  paykit.tax.rate(country: str) -> str       sales tax rate for a country, e.g. '0.07'
  paykit.fx.convert(amount, frm: str, to: str)   currency conversion quote
  paykit.customers.get(customer_id) -> dict  keys id, name, country
  paykit.ledger.post(customer_id, amount, memo='') -> dict   may raise paykit.RateLimited"""}
