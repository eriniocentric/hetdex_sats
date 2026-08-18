#!/usr/bin/env python
"""
diagnose_gap.py -- figure out why the 2023-08..2024-07 TLE nights won't fetch.

Run this from inside crossmatch/ (it imports fetch_tles.py from there), with
your Space-Track credentials set the usual way:

    export SPACETRACK_IDENTITY="you@example.com"
    export SPACETRACK_PASSWORD="..."
    python diagnose_gap.py

Or edit TEST_NIGHTS below to poke a specific date. It queries ONE night's
whole-catalog gp_history window directly (bypassing fetch_tles.main()'s
error handling) and prints exactly what Space-Track sends back: status
code, response size, headers, and the first 500 chars of the body. That's
enough to tell whether this is:

  - a genuine oversized response (Space-Track's ~500,000-row cap, or a
    request that's simply too big for the API to serve reliably at the
    2023-2024 satellite population density)
  - a server-side timeout (slow response, connection dropped)
  - an auth/permission issue specific to this class or date range
  - something else entirely
"""
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, ".")
import fetch_tles as F

# A handful of nights spanning the reported gap (2023-08 .. 2024-07).
# MJD roughly: 2023-08-15 ~ 60171, 2023-11-01 ~ 60249, 2024-02-01 ~ 60341,
# 2024-05-01 ~ 60431. Edit freely -- these don't need to be exact shot nights,
# just representative dates inside the gap.
TEST_DATES = [
    "2023-08-15",
    "2023-11-01",
    "2024-02-01",
    "2024-05-01",
]

# For comparison, one night well *outside* the gap that is known to work.
CONTROL_DATE = "2020-06-15"


def probe(session, limiter, date_str, pad_days=1.0):
    center = datetime.strptime(date_str, "%Y-%m-%d")
    s = center - timedelta(days=pad_days)
    e = center + timedelta(days=pad_days)
    url = F.gp_history_url(s, e)
    print(f"\n{'='*70}\nProbing {date_str}  (window {s:%Y-%m-%d} .. {e:%Y-%m-%d})")
    print(f"URL: {url}")

    import requests
    t0 = time.time()
    limiter.wait()
    try:
        r = session.get(url, timeout=600)
    except requests.RequestException as exc:
        dt = time.time() - t0
        print(f"  EXCEPTION after {dt:.1f}s: {type(exc).__name__}: {exc}")
        return
    dt = time.time() - t0

    print(f"  HTTP {r.status_code}  in {dt:.1f}s")
    print(f"  response bytes: {len(r.content)}")
    print(f"  content-type: {r.headers.get('Content-Type')}")
    # Space-Track sometimes signals truncation / errors via headers or body text
    for h in ("X-RateLimit-Remaining", "X-RateLimit-Limit", "Retry-After"):
        if h in r.headers:
            print(f"  {h}: {r.headers[h]}")

    if r.status_code == 200:
        n_rec = r.text.count("\n1 ")
        print(f"  parsed record count (3LE '1 ' lines): {n_rec}")
        if n_rec == 0 and len(r.text) > 0:
            print(f"  body is non-empty but has 0 records -- first 500 chars:")
            print("  " + repr(r.text[:500]))
    else:
        print(f"  body (first 500 chars): {r.text[:500]!r}")


def main():
    identity, password = F.load_credentials(
        __import__("argparse").Namespace(identity=None, password=None,
                                          config="~/.spacetrack.ini")
    )
    print(f"logging in as {identity} ...")
    session = F.make_session(identity, password)
    limiter = F.RateLimiter()
    print("login OK")

    print("\n--- CONTROL (known-good era) ---")
    probe(session, limiter, CONTROL_DATE)

    print("\n\n--- GAP WINDOW ---")
    for d in TEST_DATES:
        probe(session, limiter, d)


if __name__ == "__main__":
    main()