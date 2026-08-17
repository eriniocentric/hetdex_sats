#!/usr/bin/env python
"""
fetch_tles.py -- download the archival TLEs needed to identify HETDEX streaks.

Pulls, from Space-Track.org:

  1. `gp_history` TLEs in a +/- 1 day epoch window around every observing night
     that contains a streak, one gzipped 3LE file per night;
  2. the full `satcat` metadata table (object name, COSPAR ID, type, country,
     launch/decay dates, RCS size), once.

Why per night, and why 3LE:  a whole night of `gp_history` is ~20-30k records.
In JSON that is 30-50 MB per night (~15 GB for the survey); in 3LE it is about
170 bytes/record, i.e. 3-5 MB per night and ~1.5 GB for all of PDR1.  Object
metadata that 3LE drops is recovered once from satcat and joined by NORAD ID.

Space-Track asks API users to stay under 30 queries/minute and 300/hour; this
script enforces both, caches every night to disk, and is safe to interrupt and
re-run -- it skips nights already present.  A full cold run over PDR1 takes
roughly 1.5-2 hours, almost all of it waiting on the rate limiter.

Credentials, in order of precedence:
    --identity / --password
    $SPACETRACK_IDENTITY / $SPACETRACK_PASSWORD
    ~/.spacetrack.ini   ->   [spacetrack]
                             identity = you@example.edu
                             password = ...

Usage
-----
    python fetch_tles.py --catalog HETDEX_PDR1_sats.fits --cache-dir tle_cache
    python fetch_tles.py --catalog ... --dry-run     # just list what it would do

Space-Track's user agreement allows this use but not redistribution of the raw
element sets; NORAD IDs and derived quantities in your paper are fine.
"""

from __future__ import annotations

import argparse
import configparser
import gzip
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import numpy as np

BASE = "https://www.space-track.org"
LOGIN_URL = f"{BASE}/ajaxauth/login"
QUERY = f"{BASE}/basicspacedata/query"

# HET is at UTC-6 (no DST correction needed: we only want a night label).
SITE_UTC_OFFSET_HOURS = -6.0


# --------------------------------------------------------------- night keys
def night_id(mjd):
    """Integer label for the local observing night containing `mjd`.

    Anchored on local noon so that after-midnight shots stay with the evening
    they belong to:  floor(mjd - 0.5 + utc_offset/24).
    """
    return np.floor(np.asarray(mjd, float) - 0.5 + SITE_UTC_OFFSET_HOURS / 24.0).astype(int)


def mjd_to_datetime(mjd):
    return datetime(1858, 11, 17) + timedelta(days=float(mjd))


def datetime_to_mjd(dt):
    return (dt - datetime(1858, 11, 17)).total_seconds() / 86400.0


# ------------------------------------------------------------- credentials
def load_credentials(args):
    if args.identity and args.password:
        return args.identity, args.password
    ident = os.environ.get("SPACETRACK_IDENTITY")
    pw = os.environ.get("SPACETRACK_PASSWORD")
    if ident and pw:
        return ident, pw
    ini = Path(args.config).expanduser()
    if ini.exists():
        cp = configparser.ConfigParser()
        cp.read(ini)
        if cp.has_section("spacetrack"):
            return cp["spacetrack"].get("identity"), cp["spacetrack"].get("password")
    raise SystemExit(
        "No Space-Track credentials found.  Set SPACETRACK_IDENTITY and\n"
        "SPACETRACK_PASSWORD, pass --identity/--password, or create\n"
        f"{ini} with a [spacetrack] section.  Accounts are free at\n"
        "https://www.space-track.org/auth/createAccount"
    )


# ------------------------------------------------------------ rate limiting
class RateLimiter:
    """Enforce Space-Track's 30/min and 300/hour guidance simultaneously."""

    def __init__(self, per_minute=25, per_hour=280):
        self.per_minute = per_minute
        self.per_hour = per_hour
        self.stamps = []

    def wait(self):
        while True:
            now = time.time()
            self.stamps = [t for t in self.stamps if now - t < 3600.0]
            in_min = sum(1 for t in self.stamps if now - t < 60.0)
            in_hour = len(self.stamps)
            if in_min < self.per_minute and in_hour < self.per_hour:
                self.stamps.append(now)
                return
            waits = []
            if in_min >= self.per_minute:
                oldest_min = min(t for t in self.stamps if now - t < 60.0)
                waits.append(60.0 - (now - oldest_min) + 0.5)
            if in_hour >= self.per_hour:
                waits.append(3600.0 - (now - min(self.stamps)) + 0.5)
            sleep_for = max(1.0, min(waits))
            print(f"    [rate limit] sleeping {sleep_for:.0f}s "
                  f"({in_min}/min, {in_hour}/hr)", flush=True)
            time.sleep(sleep_for)


# ------------------------------------------------------------------ session
def make_session(identity, password):
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = "HETDEX-satellite-streak-study/1.0"
    r = s.post(LOGIN_URL, data={"identity": identity, "password": password}, timeout=60)
    r.raise_for_status()
    if "Failed" in r.text or "login" in r.text.lower() and len(r.text) > 200:
        raise SystemExit("Space-Track login rejected -- check credentials.")
    return s


def st_get(session, url, limiter, retries=4):
    """GET with rate limiting and exponential backoff on 429/5xx."""
    import requests
    for attempt in range(retries):
        limiter.wait()
        try:
            r = session.get(url, timeout=600)
        except requests.RequestException as exc:
            wait = 30 * (attempt + 1)
            print(f"    network error ({exc}); retrying in {wait}s", flush=True)
            time.sleep(wait)
            continue
        if r.status_code == 200:
            return r.text
        if r.status_code == 204:
            # No Content: the query itself succeeded, it just matched zero
            # records for this EPOCH range. Space-Track returns this instead
            # of a 200 with an empty body -- treat it as "zero records", not
            # an error, or a single empty night takes the whole batch down.
            return ""
        if r.status_code in (429, 500, 502, 503, 504):
            wait = 60 * (attempt + 1)
            print(f"    HTTP {r.status_code}; backing off {wait}s", flush=True)
            time.sleep(wait)
            continue
        raise RuntimeError(f"HTTP {r.status_code} for {url}\n{r.text[:500]}")
    raise RuntimeError(f"giving up on {url}")


# ------------------------------------------------------------------ queries
def gp_history_url(start_dt, end_dt):
    """gp_history restricted to an EPOCH range, returned as 3LE."""
    fmt = "%Y-%m-%d %H:%M:%S"
    rng = quote(f"{start_dt.strftime(fmt)}--{end_dt.strftime(fmt)}", safe="-:")
    order = quote("NORAD_CAT_ID asc,EPOCH asc", safe=",")
    return f"{QUERY}/class/gp_history/EPOCH/{rng}/orderby/{order}/format/3le"


def satcat_url():
    return (f"{QUERY}/class/satcat/orderby/{quote('NORAD_CAT_ID asc', safe=',')}"
            f"/format/json")


def gp_history_norad_url(norad_id, start_dt, end_dt):
    """gp_history for ONE object across an EPOCH range.

    `gp_history_url` above asks for the whole catalog in a window; that is
    fine at the default +/-1 day pad but was observed, empirically, to
    either hit Space-Track's ~500,000-row response cap (silently truncated,
    sorted by NORAD_CAT_ID -- so higher-numbered, more recently catalogued
    objects are the ones cut off) or get rejected outright (HTTP 204) once
    widened to +/-30 days. Filtering by NORAD_CAT_ID keeps the response to a
    handful of records regardless of window width, so it is safe to widen
    this one a lot when a per-night fetch missed a specific object.
    """
    fmt = "%Y-%m-%d %H:%M:%S"
    rng = quote(f"{start_dt.strftime(fmt)}--{end_dt.strftime(fmt)}", safe="-:")
    order = quote("EPOCH asc", safe=",")
    return (f"{QUERY}/class/gp_history/NORAD_CAT_ID/{int(norad_id)}"
           f"/EPOCH/{rng}/orderby/{order}/format/3le")


def fetch_missing_norads(pairs, cache_dir, pad_days=60.0,
                         identity=None, password=None,
                         config="~/.spacetrack.ini"):
    """Targeted per-object TLE fetch, for objects a per-night catalog-wide
    fetch missed entirely (see `gp_history_norad_url` for why this is safer
    than just widening the whole-catalog fetch's pad).

    Writes one small file per object, `norad_<id>.3le.gz`, into `cache_dir`
    alongside the per-night files -- `match_streaks.rematch_by_norad` checks
    there as a fallback when an object is absent from the night's own file.
    Reused across calls: an object already fetched is not re-requested.

    Parameters
    ----------
    pairs : iterable of (norad_id, center_mjd) -- center_mjd is normally the
            streak's mjd_shot
    pad_days : half-width of the EPOCH window searched, days. 60 is a
               generous default since the query is cheap regardless of width.

    Returns
    -------
    {norad_id: Path or None}. None means nothing was found even at this
    width -- most likely a genuine Space-Track archive gap for that object
    around this date, not a fetch problem, so widening `pad_days` further
    before concluding that is reasonable but unlikely to change the answer.
    """
    ns = argparse.Namespace(identity=identity, password=password, config=config)
    identity, password = load_credentials(ns)
    session = make_session(identity, password)
    limiter = RateLimiter()

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    pairs = list(pairs)
    out = {}
    for i, (norad, mjd) in enumerate(pairs, 1):
        norad = int(norad)
        path = cache / f"norad_{norad}.3le.gz"
        if path.exists() and path.stat().st_size > 0:
            print(f"[{i}/{len(pairs)}] norad {norad}: already cached")
            out[norad] = path
            continue
        center = mjd_to_datetime(float(mjd))
        s = center - timedelta(days=pad_days)
        e = center + timedelta(days=pad_days)
        print(f"[{i}/{len(pairs)}] norad {norad}  "
              f"EPOCH {s:%Y-%m-%d} -- {e:%Y-%m-%d}", flush=True)
        try:
            text = st_get(session, gp_history_norad_url(norad, s, e), limiter)
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
            out[norad] = None
            continue
        n_rec = text.count("\n1 ")
        if n_rec == 0:
            print(f"    nothing within +/-{pad_days:.0f} days -- likely a "
                  "real Space-Track gap for this object, not a fetch issue")
            out[norad] = None
            continue
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            fh.write(text)
        tmp.replace(path)
        print(f"    {n_rec} element sets -> {path.name}")
        out[norad] = path
    return out


# --------------------------------------------------------------------- main
def read_shot_times(catalog):
    from astropy.table import Table
    info = Table.read(catalog, hdu="INFO")
    return np.asarray(info["mjd_shot"], dtype=float)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default="HETDEX_PDR1_sats.fits",
                    help="streak catalog; only mjd_shot is read")
    ap.add_argument("--cache-dir", default="tle_cache")
    ap.add_argument("--pad-days", type=float, default=1.0,
                    help="TLE epoch window half-width around each night (default 1.0)")
    ap.add_argument("--identity")
    ap.add_argument("--password")
    ap.add_argument("--config", default="~/.spacetrack.ini")
    ap.add_argument("--skip-satcat", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-nights", type=int, default=0,
                    help="stop after N new nights (useful for a trial run)")
    ap.add_argument("--start-date", default=None,
                    help="only nights on or after this UTC date, YYYY-MM-DD")
    ap.add_argument("--end-date", default=None,
                    help="only nights on or before this UTC date, YYYY-MM-DD")
    ap.add_argument("--latest", action="store_true",
                    help="work backwards from the most recent night; combine "
                         "with --max-nights to grab recent data first")
    args = ap.parse_args(argv)

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    mjds = read_shot_times(args.catalog)
    nights = defaultdict(list)
    for m in mjds:
        nights[int(night_id(m))].append(m)

    lo_mjd = datetime_to_mjd(datetime.strptime(args.start_date, "%Y-%m-%d")) \
        if args.start_date else -np.inf
    hi_mjd = datetime_to_mjd(datetime.strptime(args.end_date, "%Y-%m-%d")) + 1.0 \
        if args.end_date else np.inf

    todo, skipped_date, have = [], 0, 0
    for nid in sorted(nights):
        lo, hi = min(nights[nid]), max(nights[nid])
        label = mjd_to_datetime(lo).strftime("%Y-%m-%d")
        path = cache / f"gp_{nid}_{label}.3le.gz"
        if path.exists() and path.stat().st_size > 0:
            have += 1
            continue
        if not (lo_mjd <= lo < hi_mjd):
            skipped_date += 1
            continue
        todo.append((nid, lo, hi, path))

    if args.latest:
        todo.reverse()

    print(f"{len(mjds)} streaks -> {len(nights)} observing nights "
          f"({have} already cached)")
    if skipped_date:
        print(f"{skipped_date} nights outside the requested date range")
    print(f"{len(todo)} to fetch"
          + (f", newest first, capped at {args.max_nights}" if args.latest
             and args.max_nights else ""))
    if todo:
        print(f"  range: {mjd_to_datetime(todo[0][1]):%Y-%m-%d} .. "
              f"{mjd_to_datetime(todo[-1][1]):%Y-%m-%d}")
    if args.dry_run:
        for nid, lo, hi, path in todo[:10]:
            s = mjd_to_datetime(lo - args.pad_days)
            e = mjd_to_datetime(hi + args.pad_days)
            print(f"  {path.name}: EPOCH {s:%Y-%m-%d %H:%M} -- {e:%Y-%m-%d %H:%M}")
        if len(todo) > 10:
            print(f"  ... and {len(todo) - 10} more")
        est = len(todo) / 280.0
        print(f"estimated wall time at 280 queries/hr: {est:.1f} h")
        return 0

    identity, password = load_credentials(args)
    session = make_session(identity, password)
    limiter = RateLimiter()

    satcat_path = cache / "satcat.json.gz"
    if not args.skip_satcat and not satcat_path.exists():
        print("fetching satcat ...", flush=True)
        text = st_get(session, satcat_url(), limiter)
        with gzip.open(satcat_path, "wt", encoding="utf-8") as fh:
            fh.write(text)
        print(f"  wrote {satcat_path} ({satcat_path.stat().st_size/1e6:.1f} MB)")

    done = 0
    failed = []
    for nid, lo, hi, path in todo:
        s = mjd_to_datetime(lo - args.pad_days)
        e = mjd_to_datetime(hi + args.pad_days)
        print(f"[{done+1}/{len(todo)}] {path.name}  "
              f"EPOCH {s:%Y-%m-%d %H:%M} -- {e:%Y-%m-%d %H:%M}", flush=True)
        # One night's failure -- a 204, a network error that outlasts
        # st_get's retries, whatever -- must not abort the rest of the
        # batch. Previously an uncaught exception here killed every
        # remaining night in `todo`, which is especially bad when the
        # caller has already deleted the old file for a targeted refetch:
        # the night ends up with *nothing* cached, worse than before.
        try:
            text = st_get(session, gp_history_url(s, e), limiter)
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
            print(f"    -> {path.name} left unfetched; re-run to retry just "
                  "this night (already-cached nights are skipped)")
            failed.append((nid, path.name, str(exc)))
            continue
        n_rec = text.count("\n1 ")
        if n_rec == 0:
            # A genuinely empty response is exceptionally unlikely for a
            # whole-catalog window (tens of thousands of objects; something
            # is normally updated every day) -- and at wider pad_days this
            # was directly observed to actually mean "query rejected as too
            # expensive" (HTTP 204) rather than "confirmed nothing here"
            # (CLAUDE.md gotcha 7). NEVER cache this as if it were real: a
            # written empty file reads as "already fetched" on every future
            # run and would silently and permanently erase whatever good,
            # narrower data used to be at this path. Leave it unfetched so
            # it gets retried instead.
            print("    WARNING: zero records returned -- NOT caching this "
                  "(would silently erase any existing data at this path); "
                  "left unfetched for retry", flush=True)
            failed.append((nid, path.name, "zero records returned"))
            continue
        # tmp + rename: never leave `path` half-written, and never remove an
        # existing good file before its replacement is confirmed on disk.
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            fh.write(text)
        tmp.replace(path)
        print(f"    {n_rec} element sets, {path.stat().st_size/1e6:.1f} MB")
        done += 1
        if args.max_nights and done >= args.max_nights:
            print("reached --max-nights, stopping")
            break

    if failed:
        print(f"\n{len(failed)}/{len(todo)} nights FAILED and are still "
              "missing from the cache:")
        for nid, name, exc in failed:
            print(f"  {nid}  {name}  {exc.splitlines()[0] if exc else ''}")
        print("re-run this same command to retry just the failed nights -- "
              "everything else already fetched will be skipped")

    print("done." if not failed else "done, with failures (see above).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
