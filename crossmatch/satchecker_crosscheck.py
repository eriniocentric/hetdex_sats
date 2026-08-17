#!/usr/bin/env python
"""
satchecker_crosscheck.py -- independent validation of the Space-Track matches
using the IAU CPS SatChecker FOV API.

SatChecker does the SGP4 propagation and the frame transformations server-side,
using its own TLE archive, so agreement between it and match_streaks.py is a
genuine end-to-end check of the identification: element-set selection, time
convention, frame handling and scoring all have to be right on both sides.

Coverage caveat: SatChecker's TLE archive begins in July 2019 (MJD 58665), so
the ~55 pre-2019.5 streaks in HETDEX PDR1 cannot be cross-checked this way and
are skipped.

This is a *sampling* tool, not a second full pipeline -- each streak costs an
asynchronous job plus polling.  Default is 40 randomly chosen matched streaks,
which is plenty to establish agreement.

Usage
-----
    python satchecker_crosscheck.py --matched HETDEX_PDR1_sats_matched.fits \
                                    --catalog HETDEX_PDR1_sats.fits \
                                    --n-sample 40 --out satchecker_check.csv

API reference: https://satchecker.readthedocs.io/en/stable/fov.html
"""

from __future__ import annotations

import argparse
import csv
import sys
import time

import numpy as np

import satstreak_core as core
from satstreak_core import MatchConfig

BASE = "https://satchecker.cps.iau.org"
PASSES = f"{BASE}/fov/satellite-passes/"
STATUS = f"{BASE}/fov/task-status/"
MJD0 = 2400000.5
SATCHECKER_TLE_START_MJD = 58665.0      # 2019-07-01


def request_passes(session, lat, lon, elev, start_mjd, duration_s,
                   ra, dec, radius_deg, poll_s=3.0, timeout_s=300.0):
    """Submit an FOV job and poll until it finishes.  Returns the satellites
    dict, or {} on failure."""
    params = dict(latitude=lat, longitude=lon, elevation=elev,
                  start_time_jd=start_mjd + MJD0, duration=duration_s,
                  ra=ra, dec=dec, fov_radius=radius_deg,
                  group_by="satellite")
    r = session.get(PASSES, params=params, timeout=120)
    r.raise_for_status()
    payload = r.json()
    if isinstance(payload, list):
        payload = payload[0]

    if "task_id" not in payload:                    # synchronous reply
        return payload.get("data", {}).get("satellites", {})

    task_id = payload["task_id"]
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(poll_s)
        s = session.get(STATUS + task_id, timeout=60)
        s.raise_for_status()
        st = s.json()
        if isinstance(st, list):
            st = st[0]
        state = str(st.get("status", "")).upper()
        if state == "SUCCESS":
            return st.get("data", {}).get("satellites", {})
        if state in ("FAILURE", "REVOKED"):
            return {}
    return {}


def best_match(satellites, row, cfg):
    """Score every satellite SatChecker returned against the streak segment."""
    best = None
    for key, sat in satellites.items():
        pos = sat.get("positions", [])
        if len(pos) < 3:
            continue
        ra = np.array([p["ra"] for p in pos], dtype=float)
        dec = np.array([p["dec"] for p in pos], dtype=float)
        mjd = np.array([p["julian_date"] for p in pos], dtype=float) - MJD0
        order = np.argsort(mjd)
        units = core.radec_to_unit(ra[order], dec[order])
        res = core.score_track(units, mjd[order],
                               row["ra_start"], row["dec_start"],
                               row["ra_end"], row["dec_end"], cfg)
        if res is None:
            continue
        res["norad"] = int(sat.get("norad_id", -1))
        res["name"] = sat.get("name", key)
        if best is None or res["score"] < best["score"]:
            best = res
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default="HETDEX_PDR1_sats.fits")
    ap.add_argument("--matched", default="HETDEX_PDR1_sats_matched.fits")
    ap.add_argument("--out", default="satchecker_check.csv")
    ap.add_argument("--n-sample", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fov-radius", type=float, default=1.0,
                    help="search radius handed to SatChecker, deg")
    ap.add_argument("--max-perp", type=float, default=900.0)
    ap.add_argument("--max-pa", type=float, default=6.0)
    ap.add_argument("--sleep", type=float, default=2.0,
                    help="pause between jobs, to stay polite")
    ap.add_argument("--unmatched", action="store_true",
                    help="search SatChecker for streaks this pipeline did "
                         "NOT match, instead of validating streaks it did. "
                         "Verdict is 'found' / 'not-found' rather than "
                         "'agree' / 'DISAGREE' / 'no-satchecker-match'; "
                         "'found' rows carry a candidate norad_id for "
                         "match_streaks.rematch_by_norad to re-score against "
                         "our own geometry. Pass a large --n-sample to cover "
                         "every unmatched streak rather than a random "
                         "subset -- min(n_sample, pool size) is taken, so "
                         "any n_sample >= the unmatched count runs all of "
                         "them.")
    args = ap.parse_args(argv)

    import requests
    from astropy.io import fits
    from astropy.table import Table, join

    cfg = MatchConfig(max_perp_arcsec=args.max_perp, max_pa_deg=args.max_pa)

    info = Table.read(args.catalog, hdu="INFO")
    match = Table.read(args.matched, hdu="MATCH")
    with fits.open(args.catalog) as h:
        ph = h[0].header
        lat = ph.get("SITELAT", 30.681436)
        lon = ph.get("SITELONG", -104.014744)
        elev = ph.get("SITEELEV", 2026.0)

    keep = [c for c in ("streak_id", "norad_id", "object_name",
                        "match_perp_arcsec", "match_pa_diff_deg",
                        "crossing_dt_s", "matched") if c in match.colnames]
    # metadata_conflicts="silent": both HDUs carry an EXTNAME meta key, which
    # astropy cannot merge.  Nothing here reads it, so the warning is noise.
    tbl = join(info, match[keep], keys="streak_id",
               metadata_conflicts="silent")

    ok = (tbl["mjd_shot"] >= SATCHECKER_TLE_START_MJD)
    if "matched" in tbl.colnames:
        ok &= (~tbl["matched"] if args.unmatched else tbl["matched"])
    pool = np.where(ok)[0]
    n_skip = int((tbl["mjd_shot"] < SATCHECKER_TLE_START_MJD).sum())
    kind = "unmatched" if args.unmatched else "matched"
    print(f"{len(pool)} {kind} streaks eligible; "
          f"{n_skip} predate SatChecker's archive", flush=True)
    if len(pool) == 0:
        print("nothing to check")
        return 1

    rng = np.random.default_rng(args.seed)
    sample = rng.choice(pool, size=min(args.n_sample, len(pool)), replace=False)

    session = requests.Session()
    session.headers["User-Agent"] = "HETDEX-satellite-streak-study/1.0"

    rows = []
    agree = disagree = nofind = 0
    interrupted = False
    for i, idx in enumerate(sample, 1):
        row = tbl[int(idx)]
        t_lo, t_hi = cfg.window_mjd(float(row["mjd_shot"]), float(row["exptime"]))
        duration = (t_hi - t_lo) * 86400.0
        try:
            sats = request_passes(session, lat, lon, elev, t_lo, duration,
                                  float(row["ra_cen_spax"]),
                                  float(row["dec_cen_spax"]),
                                  args.fov_radius)
        except KeyboardInterrupt:
            # Ctrl-C / Jupyter interrupt: keep whatever has been collected
            # rather than throwing away several minutes of API calls.
            print(f"\ninterrupted after {i-1} streaks -- writing partial results",
                  flush=True)
            interrupted = True
            break
        except Exception as exc:
            print(f"[{i}/{len(sample)}] streak {row['streak_id']}: API error {exc}")
            continue

        b = best_match(sats, row, cfg)
        st_norad = (int(row["norad_id"])
                   if ("norad_id" in tbl.colnames and not args.unmatched) else -1)
        sc_norad = int(b["norad"]) if b else -1

        if args.unmatched:
            # There is no space-track identification to agree/disagree with
            # by construction (that is why the streak is in this pool) --
            # the only question is whether SatChecker's own archive turned
            # up a scoreable candidate at all.
            if sc_norad < 0:
                nofind += 1
                verdict = "not-found"
            else:
                agree += 1          # reused as the "found" counter below
                verdict = "found"
        else:
            if sc_norad < 0:
                nofind += 1
                verdict = "no-satchecker-match"
            elif sc_norad == st_norad:
                agree += 1
                verdict = "agree"
            else:
                disagree += 1
                verdict = "DISAGREE"

        # flush: this is the only progress indicator during a multi-minute run
        print(f"[{i}/{len(sample)}] streak {int(row['streak_id']):4d}  "
              f"space-track={st_norad:>7d}  satchecker={sc_norad:>7d}  "
              f"n_returned={len(sats):3d}  {verdict}", flush=True)

        st_perp = (float(row["match_perp_arcsec"])
                   if ("match_perp_arcsec" in tbl.colnames and not args.unmatched)
                   else np.nan)
        rows.append(dict(
            streak_id=int(row["streak_id"]), shotid=int(row["shotid"]),
            mjd_shot=float(row["mjd_shot"]),
            spacetrack_norad=st_norad,
            spacetrack_perp_arcsec=st_perp,
            satchecker_norad=sc_norad,
            satchecker_name=b["name"] if b else "",
            satchecker_perp_arcsec=b["perp_arcsec"] if b else np.nan,
            satchecker_pa_diff_deg=b["pa_diff_deg"] if b else np.nan,
            satchecker_n_returned=len(sats),
            verdict=verdict))
        time.sleep(args.sleep)

    if rows:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    n = agree + disagree + nofind
    tail = "  (PARTIAL -- interrupted)" if interrupted else ""
    if args.unmatched:
        print(f"\nfound {agree}/{n} candidate identifications, "
              f"{nofind} with nothing returned  ->  {args.out}{tail}")
    else:
        print(f"\nagree {agree}/{n}, disagree {disagree}, "
              f"no SatChecker match {nofind}  ->  {args.out}{tail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
