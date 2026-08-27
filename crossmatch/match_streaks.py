#!/usr/bin/env python
"""
match_streaks.py -- identify the satellite responsible for each HETDEX streak.

Reads HETDEX_PDR1_sats.fits and the TLE cache built by fetch_tles.py,
propagates every catalogued object over each shot's exposure window with SGP4,
and associates propagated tracks with observed streak segments.

Pipeline per shot
-----------------
1. Pick, for every object with an element set that night, the TLE whose epoch
   is closest to the shot time.
2. Coarse pass: propagate all ~20-30k objects on a 20 s grid across the search
   window, convert TEME -> topocentric GCRS, keep anything that comes within
   `--coarse-radius` degrees of the field.
3. Fine pass: re-propagate the survivors on a 0.5 s grid.
4. Score each candidate track against each streak segment on
     (a) perpendicular offset of the streak midpoint from the model great
         circle, and
     (b) position-angle agreement,
   both computed spherically from the streak *endpoints*, so no convention
   from the input catalog is assumed.
5. Record the best match plus the next `--keep` candidates, with geometry,
   illumination and orbit class.

Frames and time
---------------
* TLE epochs and mjd_shot are UTC; SGP4 is evaluated on UTC Julian dates.
* SGP4 returns TEME.  TEME -> GCRS is a pure rotation, computed once per shot
  (it varies only through precession/nutation, i.e. negligibly over 20 min).
* The observer's GCRS position is computed per timestep from the HET site
  coordinates in the catalog header, and subtracted to give the *topocentric*
  direction.  No aberration correction is applied, deliberately: the HETDEX
  WCS is tied to catalog star positions, so the same annual aberration is
  removed from the satellite direction by the astrometric solution.  The
  residual error is second order, <0.1".
* GCRS and ICRS axes are identical by construction, so the resulting RA/Dec is
  directly comparable to the catalog's.

Accuracy expectation
--------------------
A TLE ~12 h from epoch is typically good to a few km, dominated by along-track
error.  At LEO ranges that is arcminutes along the track but only ~0.1-1' across
it, which is why the perpendicular offset -- not the separation -- is the
primary discriminant.  Default tolerances (900" perpendicular, 6 deg in PA) are
loose on purpose; tighten them after inspecting the residual distributions.

Usage
-----
    python match_streaks.py --catalog HETDEX_PDR1_sats.fits \
                            --cache-dir tle_cache \
                            --out HETDEX_PDR1_sats_matched.fits

    python match_streaks.py ... --limit-nights 3     # quick trial
    python match_streaks.py ... --n-jobs 4

Requires: numpy, astropy, sgp4  (pip install sgp4), optionally joblib.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import satstreak_core as core
from satstreak_core import MatchConfig

MJD0 = 2400000.5


# --------------------------------------------------------- survey timing
def load_survey_timing(h5_path):
    """Load precise per-dither shutter timing from survey_hdr5.h5.

    Reconstructs float64-precision dither start times from the ``date`` and
    ``time`` columns (which together give sub-second accuracy), combined with
    ``darktime`` (wall-clock time per dither = exposure + readout) and
    ``exptime``.  The float32 ``mjd`` column in the survey file has a
    quantisation step of ~338 s — large enough to shift the search window past
    a real crossing.

    Returns
    -------
    dict : shotid (int) -> dict with keys
        dither_open  : np.ndarray, shape (3,), float64 — MJD shutter-open
        dither_close : np.ndarray, shape (3,), float64 — MJD shutter-close
        fwhm_virus   : float — seeing FWHM (arcsec), NaN if unavailable
    """
    import tables as tb
    from astropy.time import Time as AstTime

    h5 = tb.open_file(str(h5_path), "r")
    survey = h5.root.Survey
    out = {}
    for row in survey.iterrows():
        shotid = int(row["shotid"])
        if shotid == 0:
            continue
        # Precise MJD of dither-0 start from date + time
        d = str(int(row["date"]))
        t = row["time"]
        t = t.decode() if isinstance(t, bytes) else str(t)
        if len(t.strip()) < 5:
            continue
        hh, mm = int(t[0:2]), int(t[2:4])
        ss_frac = float(t[4:]) / 10.0
        iso = f"{d[:4]}-{d[4:6]}-{d[6:8]}T{hh:02d}:{mm:02d}:{ss_frac:06.3f}"
        t0 = AstTime(iso, format="isot", scale="utc").mjd

        dt = np.asarray(row["darktime"], dtype=np.float64)
        et = np.asarray(row["exptime"], dtype=np.float64)

        opens = np.empty(3, dtype=np.float64)
        closes = np.empty(3, dtype=np.float64)
        opens[0] = t0
        opens[1] = t0 + dt[0] / 86400.0
        opens[2] = t0 + (dt[0] + dt[1]) / 86400.0
        closes[0] = opens[0] + et[0] / 86400.0
        closes[1] = opens[1] + et[1] / 86400.0
        closes[2] = opens[2] + et[2] / 86400.0

        fwhm = float(row["fwhm_virus"])
        if fwhm <= 0:
            fwhm = np.nan
        out[shotid] = dict(dither_open=opens, dither_close=closes,
                           fwhm_virus=fwhm)
    h5.close()
    return out


def apply_survey_timing(info, survey_timing, sat_tracks_path=None):
    """Patch *info* table in-place with precise timing from the survey file.

    Adds or replaces the following columns:

    * ``mjd_shot``     — precise MJD of dither-0 shutter open (float64)
    * ``dither_open``   — (N, 3) precise shutter-open MJD per dither
    * ``dither_close``  — (N, 3) precise shutter-close MJD per dither
    * ``expnum``        — dither number (1-3) the streak was detected in
    * ``shot_span_s``   — total time first-open to last-close (seconds)

    ``expnum`` is looked up by joining on (shotid, streak_slope,
    streak_intercept) with satellite_tracks.txt.  Falls back to 0 (unknown)
    for unmatched rows.
    """
    n = len(info)
    d_open = np.full((n, 3), np.nan, dtype=np.float64)
    d_close = np.full((n, 3), np.nan, dtype=np.float64)
    shot_span = np.full(n, np.nan, dtype=np.float64)
    fwhm = np.full(n, np.nan, dtype=np.float64)

    for i, row in enumerate(info):
        sid = int(row["shotid"])
        st = survey_timing.get(sid)
        if st is None:
            continue
        d_open[i] = st["dither_open"]
        d_close[i] = st["dither_close"]
        info["mjd_shot"][i] = st["dither_open"][0]
        shot_span[i] = (st["dither_close"][2] - st["dither_open"][0]) * 86400.0
        fwhm[i] = st.get("fwhm_virus", np.nan)

    info["dither_open"] = d_open
    info["dither_close"] = d_close
    info["shot_span_s"] = shot_span
    info["fwhm_virus"] = fwhm

    # --- look up expnum from satellite_tracks.txt ---
    expnum = np.zeros(n, dtype=np.int16)
    if sat_tracks_path is not None:
        from astropy.table import Table as AstTable
        sat_tab = AstTable.read(str(sat_tracks_path), format="ascii",
                                names=["shotid", "expnum", "slope", "intercept"])
        sat_tab["shotid"] = sat_tab["shotid"].astype(np.int64)
        for i, row in enumerate(info):
            sid = int(row["shotid"])
            sel = ((sat_tab["shotid"] == sid)
                   & (np.abs(sat_tab["slope"] - float(row["streak_slope"])) < 1e-4)
                   & (np.abs(sat_tab["intercept"]
                             - float(row["streak_intercept"])) < 1e-4))
            matches = sat_tab[sel]
            if len(matches):
                expnum[i] = int(matches["expnum"][0])
    info["expnum"] = expnum

    # Scalar MJD open/close for the specific dither each streak was detected
    # in.  These go into the publication table (MRT cannot hold shape-3 arrays).
    dith_open_scalar = np.full(n, np.nan, dtype=np.float64)
    dith_close_scalar = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        e = int(expnum[i])
        if e >= 1 and np.isfinite(d_open[i, e - 1]):
            dith_open_scalar[i] = d_open[i, e - 1]
            dith_close_scalar[i] = d_close[i, e - 1]
    info["dither_mjd_open"] = dith_open_scalar
    info["dither_mjd_close"] = dith_close_scalar


def shutter_open_grid(dither_open, dither_close, step_s, margin_s=0.0):
    """Build a sorted time array covering only shutter-open intervals.

    Parameters
    ----------
    dither_open, dither_close : array-like, shape (3,)
        MJD of shutter open / close per dither.
    step_s : float
        Time step in seconds.
    margin_s : float
        Margin to add before/after each interval (seconds).

    Returns
    -------
    mjd : np.ndarray, float64 — sorted time samples
    """
    margin = margin_s / 86400.0
    parts = []
    for d in range(3):
        t_lo = dither_open[d] - margin
        t_hi = dither_close[d] + margin
        n = max(2, int(np.ceil((t_hi - t_lo) * 86400.0 / step_s)) + 1)
        parts.append(np.linspace(t_lo, t_hi, n))
    return np.concatenate(parts)


# ------------------------------------------------------------- TLE handling
def tle_epoch_to_mjd(line1):
    """Epoch of a TLE line 1 as MJD (UTC)."""
    yy = int(line1[18:20])
    doy = float(line1[20:32])
    year = 2000 + yy if yy < 57 else 1900 + yy
    dt = datetime(year, 1, 1) + timedelta(days=doy - 1.0)
    return (dt - datetime(1858, 11, 17)).total_seconds() / 86400.0


def load_3le(path):
    """Parse a gzipped 3LE file -> list of dicts."""
    opener = gzip.open if str(path).endswith(".gz") else open
    out = []
    name = None
    l1 = None
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if line.startswith("0 "):
                name = line[2:].strip()
                l1 = None
            elif line.startswith("1 ") and len(line) >= 69:
                l1 = line
            elif line.startswith("2 ") and len(line) >= 69 and l1 is not None:
                try:
                    norad = int(l1[2:7])
                    epoch = tle_epoch_to_mjd(l1)
                except ValueError:
                    l1 = None
                    continue
                out.append(dict(norad=norad, name=name or f"NORAD {norad}",
                                line1=l1, line2=line, epoch_mjd=epoch))
                l1 = None
    return out


def select_nearest_epoch(records, t_mjd):
    """One element set per object: the one with epoch closest to t_mjd."""
    best = {}
    for rec in records:
        n = rec["norad"]
        dt = abs(rec["epoch_mjd"] - t_mjd)
        if n not in best or dt < best[n][0]:
            best[n] = (dt, rec)
    return [v[1] for v in best.values()]


# Only these satcat fields are used; keeping just them shrinks the table from
# ~60 MB to ~6 MB, which matters because it crosses a process boundary.
SATCAT_FIELDS = ("OBJECT_NAME", "SATNAME", "OBJECT_ID", "OBJECT_TYPE",
                 "COUNTRY", "LAUNCH", "RCS_SIZE")

_SATCAT_MEMO = {}


def load_satcat(cache_dir):
    """NORAD ID -> metadata dict, from the cached Space-Track satcat."""
    path = Path(cache_dir) / "satcat.json.gz"
    if not path.exists():
        print(f"  note: {path} not found; object metadata will be TLE-name only")
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rows = json.load(fh)
    out = {}
    for r in rows:
        try:
            out[int(r["NORAD_CAT_ID"])] = {k: r[k] for k in SATCAT_FIELDS
                                           if k in r}
        except (KeyError, TypeError, ValueError):
            continue
    return out


def get_satcat(source):
    """Accept either a loaded satcat dict or a cache directory.

    Passing the directory is strongly preferred when n_jobs > 1: joblib pickles
    every argument once per task, so handing a 60k-entry dict to 460 tasks
    would ship it across the process boundary 460 times.  Given a path, each
    worker loads it once and memoises it here for the life of the process.
    """
    if isinstance(source, dict):
        return source
    key = str(source)
    if key not in _SATCAT_MEMO:
        _SATCAT_MEMO[key] = load_satcat(key)
    return _SATCAT_MEMO[key]


# ------------------------------------------------------------ astropy glue
def teme_to_gcrs_matrix(t):
    """3x3 rotation M with v_gcrs = M @ v_teme, at a single astropy Time."""
    import astropy.units as u
    from astropy.coordinates import GCRS, TEME, CartesianRepresentation
    basis = CartesianRepresentation(np.eye(3) * u.km)
    g = TEME(basis, obstime=t).transform_to(GCRS(obstime=t))
    M = np.asarray(g.cartesian.xyz.to_value(u.km))
    # sanity: must be orthonormal
    err = np.abs(M @ M.T - np.eye(3)).max()
    if err > 1e-8:
        raise RuntimeError(f"TEME->GCRS matrix not orthonormal (err={err:.2e})")
    return M


def observer_and_sun(times, location):
    """Geocentric GCRS positions (km) of the observer and the Sun."""
    import astropy.units as u
    from astropy.coordinates import get_sun
    obs = location.get_gcrs_posvel(times)[0].xyz.to_value(u.km).T   # (nt, 3)
    sun = get_sun(times).cartesian.xyz.to_value(u.km).T             # (nt, 3)
    return np.ascontiguousarray(obs), np.ascontiguousarray(sun)


def propagate(satrecs, times_mjd):
    """SGP4 for many objects at many times.  Returns TEME positions (km) with
    shape (nsat, nt, 3) and a boolean mask of usable results."""
    from sgp4.api import SatrecArray
    jd = np.floor(np.asarray(times_mjd, float) + MJD0 - 0.5) + 0.5
    fr = (np.asarray(times_mjd, float) + MJD0) - jd
    arr = SatrecArray(satrecs)
    err, r, _v = arr.sgp4(jd, fr)
    return r, err == 0


def site_location(site_geodetic):
    """astropy EarthLocation from the (lat, lon, height_m) header triple."""
    import astropy.units as u
    from astropy.coordinates import EarthLocation
    lat, lon, height = site_geodetic
    return EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, height * u.m)


def build_satrecs(records, mjd_shot, norads=None, cache=None):
    """Nearest-epoch element set per object -> (satrecs, metadata).

    `cache` is an optional dict reused across the shots of one night.  Shots
    hours apart usually select the same element set for most objects, so this
    avoids re-running twoline2rv on ~25k TLEs for every shot.
    """
    from sgp4.api import Satrec
    chosen = select_nearest_epoch(records, mjd_shot)
    if norads is not None:
        want = set(int(n) for n in norads)
        chosen = [r for r in chosen if r["norad"] in want]
    satrecs, meta = [], []
    for rec in chosen:
        key = (rec["norad"], rec["epoch_mjd"])
        sr = cache.get(key) if cache is not None else None
        if sr is None:
            try:
                sr = Satrec.twoline2rv(rec["line1"], rec["line2"])
            except Exception:
                continue
            if cache is not None:
                cache[key] = sr
        satrecs.append(sr)
        meta.append(rec)
    return satrecs, meta


def shot_tracks(shot_rows, records, site_geodetic, cfg, norads,
                step_s=None):
    """Topocentric tracks of specific objects over a shot's search window.

    Returns {norad: dict(mjd, ra, dec, range_km, alt_deg)}.  Used by the
    notebook to overlay candidate tracks on an observed streak; not part of
    the matching path.
    """
    from astropy.time import Time

    location = site_location(site_geodetic)
    mjd_shot = float(shot_rows[0]["mjd_shot"])
    exptime = float(shot_rows[0]["exptime"])
    step = step_s if step_s else cfg.fine_step_s

    d_open = shot_rows[0].get("dither_open")
    d_close = shot_rows[0].get("dither_close")
    has_timing = (d_open is not None
                  and np.all(np.isfinite(d_open))
                  and np.all(np.isfinite(d_close)))
    if has_timing:
        mjd = shutter_open_grid(
            np.asarray(d_open, np.float64),
            np.asarray(d_close, np.float64),
            step, margin_s=cfg.margin_before_s)
        t_lo, t_hi = mjd[0], mjd[-1]
    else:
        t_lo, t_hi = cfg.window_mjd(mjd_shot, exptime)
        n = max(8, int(np.ceil((t_hi - t_lo) * 86400.0 / step)) + 1)
        mjd = np.linspace(t_lo, t_hi, n)

    satrecs, meta = build_satrecs(records, mjd_shot, norads=norads)
    if not satrecs:
        return {}
    times = Time(mjd, format="mjd", scale="utc")
    M = teme_to_gcrs_matrix(Time(0.5 * (t_lo + t_hi), format="mjd", scale="utc"))
    obs, _sun = observer_and_sun(times, location)
    up = _observer_up_gcrs(location, times)

    r_teme, ok = propagate(satrecs, mjd)
    r_gcrs = np.einsum("ij,ntj->nti", M, r_teme)
    topo = r_gcrs - obs[None, :, :]
    rng = np.linalg.norm(topo, axis=-1)
    u_topo = topo / np.where(rng[..., None] > 0, rng[..., None], 1.0)

    out = {}
    for j, rec in enumerate(meta):
        good = ok[j]
        if good.sum() < 2:
            continue
        ra, dec = core.unit_to_radec(u_topo[j][good])
        out[rec["norad"]] = dict(
            name=rec["name"], mjd=mjd[good], ra=ra, dec=dec,
            range_km=rng[j][good],
            alt_deg=90.0 - core.angsep(u_topo[j][good], up[good]))
    return out


# --------------------------------------------------------------- one shot
def _shot_time_grids(shot_rows, cfg):
    """Return (mjd_shot, t_lo, t_hi, mjd_coarse, mjd_fine).

    When precise per-dither timing is available in the row dicts (from
    ``apply_survey_timing``), the grids cover only shutter-open intervals —
    readout gaps between dithers are excluded.  Otherwise falls back to one
    continuous window from ``cfg.window_mjd``.
    """
    mjd_shot = float(shot_rows[0]["mjd_shot"])
    exptime = float(shot_rows[0]["exptime"])

    d_open = shot_rows[0].get("dither_open")
    d_close = shot_rows[0].get("dither_close")
    has_timing = (d_open is not None
                  and np.all(np.isfinite(d_open))
                  and np.all(np.isfinite(d_close)))

    if has_timing:
        d_open = np.asarray(d_open, dtype=np.float64)
        d_close = np.asarray(d_close, dtype=np.float64)
        mjd_coarse = shutter_open_grid(
            d_open, d_close, cfg.coarse_step_s, margin_s=cfg.margin_before_s)
        mjd_fine = shutter_open_grid(
            d_open, d_close, cfg.fine_step_s, margin_s=cfg.margin_before_s)
        t_lo = mjd_coarse[0]
        t_hi = mjd_coarse[-1]
    else:
        t_lo, t_hi = cfg.window_mjd(mjd_shot, exptime)
        n_coarse = max(4, int(np.ceil(
            (t_hi - t_lo) * 86400.0 / cfg.coarse_step_s)) + 1)
        mjd_coarse = np.linspace(t_lo, t_hi, n_coarse)
        n_fine = max(8, int(np.ceil(
            (t_hi - t_lo) * 86400.0 / cfg.fine_step_s)) + 1)
        mjd_fine = np.linspace(t_lo, t_hi, n_fine)

    return mjd_shot, t_lo, t_hi, mjd_coarse, mjd_fine


def process_shot(shot_rows, records, site_geodetic, cfg, satcat,
                 satrec_cache=None):
    """Match every streak in one shot.  Returns a list of result dicts."""
    from astropy.time import Time

    satcat = get_satcat(satcat)
    location = site_location(site_geodetic)

    mjd_shot, t_lo, t_hi, mjd_coarse, mjd_fine = _shot_time_grids(
        shot_rows, cfg)
    exptime = float(shot_rows[0]["exptime"])

    satrecs, meta = build_satrecs(records, mjd_shot, cache=satrec_cache)
    if not satrecs:
        return []

    # ---- coarse pass
    t_coarse = Time(mjd_coarse, format="mjd", scale="utc")
    M = teme_to_gcrs_matrix(Time(0.5 * (t_lo + t_hi), format="mjd", scale="utc"))
    obs_c, _sun_c = observer_and_sun(t_coarse, location)

    # circular mean, so a field straddling RA = 0 does not collapse to RA = 180
    field = core.normalize(np.mean(
        [core.radec_to_unit(row["ra_cen_spax"], row["dec_cen_spax"])
         for row in shot_rows], axis=0))

    # Chunked over objects: only the per-object minimum separation is kept, so
    # peak memory is set by sat_chunk rather than by the catalog size.  This is
    # what lets you raise n_jobs without running the workers out of memory.
    n_sat = len(satrecs)
    min_sep = np.full(n_sat, np.inf)
    threshold = np.full(n_sat, cfg.coarse_radius_deg)
    step = max(1, int(cfg.sat_chunk))
    for start in range(0, n_sat, step):
        sl = slice(start, min(start + step, n_sat))
        r_teme, ok = propagate(satrecs[sl], mjd_coarse)
        topo = np.einsum("ij,ntj->nti", M, r_teme) - obs_c[None, :, :]
        u_c = core.normalize(topo)
        sep = core.angsep(u_c, field[None, None, :])         # (nchunk, nt)
        min_sep[sl] = np.min(np.where(ok, sep, np.inf), axis=1)
        # widen per object for motion between coarse samples -- see
        # core.coarse_threshold for why a fixed radius silently drops fast
        # LEO crossings
        threshold[sl] = core.coarse_threshold(cfg.coarse_radius_deg, u_c, ok)
        del r_teme, topo, u_c, sep

    cand = np.where(min_sep < threshold)[0]
    if cand.size == 0:
        return [_no_match(row, mjd_shot, len(satrecs)) for row in shot_rows]

    # ---- fine pass, survivors only
    t_fine = Time(mjd_fine, format="mjd", scale="utc")
    obs_f, sun_f = observer_and_sun(t_fine, location)

    sub = [satrecs[i] for i in cand]
    r_teme_f, ok_f = propagate(sub, mjd_fine)
    r_gcrs_f = np.einsum("ij,ntj->nti", M, r_teme_f)
    topo_f = r_gcrs_f - obs_f[None, :, :]
    rng_f = np.linalg.norm(topo_f, axis=-1)
    u_f = topo_f / np.where(rng_f[..., None] > 0, rng_f[..., None], 1.0)

    # geodetic "up" at the observer, for altitude; geocentric-up would be off
    # by up to 0.19 deg at HET's latitude.
    up_gcrs = _observer_up_gcrs(location, t_fine)

    results = []
    for row in shot_rows:
        scored = []
        for j in range(len(sub)):
            good = ok_f[j]
            if good.sum() < 3:
                continue
            res = core.score_track(u_f[j][good], mjd_fine[good],
                                   row["ra_start"], row["dec_start"],
                                   row["ra_end"], row["dec_end"], cfg)
            if res is None:
                continue
            gi = np.where(good)[0][res["idx"]]      # index in the full fine grid
            alt = 90.0 - core.angsep(u_f[j, gi], up_gcrs[gi])
            if alt < cfg.min_altitude_deg:
                continue
            illum = int(core.illumination_state(r_gcrs_f[j, gi], sun_f[gi]))
            if cfg.require_sunlit and illum != core.ILLUM_SUNLIT:
                continue
            rec = meta[cand[j]]
            res.update(
                norad=rec["norad"], name=rec["name"],
                tle_epoch_mjd=rec["epoch_mjd"],
                tle_age_hours=(mjd_shot - rec["epoch_mjd"]) * 24.0,
                range_km=float(rng_f[j, gi]),
                alt_deg=float(alt),
                illum=illum,
                phase_deg=float(core.phase_angle(r_gcrs_f[j, gi], obs_f[gi], sun_f[gi])),
                sat_geocentric_km=float(np.linalg.norm(r_gcrs_f[j, gi])),
                sun_alt_deg=float(90.0 - core.angsep(
                    core.normalize(sun_f[gi] - obs_f[gi]), up_gcrs[gi])),
                mean_motion=float(satrecs[cand[j]].no_kozai * 1440.0 / (2 * np.pi)),
                ecc=float(satrecs[cand[j]].ecco),
                inc_deg=float(np.degrees(satrecs[cand[j]].inclo)),
            )
            scored.append(res)

        scored.sort(key=lambda d: d["score"])
        results.append(_pack(row, scored, mjd_shot, exptime, len(satrecs),
                             len(sub), cfg, satcat))
    return results


# Columns of INFO that the matcher actually reads.
NEEDED_COLS = ("streak_id", "shotid", "mjd_shot", "exptime",
               "ra_cen_spax", "dec_cen_spax",
               "ra_start", "dec_start", "ra_end", "dec_end",
               "seg_len_arcsec", "streak_pa", "g_mag", "area_arcsec2")

# Optional columns added by apply_survey_timing; included in row dicts when
# present but never required (backwards-compatible with old INFO tables).
_TIMING_COLS = ("dither_open", "dither_close", "expnum", "shot_span_s",
                "fwhm_virus")


def run_night(path, shot_groups, site, cfg, satcat):
    """Match every shot taken on one night.

    Module level so joblib can pickle it; `shot_groups` is a list of lists of
    plain dicts, and `satcat` is normally the cache directory (see get_satcat).
    One Satrec cache is shared by all shots on the night.
    """
    if path is None:
        return [_no_match(r, r["mjd_shot"], 0)
                for rows in shot_groups for r in rows]
    satcat = get_satcat(satcat)
    records = load_3le(path)
    satrec_cache = {}
    out = []
    for rows in shot_groups:
        out.extend(process_shot(rows, records, site, cfg, satcat,
                                satrec_cache=satrec_cache))
    return out


def _observer_up_gcrs(location, times):
    """Unit geodetic zenith vector at the observer, in GCRS, per timestep."""
    import astropy.units as u
    from astropy.coordinates import AltAz, GCRS, SkyCoord
    zen = SkyCoord(alt=np.full(len(times), 90.0) * u.deg,
                   az=np.zeros(len(times)) * u.deg,
                   frame=AltAz(obstime=times, location=location))
    # GCRS(obstime=times), not the string "gcrs": the latter builds a frame at
    # the default epoch, which makes the chain detour through ICRS and apply
    # then remove aberration at two different times (~40" of slop).
    g = zen.transform_to(GCRS(obstime=times))
    return np.ascontiguousarray(g.cartesian.xyz.value.T)


def _no_match(row, mjd_shot, n_prop):
    return dict(streak_id=int(row["streak_id"]), shotid=int(row["shotid"]),
                n_propagated=n_prop, n_candidates=0, norad_id=-1)


def _pack(row, scored, mjd_shot, exptime, n_prop, n_close, cfg, satcat):
    out = dict(streak_id=int(row["streak_id"]), shotid=int(row["shotid"]),
               n_propagated=n_prop, n_close=n_close, n_candidates=len(scored))
    if not scored:
        out["norad_id"] = -1
        return out

    b = scored[0]
    sc = satcat.get(b["norad"], {})
    _, peri, apo = core.orbit_geometry(b["mean_motion"], b["ecc"])
    name = sc.get("OBJECT_NAME") or sc.get("SATNAME") or b["name"]

    # NOTE: deliberately *not* `exptime` (the per-row catalog column, which
    # varies 366.9-728.0 s and is shutter/overhead bookkeeping). HETDEX flux
    # calibration normalizes every shot's spectrum to a fixed 360 s
    # reference regardless of the actual dither length, so that is the value
    # `g_mag` implicitly assumes -- core.instantaneous_magnitude defaults to
    # it via HETDEX_CAL_EXPTIME_S. Using `exptime` here previously over- or
    # under-corrected m_inst by up to ~0.8 mag for streaks far from 360 s.
    #
    # area_arcsec2 corrects for incomplete IFU coverage along the streak:
    # fill = area / (STREAK_WIDTH * seg_len), capped at 1.0, then
    # t_cross = fill * seg_len / rate.  STREAK_WIDTH_ARCSEC is calibrated
    # from the 22 single-IFU streaks (no inter-IFU gaps).
    m_inst, t_cross = core.instantaneous_magnitude(
        row["g_mag"], row["seg_len_arcsec"], b["rate_arcsec_s"],
        area_arcsec2=row.get("area_arcsec2"))

    out.update(
        norad_id=int(b["norad"]),
        object_name=str(name)[:40],
        object_id=str(sc.get("OBJECT_ID", ""))[:16],
        object_type=str(sc.get("OBJECT_TYPE", ""))[:12],
        country=str(sc.get("COUNTRY", ""))[:8],
        launch_date=str(sc.get("LAUNCH", ""))[:10],
        rcs_size=str(sc.get("RCS_SIZE", ""))[:8],
        constellation=core.classify_constellation(name),
        orbit_class=core.classify_orbit(peri, apo, b["ecc"]),
        perigee_km=float(peri), apogee_km=float(apo),
        inclination_deg=b["inc_deg"], eccentricity=b["ecc"],
        period_min=float(1440.0 / b["mean_motion"]) if b["mean_motion"] else np.nan,
        # --- match quality
        match_perp_arcsec=b["perp_arcsec"],
        match_sep_arcsec=b["sep_mid_arcsec"],
        match_pa_diff_deg=b["pa_diff_deg"],
        match_end_a_arcsec=b["end_a_arcsec"],
        match_end_b_arcsec=b["end_b_arcsec"],
        match_score=b["score"],
        match_time_offset_s=b["time_offset_s"],
        at_window_edge=bool(b["at_edge"]),
        model_pa_deg=b["model_pa_deg"],
        streak_pa_sph_deg=b["streak_pa_deg"],
        crossing_mjd=b["crossing_mjd"],
        crossing_dt_s=(b["crossing_mjd"] - mjd_shot) * 86400.0,
        tle_epoch_mjd=b["tle_epoch_mjd"],
        tle_age_hours=b["tle_age_hours"],
        # --- geometry / illumination
        range_km=b["range_km"],
        sat_height_km=b["sat_geocentric_km"] - core.R_EARTH_EQ_KM,
        alt_deg=b["alt_deg"],
        sun_alt_deg=b["sun_alt_deg"],
        phase_angle_deg=b["phase_deg"],
        illum_state=b["illum"],
        illum_label=core.ILLUM_LABELS[b["illum"]],
        ang_rate_arcsec_s=b["rate_arcsec_s"],
        ang_rate_deg_s=b["rate_arcsec_s"] / 3600.0,
        # --- derived photometry
        t_cross_s=float(t_cross),
        g_mag_inst=float(m_inst),
        g_mag_inst_550km=float(core.normalize_magnitude_to_range(m_inst, b["range_km"])),
        # --- ambiguity
        second_norad=int(scored[1]["norad"]) if len(scored) > 1 else -1,
        second_score=float(scored[1]["score"]) if len(scored) > 1 else np.nan,
        score_margin=float(scored[1]["score"] - b["score"]) if len(scored) > 1 else np.inf,
        unambiguous=bool(len(scored) == 1 or scored[1]["score"] > 3.0 * b["score"] + 1.0),
    )
    out["_all"] = scored[:cfg.keep_n_candidates]
    return out


# --------------------------------------------------------------------- I/O
def write_output(path, info, results, cfg, catalog_name=""):
    from astropy.io import fits
    from astropy.table import Table

    by_id = {r["streak_id"]: r for r in results}
    n = len(info)

    def col(key, default, dtype):
        vals = [by_id.get(int(s), {}).get(key, default) for s in info["streak_id"]]
        return np.array(vals, dtype=dtype)

    t = Table()
    t["streak_id"] = np.asarray(info["streak_id"], dtype=np.int32)
    t["shotid"] = np.asarray(info["shotid"], dtype=np.int64)
    t["norad_id"] = col("norad_id", -1, np.int32)
    for k, d in [("object_name", ""), ("object_id", ""), ("object_type", ""),
                 ("country", ""), ("launch_date", ""), ("rcs_size", ""),
                 ("constellation", ""), ("orbit_class", ""), ("illum_label", "")]:
        t[k] = col(k, d, "U40")
    for k in ["match_perp_arcsec", "match_sep_arcsec", "match_pa_diff_deg",
              "match_end_a_arcsec", "match_end_b_arcsec", "match_score",
              "model_pa_deg", "streak_pa_sph_deg", "crossing_dt_s",
              "tle_age_hours", "range_km", "sat_height_km", "alt_deg",
              "sun_alt_deg", "phase_angle_deg", "ang_rate_arcsec_s",
              "ang_rate_deg_s", "t_cross_s", "g_mag_inst", "g_mag_inst_550km",
              "perigee_km", "apogee_km", "inclination_deg", "eccentricity",
              "period_min", "second_score", "score_margin",
              "match_time_offset_s"]:
        t[k] = col(k, np.nan, np.float64)
    for k in ["crossing_mjd", "tle_epoch_mjd"]:
        t[k] = col(k, np.nan, np.float64)
    for k in ["n_propagated", "n_close", "n_candidates", "illum_state",
              "second_norad"]:
        t[k] = col(k, 0, np.int32)
    t["unambiguous"] = col("unambiguous", False, bool)
    t["at_window_edge"] = col("at_window_edge", False, bool)
    t["matched"] = t["norad_id"] > 0

    # long-form candidate list, for vetting ambiguous cases
    rows = []
    for r in results:
        for rank, c in enumerate(r.get("_all", [])):
            rows.append((r["streak_id"], rank, c["norad"], str(c["name"])[:40],
                         c["perp_arcsec"], c["pa_diff_deg"], c["score"],
                         c["range_km"], c["rate_arcsec_s"], c["illum"],
                         (c["crossing_mjd"])))
    cnames = ["streak_id", "rank", "norad_id", "object_name", "perp_arcsec",
              "pa_diff_deg", "score", "range_km", "ang_rate_arcsec_s",
              "illum_state", "crossing_mjd"]
    cdtype = [np.int32, np.int16, np.int32, "U40", np.float64, np.float64,
              np.float64, np.float64, np.float64, np.int16, np.float64]
    cand = Table(names=cnames, dtype=cdtype)
    for r_ in rows:
        cand.add_row(r_)

    hdr = fits.Header()
    hdr["ORIGIN"] = "match_streaks.py"
    hdr["DATE"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    hdr["SRCCAT"] = (Path(catalog_name).name if catalog_name else "",
                     "input streak catalog")
    hdr["TLESRC"] = ("space-track gp_history", "archival element sets")
    hdr["COARSER"] = (cfg.coarse_radius_deg, "coarse search radius [deg]")
    hdr["MAXPERP"] = (cfg.max_perp_arcsec, "max perpendicular offset [arcsec]")
    hdr["MAXPA"] = (cfg.max_pa_deg, "max PA difference [deg]")
    hdr["FINESTP"] = (cfg.fine_step_s, "fine propagation step [s]")
    hdr["NDITHER"] = (cfg.n_dither, "dithers assumed per shot")
    hdr["MARGB"] = (cfg.margin_before_s, "search window pad before mjd_shot [s]")
    hdr["MARGA"] = (cfg.margin_after_s, "search window pad after last dither [s]")
    hdr["EXPSPAN"] = (getattr(cfg, "exposure_span_s", 0.0) or 0.0,
                      "shot duration used for the window [s]")
    hdr["MAXTOFF"] = (getattr(cfg, "max_time_offset_s", 0.0),
                      "max along-track time offset [s]")
    hdr["SHUTPRE"] = ("dither_open" in info.colnames,
                      "precise per-dither shutter timing used")
    hdr["NMATCH"] = (int(t["matched"].sum()), "streaks with a match")
    hdr["NUNAMB"] = (int((t["matched"] & t["unambiguous"]).sum()),
                     "unambiguous matches")

    # table_to_hdu (not BinTableHDU) so unicode columns are encoded to ASCII
    # character arrays rather than raising.
    h_match = fits.table_to_hdu(t)
    h_match.name = "MATCH"
    h_cand = fits.table_to_hdu(cand)
    h_cand.name = "CANDIDATES"
    fits.HDUList([fits.PrimaryHDU(header=hdr), h_match, h_cand]).writeto(
        path, overwrite=True)
    return t


def gather_for_plot(streak_ids, info, match_table, cand_table, cache_dir,
                    site, cfg, max_candidates=5):
    """Collect everything `satstreak_plots.plot_match_grid` needs.

    Groups the requested streaks by observing night so each 3LE file is parsed
    once rather than once per streak -- that parse dominates the cost.

    Returns a list of {streak, tracks, best_norad, label} dicts, skipping any
    streak whose night has no cached elements.
    """
    from fetch_tles import night_id

    sid_list = [int(s) for s in streak_ids]
    info_ids = np.asarray(info["streak_id"]).astype(int)
    match_ids = np.asarray(match_table["streak_id"]).astype(int)
    cand_ids = (np.asarray(cand_table["streak_id"]).astype(int)
                if cand_table is not None and len(cand_table) else
                np.array([], dtype=int))
    files = find_night_files(cache_dir)

    by_night = defaultdict(list)
    for sid in sid_list:
        w = np.where(info_ids == sid)[0]
        if not len(w):
            continue
        by_night[int(night_id(info[int(w[0])]["mjd_shot"]))].append(sid)

    out = {}
    for nid, sids in by_night.items():
        path = files.get(nid)
        if path is None:
            continue
        records = load_3le(path)
        extra = tuple(c for c in _TIMING_COLS if c in info.colnames)
        all_cols = NEEDED_COLS + extra
        for sid in sids:
            row = info[int(np.where(info_ids == sid)[0][0])]
            streak = {}
            for k in all_cols:
                v = row[k]
                if hasattr(v, "shape") and v.shape:
                    streak[k] = np.asarray(v, dtype=np.float64)
                elif hasattr(v, "item"):
                    streak[k] = v.item()
                else:
                    streak[k] = v
            shot_rows = []
            for r in info[np.asarray(info["shotid"]).astype(int)
                          == int(row["shotid"])]:
                d = {}
                for k in all_cols:
                    v = r[k]
                    if hasattr(v, "shape") and v.shape:
                        d[k] = np.asarray(v, dtype=np.float64)
                    elif hasattr(v, "item"):
                        d[k] = v.item()
                    else:
                        d[k] = v
                shot_rows.append(d)

            norads = [int(n) for n in
                      np.asarray(cand_table["norad_id"])[cand_ids == sid][:max_candidates]] \
                if len(cand_ids) else []
            tracks = (shot_tracks(shot_rows, records, site, cfg, norads=norads)
                      if norads else {})

            mrow = match_table[int(np.where(match_ids == sid)[0][0])]
            best = int(mrow["norad_id"])
            name = str(mrow["object_name"]).strip() if "object_name" in \
                match_table.colnames else ""
            hgt = float(mrow["sat_height_km"]) if "sat_height_km" in \
                match_table.colnames else np.nan
            label = f"{sid}: {name[:18]}"
            if np.isfinite(hgt):
                label += f"\n{hgt:,.0f} km"
            out[sid] = dict(streak=streak, tracks=tracks,
                            best_norad=best if best > 0 else None,
                            label=label)

    return [out[s] for s in sid_list if s in out]


# ------------------------------------------------------------- diagnostics
def assess(match_table, info=None, exposure_span_s=1320.0):
    """Post-run quality assessment: residuals, purity, and what got missed.

    Prints tolerance recommendations derived from the data rather than from
    guesses, so a second pass can cut chance contamination without losing real
    matches.
    """
    t = match_table
    m = t[t["norad_id"] > 0]
    n, nm = len(t), len(m)
    print(f"matched {nm}/{n} ({100.0*nm/max(n,1):.1f}%), "
          f"{int(t['unambiguous'].sum())} unambiguous\n")
    if nm == 0:
        return

    # --- residuals: how far inside the tolerances are the matches?
    print("residual percentiles (50 / 90 / 99 / max):")
    for col, unit, tol in [("match_perp_arcsec", '"', None),
                           ("match_pa_diff_deg", " deg", None)]:
        v = np.asarray(m[col], float)
        v = v[np.isfinite(v)]
        q = np.percentile(v, [50, 90, 99])
        print(f"  {col:20s} {q[0]:8.2f} {q[1]:8.2f} {q[2]:8.2f} "
              f"{v.max():8.2f}{unit}")
    # Scale from the MEDIAN, not a high percentile.  The upper percentiles are
    # truncated by the cut that produced this table, so "99th x 2" drifts back
    # towards -- or past -- the tolerance already in use and recommends nothing.
    # The core of a genuine match distribution is tight; the tail is where the
    # chance coincidences live.
    perp = np.asarray(m["match_perp_arcsec"], float)
    dpa = np.asarray(m["match_pa_diff_deg"], float)
    perp, dpa = perp[np.isfinite(perp)], dpa[np.isfinite(dpa)]
    p50, a50 = np.median(perp), np.median(dpa)
    sug_p, sug_a = max(60.0, 10 * p50), max(0.5, 10 * a50)
    keep = float(np.mean((perp <= sug_p) & (dpa <= sug_a)))
    print(f"\n  suggested second-pass cuts (10x median, robust to the tail):")
    print(f"    max_perp_arcsec = {sug_p:.0f}    max_pa_deg = {sug_a:.2f}")
    print(f"    would retain {100*keep:.1f}% of current matches")
    print("  chance coincidences scale with max_perp x max_pa, so this is a "
          f"{(900.0*6.0)/(sug_p*sug_a):.0f}x reduction in chance area "
          "relative to the 900\"/6deg defaults")

    # --- purity: umbra matches are physically impossible
    umbra = m[m["illum_state"] == 0]
    print(f"\npurity: {len(umbra)}/{nm} matches in Earth's shadow "
          f"({100.0*len(umbra)/nm:.1f}% false-positive floor)")
    if len(umbra) and "streak_id" in m.colnames:
        print("  false positives to inspect:",
              [int(s) for s in umbra["streak_id"][:10]])

    # --- timing: should populate the exposure, not pile at an edge
    dt = np.asarray(m["crossing_dt_s"], float)
    dt = dt[np.isfinite(dt)]
    if len(dt):
        q = np.percentile(dt, [0, 16, 50, 84, 100])
        print(f"\ncrossing time within the {exposure_span_s:.0f} s shot:")
        print(f"  min {q[0]:.0f}  16% {q[1]:.0f}  median {q[2]:.0f}  "
              f"84% {q[3]:.0f}  max {q[4]:.0f} s")
        frac_early = float((dt < exposure_span_s / 2).mean())
        print(f"  {100*frac_early:.0f}% fall in the first half "
              "(50% would mean uniform across the shot)")
        if q[0] < 5 or q[4] > exposure_span_s - 5:
            print("  WARNING: crossings at the window edge -- widen the window "
                  "and re-check, some real crossings may lie outside it")

    # --- what the matches are
    if "object_type" in m.colnames:
        print("\nobject type (the 'Other' constellation bucket is mostly "
              "debris and spent stages):")
        names, counts = np.unique(m["object_type"], return_counts=True)
        for i in np.argsort(-counts):
            lab = names[i] if str(names[i]).strip() else "(blank)"
            print(f"  {lab:>14s}: {counts[i]:4d}  "
                  f"{100.0*counts[i]/nm:5.1f}%")

    # --- what got missed
    miss = t[t["norad_id"] <= 0]
    if len(miss) and info is not None:
        try:
            key = {int(s): i for i, s in enumerate(info["streak_id"])}
            idx = [key[int(s)] for s in miss["streak_id"] if int(s) in key]
            sub = info[idx]
            print(f"\n{len(miss)} unmatched streaks:")
            for col in ("g_mag", "seg_len_arcsec", "mjd_shot"):
                if col in sub.colnames:
                    a = np.asarray(sub[col], float)
                    b = np.asarray(info[col], float)
                    print(f"  {col:16s} median {np.nanmedian(a):9.2f}  "
                          f"(all streaks: {np.nanmedian(b):9.2f})")
            print("  faint or short streaks are the usual cause; an uncatalogued "
                  "object is the other")
        except Exception as exc:
            print(f"  (could not profile unmatched streaks: {exc})")
    return m



def triage(match_table):
    """Localise a zero-match run using the per-streak counters.

    n_propagated = 0  -> no element sets for that night (TLE cache gap)
    n_close      = 0  -> nothing came within coarse_radius of the field, which
                         points at the time window, the site, or the frames
    n_close > 0 but n_candidates = 0
                      -> objects were near the field but none passed the
                         perpendicular / PA tolerances
    """
    t = match_table
    n = len(t)
    print(f"{n} streaks")
    print(f"  n_propagated == 0 : {int((t['n_propagated'] == 0).sum())}"
          "   (no TLEs loaded for the night)")
    print(f"  n_close      == 0 : {int((t['n_close'] == 0).sum())}"
          "   (nothing near the field -> window/site/frames)")
    both = (t["n_close"] > 0) & (t["n_candidates"] == 0)
    print(f"  near but unscored : {int(both.sum())}"
          "   (tolerances too tight, or wrong window)")
    print(f"  matched           : {int((t['norad_id'] > 0).sum())}")
    for name in ("n_propagated", "n_close", "n_candidates"):
        v = np.asarray(t[name])
        print(f"  {name:14s} min {v.min():6d}  median {int(np.median(v)):6d}"
              f"  max {v.max():6d}")
    return t


def audit_false_positives(match_table, exposure_span_s=1320.0,
                          margin_before_s=60.0, margin_after_s=60.0,
                          guard_s=10.0):
    """Estimate the chance-match rate from crossings outside the shutter.

    The search window deliberately extends `margin_before_s` before the shot
    starts and `margin_after_s` past its end.  **The shutter is closed then**,
    so any match whose fitted crossing falls in those margins is impossible by
    construction -- a chance alignment, not an identification.

    That gives a calibrated false-positive rate for free: measure the match
    density per second in the margins, multiply by the exposure duration, and
    you have the expected number of chance matches contaminating the real ones.
    No simulation and no assumptions required.

    `guard_s` excludes a few seconds either side of the shutter boundaries,
    where timing slop makes the classification ambiguous.
    """
    t = match_table
    m = t[t["norad_id"] > 0]
    dt = np.asarray(m["crossing_dt_s"], float)
    good = np.isfinite(dt)
    m, dt = m[good], dt[good]
    if len(dt) == 0:
        print("no matches to audit")
        return None

    # Matches whose closest approach was pinned to the window boundary are not
    # measurements of a crossing time -- the true encounter lies outside the
    # searched window.  They pile up at the edges and would masquerade as
    # "outside the shutter", corrupting the rate estimate, so exclude them and
    # report them separately.
    if "at_window_edge" in m.colnames:
        edge = np.asarray(m["at_window_edge"], bool)
    else:
        edge = np.zeros(len(dt), bool)
    if edge.any():
        print(f"  note: {int(edge.sum())} matches pinned to the window edge "
              "(true crossing lies outside the window); excluded from the rate")

    before = (dt < -guard_s) & ~edge
    after = (dt > exposure_span_s + guard_s) & ~edge
    outside = before | after
    inside = (~outside) & (~edge)

    t_margin = max(margin_before_s - guard_s, 0) + max(margin_after_s - guard_s, 0)
    n_out, n_in = int(outside.sum()), int(inside.sum())

    print(f"{len(dt)} matches: {n_in} during the exposure, "
          f"{n_out} outside the shutter")
    print(f"  margin sampled: {t_margin:.0f} s   exposure: {exposure_span_s:.0f} s")
    if t_margin <= 0:
        print("  no margin searched -- widen margin_before_s/margin_after_s "
              "to calibrate this")
        return None

    rate = n_out / t_margin
    expected = rate * exposure_span_s
    # the margin holds few counts, so this is Poisson-limited -- quote the error
    err = np.sqrt(max(n_out, 1)) / t_margin * exposure_span_s
    print(f"\n  chance-match rate      : {rate*3600:.1f} per hour of search")
    print(f"  expected chance matches inside the exposure: "
          f"{expected:.1f} +/- {err:.1f}  (Poisson on {n_out} margin counts)")
    if n_out < 5:
        print("  NB: few margin counts -- treat this as an order-of-magnitude "
              "estimate, and widen the margins to sharpen it")
    if n_in:
        purity = max(0.0, 1.0 - expected / n_in)
        print(f"  implied purity         : {100*purity:.0f}% "
              f"({n_in - expected:.0f} of {n_in} genuine)")
        if purity < 0.7:
            print("  -> LOW. Tighten max_perp_arcsec / max_pa_deg and re-run; "
                  "chance matches scale with the tolerance area.")

    # what distinguishes the impossible matches from the rest?
    def prof(mask, lab):
        if mask.sum() == 0:
            print(f"  {lab:>18s}: none")
            return
        row = [f"n={int(mask.sum()):3d}"]
        for col, fmt in [("match_perp_arcsec", "{:7.1f}\""),
                         ("match_pa_diff_deg", "{:6.2f}deg"),
                         ("sat_height_km", "{:8.0f}km"),
                         ("ang_rate_arcsec_s", "{:7.0f}\"/s")]:
            if col in m.colnames:
                v = np.asarray(m[col], float)[mask]
                v = v[np.isfinite(v)]
                row.append(fmt.format(np.median(v)) if v.size else "     --")
        print(f"  {lab:>18s}: " + "  ".join(row))

    print("\n  median properties (impossible vs plausible):")
    prof(outside, "outside shutter")
    prof(inside, "during exposure")
    print("  a distinct population outside the shutter -- typically slow, "
          "distant objects -- \n  is the signature of chance alignment, and "
          "tells you which cut will remove it")
    return dict(n_inside=n_in, n_outside=n_out, rate_per_s=rate,
                expected_chance=expected)


def diagnose_streak(streak_id, info, cache_dir, site, window_min=45.0,
                    top=15, radius_deg=10.0, fine_step_s=1.0):
    """Wide-open search around one streak, ignoring the usual tolerances.

    Searches +/- `window_min` minutes about mjd_shot with a large radius and no
    effective perpendicular/PA cut, then ranks everything it finds by
    perpendicular offset.  Reading the resulting table answers the three
    questions that matter:

    * Is the right object there at all, just outside the normal window?  Look
      at `dt_s` -- if the best-fitting track crosses at, say, -900 s, then
      mjd_shot is not the start of the first dither and the window needs
      moving, not widening.
    * Is there a systematic offset?  If the best perpendicular offsets cluster
      at some non-zero value with good PA agreement, suspect the frames or the
      site coordinates.
    * Is nothing near the field at any time?  Then the TLE cache, the time
      scale, or the field coordinates are wrong.
    """
    from astropy.table import Table
    from fetch_tles import night_id

    row = info[np.asarray(info["streak_id"]) == int(streak_id)]
    if len(row) == 0:
        raise ValueError(f"streak_id {streak_id} not in the catalog")
    row = row[0]
    extra = tuple(c for c in _TIMING_COLS if c in info.colnames)
    all_cols = NEEDED_COLS + extra
    shot_rows = []
    for r in info[np.asarray(info["shotid"]) == row["shotid"]]:
        d = {}
        for k in all_cols:
            v = r[k]
            if hasattr(v, "shape") and v.shape:
                d[k] = np.asarray(v, dtype=np.float64)
            elif hasattr(v, "item"):
                d[k] = v.item()
            else:
                d[k] = v
        shot_rows.append(d)
    target = [r for r in shot_rows if int(r["streak_id"]) == int(streak_id)]

    nid = int(night_id(row["mjd_shot"]))
    path = find_night_files(cache_dir).get(nid)
    print(f"streak {streak_id}  shot {row['shotid']}  night {nid}")
    print(f"  mjd_shot {row['mjd_shot']:.10f}  exptime {row['exptime']:.1f} s")
    if "dither_open" in info.colnames and np.all(np.isfinite(row["dither_open"])):
        d_open = np.asarray(row["dither_open"], dtype=np.float64)
        d_close = np.asarray(row["dither_close"], dtype=np.float64)
        for d in range(3):
            print(f"  dither {d+1}: open={d_open[d]:.10f}  close={d_close[d]:.10f}  "
                  f"({(d_close[d]-d_open[d])*86400:.1f}s)")
        if "expnum" in info.colnames:
            print(f"  streak in dither {int(row['expnum'])}")
    print(f"  field {row['ra_cen_spax']:.4f} {row['dec_cen_spax']:+.4f}")
    print(f"  TLE file: {path}")
    if path is None:
        print("  -> no cached TLEs for this night; fetch it first")
        return None
    records = load_3le(path)
    norads = {r["norad"] for r in records}
    print(f"  {len(records)} element sets, {len(norads)} distinct objects")
    ep = np.array([r["epoch_mjd"] for r in records])
    print(f"  TLE epochs span {ep.min():.3f} to {ep.max():.3f} "
          f"(shot at {row['mjd_shot']:.3f})")
    if ep.min() > row["mjd_shot"] or ep.max() < row["mjd_shot"]:
        print("  -> WARNING: the shot time is outside the TLE epoch range")

    # deliberately permissive: nothing is rejected on geometry
    cfg = MatchConfig(coarse_radius_deg=radius_deg, coarse_step_s=5.0,
                      fine_step_s=fine_step_s,
                      max_perp_arcsec=radius_deg * 3600.0, max_pa_deg=90.1,
                      exposure_span_s=0.0, n_dither=0,
                      margin_before_s=window_min * 60.0,
                      margin_after_s=window_min * 60.0,
                      min_altitude_deg=-90.0, require_sunlit=False,
                      keep_n_candidates=max(top, 50))
    t_lo, t_hi = cfg.window_mjd(float(row["mjd_shot"]), float(row["exptime"]))
    print(f"  searching {(t_hi - t_lo) * 1440.0:.0f} min window, "
          f"radius {radius_deg} deg")

    res = process_shot(target, records, site, cfg, cache_dir)
    if not res or not res[0].get("_all"):
        print("\n  nothing found even wide open -> suspect the TLE cache, the "
              "time scale, or the field coordinates")
        return None

    scored = res[0]["_all"]
    t0 = float(row["mjd_shot"])

    def as_row(c):
        return (c["norad"], str(c["name"])[:22],
                round(c["perp_arcsec"], 1), round(c["pa_diff_deg"], 2),
                round((c["crossing_mjd"] - t0) * 86400.0, 1),
                round(c["alt_deg"], 1), round(c["range_km"]),
                core.ILLUM_LABELS[c["illum"]])

    cols = ("norad", "name", "perp_asec", "dPA_deg", "dt_s", "alt_deg",
            "range_km", "illum")
    perp = np.array([c["perp_arcsec"] for c in scored])
    dpa = np.array([c["pa_diff_deg"] for c in scored])
    print(f"\n  {res[0]['n_close']} objects within {radius_deg} deg; "
          f"{len(scored)} scored")
    print(f"  perpendicular offset: min {perp.min():.0f}\" "
          f"median {np.median(perp):.0f}\"")
    print(f"  |dPA|:                min {dpa.min():.2f} deg "
          f"median {np.median(dpa):.2f} deg")

    print(f"\n  --- top {min(top, len(scored))} by perpendicular offset "
          f"(is the right object here at the wrong time?) ---")
    by_perp = [as_row(c) for c in sorted(scored, key=lambda d: d["perp_arcsec"])[:top]]
    Table(rows=by_perp, names=cols).pprint_all()

    print(f"\n  --- top 5 by PA agreement (is there a systematic offset?) ---")
    by_pa = [as_row(c) for c in sorted(scored, key=lambda d: d["pa_diff_deg"])[:5]]
    Table(rows=by_pa, names=cols).pprint_all()

    b = by_perp[0]
    print(f"\n  best offset {b[2]}\" at dt = {b[4]} s, |dPA| = {b[3]} deg")
    if b[2] < 900 and abs(b[4]) > 60:
        print(f"  -> good geometry {b[4]} s away from mjd_shot: SHIFT the "
              f"window (n_dither / margin_before_s), do not just widen it")
    elif b[2] > 900 and dpa.min() < 2.0:
        print("  -> tracks parallel to the streak but displaced: suspect a "
              "systematic (site coordinates, or the streak endpoints)")
    elif b[2] > 900:
        print("  -> nothing lines up even wide open; check that ra_start/"
              "dec_start/ra_end/dec_end really are the two ends of the track")
    return Table(rows=by_perp, names=cols)


# =====================================================================
# RECONSTRUCTED 2026-08-13.  This function (and audit_tle_cache /
# audit_tle_age below the diagnostics section, near cache_coverage) were
# added in a separate JupyterLab session and the source was lost when
# match_streaks.py was overwritten wholesale from a version that predated
# them. The original implementations were not recoverable (checked
# .ipynb_checkpoints and the IPython history sqlite db -- both had call
# sites but no `def`). This is a fresh implementation built from those call
# sites, `_PACK_COLS` in the notebook history (which matches this file's
# actual _pack()/write_output() schema exactly), and the surrounding
# functions' conventions. It has NOT been run against the original data and
# has NOT been compared against the lost version's actual output.
#
# Test before trusting it for the paper: run it on a couple of pairs you can
# sanity-check by eye (e.g. via diagnose_streak on the same streak_id first),
# and diff the returned match_perp_arcsec / match_pa_diff_deg against what
# diagnose_streak reports for the same object. rematch_by_norad writes to
# the science table via the notebook cell that calls it -- do this before
# re-running that cell for real.
# =====================================================================
def rematch_by_norad(pairs, info, cache_dir, site, cfg, satcat=None,
                     force=False, verbose=True):
    """Re-run our own SGP4 geometry for specific (streak_id, norad_id) pairs.

    For streaks where an independent source (typically the SatChecker
    cross-check, section 3b.i) suggests an identification our own
    coarse-to-fine search did not surface or scored below tolerance, this
    targets exactly that one object on that one night -- instead of the full
    ~25k-object catalog -- and scores it with the *same* `core.score_track`
    used everywhere else in this file. That keeps the reported geometry
    (`match_perp_arcsec`, `match_pa_diff_deg`, ...) and derived photometry
    (`g_mag_inst`) self-consistent with the rest of the MATCH table, rather
    than copying SatChecker's own numbers in directly.

    By default (`force=False`) `cfg`'s own `max_perp_arcsec` / `max_pa_deg`
    tolerances still apply, so a pair our geometry genuinely cannot
    corroborate is skipped (printed, not silently dropped) rather than
    force-written into the table -- adopting an identification our own
    geometry disagrees with deserves a `diagnose_streak` look, not a bypass.
    `force=True` drops the perpendicular/PA cut (the along-track
    "was the object actually there" time-offset cut still applies, since
    that one is not a tuning choice) so you can inspect the geometry for a
    marginal case before deciding by hand. Results returned this way carry
    `forced=True` and should be reviewed, not written blind.

    Parameters
    ----------
    pairs : list of (streak_id, norad_id)
    info  : the INFO table
    cache_dir : TLE cache directory
    site  : (lat, lon, height_m) geodetic triple, as used elsewhere in this
            module
    cfg   : MatchConfig
    satcat : loaded satcat dict, or a cache directory (see `get_satcat`);
             defaults to `cache_dir`

    Returns
    -------
    list of dicts, each shaped like `_pack()`'s normal per-streak output
    (same keys `write_output` reads) plus `streak_id`, and `forced=True`
    when `force=True`. One entry per pair that found and scored the
    requested object; pairs that failed are printed and omitted, not
    padded with a null entry.
    """
    from astropy.time import Time
    from fetch_tles import night_id

    pairs = list(pairs)
    satcat = get_satcat(satcat if satcat is not None else cache_dir)
    location = site_location(site)
    info_ids = np.asarray(info["streak_id"]).astype(int)
    files = find_night_files(cache_dir)
    records_cache = {}   # night_id -> parsed 3LE records, shared across pairs

    out = []
    for sid, norad in pairs:
        sid = int(sid)
        norad = int(norad)
        w = np.where(info_ids == sid)[0]
        if len(w) == 0:
            print(f"  streak {sid}: not in the catalog, skipped")
            continue
        row = info[int(w[0])]
        extra = tuple(c for c in _TIMING_COLS if c in info.colnames)
        row_d = {}
        for k in NEEDED_COLS + extra:
            v = row[k]
            if hasattr(v, "shape") and v.shape:
                row_d[k] = np.asarray(v, dtype=np.float64)
            elif hasattr(v, "item"):
                row_d[k] = v.item()
            else:
                row_d[k] = v
        mjd_shot = float(row_d["mjd_shot"])
        exptime = float(row_d["exptime"])
        night = int(night_id(mjd_shot))

        path = files.get(night)
        if path is None:
            print(f"  streak {sid} (norad {norad}): no cached TLEs for "
                  f"night {night}, skipped")
            continue
        if night not in records_cache:
            records_cache[night] = load_3le(path)
        records = records_cache[night]

        satrecs, meta = build_satrecs(records, mjd_shot, norads=[norad])
        if not satrecs:
            # Fall back to a targeted per-object fetch, if one has been
            # cached (fetch_tles.fetch_missing_norads writes these). Not
            # fetched here on demand -- this function stays network-free and
            # synchronous like the rest of the matcher; run the fetch first
            # for any pairs that reach this branch.
            supp_path = Path(cache_dir) / f"norad_{norad}.3le.gz"
            if supp_path.exists():
                supp_records = load_3le(supp_path)
                satrecs, meta = build_satrecs(supp_records, mjd_shot,
                                              norads=[norad])
            if not satrecs:
                hint = ("checked the per-night cache and the targeted "
                        "supplemental fetch, neither has it"
                        if supp_path.exists() else
                        "not in the per-night cache; no supplemental fetch "
                        f"found either -- try fetch_tles.fetch_missing_norads("
                        f"[({norad}, {mjd_shot})], CACHE_DIR) first")
                print(f"  streak {sid}: no element set for norad {norad} on "
                      f"night {night} ({hint}), skipped")
                continue

        d_open = row_d.get("dither_open")
        d_close = row_d.get("dither_close")
        has_timing = (d_open is not None
                      and np.all(np.isfinite(d_open))
                      and np.all(np.isfinite(d_close)))
        if has_timing:
            mjd_fine = shutter_open_grid(
                np.asarray(d_open, np.float64),
                np.asarray(d_close, np.float64),
                cfg.fine_step_s, margin_s=cfg.margin_before_s)
            t_lo, t_hi = mjd_fine[0], mjd_fine[-1]
        else:
            t_lo, t_hi = cfg.window_mjd(mjd_shot, exptime)
            n_fine = max(8, int(np.ceil(
                (t_hi - t_lo) * 86400.0 / cfg.fine_step_s)) + 1)
            mjd_fine = np.linspace(t_lo, t_hi, n_fine)
        t_fine = Time(mjd_fine, format="mjd", scale="utc")
        Mrot = teme_to_gcrs_matrix(Time(0.5 * (t_lo + t_hi), format="mjd", scale="utc"))
        obs_f, sun_f = observer_and_sun(t_fine, location)
        up_gcrs = _observer_up_gcrs(location, t_fine)

        r_teme, ok = propagate(satrecs, mjd_fine)
        r_gcrs = np.einsum("ij,ntj->nti", Mrot, r_teme)
        topo = r_gcrs - obs_f[None, :, :]
        rng = np.linalg.norm(topo, axis=-1)
        u_f = topo / np.where(rng[..., None] > 0, rng[..., None], 1.0)

        good = ok[0]
        if good.sum() < 3:
            print(f"  streak {sid}: norad {norad} not usably propagated in "
                  f"this window, skipped")
            continue

        use_cfg = cfg
        if force:
            import copy
            use_cfg = copy.copy(cfg)
            use_cfg.max_perp_arcsec = 1e9
            use_cfg.max_pa_deg = 180.0

        res = core.score_track(u_f[0][good], mjd_fine[good],
                               row_d["ra_start"], row_d["dec_start"],
                               row_d["ra_end"], row_d["dec_end"], use_cfg)
        if res is None:
            print(f"  streak {sid}: norad {norad} propagated but did not "
                  f"score (failed the along-track cut, or the tolerance cut "
                  f"with force=False)")
            continue

        gi = np.where(good)[0][res["idx"]]
        alt = 90.0 - core.angsep(u_f[0, gi], up_gcrs[gi])
        illum = int(core.illumination_state(r_gcrs[0, gi], sun_f[gi]))
        rec = meta[0]
        res.update(
            norad=rec["norad"], name=rec["name"],
            tle_epoch_mjd=rec["epoch_mjd"],
            tle_age_hours=(mjd_shot - rec["epoch_mjd"]) * 24.0,
            range_km=float(rng[0, gi]), alt_deg=float(alt), illum=illum,
            phase_deg=float(core.phase_angle(r_gcrs[0, gi], obs_f[gi], sun_f[gi])),
            sat_geocentric_km=float(np.linalg.norm(r_gcrs[0, gi])),
            sun_alt_deg=float(90.0 - core.angsep(
                core.normalize(sun_f[gi] - obs_f[gi]), up_gcrs[gi])),
            mean_motion=float(satrecs[0].no_kozai * 1440.0 / (2 * np.pi)),
            ecc=float(satrecs[0].ecco),
            inc_deg=float(np.degrees(satrecs[0].inclo)),
        )

        packed = _pack(row, [res], mjd_shot, exptime, 1, 1, cfg, satcat)
        packed["streak_id"] = sid
        packed.pop("_all", None)
        if force:
            packed["forced"] = True
        out.append(packed)
        if verbose:
            tag = " (forced, tolerance bypassed)" if force else ""
            print(f"  streak {sid}: norad {norad} -> "
                  f"perp {res['perp_arcsec']:.1f}\"  "
                  f"dPA {res['pa_diff_deg']:.2f} deg{tag}")

    if verbose:
        print(f"\n{len(out)}/{len(pairs)} pairs scored")
    return out


# --------------------------------------------------- publication table
# (source, column, output name, unit, description)
#   source "I" = INFO (the streak catalog), "M" = MATCH (the identification)
# Descriptions are mandatory for the AAS machine-readable format, so they live
# here rather than being invented at write time.
PUB_COLUMNS = [
    ("I", "streak_id", "ID", "", "Streak identifier"),
    ("I", "shotid", "Shot", "", "HETDEX shot identifier (YYYYMMDDsss)"),
    ("I", "mjd_shot", "MJD", "d", "Modified Julian Date of shot start (UTC)"),
    ("I", "exptime", "Texp", "s", "Exposure time per dither"),
    ("I", "ra_cen_spax", "RAcen", "deg", "Right ascension of streak centroid (J2000)"),
    ("I", "dec_cen_spax", "DEcen", "deg", "Declination of streak centroid (J2000)"),
    ("I", "ra_start", "RAsta", "deg", "Right ascension of first streak endpoint"),
    ("I", "dec_start", "DEsta", "deg", "Declination of first streak endpoint"),
    ("I", "ra_end", "RAend", "deg", "Right ascension of second streak endpoint"),
    ("I", "dec_end", "DEend", "deg", "Declination of second streak endpoint"),
    ("I", "seg_len_arcsec", "Length", "arcsec", "Observed streak segment length"),
    ("I", "streak_pa", "PA", "deg", "Position angle of streak, east of north"),
    ("I", "area_arcsec2", "Area", "arcsec2", "Streak area from flagged spaxels"),
    ("I", "n_ifu", "Nifu", "", "Number of IFUs contributing"),
    ("I", "g_mag", "gmag", "mag", "SDSS g AB magnitude of the summed streak spectrum"),
    ("I", "sb_mag_arcsec2", "gSB", "mag", "Surface brightness in SDSS g per square arcsec"),
    ("I", "mean_snr", "SNR", "", "Mean signal-to-noise per pixel, 3800-5500 AA"),
    # --- identification, blank where no match was found
    ("M", "norad_id", "NORAD", "", "NORAD catalog number of the matched object"),
    ("M", "object_name", "Name", "", "Object name"),
    ("M", "object_id", "COSPAR", "", "COSPAR international designator"),
    ("M", "object_type", "Type", "", "Object type (PAYLOAD, ROCKET BODY, DEBRIS)"),
    ("M", "country", "Owner", "", "Owner or operator code"),
    ("M", "launch_date", "Launch", "", "Launch date (YYYY-MM-DD)"),
    ("M", "constellation", "Constel", "", "Constellation membership"),
    ("M", "orbit_class", "Class", "", "Orbit class (LEO, MEO, GEO, HEO)"),
    ("M", "match_perp_arcsec", "Resid", "arcsec", "Perpendicular offset of streak from the propagated track"),
    ("M", "match_pa_diff_deg", "dPA", "deg", "Position-angle difference between streak and track"),
    ("M", "unambiguous", "Uniq", "", "Flag, 1 if the best match is clearly separated from the next"),
    ("M", "crossing_mjd", "MJDx", "d", "Modified Julian Date of closest approach (UTC)"),
    ("M", "tle_age_hours", "TLEage", "h", "Shot time minus element-set epoch"),
    ("M", "range_km", "Range", "km", "Topocentric slant range to the object"),
    ("M", "sat_height_km", "Height", "km", "Height of the object above the Earth ellipsoid"),
    ("M", "ang_rate_arcsec_s", "Rate", "arcsec/s", "Topocentric angular rate"),
    ("M", "inclination_deg", "Incl", "deg", "Orbital inclination"),
    ("M", "perigee_km", "Perigee", "km", "Perigee height"),
    ("M", "apogee_km", "Apogee", "km", "Apogee height"),
    ("M", "period_min", "Period", "min", "Orbital period"),
    ("M", "g_mag_inst", "ginst", "mag", "Instantaneous SDSS g magnitude, trail-rate corrected"),
    ("I", "fwhm_virus", "FWHM", "arcsec", "Seeing FWHM from VIRUS guider"),
    ("I", "expnum", "Dither", "", "Dither number (1-3) in which streak was detected"),
    ("I", "dither_mjd_open", "MJDopen", "d", "MJD of shutter open for the streak dither (UTC)"),
    ("I", "dither_mjd_close", "MJDclos", "d", "MJD of shutter close for the streak dither (UTC)"),
]


def write_publication_table(catalog, matched, out_base="HETDEX_PDR1_sats_ids",
                            formats=("fits", "mrt", "csv"), verbose=True,
                            info_table=None):
    """Curated streak + identification table, ready for a journal.

    Joins the streak catalog to the identifications, keeps a publishable subset
    of columns with units and descriptions attached, and **masks the
    identification columns wherever no match was found** so unmatched streaks
    read as blanks rather than as sentinel values like -1.

    `mrt` is the AAS machine-readable format, which requires a description on
    every column -- hence PUB_COLUMNS.  Writing it can be fussy about dtypes,
    so each format is attempted independently and a failure in one does not
    lose the others.

    Parameters
    ----------
    info_table : astropy.table.Table, optional
        If provided, use this in-memory INFO table instead of reading from
        *catalog*.  This allows timing columns added by `apply_survey_timing`
        (which patches the table in memory, not on disk) to flow through to
        the publication output.
    """
    import astropy.units as u
    from astropy.table import Column, MaskedColumn, Table

    info = info_table if info_table is not None else Table.read(catalog, hdu="INFO")
    match = Table.read(matched, hdu="MATCH")
    for t in (info, match):
        for c in t.colnames:
            if t[c].dtype.kind == "S":
                t[c] = np.char.decode(t[c], "utf-8")

    if not np.array_equal(np.asarray(info["streak_id"]),
                          np.asarray(match["streak_id"])):
        raise ValueError("INFO and MATCH are not row-aligned")

    unmatched = ~np.asarray(match["matched"], bool)
    out = Table()
    for src, col, name, unit, desc in PUB_COLUMNS:
        t = info if src == "I" else match
        if col not in t.colnames:
            if verbose:
                print(f"  skipping {col}: not in {'INFO' if src=='I' else 'MATCH'}")
            continue
        data = np.asarray(t[col])
        if data.dtype.kind == "b":                 # MRT dislikes booleans
            data = data.astype(np.int16)
        elif data.dtype.kind == "f" and data.dtype.itemsize < 8:
            data = data.astype(np.float64)         # MRT cannot format float32
        if src == "M":
            data = MaskedColumn(data, mask=unmatched)
        else:
            data = Column(data)
        data.name = name
        data.description = desc
        if unit:
            try:
                data.unit = u.Unit(unit)
            except Exception:
                data.unit = unit
        out[name] = data

    n_id = int((~unmatched).sum())
    out.meta["comments"] = [
        "Satellite streaks in HETDEX PDR1 with orbital identifications.",
        f"{len(out)} streaks; {n_id} identified against archival two-line "
        "element sets.",
        "Identification columns are blank where no match was found.",
        "Positions are J2000; times are UTC.",
        "Resid is the perpendicular offset of the streak from the propagated "
        "track and is the primary match discriminant; along-track separation "
        "is dominated by element-set timing error and is not used.",
    ]

    # FITS uses float32 (sufficient precision, smaller file) except for MJD
    # columns where float32 quantisation (~338 s at MJD ~58000) loses the
    # precise timing we just reconstructed.
    # MRT/CSV keep float64 — the MRT writer cannot format float32.
    _keep_f64 = {"MJD", "MJDx", "MJDopen", "MJDclos"}
    out_fits = out.copy()
    for _cn in out_fits.colnames:
        if out_fits[_cn].dtype == np.float64 and _cn not in _keep_f64:
            out_fits[_cn] = out_fits[_cn].astype(np.float32)

    written = []
    for fmt in formats:
        path = f"{out_base}.{ {'mrt': 'txt'}.get(fmt, fmt) }"
        try:
            if fmt == "fits":
                from astropy.io import fits as _fits
                hdu = _fits.table_to_hdu(out_fits)
                hdu.name = "INFO"
                _fits.HDUList([_fits.PrimaryHDU(), hdu]).writeto(
                    path, overwrite=True)
            elif fmt == "mrt":
                out.write(path, format="ascii.mrt", overwrite=True)
                # astropy puts "? " at the start of the description for
                # MaskedColumns; CDS/AAS standard puts "?" before the byte
                # range.  Post-process to move it.
                import re as _re
                # Matches a column-def line (optional leading space, then
                # byte_range format unit label, then "? " before description)
                _col = _re.compile(
                    r'^( *\d+- *\d+\s+\S+\s+\S+\s+\S+)\s+\? ')
                with open(path) as _f:
                    _lines = _f.readlines()
                _fixed = [_col.sub(r'?\1  ', _l) for _l in _lines]
                with open(path, 'w') as _f:
                    _f.writelines(_fixed)
            elif fmt == "csv":
                out.write(path, format="csv", overwrite=True)
            else:
                continue
            written.append(path)
            if verbose:
                print(f"  wrote {path}")
        except Exception as exc:
            print(f"  {fmt} failed: {type(exc).__name__}: {exc}")

    if verbose:
        print(f"\n{len(out)} rows x {len(out.colnames)} columns; "
              f"{n_id} identified ({100*n_id/max(len(out),1):.1f}%)")
    return out


# ------------------------------------------------------- job orchestration
def read_site(catalog):
    """(lat, lon, height_m) from the catalog primary header."""
    from astropy.io import fits
    with fits.open(catalog) as h:
        ph = h[0].header
        return (ph.get("SITELAT", 30.681436), ph.get("SITELONG", -104.014744),
                ph.get("SITEELEV", 2026.0))


def find_night_files(cache_dir):
    """{night_id: path} for the cached 3LE files."""
    files = {}
    for p in sorted(Path(cache_dir).glob("gp_*.3le.gz")):
        try:
            files[int(p.name.split("_")[1])] = p
        except (IndexError, ValueError):
            continue
    return files


def cache_coverage(info, cache_dir, verbose=True):
    """Which observing nights have cached element sets, and which do not.

    A missing night is invisible in the results: its streaks simply come back
    unmatched, which is easy to misread as "no satellite found" when in fact
    nothing was ever searched.  Run this before interpreting a match rate.
    """
    from fetch_tles import night_id

    def _d(m):
        return (datetime(1858, 11, 17)
                + timedelta(days=float(m))).strftime("%Y-%m-%d")

    files = find_night_files(cache_dir)
    nights = defaultdict(list)
    for row in info:
        nights[int(night_id(row["mjd_shot"]))].append(float(row["mjd_shot"]))

    have, missing = [], []
    for nid, mjds in nights.items():
        (have if nid in files else missing).append((nid, min(mjds), len(mjds)))

    out = dict(n_needed=len(nights), n_cached=len(have), n_missing=len(missing),
               missing_nights=sorted(n for n, _m, _c in missing),
               stranded_streaks=sum(c for _n, _m, c in missing))
    if verbose:
        print(f"observing nights needed : {len(nights)}")
        if have:
            print(f"  cached   : {len(have):4d}   "
                  f"{_d(min(h[1] for h in have))} .. {_d(max(h[1] for h in have))}")
        if missing:
            print(f"  MISSING  : {len(missing):4d}   "
                  f"{_d(min(m[1] for m in missing))} .. "
                  f"{_d(max(m[1] for m in missing))}")
            print(f"  streaks stranded by the gap: {out['stranded_streaks']}"
                  f" of {len(info)}")
            print(f"  est. fetch time: {len(missing)/280.0:.1f} h at the "
                  "Space-Track rate limit")
            print("  -> those streaks cannot match; do not read them as "
                  "unidentified")
        else:
            print("  complete: every night has cached elements")
    return out


# RECONSTRUCTED 2026-08-13 -- see the note above rematch_by_norad for why,
# and the same caveat: not run against the original data, not diffed
# against the lost version's actual output. Treat `suspect_nights` as a
# starting point for `diagnose_streak` / a re-fetch, not a verdict.
def audit_tle_cache(cache_dir, full_scan=False, size_frac=0.5, min_bytes=2000,
                    count_frac=0.5, verbose=True):
    """Find corrupt or partial TLE downloads already sitting in the cache.

    Complements `cache_coverage`, which finds *missing* nights: this finds
    nights whose file exists but is truncated or empty -- typically a
    Space-Track connection that dropped mid-download rather than a fetch
    that never happened. Those look "cached" to `cache_coverage` and
    `output_is_current`, but propagate few or no objects.

    Fast pass (default, `full_scan=False`): flags a night if its compressed
    file size is below `min_bytes`, or below `size_frac` of the cache's
    median file size. Just stats the files, does not open them.

    `full_scan=True` additionally decompresses and parses every cached file
    with `load_3le` to get an exact element-set count per night, and flags
    nights whose count is 0 or below `count_frac` of the median. Touches
    every file on disk -- meant for confirming a night the fast pass (or
    `cache_coverage`) already flagged as suspect, not for routine use.

    Returns
    -------
    dict with `n_files`, `median_size_mb`, `suspect_nights` (night ids,
    union of size- and count-based flags if `full_scan`), and, if
    `full_scan`, `counts` ({night_id: n_objects}) and `median_n_objects`.
    """
    files = find_night_files(cache_dir)
    if not files:
        if verbose:
            print(f"no cached TLE files found in {cache_dir}")
        return dict(n_files=0, median_size_mb=0.0, suspect_nights=[])

    nids = sorted(files)
    sizes = np.array([files[n].stat().st_size for n in nids], dtype=float)
    med_size = float(np.median(sizes))
    size_suspect = [n for n, s in zip(nids, sizes)
                    if s < min_bytes or s < size_frac * med_size]

    if verbose:
        print(f"{len(files)} cached nights, {sizes.sum()/1e6:.1f} MB total, "
              f"median {med_size/1e3:.1f} KB/night")
        if size_suspect:
            print(f"  {len(size_suspect)} nights flagged on file size "
                  f"(< {min_bytes} B or < {100*size_frac:.0f}% of median):")
            for n in sorted(size_suspect, key=lambda n: sizes[nids.index(n)])[:15]:
                print(f"    {n}  {files[n].name}  "
                      f"{files[n].stat().st_size/1e3:.1f} KB")
        else:
            print("  no nights flagged on file size")

    out = dict(n_files=len(files), median_size_mb=med_size / 1e6,
               suspect_nights=list(size_suspect))

    if full_scan:
        counts = {}
        for n in nids:
            try:
                counts[n] = len(load_3le(files[n]))
            except Exception as exc:
                counts[n] = 0
                if verbose:
                    print(f"  {n}: failed to parse ({type(exc).__name__}: {exc})")
        cvals = np.array([counts[n] for n in nids], dtype=float)
        med_count = float(np.median(cvals[cvals > 0])) if (cvals > 0).any() else 0.0
        count_suspect = [n for n in nids
                         if counts[n] == 0 or
                         (med_count > 0 and counts[n] < count_frac * med_count)]
        if verbose:
            print(f"\nfull scan: median {med_count:.0f} element sets/night")
            if count_suspect:
                print(f"  {len(count_suspect)} nights flagged on object count "
                      f"(0, or < {100*count_frac:.0f}% of median):")
                for n in sorted(count_suspect, key=lambda n: counts[n])[:15]:
                    print(f"    {n}  {files[n].name}  {counts[n]} objects")
            else:
                print("  no nights flagged on object count")
        out.update(counts=counts, median_n_objects=med_count,
                   suspect_nights=sorted(set(size_suspect) | set(count_suspect)))

    return out


def audit_tle_age(match_table, warn_hours=48.0, extreme_hours=168.0):
    """TLE age distribution for matched streaks -- catches object-level gaps.

    A whole missing observing night shows up as `n_propagated == 0` (see
    `triage`) and is easy to spot. A narrower, easier-to-miss gap is a single
    *object* whose element set was not updated near a given night even
    though that night's cache file exists: the matcher still finds some
    epoch for it (nothing stops it), but propagates from an element set that
    may be stale by days, which degrades along-track accuracy and can push a
    genuine crossing outside the along-track time-offset cut. This surfaces
    that case via `tle_age_hours` (`|shot time - element set epoch|`,
    written by `_pack`), which `triage` does not look at.

    Returns dict with `median_hours`, `n_over_warn`, `n_over_extreme`.
    """
    t = match_table
    m = t[np.asarray(t["norad_id"]) > 0] if "norad_id" in t.colnames \
        else t[np.asarray(t["matched"], bool)]
    age = np.asarray(m["tle_age_hours"], float)
    age = age[np.isfinite(age)]
    if age.size == 0:
        print("no matched streaks with a finite TLE age")
        return None

    q = np.percentile(age, [50, 90, 99])
    print(f"TLE age for {age.size} matched streaks (hours):")
    print(f"  median {q[0]:.1f}   90th {q[1]:.1f}   99th {q[2]:.1f}   "
          f"max {age.max():.1f}")

    warn = int((age > warn_hours).sum())
    extreme = int((age > extreme_hours).sum())
    print(f"  > {warn_hours:.0f} h old: {warn} ({100*warn/age.size:.1f}%)")
    print(f"  > {extreme_hours:.0f} h old: {extreme} ({100*extreme/age.size:.1f}%)")
    if extreme and "streak_id" in m.colnames:
        worst_mask = np.asarray(m["tle_age_hours"], float) > extreme_hours
        worst = m[worst_mask]
        order = np.argsort(-np.asarray(worst["tle_age_hours"], float))
        ids = [int(s) for s in np.asarray(worst["streak_id"])[order][:10]]
        print(f"  worst-affected streak_ids: {ids}")
        print("  -> stale element set for that specific object, not a "
              "missing night; a fresh fetch may not help if Space-Track has "
              "nothing closer in time for that object")
    return dict(median_hours=float(q[0]), n_over_warn=warn,
               n_over_extreme=extreme)


# header keyword -> MatchConfig attribute, for the staleness check
_CFG_HEADER_KEYS = {"COARSER": "coarse_radius_deg", "MAXPERP": "max_perp_arcsec",
                    "MAXPA": "max_pa_deg", "FINESTP": "fine_step_s",
                    "MARGB": "margin_before_s", "MARGA": "margin_after_s",
                    "EXPSPAN": "exposure_span_s", "MAXTOFF": "max_time_offset_s"}


def output_is_current(out_path, cache_dir, catalog=None, cfg=None):
    """True if `out_path` exists, post-dates its inputs, and was built with the
    same settings.

    Lets the notebook skip a multi-minute rematch when nothing has changed,
    while still rerunning automatically after new nights are fetched or the
    catalog is replaced.  mtime-based rather than hashed: simple, and wrong
    only in the harmless direction (an extra rerun).

    Pass `cfg` to also compare the tolerances recorded in the output header.
    Without it, tightening `max_perp_arcsec` and re-running would silently
    reload the old, looser result -- the one case where mtimes cannot help,
    because no file on disk has changed.
    """
    out = Path(out_path)
    if not out.exists():
        return False

    newest = 0.0
    for p in Path(cache_dir).glob("gp_*.3le.gz"):
        newest = max(newest, p.stat().st_mtime)
    sc = Path(cache_dir) / "satcat.json.gz"
    if sc.exists():
        newest = max(newest, sc.stat().st_mtime)
    if catalog and Path(catalog).exists():
        newest = max(newest, Path(catalog).stat().st_mtime)
    if out.stat().st_mtime < newest:
        return False

    if cfg is not None:
        from astropy.io import fits
        try:
            hdr = fits.getheader(str(out), 0)
        except Exception:
            return False
        for key, attr in _CFG_HEADER_KEYS.items():
            want = getattr(cfg, attr, None)
            if want is None:
                continue
            got = hdr.get(key)
            if got is None or abs(float(got) - float(want)) > 1e-9:
                print(f"  settings changed ({attr}: {got} -> {want}) "
                      "- rematch required")
                return False
    return True


def build_jobs(info, cache_dir, limit_nights=0, cached_only=False,
               latest=False, start_mjd=None, end_mjd=None):
    """Group streaks into shots and shots into nights.

    Returns [(night_id, tle_path_or_None, [[row dicts], ...]), ...].  Rows are
    plain dicts so they survive the trip to a joblib worker.

    Parameters
    ----------
    limit_nights : keep only this many nights (0 = all)
    cached_only  : drop nights with no TLE file.  Recommended whenever you have
                   fetched a subset -- otherwise `limit_nights` silently picks
                   the *earliest* nights, which may not be the ones you
                   downloaded, and every streak comes back n_propagated = 0.
    latest       : take the most recent nights rather than the earliest
    start_mjd, end_mjd : restrict to a date range
    """
    from fetch_tles import night_id
    files = find_night_files(cache_dir)

    # Include optional timing columns when present in the info table.
    extra = tuple(c for c in _TIMING_COLS if c in info.colnames)
    all_cols = NEEDED_COLS + extra

    shots = defaultdict(list)
    for row in info:
        d = {}
        for k in all_cols:
            v = row[k]
            # numpy arrays (dither_open/close, shape-3) must stay as arrays
            if hasattr(v, "shape") and v.shape:
                d[k] = np.asarray(v, dtype=np.float64)
            elif hasattr(v, "item"):
                d[k] = v.item()
            else:
                d[k] = v
        shots[int(row["shotid"])].append(d)
    nights = defaultdict(list)
    for _sid, rows in shots.items():
        nights[int(night_id(rows[0]["mjd_shot"]))].append(rows)

    keys = sorted(nights)
    if start_mjd is not None:
        keys = [k for k in keys
                if min(r[0]["mjd_shot"] for r in nights[k]) >= start_mjd]
    if end_mjd is not None:
        keys = [k for k in keys
                if min(r[0]["mjd_shot"] for r in nights[k]) <= end_mjd]
    if cached_only:
        keys = [k for k in keys if k in files]
    if latest:
        keys = keys[::-1]
    if limit_nights:
        keys = keys[:limit_nights]
    return [(k, files.get(k), nights[k]) for k in keys]


def run_all(jobs, site, cfg, satcat, n_jobs=1, verbose=True):
    """Run the matcher over prepared jobs.  Returns the flat result list.

    Parallelism is over nights, which balances well: nights carry a similar
    number of shots and each is independent.  `n_jobs=-1` uses every core.

    Pass `satcat` as the cache-directory path rather than a loaded dict when
    n_jobs > 1 -- see get_satcat for why.

    Peak memory is roughly `cfg.sat_chunk` x n_timesteps x 120 bytes per
    worker (~30 MB at the defaults), so worker count is limited by cores
    rather than RAM.  Halve `cfg.sat_chunk` if you are still tight.
    """
    if n_jobs != 1 and isinstance(satcat, dict) and satcat:
        print("  note: pass the cache directory instead of a loaded satcat "
              "dict when running in parallel -- joblib pickles it per task")

    t_start = time.time()
    results = []
    if n_jobs != 1:
        from joblib import Parallel, delayed
        tasks = [delayed(run_night)(path, groups, site, cfg, satcat)
                 for _nid, path, groups in jobs]
        kw = dict(n_jobs=n_jobs, verbose=5 if verbose else 0)
        try:
            # stops each worker's BLAS spawning its own thread pool on top of
            # the process pool; not accepted by every joblib version
            chunks = Parallel(inner_max_num_threads=1, **kw)(tasks)
        except TypeError:
            chunks = Parallel(**kw)(tasks)
        for c in chunks:
            results.extend(c)
        if verbose:
            el = time.time() - t_start
            print(f"  {len(jobs)} nights in {el:.0f}s "
                  f"({el/max(len(jobs),1):.1f}s/night wall)")
    else:
        for i, (_nid, path, groups) in enumerate(jobs, 1):
            results.extend(run_night(path, groups, site, cfg, satcat))
            if verbose and (i % 10 == 0 or i == len(jobs)):
                el = time.time() - t_start
                print(f"  {i}/{len(jobs)} nights, {el:.0f}s "
                      f"({el/i:.1f}s/night)", flush=True)
    return results


def summarise(t):
    """Print the headline numbers from a MATCH table."""
    n_match = int(t["matched"].sum())
    print(f"matched {n_match}/{len(t)} streaks "
          f"({100.0*n_match/max(len(t),1):.1f}%); "
          f"{int((t['matched'] & t['unambiguous']).sum())} unambiguous")
    if not n_match:
        return
    m = t[t["matched"]]
    print(f"  median |perp| = {np.median(m['match_perp_arcsec']):.1f}\"")
    print(f"  median |dPA|  = {np.median(m['match_pa_diff_deg']):.2f} deg")
    print(f"  crossing time rel. mjd_shot: "
          f"median {np.median(m['crossing_dt_s']):.0f}s, "
          f"16-84% {np.percentile(m['crossing_dt_s'], 16):.0f} to "
          f"{np.percentile(m['crossing_dt_s'], 84):.0f}s")
    n_umbra = int((m["illum_state"] == 0).sum())
    print(f"  in umbra (impossible -> false positives): {n_umbra}")
    vals, counts = np.unique(m["constellation"], return_counts=True)
    for v, c in sorted(zip(vals, counts), key=lambda x: -x[1])[:10]:
        print(f"  {v:>14s}: {c}")


# -------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default="HETDEX_PDR1_sats.fits")
    ap.add_argument("--cache-dir", default="tle_cache")
    ap.add_argument("--out", default="HETDEX_PDR1_sats_matched.fits")
    ap.add_argument("--coarse-radius", type=float, default=5.0)
    ap.add_argument("--coarse-step", type=float, default=20.0)
    ap.add_argument("--fine-step", type=float, default=0.5)
    ap.add_argument("--max-perp", type=float, default=900.0,
                    help="max perpendicular offset, arcsec")
    ap.add_argument("--max-pa", type=float, default=6.0, help="max |dPA|, deg")
    ap.add_argument("--n-dither", type=int, default=3)
    ap.add_argument("--margin-before", type=float, default=120.0)
    ap.add_argument("--margin-after", type=float, default=300.0)
    ap.add_argument("--require-sunlit", action="store_true")
    ap.add_argument("--keep", type=int, default=5, help="candidates kept per streak")
    ap.add_argument("--limit-nights", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="parallel workers over nights; -1 uses every core")
    ap.add_argument("--sat-chunk", type=int, default=4000,
                    help="objects propagated per block in the coarse pass; "
                         "sets peak memory per worker")
    args = ap.parse_args(argv)

    from astropy.table import Table

    cfg = MatchConfig(coarse_radius_deg=args.coarse_radius,
                      coarse_step_s=args.coarse_step,
                      fine_step_s=args.fine_step,
                      max_perp_arcsec=args.max_perp,
                      max_pa_deg=args.max_pa,
                      n_dither=args.n_dither,
                      margin_before_s=args.margin_before,
                      margin_after_s=args.margin_after,
                      require_sunlit=args.require_sunlit,
                      keep_n_candidates=args.keep,
                      sat_chunk=args.sat_chunk)

    info = Table.read(args.catalog, hdu="INFO")
    site = read_site(args.catalog)
    print(f"{len(info)} streaks, site lat={site[0]} lon={site[1]} elev={site[2]}")

    print(f"satcat entries: {len(get_satcat(args.cache_dir))}")

    jobs = build_jobs(info, args.cache_dir, limit_nights=args.limit_nights)
    print(f"TLE night files found: {len(find_night_files(args.cache_dir))}")
    print(f"processing {len(jobs)} nights / "
          f"{sum(len(g) for _n, _p, g in jobs)} shots on {args.n_jobs} job(s)")

    # pass the directory, not the dict: each worker loads and memoises it once
    results = run_all(jobs, site, cfg, args.cache_dir, n_jobs=args.n_jobs)

    t = write_output(args.out, info, results, cfg, args.catalog)
    print()
    summarise(t)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
