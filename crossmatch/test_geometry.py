#!/usr/bin/env python
"""
test_geometry.py -- unit tests for satstreak_core.

Pure numpy: runs without astropy, sgp4 or a network connection, so it can be
executed anywhere before committing to a multi-hour Space-Track run.

    python test_geometry.py
"""

import numpy as np

import satstreak_core as core

FAILURES = []


def check(name, got, want, tol):
    ok = np.all(np.abs(np.asarray(got) - np.asarray(want)) <= tol)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {np.round(got, 6)} "
          f"want {np.round(want, 6)} (tol {tol})")
    if not ok:
        FAILURES.append(name)


def check_true(name, cond):
    print(f"{'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILURES.append(name)


# ------------------------------------------------------- vectors and angles
def test_roundtrip():
    rng = np.random.default_rng(1)
    ra = rng.uniform(0, 360, 500)
    dec = rng.uniform(-80, 80, 500)
    ra2, dec2 = core.unit_to_radec(core.radec_to_unit(ra, dec))
    check("radec roundtrip RA", np.max(np.abs(ra2 - ra)), 0.0, 1e-9)
    check("radec roundtrip Dec", np.max(np.abs(dec2 - dec)), 0.0, 1e-9)


def test_angsep():
    a = core.radec_to_unit(0.0, 0.0)
    check("angsep 90 deg", core.angsep(a, core.radec_to_unit(90.0, 0.0)), 90.0, 1e-9)
    check("angsep pole", core.angsep(a, core.radec_to_unit(0.0, 90.0)), 90.0, 1e-9)
    check("angsep 1 arcsec",
          core.angsep(a, core.radec_to_unit(1 / 3600.0, 0.0)) * 3600.0, 1.0, 1e-6)
    # cos(dec) compression must be present
    b = core.radec_to_unit(0.0, 60.0)
    check("angsep cos(dec)",
          core.angsep(b, core.radec_to_unit(1.0, 60.0)), 0.5, 1e-3)


def test_position_angle():
    o = core.radec_to_unit(0.0, 0.0)
    check("PA north", core.position_angle(o, core.radec_to_unit(0.0, 1.0)), 0.0, 1e-6)
    check("PA east", core.position_angle(o, core.radec_to_unit(1.0, 0.0)), 90.0, 1e-6)
    check("PA south", core.position_angle(o, core.radec_to_unit(0.0, -1.0)), 180.0, 1e-6)
    check("PA west", core.position_angle(o, core.radec_to_unit(359.0, 0.0)), 270.0, 1e-6)
    # at high declination the same RA offset is a larger PA swing
    o2 = core.radec_to_unit(0.0, 60.0)
    pa = core.position_angle(o2, core.radec_to_unit(1.0, 61.0))
    check_true("PA at dec=60 tilts east of the dec=0 value", 20.0 < pa < 30.0)


def test_pa_difference():
    check("dPA 179 vs 1", core.pa_difference(179.0, 1.0), 2.0, 1e-9)
    check("dPA 10 vs 100", core.pa_difference(10.0, 100.0), 90.0, 1e-9)
    check("dPA 350 vs 170", core.pa_difference(350.0, 170.0), 0.0, 1e-9)
    check("dPA 45 vs 225", core.pa_difference(45.0, 225.0), 0.0, 1e-9)


def test_great_circle():
    pole = core.great_circle_pole(core.radec_to_unit(0.0, 0.0),
                                  core.radec_to_unit(90.0, 0.0))
    check("equator pole is +z", pole, [0, 0, 1], 1e-12)
    check("perp 5 deg",
          core.perp_sep_to_great_circle(core.radec_to_unit(30.0, 5.0), pole), 5.0, 1e-9)
    check("perp on-circle",
          core.perp_sep_to_great_circle(core.radec_to_unit(30.0, 0.0), pole), 0.0, 1e-12)
    mid = core.midpoint(core.radec_to_unit(10.0, 0.0), core.radec_to_unit(20.0, 0.0))
    ra, dec = core.unit_to_radec(mid)
    check("midpoint RA", ra, 15.0, 1e-9)
    check("midpoint Dec", dec, 0.0, 1e-9)


# ------------------------------------------------------------- illumination
def test_illumination():
    sun = np.array([1.496e8, 0.0, 0.0])
    check("sunlit (day side)",
          core.illumination_state(np.array([6878.0, 0, 0]), sun), core.ILLUM_SUNLIT, 0)
    check("umbra (directly behind Earth)",
          core.illumination_state(np.array([-6878.0, 0, 0]), sun), core.ILLUM_UMBRA, 0)
    check("penumbra (grazing)",
          core.illumination_state(np.array([-6878.0, 6360.0, 0]), sun),
          core.ILLUM_PENUMBRA, 0)
    check("sunlit (well outside the cone)",
          core.illumination_state(np.array([-6878.0, 9000.0, 0]), sun),
          core.ILLUM_SUNLIT, 0)
    # a satellite high above the terminator at 550 km should be lit while the
    # ground below is dark -- the reason twilight streaks exist at all
    r = 6378.137 + 550.0
    check("terminator-crossing satellite is lit",
          core.illumination_state(np.array([-100.0, r, 0]), sun), core.ILLUM_SUNLIT, 0)
    # vectorised
    batch = np.array([[6878.0, 0, 0], [-6878.0, 0, 0]])
    st = core.illumination_state(batch, np.broadcast_to(sun, (2, 3)))
    check("vectorised illumination", st, [core.ILLUM_SUNLIT, core.ILLUM_UMBRA], 0)


def test_phase_angle():
    sun = np.array([1.496e8, 0.0, 0.0])
    sat = np.array([7000.0, 0.0, 0.0])
    obs = np.array([6378.0, 0.0, 0.0])          # observer directly sunward
    check("phase angle 180 (observer between sat and sun is 180)",
          core.phase_angle(sat, obs, sun), 180.0, 1e-6)
    obs2 = np.array([0.0, 6378.0, 0.0])
    pa = core.phase_angle(sat, obs2, sun)
    check_true("phase angle in range", 0.0 <= pa <= 180.0)


# ----------------------------------------------------------- orbit elements
def test_orbit_geometry():
    a, peri, apo = core.orbit_geometry(15.5, 0.0006)
    check("ISS-like semi-major axis", a, 6795.0, 15.0)
    check_true("ISS-like perigee 380-450 km", 380 < peri < 450)
    check_true("ISS-like classified LEO", core.classify_orbit(peri, apo, 0.0006) == "LEO")

    a, peri, apo = core.orbit_geometry(1.00270, 0.0002)
    check("GEO semi-major axis", a, 42164.0, 30.0)
    check_true("GEO altitude ~35786 km", 35700 < peri < 35870)
    check_true("GEO classified", core.classify_orbit(peri, apo, 0.0002) == "GEO")

    # Starlink shell-1 sits at 550 km, i.e. n = 15.05 rev/day
    a, peri, apo = core.orbit_geometry(15.05, 0.0001)
    check("Starlink shell height", peri, 550.0, 10.0)

    a, peri, apo = core.orbit_geometry(2.006, 0.72)     # Molniya-like
    check_true("Molniya classified HEO",
               core.classify_orbit(peri, apo, 0.72) == "HEO")


def test_constellations():
    cases = {"STARLINK-1234": "Starlink", "ONEWEB-0088": "OneWeb",
             "IRIDIUM 33 DEB": "Iridium-NEXT", "COSMOS 2251": "Cosmos",
             "FALCON 9 R/B": "Other", "": "Unknown"}
    for name, want in cases.items():
        got = core.classify_constellation(name)
        check_true(f"constellation {name!r} -> {want}", got == want)


# ---------------------------------------------------------------- scoring
def synthetic_track(ra0, ra1, dec, n=400, t0=59000.0, dt_s=0.5):
    ra = np.linspace(ra0, ra1, n)
    d = np.full(n, dec)
    mjd = t0 + np.arange(n) * dt_s / 86400.0
    return core.radec_to_unit(ra, d), mjd


def test_score_track():
    cfg = core.MatchConfig()

    # exact hit: streak lies on the model great circle
    units, mjd = synthetic_track(100.0, 101.0, 0.0)
    res = core.score_track(units, mjd, 100.4, 0.0, 100.6, 0.0, cfg)
    check_true("exact hit is found", res is not None)
    check("exact hit perp", res["perp_arcsec"], 0.0, 1e-6)
    check("exact hit dPA", res["pa_diff_deg"], 0.0, 1e-6)
    # 1 deg of RA at dec=0 in 400 samples x 0.5 s = 199.5 s -> 18.0 "/s
    check("angular rate", res["rate_arcsec_s"], 3600.0 / 199.5, 0.05)
    check("streak PA is 90 (due east)", res["streak_pa_deg"], 90.0, 1e-6)

    # offset perpendicular by 0.05 deg = 180"
    units, mjd = synthetic_track(100.0, 101.0, 0.05)
    res = core.score_track(units, mjd, 100.4, 0.0, 100.6, 0.0, cfg)
    check_true("offset track still found", res is not None)
    check("offset perp ~180 arcsec", res["perp_arcsec"], 180.0, 5.0)

    # wrong orientation must be rejected
    units, mjd = synthetic_track(100.0, 101.0, 0.0)
    res = core.score_track(units, mjd, 100.5, -0.1, 100.5, 0.1, cfg)
    check_true("perpendicular streak rejected on PA", res is None)

    # far away must be rejected
    units, mjd = synthetic_track(100.0, 101.0, 30.0)
    res = core.score_track(units, mjd, 100.4, 0.0, 100.6, 0.0, cfg)
    check_true("distant track rejected", res is None)

    # ordering: the closer of two tracks must score better
    cfg2 = core.MatchConfig()
    near, m1 = synthetic_track(100.0, 101.0, 0.02)
    far, m2 = synthetic_track(100.0, 101.0, 0.15)
    r1 = core.score_track(near, m1, 100.4, 0.0, 100.6, 0.0, cfg2)
    r2 = core.score_track(far, m2, 100.4, 0.0, 100.6, 0.0, cfg2)
    check_true("nearer track scores lower", r1["score"] < r2["score"])

    # a track at dec=60 exercises the cos(dec) handling
    units = core.radec_to_unit(np.linspace(100.0, 102.0, 400), np.full(400, 60.0))
    mjd = 59000.0 + np.arange(400) * 0.5 / 86400.0
    res = core.score_track(units, mjd, 100.9, 60.0, 101.1, 60.0, cfg)
    check_true("dec=60 track found", res is not None)
    check("dec=60 perp small", res["perp_arcsec"], 0.0, 1.0)


def test_must_actually_be_there():
    """A satellite on a great circle that passes through the streak, but which
    is degrees away along that circle during the window, must be rejected.

    Regression test: `perp` only constrains the great circle, so without a
    time-offset cut these score perfectly (perp = 0.0") and pile up at the
    window edge, inflating the apparent match count.
    """
    cfg = core.MatchConfig()
    mjd = 59000.0 + np.arange(400) * 0.5 / 86400.0

    def track(ra0, ra1):
        return core.radec_to_unit(np.linspace(ra0, ra1, 400), np.zeros(400))

    # control: the satellite really does cross the streak
    r = core.score_track(track(100.0, 101.0), mjd, 100.4, 0.0, 100.6, 0.0, cfg)
    check_true("satellite crossing the streak is accepted", r is not None)
    check_true("crossing is not flagged at the window edge", not r["at_edge"])
    check_true("time offset is sub-second", r["time_offset_s"] < 1.0)

    # same great circle, wrong place along it -- all must be rejected
    for lo, hi, lab in [(101.0, 109.0, "1-9 deg away"),
                        (103.0, 111.0, "3-11 deg away"),
                        (106.0, 114.0, "6-14 deg away")]:
        r = core.score_track(track(lo, hi), mjd, 100.4, 0.0, 100.6, 0.0, cfg)
        check_true(f"off-track-position rejected ({lab})", r is None)

    # slow distant object: the same angular offset is a much larger time offset
    slow_on = core.radec_to_unit(np.linspace(100.5, 105.8, 400), np.zeros(400))
    r = core.score_track(slow_on, mjd, 100.4, 0.0, 100.6, 0.0, cfg)
    check_true("slow object actually at the streak is accepted", r is not None)
    slow_off = core.radec_to_unit(np.linspace(103.0, 108.3, 400), np.zeros(400))
    r = core.score_track(slow_off, mjd, 100.4, 0.0, 100.6, 0.0, cfg)
    check_true("slow object 2.5 deg away is rejected", r is None)

    # a realistic sampling offset must NOT be rejected
    r = core.score_track(track(100.0, 101.0), mjd, 100.4012, 0.0, 100.6012, 0.0, cfg)
    check_true("half-step sampling offset still accepted", r is not None)

    # the cut must be configurable and actually bite
    loose = core.MatchConfig(max_time_offset_s=1e6)
    r = core.score_track(track(103.0, 111.0), mjd, 100.4, 0.0, 100.6, 0.0, loose)
    check_true("disabling the cut restores the old (wrong) behaviour",
               r is not None and r["at_edge"])


def test_coarse_threshold():
    """A fixed coarse radius silently drops fast LEO crossings; the rate-aware
    threshold must not.  Regression test for the zero-match bug."""
    field = core.radec_to_unit(150.0, 55.0)
    cosd = np.cos(np.radians(55.0))

    def survives(rate_deg_s, step_s, tc, radius=5.0, half=120.0):
        t = np.arange(-half, half + 1e-9, step_s)
        off = (t - tc) * rate_deg_s
        m = np.abs(off) < 40.0            # keep it from wrapping the sky
        off = off[m]
        u = core.radec_to_unit(150.0 + off / cosd, np.full(len(off), 55.0))
        sep = core.angsep(u, field[None, :])
        thr = core.coarse_threshold(radius, u[None, ...])[0]
        return sep.min() < radius, sep.min() < thr

    # crossing exactly half a coarse step off the sample grid
    for rate, fixed_should_catch in [(0.20, True), (0.40, True),
                                     (0.67, False), (1.40, False)]:
        fixed, aware = survives(rate, 20.0, tc=10.0)
        check_true(f"rate {rate} deg/s: fixed radius catches = {fixed_should_catch}",
                   fixed == fixed_should_catch)
        check_true(f"rate {rate} deg/s: rate-aware catches", aware)

    # across all sampling phases the rate-aware threshold must never miss
    n_miss_fixed = n_miss_aware = 0
    for tc in np.linspace(0, 20, 41):
        fixed, aware = survives(0.67, 20.0, tc)
        n_miss_fixed += (not fixed)
        n_miss_aware += (not aware)
    check_true(f"fixed radius loses crossings ({n_miss_fixed}/41)", n_miss_fixed > 0)
    check("rate-aware loses none", n_miss_aware, 0, 0)

    # threshold must reduce to the base radius for a stationary object
    u = np.broadcast_to(core.radec_to_unit(10.0, 10.0), (1, 8, 3))
    check("static object keeps base radius",
          core.coarse_threshold(5.0, u)[0], 5.0, 1e-9)
    # and grow with rate
    slow = core.radec_to_unit(np.linspace(0, 1, 8), np.zeros(8))[None, ...]
    fast = core.radec_to_unit(np.linspace(0, 20, 8), np.zeros(8))[None, ...]
    check_true("threshold grows with angular rate",
               core.coarse_threshold(5.0, fast)[0] >
               core.coarse_threshold(5.0, slow)[0] > 5.0)
    # validity mask must suppress bogus jumps
    u2 = core.radec_to_unit(np.array([0., 0.1, 170., 0.3]), np.zeros(4))[None, ...]
    ok = np.array([[True, True, False, True]])
    check_true("masked samples excluded from the travel estimate",
               core.coarse_threshold(5.0, u2, ok)[0] < 6.0)


def test_photometry():
    # 500" streak at 1000 "/s -> 0.5 s crossing in a 367 s exposure
    m, t_cross = core.instantaneous_magnitude(17.0, 500.0, 1000.0, 367.0)
    check("crossing time", t_cross, 0.5, 1e-9)
    check("instantaneous mag", m, 17.0 + 2.5 * np.log10(0.5 / 367.0), 1e-9)
    check_true("instantaneous mag is brighter than trail mag", m < 17.0)
    check("range normalisation at 550 km",
          core.normalize_magnitude_to_range(7.0, 550.0), 7.0, 1e-12)
    check("range normalisation at 1100 km",
          core.normalize_magnitude_to_range(7.0, 1100.0), 7.0 - 5 * np.log10(2.0), 1e-12)


if __name__ == "__main__":
    for fn in [test_roundtrip, test_angsep, test_position_angle,
               test_pa_difference, test_great_circle, test_illumination,
               test_phase_angle, test_orbit_geometry, test_constellations,
               test_score_track, test_must_actually_be_there,
               test_coarse_threshold, test_photometry]:
        print(f"\n--- {fn.__name__}")
        fn()
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES: {FAILURES}")
        raise SystemExit(1)
    print("all tests passed")
