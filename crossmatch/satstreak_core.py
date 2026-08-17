"""
satstreak_core.py -- pure-numpy geometry, illumination and scoring routines for
identifying HETDEX satellite streaks against archival TLEs.

Deliberately has NO astropy / sgp4 / skyfield dependency so it can be unit
tested standalone (see test_geometry.py).  All angles in degrees unless the
name ends in _rad.  All vectors are right-handed equatorial Cartesian with the
same axes as ICRS/GCRS (they are identical to <1 mas for our purposes).

Conventions
-----------
* Sky directions are unit 3-vectors: x -> (RA=0, Dec=0), z -> Dec=+90.
* Position angle (PA) is measured East of North, i.e. from +Dec toward +RA,
  in the tangent plane at the reference point.  This is the standard IAU
  convention and is computed spherically (it implicitly carries the cos(dec)
  factor).  NOTE: this is *not* the same as atan(1/streak_slope) from a plain
  RA/Dec linear fit -- see README_SATMATCH.md, "PA conventions".
* Streak tracks are compared to model tracks using great-circle geometry, so
  no convention from the input catalog is assumed except the endpoint
  coordinates themselves.

Author: written for Erin Cooper's HETDEX PDR1 satellite-streak paper.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- constants
DEG = np.pi / 180.0
R_EARTH_EQ_KM = 6378.137        # WGS84 equatorial radius
R_SUN_KM = 695700.0             # IAU 2015 nominal solar radius
MU_EARTH = 398600.4418          # km^3 / s^2
XKMPER = 6378.135               # km, the Earth radius *inside* SGP4/WGS72

ILLUM_UMBRA, ILLUM_PENUMBRA, ILLUM_SUNLIT = 0, 1, 2
ILLUM_LABELS = {0: "umbra", 1: "penumbra", 2: "sunlit"}


# ------------------------------------------------------------ vector basics
def normalize(v):
    """Return v scaled to unit length along the last axis."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0.0, 1.0, n)


def radec_to_unit(ra_deg, dec_deg):
    """(RA, Dec) in degrees -> unit vector(s) of shape (..., 3)."""
    ra = np.asarray(ra_deg, dtype=float) * DEG
    dec = np.asarray(dec_deg, dtype=float) * DEG
    c = np.cos(dec)
    return np.stack([c * np.cos(ra), c * np.sin(ra), np.sin(dec)], axis=-1)


def unit_to_radec(v):
    """Unit vector(s) (..., 3) -> (RA, Dec) in degrees, RA wrapped to [0, 360)."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1)
    n = np.where(n == 0.0, 1.0, n)
    ra = np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(v[..., 2] / n, -1.0, 1.0)))
    return ra, dec


def angsep(u1, u2):
    """Angular separation in degrees between direction(s), via atan2 (robust
    at both small and large separations, unlike acos(dot))."""
    a = normalize(u1)
    b = normalize(u2)
    cross = np.cross(a, b)
    dot = np.sum(a * b, axis=-1)
    return np.degrees(np.arctan2(np.linalg.norm(cross, axis=-1), dot))


def local_east_north(u):
    """Unit East and North vectors in the tangent plane at direction u.

    Degenerate exactly at the celestial poles; HETDEX never observes there.
    """
    u = normalize(u)
    z = np.zeros_like(u)
    z[..., 2] = 1.0
    east = normalize(np.cross(z, u))
    north = np.cross(u, east)
    return east, north


def position_angle(u_from, u_to):
    """PA (deg, East of North, [0, 360)) of u_to as seen from u_from."""
    u_from = normalize(u_from)
    u_to = normalize(u_to)
    east, north = local_east_north(u_from)
    # component of u_to perpendicular to u_from
    d = u_to - u_from * np.sum(u_from * u_to, axis=-1, keepdims=True)
    return np.degrees(np.arctan2(np.sum(d * east, axis=-1),
                                 np.sum(d * north, axis=-1))) % 360.0


def pa_difference(pa1, pa2):
    """Difference between two position angles modulo 180 deg, in [0, 90].

    Streaks have no direction of travel, so PA and PA+180 are equivalent.
    """
    d = np.abs(np.asarray(pa1, float) - np.asarray(pa2, float)) % 180.0
    return np.minimum(d, 180.0 - d)


def great_circle_pole(u1, u2):
    """Unit normal to the great circle through u1 and u2."""
    return normalize(np.cross(normalize(u1), normalize(u2)))


def perp_sep_to_great_circle(p, pole):
    """Signed-magnitude perpendicular angular distance (deg) from direction p
    to the great circle whose pole is `pole`."""
    return np.abs(90.0 - angsep(p, pole))


def midpoint(u1, u2):
    """Great-circle midpoint of two directions."""
    return normalize(normalize(u1) + normalize(u2))


# ------------------------------------------------------------- illumination
def illumination_state(sat_gcrs_km, sun_gcrs_km):
    """Conical Earth-shadow test (Vallado, Fundamentals of Astrodynamics, §5.3).

    Parameters
    ----------
    sat_gcrs_km : (..., 3) geocentric satellite position, km
    sun_gcrs_km : (..., 3) geocentric Sun position, km

    Returns
    -------
    state : integer array, 0 = umbra, 1 = penumbra, 2 = sunlit
    """
    r = np.asarray(sat_gcrs_km, dtype=float)
    s = np.asarray(sun_gcrs_km, dtype=float)
    r_sun = np.linalg.norm(s, axis=-1)
    sun_dir = s / r_sun[..., None]

    # component of the satellite position along the Earth->Sun direction
    proj = np.sum(r * sun_dir, axis=-1)
    perp = np.linalg.norm(r - proj[..., None] * sun_dir, axis=-1)

    # half-angles of the umbral (converging) and penumbral (diverging) cones
    alpha_u = np.arcsin(np.clip((R_SUN_KM - R_EARTH_EQ_KM) / r_sun, -1, 1))
    alpha_p = np.arcsin(np.clip((R_SUN_KM + R_EARTH_EQ_KM) / r_sun, -1, 1))

    behind = -proj                       # distance down-Sun of the Earth centre
    r_umbra = R_EARTH_EQ_KM - behind * np.tan(alpha_u)
    r_penumbra = R_EARTH_EQ_KM + behind * np.tan(alpha_p)

    state = np.full(np.shape(perp), ILLUM_SUNLIT, dtype=np.int8)
    shadow_side = behind > 0.0
    state = np.where(shadow_side & (perp < r_penumbra), ILLUM_PENUMBRA, state)
    state = np.where(shadow_side & (perp < r_umbra), ILLUM_UMBRA, state)
    return state


def phase_angle(sat_gcrs_km, obs_gcrs_km, sun_gcrs_km):
    """Solar phase angle at the satellite (deg): the Sun-satellite-observer
    angle.  0 deg = fully lit face toward the observer."""
    to_obs = normalize(np.asarray(obs_gcrs_km, float) - np.asarray(sat_gcrs_km, float))
    to_sun = normalize(np.asarray(sun_gcrs_km, float) - np.asarray(sat_gcrs_km, float))
    return angsep(to_obs, to_sun)


# ----------------------------------------------------------- orbit elements
def orbit_geometry(mean_motion_rev_day, eccentricity):
    """Semi-major axis, perigee and apogee heights (km) from TLE mean elements.

    Uses the WGS72 Earth radius that SGP4 itself assumes, so these agree with
    Space-Track's own APOGEE/PERIGEE columns to ~1 km.
    """
    n_rad_s = np.asarray(mean_motion_rev_day, float) * 2.0 * np.pi / 86400.0
    a = (MU_EARTH / n_rad_s ** 2) ** (1.0 / 3.0)
    e = np.asarray(eccentricity, float)
    return a, a * (1.0 - e) - XKMPER, a * (1.0 + e) - XKMPER


def classify_orbit(perigee_km, apogee_km, eccentricity):
    """Coarse orbit class string from perigee/apogee heights."""
    p = float(perigee_km)
    a = float(apogee_km)
    e = float(eccentricity)
    if e > 0.25 and a > 2000.0:
        return "HEO"
    if a < 2000.0:
        return "LEO"
    if 34000.0 < p and a < 37500.0:
        return "GEO"
    if a >= 37500.0:
        return "HEO"
    return "MEO"


# Ordered: first match wins, so put more specific patterns first.
CONSTELLATION_PATTERNS = [
    ("Starlink", ("STARLINK",)),
    ("OneWeb", ("ONEWEB",)),
    ("Kuiper", ("KUIPER",)),
    ("Qianfan/G60", ("QIANFAN", "G60")),
    ("Guowang", ("GUOWANG", "SATNET", "XINGWANG")),
    ("Iridium-NEXT", ("IRIDIUM",)),
    ("Globalstar", ("GLOBALSTAR",)),
    ("Orbcomm", ("ORBCOMM",)),
    ("Planet/Flock", ("FLOCK", "SKYSAT", "DOVE")),
    ("Spire/Lemur", ("LEMUR",)),
    ("SpaceBEE", ("SPACEBEE",)),
    ("Yaogan", ("YAOGAN",)),
    ("Cosmos", ("COSMOS", "KOSMOS")),
    ("GPS/GNSS", ("NAVSTAR", "GPS ", "GALILEO", "GLONASS", "BEIDOU")),
]


def classify_constellation(object_name):
    """Map an object name onto a constellation label ('Other' if unmatched)."""
    if not object_name:
        return "Unknown"
    nm = str(object_name).upper()
    for label, pats in CONSTELLATION_PATTERNS:
        if any(p in nm for p in pats):
            return label
    return "Other"


# ----------------------------------------------------------- coarse filtering
def coarse_threshold(radius_deg, u_samples, ok=None):
    """Per-object coarse-search threshold, widened for inter-sample motion.

    The coarse grid is sampled far more slowly than a LEO object moves: at
    ~0.7 deg/s a satellite covers ~14 deg between 20 s samples.  A fixed radius
    therefore steps straight over real crossings -- it only catches an object
    if a sample happens to land while it is inside the radius -- and biases the
    survivors towards slow, high-altitude objects.

    Widening each object's threshold by half its own largest inter-sample
    motion makes the filter safe for any `coarse_step_s` and any orbit, at the
    cost of passing more objects to the fine pass (which is cheap, and which
    applies the real geometric cut anyway).

    Parameters
    ----------
    radius_deg : base search radius
    u_samples  : (..., nt, 3) topocentric unit vectors
    ok         : (..., nt) optional validity mask

    Returns
    -------
    threshold : (...,) array of per-object thresholds in degrees
    """
    u = np.asarray(u_samples, dtype=float)
    if u.shape[-2] < 2:
        return np.full(u.shape[:-2], float(radius_deg))
    travel = angsep(u[..., 1:, :], u[..., :-1, :])
    if ok is not None:
        ok = np.asarray(ok, dtype=bool)
        travel = np.where(ok[..., 1:] & ok[..., :-1], travel, 0.0)
    return float(radius_deg) + 0.5 * travel.max(axis=-1)


# ------------------------------------------------------------------ scoring
class MatchConfig:
    """Tolerances and search geometry for the streak/TLE association."""

    def __init__(self,
                 coarse_radius_deg=5.0,
                 coarse_step_s=20.0,
                 fine_step_s=0.5,
                 max_perp_arcsec=900.0,
                 max_pa_deg=6.0,
                 max_time_offset_s=2.0,
                 sigma_perp_arcsec=120.0,
                 sigma_pa_deg=1.5,
                 margin_before_s=60.0,
                 margin_after_s=60.0,
                 exposure_span_s=1320.0,
                 n_dither=3,
                 min_altitude_deg=0.0,
                 require_sunlit=False,
                 keep_n_candidates=5,
                 sat_chunk=4000):
        self.coarse_radius_deg = coarse_radius_deg
        self.coarse_step_s = coarse_step_s
        self.fine_step_s = fine_step_s
        self.max_perp_arcsec = max_perp_arcsec
        self.max_pa_deg = max_pa_deg
        # The satellite must actually BE at the streak, not merely lie on a
        # great circle that passes through it.  Expressed as a time offset
        # rather than an angle, because the tolerable angular offset scales
        # with the object's own angular rate:
        #     t_off = (distance to the streak midpoint) / (angular rate)
        # 2 s comfortably covers the fine sampling half-step plus TLE
        # along-track error (~1-5 km, i.e. well under a second), while
        # rejecting an object parked degrees away along the same track.
        self.max_time_offset_s = max_time_offset_s
        self.sigma_perp_arcsec = sigma_perp_arcsec
        self.sigma_pa_deg = sigma_pa_deg
        self.margin_before_s = margin_before_s
        self.margin_after_s = margin_after_s
        # Confirmed by Erin: mjd_shot is the START of a 22 min (1320 s) shot.
        # When set, this overrides the n_dither x exptime guess -- `exptime` is
        # the per-dither value and varies (366.9-728.0 s), so it cannot be used
        # to reconstruct the shot span reliably.
        self.exposure_span_s = exposure_span_s
        self.n_dither = n_dither
        self.min_altitude_deg = min_altitude_deg
        self.require_sunlit = require_sunlit
        self.keep_n_candidates = keep_n_candidates
        # Objects propagated per block in the coarse pass.  Peak memory is
        # roughly sat_chunk * n_timesteps * 120 bytes, so 4000 keeps a worker
        # near 30 MB instead of the ~200 MB an all-at-once pass would need.
        # Lower it if you are running many workers on a small machine.
        self.sat_chunk = sat_chunk

    def window_mjd(self, mjd_shot, exptime_s):
        """Search window for a shot: [mjd_shot, mjd_shot + shot span].

        `mjd_shot` is the start of the shot, which runs for
        `exposure_span_s` (1320 s = 22 min for HETDEX).  Small margins on each
        side absorb timing slop.  If `exposure_span_s` is None the span falls
        back to `n_dither * exptime_s`.

        The fitted crossing time is still reported per streak, so the
        `crossing_dt_s` histogram remains a useful check: it should fill the
        0-1320 s range rather than clustering at an edge.
        """
        span = (self.exposure_span_s if self.exposure_span_s
                else self.n_dither * exptime_s)
        return (mjd_shot - self.margin_before_s / 86400.0,
                mjd_shot + (span + self.margin_after_s) / 86400.0)


def score_track(model_units, model_mjd, ra_start, dec_start, ra_end, dec_end,
                cfg):
    """Compare one propagated track against one observed streak segment.

    Parameters
    ----------
    model_units : (nt, 3) topocentric unit vectors of the satellite
    model_mjd   : (nt,) times of those samples
    ra_start, dec_start, ra_end, dec_end : streak endpoints, degrees
    cfg         : MatchConfig

    Returns
    -------
    dict or None.  None if the track never comes close enough / has the wrong
    orientation.  Keys:
        perp_arcsec   perpendicular offset of the streak midpoint from the
                      model great circle (the primary discriminant)
        sep_mid_arcsec  angular distance from streak midpoint to the nearest
                      model sample (includes the along-track component, which
                      is dominated by TLE timing error and is only a weak
                      constraint)
        pa_diff_deg   |PA(model) - PA(streak)| mod 180
        end_a_arcsec, end_b_arcsec  perpendicular offsets of the two endpoints
        crossing_mjd  time of closest approach to the streak midpoint
        idx           index of that sample
        rate_arcsec_s topocentric angular rate at closest approach
        model_pa_deg  model track PA at closest approach
        streak_pa_deg observed segment PA (spherical, East of North)
        score         chi2-like, lower is better
    """
    model_units = np.asarray(model_units, dtype=float)
    if model_units.shape[0] < 3:
        return None

    a = radec_to_unit(ra_start, dec_start)
    b = radec_to_unit(ra_end, dec_end)
    mid = midpoint(a, b)
    streak_pa = position_angle(mid, b)

    # --- nearest model sample to the streak midpoint
    seps = angsep(model_units, mid[None, :])
    k = int(np.argmin(seps))
    sep_mid = seps[k]

    # cheap rejection before the more careful geometry
    if sep_mid > 3.0 * cfg.coarse_radius_deg:
        return None

    # --- local tangent of the model track, from neighbouring samples
    k0 = max(k - 1, 0)
    k1 = min(k + 1, model_units.shape[0] - 1)
    if k0 == k1:
        return None
    p0, p1 = model_units[k0], model_units[k1]
    if angsep(p0, p1) < 1e-9:
        return None

    pole = great_circle_pole(p0, p1)
    perp = perp_sep_to_great_circle(mid, pole) * 3600.0
    end_a = perp_sep_to_great_circle(a, pole) * 3600.0
    end_b = perp_sep_to_great_circle(b, pole) * 3600.0

    # PA from the bracketing pair, not from model_units[k] -> p1: when the
    # closest approach lands on the final sample those two coincide and the
    # tangent direction becomes numerically ill-conditioned.
    model_pa = position_angle(p0, p1)
    dpa = float(pa_difference(model_pa, streak_pa))

    if perp > cfg.max_perp_arcsec or dpa > cfg.max_pa_deg:
        return None

    dt = (model_mjd[k1] - model_mjd[k0]) * 86400.0
    rate = float(angsep(p0, p1) * 3600.0 / dt) if dt > 0 else np.nan

    # --- was the satellite actually there?
    # `perp` only tests the great circle, which an object sitting degrees away
    # along that same circle satisfies perfectly.  Split the total miss
    # distance into its cross-track and along-track parts,
    #     sep^2 = perp^2 + along^2,
    # and convert only the along-track part into a time offset.  Mixing them
    # would penalise a genuine match that has a real cross-track residual, and
    # would do so hardest for slow objects.
    sep_mid_arcsec = float(sep_mid * 3600.0)
    along = np.sqrt(max(sep_mid_arcsec ** 2 - perp ** 2, 0.0))
    t_off = along / rate if (rate and np.isfinite(rate) and rate > 0) else np.inf
    if t_off > cfg.max_time_offset_s:
        return None

    # closest approach pinned to the first or last sample means the true
    # encounter lies outside the searched window; the fitted time is a lower
    # bound, not a measurement
    at_edge = bool(k == 0 or k == model_units.shape[0] - 1)

    score = (perp / cfg.sigma_perp_arcsec) ** 2 + (dpa / cfg.sigma_pa_deg) ** 2

    return dict(time_offset_s=float(t_off), at_edge=at_edge,
                perp_arcsec=float(perp),
                sep_mid_arcsec=float(sep_mid * 3600.0),
                pa_diff_deg=dpa,
                end_a_arcsec=float(end_a),
                end_b_arcsec=float(end_b),
                crossing_mjd=float(model_mjd[k]),
                idx=k,
                rate_arcsec_s=rate,
                model_pa_deg=float(model_pa),
                streak_pa_deg=float(streak_pa),
                score=float(score))


# ------------------------------------------------- streak photometry helpers
# Every HETDEX shot's calibrated spectrum is flux-normalized to a fixed
# nominal single-dither exposure of 360 s, regardless of the per-shot
# `exptime` metadata column (which varies 366.9-728.0 s across the survey and
# reflects shutter-open/overhead bookkeeping, not the flux-calibration
# reference). `g_mag` is therefore synthesised as if the source shone for
# 360 s, not for `exptime`. Confirmed by Erin, 2026-08-13; supersedes the
# earlier "put the flux on a single-dither basis via EXPDILUT" assumption,
# which used the per-row `exptime` value and over/under-corrected by up to
# ~0.8 mag for the ~82 streaks with exptime far from 360 s.
HETDEX_CAL_EXPTIME_S = 360.0


def instantaneous_magnitude(g_mag, seg_len_arcsec, rate_arcsec_s,
                            exptime_s=HETDEX_CAL_EXPTIME_S):
    """Convert a trail-integrated magnitude to an instantaneous point-source
    magnitude.

    The catalog `g_mag` is synthesised from a spectrum calibrated as if the
    source shone for a fixed reference exposure -- **360 s for HETDEX,
    `HETDEX_CAL_EXPTIME_S`, not the per-row `exptime` catalog column** -- but
    the satellite only illuminated the summed spaxels for

        t_cross = seg_len_arcsec / rate_arcsec_s

    seconds.  Hence

        m_inst = g_mag + 2.5 * log10(t_cross / exptime)

    which is *brighter* (smaller) than g_mag because t_cross << exptime.
    `exptime_s` defaults to the HETDEX calibration reference; pass an
    explicit value only if you have confirmed a different reference applies
    (e.g. for a non-HETDEX catalog reusing this function).
    """
    rate = np.asarray(rate_arcsec_s, float)
    t_cross = np.asarray(seg_len_arcsec, float) / np.where(rate > 0, rate, np.nan)
    return np.asarray(g_mag, float) + 2.5 * np.log10(t_cross / float(exptime_s)), t_cross


def normalize_magnitude_to_range(mag, range_km, ref_km=550.0):
    """Range-normalised magnitude, m(550 km) = m - 5 log10(range / 550).

    This is the quantity tabulated by Mallama (2021) and used throughout the
    satellite-brightness literature.
    """
    r = np.asarray(range_km, float)
    return np.asarray(mag, float) - 5.0 * np.log10(r / ref_km)
