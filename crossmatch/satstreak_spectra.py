"""
satstreak_spectra.py -- stacking the HETDEX streak spectra by satellite type,
and splitting Starlink by brightness-mitigation era.

Streak spectra are reflected sunlight, so every stack is a solar spectrum
modulated by the satellite's surface. Differences *between* classes therefore
live in the continuum slope and in how deep the solar absorption lines are
relative to it -- not in the line positions. That is why the default is to
normalise each spectrum before combining: it compares colour and reflectance
shape rather than brightness, which is what a coating change would alter.

Depends on numpy + matplotlib; astropy only inside `load_spectra`.
"""

from __future__ import annotations

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# Solar absorption features inside the HETDEX range (3470-5540 A), air
# wavelengths.  These are the sanity check on any stack: if they are absent,
# the stack is noise rather than reflected sunlight.
FRAUNHOFER = {
    "Ca II K": 3933.7, "Ca II H": 3968.5, "H-delta": 4101.7,
    "Ca I": 4226.7, "G band": 4300.0, "H-gamma": 4340.5,
    "H-beta": 4861.3, "Mg b": 5175.0,
}

# Starlink brightness mitigation, keyed on the satellite's LAUNCH date -- the
# hardware generation is a property of the spacecraft, not of when it happened
# to be observed.
#   DarkSat  (darkened coating, abandoned - thermal problems)  2020-01-07
#   VisorSat prototype                                          2020-06-04
#   all new satellites carry visors                             2020-08-07
#   v1.5, visor dropped for laser links, dielectric mirror film 2021-09-14
#   Gen2 mini, improved dielectric mirror                       2023-02-27
STARLINK_EPOCHS = [
    ("pre-mitigation", None, 59068.0),      # before visors became standard
    ("VisorSat", 59068.0, 59471.0),
    ("v1.5 mirror film", 59471.0, 60002.0),
    ("Gen2 mini", 60002.0, None),
]

EPOCH_COLORS = {"pre-mitigation": "#d62728", "VisorSat": "#1f77b4",
                "v1.5 mirror film": "#2ca02c", "Gen2 mini": "#9467bd",
                "unknown": "#999999"}


# ------------------------------------------------------------------ helpers
def date_to_mjd(s):
    """'YYYY-MM-DD' -> MJD.  NaN if unparseable (satcat has blanks)."""
    try:
        d = datetime.strptime(str(s).strip()[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return np.nan
    return (d - datetime(1858, 11, 17)).days


def starlink_epoch(launch_date, epochs=STARLINK_EPOCHS):
    """Map a launch date onto a mitigation era label."""
    mjd = date_to_mjd(launch_date)
    if not np.isfinite(mjd):
        return "unknown"
    for label, lo, hi in epochs:
        if (lo is None or mjd >= lo) and (hi is None or mjd < hi):
            return label
    return "unknown"


def add_starlink_epoch(match_table, epochs=STARLINK_EPOCHS):
    """Array of epoch labels for a MATCH table; non-Starlink rows are ''."""
    names = np.asarray(match_table["object_name"])
    launch = np.asarray(match_table["launch_date"])
    out = []
    for nm, ld in zip(names, launch):
        nm = nm.decode() if isinstance(nm, bytes) else str(nm)
        ld = ld.decode() if isinstance(ld, bytes) else str(ld)
        out.append(starlink_epoch(ld, epochs) if "STARLINK" in nm.upper() else "")
    return np.array(out, dtype=object)


def load_spectra(catalog):
    """(wave, flux, err) from the catalog's WAVE/SPECTRA/ERRORS HDUs.

    Flux is in 1e-17 erg/s/cm2/A and already dilution-corrected (x N_EXP), so
    it reflects in-exposure brightness.  Row i matches row i of INFO.
    """
    from astropy.io import fits
    with fits.open(catalog) as h:
        wave = np.asarray(h["WAVE"].data, dtype=float)
        flux = np.asarray(h["SPECTRA"].data, dtype=float)
        err = np.asarray(h["ERRORS"].data, dtype=float)
    return wave, flux, err


# ------------------------------------------------------------------ stacking
def stack(flux, err, mask=None, wave=None, method="median",
          normalize="band", norm_band=(4500.0, 5200.0), clip=5.0):
    """Combine spectra into one stack.

    Parameters
    ----------
    method : "median" (robust, default) or "ivar" (inverse-variance mean)
    normalize : "band"   scale each spectrum to unit median in `norm_band`
                         -- compares SHAPE, which is what a coating changes
                "median" scale to unit median over the whole range
                "none"   keep absolute flux -- compares BRIGHTNESS, and is
                         then dominated by range and angular rate rather than
                         by the satellite's surface
    clip : sigma-clip threshold per wavelength before combining, None to skip

    Returns (stacked_flux, stacked_err, n_used).
    """
    f = np.array(flux, dtype=float, copy=True)
    e = np.array(err, dtype=float, copy=True)
    if mask is not None:
        f, e = f[mask], e[mask]
    if f.ndim == 1:
        f, e = f[None, :], e[None, :]
    n_in = f.shape[0]
    if n_in == 0:
        return np.full(flux.shape[-1], np.nan), np.full(flux.shape[-1], np.nan), 0

    # --- per-spectrum normalisation
    if normalize == "band" and wave is not None:
        sel = (wave >= norm_band[0]) & (wave <= norm_band[1])
        scale = np.nanmedian(np.where(sel[None, :], f, np.nan), axis=1)
    elif normalize in ("band", "median"):
        scale = np.nanmedian(f, axis=1)
    else:
        scale = np.ones(n_in)
    scale = np.where(np.isfinite(scale) & (scale != 0), scale, np.nan)
    f = f / scale[:, None]
    e = e / np.abs(scale)[:, None]

    good = np.isfinite(f) & np.isfinite(e) & (e > 0)

    # --- sigma clip per wavelength
    if clip:
        med = np.nanmedian(np.where(good, f, np.nan), axis=0)
        mad = np.nanmedian(np.abs(np.where(good, f, np.nan) - med), axis=0)
        sd = 1.4826 * mad
        with np.errstate(invalid="ignore"):
            good &= np.abs(f - med) <= clip * np.where(sd > 0, sd, np.inf)

    n_used = good.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        if method == "ivar":
            w = np.where(good, 1.0 / e ** 2, 0.0)
            sw = w.sum(axis=0)
            out = np.where(sw > 0, (np.where(good, f, 0.0) * w).sum(axis=0) / sw,
                           np.nan)
            out_e = np.where(sw > 0, 1.0 / np.sqrt(sw), np.nan)
        else:
            out = np.nanmedian(np.where(good, f, np.nan), axis=0)
            sd = np.nanstd(np.where(good, f, np.nan), axis=0)
            # standard error of the median for a normal parent
            out_e = np.where(n_used > 1,
                             1.2533 * sd / np.sqrt(np.maximum(n_used, 1)), np.nan)
    return out, out_e, int(np.median(n_used))


def stack_by_group(flux, err, labels, wave=None, min_n=3, **kw):
    """{label: (flux, err, n)} for every group with at least `min_n` members."""
    labels = np.asarray(labels, dtype=object)
    out = {}
    for lab in [l for l in dict.fromkeys(labels) if str(l) != ""]:
        m = labels == lab
        if m.sum() < min_n:
            continue
        out[lab] = stack(flux, err, mask=m, wave=wave, **kw)
    return out


# ------------------------------------------------------------------- figures
def mark_fraunhofer(ax, lines=None, fontsize=5.5, color="0.55"):
    """Dotted verticals at the solar absorption features."""
    for name, wl in (lines or FRAUNHOFER).items():
        lo, hi = ax.get_xlim()
        if lo <= wl <= hi:
            ax.axvline(wl, color=color, ls=":", lw=0.6, zorder=0)
            ax.annotate(name, xy=(wl, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(1.5, -2), textcoords="offset points",
                        rotation=90, va="top", ha="left",
                        fontsize=fontsize, color=color)


def plot_spectral_stacks(wave, stacks, colors=None, ratio_to=None,
                         smooth=0, figsize=None, ylabel=None,
                         xlim=(3500, 5500)):
    """Overlaid stacks, optionally with a ratio panel underneath.

    `ratio_to` names one stack to divide the others by -- the most sensitive
    way to see a continuum-slope difference, since a ratio of two solar
    spectra cancels the solar features and leaves only the reflectance
    difference.
    """
    def _smooth(y, w):
        if not w or w < 2:
            return y
        k = np.ones(int(w)) / float(int(w))
        return np.convolve(y, k, mode="same")

    nrow = 2 if ratio_to else 1
    fig, axes = plt.subplots(nrow, 1, sharex=True, squeeze=False,
                            figsize=figsize or (7.1, 2.6 * nrow),
                            gridspec_kw={"height_ratios": [2, 1]} if ratio_to
                            else None)
    ax = axes[0, 0]
    for lab, (f, e, n) in stacks.items():
        c = (colors or {}).get(lab)
        ax.plot(wave, _smooth(f, smooth), lw=0.9, color=c, label=f"{lab} (n={n})")
        if e is not None and np.isfinite(e).any():
            ax.fill_between(wave, _smooth(f - e, smooth), _smooth(f + e, smooth),
                            color=c, alpha=0.18, lw=0)
    ax.set_ylabel(ylabel or "normalised flux")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    mark_fraunhofer(ax)

    if ratio_to and ratio_to in stacks:
        ref = stacks[ratio_to][0]
        rax = axes[1, 0]
        for lab, (f, _e, _n) in stacks.items():
            if lab == ratio_to:
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where(np.isfinite(ref) & (ref != 0), f / ref, np.nan)
            rax.plot(wave, _smooth(r, smooth), lw=0.9,
                     color=(colors or {}).get(lab))
        rax.axhline(1.0, color="k", lw=0.6, ls="--")
        rax.set_ylabel(f"/ {ratio_to}")
        rax.set_xlabel(r"observed wavelength [$\AA$]")
    else:
        ax.set_xlabel(r"observed wavelength [$\AA$]")

    for row_axes in axes:
        for a in row_axes:
            a.set_xlim(xlim)
    fig.tight_layout()
    return fig


def line_depth(wave, flux, center, half_width=8.0, cont_gap=25.0,
               cont_width=25.0):
    """Fractional depth of an absorption line against the local continuum.

    `1 - F_line / F_continuum`, with the continuum taken from two windows
    straddling the line.  0 means no line; 0.3 means the core is 30% below the
    surrounding continuum.

    This is the quantitative form of "does the stack look like sunlight".
    Reflected solar spectra show the Fraunhofer series at a characteristic
    depth; detector artefacts, cosmic rays and non-solar emitters do not.
    """
    wave = np.asarray(wave, float)
    flux = np.asarray(flux, float)
    core = np.abs(wave - center) <= half_width
    left = (wave >= center - cont_gap - cont_width) & (wave <= center - cont_gap)
    right = (wave >= center + cont_gap) & (wave <= center + cont_gap + cont_width)
    cont = left | right
    if core.sum() < 2 or cont.sum() < 4:
        return np.nan
    fc = np.nanmedian(flux[core])
    fb = np.nanmedian(flux[cont])
    if not np.isfinite(fb) or fb == 0:
        return np.nan
    return float(1.0 - fc / fb)


def solar_line_report(wave, stacks, lines=None):
    """Print Fraunhofer depths for each stack -- the 'is this sunlight' test."""
    lines = lines or {k: v for k, v in FRAUNHOFER.items()
                      if k in ("Ca II K", "Ca II H", "H-delta", "G band",
                               "H-beta", "Mg b")}
    names = list(lines)
    print(f"{'stack':>22} {'n':>4} " + " ".join(f"{n:>9}" for n in names))
    for lab, (f, _e, n) in stacks.items():
        vals = [line_depth(wave, f, lines[n_]) for n_ in names]
        cells = " ".join("     --  " if not np.isfinite(v) else f"{v:9.3f}"
                         for v in vals)
        print(f"{str(lab)[:22]:>22} {n:4d} " + cells)
    print("\nfractional depth below the local continuum; reflected sunlight "
          "shows\nthe series at comparable depth across every stack of real "
          "satellites.")


def continuum_slope(wave, flux, blue=(3700.0, 4100.0), red=(5000.0, 5400.0)):
    """Crude colour: median red flux over median blue flux.

    A single number per stack, useful for ordering coatings without reading
    curves off a plot.  >1 means redder than the comparison.
    """
    b = (wave >= blue[0]) & (wave <= blue[1])
    r = (wave >= red[0]) & (wave <= red[1])
    fb = np.nanmedian(np.asarray(flux)[b])
    fr = np.nanmedian(np.asarray(flux)[r])
    return float(fr / fb) if np.isfinite(fb) and fb != 0 else np.nan
