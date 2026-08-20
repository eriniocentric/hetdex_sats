"""
satstreak_plots.py -- diagnostic and publication figures for the streak/TLE
matching, in the ApJL style used elsewhere in the project.

Every function takes plain numpy arrays (no astropy Tables required) and
returns a matplotlib Figure, so they can be exercised without the full
pipeline.  `save()` writes 200 dpi PNG into PLOT_DIR, matching the existing
convention -- PNG only, no PDF.
"""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import satstreak_core as core

PLOT_DIR = "trend_plots/"
ONE_COL = 3.5      # inches, ApJL single column
TWO_COL = 7.1      # inches, ApJL double column

# consistent colours for the common constellations
CONSTELLATION_COLORS = {
    "Starlink": "#d62728", "OneWeb": "#1f77b4", "Kuiper": "#9467bd",
    "Iridium-NEXT": "#2ca02c", "Globalstar": "#8c564b", "Cosmos": "#e377c2",
    "Planet/Flock": "#7f7f7f", "GPS/GNSS": "#bcbd22", "Other": "#17becf",
    "Unknown": "#999999",
}
ILLUM_COLORS = {2: "#d95f02", 1: "#7570b3", 0: "#1b9e77"}


def as_str_array(col):
    """Normalise a label column to python `str`.

    Character columns round-tripped through FITS often come back as bytes
    (dtype 'S').  `b"Other"` is a str-like *sequence* to matplotlib, which
    reads it as five per-bar labels and raises

        ValueError: number of labels (5) does not match number of bars (24)

    Decoding here means every entry point -- including summary_panel, which
    passes raw table columns straight through -- is safe.
    """
    out = []
    for v in np.asarray(col).ravel():
        if isinstance(v, (bytes, np.bytes_)):
            v = v.decode("utf-8", "replace")
        s = str(v).strip()
        out.append(s if s else "Unknown")
    return np.array(out, dtype=object)


def apjl_style(fontsize=8):
    """rcParams for ApJL figures: serif, ticks inside, minor ticks, 200 dpi."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "legend.fontsize": fontsize - 1,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.0,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def save(fig, name, plot_dir=None):
    """Write a 200 dpi PNG (no PDF) and return the path."""
    d = plot_dir if plot_dir is not None else PLOT_DIR
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name if name.endswith(".png") else name + ".png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


# ------------------------------------------------------ per-streak overlay
HET_FOV_RADIUS_ARCMIN = 11.0     # HETDEX focal plane, 22 arcmin across


def plot_streak_match(streak, tracks, best_norad=None, ax=None,
                      pad_arcmin=None, view="auto", max_tracks=6, title=None,
                      fov_radius_arcmin=HET_FOV_RADIUS_ARCMIN, legend=True,
                      min_pad_arcmin=2.5):
    """Observed streak segment and candidate satellite tracks, on the sky.

    Parameters
    ----------
    streak : mapping with ra_start, dec_start, ra_end, dec_end and optionally
             ra_cen_spax / dec_cen_spax and streak_id
    tracks : {norad: {"ra":…, "dec":…, "mjd":…, "name":…}} from
             match_streaks.shot_tracks
    best_norad : highlighted in red and drawn on top
    view : "focal_plane" always frames the full HETDEX field; "streak" zooms to
           the measured segment; "auto" (default) uses the focal plane unless
           the streak is short enough to vanish in it, then zooms.
    pad_arcmin : explicit half-width, overriding `view`.

    Streak lengths span 0.6-17 arcmin (median 11), so a single fixed scale
    either buries the short ones or crops the long ones -- hence the auto mode.
    The HETDEX field circle is drawn whenever it falls inside the frame, giving
    the segment a physical reference.

    Offsets are a tangent plane about the streak midpoint, with +x East, so the
    panel reads like the sky: North up, East left.
    """
    a = core.radec_to_unit(streak["ra_start"], streak["dec_start"])
    b = core.radec_to_unit(streak["ra_end"], streak["dec_end"])
    mid = core.midpoint(a, b)
    ra0, dec0 = core.unit_to_radec(mid)
    cosd = np.cos(np.radians(dec0))

    def offsets(ra, dec):
        dra = (np.asarray(ra, float) - ra0 + 180.0) % 360.0 - 180.0
        return dra * cosd * 60.0, (np.asarray(dec, float) - dec0) * 60.0  # arcmin

    # --- measured segment, and the frame that shows it best
    seg_x, seg_y = offsets([streak["ra_start"], streak["ra_end"]],
                           [streak["dec_start"], streak["dec_end"]])
    seg_len = float(np.hypot(seg_x[1] - seg_x[0], seg_y[1] - seg_y[0]))

    if pad_arcmin is None:
        # `min_pad_arcmin` keeps the very short streaks at a sane scale: at
        # +/-0.7' a 20" positional residual fills a seventh of the panel and
        # looks like a gross mismatch, when it is in fact a good fit.
        if view == "streak":
            pad_arcmin = max(0.75 * seg_len, min_pad_arcmin)
        elif view == "focal_plane":
            pad_arcmin = fov_radius_arcmin * 1.15
        else:                                    # auto
            pad_arcmin = (fov_radius_arcmin * 1.15
                          if seg_len > 0.25 * fov_radius_arcmin
                          else max(0.75 * seg_len, min_pad_arcmin))

    if ax is None:
        fig, ax = plt.subplots(figsize=(ONE_COL, ONE_COL))
    else:
        fig = ax.figure

    # HETDEX focal plane, for physical scale
    if fov_radius_arcmin and pad_arcmin > 0.5 * fov_radius_arcmin:
        ax.add_patch(plt.Circle((0, 0), fov_radius_arcmin, fill=False,
                                color="0.75", lw=0.8, ls="--", zorder=1))
        if pad_arcmin > fov_radius_arcmin:
            # just inside the circle, so it never collides with it or the frame
            ax.annotate(f"HETDEX field ({2*fov_radius_arcmin:.0f}$^\\prime$)",
                        xy=(0, fov_radius_arcmin), xytext=(0, -7),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=6, color="0.55", zorder=1)

    # candidate tracks -- say so explicitly when there are none, rather than
    # silently drawing a bare streak that looks like a failed plot
    if not tracks:
        ax.annotate("no candidate track\n(unmatched streak)", xy=(0.5, 0.06),
                    xycoords="axes fraction", ha="center", fontsize=6,
                    color="#b03030")
    items = list(tracks.items())
    if best_norad is not None:
        items.sort(key=lambda kv: kv[0] != best_norad)
    for i, (norad, tr) in enumerate(items[:max_tracks]):
        x, y = offsets(tr["ra"], tr["dec"])
        # Keep a generous window and let matplotlib clip to the axes.  A tight
        # window fails badly on fast objects: at 0.5 s sampling a LEO track has
        # samples ~20 arcmin apart, so a zoomed panel can contain zero or one
        # of them and the track would silently vanish.  The tangent plane is
        # only trustworthy near the origin, hence the 10 deg guard, which also
        # stops an RA wrap drawing a line across the panel.
        keep = (np.abs(x) < 600.0) & (np.abs(y) < 600.0)
        if keep.sum() < 2:
            continue
        is_best = (norad == best_norad)
        ax.plot(x[keep], y[keep],
                color="#d62728" if is_best else "0.6",
                lw=1.4 if is_best else 0.7,
                zorder=4 if is_best else 2, clip_on=True,
                label=f"{tr.get('name', norad)} ({norad})" if i < 4 else None)

    # measured streak: solid, with distinguishable ends so the sense of the
    # segment is readable.  (The previous white hairline overlay made short
    # streaks look like a hollow blob.)
    ax.plot(seg_x, seg_y, color="k", lw=3.0, solid_capstyle="butt", zorder=5,
            label=f"measured streak ({seg_len:.1f}$^\\prime$)")
    ax.plot(seg_x[0], seg_y[0], "o", color="k", ms=4, zorder=6)
    ax.plot(seg_x[1], seg_y[1], "s", color="k", ms=4, zorder=6)

    if "ra_cen_spax" in streak:
        xc, yc = offsets(streak["ra_cen_spax"], streak["dec_cen_spax"])
        ax.plot(xc, yc, "+", color="w", ms=6, mew=1.1, zorder=7,
                label="SAT spaxel centroid")

    ax.set_xlim(pad_arcmin, -pad_arcmin)          # East left
    ax.set_ylim(-pad_arcmin, pad_arcmin)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\Delta\alpha\cos\delta$ [arcmin]")
    ax.set_ylabel(r"$\Delta\delta$ [arcmin]")
    if title is None and "streak_id" in streak:
        title = (f"streak {int(streak['streak_id'])}  "
                 f"({seg_len:.1f}$^\\prime$ measured)")
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.18), frameon=False,
                  handlelength=1.4, fontsize=6, ncol=2)
    return fig


def plot_match_grid(entries, ncols=3, nrows=3, panel_in=2.2, share_scale=True,
                    **kw):
    """Grid of per-streak overlays.

    `entries` is a list of dicts as produced by
    `match_streaks.gather_for_plot`, each with keys `streak`, `tracks`,
    `best_norad` and optionally `label`.  Panels beyond the supplied entries
    are blanked, and a single shared legend replaces the per-panel ones.

    `share_scale` (default) puts every panel on one angular scale, so the
    residuals really are comparable across the grid.  Without it a 0.9' streak
    is drawn at fifteen times the magnification of a 15' one, and a perfectly
    good match on the short streak looks alarming next to its neighbours.
    """
    n = min(len(entries), ncols * nrows)
    if share_scale and "pad_arcmin" not in kw:
        kw["pad_arcmin"] = HET_FOV_RADIUS_ARCMIN * 1.15
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_in * ncols, panel_in * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes[n:]:
        ax.axis("off")

    for ax, e in zip(axes[:n], entries[:n]):
        plot_streak_match(e["streak"], e.get("tracks", {}),
                          best_norad=e.get("best_norad"), ax=ax,
                          legend=False, title=e.get("label"), **kw)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=5)
        if ax.title.get_text():
            ax.set_title(ax.title.get_text(), fontsize=6)

    # shared axis labels instead of nine copies
    fig.supxlabel(r"$\Delta\alpha\cos\delta$ [arcmin]", fontsize=7)
    fig.supylabel(r"$\Delta\delta$ [arcmin]", fontsize=7)

    handles = [plt.Line2D([], [], color="#d62728", lw=1.4),
               plt.Line2D([], [], color="0.6", lw=0.7),
               plt.Line2D([], [], color="k", lw=3.0),
               plt.Line2D([], [], color="w", marker="+", mec="k", ls="none")]
    fig.legend(handles,
               ["best-match track", "other candidates", "measured streak",
                "SAT spaxel centroid"],
               loc="lower center", ncol=4, frameon=False, fontsize=6,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    return fig


# ------------------------------------------------- time-window calibration
def plot_crossing_times(dt_s, exptime=None, n_dither=3, bins=80, ax=None):
    """Histogram of fitted crossing time minus mjd_shot.

    If the time convention and frames are right, this clusters at one offset
    per dither.  A flat distribution means something upstream is wrong.
    """
    dt_s = np.asarray(dt_s, float)
    dt_s = dt_s[np.isfinite(dt_s)]
    if ax is None:
        fig, ax = plt.subplots(figsize=(ONE_COL, 2.3))
    else:
        fig = ax.figure

    ax.hist(dt_s, bins=bins, color="0.35", histtype="stepfilled", lw=0)
    if exptime:
        for k in range(int(n_dither) + 1):
            ax.axvline(k * exptime, color="#d62728", ls="--", lw=0.7,
                       label="dither boundary" if k == 0 else None)
    ax.axvline(0.0, color="k", ls=":", lw=0.7, label="mjd_shot")
    ax.set_xlabel("crossing time $-$ mjd_shot [s]")
    ax.set_ylabel("streaks")
    ax.legend(frameon=False, loc="upper right")
    return fig


# ------------------------------------------------------------ match quality
def plot_residuals(perp_arcsec, pa_diff_deg, illum_state=None,
                   max_perp=None, max_pa=None):
    """Three-panel QA: perpendicular residual, PA residual, and the
    illumination purity check."""
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL, 2.0))

    perp = np.asarray(perp_arcsec, float)
    perp = perp[np.isfinite(perp)]
    axes[0].hist(perp, bins=40, color="0.35", lw=0)
    if max_perp:
        axes[0].axvline(max_perp, color="#d62728", ls="--", lw=0.7)
    axes[0].set_xlabel("perpendicular offset [arcsec]")
    axes[0].set_ylabel("streaks")
    if len(perp):
        axes[0].text(0.95, 0.92, f"median {np.median(perp):.0f}$^{{\\prime\\prime}}$",
                     transform=axes[0].transAxes, ha="right", va="top")

    dpa = np.asarray(pa_diff_deg, float)
    dpa = dpa[np.isfinite(dpa)]
    axes[1].hist(dpa, bins=40, color="0.35", lw=0)
    if max_pa:
        axes[1].axvline(max_pa, color="#d62728", ls="--", lw=0.7)
    axes[1].set_xlabel(r"$|\Delta\mathrm{PA}|$ [deg]")
    if len(dpa):
        axes[1].text(0.95, 0.92, f"median {np.median(dpa):.2f}$^\\circ$",
                     transform=axes[1].transAxes, ha="right", va="top")

    if illum_state is not None:
        st = np.asarray(illum_state, int)
        labels = [core.ILLUM_LABELS[k] for k in (2, 1, 0)]
        counts = [int((st == k).sum()) for k in (2, 1, 0)]
        axes[2].bar(labels, counts,
                    color=[ILLUM_COLORS[k] for k in (2, 1, 0)], width=0.6)
        axes[2].set_ylabel("matches")
        n_bad = counts[2]
        axes[2].set_title(f"{n_bad} in umbra = false positives", fontsize=7)
    else:
        axes[2].axis("off")
    fig.tight_layout()
    return fig


# -------------------------------------------------------- population plots
def plot_constellation_history(mjd, constellation, bins=24, ax=None,
                               min_count=3):
    """Stacked history of matched streaks by constellation."""
    mjd = np.asarray(mjd, float)
    con = as_str_array(constellation)
    good = np.isfinite(mjd)
    mjd, con = mjd[good], con[good]

    if ax is None:
        fig, ax = plt.subplots(figsize=(TWO_COL, 2.4))
    else:
        fig = ax.figure
    if len(mjd) == 0:
        return fig

    edges = np.linspace(mjd.min(), mjd.max(), bins + 1)
    labels, counts = np.unique(con, return_counts=True)
    order = [l for _, l in sorted(zip(counts, labels), reverse=True)
             if _ >= min_count]
    other = [l for l in labels if l not in order]

    stack = []
    names = []
    for lab in order:
        stack.append(np.histogram(mjd[con == lab], bins=edges)[0])
        names.append(str(lab))
    if other:
        m = np.isin(con, other)
        stack.append(np.histogram(mjd[m], bins=edges)[0])
        names.append("other")

    centres = 0.5 * (edges[1:] + edges[:-1])
    width = np.diff(edges)
    bottom = np.zeros(bins)
    for arr, lab in zip(stack, names):
        ax.bar(centres, arr, width=width, bottom=bottom, label=lab,
               color=CONSTELLATION_COLORS.get(lab, "#cccccc"),
               edgecolor="none")
        bottom += arr

    ax.set_xlabel("MJD")
    ax.set_ylabel("matched streaks")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    return fig


ORBIT_CLASS_COLORS = {"LEO": "#E64B35", "MEO": "#4DBBD5",
                      "GEO": "#00A087", "HEO": "#3C5488",
                      "Unknown": "#8491B4", "Unidentified": "#999999"}
ORBIT_CLASS_ORDER = ["LEO", "MEO", "GEO", "HEO", "Unknown"]


MJD_J2000 = 51544.5          # MJD of 2000-01-01T00:00 UTC


def mjd_to_year(mjd):
    """Decimal year, for readable time axes."""
    return 2000.0 + (np.asarray(mjd, float) - MJD_J2000) / 365.25


def year_to_mjd(year):
    """Inverse of mjd_to_year."""
    return (np.asarray(year, float) - 2000.0) * 365.25 + MJD_J2000


def plot_orbit_class_history(mjd, orbit_class, sat_height_km=None, bins=16,
                             shots_per_bin=None, figsize=None):
    """Orbit-class composition of the matched streaks over time.

    Three panels: absolute counts, the same normalised to a fraction, and
    (given `sat_height_km`) the altitudes themselves.

    **Counts are not a rate.** They fold in how many shots the survey took in
    each interval, which varies a lot across PDR1.  The fraction panel is the
    one that is safe to interpret without a denominator; pass `shots_per_bin`
    (an array matching `bins`) to convert the top panel into streaks per shot.
    """
    mjd = np.asarray(mjd, float)
    cls = as_str_array(orbit_class)
    good = np.isfinite(mjd)
    mjd, cls = mjd[good], cls[good]
    if sat_height_km is not None:
        hgt = np.asarray(sat_height_km, float)[good]

    nrow = 3 if sat_height_km is not None else 2
    fig, axes = plt.subplots(nrow, 1, sharex=True,
                             figsize=figsize or (ONE_COL * 1.6, 1.7 * nrow))
    if len(mjd) == 0:
        return fig

    edges = np.linspace(mjd.min(), mjd.max(), bins + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    width = np.diff(edges)
    present = [c for c in ORBIT_CLASS_ORDER if (cls == c).sum()]
    present += [c for c in np.unique(cls) if c not in ORBIT_CLASS_ORDER]

    stacks = {c: np.histogram(mjd[cls == c], bins=edges)[0] for c in present}
    total = np.sum(list(stacks.values()), axis=0).astype(float)

    # --- counts (or rate per shot)
    bottom = np.zeros(bins)
    scale = np.ones(bins)
    if shots_per_bin is not None:
        scale = np.where(np.asarray(shots_per_bin, float) > 0,
                         np.asarray(shots_per_bin, float), np.nan)
    for c in present:
        axes[0].bar(centres, stacks[c] / scale, width=width, bottom=bottom,
                    color=ORBIT_CLASS_COLORS.get(c, "#cccccc"), label=c,
                    edgecolor="none")
        bottom += stacks[c] / scale
    axes[0].set_ylabel("streaks per shot" if shots_per_bin is not None
                       else "matched streaks")
    axes[0].legend(frameon=False, ncol=len(present), fontsize=6,
                   loc="upper left")
    if shots_per_bin is None:
        axes[0].text(0.99, 0.92, "counts, not a rate: survey cadence varies",
                     transform=axes[0].transAxes, ha="right", va="top",
                     fontsize=5.5, color="0.4")

    # --- fraction (safe without a denominator)
    bottom = np.zeros(bins)
    with np.errstate(invalid="ignore"):
        for c in present:
            frac = np.where(total > 0, stacks[c] / total, np.nan)
            axes[1].bar(centres, frac, width=width, bottom=bottom,
                        color=ORBIT_CLASS_COLORS.get(c, "#cccccc"),
                        edgecolor="none")
            bottom += np.nan_to_num(frac)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("fraction")

    # --- altitudes
    if sat_height_km is not None:
        for c in present:
            k = cls == c
            axes[2].plot(mjd[k], hgt[k], ".", ms=2.5, alpha=0.7,
                         color=ORBIT_CLASS_COLORS.get(c, "#cccccc"))
        axes[2].set_yscale("log")
        axes[2].set_ylabel("height [km]")
        for shell, lab in [(550, "Starlink"), (1200, "OneWeb"),
                           (20200, "GNSS"), (35786, "GEO")]:
            if hgt.min() <= shell <= hgt.max():
                axes[2].axhline(shell, color="0.7", ls=":", lw=0.6)
                axes[2].annotate(lab, xy=(1.0, shell),
                                 xycoords=("axes fraction", "data"),
                                 xytext=(2, 0), textcoords="offset points",
                                 fontsize=5.5, va="center", color="0.5")

    axes[-1].set_xlabel("MJD")
    sec = axes[0].secondary_xaxis("top", functions=(mjd_to_year, year_to_mjd))
    sec.set_xlabel("year", fontsize=7)
    fig.tight_layout()
    return fig


def orbit_class_table(mjd, orbit_class, by="year"):
    """Counts and fractions per orbit class per year, printed."""
    mjd = np.asarray(mjd, float)
    cls = as_str_array(orbit_class)
    good = np.isfinite(mjd)
    mjd, cls = mjd[good], cls[good]
    yr = mjd_to_year(mjd).astype(int)
    present = [c for c in ORBIT_CLASS_ORDER if (cls == c).sum()]
    present += [c for c in np.unique(cls) if c not in ORBIT_CLASS_ORDER]

    print(f"{'year':>6} {'N':>5} " + " ".join(f"{c:>13}" for c in present))
    for y in np.unique(yr):
        k = yr == y
        n = int(k.sum())
        cells = []
        for c in present:
            m = int((cls[k] == c).sum())
            cells.append(f"{m:4d} ({100*m/n:4.0f}%)")
        print(f"{y:>6} {n:5d} " + " ".join(f"{s:>13}" for s in cells))
    print(f"{'all':>6} {len(cls):5d} " + " ".join(
        f"{int((cls==c).sum()):4d} ({100*(cls==c).mean():4.0f}%)".rjust(13)
        for c in present))


def plot_orbit_class_counts(mjd, orbit_class, bins=12, ax=None,
                            shot_mjd=None, rate_classes=("LEO", "GEO")):
    """Grouped (side-by-side) histogram of matched streaks by orbit class.

    Each MJD bin has one bar per orbit class placed next to each other,
    so classes are directly comparable without decoding a stack.

    Parameters
    ----------
    shot_mjd : array-like, optional
        MJD of every survey shot (not just ones with streaks).  When supplied,
        dashed rate lines (streaks per shot) are overlaid on a twin y-axis for
        the classes listed in `rate_classes`.
    rate_classes : sequence of str
        Which orbit classes to draw rate lines for (default LEO and GEO).
    """
    mjd = np.asarray(mjd, float)
    cls = as_str_array(orbit_class)
    good = np.isfinite(mjd)
    mjd, cls = mjd[good], cls[good]

    if ax is None:
        fig, ax = plt.subplots(figsize=(ONE_COL, 2.3))
    else:
        fig = ax.figure
    if len(mjd) == 0:
        return fig

    edges = np.linspace(mjd.min(), mjd.max(), bins + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    bin_w = np.diff(edges).mean()
    present = [c for c in ORBIT_CLASS_ORDER if (cls == c).sum()]
    present += [c for c in np.unique(cls) if c not in ORBIT_CLASS_ORDER]
    n = len(present)
    bar_w = bin_w * 0.8 / n

    stacks = {}
    for i, c in enumerate(present):
        counts = np.histogram(mjd[cls == c], bins=edges)[0]
        stacks[c] = counts
        offset = (i - (n - 1) / 2) * bar_w
        ax.bar(centres + offset, counts, width=bar_w,
               color=ORBIT_CLASS_COLORS.get(c, "#cccccc"), label=c,
               edgecolor="none")

    ax.set_xlabel("MJD")
    ax.set_ylabel("Streaks")
    ax.legend(frameon=False, ncol=1, fontsize=7, loc="upper left")

    sec = ax.secondary_xaxis("top", functions=(mjd_to_year, year_to_mjd))
    sec.set_xlabel("Year", fontsize=7)
    sec.tick_params(labelsize=7)

    # --- optional rate overlay ---
    if shot_mjd is not None:
        shot_mjd = np.asarray(shot_mjd, float)
        shot_mjd = shot_mjd[np.isfinite(shot_mjd)]
        shots_per_bin, _ = np.histogram(shot_mjd, bins=edges)
        scale = np.where(shots_per_bin > 0, shots_per_bin.astype(float), np.nan)

        ax2 = ax.twinx()
        rate_line_styles = ["--", ":"]
        for j, c in enumerate(rate_classes):
            if c not in stacks:
                continue
            rate = stacks[c] / scale
            ls = rate_line_styles[j % len(rate_line_styles)]
            ax2.plot(centres, rate,
                     color=ORBIT_CLASS_COLORS.get(c, "#cccccc"),
                     ls=ls, lw=1.2, alpha=0.7,
                     label=f"{c} rate")
        ax2.set_ylabel("Streaks/shot", fontsize=7)
        ax2.tick_params(axis="y", labelsize=6)
        ax2.yaxis.set_label_position("right")
        ax2.legend(frameon=False, fontsize=7, loc="upper center",
                   bbox_to_anchor=(0.5, 1.0), ncol=1)

    return fig


def plot_orbit_distribution(sat_height_km, orbit_class=None, ax=None):
    """Stacked histogram of satellite altitude, coloured by orbit class."""
    h = np.asarray(sat_height_km, float)
    good = np.isfinite(h)
    h_all = h[good]
    if ax is None:
        fig, ax = plt.subplots(figsize=(ONE_COL, 2.3))
    else:
        fig = ax.figure
    if len(h_all) == 0:
        return fig

    log_bins = np.logspace(np.log10(max(h_all.min(), 100.0)),
                           np.log10(h_all.max() * 1.05), 20)
    if orbit_class is not None:
        cls = as_str_array(orbit_class)[good]
        present = [c for c in ORBIT_CLASS_ORDER if (cls == c).sum()]
        present += [c for c in np.unique(cls) if c not in ORBIT_CLASS_ORDER]
        n = len(present)
        # Log-space stagger: LEO stays at bin centre (aligns with Starlink line);
        # other classes alternate left/right in log space.
        log_bw = np.diff(np.log10(log_bins)).mean()
        bar_w_log = log_bw * 1.1 / n
        log_cen = 0.5 * (np.log10(log_bins[1:]) + np.log10(log_bins[:-1]))
        non_leo = [c for c in present if c != "LEO"]
        offsets = {"LEO": 0.0}
        for j, c in enumerate(non_leo):
            side = 1 if j % 2 == 0 else -1
            offsets[c] = side * ((j // 2) + 1) * bar_w_log
        for c in present:
            off = offsets.get(c, 0.0)
            cen = 10 ** (log_cen + off)
            bw  = 10 ** (log_cen + bar_w_log / 2) - 10 ** (log_cen - bar_w_log / 2)
            counts = np.histogram(h_all[cls == c], bins=log_bins)[0]
            ax.bar(cen, counts, width=bw,
                   color=ORBIT_CLASS_COLORS.get(c, "#cccccc"),
                   edgecolor="none", label=c)
        ax.legend(frameon=False, ncol=1, fontsize=7, loc="upper right")
    else:
        ax.hist(h_all, bins=log_bins, color="0.35", lw=0)

    ax.set_xscale("log")
    ax.set_xlabel("Orbital altitude [km]")
    ax.set_ylabel("Streaks")
    for shell, lab in [(550, "Starlink"), (1200, "OneWeb")]:
        if h_all.min() <= shell <= h_all.max():
            ax.axvline(shell, color="0.3", ls="--", lw=0.7)
            ax.annotate(lab, xy=(shell, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(2, -2), textcoords="offset points",
                        fontsize=6, rotation=90, va="top", ha="left",
                        color="0.3")
    return fig


HET_LAT_DEG = 30.681      # McDonald Observatory geodetic latitude


def _starlink_pa(inclination_deg, lat_deg=HET_LAT_DEG):
    """Expected ascending/descending streak PA (deg E of N) at `lat_deg`
    for a circular orbit at `inclination_deg`.

    Derived from the spherical triangle for the ground track: at latitude φ
    with inclination i, the heading of the ascending track satisfies
    sin(az) = cos(i) / cos(φ).  Returns (pa_asc, pa_desc) or None if the
    inclination does not reach the observer latitude.
    """
    ci = np.cos(np.radians(inclination_deg))
    cf = np.cos(np.radians(lat_deg))
    ratio = ci / cf
    if abs(ratio) > 1.0:
        return None
    pa_asc = float(np.degrees(np.arcsin(ratio))) % 360.0
    pa_desc = (180.0 - pa_asc) % 360.0
    return pa_asc, pa_desc


def plot_pa_rose(pa_deg, orbit_class=None, bins=24, half=True,
                 site_lat_deg=HET_LAT_DEG, ax=None):
    """Polar rose diagram of streak position angles (degrees E of N, 0–360).

    If `orbit_class` is supplied, bars are stacked and coloured by class.
    `half=True` centres the visible 180° arc on the modal direction.
    Dashed grey radial lines mark the expected ascending and descending
    streak PAs for the main Starlink shells (53° and 70° inclination).
    """
    if pa_deg is None:
        pa = np.array([], float)
    else:
        pa = np.asarray(pa_deg, float).ravel()
    good = np.isfinite(pa)
    pa = pa[good]
    pa_mod = pa % 360.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(ONE_COL, ONE_COL),
                               subplot_kw=dict(projection="polar"))
    else:
        fig = ax.figure

    # --- half-rose: re-centre all angles on the modal direction so the
    # arc always runs from -90° to +90° with no wrap-around issues.
    if half and len(pa_mod):
        probe = np.linspace(0.0, 360.0, 361, endpoint=False)
        density = np.array([
            np.sum(np.abs(((pa_mod - a + 180) % 360) - 180) < 15)
            for a in probe])
        peak = float(probe[np.argmax(density)])
        # Shift angles so peak → 0, range [-180, 180]
        pa_c = ((pa_mod - peak + 180) % 360) - 180
        in_arc = np.abs(pa_c) <= 90.0
        pa_rad = np.radians(pa_c)
        edges = np.linspace(-np.pi / 2, np.pi / 2, bins // 2 + 1)
        ax.set_thetamin(-90)
        ax.set_thetamax(90)
        # Tick labels: actual compass directions at -90, -45, 0, +45, +90
        _dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        def _compass(d):
            return _dirs[int(round(d / 45)) % 8]
        tick_degs = np.radians([-90, -45, 0, 45, 90])
        tick_labels = [_compass((peak + d) % 360) for d in (-90, -45, 0, 45, 90)]
        ax.set_xticks(tick_degs)
        ax.set_xticklabels(tick_labels, fontsize=5)
    else:
        in_arc = np.ones(len(pa_mod), bool)
        pa_rad = np.radians(pa_mod)
        edges = np.linspace(0.0, 2 * np.pi, bins + 1)
        ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                           fontsize=5)

    width = edges[1] - edges[0]

    if orbit_class is not None:
        cls = as_str_array(orbit_class)[good]
        present = [c for c in ORBIT_CLASS_ORDER if (cls == c).sum()]
        present += [c for c in np.unique(cls) if c not in ORBIT_CLASS_ORDER]
        bottom = np.zeros(len(edges) - 1)
        for c in present:
            mask = in_arc & (cls == c)
            counts, _ = np.histogram(pa_rad[mask], bins=edges)
            ax.bar(edges[:-1], counts, width=width, bottom=bottom,
                   align="edge", color=ORBIT_CLASS_COLORS.get(c, "#cccccc"),
                   label=c, edgecolor="white", linewidth=0.3)
            bottom += counts
        ax.legend(frameon=False, fontsize=5.5, ncol=1,
                  loc="lower left", bbox_to_anchor=(-0.18, -0.18))
    else:
        counts, _ = np.histogram(pa_rad[in_arc], bins=edges)
        ax.bar(edges[:-1], counts, width=width, align="edge",
               color="0.35", edgecolor="white", linewidth=0.3)

    # --- Starlink shell overlays (ascending + descending per inclination)
    r_max = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0
    _sl_shells = [(53.0, "--", "53°"), (70.0, ":", "70°")]
    for inc, ls, lbl in _sl_shells:
        pas = _starlink_pa(inc, lat_deg=site_lat_deg)
        if pas is None:
            continue
        for pa_actual in pas:
            if half:
                pa_plot = ((pa_actual - peak + 180) % 360) - 180
                if abs(pa_plot) > 90.0:
                    continue
                theta = np.radians(pa_plot)
            else:
                theta = np.radians(pa_actual)
            ax.plot([theta, theta], [0, r_max * 0.92],
                    color="0.55", ls=ls, lw=0.9, zorder=5)
            ax.annotate(lbl, xy=(theta, r_max * 0.94),
                        xycoords="data", ha="center", va="bottom",
                        fontsize=4, color="0.45")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.tick_params(axis="y", labelsize=4)
    # Label below the arc — polar axes don't have a standard xlabel,
    # so place it as an axes-fraction annotation.
    ax.annotate("position angle (° E of N)",
                xy=(0.5, -0.06), xycoords="axes fraction",
                ha="center", va="top", fontsize=6, color="0.2")
    return fig


def plot_pa_histogram(pa_deg, orbit_class=None, bins=18,
                      site_lat_deg=HET_LAT_DEG, ax=None):
    """Histogram of streak position angles (0–180°, E of N).

    Bars are stacked by orbit class when `orbit_class` is supplied.
    Vertical dashed lines mark the expected ascending/descending PAs for the
    Starlink 53° shell at `site_lat_deg`.
    """
    if pa_deg is None:
        pa = np.array([], float)
    else:
        pa = np.asarray(pa_deg, float).ravel()
    good = np.isfinite(pa)
    pa = pa[good] % 360.0
    # Fold to 0–180°: directions > 180° are the same line, opposite sense
    pa = np.where(pa > 180.0, pa - 180.0, pa)

    if ax is None:
        fig, ax = plt.subplots(figsize=(ONE_COL, 2.3))
    else:
        fig = ax.figure
    if len(pa) == 0:
        return fig

    edges = np.linspace(0, 180, bins + 1)

    if orbit_class is not None:
        cls = as_str_array(orbit_class)[good]
        present = [c for c in ORBIT_CLASS_ORDER if (cls == c).sum()]
        present += [c for c in np.unique(cls) if c not in ORBIT_CLASS_ORDER]
        n = len(present)
        bin_w = np.diff(edges).mean()
        bar_w = bin_w * 1.1 / n
        centres = 0.5 * (edges[1:] + edges[:-1])
        non_leo = [c for c in present if c != "LEO"]
        offsets = {"LEO": 0.0}
        for j, c in enumerate(non_leo):
            side = 1 if j % 2 == 0 else -1
            offsets[c] = side * ((j // 2) + 1) * bar_w
        for c in present:
            counts = np.histogram(pa[cls == c], bins=edges)[0]
            ax.bar(centres + offsets.get(c, 0.0), counts, width=bar_w,
                   color=ORBIT_CLASS_COLORS.get(c, "#cccccc"),
                   edgecolor="none", label=c)
    else:
        ax.hist(pa, bins=edges, color="0.35", lw=0)

    # Starlink shell PA overlays — styled like the top-right altitude panel
    _sl_shells = [(53.0, ":", "Starlink 53°"), (70.0, "--", "Starlink 70°")]
    for inc, ls, lbl in _sl_shells:
        pas = _starlink_pa(inc, lat_deg=site_lat_deg)
        if pas is None:
            continue
        for pa_val in pas:
            pa_plot = pa_val if pa_val <= 180.0 else pa_val - 180.0
            ax.axvline(pa_plot, color="0.3", ls=ls, lw=0.9, zorder=5)
            ax.annotate(lbl, xy=(pa_plot, 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(2, -2), textcoords="offset points",
                        fontsize=6, rotation=90, va="top", ha="left",
                        color="0.3")

    # GEO belt: apparent track is inclined to the equator by the orbital
    # inclination, so PA ~ 90 - i.  Median GEO-class inclination is ~14 deg.
    _geo_pa = 90.0 - 14.0
    ax.axvline(_geo_pa, color="0.3", ls="-.", lw=0.9, zorder=5)
    ax.annotate(r"GEO $90^\circ - i$", xy=(_geo_pa, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(2, -2), textcoords="offset points",
                fontsize=6, rotation=90, va="top", ha="left", color="0.3")
    
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel(r"Streak position angle [$^{\circ}$ E of N]")
    ax.set_ylabel("Streaks")
    return fig


def plot_magnitude_vs_range(range_km, g_mag_inst, constellation=None,
                            colors=None, ax=None):
    """Instantaneous magnitude against slant range, with the inverse-square
    locus for reference."""
    r = np.asarray(range_km, float)
    m = np.asarray(g_mag_inst, float)
    good = np.isfinite(r) & np.isfinite(m)
    r, m = r[good], m[good]

    if ax is None:
        fig, ax = plt.subplots(figsize=(ONE_COL, 2.6))
    else:
        fig = ax.figure
    if len(r) == 0:
        return fig

    if constellation is not None:
        cmap = colors if colors is not None else CONSTELLATION_COLORS
        con = as_str_array(constellation)[good]
        for lab in np.unique(con):
            lab = str(lab)
            k = con == lab
            ax.plot(r[k], m[k], ".", ms=3, alpha=0.8, label=lab,
                    color=cmap.get(lab, "#cccccc"))
        ax.legend(frameon=False, ncol=1, fontsize=7, loc="upper left")
    else:
        ax.plot(r, m, ".", ms=3, color="0.35", alpha=0.8)

    rr = np.linspace(r.min(), r.max(), 100)
    m550 = np.median(m - 5 * np.log10(r / 550.0))
    ax.plot(rr, m550 + 5 * np.log10(rr / 550.0), "k--", lw=0.8,
            label=r"$5\log_{10}(d/550)$")
    ax.invert_yaxis()
    ax.set_xlabel("Slant range [km]")
    ax.set_ylabel(r"$g_{\rm inst}$ [mag]")
    return fig


def summary_panel(match, exptime=None):
    """Four-panel summary from a MATCH table (anything with the right keys:
    an astropy Table, a dict of arrays, or a pandas DataFrame).

    Top-left     — orbit-class counts vs time
    Top-right    — satellite altitude distribution, stacked by orbit class
    Bottom-left  — streak PA rose diagram, stacked by orbit class
    Bottom-right — instantaneous magnitude vs slant range, coloured by class
    """
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(TWO_COL, 5.4))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.3],
                  hspace=0.22, wspace=0.22)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])

    oc = match["orbit_class"]
    plot_orbit_class_counts(match["crossing_mjd"], oc, ax=ax00)
    plot_orbit_distribution(match["sat_height_km"], orbit_class=oc, ax=ax01)

    try:
        pa = match["streak_pa_sph_deg"]
    except (KeyError, IndexError, ValueError):
        pa = None
    plot_pa_histogram(pa, orbit_class=oc, ax=ax10)

    plot_magnitude_vs_range(match["range_km"], match["g_mag_inst"],
                            constellation=oc, colors=ORBIT_CLASS_COLORS,
                            ax=ax11)

    # legend only on top-left; strip from all other panels
    for ax in [ax01, ax10, ax11]:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    return fig


def plot_orbit_spectra(pub, catalog, smooth=5, figsize=None):
    """Normalised stacked spectra by orbit class with a brightness inset.

    Each class stack is normalised to unit median flux in 4500-5200 AA so that
    reflectance *shape* (coating colour, Fraunhofer depths) is compared rather
    than brightness.  The inset bar chart restores the brightness context by
    showing median instantaneous g magnitude per class (inverted axis: brighter
    at top).

    Parameters
    ----------
    pub : astropy Table
        Publication table (from ``HETDEX_PDR1_satellites.fits``) with columns
        ``orbit_class``, ``matched``, and ``gmag`` (trail magnitude, MRT name
        unchanged from the publication table).
    catalog : str
        Path to the FITS file containing ``WAVE`` / ``SPECTRA`` / ``ERRORS``
        image HDUs in the same row order as ``pub``.
    smooth : int
        Boxcar smoothing width in pixels for display only (default 5, ~10 AA).
    figsize : tuple, optional
        Defaults to (TWO_COL, 2.6).
    """
    import satstreak_spectra as S

    wave, flux, err = S.load_spectra(catalog)

    if "matched" in pub.colnames:
        matched = np.asarray(pub["matched"], bool)
    else:
        norad   = pub["NORAD"]
        matched = ~np.asarray(norad.mask if hasattr(norad, "mask")
                              else np.zeros(len(pub), bool))
    oc = np.array(
        [((s.decode() if isinstance(s, bytes) else str(s)) if matched[i] else "Unidentified")
         for i, s in enumerate(pub["orbit_class"])],
        dtype=object,
    )

    stacks = S.stack_by_group(flux, err, oc, wave=wave, min_n=3,
                               normalize="band")
    order = [c for c in ORBIT_CLASS_ORDER if c in stacks] + \
            (["Unidentified"] if "Unidentified" in stacks else [])

    def _smooth(y, w):
        if not w or w < 2:
            return y
        k = np.ones(int(w)) / float(int(w))
        return np.convolve(y, k, mode="same")

    fig, ax = plt.subplots(figsize=figsize or (TWO_COL, 2.6))

    for cls in order:
        f, e, _ = stacks[cls]
        n_streaks = int((oc == cls).sum())
        c = ORBIT_CLASS_COLORS.get(cls, "#cccccc")
        g = np.asarray(pub["gmag"][oc == cls], float)
        g_med = np.nanmedian(g[np.isfinite(g)]) if np.isfinite(g).any() else float("nan")
        g_str = f"{g_med:.1f}" if np.isfinite(g_med) else "--"
        ax.plot(wave, _smooth(f, smooth), lw=0.9, color=c,
                label=f"{cls}  N\u2009=\u2009{n_streaks},  $g_{{\\rm med}}$\u2009=\u2009{g_str}")
        if e is not None and np.any(np.isfinite(e)):
            ax.fill_between(wave,
                            _smooth(f - e, smooth), _smooth(f + e, smooth),
                            color=c, alpha=0.18, lw=0)

    ax.set_xlabel(r"Observed wavelength [$\AA$]")
    ax.set_ylabel("Normalized flux")
    ax.set_xlim(3500, 5500)
    ax.legend(frameon=False, fontsize=7, ncol=1, loc="lower right")
    S.mark_fraunhofer(ax)

    fig.tight_layout()
    return fig


def full_summary_panel(match, pub, catalog, smooth=5, figsize=None,
                       shot_mjd=None):
    """Combined figure: normalised spectra by orbit class (top, full width)
    above the four-panel summary (counts, altitude, PA, magnitude).

    Parameters
    ----------
    match : astropy Table
        Matched rows only, with columns already renamed to long internal names
        (``crossing_mjd``, ``orbit_class``, ``sat_height_km``, ``range_km``,
        ``g_mag_inst``, ``streak_pa_sph_deg``).
    pub : astropy Table
        Full 527-row publication table (all streaks, including unmatched),
        with the same column renames applied plus ``gmag`` and ``NORAD``.
    catalog : str
        Path to the FITS file containing ``WAVE`` / ``SPECTRA`` / ``ERRORS``
        image HDUs in the same row order as ``pub``.
    smooth : int
        Boxcar smoothing width in pixels for the spectra (default 5, ~10 AA).
    figsize : tuple, optional
        Defaults to (TWO_COL, 8.0).
    shot_mjd : array-like, optional
        MJD of every survey shot.  When supplied, dashed LEO and GEO rate
        lines (streaks per shot) are overlaid on the middle-left counts panel.
    """
    import satstreak_spectra as S
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=figsize or (TWO_COL, 7.4))
    gs = GridSpec(3, 2, figure=fig,
                  height_ratios=[1.1, 1.0, 1.2],
                  hspace=0.30, wspace=0.30)

    ax_sp = fig.add_subplot(gs[0, :])   # spectrum — full width
    ax00  = fig.add_subplot(gs[1, 0])   # counts vs time
    ax01  = fig.add_subplot(gs[1, 1])   # altitude distribution
    ax10  = fig.add_subplot(gs[2, 0])   # PA histogram
    ax11  = fig.add_subplot(gs[2, 1])   # magnitude vs range

    # ------------------------------------------------------------------ spectra
    wave, flux, err = S.load_spectra(catalog)

    if "matched" in pub.colnames:
        matched = np.asarray(pub["matched"], bool)
    else:
        norad   = pub["NORAD"]
        matched = ~np.asarray(norad.mask if hasattr(norad, "mask")
                              else np.zeros(len(pub), bool))

    oc_pub = np.array(
        [((s.decode() if isinstance(s, bytes) else str(s)) if matched[i] else "Unidentified")
         for i, s in enumerate(pub["orbit_class"])],
        dtype=object,
    )

    stacks = S.stack_by_group(flux, err, oc_pub, wave=wave, min_n=3,
                               normalize="band")
    sp_order = [c for c in ORBIT_CLASS_ORDER if c in stacks] + \
               (["Unidentified"] if "Unidentified" in stacks else [])

    def _smooth(y, w):
        if not w or w < 2:
            return y
        k = np.ones(int(w)) / float(int(w))
        return np.convolve(y, k, mode="same")

    for cls in sp_order:
        f, e, _ = stacks[cls]
        n_streaks = int((oc_pub == cls).sum())
        c = ORBIT_CLASS_COLORS.get(cls, "#cccccc")
        g = np.asarray(pub["gmag"][oc_pub == cls], float)
        g_med = np.nanmedian(g[np.isfinite(g)]) if np.isfinite(g).any() else float("nan")
        g_str = f"{g_med:.1f}" if np.isfinite(g_med) else "--"
        ax_sp.plot(wave, _smooth(f, smooth), lw=0.9, color=c,
                   label=f"{cls}  N\u2009=\u2009{n_streaks},  "
                         f"$g_{{\\rm med}}$\u2009=\u2009{g_str}")
        if e is not None and np.any(np.isfinite(e)):
            ax_sp.fill_between(wave,
                               _smooth(f - e, smooth), _smooth(f + e, smooth),
                               color=c, alpha=0.18, lw=0)

    ax_sp.set_xlabel(r"Observed wavelength [$\AA$]")
    ax_sp.set_ylabel("Normalized flux")
    ax_sp.set_xlim(3500, 5500)
    ax_sp.legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
    S.mark_fraunhofer(ax_sp)

    # ------------------------------------------------------------ four panels
    oc = match["orbit_class"]
    plot_orbit_class_counts(match["crossing_mjd"], oc, ax=ax00, shot_mjd=shot_mjd)
    plot_orbit_distribution(match["sat_height_km"], orbit_class=oc, ax=ax01)

    try:
        pa = match["streak_pa_sph_deg"]
    except (KeyError, IndexError, ValueError):
        pa = None
    plot_pa_histogram(pa, orbit_class=oc, ax=ax10)

    plot_magnitude_vs_range(match["range_km"], match["g_mag_inst"],
                            constellation=oc, colors=ORBIT_CLASS_COLORS, ax=ax11)

    # legend only on counts panel; strip from the other three
    for ax in [ax01, ax10, ax11]:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    fig.subplots_adjust(left=0.09, right=0.93, top=0.94, bottom=0.07,
                        hspace=0.30, wspace=0.30)
    return fig
