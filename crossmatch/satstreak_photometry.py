"""
satstreak_photometry.py -- why a GEO object and a Starlink can leave streaks of
similar measured brightness despite differing by ~8 mag in distance.

The measured trail magnitude decomposes exactly:

    g_measured  =  g(550 km)  +  range_penalty  +  trail_dilution

    range_penalty  = 5 log10(range / 550 km)          # inverse square
    trail_dilution = 2.5 log10(T_cal / t_cross)       # smearing along the trail
    t_cross        = seg_len / angular_rate           # dwell on the spaxels
    T_cal          = 360 s, the HETDEX flux-calibration reference exposure
                     (fixed; NOT the per-shot `exptime` catalog column, which
                     varies 366.9-728.0 s and is shutter/overhead bookkeeping,
                     not the calibration basis -- see
                     core.HETDEX_CAL_EXPTIME_S)

The two penalties pull in opposite directions.  A distant object pays a large
range penalty but a small dilution, because it crawls and dwells; a LEO object
pays almost no range penalty but is smeared across the detector in a fraction
of a second.  The partial cancellation is why HETDEX detects high-orbit debris
at all.

Both penalties are read straight from the match table rather than recomputed,
so the budget is self-consistent with `g_mag_inst` by construction:

    range_penalty  = g_mag_inst - g_mag_inst_550km
    trail_dilution = g_mag      - g_mag_inst

Separate module so it can be dropped in without merging against edited files.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

CLASS_ORDER = ["LEO", "MEO", "GEO", "HEO"]
CLASS_COLORS = {"LEO": "#d62728", "MEO": "#1f77b4",
                "GEO": "#2ca02c", "HEO": "#9467bd"}


def _asstr(col):
    out = []
    for v in np.asarray(col).ravel():
        if isinstance(v, (bytes, np.bytes_)):
            v = v.decode("utf-8", "replace")
        out.append(str(v).strip())
    return np.array(out, dtype=object)


def magnitude_budget(match, info=None, classes=None, min_n=3):
    """Per-orbit-class magnitude budget.

    `match` needs g_mag_inst, g_mag_inst_550km, range_km, ang_rate_arcsec_s,
    orbit_class and matched.  `g_mag` comes from `info` if given, else from
    `match` if it carries one.

    Returns a list of dicts, one per class, with medians and 16-84 spreads.
    """
    m = match[np.asarray(match["matched"], bool)] if "matched" in match.colnames \
        else match
    keep = np.asarray(match["matched"], bool) if "matched" in match.colnames \
        else np.ones(len(match), bool)

    g_inst = np.asarray(m["g_mag_inst"], float)
    g_550 = np.asarray(m["g_mag_inst_550km"], float)
    rng = np.asarray(m["range_km"], float)
    rate = np.asarray(m["ang_rate_arcsec_s"], float)
    cls = _asstr(m["orbit_class"])

    if info is not None and "g_mag" in info.colnames:
        g_meas = np.asarray(info["g_mag"], float)[keep]
    elif "g_mag" in m.colnames:
        g_meas = np.asarray(m["g_mag"], float)
    else:
        g_meas = np.full(len(g_inst), np.nan)

    range_pen = g_inst - g_550          # 5 log10(range/550)
    trail_dil = g_meas - g_inst         # 2.5 log10(exptime/t_cross)

    order = classes or [c for c in CLASS_ORDER if (cls == c).sum() >= min_n]
    order += [c for c in np.unique(cls)
              if c not in CLASS_ORDER and (cls == c).sum() >= min_n]

    def q(v, k):
        v = v[k]
        v = v[np.isfinite(v)]
        if v.size == 0:
            return np.nan, np.nan
        return float(np.median(v)), float(np.percentile(v, 84)
                                          - np.percentile(v, 16))

    rows = []
    for c in order:
        k = cls == c
        row = dict(orbit_class=c, n=int(k.sum()))
        for name, v in (("g550", g_550), ("range_penalty", range_pen),
                        ("g_inst", g_inst), ("trail_dilution", trail_dil),
                        ("g_measured", g_meas), ("range_km", rng),
                        ("rate_arcsec_s", rate)):
            row[name], row[name + "_spread"] = q(v, k)
        rows.append(row)
    return rows


def print_magnitude_budget(rows):
    """The table, laid out so the two penalties read as a sum."""
    print(f"{'class':>6} {'n':>4} | {'g(550km)':>9} {'+range':>8} "
          f"{'= g_inst':>9} {'+trail':>8} {'= g_meas':>9} | "
          f"{'range km':>9} {'rate as/s':>10}")
    print("-" * 96)
    for r in rows:
        print(f"{r['orbit_class']:>6} {r['n']:4d} | "
              f"{r['g550']:9.2f} {r['range_penalty']:8.2f} "
              f"{r['g_inst']:9.2f} {r['trail_dilution']:8.2f} "
              f"{r['g_measured']:9.2f} | "
              f"{r['range_km']:9.0f} {r['rate_arcsec_s']:10.0f}")
    print("\nmedians. g_measured = g(550km) + range_penalty + trail_dilution.")
    if len(rows) >= 2:
        a, b = rows[0], rows[-1]
        dr = b["range_penalty"] - a["range_penalty"]
        dt = b["trail_dilution"] - a["trail_dilution"]
        print(f"\n{b['orbit_class']} vs {a['orbit_class']}: "
              f"range penalty {dr:+.1f} mag, trail dilution {dt:+.1f} mag, "
              f"net {dr+dt:+.1f} mag.")
        print(f"The distance penalty is largely cancelled by the slower "
              f"object's longer dwell.")


def plot_magnitude_budget(rows, figsize=None):
    """Two panels: the budget as stacked bars, and where each class lands."""
    fig, axes = plt.subplots(1, 2, figsize=figsize or (7.1, 2.8),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    labs = [r["orbit_class"] for r in rows]
    x = np.arange(len(rows))
    g550 = np.array([r["g550"] for r in rows])
    rp = np.array([r["range_penalty"] for r in rows])
    td = np.array([r["trail_dilution"] for r in rows])

    ax = axes[0]
    ax.bar(x, g550, color="0.55", label="intrinsic, g(550 km)")
    ax.bar(x, rp, bottom=g550, color="#4c72b0", label="range penalty")
    ax.bar(x, td, bottom=g550 + rp, color="#dd8452", label="trail dilution")
    for xi, tot in zip(x, g550 + rp + td):
        ax.plot([xi - 0.4, xi + 0.4], [tot, tot], color="k", lw=1.4)
        ax.annotate(f"{tot:.1f}", xy=(xi, tot), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={r['n']})" for l, r in zip(labs, rows)])
    ax.set_ylabel("magnitude")
    # bars fill the axes from zero, so the legend goes underneath rather than
    # on top of the total annotations
    ax.set_ylim(0, np.nanmax(g550 + rp + td) * 1.12)
    ax.legend(frameon=False, fontsize=6, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), handlelength=1.2)
    ax.set_title("what makes up the measured trail magnitude", fontsize=7)

    ax = axes[1]
    for i, r in enumerate(rows):
        c = CLASS_COLORS.get(r["orbit_class"], "#888888")
        ax.plot([0, 1, 2], [r["g550"], r["g_inst"], r["g_measured"]],
                "o-", ms=4, lw=1.2, color=c, label=r["orbit_class"])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["intrinsic\ng(550 km)", "apparent\ng_inst",
                        "measured\ng trail"], fontsize=6)
    ax.invert_yaxis()
    ax.set_ylabel("magnitude (brighter up)")
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("classes converge as measured", fontsize=7)
    fig.tight_layout()
    return fig


def plot_brightness_distributions(match, info=None, figsize=None, bins=18):
    """Distributions of intrinsic vs measured brightness by class.

    The point of the pair: intrinsic brightness separates the classes by many
    magnitudes, while the measured trail magnitude does not.
    """
    keep = np.asarray(match["matched"], bool) if "matched" in match.colnames \
        else np.ones(len(match), bool)
    m = match[keep]
    cls = _asstr(m["orbit_class"])
    g550 = np.asarray(m["g_mag_inst_550km"], float)
    if info is not None and "g_mag" in info.colnames:
        g_meas = np.asarray(info["g_mag"], float)[keep]
    else:
        g_meas = np.asarray(m["g_mag"], float) if "g_mag" in m.colnames \
            else np.full(len(g550), np.nan)

    order = [c for c in CLASS_ORDER if (cls == c).sum() >= 3]
    fig, axes = plt.subplots(1, 2, figsize=figsize or (7.1, 2.4), sharey=True)
    for ax, v, lab in ((axes[0], g550, "intrinsic, g at 550 km"),
                       (axes[1], g_meas, "measured trail g")):
        good = np.isfinite(v)
        if good.sum() == 0:
            continue
        rng = (np.nanpercentile(v[good], 1), np.nanpercentile(v[good], 99))
        for c in order:
            k = (cls == c) & good
            if k.sum() < 3:
                continue
            ax.hist(v[k], bins=bins, range=rng, histtype="step", lw=1.2,
                    color=CLASS_COLORS.get(c, "#888888"), label=c)
        ax.set_xlabel(lab)
        ax.invert_xaxis()
    axes[0].set_ylabel("streaks")
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    return fig
