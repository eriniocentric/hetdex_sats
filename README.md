# hetdex_sats

Analysis code, catalog, and figures for the HETDEX PDR1 satellite-streak paper.

We identify satellite trails in the HETDEX Public Data Release 1 (PDR1) fiber
datacubes via the pipeline `SAT` mask, aggregate the masked spaxels per shot,
separate distinct passes using the known satellite-track catalog, and build a
value-added catalog of streak spectra and geometry (`intermediate/HETDEX_PDR1_sats.fits`),
then cross-match each streak against archival TLEs to produce the final
identified catalog `HETDEX_PDR1_satellites.fits` (527 streaks, 468 identified).

## Contents

| Path | Description |
|------|-------------|
| `HETDEX_PDR1_satellites.fits` | **Final publication catalog**: 527 streaks, 468 identified, with spectra and CANDIDATES HDUs. |
| `HETDEX_PDR1_satellites.txt` | AAS machine-readable text (MRT format). |
| `HETDEX_PDR1_satellites.csv` | Plain CSV. |
| `crossmatch_and_make_catalog.ipynb` | SGP4 identification pipeline — fetches TLEs, matches all 527 streaks, writes the three files above. **Run this to reproduce.** |
| `crossmatch_figures.ipynb` | Publication figures and match gallery from `HETDEX_PDR1_satellites.fits`. |
| `intermediate/` | Input streak catalog `HETDEX_PDR1_sats.fits` and its documentation. |
| `crossmatch/` | Pipeline modules and diagnostic notebooks (see below). |
| `figures/` | Final paper figures (written by `crossmatch_figures.ipynb`). |
| `satellite_streak_spectra_by_shot.ipynb` | Builds `intermediate/HETDEX_PDR1_sats.fits` from the PDR1 datacubes (SAT-mask extraction, track assignment, photometry). |
| `satellite_streak_analysis.ipynb` | Trend analysis: magnitude/surface-brightness distributions, PA rose, brightness vs time / twilight / azimuth, monthly contamination fraction, and comparison to the active-satellite count (GCAT). |
| `satellite_streak_figure.ipynb` | Publication figure(s) for individual streaks. |
| `paper_figures_combined.ipynb` | Assembles the combined multi-panel paper figures. |
| `satellite_tracks.txt` | Per-shot/exposure satellite track lines (`shotid expnum slope intercept`), from `hetdex_api`. |
| `sunset_cache*.fits` | Cached sunset / astronomical-twilight times per observing night. |
| `focal_plane_diagnostics/` | Per-shot focal-plane plots: IFU footprints + streak path + endpoint coordinates. |
| `shot_review_plots/` | Per-shot review plots: IFU white-light images with SAT contours + summed spectra. |

## Orbital identification (`crossmatch/`)

The `crossmatch/` directory contains the SGP4 pipeline that matches each streak
to a catalogued satellite and the figure notebooks. The final publication table `HETDEX_PDR1_satellites.fits` is
written to the repo root by `crossmatch_and_make_catalog.ipynb`.

| File | Description |
|------|-------------|
| `satellite_streak_identification.ipynb` | Development/diagnostic notebook for the matching algorithm. |
| `match_streaks.py` | SGP4 propagation, scoring, and publication-table writer. |
| `satstreak_core.py` | Geometry, illumination, scoring — pure NumPy, 82 unit tests. |
| `satstreak_plots.py` | ApJL-style figures. |
| `satstreak_spectra.py` | Spectral stacking, Fraunhofer line depths, Starlink eras. |
| `satstreak_photometry.py` | Magnitude budget by orbit class. |
| `fetch_tles.py` | Space-Track `gp_history` downloader and TLE cache manager. |
| `satchecker_crosscheck.py` | Independent validation via IAU CPS SatChecker API. |
| `test_geometry.py` | 82 offline unit tests (~1 s): `python test_geometry.py`. |

Technical context for the whole repo, including this pipeline, is in the
top-level `CLAUDE.md`.

**Prerequisites:**
```
pip install sgp4 joblib
```
A [Space-Track](https://www.space-track.org) account is required to fetch TLEs.
Store credentials in `~/.spacetrack.ini`:
```ini
[spacetrack]
identity = you@example.com
password = yourpassword
```

**Run order:** `crossmatch_and_make_catalog.ipynb` (repo root) → `crossmatch_figures.ipynb`.
The TLE cache (`crossmatch/tle_cache/`, ~1.5 GB) is not included in this
repository (Space-Track redistribution terms). It is populated automatically
when you run `crossmatch_and_make_catalog.ipynb` with valid credentials.

**Output:** `HETDEX_PDR1_satellites.fits` (repo root) — 527 streaks, 468
identified (88.8%): 462 via Space-Track SGP4 propagation and 6 via SatChecker,
distinguished by the `IDsrc` column. Contains the publication table plus
WAVE/SPECTRA/ERRORS and CANDIDATES HDUs, so it is fully self-contained.

Summary of the identifications:

| | |
|---|---|
| identified | 468 / 527 (88.8%) |
| unambiguous | 463 / 468 (98.9%) |
| median cross-track residual | 14.0″ (90th pct 71.7″) |
| median PA residual | 0.08° (90th pct 0.59°) |
| umbra false-positive floor | 6 (1.3%) |
| object type | 254 payload, 166 rocket body, 48 debris |
| orbit class | 191 LEO, 64 MEO, 97 GEO, 116 HEO |
| constellation | 34 Starlink, 50 Cosmos, 21 OneWeb, 15 Globalstar |

**Search window.** Each shot is searched over
`[mjd_shot − 60 s, mjd_shot + T_shot + 60 s]` where
`T_shot = 3 × exptime + 240 s` — three dithers plus ~4 min of overhead.
`T_shot` must be evaluated per shot: `exptime` varies 366.9–728.0 s, so the
window ranges from 1341 s at the median exposure to 2423 s at the longest.
This is `MatchConfig.overhead_s` (240 s) with `exposure_span_s = None`;
setting `exposure_span_s` overrides the formula with a fixed span.

---

## Manually removed observations

Five observations were removed from the analysis after spectral inspection
revealed the emission to be meteors rather than reflected sunlight from
satellites:

| shotid | reason |
|--------|--------|
| 20190731019 | meteor (emission features in spectrum) |
| 20200523028 | meteor |
| 20220928016 | meteor |
| 20230720009 | meteor |
| 20240311025 | meteor |

These shots are excluded from `intermediate/HETDEX_PDR1_sats.fits` and all downstream
products.

---

## Method (brief)

1. **Select** IFUs flagged for satellites in the PDR1 IFU index (`flag_satellite < 0.9`).
2. **Extract** SAT-masked spaxels per IFU from the datacubes, applying quality-bit
   masking (MAIN, FTF, BADPIX, BADAMP) but keeping the satellite flux.
3. **Assign** spaxels to individual passes by proximity to the track lines in
   `satellite_tracks.txt`, so shots with multiple crossings are split.
4. **Sum** flux and propagate errors per streak; correct for exposure dilution
   (the trail appears in one of the co-added dithers).
5. **Measure** SDSS-*g* AB magnitude (via `speclite`), surface brightness, an
   on-track spaxel centroid, observed segment endpoints, and position angle.

## Reproducing

Requires the HETDEX PDR1 datacubes (`dex_cube_*.fits`) and IFU index, plus
`numpy`, `astropy`, `speclite`, `joblib`, `matplotlib`. Set `pdr_dir` at the top
of `satellite_streak_spectra_by_shot.ipynb` to your PDR1 path and run top to
bottom to regenerate `intermediate/HETDEX_PDR1_sats.fits`; then run
`crossmatch_and_make_catalog.ipynb` to produce the final catalog; then run
`crossmatch_figures.ipynb` for the publication figures.

## Data provenance

Built from HETDEX PDR1 (Hobby-Eberly Telescope Dark Energy Experiment). Satellite
track lines from the `hetdex_api` known-issues list. Active-satellite counts in
the trend analysis use GCAT (J. McDowell, planet4589.org/space/gcat), CC-BY.

## Citation

If you use this catalog or code, please cite the accompanying paper (Mentuch Cooper et
al., in prep.) and HETDEX PDR1 (Mentuch Cooper et al. 2026, ApJS, 284, 67)
