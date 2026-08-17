# Satellite identification for HETDEX PDR1 streaks — project context

Working context for `~/work/pdr1/claude-tests/satellites/crossmatch/`.
Named `CLAUDE.md` so an agent working in this directory picks it up
automatically; it reads as plain project documentation either way.

**Goal.** Identify which catalogued satellite produced each streak in
`HETDEX_PDR1_sats.fits`, by propagating archival TLEs with SGP4 and matching
tracks to the measured streak segments. Feeds an ApJL letter on satellite
contamination of HETDEX IFU spectroscopy.

---

## Current state

| | |
|---|---|
| matched | **439 / 527 (83.3%)**, 433 unambiguous |
| median perpendicular residual | **13.9″** (tolerance 900″) |
| median PA residual | **0.08°** (tolerance 6°) |
| umbra false-positive floor | 8/439 = **1.8%** |
| independent validation | **97%** agreement with IAU CPS SatChecker (31/32 resolvable) |
| object mix | PAYLOAD 55%, ROCKET BODY 34%, DEBRIS 10% |

**TLE cache is incomplete**: nights from 2023-08 to 2024-07 (71 nights,
117 streaks) are not fetched. Most of the 88 unmatched streaks are that gap,
not unidentified objects. Fetch before quoting any population fraction.

---

## Layout

```
crossmatch/
├── satellite_streak_identification.ipynb   pipeline driver — start here
├── orbit_class_analysis.ipynb              population figures vs time
├── spectral_stacks_and_starlink.ipynb      spectral stacks; Starlink coating eras
├── satstreak_core.py     geometry, illumination, scoring — pure numpy, 82 tests
├── match_streaks.py      SGP4 + astropy frames, matcher, diagnostics, pub table
├── fetch_tles.py         Space-Track downloader + cache
├── satstreak_plots.py    ApJL-style figures
├── satstreak_spectra.py  spectral stacking, Fraunhofer depths, Starlink eras
├── satstreak_photometry.py  magnitude budget by orbit class
├── satchecker_crosscheck.py   independent validation via IAU CPS API
├── test_geometry.py      82 unit tests, offline, ~1 s
├── tle_cache/            gp_<night>_<date>.3le.gz + satcat.json.gz  (~1.5 GB full)
└── trend_plots/          200 dpi PNG output
../hetdex_sats/HETDEX_PDR1_sats.fits        input catalog
```

**Notebook order.** The pipeline notebook produces
`HETDEX_PDR1_sats_matched.fits`; the other two read it and can be run in either
order afterwards. All three skip work already done — see Performance.

Dependencies beyond the usual stack: `pip install sgp4 joblib`.
Space-Track credentials in `~/.spacetrack.ini`:

```ini
[spacetrack]
identity = you@utexas.edu
password = ...
```

---

## Catalog facts (verified against the file, not the README)

- **527 streaks across 492 shots**, `streak_id` 0–526 contiguous. The bundled
  README says 533/497 and documents 22 INFO columns; the file has 21 (no
  `streak_pa_meas`). The README predates this extraction — reconcile before
  the numbers go in the paper.
- **`mjd_shot` is the START of a 22-minute (1320 s) shot** (confirmed by Erin).
  `exptime` is per-dither and varies 366.9–728.0 s, with 82 streaks above
  500 s, so it *cannot* be used to reconstruct the shot span. Set explicitly
  via `MatchConfig.exposure_span_s = 1320.0`.
- `mjd_shot` is UTC and falls 1.4–11.9 h after midnight UTC on the date encoded
  in `shotid` — i.e. 19:24–05:55 local. Consistent, as documented.
- **`streak_pa` is already a proper spherical PA** east of north. An earlier
  claim that it needed a cos δ correction was wrong: measured against all 527
  streaks the median difference from a spherical PA is 0.000° and the maximum
  0.020°. No convention juggling needed.
- Endpoint separation matches `seg_len_arcsec` to 1 part in 10⁴; endpoints and
  centroid sit on `DEC = intercept + RA × slope` to a median 0.29″.
- Streak lengths 0.64′–17.2′, **median 11.1′**; only 22 streaks are under 2′.
- 370 observing nights. Streaks per year: 2017:4, 2018:24, 2019:47, 2020:63,
  2021:90, 2022:124, 2023:111, 2024:64.

---

## How the matching works

Per shot, over `[mjd_shot − 60 s, mjd_shot + 1320 s + 60 s]`:

1. **Element selection** — nearest-epoch TLE per object for that night.
2. **Coarse pass** — propagate all ~25k objects on a 20 s grid, keep anything
   within `coarse_radius_deg` *widened by half its own inter-sample motion*
   (see gotcha 1).
3. **Fine pass** — re-propagate survivors at 0.5 s.
4. **Score** — perpendicular offset of the streak midpoint from the model great
   circle, plus PA agreement, both computed spherically from the streak
   endpoints. Plus an along-track time-offset cut (gotcha 2).
5. **Rank**, keep the best plus 4 runners-up, flag ambiguity.

**Why perpendicular offset is the discriminant.** TLE error is dominated by the
along-track component, which shifts *when* a satellite reaches a point, not
*where the track lies*. Cross-track error is ~0.1–1′ at LEO; the perpendicular
residual is therefore the meaningful quantity. `match_sep_arcsec` is reported
but is a much weaker constraint.

**Frames.** SGP4 gives TEME; TEME→GCRS is a pure rotation computed once per
shot. Observer GCRS position subtracted per timestep for topocentric
directions. GCRS and ICRS axes are identical, so results compare directly with
the WCS. **No aberration correction is applied, deliberately** — the HETDEX
astrometric solution is tied to catalog star positions, so the same annual
aberration is already removed by the plate solution; the residual is <0.1″.
Altitude uses the geodetic zenith (0.19° from geocentric at HET's latitude).

**Why Space-Track.** CelesTrak's free endpoints serve current elements only
(historical archive is a paid subscription). SatChecker's archive starts
July 2019, which would drop the 55 pre-2019.5 streaks — precisely the
pre-Starlink baseline. Space-Track `gp_history` covers 2017–2024. 3LE format
is used over JSON: ~170 bytes/record vs ~1.5 kB, i.e. 1.5 GB rather than 15 GB.

---

## Gotchas — bugs that reached the science

Each produced plausible-looking output. All are now covered by regression tests.

1. **Fixed-radius coarse filter.** Sampling at 20 s while a LEO object moves
   ~0.7°/s means ~14° of travel between samples. A fixed 5° radius stepped over
   **27% of true crossings** at 0.67°/s, and did so *selectively* — slow objects
   lingered and survived, fast ones were dropped. Symptom: zero matches with
   ~40 objects reported "near the field". Fixed by widening each object's
   threshold by half its own largest inter-sample motion
   (`core.coarse_threshold`).

2. **No along-track constraint.** `perp` tests only the *great circle*, so an
   object sitting degrees away along the same track scored `perp = 0.0″`,
   `dPA = 0.00°` — a perfect match to a satellite that was never there. These
   pin to the window edge and surfaced as crossing times outside the shutter.
   Fixed by decomposing `sep² = perp² + along²` and converting **only** the
   along-track part to a time offset (`max_time_offset_s = 2 s`). Using total
   separation instead penalises genuine cross-track error, worst for slow
   objects — that first attempt was caught by the existing test suite.

3. **Bytes vs str from FITS.** String columns round-trip as bytes; matplotlib
   reads `b"Other"` as five per-bar labels →
   `ValueError: number of labels (5) does not match number of bars (24)`. It
   also silently defeated the constellation colour map. Decode after any
   `Table.read`, or rely on `P.as_str_array`.

4. **Sparse track sampling at high zoom.** At 0.5 s a LEO track has samples
   ~20′ apart, so a zoomed panel could contain fewer than two and the track
   vanished from the plot. Now clipped by the axes instead of pre-filtered.

5. **Tolerance suggestions from truncated percentiles.** Scaling from the 99th
   percentile drifts back toward the cut that produced the table and can
   recommend a *looser* cut. `assess()` now scales from the median.

6. **`mjd_to_year` epoch constant** was wrong by 41 years. Fixed, round trip
   tested.

7. **A single-night TLE gap is not a corrupt-file problem, and does not fix
   the way you'd expect.** `fetch_tles.gp_history_url` restricts each night's
   query to `EPOCH` within `pad_days` (default 1.0) of the shot — objects
   Space-Track didn't re-track that specific day (common for old,
   low-priority debris/rocket bodies) are simply absent from that night's
   file, correctly, not because of a bad download. Two more bugs found
   chasing this on 2026-08-13:
   - `fetch_tles.st_get` treated Space-Track's HTTP 204 ("query succeeded,
     zero rows matched") as fatal, and a single night's empty response
     aborted the whole batch mid-loop — after files had already been
     deleted for a targeted refetch, leaving them worse off than before
     (uncached, not just narrow). Fixed: 204 now returns `""`; any other
     per-night failure is caught, logged, and the batch continues, with
     failed nights reported at the end and safely retryable.
   - **Widening `pad_days` on the whole-catalog query does not work.**
     Tried `--pad-days 30`: 5/10 target nights came back capped at exactly
     `500000` records (Space-Track's `gp_history` response limit, truncated
     in `NORAD_CAT_ID` order — cutting off higher-numbered, more recently
     catalogued objects first) and the other 5 came back with 0 records
     (the same query, rejected outright). A 60-day, whole-catalog
     `gp_history` query is simply too expensive for the API to serve
     reliably. Fixed properly with a **targeted, single-object** query
     instead — `fetch_tles.gp_history_norad_url` / `fetch_missing_norads`
     filter by `NORAD_CAT_ID` as well as `EPOCH`, so the response is a
     handful of rows regardless of window width. `M.rematch_by_norad` checks
     for a `norad_<id>.3le.gz` supplemental file (what `fetch_missing_norads`
     writes) as a fallback when the per-night file doesn't have the object,
     but does not fetch on demand itself — run the fetch first:
     ```python
     import fetch_tles as F
     F.fetch_missing_norads([(norad, mjd_shot), ...], CACHE_DIR)
     ```
   - If `fetch_missing_norads` still returns `None` for an object even at a
     wide window, that's a real Space-Track archive gap for it, not
     anything fetchable — the honest result is an unverified SatChecker-only
     identification (`id_source="satchecker"`), not a bug to keep chasing.
   - **Worst consequence, now fixed**: `main()` originally wrote the "0
     records" response to disk as if it were a confirmed real result. Since
     a non-empty file reads as "already cached" forever, this **silently and
     permanently erased 5 nights' worth of real, narrower TLE data**
     (`59642`, `60292`, `60374`, `60402`, `60430` — 0.1 KB files, caught by
     `audit_tle_cache`'s size check) with the empty result of a rejected
     wide-pad query. `main()` no longer writes anything for a zero-record
     response; it leaves the night unfetched and reports it alongside real
     failures, so it gets retried rather than frozen as empty. Those 5
     nights need restoring with a plain default-pad_days fetch before
     anything downstream trusts them again.

### Two claims that were wrong and are retracted

- **`streak_pa` needs a cos δ correction.** It does not — see catalog facts.
- **The high-orbit population is contamination.** It is real; see below.

---

## The high-orbit population is real

Roughly 43% of matches in a partial run sat above 15,000 km, which looks wrong
given MEO/GEO number in the hundreds against ~25,000 LEO objects. It survives
scrutiny:

- **SatChecker agrees 11/11** on resolvable high-altitude identifications
  (100%), versus 20/21 (95%) for LEO/mid.
- The mechanism is trail surface brightness, **∝ 1/(d²ω)**. A slow object
  dwells far longer on each spaxel, recovering ~3.4 mag of the GEO distance
  penalty. Measured `g_mag_inst ≈ 13.3` at ~36,000 km is exactly where real GEO
  objects sit (V ≈ 11–13).
- Typical identifications are Titan 3C Transtage debris, Breeze-M tanks,
  SL-8/Delta-2 upper stages — well-known GEO-region populations.

**Two checks that do NOT discriminate up there**, recorded so they are not
relied on again:

- **Umbra fraction.** Earth's shadow subtends ~133° of a LEO orbit, ~27° at MEO,
  ~17° at GEO. High-altitude objects are sunlit nearly always, so 0% umbra says
  nothing. It remains a good purity check at LEO.
- **Perpendicular residual.** It is *smaller* for slow objects (9.6″ vs 24.3″)
  purely because 0.5 s sampling puts their track samples 7.5″ apart versus 611″
  for a fast LEO object. Measurement precision, not match quality.

---

## Performance

Parallelism is over nights (independent, similar sizes). `n_jobs=-1` uses all
cores. Three things make high core counts safe:

- **Peak memory is bounded by `cfg.sat_chunk`** (default 4000 objects per
  coarse-pass block, ~30 MB/worker), not by catalog size. This is what
  previously forced `N_JOBS=4`.
- **satcat is trimmed to 7 fields and loaded worker-side.** joblib pickles every
  argument once per task — pass the cache *directory*, not a loaded dict.
- **Satrec objects cached per night**, so `twoline2rv` is not re-run on ~25k
  TLEs for every shot.

The **download is deliberately serial**: Space-Track rate-limits per account
(30/min, 300/hr). Full fetch ~1.5–2 h; full match ~15 min on 16 cores.

Both notebooks skip work that is already done: `M.output_is_current(out,
cache_dir, catalog, cfg=cfg)` compares the output's mtime against every TLE
file, the satcat and the catalog, *and* compares the tolerances recorded in the
output header. Change `max_perp_arcsec` and it rematches; change nothing and it
loads from disk. `FORCE_TRIAL` / `FORCE_MATCH` override.

---

## Output columns

`HETDEX_PDR1_sats_matched.fits`, HDU `MATCH`, same row order as `INFO`:

| group | columns |
|---|---|
| identity | `norad_id`, `object_name`, `object_id`, `object_type`, `country`, `launch_date`, `rcs_size`, `constellation` |
| quality | `match_perp_arcsec`, `match_pa_diff_deg`, `match_sep_arcsec`, `match_time_offset_s`, `match_score`, `second_norad`, `score_margin`, `unambiguous`, `at_window_edge`, `matched` |
| timing | `crossing_mjd`, `crossing_dt_s`, `tle_epoch_mjd`, `tle_age_hours` |
| geometry | `range_km`, `sat_height_km`, `alt_deg`, `ang_rate_arcsec_s` |
| illumination | `illum_state` (0 umbra/1 penumbra/2 sunlit), `phase_angle_deg`, `sun_alt_deg` |
| orbit | `perigee_km`, `apogee_km`, `inclination_deg`, `eccentricity`, `period_min`, `orbit_class` |
| photometry | `t_cross_s`, `g_mag_inst`, `g_mag_inst_550km` |
| diagnostics | `n_propagated`, `n_close`, `n_candidates` |

HDU `CANDIDATES` holds the top 5 per streak for vetting ambiguous cases.

**`range_km` is slant range** — the line-of-sight distance, not altitude. A
550 km satellite spans 550–1,800 km in range depending on elevation, which is
why the low-range end of the magnitude–range plot is a smear.

### Identification provenance — `id_source`

Two pipelines can identify a streak, and they must stay distinguishable.

- `spacetrack` — this matcher's own SGP4 result.
- `satchecker` — patched in by notebook §5b from the IAU CPS API, for streaks
  this matcher searched and rejected.

Set in §5a and written into the MATCH HDU; §5b tags the rows it patches;
published as `IDsrc`. Registered for the publication table by appending to
`M.PUB_COLUMNS` at runtime rather than editing the module.

**Two reasons this matters.** Without it the publication table mixes two
identification methods indistinguishably. And the SatChecker cross-check (§3b.i)
stops being independent of a catalog that has SatChecker results merged into it
— hence the ordering constraint: **always run §3b.i before §5b.**

**Open question before trusting the merged rows.** These are streaks this
matcher *searched and rejected*. A different TLE source (CelesTrak
supplemental) is a good reason to adopt SatChecker's answer; our own tolerances
having rejected the candidate is a reason to be wary, since those are exactly
the marginal cases the cuts exist to exclude. Run `M.diagnose_streak` on a few
before treating them as identifications.

### Publication table

`M.write_publication_table(CATALOG, OUT, out_base="HETDEX_PDR1_sats_ids")`
(pipeline notebook §11) writes FITS, AAS machine-readable text and CSV.

- **42 curated columns**, not the full join, each with a unit and a
  description — the MRT format requires descriptions, and they are what make
  the table usable by anyone else. Definitions in `M.PUB_COLUMNS`.
- **Identification columns are masked where no match was found**, so unmatched
  streaks read as blanks rather than `-1`. With ~17% unmatched, sentinels in a
  published table are a real misreading risk.
- Labels are ≤8 characters and booleans are cast to `int16`, both MRT
  constraints.
- §10 still writes the full `JOINED_OUT` join as the working file.

---

## Why high-orbit objects are detectable — the magnitude budget

`satstreak_photometry.py`. The measured trail magnitude decomposes exactly:

```
g_measured = g(550 km) + range_penalty + trail_dilution

range_penalty  = 5   log10(range / 550 km)      = g_mag_inst - g_mag_inst_550km
trail_dilution = 2.5 log10(exptime / t_cross)   = g_mag      - g_mag_inst
t_cross        = seg_len / angular_rate
```

**The two penalties oppose each other**, which is the whole reason HETDEX
records high-orbit debris. At the median 666″ streak and 367 s exposure:

| | range | rate | t_cross | +range | +trail | total |
|---|---|---|---|---|---|---|
| LEO | 1,000 km | 1500″/s | 0.44 s | +1.3 | +7.3 | 8.6 |
| GEO | 38,000 km | 15″/s | 44 s | +9.2 | +2.3 | 11.5 |

GEO pays 7.9 mag more in distance but saves 5.0 mag in smearing, so the net
penalty differs by only ~2.9 mag rather than ~7.9. Intrinsic brightness
separates the orbit classes by many magnitudes; **measured trail magnitude
barely separates them at all.**

Both penalties are read from the match table rather than recomputed, so the
budget is self-consistent with the photometry by construction — the identity is
asserted to close per class in testing. Because both terms are *differences*,
class-to-class and era-to-era comparisons survive the `g_mag_inst` calibration
caveat even if the absolute zero point moves.

```python
rows = PH.magnitude_budget(match, info=info); PH.print_magnitude_budget(rows)
PH.plot_magnitude_budget(rows)            # stacked penalties + convergence
PH.plot_brightness_distributions(match, info=info)
```

---

## Spectral analysis

`spectral_stacks_and_starlink.ipynb` + `satstreak_spectra.py`. Streak spectra
are reflected sunlight, so every stack is a solar spectrum modulated by the
spacecraft surface — differences live in the **continuum slope** and in line
depth relative to it, not in line positions.

- **Normalisation matters.** Default `normalize="band"` scales each spectrum to
  unit median at 4500–5200 Å before combining, comparing reflectance *shape*.
  `normalize="none"` compares absolute flux, which is then dominated by range
  and angular rate rather than by the surface.
- **The ratio panel is the sensitive view.** Dividing two solar spectra cancels
  the Fraunhofer series and leaves only the reflectance difference. A sloped
  ratio is a real colour difference; flat at 1.0 means indistinguishable at
  this S/N.
- `S.line_depth` / `S.solar_line_report` quantify "is this sunlight" as
  fractional depth below the local continuum. Verified on synthetic spectra:
  Ca II K reads 0.436 for a solar stack versus 0.002 for a featureless
  continuum.
- `S.continuum_slope` gives one red/blue number per stack for ordering
  coatings without reading curves off a plot.

### Starlink mitigation eras

Assigned from **launch date**, not observation date — the coating belongs to
the spacecraft. Boundaries in `S.STARLINK_EPOCHS`:

| era | from | MJD | rationale |
|---|---|---|---|
| pre-mitigation | — | — | before visors became standard |
| VisorSat | 2020-08-07 | 59068 | all new satellites carry sun visors |
| v1.5 mirror film | 2021-09-14 | 59471 | visor dropped for laser links, dielectric mirror |
| Gen2 mini | 2023-02-27 | 60002 | larger bus, improved dielectric mirror |

DarkSat (2020-01-07) was a one-off darkened-coating test abandoned over thermal
problems; it falls in the pre-mitigation bin.

**Phase angle is the main confounder** for any brightness comparison and is not
controlled. Compare phase distributions across eras before attributing a
magnitude difference to coatings. Note also that Gen2 mini launched Feb 2023
and the uncached gap starts 2023-08 — the newest generation is largely missing
until the fetch completes.

### The unmatched streaks

§9 of the spectral notebook stacks them, which answers a question the geometry
cannot: are they reflected sunlight? **Split them first**, using
`n_propagated`:

- `n_propagated == 0` — night never searched, no TLEs cached. Uninformative
  about identification, but an excellent **control**: it should stack
  identically to the matched sample.
- `n_propagated > 0`, no candidate — genuinely searched and not found. The
  informative sample.

Fraunhofer lines at matched depth ⇒ real satellites that went unidentified
(uncatalogued, or no public element set) — a publishable fraction. Lines absent
with a smooth continuum ⇒ not sunlight: artefacts or cosmic rays. Emission
features ⇒ aircraft strobes or municipal lighting. Check `mean_snr` before
reading a shallow-line result as physical.

---

## Open items

1. **Fetch the missing 71 nights** (2023-08 → 2024-07). Section 1b of the
   pipeline notebook reports the gap; the fetch is incremental and the match
   then re-runs itself.
2. **`SHOTS_PER_BIN`** in `orbit_class_analysis.ipynb` §3 — counts are not
   rates without a shot denominator, and PDR1's cadence is uneven. Monthly
   contamination fractions computed elsewhere should slot in.
3. ~~Verify `g_mag_inst`.~~ **Resolved 2026-08-13.** It was normalizing by the
   per-row `exptime` catalog column (366.9–728.0 s), but Erin confirmed HETDEX
   flux calibration is referenced to a fixed 360 s exposure regardless of the
   actual dither length — `exptime` is shutter/overhead bookkeeping, not the
   calibration basis. Fixed: `core.instantaneous_magnitude` now defaults to
   `core.HETDEX_CAL_EXPTIME_S = 360.0`, and the `match_streaks.py` call site no
   longer passes `exptime`. Shifts `g_mag_inst` by up to ~0.8 mag for the ~82
   streaks with `exptime` far from 360 s; negligible near the median (367 s).
   Does not change the LEO/GEO worked example below (both used median values
   near 360 s already). **Still open: re-run §5 of the pipeline notebook** so
   `HETDEX_PDR1_sats_matched.fits`, the publication table, and the magnitude
   budget figures pick up the corrected values before they reach the paper —
   the still-outstanding ~3 mag gap vs. Mallama's Starlink values was only
   partly this effect and needs the post-fix numbers to assess.
4. **Second pass with tighter cuts** (~`max_perp_arcsec=139`, `max_pa_deg=0.8`,
   from 10× median) — a ~50× reduction in chance area, retaining ~93%.
5. **Inspect the 8 umbra matches** (`streak_id` 32, 82, 83, 149, 358, 362, 407,
   417) — physically impossible, so a direct false-positive sample.
6. **One SatChecker disagreement**: streak 459, space-track 11057 vs satchecker
   8438. Both old objects; likely genuine ambiguity for the systematics
   discussion.
7. **`crossing_dt_s` is 67% in the first half** of the shot rather than uniform.
   Unexplained. Plot the histogram — three dither clumps would confirm the
   timing model.
8. **Check the MRT output** (`HETDEX_PDR1_sats_ids.txt`). The AAS
   machine-readable writer is fussy about dtypes and was never exercised
   end to end; FITS and CSV are written independently so a failure there costs
   nothing else.
9. **Stack the unmatched streaks** once the fetch completes — right now the
   "searched, no match" group is the thin remainder after the coverage gap.
10. **Decide whether to keep the §5b SatChecker merges.** Diagnose a few first
    (see *Identification provenance*); `id_source` means the catalog can be
    published either way, with or without them.
11. **Sanity-check the reconstructed `rematch_by_norad`** against
    `diagnose_streak` on one or two pairs before trusting §5b again — see
    *Functions added outside this history* for what was lost and rebuilt, and
    why.
12. **Re-run §5** (full match) with the copied-over `.py` files so
    `g_mag_inst`/`g_mag_inst_550km` reflect the 360 s calibration fix
    (Open item 3) before regenerating the publication table or magnitude
    budget figures.

---

## Conventions

- Figures: ApJL style — serif 8 pt, ticks inside, minor ticks, **200 dpi PNG
  only** (no PDF), to `trend_plots/`. One column 3.5″, two column 7.1″.
- `%autoreload 2` is on in both notebooks; edits to the modules take effect on
  the next cell run. Newly *added* module-level names occasionally need a
  kernel restart.
- Run tests after any edit to `satstreak_core.py`: `python test_geometry.py`,
  82 assertions, ~1 s, no network.
- No subprocess or `!` shell cells — output must stream into the cell and drive
  the `[*]` indicator.
- Space-Track's user agreement permits this use but **not redistribution of raw
  element sets**. Publishing NORAD IDs and derived quantities is fine.
- **Notebook flags have safe resting states**: `DRY_RUN = True` (§1b.ii deletes
  TLE files when False), `FORCE_MATCH` / `FORCE_TRIAL = False` (the staleness
  guard rematches on its own when the cache or `cfg` changes). Pass `cfg=cfg`
  to `output_is_current` or a tolerance change goes undetected.
- Cells that read a file produced by a guarded cell must check
  `os.path.exists` first, or a `False` flag upstream becomes a
  `FileNotFoundError` downstream.
- When patching a notebook programmatically, **match cells on a unique,
  anchored pattern and assert the hit count**. A loose search for `"hstack"`
  once matched the config cell, because it imports `hstack`, and overwrote
  every import in the notebook.

## Functions added outside this history — lost, then reconstructed

`M.audit_tle_cache`, `M.audit_tle_age` and `M.rematch_by_norad` were added to
`match_streaks.py` in a separate JupyterLab session. On 2026-08-13 the whole
file was overwritten with a version that predated them (an exptime-calibration
fix was applied and the file copied over wholesale instead of patched), and
the originals were lost — `.ipynb_checkpoints` and the IPython history sqlite
db (`~/.ipython/profile_default/history.sqlite`) both had call sites for all
three but no `def`.

**They have been reconstructed from those call sites** — the notebook history
cells showed `_PACK_COLS`, which matches this file's actual `_pack()` /
`write_output()` schema exactly, giving high confidence in the intended
output shape — plus the conventions of the surrounding functions
(`build_satrecs(..., norads=...)`, `score_track`, `_pack` itself are reused
directly rather than reimplemented). **This is not the original code.** It has
not been run against the real cache or catalog, and cannot be diffed against
whatever the original actually did.

Before trusting it:
- `audit_tle_cache(cache_dir, full_scan=False)` — file-size heuristic for
  partial/corrupt downloads, matching the two-step usage seen in history
  (fast pass, then `full_scan=True` to confirm). Reasonable to trust as-is;
  worst case it flags nothing or flags something benign — it does not write
  anywhere.
- `audit_tle_age(match_table)` — reads `tle_age_hours`, already a real,
  previously-verified column; low risk for the same reason.
- **`rematch_by_norad(pairs, info, cache_dir, site, cfg, force=False)`** —
  the one that matters. It re-propagates a single named object for a single
  streak and scores it with the same `core.score_track` used everywhere
  else, so `match_perp_arcsec` etc. stay self-consistent with the rest of
  the table — but whether it should *enforce* `cfg`'s normal tolerances
  (what this reconstruction does by default) or bypass them the way the lost
  version might have is a real design choice I inferred, not recovered.
  **Sanity-check it before the §5b cell writes to the science table again**:
  run it on one or two of the `satchecker_unmatched.csv` "found" pairs, and
  compare `match_perp_arcsec` / `match_pa_diff_deg` against what
  `M.diagnose_streak(streak_id, ...)` reports for the same object on the same
  night. Disagreement between the two would mean the reconstruction's
  geometry itself is wrong, not just its policy choices.

Anything added to `satstreak_photometry.py` / `satstreak_spectra.py` is
deliberately in separate modules so it can be updated without merging against
edits made elsewhere.

**Lesson for next time**: don't copy a whole file over a live environment
that may have diverged; patch the specific change, or diff first.

## Citations

- IAU CPS SatChecker — [arXiv:2408.16026](https://arxiv.org/abs/2408.16026)
- Space-Track.org for archival GP data
- Mallama (2021) for satellite instantaneous magnitudes
- Software to cite: `sgp4`, astropy, numpy, scipy, matplotlib, hetdex-api, dexcube
