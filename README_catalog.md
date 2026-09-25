# HETDEX PDR1 Satellite Streak Catalog

`HETDEX_PDR1_satellites.fits` is the catalog of satellite streaks identified
in HETDEX Public Data Release 1 (PDR1) integral-field spectroscopy, with
orbital identifications and per-streak extracted spectra. Companion files
`HETDEX_PDR1_satellites.txt` (AAS machine-readable table format) and
`HETDEX_PDR1_satellites.csv` contain the same per-streak table (the `INFO`
extension below) without the spectra, errors, or candidate-match data — use
the `.fits` file if you need those.

**527** satellite streaks total; **475** identified against archival TLEs
(468 via Space-Track, 7 via SatChecker).

## FITS structure

| # | EXTNAME | Type | Shape | Contents |
|---|---------|------|-------|----------|
| 0 | (primary) | — | — | empty primary header |
| 1 | `INFO` | BINTABLE | 527 rows × 43 cols | one row per streak: geometry, photometry, timing, orbital identification (see below) |
| 2 | `WAVE` | IMAGE | 1036 | common wavelength grid, Å (3470–5540 Å, 2 Å/pixel) |
| 3 | `SPECTRA` | IMAGE | 527 × 1036 | extracted flux per streak; row *i* matches row *i* of `INFO` |
| 4 | `ERRORS` | IMAGE | 527 × 1036 | propagated 1σ flux errors; row *i* matches row *i* of `INFO` |
| 5 | `CANDIDATES` | BINTABLE | 582 rows × 11 cols | up to 5 ranked orbital-match candidates per streak (see below) |

**Flux units:** `SPECTRA`/`ERRORS` are in $10^{-17}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\text{\AA}^{-1}}$,
already corrected for exposure dilution (×3, one value per shot rather than
per dither), so they reflect in-exposure brightness.

## `INFO` columns

| Column | Unit | Description |
|---|---|---|
| `ID` | — | Streak/row identifier |
| `Shot` | — | HETDEX shot ID |
| `MJD` | d | Shutter-open MJD of the shot's first dither |
| `Texp` | s | Exposure time of the dither the streak was detected in |
| `RAcen`, `DEcen` | deg | Streak midpoint (J2000) |
| `RAsta`, `DEsta` | deg | Streak start point |
| `RAend`, `DEend` | deg | Streak end point |
| `Length` | arcsec | Streak length |
| `PA` | deg | Streak position angle |
| `Area` | arcsec² | Masked area |
| `Nifu` | — | Number of IFUs affected |
| `gmag` | mag | Exposure-averaged $g$-band magnitude |
| `gSB` | mag arcsec⁻² | $g$-band surface brightness |
| `SNR` | — | Signal-to-noise ratio |
| `NORAD` | — | NORAD catalog ID of the matched object (`999999` = unidentified) |
| `Name` | — | Matched object's name |
| `COSPAR` | — | International (COSPAR) designator |
| `Type` | — | Object type |
| `Owner` | — | Owner/operator code |
| `Launch` | — | Launch date |
| `Constel` | — | Constellation (e.g. Starlink), if applicable |
| `Class` | — | Orbit class: `LEO`, `MEO`, `GEO`, or `HEO` |
| `Resid` | arcsec | Perpendicular offset of the streak from the propagated track (primary match discriminant) |
| `dPA` | deg | Position-angle residual between streak and propagated track |
| `Uniq` | — | 1 if the best match is clearly separated from the runner-up (see paper §Methods); null = `16959` |
| `MJDx` | d | MJD of closest approach (crossing epoch) of the matched object |
| `TLEage` | h | Age of the TLE used, relative to the crossing epoch |
| `Range` | km | Range to the satellite at crossing |
| `Height` | km | Altitude at crossing |
| `Rate` | arcsec/s | Angular rate at crossing |
| `Incl` | deg | Orbital inclination |
| `Perigee`, `Apogee` | km | Perigee / apogee altitude |
| `Period` | min | Orbital period |
| `ginst` | mag | Instantaneous (transit-time-corrected) $g$ magnitude |
| `FWHM` | arcsec | Seeing FWHM for the dither |
| `Dither` | — | Dither number (1–3) the streak was detected in |
| `MJDopen`, `MJDclos` | d | Shutter-open / -close MJD of that specific dither |
| `IDsrc` | — | Identification source: `spacetrack`, `satchecker`, or blank if unidentified |

Rows with `NORAD = 999999` / blank `IDsrc` (52 streaks) have no accepted
orbital match; their `Class`, `Name`, and other identification-dependent
columns are correspondingly blank/null.

## `CANDIDATES` columns

Up to 5 ranked candidate matches per streak (`rank` 0 = best), linked to
`INFO` by `streak_id`:

| Column | Unit | Description |
|---|---|---|
| `streak_id` | — | Links to `INFO.ID` |
| `rank` | — | 0 = best match, increasing = lower-ranked |
| `norad_id` | — | Candidate object's NORAD ID |
| `object_name` | — | Candidate object's name |
| `perp_arcsec` | arcsec | Perpendicular offset for this candidate |
| `pa_diff_deg` | deg | Position-angle difference for this candidate |
| `score` | — | Match score (lower is better); see paper §Methods for the scoring formula |
| `range_km` | km | Range to the candidate at closest approach |
| `ang_rate_arcsec_s` | arcsec/s | Candidate's angular rate at closest approach |
| `illum_state` | — | Illumination state at crossing (sunlit / penumbra / umbra) |
| `crossing_mjd` | d | MJD of closest approach for this candidate |

## Orbit classification

Orbit class (`INFO.Class`) is assigned from the propagated perigee $h_p$,
apogee $h_a$, and eccentricity $e$:

- **LEO:** $h_a < 2000$ km
- **GEO:** $h_p > 34{,}000$ km and $h_a < 37{,}500$ km
- **HEO:** $e > 0.25$ and $h_a > 2000$ km, or $h_a \geq 37{,}500$ km
- **MEO:** otherwise

## Reproducing / further detail

Full column definitions also appear as Table 1 of the associated article.
The pipeline that produced this catalog (TLE cross-matching, spectral
extraction) is included separately in this deposit's code archive.
