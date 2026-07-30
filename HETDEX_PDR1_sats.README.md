# HETDEX_PDR1_sats.fits — catalog description

Satellite-streak catalog from HETDEX PDR1. One row per distinct satellite pass
("streak"): **533 streaks** across **497 shots** (34 shots contain ≥2 passes).
Each streak has a summed spectrum, propagated errors, SDSS-*g* photometry, and
on-sky geometry.

## File structure

| # | HDU | Type | Shape | Content |
|---|-----|------|-------|---------|
| 0 | `PRIMARY` | — | — | Metadata header (see below). |
| 1 | `INFO` | BinTable | 533 rows × 22 cols | One row per streak (scalar quantities). |
| 2 | `SPECTRA` | Image | (533, 1036) | Summed flux per streak. |
| 3 | `ERRORS` | Image | (533, 1036) | Propagated 1σ errors. |
| 4 | `WAVE` | Image | (1036,) | Wavelength grid (Å) for axis 1 of `SPECTRA`/`ERRORS`. |

Row *i* of `SPECTRA`/`ERRORS` corresponds to row *i* of `INFO`. The wavelength
axis is 3470–5540 Å at 2 Å/pixel (1036 channels), also encoded as a WCS on the
image HDUs (`CRVAL1`, `CDELT1`, `CTYPE1='WAVE'`, `CUNIT1='Angstrom'`).

**Spectral flux unit:** 10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹ (HETDEX PDR1 convention). Flux is
**dilution-corrected** (multiplied by `N_EXP`; see `EXPDILUT`/`N_EXP`) so it
reflects the in-exposure brightness rather than the dither-averaged cube value.

## `INFO` columns

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `streak_id` | int32 | — | Global running index (0-based, unique). Primary key. |
| `shotid` | int64 | — | HETDEX shot identifier (YYYYMMDDsss). |
| `i` | int16 | — | Streak index within the shot (0-based). |
| `n_ifu` | int16 | — | Number of IFUs contributing to the streak. |
| `n_spax` | int32 | — | Total SAT spaxels summed. |
| `area_arcsec2` | float32 | arcsec² | Streak area (`n_spax` × 0.25). |
| `ra_cen_spax` | float32 | deg | Flux-independent centroid RA of the SAT spaxels (on the streak). |
| `dec_cen_spax` | float32 | deg | Centroid Dec of the SAT spaxels. |
| `ra_start` | float32 | deg | RA of the observed endpoint at minimum along-track projection. |
| `dec_start` | float32 | deg | Dec of that endpoint. |
| `ra_end` | float32 | deg | RA of the observed endpoint at maximum along-track projection. |
| `dec_end` | float32 | deg | Dec of that endpoint. |
| `seg_len_arcsec` | float32 | arcsec | Observed segment length along the track. |
| `streak_pa` | float32 | deg | Position angle of the satellite track (from `streak_slope`), E of N, [0,180). |
| `streak_pa_meas` | float32 | deg | *Diagnostic:* PA measured from the SAT spaxels per IFU (WCS/SVD, median). |
| `streak_slope` | float32 | — | Track slope (DEC = `intercept` + RA × slope). |
| `streak_intercept` | float32 | deg | Track intercept. |
| `exptime` | float32 | s | Shot exposure time (from IFU index). |
| `mjd_shot` | float64 | day | Shot MJD (UTC). |
| `g_mag` | float32 | mag | SDSS-*g* AB magnitude of the summed streak spectrum. |
| `sb_mag_arcsec2` | float32 | mag arcsec⁻² | Surface brightness, `g_mag` + 2.5 log₁₀(`area_arcsec2`). |
| `mean_snr` | float32 | — | Mean S/N per pixel in the *g*-band window (3800–5500 Å). |

Notes: All RA/Dec are float32 (≈0.1″ near RA 360°); `mjd_shot` is float64 to
preserve timing. `streak_pa` is the catalog-track PA and is the recommended
value; `streak_pa_meas` is retained for QA and agrees with `streak_pa` to ~1.7°
(median). A per-dither exposure start can be reconstructed as
`[mjd_shot, mjd_shot + exptime/86400]` (no dither cadence is assumed).

## Key `PRIMARY` header keywords

| Keyword | Meaning |
|---------|---------|
| `NSTREAKS` / `NSHOTS` | Number of streaks / shots with ≥1 streak. |
| `SAT_BIT` | Datacube mask bit used to select satellite spaxels (1024). |
| `PIX_SCAL` | Spaxel scale (0.5 arcsec/pixel). |
| `CDELT3` | Å per spectral bin (2.0). |
| `FILTER` | Filter for the *g*-band photometry (`sdss2010-g`). |
| `EXPDILUT` / `N_EXP` | Exposure-dilution correction applied / co-added dithers (factor). |
| `CRVAL1W` | Start wavelength of the spectral axis (Å). |
| `SITELAT` / `SITELONG` / `SITEELEV` | HET site: +30.681436°, −104.014744°, 2026 m. |

## Quick start

```python
from astropy.table import Table
from astropy.io import fits

info = Table.read("HETDEX_PDR1_sats.fits", hdu="INFO")
with fits.open("HETDEX_PDR1_sats.fits") as h:
    wave    = h["WAVE"].data          # (1036,) Angstrom
    spectra = h["SPECTRA"].data       # (533, 1036)
    errors  = h["ERRORS"].data

# spectrum of streak with streak_id == k
k = 0
flux, err = spectra[k], errors[k]
```
