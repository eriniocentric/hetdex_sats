# hetdex_sats

Analysis code, catalog, and figures for the HETDEX PDR1 satellite-streak paper.

We identify satellite trails in the HETDEX Public Data Release 1 (PDR1) fiber
datacubes via the pipeline `SAT` mask, aggregate the masked spaxels per shot,
separate distinct passes using the known satellite-track catalog, and build a
value-added catalog of streak spectra and geometry: `HETDEX_PDR1_sats.fits`
(533 streaks over 497 shots).

## Contents

| Path | Description |
|------|-------------|
| `HETDEX_PDR1_sats.fits` | The satellite-streak catalog: geometry + summed spectra (see [`HETDEX_PDR1_sats.README.md`](HETDEX_PDR1_sats.README.md)). |
| `satellite_streak_spectra_by_shot.ipynb` | Builds the catalog from the PDR1 datacubes (SAT-mask extraction, track assignment, photometry). Writes `HETDEX_PDR1_sats.fits`. |
| `satellite_streak_analysis.ipynb` | Trend analysis: magnitude/surface-brightness distributions, PA rose, brightness vs time / twilight / azimuth, monthly contamination fraction, and comparison to the active-satellite count (GCAT). |
| `satellite_streak_figure.ipynb` | Publication figure(s) for individual streaks. |
| `paper_figures_combined.ipynb` | Assembles the combined multi-panel paper figures. |
| `satellite_tracks.txt` | Per-shot/exposure satellite track lines (`shotid expnum slope intercept`), from `hetdex_api`. |
| `sunset_cache*.fits` | Cached sunset / astronomical-twilight times per observing night. |
| `figures/` | Final paper figures. |
| `focal_plane_diagnostics/` | Per-shot focal-plane plots: IFU footprints + streak path + endpoint coordinates. |
| `shot_review_plots/` | Per-shot review plots: IFU white-light images with SAT contours + summed spectra. |

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
bottom to regenerate `HETDEX_PDR1_sats.fits`; then run
`satellite_streak_analysis.ipynb` for the trend figures. 

## Data provenance

Built from HETDEX PDR1 (Hobby-Eberly Telescope Dark Energy Experiment). Satellite
track lines from the `hetdex_api` known-issues list. Active-satellite counts in
the trend analysis use GCAT (J. McDowell, planet4589.org/space/gcat), CC-BY.

## Citation

If you use this catalog or code, please cite the accompanying paper (Mentuch Cooper et
al., in prep.) and HETDEX PDR1 (Mentuch Cooper et al. 2026, ApJS, 284, 67)
