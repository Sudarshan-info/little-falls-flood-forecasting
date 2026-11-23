# Little Falls flood forecasting

Time-Series Forecasting course project on one-hour-ahead river-stage prediction and a supplemental staff-review alert at Little Falls, New Jersey.

| Course material | Where to find it |
|---|---|
| Academic report | [Final academic report PDF](course_report/final_review/academic_report_submission_final.pdf) · [Editable Markdown](course_report/final_review/academic_report_submission_final.md) |
| Research poster | [36 × 24 in PDF](course_report/final_review/Little_Falls_Research_Poster_36x24_final_review.pdf) |
| Optional five-minute slides | [PowerPoint](course_report/final_review/Little_Falls_5_Minute_Presentation_Submission.pptx) |
| Source and tests | [Python modules](src/) · [Offline tests](tests/) |
| Data | [Public dataset-v1.0.0 release](https://github.com/Sudarshan-info/little-falls-flood-forecasting/releases/tag/dataset-v1.0.0) |
| V1 final-test results | [Frozen V1 research and reproduction instructions](docs/research_v1/) |
| V2 historical validation | [Protocol, saved results, and reproduction instructions](docs/research_v2/) |

V1 uses 2023 for model selection and 2024–2025 for the final test. V2 uses five earlier validation folds and has no new independent final test. V1 ARIMA overlapped all three final-test floods, with first matched alerts 15, 60 and 0 minutes before raw-stage crossings. The proposed 7-ft alert is for supplemental staff review. The report's $1,325 difference versus persistence comes from an assumed-cost scenario.

### Rebuild the course files

The editable report, figure assets and poster source are in [the course folder](course_report/final_review/). With Chrome installed, run the following in PowerShell to rebuild the final report PDF. The poster and slides are alternate formats for the five-minute presentation.

```powershell
.\.venv\Scripts\python.exe -m pip install Markdown
.\.venv\Scripts\python.exe -B course_report/final_review/build_course_deliverables.py --chrome 'C:\Program Files\Google\Chrome\Application\chrome.exe'
```

This research project forecasts the next hourly river stage at USGS station **01389500**, Passaic River at Little Falls, New Jersey (NOAA gage **LTFN4**). A second task is predicting whether the next hour reaches **7.0 ft**, the minor flood threshold. The thresholds are 6.5 ft (action), 7.0 ft (minor), 8.0 ft (moderate), and 9.0 ft (major). Stage is in feet relative to the gage datum.

Sources: [USGS station and parameter](https://waterdata.usgs.gov/nwis/uv?legacy=1&site_no=01389500), [USGS continuous API schema](https://api.waterdata.usgs.gov/ogcapi/v1/collections/continuous/schema?f=html), and [NOAA flood thresholds](https://water.noaa.gov/gauges/ltfn4).

## Windows setup and offline tests

From the repository directory in PowerShell, with Python 3.11 or 3.12 installed:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

In VS Code, select `.venv\Scripts\python.exe` with **Python: Select Interpreter**. The tests use mocked HTTP responses and temporary files. For a separate synthetic-data preparation check:

```powershell
.\.venv\Scripts\python.exe -m src.demo_synthetic
.\.venv\Scripts\python.exe -m src.prepare --synthetic
```

## Canonical study | October 2007–December 2025

The canonical raw and hourly files are `data/raw/measurements.csv` and `data/processed/hourly_stage.csv`. The historical exploration remains under `data/exploration/2007_2014/`. The original working 2015–2025 CSVs are preserved with verified SHA-256 hashes under `data/backups/pre-canonical-2015-2025/`, alongside the earlier 2021–2025 archive.

### Reproduce from the versioned dataset

Download the ZIP and its `.zip.sha256` file from the public [dataset-v1.0.0 release](https://github.com/Sudarshan-info/little-falls-flood-forecasting/releases/tag/dataset-v1.0.0). The ZIP contains the two canonical CSVs, source URLs and QC metadata, and an internal `SHA256SUMS.txt` manifest. GitHub also provides separate source-code archives.

Run these commands from a fresh project checkout with both release assets in the current directory. Copy the two verified CSVs directly from the archive to reproduce the published modeling inputs.

```powershell
$archive = 'little-falls-usgs-01389500-2007-2025-v1.0.0.zip'
$hash = (Get-Content "$archive.sha256").Split()[0]
if ((Get-FileHash $archive -Algorithm SHA256).Hash -ne $hash) { throw 'Archive checksum mismatch' }
Expand-Archive $archive -DestinationPath data/release-v1.0.0
Copy-Item data/release-v1.0.0/raw/measurements.csv data/raw/measurements.csv
Copy-Item data/release-v1.0.0/processed/hourly_stage.csv data/processed/hourly_stage.csv
Get-FileHash data/raw/measurements.csv,data/processed/hourly_stage.csv -Algorithm SHA256
```

Compare the last two digests with the CSV entries in `data/release-v1.0.0/SHA256SUMS.txt`. The archive preserves the verified file bytes. For a byte-for-byte regeneration, use the Python and pandas versions recorded in `metadata.json` and the same platform line-ending behavior. The tagged repository includes the [QC manifest](outputs/canonical_qc/dataset_qc.json), [annual statistics](outputs/canonical_qc/annual_statistics.csv), and [fixed split counts](outputs/canonical_qc/chronological_split.csv).

With the original source caches and backed-up snapshots available locally, rebuild the staged files and QC reports:

```powershell
.\.venv\Scripts\python.exe -B -m src.finalize_dataset
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

The staging command reconciles the saved USGS responses without downloading or altering either working CSV. It reports duplicate timestamps for review and writes candidates to `data/canonical_stage/`. Annual summaries, event dates, gap inventories, and the dataset hash manifest are under `outputs/canonical_qc/`. Dataset CSVs, source caches and backups stay out of Git.

The fixed UTC target periods are **training October 2007–2022**, **validation 2023**, and **test 2024–2025**. The final partial bucket at `2026-01-01 00:00 UTC` is retained for provenance and excluded from every split. A model feature window uses only observations preceding its target, and any missing hour within the lookback or target makes that window ineligible. State-space models treat missing observations explicitly and apply the gap check to their scored lag spans. Learned scaling uses training data, with settings selected on validation before scoring test.

The primary target remains **stage ≥7.0 ft**. Separate flood events with at least 24 consecutive observed hourly stages below 7.0 ft. Missing hours remain unknown. **6.5 ft is action-stage context.** Record event-level metrics alongside hourly metrics. The three events in each holdout still limit operational claims, and this archive does not establish real-time publication latency.

## Archived Checkpoint 6 | 2015–2025

Checkpoint 6 covered **2015-01-01 through 2025-12-31 UTC**. The downloader fetched 2015–2020 and reused the verified 2021–2025 monthly cache. The overlap matched the previous raw snapshot exactly after CSV parsing. Research outputs from that checkpoint remain in place.

To audit the 2015–2025 snapshot in an isolated copy with those working CSVs:

```powershell
.\.venv\Scripts\python.exe -B -m src.checkpoint6
```

To reproduce from the source, use `src.fetch --start 2015-01-01 --end 2025-12-31`, then `src.prepare`, followed by the audit. The downloader requires explicit `--overwrite` for an existing measurements file. Back up both CSVs and historical outputs before reproducing into working paths.

The extended snapshot has 384,963 source readings and 96,432 hourly study cutoffs. Preparation retains one extra partial row at `2026-01-01 00:00 UTC` for provenance. The fixed study split excludes that row. Raw QC found 766 missing quarter-hour slots, one extra off-grid reading, and zero duplicate timestamps. All source rows are approved, with blank qualifiers. The 186 missing hours are all in training, including a 184-hour outage from 2018-12-30 through 2019-01-07. The stages immediately before and after the long outage were 5.13 ft. Flood occurrence during the outage is unknown. Gaps remain blank. Never interpolate across them with observations from the future or treat unavailable flood labels as non-floods. QC plots and CSVs are in `outputs/checkpoint6/`.

| UTC target cutoffs | Hourly rows | Missing hours | Observed flood events | Flood hours |
|---|---:|---:|---:|---:|
| Training, 2015–2022 | 70,128 | 186 | 4 | 249 |
| Validation, 2023 | 8,760 | 0 | 3 | 237 |
| Test, 2024–2025 | 17,544 | 0 | 3 | 282 |

Observed events use stage at or above 7 ft. Exactly 24 consecutive observed hourly values below 7 ft are sufficient to separate events. Each shorter below-threshold break remains part of the same event. Missing hours are unknown. In this snapshot no observed event spans an outage, all adjacent events have at least 24 confirmed below-threshold hours between them, and no event crosses a split boundary.

These are fixed target-time splits. Forecast inputs at a boundary may use earlier observed stages that were already available at the forecast origin. Fit scaling and other learned transformations on training data. Select settings and warning thresholds using validation, then freeze them before scoring test. The neural trial loop builds test windows only after selection. Classical example orders must be chosen without repeatedly scoring test. Use a fresh set of model artifacts for the extended snapshot. Earlier prediction files and saved model settings belong to the earlier study.

The extended record supports stage-forecast experiments and exploratory flood-classification experiments. Three independent validation events and three test events are too few for a strong operational warning claim. The observation archive does not capture live publication delays. Hourly end-of-bin causality therefore establishes event-time alignment, while operational claims also require checking real publication latency.

## Archived study | 2021–2025

The original study covered **2021-01-01 through 2025-12-31 UTC**. The USGS continuous-data query uses location `USGS-01389500` and parameter `00065` (gage height). These historical commands apply to an isolated copy of the original study:

```powershell
.\.venv\Scripts\python.exe -m src.fetch --start 2021-01-01 --end 2025-12-31
.\.venv\Scripts\python.exe -m src.prepare
.\.venv\Scripts\python.exe -m src.qc
.\.venv\Scripts\python.exe -m src.baselines
```

Fetch dates are inclusive UTC days. Monthly caches are keyed by station, parameter, and exact date slice. Fetch requires `--overwrite` to replace an existing raw `measurements.csv`. Preserve source files and record the retrieval date before regenerating the snapshot. USGS provisional data can be revised. The [USGS download tool](https://api.waterdata.usgs.gov/download/) is another source. Adapt its CSV to the project's expected columns, including `time_utc`, `stage_ft`, and `unit`, and retain source approval, qualifier, and series ID where available.

Preparation uses the last valid reading in each UTC hour ending at the labeled cutoff. The 11:00 row uses readings after 10:00 through 11:00, and `observed_time_utc` records the chosen reading's original timestamp. Missing hourly values stay missing. The hourly file has partial buckets at the beginning and end of this download. Raw data, prepared data, cache files, prediction rows, full-resolution stage figures, and model weights stay local. Only selected small summary files are tracked.

### Archived audit and evaluation limits

The audit rebuilds the 43,825 hourly rows from 175,293 raw readings, checks for source-grid gaps and readings after the hourly cutoff, and writes figures and tables under `outputs/`. The annual summary reports strict and inclusive threshold counts. Flood events use the hourly 7.0 ft threshold. A dip with fewer than 24 consecutive below-threshold hourly readings is part of the same event. A dip of 24 hours or longer separates events. Durations include both endpoint hours and any bridged dips. The sensitivity CSV compares several other separation lengths without changing the 7 ft threshold.

The event count stays at eight for 6, 12, 24, 48, and 72 hours of separation. A one-hour rule yields ten events and a one-week rule yields seven. The 70/15/15 chronological split places all 8 events under the 24-hour rule and all 694 flood hours in training. Validation and test have zero events and zero flood hours under every rule tested. Persistence and the 24-hour-lag baseline measure stage forecast errors in the test period, when stage stays below flood level. Flood recall, missed-event rates, and warning lead time require independent floods in held-out periods. Avoid random hour-level splits, which can put different hours from the same event in training and evaluation.
