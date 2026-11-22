# Little Falls flood forecasting

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

## Real observations and preparation

The study snapshot covers **2021-01-01 through 2025-12-31 UTC**. The USGS continuous-data query uses location `USGS-01389500` and parameter `00065` (gage height). If the files already exist, skip the download and preparation and run the audit directly.

```powershell
.\.venv\Scripts\python.exe -m src.fetch --start 2021-01-01 --end 2025-12-31
.\.venv\Scripts\python.exe -m src.prepare
.\.venv\Scripts\python.exe -m src.qc
.\.venv\Scripts\python.exe -m src.baselines
```

Fetch dates are inclusive UTC days. Monthly caches are keyed by station, parameter, and exact date slice. Fetch requires `--overwrite` to replace an existing raw `measurements.csv`. Preserve source files and record the retrieval date before regenerating the snapshot. USGS provisional data can be revised. The [USGS download tool](https://api.waterdata.usgs.gov/download/) is another source. Adapt its CSV to the project's expected columns, including `time_utc`, `stage_ft`, and `unit`, and retain source approval, qualifier, and series ID where available.

Preparation uses the last valid reading in each UTC hour ending at the labeled cutoff. The 11:00 row uses readings after 10:00 through 11:00, and `observed_time_utc` records the chosen reading's original timestamp. Missing hourly values stay missing. The hourly file has partial buckets at the beginning and end of this download. Raw data, prepared data, cache files, prediction rows, full-resolution stage figures, and model weights stay local. Only selected small summary files are tracked.

## Audit and evaluation limits

The audit rebuilds the 43,825 hourly rows from 175,293 raw readings, checks for source-grid gaps and readings after the hourly cutoff, and writes figures and tables under `outputs/`. The annual summary reports strict and inclusive threshold counts. Flood events use the hourly 7.0 ft threshold. A dip with fewer than 24 consecutive below-threshold hourly readings is part of the same event. A dip of 24 hours or longer separates events. Durations include both endpoint hours and any bridged dips. The sensitivity CSV compares several other separation lengths without changing the 7 ft threshold.

The 70/15/15 chronological split places all 8 flood events and 694 flood hours in training. Validation and test have zero events and zero flood hours. Persistence and the 24-hour-lag baseline measure stage forecast errors on the dry test period. Flood recall, missed-event rates, and warning lead time require independent floods in held-out periods. Avoid random hour-level splits, which can put different hours from the same event in training and evaluation.

## Course and dissertation scope

- **Time-Series course.** Focus on one-hour-ahead stage forecasting, exploratory diagnostics, transparent baselines, and comparisons with classical models. Select model orders with training diagnostics and validation. Compare models on the same target timestamps.
- **SAU dissertation.** Treat flood warning as a separate research question. The Sugeno-style neuro-fuzzy classifier is a feasibility demonstration. Dissertation claims need a defensible event-level evaluation, including missed floods, false alarms, and warning lead time, with enough distinct held-out floods.

Later experiments, after the evaluation design is settled:

```powershell
python -m src.audit
python -m src.classical arima --p 2 --d 1 --q 1
python -m src.neural rnn --trials 2
python -m src.neural lstm --trials 2
python -m src.quantile
python -m src.decision
python -m src.sau
python -m src.compare
```

The model settings above are examples. The present split cannot support operational flood-warning claims.
