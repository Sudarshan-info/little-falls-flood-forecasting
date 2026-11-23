# V2 historical validation

V2 compares model settings on five historical validation folds. Frozen V1 supplies the 2024–2025 final test. V2 uses observations through 2023 and has no fresh independent test.

## Data and assessment

Use the two canonical CSVs from the [dataset release](https://github.com/Sudarshan-info/little-falls-flood-forecasting/releases/tag/dataset-v1.0.0). The loader checks both SHA-256 hashes and reads the hourly grid from 2007-10-01 00:00 UTC to 2023-12-31 23:00 UTC. Each fold starts training from that same October 2007 cutoff and ends at 23:00 UTC on December 31 before the validation year. Validation runs from January 1 at 00:00 to December 31 at 23:00 UTC in 2014, 2018, 2021, 2022 and 2023. See the [exact fold boundaries and rise thresholds](folds.csv).

Each hourly cutoff uses the last observation in the preceding hour, including the cutoff itself. Missing hours stay on the grid and are left unfilled. All point models score one hour ahead on 43,775 common targets with a finite target and 48 finite preceding hours. Parameters and scalers exclude the outer year. Classical fits use the latest contiguous training block and update their filter with observed history while keeping parameters fixed. Neural models reserve the last 90 days before validation for early stopping. Training and scaling use the earlier portion.

Ranking uses pooled hourly RMSE. Eligibility requires five successful folds and RMSE on observed stages below 6.5 ft at most 105% of persistence. Rapid rise uses the training-only 99th percentile of hourly stage change in each fold. Other subsets are within 0.5 ft of 7.0 ft and at or above 7.0 ft.

## Search space

- SARIMAX fits use a constant for undifferenced models and no drift otherwise, with a 24-iteration optimizer cap. The AR orders were 1, 2, 3, 4, 6, 8, 12 and 24. Additional orders were (0,1,0), (1,1,0), (2,1,0), (4,1,0), (0,1,2), (1,1,1), (2,1,1), (3,1,2) and (2,0,1). The default cap was 8,000 contiguous hours. AR(4), AR(8), ARIMA(1,1,1), ARIMA(2,1,1) and ARIMA(2,1,0) also used a 32,000-hour cap. Earlier gaps sometimes made both caps equivalent. Nonconverged fits are ineligible.
- RNN and LSTM compared 16 and 36 epochs on 24-hour inputs, 16 units, two layers, batch 384, learning rate 0.002, Adam without weight decay and up to 18,000 evenly sampled complete windows. The larger RNN used 48 hours, 32 units, one layer, batch 512, learning rate 0.001, weight decay 0.0001 and 26 epochs. On 2021 and 2023 the LSTM pilots used a 10-epoch cap and compared a 6,000-window control against a larger architecture, 9,000-window coverage, rise-weighted sampling and past-change features. Exact configurations are in the source and fold tables.
- Main trials use seed 42. Neural gradients are clipped at 1 and the best inner-loss weights are restored after 10 epochs without improvement exceeding 0.00000001. Seed pilots use 42, 43 and 44 in 2023 with 6,000 windows and 10 epochs. The separate 2018, 2021 and 2023 interval pilot uses the same reduced budget and seed 42.

| Finalist | Setting | Pooled RMSE (ft) | Pooled MAE (ft) |
|---|---|---:|---:|
| Persistence | Latest stage | 0.03436 | 0.01393 |
| AR(3) | 8,000-hour cap | 0.02747 | 0.01218 |
| ARIMA(2,0,1) | 8,000-hour cap | 0.02746 | 0.01217 |
| RNN | 36-epoch control | 0.02758 | 0.01215 |
| LSTM | 36-epoch control | 0.02788 | 0.01201 |

## Saved evidence

| Evidence | Files |
|---|---|
| Protocol, software and status counts | [Selection rules](protocol.json), [alignment](alignment.json), [runtime versions and hashes](reproducibility.json), [trial totals](runtime.csv) |
| Candidate summaries and per-fold results | Baseline [summary](baseline_summary.csv) and [folds](baseline_folds.csv), classical [summary](classical_summary.csv) and [folds](classical_folds.csv), RNN [summary](rnn_summary.csv) and [folds](rnn_folds.csv), LSTM [summary](lstm_summary.csv) and [folds](lstm_folds.csv) |
| Selected settings | [Classical](classical_selected.json), [neural](neural_selected.json) |
| Matched controls and seeds | [Fitting history](classical_history_comparison.csv), [neural controls](paired_neural_controls.csv), [RNN seeds](rnn_seeds_summary.csv), [LSTM seeds](lstm_seeds_summary.csv), [saved 2023 V1 comparison](matched_2023_v1_comparison.csv) |
| Rapid-rise and flood evidence | [Finalist subsets](validation_finalists.csv), [hourly onset and overlap](finalists_flood_events.csv), interval [summary](interval_summary.csv) and [folds](interval_folds.csv), [validation checks](validation_checks.json) |

On matched folds, the 36-epoch controls lowered overall RMSE by 0.00054 ft for RNN and 0.00042 ft for LSTM. Longer classical fitting history improved pooled RMSE for zero of five paired orders. On 398 rapid-rise targets, finalist RMSEs were 0.29707 ft for persistence, 0.21855 for AR, 0.21851 for ARIMA, 0.22683 for RNN and 0.22200 for LSTM. All finalists overlapped eight out of eight observed events. Persistence, AR, ARIMA, RNN and LSTM hit 0, 2, 2, 0 and 3 of the eight hourly onset targets respectively. V2 records timing between hourly targets. The course report separately gives V1 leads relative to raw crossings. The V2 interval pilot covered 13/261 rapid rises with free bounds and 3/261 with ordered bounds.

## Reproduction and limits

Use a clean checkout. Place the release CSVs in data/raw/measurements.csv and data/processed/hourly_stage.csv following the [root README](../../README.md). Install [project dependencies](../../requirements.txt), using the recorded Windows Python 3.11.9 and library versions in [reproducibility.json](reproducibility.json) where available. The original runs used CPU Torch with two threads. The exact package index, transitive versions, CPU and BLAS build were not captured.

    python -B -m unittest discover -s tests -p test_experiment_v2.py
    python -B -m src.experiment_v2 --phase all

The second command fits models and writes new artifacts to outputs/experiments/stage_v2_optimization/. See the [driver](../../src/experiment_v2.py), [neural implementation](../../src/experiment_v2_neural.py) and [offline tests](../../tests/test_experiment_v2.py). Run with a fresh output directory for a clean reproduction. The original trial caches, complete traces, checkpoints and private local logs are excluded. AR(24) and larger RNN pruning depend on elapsed time, and numerical libraries can affect optimization and convergence. Exact rerun equivalence has not been verified. The saved tables are the record of the original run.

The folds guided model selection, only eight historical flood events were observed, the seed sensitivity used a reduced budget, and live warning delay was not measured. The interval and alternative LSTM comparisons were bounded pilots.
