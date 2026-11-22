# Little Falls one-hour river-stage forecasts V1

**Frozen experiment:** `stage_v1_20261122`

This study compares one-hour-ahead stage forecasts for USGS gage **01389500**, Passaic River at Little Falls, New Jersey, using parameter **00065** (gage height in feet). On all **17,544** hourly test targets, AR(4), ARIMA(1,1,1), LSTM, and RNN had lower RMSE and MAE than persistence. AR(4) and ARIMA were practically tied. The best flood-warning models recognized all three test episodes and issued their first matched warnings **15, 60, and 0 minutes** before the respective raw USGS crossings. Three events provide limited evidence about performance on future floods.

## Data and forecast design

Dataset `dataset-v1.0.0` covers October 2007 through December 2025 UTC. There are 160,008 study cutoffs, including 313 missing hours. Training covers October 2007–2022, validation covers 2023, and test covers 2024–2025. The final partial cutoff at 2026-01-01 00:00 UTC is outside the study. At cutoff *t*, the hourly series uses the last available archived reading in *(t − 1 hour, t]*. The forecast targets the next cutoff, *t + 1 hour*. Explicit windows with missing observations are excluded. The 184-hour and 42-hour outages remain unfilled. During rolling evaluation, observed test stages become available as inputs at their cutoffs. Model parameters and selection remain fixed.

## Training exposure and model selection

| Model | Frozen specification | Training and selection |
|---|---|---|
| Persistence | Latest observed stage | Fixed one-hour lag |
| Daily naive | Stage at the target time minus 24 hours | Fixed 24-hour lag |
| MA(2) | (0,0,2), intercept, two moving-average error terms | Lowest validation RMSE among converged MA candidates |
| AR(4) | (4,0,0), intercept | Lowest validation RMSE among converged AR candidates |
| ARIMA(1,1,1) | (1,1,1), no deterministic trend | Lowest validation RMSE among converged ARIMA candidates |
| RNN | 24-hour lookback, 16 hidden units, two layers | Seed 42, epoch 13 |
| LSTM | 24-hour lookback, 16 hidden units, two layers | Seed 42, epoch 16 |
| Quantile LSTMs | Same architecture, separate 0.025 and 0.975 pinball losses | Seed 42, epochs 4 and 13 by validation pinball loss |

Classical maximum-likelihood estimation used **8,000 contiguous training hours**, from **2022-02-01 16:00** through **2022-12-31 23:00 UTC**. It captured **one independent flood, 57 flood hours, and a 7.43-ft maximum**. The cap limited fit time and preserved contiguous lags. The optimizer limit was 24 iterations and AR(1) did not converge. The recurrent fits used **18,000 evenly sampled complete windows** per tuned trial. In the selected 24-hour configuration, **200 sampled target hours** reached 7 ft, with examples from **all 14 training floods**. Those windows overlap. The scaler was fitted on the **133,391** valid training observations. All selected neural configurations used Adam, batch size **384**, learning rate **0.002**, a **16-epoch cap**, and patience **7**. The validation-only LSTM repeats at seeds 43 and 44 did not change the frozen seed-42 checkpoint.

The different fitting exposures limit comparisons of the model families themselves. The common target hours support direct comparison of the fitted forecasts. Training diagnostics showed strong dependence in stage levels, and first differences supported testing integrated models. The differenced series did not meet the daily seasonality screen, so SARIMA and daily decomposition were omitted. The daily naive forecast remained as a comparator. See the [diagnostics](stationarity.json), [search tables](classical_search.csv), [seed sensitivity](lstm_seed_sensitivity.csv), and [V1 manifest](manifest.json).

## Common-target forecast accuracy

Errors are in feet. Validation contains 8,760 hours. Final test contains 17,544 hours.

| Model | Validation RMSE | Validation MAE | Test RMSE | Test MAE |
|---|---:|---:|---:|---:|
| Persistence | 0.03341 | 0.01460 | 0.02997 | 0.01244 |
| Daily naive | 0.45411 | 0.26547 | 0.37061 | 0.21357 |
| MA(2) | 0.45985 | 0.31817 | 0.43602 | 0.33296 |
| AR(4) | 0.02464 | 0.01175 | 0.02313 | 0.01043 |
| ARIMA(1,1,1) | 0.02461 | 0.01174 | 0.02315 | 0.01041 |
| RNN | 0.02685 | 0.01332 | 0.02476 | 0.01143 |
| LSTM | 0.02505 | 0.01155 | 0.02342 | 0.01017 |

AR(4) had **22.8% lower RMSE** than persistence on test, and LSTM had the lowest test MAE. ARIMA led AR on validation RMSE and test MAE. AR led on test RMSE by **0.000018 ft**. The forecasts differed by a mean absolute **0.00068 ft** and gave identical threshold decisions. The fitted AR coefficients sum to about **0.99958**, consistent with both models making small adjustments to a highly persistent stage series. Persistence itself achieved test R² of **0.99966**. Hourly observations and errors are serially dependent. The tiny AR/ARIMA gap does not establish a meaningful winner. Full-precision [validation](validation_common_metrics.csv) and [test](test_common_metrics.csv) results include R² and MAPE. The saved near-zero MAPE exclusion removed zero common observations in either split.

## Flood-warning performance and raw lead time

The flood target is **7.0 ft or higher**. The **6.5-ft action stage** is descriptive. A warning episode detects a flood when at least one forecast-positive target overlaps an observed flood-positive target. Independent episodes follow the 24-hour separation rule. Raw-stage lead measures the interval from the first matched forecast issuance to the first raw USGS reading at or above 7.0 ft.

| 2024 episode, UTC | First matched issue, AR, ARIMA, RNN and LSTM | First raw crossing | Hourly crossing | Raw-stage lead |
|---|---|---|---|---:|
| January 10 | 18:00 | 18:15 | 19:00 | **15 min** |
| March 10 | 21:00 | 22:00 | 22:00 | **60 min** |
| April 5 | 01:00 | 01:00 | 01:00 | **0 min** |

The January difference arises because the hourly row uses the reading at the end of its bucket. The original experiment timing table and hydrograph markers refer to hourly onsets. The [raw-crossing lead table](raw_crossing_leads.csv) gives both clocks for every model. USGS publication latency was unavailable, so these leads describe archived observation times.

AR, ARIMA, and LSTM each detected all three floods with **281 true-positive hours, one false-negative hour, and zero false-positive hours**. RNN had the same hits and miss with three false-positive hours. Persistence also detected all three, first issued at or after the raw crossing in each case, and missed all three onset targets. The April onset target was missed by all four leading models. Daily naive produced one separate false-alarm episode, and MA detected only one flood. See the [hourly classification table](test_flood_hourly.csv) and [flood hydrographs](test_flood_hydrographs.png).

## Rising limbs and quantile intervals

On the **135 test hours** meeting the training-derived rapid-rise threshold of approximately **0.12 ft/h**, AR and ARIMA underpredicted stage about **92%** of the time, RNN and LSTM about **96%**, and persistence **100%**. AR and ARIMA were low at all three observed crest hours, while their forecast crests arrived one hour later and slightly exceeded the observed peaks. Strong whole-period error scores can therefore coexist with slow response during a rising limb. [Rapid-rise results](test_rapid_rises.csv)

The nominal 95% intervals achieved **95.10%** validation coverage at **0.0613 ft** mean width and **95.83%** test coverage at **0.0580 ft** mean width. There were **10 raw test quantile crossings (0.057%)**, all below flood stage. The bounds were ordered for plotting and scoring, which covered two of the inverted cases. No calibration was fitted. Coverage was **95.88%** at a mean width of **0.1813 ft** on the 97 rising flood-stage test hours. Those hours come from only three events. [Test interval metrics](test_quantile_metrics.json) · [Interval figure](test_quantile_interval.png)

## Reproducibility and limits

The [manifest](manifest.json) records data and checkpoint hashes, source fingerprints, saved model settings, runtime versions, and commands. Its checkpoint and detailed forecast paths refer to the preserved local experiment. The [publication list](publication_files.txt) identifies the small evidence files and code proposed for GitHub.

This checkpoint supports the conclusion that four fitted models improved hourly forecasts over persistence on the fixed holdout. The brief archival warning leads, three-event sample, unequal training exposure, and unmeasured publication latency limit any operational or model-family claims.
