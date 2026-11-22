import unittest

import numpy as np
import pandas as pd

from src.experiment import slice_predictions
from src.experiment_evaluate import warning_metrics
from src.experiment_neural import dataset, interval_metrics


class ExperimentTests(unittest.TestCase):
    def test_baseline_uses_prior_observation_and_drops_incomplete_windows(self):
        time = pd.date_range('2023-01-01', periods=8, freq='h', tz='UTC')
        stage = pd.Series([1, 2, np.nan, 4, 5, 6, 7, 8], index=time)
        bounds = {'val': (3, 7)}
        result = slice_predictions(stage, bounds, 'val', 2)
        self.assertTrue(result.iloc[:2].isna().all())
        self.assertEqual(result.iloc[2:].tolist(), [4, 5])
        stage.iloc[6:] = 1000
        self.assertEqual(result.at[time[6]], slice_predictions(stage, bounds, 'val', 2).at[time[6]])

    def test_validation_windows_do_not_include_target_or_future_holdout(self):
        time = pd.date_range('2023-01-01', periods=8, freq='h', tz='UTC')
        stage = pd.Series(range(8), index=time, dtype=float)
        bounds = {'train': (0, 4), 'val': (4, 6), 'test': (6, 8)}
        stage.iloc[6:] = 999
        inputs, targets, idx = dataset(stage, bounds, 'val', {'lookback': 2}, {'mean': 0, 'scale': 1})
        self.assertEqual(inputs[:,:,0].tolist(), [[2, 3], [3, 4]])
        self.assertEqual(targets.tolist(), [4, 5])
        self.assertEqual(idx.tolist(), time[4:6].tolist())

    def test_event_warning_timing_uses_issuance_hour_and_reports_false_episode(self):
        time = pd.date_range('2024-01-01', periods=50, freq='h', tz='UTC')
        actual = pd.Series(2., index=time)
        predicted = actual.copy()
        actual.iloc[4:6] = 7
        predicted.iloc[3:5] = 7
        predicted.iloc[40] = 7
        scores, event, false = warning_metrics(actual, predicted, 'fixture')
        self.assertEqual((scores['TP'], scores['FP'], scores['FN']), (1, 2, 1))
        self.assertEqual((scores['detected'], scores['missed'], len(false)), (1, 0, 1))
        self.assertEqual(event[0]['lead_hours_positive_is_early'], 2)
        self.assertEqual(event[0]['first_hit_target_delay_hours'], 0)

    def test_interval_scoring_retains_crossing_count(self):
        actual = pd.Series([7., 8.])
        score = interval_metrics(actual, pd.Series([8., 7.]), pd.Series([6., 9.]),
                                 pd.Series([6., 7.]))
        self.assertEqual((score['crossings'], score['PICP'], score['MPIW_ft']), (1, 1., 2.))
