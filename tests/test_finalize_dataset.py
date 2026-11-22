import unittest

import numpy as np
import pandas as pd

from src.finalize_dataset import DuplicateTimestamps, merge_observations
from src.prepare import canonical_hourly_stage
from src.utils import CANONICAL_CUTOFFS, complete_window_targets, forecast_splits


def readings(times, stages):
    index = pd.DatetimeIndex(times, tz='UTC', name='time_utc')
    return pd.DataFrame({'stage_ft': stages}, index=index)


class FinalizeDatasetTests(unittest.TestCase):
    def test_merge_preserves_original_observations_and_missing_raw_slots(self):
        historical = readings(['2014-12-31 23:30'], [2.5])
        modern = readings(['2015-01-01 00:00', '2015-01-01 00:15'], [2.7, 2.8])
        merged = merge_observations(historical, modern)
        pd.testing.assert_frame_equal(merged, pd.concat([historical, modern]))
        self.assertNotIn(pd.Timestamp('2014-12-31 23:45', tz='UTC'), merged.index)

    def test_merge_reports_identical_and_conflicting_duplicates(self):
        times = ['2015-01-01 00:00', '2015-01-01 00:15']
        historical = readings(times, [2.5, 3.1])
        modern = readings(times, [2.5, 3.2])
        with self.assertRaises(DuplicateTimestamps) as caught:
            merge_observations(historical, modern)
        self.assertEqual(caught.exception.rows.duplicate_type.tolist(),
                         ['identical', 'identical', 'conflicting', 'conflicting'])
        self.assertEqual(set(caught.exception.rows.source_snapshot), {'2007-2014', '2015-2025'})

    def test_hourly_cutoffs_do_not_backfill_or_use_future_readings(self):
        raw = readings(['2007-10-01 05:00', '2007-10-01 05:45',
                        '2007-10-01 07:15', '2025-12-31 23:45'], [1.0, 2.0, 3.0, 4.0])
        hourly = canonical_hourly_stage(raw)
        start, validation, test, stop = CANONICAL_CUTOFFS
        self.assertTrue(hourly.stage_ft.iloc[:5].isna().all())
        self.assertEqual(hourly.stage_ft.iloc[5:7].tolist(), [1, 2])
        self.assertTrue(pd.isna(hourly.stage_ft.iloc[7]))
        self.assertEqual(hourly.stage_ft.iloc[8], 3)
        self.assertEqual(hourly.observed_time_utc.at[stop], raw.index[-1])
        split = forecast_splits(hourly.index)
        self.assertEqual([hourly.index[split[key][0]] for key in ('train', 'val', 'test')],
                         [start, validation, test])
        self.assertEqual(hourly.index[split['test'][1]], stop)
        with self.assertRaisesRegex(ValueError, 'complete hourly cutoff grid'):
            forecast_splits(hourly.index.delete(6))

    def test_gap_targets_and_windows_are_ineligible(self):
        stage = np.array([1, 2, 3, np.nan, 5, 6, 7, 8, np.inf, 10])
        self.assertEqual(np.flatnonzero(complete_window_targets(stage, 2)).tolist(), [2, 6, 7])
        self.assertFalse(complete_window_targets(stage, len(stage)).any())
        from src.neural import windows
        features, targets, positions = windows(stage, 6, len(stage), 2)
        self.assertEqual(positions.tolist(), [6, 7])
        self.assertEqual(features[:, :, 0].tolist(), [[5, 6], [6, 7]])
        self.assertEqual(targets.tolist(), [7, 8])
