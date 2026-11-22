import unittest
from unittest.mock import patch

import pandas as pd

from src.qc import flood_event_ids
from src.utils import CHECKPOINT6_CUTOFFS, forecast_splits


class Checkpoint6Tests(unittest.TestCase):
    def test_fixed_target_cutoffs_exclude_terminal_partial_bucket(self):
        start, validation, test, stop = CHECKPOINT6_CUTOFFS
        grid = pd.date_range(start, stop, freq='h')
        splits = forecast_splits(grid)
        self.assertEqual([grid[splits[name][0]] for name in ('train', 'val', 'test')],
                         [start, validation, test])
        self.assertEqual(grid[splits['test'][1]], stop)
        self.assertEqual(splits['train'][1], splits['val'][0])
        self.assertEqual(splits['val'][1], splits['test'][0])
        with self.assertRaisesRegex(ValueError, 'complete hourly cutoff grid'):
            forecast_splits(grid.delete(1))

    def test_23_below_threshold_hours_merge_but_24_separate(self):
        start = pd.Timestamp('2023-01-01', tz='UTC')
        index = pd.DatetimeIndex([start, start + pd.Timedelta(hours=24),
                                  start + pd.Timedelta(hours=49)])
        self.assertEqual(flood_event_ids(index, 24).tolist(), [1, 1, 2])

    def test_neural_selection_preparation_excludes_test(self):
        from src import neural

        start, validation, test, stop = CHECKPOINT6_CUTOFFS
        grid = pd.date_range(start, stop, freq='h')
        stage = pd.Series(2.0, index=grid)
        stage.loc[validation:] = 3.0
        stage.loc[test:] = 100.0
        with patch.object(neural, 'windows') as windows:
            data, scaler = neural.prepare(stage, 24, partitions=('train', 'val'))
        self.assertEqual(set(data), {'train', 'val'})
        self.assertEqual(scaler.mean_[0], 2.0)
        self.assertEqual(windows.call_count, 2)
        for args, _ in windows.call_args_list:
            self.assertEqual(len(args[0]), grid.get_loc(test))
