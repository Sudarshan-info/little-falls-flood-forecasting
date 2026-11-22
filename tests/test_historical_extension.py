import unittest

import pandas as pd

from src.historical_extension import classify_events


class HistoricalEventTests(unittest.TestCase):
    def test_missing_hour_never_verifies_a_24_hour_separation(self):
        grid = pd.date_range('2011-01-01', periods=100, freq='h', tz='UTC')
        stage = pd.Series(4.0, index=grid)
        stage.iloc[[30, 55]] = 7.0
        self.assertEqual(classify_events(stage).status.tolist(), ['confirmed', 'confirmed'])
        stage.iloc[40] = float('nan')
        self.assertEqual(classify_events(stage).status.tolist(), ['uncertain', 'uncertain'])

    def test_short_observed_dip_merges_but_missing_inside_is_uncertain(self):
        grid = pd.date_range('2011-01-01', periods=100, freq='h', tz='UTC')
        stage = pd.Series(4.0, index=grid)
        stage.iloc[[30, 54]] = 7.0
        event = classify_events(stage).iloc[0]
        self.assertEqual((event.status, event.duration_hours_inclusive), ('confirmed', 25))
        stage.iloc[40] = float('nan')
        self.assertEqual(classify_events(stage).status.tolist(), ['uncertain'])
