from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from src import config, demo_synthetic, fetch, prepare


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = self.enterContext(TemporaryDirectory())
        folder = Path(temporary)
        self.enterContext(patch.multiple(config, RAW=folder, PROCESSED=folder, METRICS=folder))
        self.request = self.enterContext(patch.object(
            fetch.requests, 'get', side_effect=AssertionError('Unexpected network request')))
        self.folder = folder

    def run_fetch(self, start, end, *options):
        with patch('sys.argv', ['fetch', '--start', start, '--end', end, *options]):
            fetch.main()

    def test_continuous_schema_pagination_and_end_boundary(self):
        def feature(time, site=config.SITE):
            return {'properties': {
                'monitoring_location_id': site, 'parameter_code': '00065',
                'time': time, 'value': '7.25', 'unit_of_measure': 'ft',
                'time_series_id': 'series-1', 'approval_status': 'Approved', 'qualifier': None,
            }}

        first = Mock()
        first.json.return_value = {
            'features': [feature('2024-01-31T04:00:00Z', 'other-site')],
            'links': [{'rel': 'next', 'href': '?offset=1'}],
        }
        second = Mock()
        second.json.return_value = {'features': [
            feature('2024-01-31T23:59:59.500Z'), feature('2024-02-01T00:00:00Z'),
        ]}
        self.request.side_effect = [first, second]
        rows = fetch.fetch_month('2024-01-31', '2024-01-31')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.iloc[0]['time_series_id'], 'series-1')
        self.assertEqual(rows.iloc[0]['approval_status'], 'Approved')
        self.assertEqual(rows.iloc[0]['unit'], 'ft')
        self.assertEqual(rows.iloc[0]['stage_ft'], '7.25')
        args, kwargs = self.request.call_args_list[0]
        self.assertEqual(kwargs['params']['datetime'], '2024-01-31T00:00:00Z/2024-02-01T00:00:00Z')
        self.assertEqual(self.request.call_args.args[0], args[0] + '?offset=1')

    def test_monthly_cache_is_scoped_to_exact_slice(self):
        def fake_month(begin, end):
            return pd.DataFrame([[begin, 8, 'ft', 'Approved', None, '001']], columns=list(fetch.FIELDS))

        target = self.folder / 'measurements.csv'
        (self.folder / 'usgs_2023-12.csv').write_text('unrelated legacy cache')
        with patch.object(fetch, 'fetch_month', side_effect=fake_month) as download:
            self.run_fetch('2024-01-31', '2024-02-01')
            self.assertEqual(download.call_count, 2)
            self.run_fetch('2024-01-31', '2024-02-01', '--overwrite')
            self.assertEqual(download.call_count, 2)
            self.run_fetch('2024-01-01', '2024-01-31', '--overwrite')
            self.assertEqual(download.call_count, 3)
        output = pd.read_csv(target, dtype={'time_series_id': str})
        self.assertEqual(len(output), 1)
        self.assertTrue(output.iloc[0]['time_utc'].startswith('2024-01-01'))
        self.assertEqual(output.iloc[0]['time_series_id'], '001')

    def test_invalid_ranges_and_existing_output_stop_before_network(self):
        target = self.folder / 'measurements.csv'
        target.write_text('real observations')
        for start, end, options in [
            ('2024-02-01', '2024-01-31', ['--overwrite']),
            ('2099-01-01', '2099-01-01', ['--overwrite']),
            ('2024-01-01', '2024-01-01', []),
        ]:
            with self.subTest(start=start, end=end), self.assertRaises(SystemExit):
                self.run_fetch(start, end, *options)
        self.request.assert_not_called()
        self.assertEqual(target.read_text(), 'real observations')

    def test_hourly_alignment_and_gaps(self):
        (self.folder / 'measurements.csv').write_text(
            'time_utc,stage_ft,unit\n'
            '2024-01-01T10:00:00Z,7,ft\n'
            '2024-01-01T10:45:00.500Z,8,ft\n'
            '2024-01-01T12:15:00+00:00,9,ft\n'
            '2024-01-01T13:00:00Z,10,ft\n'
        )
        with patch('sys.argv', ['prepare']):
            prepare.main()
        rows = pd.read_csv(self.folder / 'hourly_stage.csv')
        self.assertEqual(pd.to_datetime(rows.time_utc, utc=True).dt.hour.tolist(), [10, 11, 12, 13])
        self.assertEqual(rows.n_source_in_hour.tolist(), [1, 1, 0, 2])
        self.assertEqual(rows.stage_ft.dropna().tolist(), [7, 8, 10])
        self.assertEqual(pd.to_datetime(rows.observed_time_utc, format='ISO8601', utc=True).iloc[1],
                         pd.Timestamp('2024-01-01T10:45:00.500Z'))

    def test_demo_preserves_real_raw_and_processed_files(self):
        real_files = [self.folder / name for name in
                      ('measurements.csv', 'hourly_stage.csv', 'data_quality.json')]
        for path in real_files:
            path.write_text('real data')
        demo_synthetic.main()
        with patch('sys.argv', ['prepare', '--synthetic']):
            prepare.main()
        for path in real_files:
            self.assertEqual(path.read_text(), 'real data')
            self.assertTrue(path.with_name('synthetic_' + path.name).is_file())

    def test_prepare_rejects_legacy_synthetic_raw_file(self):
        (self.folder / 'measurements.csv').write_text(
            'time_utc,stage_ft,unit,approval_status\n2024-01-01T00:00:00Z,7,ft,synthetic\n')
        with patch('sys.argv', ['prepare']), self.assertRaisesRegex(SystemExit, 'Synthetic rows'):
            prepare.main()
        self.assertFalse((self.folder / 'hourly_stage.csv').exists())
