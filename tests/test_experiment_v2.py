import unittest

import numpy as np
import pandas as pd
import torch

from src.experiment_v2 import fixed_predictions, make_folds, score, trailing_block
from src.experiment_v2_neural import CONTROL, ResidualNet, windows


class VersionTwoTests(unittest.TestCase):
    def test_folds_end_in_2023_and_training_precedes_outer_year(self):
        idx = pd.date_range('2007-10-01', '2023-12-31 23:00', freq='h', tz='UTC')
        stage = pd.Series(np.arange(len(idx)), index=idx)
        for year, lo, hi, _ in make_folds(stage):
            self.assertEqual(idx[lo-1].year, year-1)
            self.assertEqual(idx[lo].year, year)
            self.assertEqual(idx[hi-1].year, year)
        self.assertEqual(hi,len(idx))

    def test_missing_hours_are_never_stitched(self):
        stage = pd.Series([1,2,np.nan,4,5])
        self.assertEqual(trailing_block(stage,10).tolist(),[4,5])
        idx = pd.date_range('2023-01-01', periods=100, freq='h', tz='UTC')
        stage = pd.Series(1.,index=idx)
        stage.iloc[47]=np.nan
        result,_ = score(stage,stage.copy(),48,100,.1)
        self.assertEqual(result['overall_n'],4)

    def test_neural_features_exclude_target_and_future_rows(self):
        values=np.arange(40,dtype='float32')
        cfg={**CONTROL,'lag':12,'features':True}
        inputs,targets,positions,_=windows(values,20,24,cfg,42)
        self.assertEqual(positions.tolist(),[20,21,22,23])
        self.assertEqual(targets.tolist(),[20,21,22,23])
        self.assertEqual(inputs[0,-1].tolist(),[19,1,3,6])
        values[24:]=999
        changed,*_=windows(values,20,24,cfg,42)
        torch.testing.assert_close(inputs,changed)

    def test_interval_architecture_remains_ordered_with_variable_inputs(self):
        model=ResidualNet('lstm',CONTROL,'ordered')
        bounds=model(torch.randn(8,12,1))
        self.assertTrue(torch.all(bounds[:,1]>bounds[:,0]))

    def test_fixed_classical_forecast_is_causal(self):
        idx = pd.date_range('2023-01-01', periods=75, freq='h', tz='UTC')
        stage = pd.Series(np.arange(75,dtype=float),index=idx)
        params={'ar.L1':.5,'sigma2':1}
        base=fixed_predictions(stage,40,60,(1,1,0),params)
        stage.iloc[50:]+=100
        changed=fixed_predictions(stage,40,60,(1,1,0),params)
        pd.testing.assert_series_equal(base.iloc[:11],changed.iloc[:11])
        self.assertGreater(changed.iloc[11],base.iloc[11])
