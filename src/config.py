from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
FIGURES = ROOT / 'outputs' / 'figures'
METRICS = ROOT / 'outputs' / 'metrics'
for p in (RAW, PROCESSED, FIGURES, METRICS):
    p.mkdir(parents=True, exist_ok=True)
SITE = 'USGS-01389500'
PARAMETER = '00065'  # USGS gage height, ft
ACTION_STAGE = 6.5
MINOR_STAGE = 7.0
MODERATE_STAGE = 8.0
MAJOR_STAGE = 9.0
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
SEED = 42
LOOKBACK = 24
