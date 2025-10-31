# EUOS25 - Absorption/Fluorescence Classification Pipeline

A minimal, extensible pipeline for the EUOS25 Kaggle competition, focusing on absorption/fluorescence classification using SMILES-based molecular features.

## Features

- **Small-scale first**: Starts with ECFP + LightGBM for rapid prototyping
- **Pluggable architecture**: Easy to swap featurizers, models, and strategies
- **Scaffold-based CV**: Proper molecular scaffold-aware cross-validation
- **Imbalance handling**: Automatic pos_weight calculation and optional sampling
- **Reproducible**: Fixed seeds and consistent data processing
- **Production-ready**: Complete CLI for all pipeline stages

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment with uv
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv sync
```

### 2. Prepare Data

Place your training data in `data/raw/train.csv` with columns:
- `ID`: Molecule identifier
- `SMILES`: SMILES string
- `Fluorescence` or `Transmittance`: Binary label

### 3. Run Pipeline

```bash
# Run complete small-scale pipeline
bash scripts/run_small.sh
```

Or run individual steps:

```bash
# Prepare data
uv run -m euos25.cli prepare \
  --input data/raw/train.csv \
  --output data/processed/train_prepared.csv

# Create scaffold splits
uv run -m euos25.cli make-splits \
  --input data/processed/train_prepared.csv \
  --output data/processed/splits_5.json \
  --folds 5 --seed 42

# Build features
uv run -m euos25.cli build-features \
  --input data/processed/train_prepared.csv \
  --output data/processed/features_small.parquet \
  --config configs/small.yaml

# Train models
uv run -m euos25.cli train \
  --features data/processed/features_small.parquet \
  --splits data/processed/splits_5.json \
  --config configs/small.yaml \
  --outdir data/models \
  --data data/processed/train_prepared.csv

# Generate predictions
uv run -m euos25.cli infer \
  --features data/processed/features_small.parquet \
  --splits data/processed/splits_5.json \
  --config configs/small.yaml \
  --model-dir data/models \
  --outdir data/preds \
  --mode oof

# Create submission
uv run -m euos25.cli submit \
  --pred data/preds/y_fluo_any_oof.csv \
  --out data/submissions/submission.csv
```

## Project Structure

```
euos25/
├── configs/                  # Configuration files
│   └── small.yaml           # Small-scale default config
├── data/
│   ├── raw/                 # Raw data
│   ├── processed/           # Processed features and splits
│   ├── models/              # Trained models
│   ├── preds/               # Predictions
│   └── submissions/         # Submission files
├── src/euos25/
│   ├── config.py            # Pydantic configuration
│   ├── cli.py               # Command-line interface
│   ├── utils/               # Utilities (I/O, metrics, etc.)
│   ├── data/                # Data processing
│   ├── featurizers/         # Feature engineering
│   ├── models/              # Model implementations
│   ├── imbalance/           # Imbalance handling
│   └── pipeline/            # Pipeline modules
├── scripts/
│   └── run_small.sh         # Quick start script
└── tests/                   # Unit tests
```

## Configuration

Edit `configs/small.yaml` to customize:

- **Featurizers**: ECFP, RDKit 2D descriptors, conjugation proxies
- **Model**: LightGBM hyperparameters
- **Imbalance**: pos_weight calculation, sampling strategies
- **CV**: Number of folds, scaffold settings

## Extending the Pipeline

### Add a New Featurizer

```python
# src/euos25/featurizers/my_featurizer.py
from euos25.featurizers.base import BaseFeaturizer

class MyFeaturizer(BaseFeaturizer):
    def transform(self, df, smiles_col="SMILES"):
        # Your feature generation logic
        return feature_df
```

Add to `configs/small.yaml`:
```yaml
featurizers:
  - name: my_featurizer
    params:
      param1: value1
```

### Add a New Model

```python
# src/euos25/models/my_model.py
from euos25.models.base import BaseClfModel

class MyModel(BaseClfModel):
    def fit(self, X, y, **kwargs):
        # Training logic
        pass

    def predict_proba(self, X):
        # Prediction logic
        pass
```

Update `pipeline/train.py` to register your model.

## Development

### Code Quality

```bash
# Format code
uv run black src/
uv run isort src/

# Lint
uv run ruff check src/

# Type checking
uv run mypy src/
```

### Testing

```bash
# Run tests
uv run pytest tests/

# With coverage
uv run pytest --cov=euos25 tests/
```

## Default Configuration

The small-scale configuration uses:

- **Features**: ECFP6 (2048 bits, counts) + RDKit 2D + Conjugation proxies
- **Model**: LightGBM (500 trees, lr=0.03, leaves=127)
- **CV**: 5-fold scaffold-based
- **Imbalance**: Automatic pos_weight = (N-P)/P
- **Metrics**: ROC AUC, PR AUC

## Tips

1. **Start small**: Use the default config for initial experiments
2. **Monitor CV**: Check fold-wise metrics in `data/models/{task}/{model}/cv_metrics.csv`
3. **Feature importance**: Models save feature importance for analysis
4. **Reproducibility**: All steps use fixed seed (default: 42)
5. **Plate effects**: Enable plate normalization if needed via config

## References

- [DEV.md](DEV.md) - Detailed development guide
- [TASK.md](TASK.md) - Development task list

## License

This project is for educational and competition purposes.
