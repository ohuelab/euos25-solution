# DEV.md — EUOS25 Minimal, Extensible Pipeline (uv-based)

本リポジトリは **EUOS25（Absorption/Fluorescence 4サブタスク）向けの小スケール最小実装** を提供しつつ、**モデル・特徴量・前処理を差し替え可能**な構造を意識しています。依存管理は **uv** を前提とし、巨大化を避けるために以下の方針で設計します。

- **Small-first**: まずは `ECFP + LightGBM` と **最小限の GNN（任意）** のみ。
- **Pluggable**: 特徴量・モデル・損失・サンプラは **プロトコル/ABC** で疎結合化。
- **Simple Config**: TOML/YAML（任意） + 1箇所の Pydantic 設定。Hydra等は不採用。
- **Repro-ready**: 再現性固定（seed）と提出物生成の一貫 CLI。

---

## 1. 環境（uv）

```bash
# Python 3.11+ 推奨
uv venv .venv
source .venv/bin/activate

uv sync
```

---

## 2. リポジトリ構成（小スケール）

```
euos25/
  README.md
  DEV.md
  pyproject.toml
  data/
    raw/                      # 提供CSV等
    processed/                # 特徴量/分割
    models/                   # 学習済み
    submissions/              # 提出CSV
  src/
    euos25/
      __init__.py
      config.py               # Pydantic 設定
      cli.py                  # 入口: uv run -m euos25.cli ...
      utils/
        io.py, seed.py, metrics.py, scaffold.py, plates.py, calibrate.py
      data/
        schema.py             # DataFrame列定義/検証
        splits.py             # Scaffold K-fold 作成
      featurizers/
        base.py               # Featurizer プロトコル/ABC
        ecfp.py               # ECFP6(count/bit)
        rdkit2d.py            # RDKit 2D 基本
        conj_proxy.py         # 共役長/D-A簡易指標
      models/
        base.py               # Model プロトコル（fit/predict_proba/save/load）
        lgbm.py               # LightGBM 分類器
        gnn.py                # 任意: GIN/MPNN の最小実装
      imbalance/
        samplers.py           # WeightedSampler / ダウンサンプル
        losses.py             # BCE/Focal（GNN用のみ）
      pipeline/
        prepare.py            # 読込→重複統合→簡易クレンジング
        features.py           # 特徴量計算と保存
        train.py              # 学習（fold ループ）
        infer.py              # 推論（fold ループ）
        ensemble.py           # rank-avg
        submit.py             # 提出CSV生成
  scripts/
    run_small.sh              # 小スケ一括（下記 Quickstart）
  tests/
    test_featurizers.py
    test_splits.py
    test_metrics.py
```

---

## 3. データ契約（schema）

**入力（train.csv 例）**
- `ID: int`
- `SMILES: str`
- `Transmittance/Fluorescence: 0/1`

**出力**
- `data/processed/features_{name}.parquet` — 行は `mol_id` index。列は featurizer 毎にプレフィクス付与。
- `data/processed/splits_{k}.json` — fold→`train_ids` / `valid_ids`。
- `data/models/{task}/{model_name}/fold{n}/...` — モデル/正規化器/設定の保存。
- `data/submissions/{task}_{timestamp}.csv` — 提出形式（`mol_id,prob`）。

---

## 4. 設定（Pydantic + YAML）

```yaml
# configs/small.yaml
seed: 42
folds: 5
scaffold_min_size: 10
featurizers:
  - name: ecfp
    params: {radius: 3, n_bits: 2048, use_counts: true}
  - name: rdkit2d
    params: {}
  - name: conj_proxy
    params: {L_cut: 4}
model:
  name: lgbm
  params:
    n_estimators: 500
    learning_rate: 0.03
    max_depth: -1
    num_leaves: 127
    subsample: 0.8
    colsample_bytree: 0.8
  imbalance:
    use_pos_weight: true
    pos_weight_from_data: true  # (N-P)/P を自動算出
plates:
  normalize: false  # 小スケでは OFF が既定
task: y_fluo_any    # 任意: y_abs_340 | y_abs_450_679 | y_fluo_340_450 | y_fluo_any
metrics: [roc_auc, pr_auc]
```

Pydantic でロードし、`cli` から渡す。将来 `model.name` を `gnn` に変えるだけで GNN 実装に差し替え可。

---

## 5. CLI（uv run）

```bash
# 分割（Scaffold K-fold）
uv run -m euos25.cli make-splits \
  --input data/raw/train.csv --output data/processed/splits_5.json \
  --folds 5 --seed 42

# 特徴量作成（小スケ）
uv run -m euos25.cli build-features \
  --input data/raw/train.csv --output data/processed/features_small.parquet \
  --config configs/small.yaml

# 学習
uv run -m euos25.cli train \
  --features data/processed/features_small.parquet \
  --splits data/processed/splits_5.json \
  --config configs/small.yaml \
  --outdir data/models

# 推論（CV-out）
uv run -m euos25.cli infer \
  --features data/processed/features_small.parquet \
  --splits data/processed/splits_5.json \
  --config configs/small.yaml \
  --outdir data/preds

# アンサンブル（rank-avg）
uv run -m euos25.cli ensemble --pred-dir data/preds --out data/preds/blended.csv

# 提出作成
uv run -m euos25.cli submit --pred data/preds/blended.csv --out data/submissions/y_fluo_any.csv
```

---

## 6. 主要コンポーネントの設計

### 6.1 Featurizer プロトコル
```python
# src/euos25/featurizers/base.py
from typing import Protocol, Dict, Any
import pandas as pd

class Featurizer(Protocol):
    name: str
    def fit(self, df: pd.DataFrame) -> "Featurizer": ...  # 多くは No-op
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...  # index=mol_id, 列にプレフィクス
    def get_params(self) -> Dict[str, Any]: ...
```
- 実装例: `ecfp.py`, `rdkit2d.py`, `conj_proxy.py`。
- `features.py` は `List[Featurizer]` を回して列結合。

### 6.2 Model プロトコル
```python
# src/euos25/models/base.py
from typing import Protocol, Dict, Any
import numpy as np
import pandas as pd

class ClfModel(Protocol):
    name: str
    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "ClfModel": ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...  # shape (n,)
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "ClfModel": ...
    def get_params(self) -> Dict[str, Any]: ...
```
- 既定: `lgbm.py`（`pos_weight` 自動算出）。
- 任意: `gnn.py`（小スケ最小：GIN 4層, Focal対応）。

### 6.3 分割 & メトリクス
- `scaffold.py`: RDKit で Murcko scaffold を計算 → stratified K-fold（陽性均衡化）。
- `metrics.py`: `roc_auc`, `pr_auc` と CV 集計、`spearman`（任意）。

### 6.4 パイプライン
- `prepare.py`: SMILES 正規化・InChIKey 1st block で重複統合、行落ちログ。
- `features.py`: Featurizer 群を適用して `parquet` に保存。
- `train.py`: 各 fold で `fit`、モデルと fold スコアを保存。
- `infer.py`: 各 fold の oof と test（任意）確率を出力。
- `ensemble.py`: 予測確率→順位に変換して平均（rank-avg）。
- `submit.py`: 提出 CSV 生成。

---

## 7. 小スケ既定（デフォルト挙動）
- **Feats**: `ECFP(count, 2048, radius=3)` + `RDKit2D` + `ConjProxy(L_cut=4)`
- **Model**: `LightGBM`（`n_estimators=500`, `learning_rate=0.03`, `num_leaves=127`, `pos_weight=(N-P)/P`）
- **CV**: `K=5` Murcko、早期停止は `valid AUC` で 50 patience。
- **Imbalance**: class weight のみ（ダウンサンプルなし）。
- **Plates**: OFF（必要なら `plates.normalize=true` で median/IQR）。
- **Calibration**: 既定 OFF（最終提出前に `calibrate.py` を実行可）。

---

## 8. 拡張ポイント
- **特徴量の追加**: `featurizers/` に新しいクラスを追加し、`configs/*.yaml` の `featurizers` に列挙。
- **モデル入替**: `models/` にクラスを追加。`model.name` を差し替え。
- **不均衡対応**: `samplers.py` で `NegativeDownSampler(ratio=30)` を実装し、`train.py` で `DataFrame.sample` による軽量実装。
- **PU 学習（任意）**: 小スケでは省略。実装する場合は `models/pu.py` を追加し、`predict_proba` をスコア化してメタ特徴に合流。
- **GNN**: `gnn.py` は **最小構成のみ**（依存が重いので任意）。

---

## 9. 再現性・ロギング
- `seed.py` で `random`, `numpy`, `torch(任意)` の種固定。
- ログは `logging` の INFO。各 fold の AUC を CSV に保存（`data/models/{task}/{model}/cv_metrics.csv`）。

---

## 10. コーディング規約
- `ruff`, `black`, `isort` を `pyproject.toml` で最小設定。
- 型注釈必須（mypy 任意）。
- 単体テスト: `pytest -q`。CI は任意。

---

## 11. Quickstart 用スクリプト（小スケ）

`scripts/run_small.sh`
```bash
set -euo pipefail
CONF=configs/small.yaml

uv run -m euos25.cli make-splits \
  --input data/raw/train.csv --output data/processed/splits_5.json \
  --folds 5 --seed 42

uv run -m euos25.cli build-features \
  --input data/raw/train.csv --output data/processed/features_small.parquet \
  --config $CONF

uv run -m euos25.cli train \
  --features data/processed/features_small.parquet \
  --splits data/processed/splits_5.json \
  --config $CONF --outdir data/models

uv run -m euos25.cli infer \
  --features data/processed/features_small.parquet \
  --splits data/processed/splits_5.json \
  --config $CONF --outdir data/preds

uv run -m euos25.cli ensemble \
  --pred-dir data/preds --out data/preds/blended.csv

uv run -m euos25.cli submit \
  --pred data/preds/blended.csv \
  --out data/submissions/$(basename ${CONF%.*})_$(date +%Y%m%d).csv
```

---

## 12. 今後の拡張（任意）
- **plates.normalize=true** の ON/OFF 比較とアンサンブル。
  - 具体的にどんな補正をすべきか？
- **負例ダウンサンプル** オプションと `pos_weight` の共存検証。
- **rank-avg vs. 重み付け平均**（Spearman で選択）。
- **GNN 多タスク化**（4ヘッド、Focal）。
- **Isotonic calibration** の最終適用。

---

### 付録: CLI 引数例（`euos25.cli`）
- `make-splits`: `--input`, `--output`, `--folds`, `--seed`
- `build-features`: `--input`, `--output`, `--config`
- `train`: `--features`, `--splits`, `--config`, `--outdir`
- `infer`: `--features`, `--splits`, `--config`, `--outdir`
- `ensemble`: `--pred-dir`, `--out`
- `submit`: `--pred`, `--out`

この DEV.md に従えば、**小スケの最小構成**から開始し、必要に応じて **プラグイン的に拡張**できます。

