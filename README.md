# Satellite Imagery Based Property Valuation

A machine learning project that predicts property prices using:
- **Tabular features** (bedrooms, bathrooms, sqft, condition, grade, etc.)
- **Satellite imagery** (fetched via Mapbox/Google Static Maps API)

## Models

1. **XGBoost** - Gradient boosting on tabular features (primary model for predictions)
2. **Hybrid CNN+Tabular** - Frozen ResNet18 + MLP fusion model

## Evaluation Metrics
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)

---

## Folder Structure

```
satellite_property_valuation/
├── kaggle_notebook.ipynb             # Main Kaggle notebook
├── colab_notebook.ipynb              # Google Colab version
├── src/
│   ├── hybrid_property_valuation.py  # Standalone Python training script
│   └── data_fetcher.py               # Satellite image fetching utility
├── notebooks/
│   ├── preprocessing.ipynb           # Data exploration & preprocessing
│   └── model_training.ipynb          # Model training notebook
├── data/
│   ├── train.xlsx                    # Training data
│   ├── test.xlsx                     # Test data
│   └── mapbox_images/                # Satellite images (img_{id}.png)
├── outputs/
│   ├── predictions.csv               # Final predictions
│   ├── xgb_model.joblib              # Saved XGBoost model
│   └── hybrid_model.pth              # Saved Hybrid model
├── requirements.txt
└── README.md
```

---

## Data Requirements

### Train columns
| Column | Description |
|--------|-------------|
| `id` | Unique identifier |
| `price` | Target variable |
| `lat`, `long` | Coordinates for satellite images |
| numeric features | bedrooms, bathrooms, sqft_living, grade, condition, etc. |

### Test columns
Same as train **excluding** `price`

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data
Place your Excel files in `data/`:
- `data/train.xlsx`
- `data/test.xlsx`

### 3. Fetch satellite images

#### Option A: Using data_fetcher.py
```bash
# Set API key
export MAPBOX_TOKEN="your_mapbox_token"
# OR
export GOOGLE_MAPS_KEY="your_google_key"

# Run fetcher
python src/data_fetcher.py
```

#### Option B: Manual download
Place satellite images in `data/mapbox_images/` with naming format: `img_{id}.png`

### 4. Run training

#### Using Python script
```bash
python src/hybrid_property_valuation.py
```

#### Using Jupyter notebook
```bash
jupyter notebook notebooks/model_training.ipynb
```

#### Using Kaggle/Colab
Upload `kaggle_notebook.ipynb` or `colab_notebook.ipynb` to the respective platform.

---

## Configuration

Key parameters in `CONFIG`:

```python
CONFIG = {
    'target_col': 'price',
    'lat_col': 'lat',
    'lon_col': 'long',
    'image_size': 224,
    'seed': 42,
    'test_size': 0.2,
    'batch_size': 32,
    'epochs': 15,
    'use_log_target': True,  # Log-transform target for better performance
}
```

---

## Model Details

### XGBoost
- `n_estimators`: 500
- `max_depth`: 6
- `learning_rate`: 0.05
- `early_stopping_rounds`: 50
- Log-transformed target with `np.log1p()` / `np.expm1()`

### Hybrid Model
- **CNN Branch**: Frozen ResNet18 backbone → 512 → 128 → 64
- **Tabular Branch**: MLP with BatchNorm → 256 → 128 → 64
- **Fusion Head**: Concatenated features → 64 → 32 → 1
- **Training**: AdamW optimizer, ReduceLROnPlateau scheduler

---

## Outputs

| File | Description |
|------|-------------|
| `predictions.csv` | Test predictions (`id`, `predicted_price`) |
| `xgb_model.joblib` | Saved XGBoost model |
| `hybrid_model.pth` | Saved PyTorch Hybrid model |
| `preprocessor.joblib` | Fitted preprocessing pipeline |
| `model_comparison.png` | RMSE/R² comparison chart |

---

## Usage Examples

### Generate predictions
```python
import joblib
import pandas as pd
import numpy as np

# Load model and preprocessor
xgb_model = joblib.load('outputs/xgb_model.joblib')
preprocessor = joblib.load('outputs/preprocessor.joblib')

# Prepare test data
test_df = pd.read_excel('data/test.xlsx')
X_test = preprocessor.transform(test_df[feature_cols])

# Predict (with log-transform)
predictions = np.expm1(xgb_model.predict(X_test))
```

### Load Hybrid model
```python
import torch

checkpoint = torch.load('outputs/hybrid_model.pth')
model = HybridModel(tabular_dim=len(feature_cols))
model.load_state_dict(checkpoint)
```

---

## Notes

- **XGBoost** is used for final predictions (better generalization on tabular data)
- **Hybrid model** leverages satellite imagery for additional spatial context
- If images are missing, a gray placeholder (128, 128, 128) is used
- Set `use_log_target=True` for better performance on skewed price distributions
