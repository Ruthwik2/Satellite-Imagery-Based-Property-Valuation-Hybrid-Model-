# Satellite Imagery Based Property Valuation (Hybrid Model)

This project trains a hybrid deep learning regression model using:
- **Tabular features** from Excel (bedrooms, bathrooms, sqft, condition, grade, etc.)
- **Satellite images** fetched using **lat/long** via an API (Mapbox or Google Static Maps)

The model uses:
- **ResNet18 (pretrained)** to extract image embeddings
- **MLP** for tabular features
- Concatenation of both -> final regression head -> **price** prediction
- **Grad-CAM** for model explainability

---

## Folder Structure

```
satellite_property_valuation/
├── src/
│   ├── hybrid_property_valuation.py  # Main training + prediction pipeline
│   └── data_fetcher.py               # Script to download satellite images from API
├── notebooks/
│   ├── preprocessing.ipynb           # Data cleaning and feature engineering
│   └── model_training.ipynb          # Training loop for the multimodal model
├── data/                             # Put your Excel files here
├── image_cache/                      # Cached satellite images (auto-created)
├── outputs/
│   └── predictions.csv               # Model predictions (id, predicted_price)
├── requirements.txt
└── README.md
```

---

## Inputs

### Required train columns
- `price` (target)
- `lat`, `long` (for images)
- plus any numeric tabular columns (e.g., bedrooms, bathrooms, sqft_living, grade, condition, etc.)

### Required test columns
- `lat`, `long`
- same tabular feature columns as train **excluding** `price`

---

## Setup

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Put your data files
Place your datasets in `data/` and rename:
- `train(1).xlsx` -> `data/train.xlsx`
- `test2.xlsx` -> `data/test.xlsx`

(Or edit the paths at the top of `src/hybrid_property_valuation.py`)

### 3) Set an API key (choose one)

#### Option A: Mapbox (recommended)
```bash
export MAPBOX_TOKEN="YOUR_TOKEN"
```
Default provider is Mapbox.

#### Option B: Google Static Maps
```bash
export GOOGLE_MAPS_KEY="YOUR_KEY"
```
Then set `provider="google"` in the Config.

### 4) Run training + prediction
```bash
python src/hybrid_property_valuation.py
```

---

## Outputs

- `outputs/predictions.csv` with columns: `id`, `predicted_price`
- Visualization outputs in `outputs/`:
  - `price_distribution.png` - Price distribution analysis
  - `correlation_heatmap.png` - Feature correlations
  - `geospatial_analysis.png` - Property locations map
  - `model_comparison.png` - XGBoost vs Hybrid comparison
  - `gradcam_sample_*.png` - Grad-CAM explainability visualizations
- During training, you’ll see epoch logs and best validation RMSE.

---

## Notes / Tips

- If no API key is provided, the code uses a blank placeholder image (pipeline still runs).
- To download images upfront for faster training, uncomment the `bulk_download_images(...)` lines in the script.
- Increase `epochs` for better performance (e.g., 20-30).
