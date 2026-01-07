#!/usr/bin/env python
"""
Satellite Imagery Based Property Valuation (Hybrid Tabular + Satellite Images)

What this does:
- Reads Excel train/test
- Uses lat/long with pre-downloaded satellite images
- Builds XGBoost baseline and Hybrid CNN+Tabular model
- Outputs predictions using XGBoost model

Usage:
1) Put your data in data/ as train.xlsx and test.xlsx
2) Ensure satellite images are in the image_cache_dir (img_{index}.png format)
3) Run:
   python src/hybrid_property_valuation.py
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from dataclasses import dataclass
from typing import List, Optional, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models


# =========================
# CONFIG
# =========================
@dataclass
class Config:
    # Paths
    train_path: str = "data/train.xlsx"
    test_path: str = "data/test.xlsx"
    image_cache_dir: str = "data/mapbox_images"
    output_dir: str = "outputs"

    # Column names
    target_col: str = "price"
    lat_col: str = "lat"
    lon_col: str = "long"

    # Image settings
    image_size: int = 224

    # Training
    seed: int = 42
    test_size: float = 0.2
    batch_size: int = 32
    num_workers: int = 2
    epochs: int = 15
    lr: float = 5e-4
    weight_decay: float = 1e-4
    use_log_target: bool = True

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


# =========================
# DATA LOADING
# =========================
def load_data(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_excel(cfg.train_path, engine="openpyxl")
    test_df = pd.read_excel(cfg.test_path, engine="openpyxl")

    train_df.columns = [c.strip() for c in train_df.columns]
    test_df.columns = [c.strip() for c in test_df.columns]
    return train_df, test_df


def get_feature_columns(train_df: pd.DataFrame, cfg: Config) -> List[str]:
    exclude_cols = {cfg.target_col, "date", "id", cfg.lat_col, cfg.lon_col}
    return [c for c in train_df.columns 
            if c not in exclude_cols and train_df[c].dtype in ['int64', 'float64']]


def build_preprocessor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


# =========================
# EVALUATION
# =========================
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name}: RMSE=${rmse:,.2f}, R²={r2:.4f}")
    return {'model': model_name, 'RMSE': rmse, 'R2': r2}


# =========================
# DATASET
# =========================
class PropertyDataset(Dataset):
    def __init__(
        self,
        X_tabular: np.ndarray,
        y: Optional[np.ndarray] = None,
        image_ids: Optional[List] = None,
        cfg: Config = CFG,
        train_mode: bool = True
    ):
        self.X_tab = X_tabular.astype(np.float32)
        self.y = y.astype(np.float32) if y is not None else None
        self.image_ids = image_ids
        self.cfg = cfg

        if train_mode:
            self.transform = T.Compose([
                T.Resize((cfg.image_size, cfg.image_size)),
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.5),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.Resize((cfg.image_size, cfg.image_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.blank = Image.new("RGB", (cfg.image_size, cfg.image_size), color=(128, 128, 128))

    def __len__(self):
        return len(self.X_tab)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx] if self.image_ids is not None else idx
        img_path = os.path.join(self.cfg.image_cache_dir, f"img_{int(img_id)}.png")

        try:
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
            else:
                img = self.blank
        except:
            img = self.blank

        img_tensor = self.transform(img)
        tab_tensor = torch.from_numpy(self.X_tab[idx])

        if self.y is None:
            return img_tensor, tab_tensor
        return img_tensor, tab_tensor, torch.tensor(self.y[idx])


# =========================
# MODEL
# =========================
class HybridModel(nn.Module):
    def __init__(self, tabular_dim: int):
        super().__init__()

        # Frozen ResNet18 backbone
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Identity()
        self.cnn = backbone
        for param in self.cnn.parameters():
            param.requires_grad = False

        # CNN feature processor
        self.cnn_processor = nn.Sequential(
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU()
        )

        # Tabular MLP
        self.tabular = nn.Sequential(
            nn.Linear(tabular_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU()
        )

        # Fusion head
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, img, tab):
        with torch.no_grad():
            img_feat = self.cnn(img)
        img_feat = self.cnn_processor(img_feat)
        tab_feat = self.tabular(tab)
        return self.head(torch.cat([img_feat, tab_feat], dim=1)).squeeze(1)


# =========================
# TRAINING
# =========================
def train_hybrid_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_val: np.ndarray,
    cfg: Config
) -> dict:
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=cfg.weight_decay
    )
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_rmse = float('inf')
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        # Training
        model.train()
        for img, tab, y in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            img, tab, y = img.to(cfg.device), tab.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad()
            loss = criterion(model(img, tab), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        preds = []
        with torch.no_grad():
            for img, tab, _ in val_loader:
                preds.extend(model(img.to(cfg.device), tab.to(cfg.device)).cpu().numpy())

        val_rmse = np.sqrt(mean_squared_error(y_val, preds))
        scheduler.step(val_rmse)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"Epoch {epoch}: Val RMSE=${val_rmse:,.0f} ✓")
        else:
            print(f"Epoch {epoch}: Val RMSE=${val_rmse:,.0f}")

    print(f"\nBest Val RMSE: ${best_rmse:,.0f}")
    return best_state


@torch.no_grad()
def predict_hybrid(model: nn.Module, loader: DataLoader, cfg: Config) -> np.ndarray:
    model.eval()
    preds = []
    for img, tab in loader:
        img, tab = img.to(cfg.device), tab.to(cfg.device)
        preds.extend(model(img, tab).cpu().numpy())
    return np.array(preds)


# =========================
# MAIN
# =========================
def main():
    set_seed(CFG.seed)
    safe_mkdir(CFG.output_dir)

    print(f"Device: {CFG.device}")

    # Load data
    train_df, test_df = load_data(CFG)
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

    # Get feature columns
    feature_cols = get_feature_columns(train_df, CFG)
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    # Train/val split
    train_data, val_data = train_test_split(
        train_df, test_size=CFG.test_size, random_state=CFG.seed
    )
    train_image_ids = train_data.index.tolist()
    val_image_ids = val_data.index.tolist()
    print(f"Training: {len(train_data)}, Validation: {len(val_data)}")

    # Preprocess
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train_data[feature_cols])
    X_val = preprocessor.transform(val_data[feature_cols])
    X_test = preprocessor.transform(test_df[feature_cols])

    # Prepare targets
    if CFG.use_log_target:
        y_train = np.log1p(train_data[CFG.target_col].values)
        y_val = np.log1p(val_data[CFG.target_col].values)
        y_train_original = train_data[CFG.target_col].values
        y_val_original = val_data[CFG.target_col].values
    else:
        y_train = train_data[CFG.target_col].values
        y_val = val_data[CFG.target_col].values
        y_train_original = y_train
        y_val_original = y_val

    # =====================
    # XGBoost Model
    # =====================
    print("\n" + "="*50)
    print("Training XGBoost Model")
    print("="*50)

    xgb_model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=CFG.seed,
        early_stopping_rounds=50,
        eval_metric='rmse',
        tree_method='hist',
        device='cuda' if CFG.device == 'cuda' else 'cpu'
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    xgb_pred_raw = xgb_model.predict(X_val)
    if CFG.use_log_target:
        xgb_pred = np.expm1(xgb_pred_raw)
        xgb_results = evaluate_model(y_val_original, xgb_pred, 'XGBoost')
    else:
        xgb_results = evaluate_model(y_val, xgb_pred_raw, 'XGBoost')

    # =====================
    # Hybrid Model
    # =====================
    print("\n" + "="*50)
    print("Training Hybrid Model")
    print("="*50)

    # Prepare datasets
    y_train_hybrid = y_train_original.astype(np.float32)
    y_val_hybrid = y_val_original.astype(np.float32)

    train_dataset = PropertyDataset(X_train, y_train_hybrid, train_image_ids, CFG, True)
    val_dataset = PropertyDataset(X_val, y_val_hybrid, val_image_ids, CFG, False)

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)

    model = HybridModel(X_train.shape[1]).to(CFG.device)
    best_state = train_hybrid_model(model, train_loader, val_loader, y_val_hybrid, CFG)

    # Evaluate Hybrid
    model.load_state_dict(best_state)
    model.eval()
    hybrid_preds = []
    with torch.no_grad():
        for img, tab, _ in val_loader:
            hybrid_preds.extend(model(img.to(CFG.device), tab.to(CFG.device)).cpu().numpy())
    hybrid_results = evaluate_model(y_val_original, np.array(hybrid_preds), 'Hybrid')

    # =====================
    # Results Summary
    # =====================
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    results_df = pd.DataFrame([xgb_results, hybrid_results])
    print(results_df.to_string(index=False))

    # =====================
    # Generate Predictions (using XGBoost)
    # =====================
    print("\n" + "="*50)
    print("Generating Predictions")
    print("="*50)

    xgb_test_preds_raw = xgb_model.predict(X_test)
    test_preds = np.expm1(xgb_test_preds_raw) if CFG.use_log_target else xgb_test_preds_raw

    submission = pd.DataFrame({
        'id': test_df['id'] if 'id' in test_df.columns else range(len(test_preds)),
        'predicted_price': test_preds
    })

    output_path = os.path.join(CFG.output_dir, 'predictions.csv')
    submission.to_csv(output_path, index=False)
    print(f"Saved {len(test_preds)} predictions to {output_path}")

    print("\n✅ Complete!")
    print(f"XGBoost RMSE: ${xgb_results['RMSE']:,.0f}")
    print(f"Hybrid RMSE: ${hybrid_results['RMSE']:,.0f}")


if __name__ == "__main__":
    main()
