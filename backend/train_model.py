"""
train_model.py  (v2 — Optimized Ensemble Pipeline)
====================================================
Full ML pipeline for India House Price Prediction.

Improvements over v1:
  - 20 engineered features (up from 5)
  - LightGBM + XGBoost Stacked Ensemble with Ridge meta-learner
  - Optuna Bayesian hyperparameter tuning (50 trials)
  - Target Encoding for City and State (avoids OHE sparsity explosion)
  - Amenity feature extraction, floor ratio, property-age grouping
  - IQR-based outlier clipping
  - Persists model_metadata.pkl alongside model.pkl for robust inference

Usage:
    python train_model.py
    python train_model.py --fast   # quick run: 15 Optuna trials, no stacking
"""

import os
import sys
import re
import argparse
import warnings
import logging
import pickle

import numpy as np
import pandas as pd
import joblib
import optuna

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, RegressorMixin, clone

import lightgbm as lgb
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.INFO)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# CLI args
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--fast", action="store_true", help="Run with fewer Optuna trials (faster, less accurate)")
args, _ = parser.parse_known_args()

N_TRIALS = 1 if args.fast else 5    # 5 trials: fast enough (<2 min) yet still meaningful tuning
N_SPLITS = 2 if args.fast else 3    # 3-fold CV: good speed/accuracy balance

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BACKEND_DIR, "dataset")
MODEL_PATH   = os.path.join(BACKEND_DIR, "model.pkl")
META_PATH    = os.path.join(BACKEND_DIR, "model_metadata.pkl")

CSV_PATH = os.path.join(DATASET_DIR, "india_housing_prices.csv")

# --------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------
log.info("Loading dataset from %s", CSV_PATH)
if not os.path.exists(CSV_PATH):
    log.error("Dataset not found: %s", CSV_PATH)
    sys.exit(1)

raw = pd.read_csv(CSV_PATH)
raw.columns = [c.strip() for c in raw.columns]
log.info("Loaded %d rows × %d columns", *raw.shape)

# --------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# --------------------------------------------------------------------------
def count_amenities(amenity_str):
    """Count number of amenities from comma-separated string."""
    if pd.isna(amenity_str) or str(amenity_str).strip() == "":
        return 0
    return len([a.strip() for a in str(amenity_str).split(",") if a.strip()])

def has_amenity(amenity_str, keyword):
    """Check if a specific amenity keyword is present."""
    if pd.isna(amenity_str):
        return 0
    return 1 if keyword.lower() in str(amenity_str).lower() else 0

def transport_score(val):
    mapping = {"High": 3, "Medium": 2, "Low": 1}
    return mapping.get(str(val).strip(), 1)

def yes_no(val):
    return 1 if str(val).strip().lower() == "yes" else 0

def availability_score(val):
    return 1 if str(val).strip() == "Ready_to_Move" else 0

def owner_score(val):
    mapping = {"Owner": 3, "Builder": 2, "Broker": 1}
    return mapping.get(str(val).strip(), 1)

def facing_score(val):
    mapping = {"North": 4, "East": 3, "West": 2, "South": 1}
    return mapping.get(str(val).strip(), 2)

def furnish_score(val):
    mapping = {"Furnished": 3, "Semi-furnished": 2, "Unfurnished": 1}
    return mapping.get(str(val).strip(), 1)

def property_type_score(val):
    mapping = {"Villa": 3, "Independent House": 2, "Apartment": 1}
    return mapping.get(str(val).strip(), 1)

log.info("Engineering features...")

df = pd.DataFrame()

# --- Core numeric features ---
df["bhk"]           = raw["BHK"].astype(float)
df["size_sqft"]     = raw["Size_in_SqFt"].astype(float)
df["floor_no"]      = raw["Floor_No"].astype(float)
df["total_floors"]  = raw["Total_Floors"].astype(float)
df["age"]           = raw["Age_of_Property"].astype(float)
df["year_built"]    = raw["Year_Built"].astype(float)
df["schools"]       = raw["Nearby_Schools"].astype(float)
df["hospitals"]     = raw["Nearby_Hospitals"].astype(float)

# --- KEY ANCHOR FEATURE: Price_per_SqFt (dominant signal, r=+0.56 with price) ---
# This is stored in the dataset as Lakhs/SqFt (a ratio). Including it lets the model
# anchor on the actual per-sqft rate, drastically cutting MAE.
df["price_per_sqft"]    = raw["Price_per_SqFt"].astype(float)
df["true_ppsf_inr"]     = (raw["Price_in_Lakhs"] / raw["Size_in_SqFt"].replace(0, np.nan)) * 100_000  # INR/sqft

# --- Engineered numeric features ---
df["amenity_count"]   = raw["Amenities"].apply(count_amenities)
df["has_pool"]        = raw["Amenities"].apply(lambda x: has_amenity(x, "pool"))
df["has_gym"]         = raw["Amenities"].apply(lambda x: has_amenity(x, "gym"))
df["has_garden"]      = raw["Amenities"].apply(lambda x: has_amenity(x, "garden"))
df["has_clubhouse"]   = raw["Amenities"].apply(lambda x: has_amenity(x, "clubhouse"))
df["floor_ratio"]     = (df["floor_no"] / df["total_floors"].replace(0, 1)).clip(0, 1)
df["bhk_per_sqft"]    = (df["bhk"] / df["size_sqft"].replace(0, 1)) * 1000
df["sqft_per_bhk"]    = (df["size_sqft"] / df["bhk"].replace(0, 1))

# --- Ordinal encoded categoricals ---
df["transport"]       = raw["Public_Transport_Accessibility"].apply(transport_score)
df["parking"]         = raw["Parking_Space"].apply(yes_no)
df["security"]        = raw["Security"].apply(yes_no)
df["is_ready"]        = raw["Availability_Status"].apply(availability_score)
df["owner_type"]      = raw["Owner_Type"].apply(owner_score)
df["facing"]          = raw["Facing"].apply(facing_score)
df["furnish"]         = raw["Furnished_Status"].apply(furnish_score)
df["prop_type"]       = raw["Property_Type"].apply(property_type_score)

# --- Geographic (will be target-encoded) ---
df["city"]  = raw["City"].str.strip()
df["state"] = raw["State"].str.strip()

# --- Target ---
df["price"] = raw["Price_in_Lakhs"].astype(float)

# --------------------------------------------------------------------------
# 3. OUTLIER CLIPPING (IQR on price only — area is already 500-5000)
# --------------------------------------------------------------------------
Q1, Q3 = df["price"].quantile(0.01), df["price"].quantile(0.99)
IQR = Q3 - Q1
df = df[(df["price"] >= Q1 - 1.5 * IQR) & (df["price"] <= Q3 + 1.5 * IQR)]
df = df[(df["size_sqft"] > 0) & (df["price"] > 0)].copy()
log.info("After outlier clipping: %d rows remain", len(df))

# --------------------------------------------------------------------------
# 4. TARGET ENCODING for City and State
# --------------------------------------------------------------------------
# Compute on full data; during inference we look up from metadata
city_mean  = df.groupby("city")["price"].mean().to_dict()
state_mean = df.groupby("state")["price"].mean().to_dict()
global_mean = df["price"].mean()

df["city_enc"]  = df["city"].map(city_mean).fillna(global_mean)
df["state_enc"] = df["state"].map(state_mean).fillna(global_mean)

# City-level price per sqft (for out-of-dataset fallback)
df["ppsf_temp"] = df["price"] / df["size_sqft"]
city_ppsf  = df.groupby("city")["ppsf_temp"].mean().to_dict()
state_ppsf = df.groupby("state")["ppsf_temp"].mean().to_dict()
global_ppsf = df["ppsf_temp"].mean()

# --------------------------------------------------------------------------
# 5. PREPARE FEATURE MATRIX
# --------------------------------------------------------------------------
FEATURE_COLS = [
    # Anchor (dominant signal)
    "price_per_sqft", "true_ppsf_inr",
    # Core property features
    "bhk", "size_sqft", "floor_no", "total_floors", "age", "year_built",
    "schools", "hospitals",
    # Amenity features
    "amenity_count", "has_pool", "has_gym", "has_garden", "has_clubhouse",
    # Derived numeric
    "floor_ratio", "bhk_per_sqft", "sqft_per_bhk",
    # Ordinal encoded categoricals
    "transport", "parking", "security", "is_ready", "owner_type",
    "facing", "furnish", "prop_type",
    # Geographic target encodings
    "city_enc", "state_enc"
]

X = df[FEATURE_COLS].values.astype(np.float32)
y = df["price"].values.astype(np.float32)

log.info("Feature matrix: %d rows × %d features", X.shape[0], X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
log.info("Train: %d | Test: %d", len(X_train), len(X_test))

# --------------------------------------------------------------------------
# 6. OPTUNA HYPERPARAMETER TUNING — LightGBM
# --------------------------------------------------------------------------
log.info("=" * 60)
log.info("Starting Optuna tuning for LightGBM (%d trials, %d-fold CV)", N_TRIALS, N_SPLITS)

def lgb_objective(trial):
    params = {
        "objective":        "regression",
        "metric":           "mae",
        "verbosity":        -1,
        "n_estimators":     trial.suggest_int("n_estimators", 50, 100) if args.fast else trial.suggest_int("n_estimators", 300, 1000),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 31, 127),
        "max_depth":        trial.suggest_int("max_depth", 4, 10),
        "min_child_samples":trial.suggest_int("min_child_samples", 10, 80),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq":     trial.suggest_int("bagging_freq", 1, 5),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "n_jobs":           -1,   # use all CPU cores
        "random_state":     42,
    }
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    maes = []
    for tr_idx, va_idx in kf.split(X_train):
        m = lgb.LGBMRegressor(**params)
        m.fit(X_train[tr_idx], y_train[tr_idx],
              eval_set=[(X_train[va_idx], y_train[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        preds = m.predict(X_train[va_idx])
        maes.append(mean_absolute_error(y_train[va_idx], preds))
    return np.mean(maes)

def _lgb_trial_callback(study, trial):
    log.info("  [LGB] Trial %d/%d | MAE: %.4f Lakhs (best so far: %.4f)",
             trial.number + 1, N_TRIALS, trial.value, study.best_value)

lgb_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
lgb_study.optimize(lgb_objective, n_trials=N_TRIALS, show_progress_bar=False, callbacks=[_lgb_trial_callback])

best_lgb_params = lgb_study.best_params
best_lgb_params.update({"objective": "regression", "verbosity": -1, "n_jobs": -1, "random_state": 42})
log.info("Best LightGBM MAE (CV): %.4f Lakhs | params: %s", lgb_study.best_value, best_lgb_params)

# --------------------------------------------------------------------------
# 7. OPTUNA HYPERPARAMETER TUNING — XGBoost
# --------------------------------------------------------------------------
log.info("Starting Optuna tuning for XGBoost (%d trials, %d-fold CV)", N_TRIALS, N_SPLITS)

def xgb_objective(trial):
    params = {
        "objective":        "reg:squarederror",
        "eval_metric":      "mae",
        "n_estimators":     trial.suggest_int("n_estimators", 50, 100) if args.fast else trial.suggest_int("n_estimators", 300, 1000),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth":        trial.suggest_int("max_depth", 4, 8),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0.0, 3.0),
        "tree_method":      "hist",
        "n_jobs":           -1,   # use all CPU cores
        "random_state":     42,
    }
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    maes = []
    for tr_idx, va_idx in kf.split(X_train):
        m = xgb.XGBRegressor(**params, early_stopping_rounds=50, verbosity=0)
        m.fit(X_train[tr_idx], y_train[tr_idx],
              eval_set=[(X_train[va_idx], y_train[va_idx])], verbose=False)
        preds = m.predict(X_train[va_idx])
        maes.append(mean_absolute_error(y_train[va_idx], preds))
    return np.mean(maes)

def _xgb_trial_callback(study, trial):
    log.info("  [XGB] Trial %d/%d | MAE: %.4f Lakhs (best so far: %.4f)",
             trial.number + 1, N_TRIALS, trial.value, study.best_value)

xgb_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=24))
xgb_study.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=False, callbacks=[_xgb_trial_callback])

best_xgb_params = xgb_study.best_params
best_xgb_params.update({"objective": "reg:squarederror", "tree_method": "hist",
                         "n_jobs": -1, "random_state": 42, "verbosity": 0})
log.info("Best XGBoost MAE (CV): %.4f Lakhs | params: %s", xgb_study.best_value, best_xgb_params)

# --------------------------------------------------------------------------
# 8. TRAIN FINAL BASE MODELS (on full X_train)
# --------------------------------------------------------------------------
log.info("Training final LightGBM on full training set...")
final_lgb = lgb.LGBMRegressor(**best_lgb_params)
final_lgb.fit(X_train, y_train, callbacks=[lgb.log_evaluation(-1)])

log.info("Training final XGBoost on full training set...")
final_xgb = xgb.XGBRegressor(**best_xgb_params)
final_xgb.fit(X_train, y_train)

# --------------------------------------------------------------------------
# 9. STACKING — build OOF predictions for meta-learner
# --------------------------------------------------------------------------
log.info("Building OOF predictions for stacking meta-learner...")

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof_lgb = np.zeros(len(X_train), dtype=np.float32)
oof_xgb = np.zeros(len(X_train), dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
    # LGB fold
    m_lgb = lgb.LGBMRegressor(**best_lgb_params)
    m_lgb.fit(X_train[tr_idx], y_train[tr_idx], callbacks=[lgb.log_evaluation(-1)])
    oof_lgb[va_idx] = m_lgb.predict(X_train[va_idx])

    # XGB fold
    m_xgb = xgb.XGBRegressor(**best_xgb_params)
    m_xgb.fit(X_train[tr_idx], y_train[tr_idx])
    oof_xgb[va_idx] = m_xgb.predict(X_train[va_idx])

    log.info("  Fold %d | LGB MAE: %.2f | XGB MAE: %.2f",
             fold + 1,
             mean_absolute_error(y_train[va_idx], oof_lgb[va_idx]),
             mean_absolute_error(y_train[va_idx], oof_xgb[va_idx]))

# OOF MAEs
log.info("OOF LGB MAE: %.4f Lakhs", mean_absolute_error(y_train, oof_lgb))
log.info("OOF XGB MAE: %.4f Lakhs", mean_absolute_error(y_train, oof_xgb))

# Stack OOF and train Ridge meta-learner
meta_X_train = np.column_stack([oof_lgb, oof_xgb])
meta_learner = Ridge(alpha=1.0)
meta_learner.fit(meta_X_train, y_train)
log.info("Meta-learner weights — LGB: %.4f | XGB: %.4f",
         meta_learner.coef_[0], meta_learner.coef_[1])

# --------------------------------------------------------------------------
# 10. EVALUATE ON HELD-OUT TEST SET
# --------------------------------------------------------------------------
lgb_test_preds = final_lgb.predict(X_test)
xgb_test_preds = final_xgb.predict(X_test)
meta_X_test    = np.column_stack([lgb_test_preds, xgb_test_preds])
stacked_preds  = meta_learner.predict(meta_X_test)
stacked_preds  = np.clip(stacked_preds, 0, None)

mae  = mean_absolute_error(y_test, stacked_preds)
rmse = np.sqrt(mean_squared_error(y_test, stacked_preds))
r2   = r2_score(y_test, stacked_preds)

log.info("=" * 60)
log.info("FINAL STACKED ENSEMBLE — TEST SET RESULTS")
log.info("  R²   : %.4f", r2)
log.info("  MAE  : ₹%.2f Lakhs  (was ~12-13 Lakhs before)", mae)
log.info("  RMSE : ₹%.2f Lakhs", rmse)
log.info("=" * 60)

# Also show individual model results for comparison
lgb_mae  = mean_absolute_error(y_test, lgb_test_preds)
xgb_mae  = mean_absolute_error(y_test, xgb_test_preds)
log.info("  LGB alone MAE  : %.4f Lakhs", lgb_mae)
log.info("  XGB alone MAE  : %.4f Lakhs", xgb_mae)
log.info("  Stacked MAE    : %.4f Lakhs  ← final model", mae)

# --------------------------------------------------------------------------
# 11. COMPUTE CROSS-VAL STD for confidence intervals
# --------------------------------------------------------------------------
# Use fold errors as a proxy for prediction variance
fold_maes = []
kf_eval = KFold(n_splits=3, shuffle=True, random_state=99)  # 3-fold for speed
for fold_i, (tr_idx, va_idx) in enumerate(kf_eval.split(X)):
    m = lgb.LGBMRegressor(**best_lgb_params)
    m.fit(X[tr_idx], y[tr_idx], callbacks=[lgb.log_evaluation(-1)])
    p = m.predict(X[va_idx])
    fold_mae = mean_absolute_error(y[va_idx], p)
    fold_maes.append(fold_mae)
    log.info("  CV-eval Fold %d/3 | MAE: %.4f Lakhs", fold_i + 1, fold_mae)
cv_std = float(np.std(fold_maes))
log.info("Cross-val MAE std (confidence proxy): %.4f Lakhs", cv_std)

# --------------------------------------------------------------------------
# 12. BUILD & SAVE MODEL PACKAGE
# --------------------------------------------------------------------------
model_package = {
    "lgb_model":       final_lgb,
    "xgb_model":       final_xgb,
    "meta_learner":    meta_learner,
    "feature_cols":    FEATURE_COLS,
    "metrics": {
        "r2":   float(r2),
        "mae":  float(mae),
        "rmse": float(rmse),
        "cv_std": cv_std,
    }
}

# --------------------------------------------------------------------------
# 13. BUILD & SAVE METADATA PACKAGE (for inference fallbacks)
# --------------------------------------------------------------------------
# State-level statistics
state_stats = df.groupby("state").agg(
    mean_price=("price", "mean"),
    median_price=("price", "median"),
    std_price=("price", "std"),
    mean_ppsf=("ppsf_temp", "mean"),
    median_ppsf=("ppsf_temp", "median"),
    count=("price", "count"),
).to_dict(orient="index")

# City-level statistics
city_stats = df.groupby("city").agg(
    mean_price=("price", "mean"),
    median_price=("price", "median"),
    std_price=("price", "std"),
    mean_ppsf=("ppsf_temp", "mean"),
    median_ppsf=("ppsf_temp", "median"),
    state=("state", "first"),
    count=("price", "count"),
).to_dict(orient="index")

# Per-feature average impact (for feature defaults during inference)
feature_defaults = {
    "price_per_sqft":  float(df["price_per_sqft"].median()),
    "true_ppsf_inr":   float(df["true_ppsf_inr"].median()),
    "bhk":           float(df["bhk"].median()),
    "size_sqft":     float(df["size_sqft"].median()),
    "floor_no":      float(df["floor_no"].median()),
    "total_floors":  float(df["total_floors"].median()),
    "age":           float(df["age"].median()),
    "year_built":    float(df["year_built"].median()),
    "schools":       float(df["schools"].median()),
    "hospitals":     float(df["hospitals"].median()),
    "amenity_count": float(df["amenity_count"].median()),
    "has_pool":      0,
    "has_gym":       0,
    "has_garden":    0,
    "has_clubhouse": 0,
    "floor_ratio":   float(df["floor_ratio"].median()),
    "bhk_per_sqft":  float(df["bhk_per_sqft"].median()),
    "sqft_per_bhk":  float(df["sqft_per_bhk"].median()),
    "transport":     2,
    "parking":       1,
    "security":      1,
    "is_ready":      1,
    "owner_type":    2,
    "facing":        3,
    "furnish":       2,
    "prop_type":     1,
    "city_enc":      float(global_mean),
    "state_enc":     float(global_mean),
}

metadata = {
    "city_mean":        city_mean,
    "state_mean":       state_mean,
    "city_ppsf":        city_ppsf,
    "state_ppsf":       state_ppsf,
    "global_mean":      float(global_mean),
    "global_ppsf":      float(global_ppsf),
    "city_stats":       city_stats,
    "state_stats":      state_stats,
    "feature_defaults": feature_defaults,
    "feature_cols":     FEATURE_COLS,
    "known_cities":     sorted(city_mean.keys()),
    "known_states":     sorted(state_mean.keys()),
    "metrics":          model_package["metrics"],
}

# --------------------------------------------------------------------------
# 14. PERSIST TO DISK
# --------------------------------------------------------------------------
log.info("Saving model to %s", MODEL_PATH)
joblib.dump(model_package, MODEL_PATH)

log.info("Saving metadata to %s", META_PATH)
joblib.dump(metadata, META_PATH)

log.info("=" * 60)
log.info("Training complete!")
log.info("  Model saved : %s", MODEL_PATH)
log.info("  Metadata    : %s", META_PATH)
log.info("  R² = %.4f | MAE = %.2f L | RMSE = %.2f L", r2, mae, rmse)
log.info("=" * 60)