"""
train_model.py
----------------
Loads the raw housing dataset, cleans messy currency-formatted price strings
(e.g. "Rs 1.99 Cr", "$450K", "45.0 L"), engineers usable numeric/categorical
features (area, bedrooms, bathrooms, location), imputes missing values, and
trains a RandomForestRegressor inside a scikit-learn Pipeline. The final
pipeline (preprocessing + model) is persisted to backend/model.pkl with
joblib so app.py can load it directly into memory at request time.
"""

import os
import re
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
backend_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(backend_dir, 'dataset', 'housing.json')
model_save_path = os.path.join(backend_dir, 'model.pkl')

# --------------------------------------------------------------------------
# Load dataset (with error handling for a missing/corrupt file)
# --------------------------------------------------------------------------
if not os.path.exists(dataset_path):
    print(f"[ERROR] Dataset not found at: {dataset_path}")
    print("Please place your housing.json file inside backend/dataset/ and retry.")
    sys.exit(1)

try:
    print(f"Loading dataset from: {dataset_path}")
    raw_df = pd.read_json(dataset_path)
except Exception as exc:
    print(f"[ERROR] Failed to read dataset: {exc}")
    sys.exit(1)

if raw_df.empty:
    print("[ERROR] Dataset is empty. Aborting training.")
    sys.exit(1)

# Normalize column names for safe/flexible lookups
raw_df.columns = [c.strip() for c in raw_df.columns]
lower_map = {c.lower(): c for c in raw_df.columns}


def find_col(*candidates):
    """Return the first matching real column name (case-insensitive)."""
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


price_col = find_col('price')
area_col = find_col('total_area', 'area', 'sqft', 'total area')
baths_col = find_col('baths', 'bathrooms', 'bathroom')
location_col = find_col('location')
title_col = find_col('property title', 'title')

missing_required = [name for name, col in [
    ('price', price_col), ('area', area_col), ('bathrooms', baths_col), ('location', location_col)
] if col is None]

if missing_required:
    print(f"[ERROR] Dataset is missing required column(s): {missing_required}")
    sys.exit(1)


# --------------------------------------------------------------------------
# Currency / numeric string cleaning
# --------------------------------------------------------------------------
def parse_numeric(val):
    """
    Convert messy currency-formatted strings into plain floats.
    Handles symbols/suffixes such as: Rs/₹, $, ',', 'Lakh'/'L', 'Crore'/'Cr', 'K'.
    Examples: "Rs 1.99 Cr" -> 19900000.0 | "$450K" -> 450000.0 | "45.0 L" -> 4500000.0
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)

    val_str = str(val).strip().replace('₹', '').replace('$', '').replace(',', '')
    val_upper = val_str.upper()

    multiplier = 1.0
    if 'CR' in val_upper:
        multiplier = 10_000_000.0  # 1 Crore = 10,000,000
        val_str = re.sub(r'CRORE[S]?|CR', '', val_upper).strip()
    elif 'LAKH' in val_upper or re.search(r'\bL\b', val_upper):
        multiplier = 100_000.0  # 1 Lakh = 100,000
        val_str = re.sub(r'LAKH[S]?|\bL\b', '', val_upper).strip()
    elif 'K' in val_upper:
        multiplier = 1_000.0
        val_str = val_upper.replace('K', '').strip()
    else:
        val_str = val_upper

    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", val_str)
        if match:
            return float(match.group()) * multiplier
        return np.nan
    except Exception:
        return np.nan


def extract_bedrooms(title):
    """Extract bedroom count from a listing title such as '3 BHK Flat...' or '1 RK Flat...'."""
    if pd.isna(title):
        return np.nan
    match = re.search(r'(\d+)\s*(?:BHK|RK)', str(title).upper())
    if match:
        return float(match.group(1))
    return np.nan


def extract_city(location):
    """
    Reduce high-cardinality free-text location strings down to a city/area
    name (the last comma-separated token), e.g. 'Kanathur Reddikuppam, Chennai'
    -> 'Chennai'. Falls back to the trimmed raw string when no comma exists.
    """
    if pd.isna(location):
        return 'Unknown'
    parts = str(location).split(',')
    return parts[-1].strip() if parts else str(location).strip()


# --------------------------------------------------------------------------
# Build the clean modelling dataframe
# --------------------------------------------------------------------------
df = pd.DataFrame()
df['price'] = raw_df[price_col].apply(parse_numeric)
df['area'] = pd.to_numeric(raw_df[area_col], errors='coerce')
df['bathrooms'] = pd.to_numeric(raw_df[baths_col], errors='coerce')
df['location'] = raw_df[location_col].apply(extract_city)

if title_col is not None:
    df['bedrooms'] = raw_df[title_col].apply(extract_bedrooms)
else:
    # Fallback heuristic if no title column is available in the dataset
    df['bedrooms'] = np.nan

# Drop rows where the target (price) or area could not be determined at all
df = df.dropna(subset=['price', 'area'])
df = df[df['price'] > 0]

print(f"Rows available for training after cleaning: {len(df)}")

X = df[['area', 'bedrooms', 'bathrooms', 'location']]
y = df['price']

numeric_features = ['area', 'bedrooms', 'bathrooms']
categorical_features = ['location']

# --------------------------------------------------------------------------
# Preprocessing + Model Pipeline
# --------------------------------------------------------------------------
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numeric_features),
    ('cat', cat_pipeline, categorical_features)
])

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=200,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

# --------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

predictions = model_pipeline.predict(X_test)
print("Model Training Complete!")
print(f"R2 Score: {r2_score(y_test, predictions):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.2f}")

# --------------------------------------------------------------------------
# Persist the trained pipeline
# --------------------------------------------------------------------------
try:
    joblib.dump(model_pipeline, model_save_path)
    print(f"Successfully saved model to: {model_save_path}")
except Exception as exc:
    print(f"[ERROR] Failed to save model: {exc}")
    sys.exit(1)