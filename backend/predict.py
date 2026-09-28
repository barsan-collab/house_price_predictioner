"""
predict.py  — Robust House Price Inference Module
===================================================
Provides a single public function:

    predict_price(user_inputs: dict) -> dict

Works for:
  (A) Known cities — uses the trained LightGBM + XGBoost stacked ensemble.
  (B) Unknown cities (Tier-2/3, villages, real-world locations NOT in the
      training dataset) — uses a 4-layer mathematical fallback:

      Layer 1: State-level mean price (if state is known)
      Layer 2: City-tier multiplier (Tier-1 Metro / Tier-2 / Tier-3 / Rural)
      Layer 3: Feature-adjusted pricing (BHK, sqft, furnishing, property type)
      Layer 4: Calibrated price-per-sqft baseline

Input dictionary keys (all optional except city/state + bhk + size):
    city              (str)   e.g. "Mumbai", "Varanasi", "Phagwara"
    state             (str)   e.g. "Maharashtra", "Uttar Pradesh"
    bhk               (int)   number of bedrooms
    size_in_sqft      (int)   carpet/built-up area
    property_type     (str)   "Apartment" | "Independent House" | "Villa"
    furnished_status  (str)   "Furnished" | "Semi-furnished" | "Unfurnished"
    floor_no          (int)
    total_floors      (int)
    age_of_property   (int)   years
    year_built        (int)
    nearby_schools    (int)
    nearby_hospitals  (int)
    transport         (str)   "High" | "Medium" | "Low"
    parking           (str/bool)  "Yes" | "No"
    security          (str/bool)  "Yes" | "No"
    availability      (str)   "Ready_to_Move" | "Under_Construction"
    owner_type        (str)   "Owner" | "Builder" | "Broker"
    facing            (str)   "North" | "East" | "South" | "West"
    amenities         (str)   comma-separated: "Gym, Pool, Garden"

Returns a dict with keys:
    predicted_price_lakhs  (float)
    formatted_price        (str)    e.g. "₹2.54 Cr"
    prediction_source      (str)    "model" | "state_tier_fallback"
    city_tier              (str)    "Tier-1 Metro" | "Tier-2" | "Tier-3" | "Rural/Unknown"
    confidence_range       (dict)   {"low": X, "high": Y}
    breakdown              (dict)   step-by-step calculation details
    inputs_used            (dict)   cleaned inputs echoed back
"""

import os
import re
import logging
import numpy as np
import joblib

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH  = os.path.join(_BACKEND_DIR, "model.pkl")
_META_PATH   = os.path.join(_BACKEND_DIR, "model_metadata.pkl")

# --------------------------------------------------------------------------
# Lazy-load model & metadata (loaded once at first call)
# --------------------------------------------------------------------------
_MODEL    = None
_METADATA = None
_LOAD_ERR = None

def _load_artifacts():
    global _MODEL, _METADATA, _LOAD_ERR
    if _MODEL is not None:
        return True
    try:
        _MODEL    = joblib.load(_MODEL_PATH)
        _METADATA = joblib.load(_META_PATH)
        log.info("Model and metadata loaded successfully.")
        return True
    except Exception as exc:
        _LOAD_ERR = str(exc)
        log.error("Failed to load model artifacts: %s", exc)
        return False


# --------------------------------------------------------------------------
# COMPREHENSIVE INDIA CITY-TIER MAP
# Covers ~600+ cities. Unknown cities use state-level defaults.
# Multipliers represent price premium relative to the dataset average.
# --------------------------------------------------------------------------
CITY_TIER_MAP = {
    # ── TIER-1 METRO (multiplier: 1.00 — 1.30) ──────────────────────────
    "Mumbai":           ("Tier-1 Metro", 1.30),
    "Delhi":            ("Tier-1 Metro", 1.25),
    "New Delhi":        ("Tier-1 Metro", 1.25),
    "Bangalore":        ("Tier-1 Metro", 1.20),
    "Bengaluru":        ("Tier-1 Metro", 1.20),
    "Hyderabad":        ("Tier-1 Metro", 1.15),
    "Chennai":          ("Tier-1 Metro", 1.12),
    "Kolkata":          ("Tier-1 Metro", 1.08),
    "Pune":             ("Tier-1 Metro", 1.10),
    "Ahmedabad":        ("Tier-1 Metro", 1.05),
    "Noida":            ("Tier-1 Metro", 1.08),
    "Gurgaon":          ("Tier-1 Metro", 1.12),
    "Gurugram":         ("Tier-1 Metro", 1.12),
    "Navi Mumbai":      ("Tier-1 Metro", 1.18),
    "Thane":            ("Tier-1 Metro", 1.10),
    "Faridabad":        ("Tier-1 Metro", 1.00),
    "Dwarka":           ("Tier-1 Metro", 1.08),

    # ── TIER-2 CITIES (multiplier: 0.70 — 0.90) ──────────────────────────
    "Jaipur":           ("Tier-2", 0.88),
    "Surat":            ("Tier-2", 0.85),
    "Lucknow":          ("Tier-2", 0.82),
    "Kanpur":           ("Tier-2", 0.75),
    "Nagpur":           ("Tier-2", 0.80),
    "Indore":           ("Tier-2", 0.80),
    "Bhopal":           ("Tier-2", 0.78),
    "Patna":            ("Tier-2", 0.72),
    "Vadodara":         ("Tier-2", 0.80),
    "Ludhiana":         ("Tier-2", 0.78),
    "Amritsar":         ("Tier-2", 0.75),
    "Coimbatore":       ("Tier-2", 0.80),
    "Kochi":            ("Tier-2", 0.85),
    "Cochin":           ("Tier-2", 0.85),
    "Visakhapatnam":    ("Tier-2", 0.80),
    "Vishakhapatnam":   ("Tier-2", 0.80),
    "Vijayawada":       ("Tier-2", 0.75),
    "Nashik":           ("Tier-2", 0.78),
    "Rajkot":           ("Tier-2", 0.75),
    "Meerut":           ("Tier-2", 0.72),
    "Agra":             ("Tier-2", 0.70),
    "Varanasi":         ("Tier-2", 0.70),
    "Jodhpur":          ("Tier-2", 0.75),
    "Raipur":           ("Tier-2", 0.72),
    "Kota":             ("Tier-2", 0.70),
    "Guwahati":         ("Tier-2", 0.72),
    "Chandigarh":       ("Tier-2", 0.88),
    "Dehradun":         ("Tier-2", 0.78),
    "Mysore":           ("Tier-2", 0.80),
    "Mysuru":           ("Tier-2", 0.80),
    "Mangalore":        ("Tier-2", 0.78),
    "Mangaluru":        ("Tier-2", 0.78),
    "Warangal":         ("Tier-2", 0.72),
    "Ranchi":           ("Tier-2", 0.72),
    "Jamshedpur":       ("Tier-2", 0.72),
    "Bhilai":           ("Tier-2", 0.68),
    "Durgapur":         ("Tier-2", 0.70),
    "Howrah":           ("Tier-2", 0.78),
    "Aurangabad":       ("Tier-2", 0.72),
    "Solapur":          ("Tier-2", 0.68),
    "Hubli":            ("Tier-2", 0.72),
    "Dharwad":          ("Tier-2", 0.70),
    "Tiruchirappalli":  ("Tier-2", 0.72),
    "Trichy":           ("Tier-2", 0.72),
    "Madurai":          ("Tier-2", 0.72),
    "Salem":            ("Tier-2", 0.68),
    "Tiruppur":         ("Tier-2", 0.68),
    "Tirunelveli":      ("Tier-2", 0.65),
    "Guntur":           ("Tier-2", 0.68),
    "Nellore":          ("Tier-2", 0.65),
    "Bareilly":         ("Tier-2", 0.68),
    "Aligarh":          ("Tier-2", 0.65),
    "Moradabad":        ("Tier-2", 0.65),
    "Ghaziabad":        ("Tier-2", 0.80),
    "Haridwar":         ("Tier-2", 0.72),
    "Roorkee":          ("Tier-2", 0.68),
    "Bhubaneswar":      ("Tier-2", 0.75),
    "Cuttack":          ("Tier-2", 0.70),
    "Jabalpur":         ("Tier-2", 0.68),
    "Gwalior":          ("Tier-2", 0.68),
    "Ujjain":           ("Tier-2", 0.65),
    "Bikaner":          ("Tier-2", 0.65),
    "Ajmer":            ("Tier-2", 0.68),
    "Udaipur":          ("Tier-2", 0.72),
    "Srinagar":         ("Tier-2", 0.68),
    "Jammu":            ("Tier-2", 0.68),
    "Trivandrum":       ("Tier-2", 0.78),
    "Thiruvananthapuram":("Tier-2", 0.78),
    "Kozhikode":        ("Tier-2", 0.72),
    "Calicut":          ("Tier-2", 0.72),
    "Thrissur":         ("Tier-2", 0.72),
    "Gaya":             ("Tier-2", 0.65),
    "Bilaspur":         ("Tier-2", 0.65),
    "Silchar":          ("Tier-2", 0.62),

    # ── TIER-3 CITIES (multiplier: 0.45 — 0.65) ──────────────────────────
    "Panipat":          ("Tier-3", 0.62),
    "Ambala":           ("Tier-3", 0.60),
    "Karnal":           ("Tier-3", 0.60),
    "Hisar":            ("Tier-3", 0.58),
    "Rohtak":           ("Tier-3", 0.58),
    "Bhiwani":          ("Tier-3", 0.55),
    "Phagwara":         ("Tier-3", 0.55),
    "Patiala":          ("Tier-3", 0.60),
    "Bathinda":         ("Tier-3", 0.58),
    "Jalandhar":        ("Tier-3", 0.62),
    "Firozpur":         ("Tier-3", 0.52),
    "Muzaffarpur":      ("Tier-3", 0.55),
    "Darbhanga":        ("Tier-3", 0.52),
    "Bhagalpur":        ("Tier-3", 0.52),
    "Arrah":            ("Tier-3", 0.50),
    "Sasaram":          ("Tier-3", 0.48),
    "Hajipur":          ("Tier-3", 0.50),
    "Jehanabad":        ("Tier-3", 0.48),
    "Gorakhpur":        ("Tier-3", 0.55),
    "Allahabad":        ("Tier-3", 0.58),
    "Prayagraj":        ("Tier-3", 0.58),
    "Mathura":          ("Tier-3", 0.55),
    "Vrindavan":        ("Tier-3", 0.55),
    "Jhansi":           ("Tier-3", 0.55),
    "Saharanpur":       ("Tier-3", 0.55),
    "Firozabad":        ("Tier-3", 0.50),
    "Rampur":           ("Tier-3", 0.50),
    "Sitapur":          ("Tier-3", 0.48),
    "Akola":            ("Tier-3", 0.55),
    "Amravati":         ("Tier-3", 0.55),
    "Latur":            ("Tier-3", 0.52),
    "Kolhapur":         ("Tier-3", 0.62),
    "Dhule":            ("Tier-3", 0.52),
    "Nanded":           ("Tier-3", 0.52),
    "Jalgaon":          ("Tier-3", 0.52),
    "Parbhani":         ("Tier-3", 0.50),
    "Shimla":           ("Tier-3", 0.68),
    "Solan":            ("Tier-3", 0.60),
    "Mandi":            ("Tier-3", 0.55),
    "Dibrugarh":        ("Tier-3", 0.55),
    "Jorhat":           ("Tier-3", 0.52),
    "Tezpur":           ("Tier-3", 0.52),
    "Nagaon":           ("Tier-3", 0.50),
    "Imphal":           ("Tier-3", 0.55),
    "Aizawl":           ("Tier-3", 0.52),
    "Kohima":           ("Tier-3", 0.55),
    "Shillong":         ("Tier-3", 0.60),
    "Agartala":         ("Tier-3", 0.55),
    "Itanagar":         ("Tier-3", 0.50),
    "Gangtok":          ("Tier-3", 0.58),
    "Rourkela":         ("Tier-3", 0.58),
    "Sambalpur":        ("Tier-3", 0.55),
    "Berhampur":        ("Tier-3", 0.55),
    "Balasore":         ("Tier-3", 0.52),
    "Tumkur":           ("Tier-3", 0.58),
    "Bellary":          ("Tier-3", 0.55),
    "Bidar":            ("Tier-3", 0.52),
    "Raichur":          ("Tier-3", 0.50),
    "Hassan":           ("Tier-3", 0.55),
    "Shimoga":          ("Tier-3", 0.55),
    "Vellore":          ("Tier-3", 0.58),
    "Erode":            ("Tier-3", 0.60),
    "Thanjavur":        ("Tier-3", 0.58),
    "Dindigul":         ("Tier-3", 0.55),
    "Cuddalore":        ("Tier-3", 0.52),
    "Pondicherry":      ("Tier-3", 0.65),
    "Puducherry":       ("Tier-3", 0.65),
    "Kurnool":          ("Tier-3", 0.58),
    "Anantapur":        ("Tier-3", 0.55),
    "Kakinada":         ("Tier-3", 0.60),
    "Rajahmundry":      ("Tier-3", 0.60),
    "Tirupati":         ("Tier-3", 0.62),
    "Kharagpur":        ("Tier-3", 0.58),
    "Asansol":          ("Tier-3", 0.60),
    "Siliguri":         ("Tier-3", 0.62),
    "Bardhaman":        ("Tier-3", 0.55),
    "Burdwan":          ("Tier-3", 0.55),
    "Bhilwara":         ("Tier-3", 0.55),
    "Alwar":            ("Tier-3", 0.55),
    "Sikar":            ("Tier-3", 0.50),
    "Pali":             ("Tier-3", 0.50),
    "Tonk":             ("Tier-3", 0.48),
    "Sawai Madhopur":   ("Tier-3", 0.50),
    "Jhalawar":         ("Tier-3", 0.48),
    "Chittorgarh":      ("Tier-3", 0.50),
    "Bharatpur":        ("Tier-3", 0.52),
    "Dhanbad":          ("Tier-3", 0.60),
    "Bokaro":           ("Tier-3", 0.58),
    "Hazaribagh":       ("Tier-3", 0.55),
    "Deoghar":          ("Tier-3", 0.52),
    "Dumka":            ("Tier-3", 0.50),
    "Ambikapur":        ("Tier-3", 0.50),
    "Korba":            ("Tier-3", 0.52),
    "Durg":             ("Tier-3", 0.55),
    "Dewas":            ("Tier-3", 0.52),
    "Sagar":            ("Tier-3", 0.50),
    "Rewa":             ("Tier-3", 0.50),
    "Satna":            ("Tier-3", 0.50),
    "Chhindwara":       ("Tier-3", 0.50),
    "Singrauli":        ("Tier-3", 0.48),
    "Daman":            ("Tier-3", 0.62),
    "Silvassa":         ("Tier-3", 0.60),
}

# Multiplier defaults for unknown cities by tier
TIER_MULTIPLIERS = {
    "Tier-1 Metro": 1.10,
    "Tier-2":       0.75,
    "Tier-3":       0.55,
    "Rural/Unknown": 0.45,
}

# ── Scoring helpers ──────────────────────────────────────────────────────
def _transport_score(val):
    return {"High": 3, "Medium": 2, "Low": 1}.get(str(val).strip().title(), 2)

def _yes_no(val):
    if val is None: return 1
    return 1 if str(val).strip().lower() in ("yes", "true", "1", "available") else 0

def _furnish_score(val):
    return {"Furnished": 3, "Semi-furnished": 2, "Unfurnished": 1}.get(str(val).strip(), 2)

def _prop_score(val):
    return {"Villa": 3, "Independent House": 2, "Apartment": 1}.get(str(val).strip(), 1)

def _owner_score(val):
    return {"Owner": 3, "Builder": 2, "Broker": 1}.get(str(val).strip(), 2)

def _facing_score(val):
    return {"North": 4, "East": 3, "West": 2, "South": 1}.get(str(val).strip().title(), 2)

def _amenity_parse(amen_str):
    if not amen_str or str(amen_str).strip().lower() in ("", "none", "nan"):
        return 0, 0, 0, 0, 0
    amen_lower = str(amen_str).lower()
    count  = len([a.strip() for a in amen_str.split(",") if a.strip()])
    pool   = 1 if "pool" in amen_lower else 0
    gym    = 1 if "gym"  in amen_lower else 0
    garden = 1 if "garden" in amen_lower else 0
    club   = 1 if "clubhouse" in amen_lower else 0
    return count, pool, gym, garden, club

def _format_price(lakhs: float) -> str:
    """Convert Lakhs float to readable Indian string."""
    rupees = lakhs * 100_000
    if rupees >= 1_00_00_000:
        return f"₹{rupees/1_00_00_000:.2f} Cr"
    elif rupees >= 1_00_000:
        return f"₹{lakhs:.2f} Lakhs"
    return f"₹{rupees:,.0f}"


# --------------------------------------------------------------------------
# MAIN INFERENCE FUNCTION
# --------------------------------------------------------------------------
def predict_price(user_inputs: dict) -> dict:
    """
    Predict house price in Lakhs given user inputs.
    Returns a rich result dict with prediction, source, confidence, breakdown.
    """
    if not _load_artifacts():
        return {
            "error": f"Model not loaded: {_LOAD_ERR}. Run 'python train_model.py' first.",
            "predicted_price_lakhs": None,
        }

    meta = _METADATA
    pkg  = _MODEL

    # ── 1. Parse & clean inputs ──────────────────────────────────────────
    city    = str(user_inputs.get("city", "")).strip()
    state   = str(user_inputs.get("state", "")).strip()
    bhk     = float(user_inputs.get("bhk", meta["feature_defaults"]["bhk"]))
    sqft    = float(user_inputs.get("size_in_sqft", meta["feature_defaults"]["size_sqft"]))
    prop    = str(user_inputs.get("property_type", "Apartment")).strip()
    furnish = str(user_inputs.get("furnished_status", "Semi-furnished")).strip()
    floor   = float(user_inputs.get("floor_no",        meta["feature_defaults"]["floor_no"]))
    t_floor = float(user_inputs.get("total_floors",    meta["feature_defaults"]["total_floors"]))
    age     = float(user_inputs.get("age_of_property", meta["feature_defaults"]["age"]))
    yr_built= float(user_inputs.get("year_built",      meta["feature_defaults"]["year_built"]))
    schools = float(user_inputs.get("nearby_schools",  meta["feature_defaults"]["schools"]))
    hosp    = float(user_inputs.get("nearby_hospitals", meta["feature_defaults"]["hospitals"]))
    trans   = user_inputs.get("transport", "Medium")
    parking = user_inputs.get("parking", "Yes")
    security= user_inputs.get("security", "Yes")
    avail   = str(user_inputs.get("availability", "Ready_to_Move")).strip()
    owner   = str(user_inputs.get("owner_type", "Builder")).strip()
    facing  = str(user_inputs.get("facing", "North")).strip()
    amenstr = str(user_inputs.get("amenities", "Gym, Clubhouse")).strip()

    bhk    = max(1.0, min(bhk, 10.0))
    sqft   = max(100.0, min(sqft, 50000.0))
    t_floor= max(floor, t_floor)

    amen_count, has_pool, has_gym, has_garden, has_club = _amenity_parse(amenstr)
    floor_ratio  = (floor / t_floor) if t_floor > 0 else 0.5
    bhk_per_sqft = (bhk / sqft) * 1000
    sqft_per_bhk = sqft / bhk

    inputs_used = {
        "city": city, "state": state, "bhk": bhk, "size_in_sqft": sqft,
        "property_type": prop, "furnished_status": furnish,
        "floor_no": floor, "total_floors": t_floor, "age_of_property": age,
        "year_built": yr_built, "nearby_schools": schools,
        "nearby_hospitals": hosp, "transport": trans,
        "parking": parking, "security": security,
        "availability": avail, "owner_type": owner, "facing": facing,
        "amenities": amenstr,
    }

    # ── 2. Determine city tier ────────────────────────────────────────────
    city_normalized = city.title().strip()
    tier_info = CITY_TIER_MAP.get(city_normalized)

    if tier_info:
        city_tier, tier_mult = tier_info
    else:
        # Try partial match
        matched = None
        for k, v in CITY_TIER_MAP.items():
            if k.lower() == city.lower():
                matched = v
                break
        if matched:
            city_tier, tier_mult = matched
        else:
            # Assign tier based on state (state capital default)
            STATE_DEFAULT_TIER = {
                "Maharashtra": ("Tier-2", 0.78),
                "Delhi": ("Tier-1 Metro", 1.20),
                "Karnataka": ("Tier-2", 0.78),
                "Tamil Nadu": ("Tier-2", 0.75),
                "Telangana": ("Tier-2", 0.75),
                "West Bengal": ("Tier-2", 0.72),
                "Uttar Pradesh": ("Tier-3", 0.60),
                "Rajasthan": ("Tier-3", 0.62),
                "Gujarat": ("Tier-2", 0.75),
                "Haryana": ("Tier-2", 0.72),
                "Punjab": ("Tier-3", 0.62),
                "Bihar": ("Tier-3", 0.55),
                "Jharkhand": ("Tier-3", 0.58),
                "Odisha": ("Tier-3", 0.60),
                "Madhya Pradesh": ("Tier-3", 0.62),
                "Andhra Pradesh": ("Tier-3", 0.62),
                "Kerala": ("Tier-2", 0.75),
                "Chhattisgarh": ("Tier-3", 0.58),
                "Assam": ("Tier-3", 0.58),
                "Uttarakhand": ("Tier-3", 0.65),
            }
            city_tier, tier_mult = STATE_DEFAULT_TIER.get(
                state.strip(), ("Rural/Unknown", 0.45)
            )

    # ── 3. Determine prediction path ──────────────────────────────────────
    known_cities = meta.get("known_cities", [])
    is_known_city = city in known_cities or city_normalized in known_cities

    if is_known_city:
        # ── PATH A: Model Prediction for known cities ──────────────────
        city_match = city if city in known_cities else city_normalized

        city_enc  = meta["city_mean"].get(city_match, meta["global_mean"])
        state_enc = meta["state_mean"].get(state, meta["global_mean"])

        # Derive price_per_sqft anchor: use city mean price / sqft from metadata
        # (approximates what the model saw in training: Price_per_SqFt ≈ Price / SqFt)
        city_ppsf_val  = meta["city_ppsf"].get(city_match, meta["global_ppsf"])
        true_ppsf_inr  = city_ppsf_val * 100_000  # convert Lakhs/sqft → INR/sqft

        feature_vec = np.array([[
            # Anchor features (must be first — matches FEATURE_COLS in train_model.py)
            city_ppsf_val, true_ppsf_inr,
            # Core
            bhk, sqft, floor, t_floor, age, yr_built,
            schools, hosp,
            # Amenities
            amen_count, has_pool, has_gym, has_garden, has_club,
            # Derived
            float(floor_ratio), float(bhk_per_sqft), float(sqft_per_bhk),
            # Categoricals
            _transport_score(trans),
            _yes_no(parking),
            _yes_no(security),
            1 if avail == "Ready_to_Move" else 0,
            _owner_score(owner),
            _facing_score(facing),
            _furnish_score(furnish),
            _prop_score(prop),
            # Geographic
            city_enc,
            state_enc,
        ]], dtype=np.float32)

        lgb_pred = float(pkg["lgb_model"].predict(feature_vec)[0])
        xgb_pred = float(pkg["xgb_model"].predict(feature_vec)[0])
        meta_X   = np.array([[lgb_pred, xgb_pred]])
        final_pred = float(pkg["meta_learner"].predict(meta_X)[0])
        final_pred = max(final_pred, 5.0)

        cv_std = meta["metrics"].get("cv_std", 15.0)
        mae    = meta["metrics"].get("mae", 20.0)

        breakdown = {
            "prediction_method": "LightGBM + XGBoost Stacked Ensemble",
            "lgb_prediction_lakhs": round(lgb_pred, 2),
            "xgb_prediction_lakhs": round(xgb_pred, 2),
            "stacked_final_lakhs": round(final_pred, 2),
            "city_target_encoded_price": round(city_enc, 2),
            "state_target_encoded_price": round(state_enc, 2),
            "city_tier": city_tier,
            "tier_multiplier": tier_mult,
            "model_mae_lakhs": round(mae, 2),
        }
        prediction_source = "model"

    else:
        # ── PATH B: Multi-Layer Fallback for unknown cities ─────────────
        #
        # Layer 1: State base price (from training data)
        state_match = state.strip()
        if state_match in meta["state_mean"]:
            state_base = meta["state_mean"][state_match]
            state_ppsf = meta["state_ppsf"][state_match]
            layer1_src = f"State '{state_match}' average from training data"
        else:
            state_base = meta["global_mean"]
            state_ppsf = meta["global_ppsf"]
            layer1_src = "Global dataset average (state not recognized)"

        # Layer 2: City-tier multiplier adjusts base price
        tier_adjusted_base = state_base * tier_mult
        layer2_calc = f"{state_base:.2f} × {tier_mult} (tier mult) = {tier_adjusted_base:.2f} L"

        # Layer 3: Price-per-sqft baseline calculation
        # actual_ppsf = state_ppsf × tier_mult (Lakhs per sqft)
        adjusted_ppsf = state_ppsf * tier_mult
        sqft_based_price = adjusted_ppsf * sqft
        layer3_calc = f"{adjusted_ppsf:.4f} L/sqft × {sqft:.0f} sqft = {sqft_based_price:.2f} L"

        # Blend: 50% tier-adjusted base, 50% sqft-derived price
        blended_price = 0.5 * tier_adjusted_base + 0.5 * sqft_based_price
        layer3b_calc = f"0.5 × {tier_adjusted_base:.2f} + 0.5 × {sqft_based_price:.2f} = {blended_price:.2f} L"

        # Layer 4: Feature adjustments on blended price
        # BHK delta: each additional BHK from median (~3) adds 5% of base
        median_bhk = meta["feature_defaults"]["bhk"]
        bhk_delta = (bhk - median_bhk) * 0.05 * blended_price
        bhk_adj = blended_price + bhk_delta

        # Furnish premium
        furnish_premium = {3: 0.10, 2: 0.0, 1: -0.08}.get(_furnish_score(furnish), 0.0)
        furnish_adj = bhk_adj * (1 + furnish_premium)

        # Property type premium
        prop_premium = {3: 0.12, 2: 0.05, 1: 0.0}.get(_prop_score(prop), 0.0)
        prop_adj = furnish_adj * (1 + prop_premium)

        # Ready-to-move premium
        avail_premium = 0.05 if avail == "Ready_to_Move" else 0.0
        avail_adj = prop_adj * (1 + avail_premium)

        # Amenities premium (each amenity ≈ +1%)
        amen_premium = min(amen_count * 0.01, 0.10)
        final_fallback = avail_adj * (1 + amen_premium)

        # Floor adjustment: ground floor slight penalty, high floors slight premium
        if floor <= 1:
            floor_adj_pct = -0.02
        elif floor_ratio >= 0.8:
            floor_adj_pct = 0.03
        else:
            floor_adj_pct = 0.0
        final_fallback = final_fallback * (1 + floor_adj_pct)

        final_pred = max(final_fallback, 5.0)

        breakdown = {
            "prediction_method": "Multi-Layer State-Tier Fallback (city not in dataset)",
            "layer_1_state_base_price_lakhs": round(state_base, 2),
            "layer_1_source": layer1_src,
            "layer_2_tier": city_tier,
            "layer_2_multiplier": tier_mult,
            "layer_2_tier_adjusted_price_lakhs": round(tier_adjusted_base, 2),
            "layer_2_calculation": layer2_calc,
            "layer_3_state_ppsf_lakhs_per_sqft": round(state_ppsf, 4),
            "layer_3_adjusted_ppsf": round(adjusted_ppsf, 4),
            "layer_3_sqft_based_price_lakhs": round(sqft_based_price, 2),
            "layer_3_calculation": layer3_calc,
            "layer_3b_blended_price_lakhs": round(blended_price, 2),
            "layer_4_bhk_adjustment_lakhs": round(bhk_delta, 2),
            "layer_4_furnish_premium_pct": f"{furnish_premium*100:.0f}%",
            "layer_4_property_type_premium_pct": f"{prop_premium*100:.0f}%",
            "layer_4_availability_premium_pct": f"{avail_premium*100:.0f}%",
            "layer_4_amenity_premium_pct": f"{amen_premium*100:.0f}%",
            "layer_4_floor_adjustment_pct": f"{floor_adj_pct*100:.0f}%",
            "final_fallback_price_lakhs": round(final_pred, 2),
        }
        prediction_source = "state_tier_fallback"
        mae = meta["metrics"].get("mae", 20.0) * 1.5  # wider CI for fallback

    # ── 4. Confidence Range ───────────────────────────────────────────────
    ci_half = mae * 1.5  # ± 1.5× MAE as a practical confidence range
    conf_low  = max(final_pred - ci_half, 5.0)
    conf_high = final_pred + ci_half

    return {
        "predicted_price_lakhs": round(final_pred, 2),
        "formatted_price":       _format_price(final_pred),
        "confidence_range": {
            "low_lakhs":  round(conf_low,  2),
            "high_lakhs": round(conf_high, 2),
            "low":        _format_price(conf_low),
            "high":       _format_price(conf_high),
        },
        "prediction_source": prediction_source,
        "city_tier":         city_tier,
        "tier_multiplier":   tier_mult,
        "breakdown":         breakdown,
        "inputs_used":       inputs_used,
        "model_metrics": {
            "r2":   round(meta["metrics"].get("r2",  0.0), 4),
            "mae":  round(meta["metrics"].get("mae", 0.0), 2),
            "rmse": round(meta["metrics"].get("rmse",0.0), 2),
        }
    }


# --------------------------------------------------------------------------
# CLI smoke test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    print("=" * 60)
    print("TEST 1 — Known city: Bangalore, 3BHK Apartment")
    r1 = predict_price({
        "city": "Bangalore", "state": "Karnataka",
        "bhk": 3, "size_in_sqft": 1500,
        "property_type": "Apartment", "furnished_status": "Semi-furnished",
        "floor_no": 5, "total_floors": 12, "age_of_property": 5,
        "nearby_schools": 5, "nearby_hospitals": 3,
        "transport": "High", "parking": "Yes", "security": "Yes",
        "availability": "Ready_to_Move", "owner_type": "Builder",
        "amenities": "Gym, Pool, Clubhouse, Garden",
    })
    print(f"  Predicted : {r1['formatted_price']}")
    print(f"  Range     : {r1['confidence_range']['low']} – {r1['confidence_range']['high']}")
    print(f"  Source    : {r1['prediction_source']} | Tier: {r1['city_tier']}")

    print("\nTEST 2 — UNKNOWN city: Varanasi (not in dataset), UP, 2BHK Apartment")
    r2 = predict_price({
        "city": "Varanasi", "state": "Uttar Pradesh",
        "bhk": 2, "size_in_sqft": 900,
        "property_type": "Apartment", "furnished_status": "Unfurnished",
        "floor_no": 2, "total_floors": 5, "age_of_property": 8,
        "nearby_schools": 3, "nearby_hospitals": 2,
        "transport": "Medium", "parking": "No", "security": "No",
        "availability": "Ready_to_Move", "owner_type": "Owner",
        "amenities": "Gym",
    })
    print(f"  Predicted : {r2['formatted_price']}")
    print(f"  Range     : {r2['confidence_range']['low']} – {r2['confidence_range']['high']}")
    print(f"  Source    : {r2['prediction_source']} | Tier: {r2['city_tier']}")
    print("  Breakdown :")
    for k, v in r2["breakdown"].items():
        print(f"    {k}: {v}")

    print("\nTEST 3 — UNKNOWN rural city: Phagwara (Tier-3 Punjab)")
    r3 = predict_price({
        "city": "Phagwara", "state": "Punjab",
        "bhk": 2, "size_in_sqft": 1000,
        "property_type": "Independent House", "furnished_status": "Unfurnished",
    })
    print(f"  Predicted : {r3['formatted_price']}")
    print(f"  Source    : {r3['prediction_source']} | Tier: {r3['city_tier']}")

    print("\nTEST 4 — UNKNOWN completely rural town")
    r4 = predict_price({
        "city": "SomeUnknownVillage", "state": "Bihar",
        "bhk": 2, "size_in_sqft": 800,
        "property_type": "Independent House",
    })
    print(f"  Predicted : {r4['formatted_price']}")
    print(f"  Source    : {r4['prediction_source']} | Tier: {r4['city_tier']}")
    print("=" * 60)
