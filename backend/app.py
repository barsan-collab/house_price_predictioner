"""
app.py  (v2 — Optimized Ensemble Backend)
==========================================
Flask web server for the House Price Prediction portfolio project.

Changes from v1:
  - Uses predict.py's predict_price() instead of raw joblib model calls.
  - API accepts full feature set (city, state, property_type, furnished_status, etc.)
  - Returns richer JSON: prediction_source, city_tier, confidence_range, breakdown.
  - /api/locations now returns cities AND states.
  - Fallback for unknown cities is now mathematically grounded (4-layer logic).
"""

import os
import logging
from datetime import datetime, timezone

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify
)
from flask_cors import CORS
import joblib
from pymongo import MongoClient
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError

# Import the new optimized predict module
from predict import predict_price, _format_price, _load_artifacts, CITY_TIER_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# App / path configuration
# --------------------------------------------------------------------------
base_dir     = os.path.abspath(os.path.dirname(__file__))
frontend_dir = os.path.abspath(os.path.join(base_dir, '..', 'frontend'))
META_PATH    = os.path.join(base_dir, 'model_metadata.pkl')
MODEL_PATH   = os.path.join(base_dir, 'model.pkl')

app = Flask(
    __name__,
    template_folder=frontend_dir,
    static_folder=frontend_dir,
    static_url_path=''
)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'house_price_prediction_secret_key')

# Allow requests from Live Server (port 5500), file://, and any localhost origin
CORS(app, resources={r'/api/*': {'origins': '*'}})

# --------------------------------------------------------------------------
# Lazy-load metadata (cities, states) from model_metadata.pkl
# --------------------------------------------------------------------------
_metadata         = None
known_cities      = []
known_states      = []
model_metrics     = {}
model_available   = False
model_load_error  = None

def _ensure_metadata():
    global _metadata, known_cities, known_states, model_metrics, model_available, model_load_error
    if _metadata is not None:
        return
    try:
        # --- Check model.pkl ---
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"model.pkl not found at {MODEL_PATH}. Run 'python train_model.py' first."
            )
        # --- Check model_metadata.pkl ---
        if not os.path.exists(META_PATH):
            raise FileNotFoundError(
                f"model_metadata.pkl not found at {META_PATH}. Run 'python train_model.py' first."
            )
        # Verify model.pkl is loadable (catches corrupt / incompatible pickles)
        test_model = joblib.load(MODEL_PATH)
        if not isinstance(test_model, dict) or 'lgb_model' not in test_model:
            raise ValueError(
                "model.pkl has unexpected structure. Re-run 'python train_model.py'."
            )
        del test_model  # free memory; predict.py will re-load lazily

        _metadata     = joblib.load(META_PATH)
        known_cities  = sorted(_metadata.get("known_cities",  []))
        known_states  = sorted(_metadata.get("known_states",  []))
        model_metrics = _metadata.get("metrics", {})
        model_available = True
        logger.info("Model & metadata loaded: %d cities, %d states", len(known_cities), len(known_states))
    except Exception as exc:
        model_load_error = str(exc)
        logger.error("Failed to load model artifacts: %s", model_load_error)

_ensure_metadata()

# --------------------------------------------------------------------------
# MongoDB connection
# --------------------------------------------------------------------------
MONGO_URI       = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
MONGO_TIMEOUT_MS = 3000

try:
    client           = MongoClient(MONGO_URI, serverSelectionTimeoutMS=MONGO_TIMEOUT_MS)
    client.admin.command('ping')
    db               = client['house_price_db']
    predictions_col  = db['predictions']
    db_available     = True
    logger.info("Connected to MongoDB at %s", MONGO_URI)
except (ServerSelectionTimeoutError, PyMongoError) as exc:
    logger.warning("MongoDB unavailable (%s). Running without persistence.", exc)
    client          = None
    predictions_col = None
    db_available    = False


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _get_recent_history(limit: int = 5):
    """Safely fetch recent predictions from MongoDB."""
    if not db_available or predictions_col is None:
        return []
    try:
        docs = list(predictions_col.find().sort('created_at', -1).limit(limit))
        for doc in docs:
            doc['_id'] = str(doc['_id'])
            ts = doc.get('created_at')
            doc['timestamp'] = ts.strftime('%Y-%m-%d %H:%M UTC') if isinstance(ts, datetime) else 'N/A'
            doc['formatted_price'] = _format_price(doc.get('predicted_price_lakhs', 0))
        return docs
    except PyMongoError as exc:
        logger.warning("Could not fetch history: %s", exc)
        return []


def _parse_int(val, default=0):
    try: return int(float(val))
    except: return default

def _parse_float(val, default=0.0):
    try: return float(val)
    except: return default


# --------------------------------------------------------------------------
# API Endpoints
# --------------------------------------------------------------------------
@app.route('/api/locations', methods=['GET'])
def api_locations():
    """Return known cities, states, and all tier-mapped cities."""
    all_tier_cities = sorted(CITY_TIER_MAP.keys())
    return jsonify({
        'success':         True,
        'cities':          known_cities,
        'states':          known_states,
        'tier_cities':     all_tier_cities,
        'model_metrics':   model_metrics,
    })


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Accept JSON prediction request, return rich JSON result.

    Accepts both canonical backend field names and the aliases sent by the
    frontend form:
        location  -> city
        area      -> size_in_sqft
        bedrooms  -> bhk
    """
    if not model_available:
        return jsonify({'success': False, 'error': f"Model unavailable: {model_load_error}"}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'success': False, 'error': 'Invalid or missing JSON payload. '
                        'Ensure Content-Type: application/json is set.'}), 400

    # ----------------------------------------------------------------
    # Resolve field aliases: frontend may send location/area/bedrooms
    # ----------------------------------------------------------------
    city  = str(data.get('city') or data.get('location') or '').strip()
    state = str(data.get('state', '')).strip()
    bhk   = _parse_float(data.get('bhk') if data.get('bhk') is not None
                         else data.get('bedrooms', 2))
    sqft  = _parse_float(data.get('size_in_sqft') if data.get('size_in_sqft') is not None
                         else data.get('area', 0))

    # ----------------------------------------------------------------
    # Validate required fields
    # ----------------------------------------------------------------
    if not city:
        return jsonify({'success': False,
                        'error': 'city (or location) is required.'}), 400
    if sqft <= 0:
        return jsonify({'success': False,
                        'error': 'size_in_sqft (or area) must be a positive number.'}), 400
    if bhk <= 0:
        return jsonify({'success': False,
                        'error': 'bhk (or bedrooms) must be at least 1.'}), 400

    inputs = {
        # Canonical names used by predict.py
        "city":              city,
        "state":             state,
        "bhk":               bhk,
        "size_in_sqft":      sqft,
        "property_type":     str(data.get('property_type', 'Apartment')),
        "furnished_status":  str(data.get('furnished_status', 'Semi-furnished')),
        "floor_no":          _parse_float(data.get('floor_no', 3)),
        "total_floors":      _parse_float(data.get('total_floors', 10)),
        "age_of_property":   _parse_float(data.get('age_of_property', 5)),
        "year_built":        _parse_float(data.get('year_built', 2015)),
        "nearby_schools":    _parse_float(data.get('nearby_schools', 5)),
        "nearby_hospitals":  _parse_float(data.get('nearby_hospitals', 3)),
        "transport":         str(data.get('transport', 'Medium')),
        "parking":           str(data.get('parking', 'Yes')),
        "security":          str(data.get('security', 'Yes')),
        "availability":      str(data.get('availability', 'Ready_to_Move')),
        "owner_type":        str(data.get('owner_type', 'Builder')),
        "facing":            str(data.get('facing', 'North')),
        "amenities":         str(data.get('amenities', '')),
        # Echo back frontend aliases so the result card can display them
        "area":              sqft,
        "bedrooms":          int(bhk),
        "bathrooms":         _parse_int(data.get('bathrooms', 2)),
        "location":          city,
    }

    try:
        result = predict_price(inputs)
        if "error" in result:
            return jsonify({'success': False, 'error': result["error"]}), 500
    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        return jsonify({'success': False, 'error': f'Prediction error: {exc}'}), 500

    # Persist to MongoDB (non-fatal)
    saved = False
    if db_available and predictions_col is not None:
        try:
            predictions_col.insert_one({
                **inputs,
                'predicted_price_lakhs': result['predicted_price_lakhs'],
                'prediction_source':     result['prediction_source'],
                'city_tier':             result['city_tier'],
                'created_at':            datetime.now(timezone.utc),
            })
            saved = True
        except PyMongoError as exc:
            logger.warning("Could not save to MongoDB: %s", exc)

    return jsonify({
        'success':          True,
        'saved_to_db':      saved,
        # Explicit 'inputs' key with frontend-friendly field names so JS can
        # always do data.inputs.location / .area / .bedrooms without crashing.
        'inputs': {
            'location':  inputs['location'],
            'area':      inputs['area'],
            'bedrooms':  inputs['bedrooms'],
            'bathrooms': inputs['bathrooms'],
        },
        **result,
    })


@app.route('/api/history', methods=['GET'])
def api_history():
    """Return recent prediction history as JSON."""
    limit = min(request.args.get('limit', 50, type=int), 200)
    return jsonify({
        'success':      True,
        'db_available': db_available,
        'predictions':  _get_recent_history(limit),
    })


@app.route('/api/city-tier', methods=['GET'])
def api_city_tier():
    """Return the tier and multiplier for any city name."""
    city_name = request.args.get('city', '').strip().title()
    info = CITY_TIER_MAP.get(city_name, ("Unknown", None))
    return jsonify({
        'city':       city_name,
        'tier':       info[0],
        'multiplier': info[1],
        'in_tier_map': info[1] is not None,
        'in_dataset':  city_name in [c.title() for c in known_cities],
    })


# --------------------------------------------------------------------------
# Page Routes
# --------------------------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    return render_template(
        'index.html',
        db_available=db_available,
        model_available=model_available,
        locations=known_cities,  # backward compat
        known_cities=known_cities,
        known_states=known_states,
        model_metrics=model_metrics,
    )


@app.route('/history', methods=['GET'])
def view_history():
    predictions = _get_recent_history(50)
    return render_template('history.html', predictions=predictions, db_available=db_available)


@app.route('/compare', methods=['GET'])
def view_compare():
    return render_template('compare.html', db_available=db_available)


@app.route('/predict', methods=['POST'])
def predict():
    """Server-side form POST handler (PRG pattern fallback for non-AJAX forms)."""
    if not model_available:
        flash(f"Model unavailable: {model_load_error}", 'error')
        return redirect(url_for('index'))

    try:
        city   = request.form.get('location', request.form.get('city', '')).strip()
        state  = request.form.get('state', '').strip()
        bhk    = _parse_float(request.form.get('bedrooms', request.form.get('bhk', '2')))
        sqft   = _parse_float(request.form.get('area', request.form.get('size_in_sqft', '0')))
        prop   = request.form.get('property_type', 'Apartment').strip()
        furnish= request.form.get('furnished_status', 'Semi-furnished').strip()

        if not city or sqft <= 0 or bhk <= 0:
            flash("City, BHK, and area are required and must be positive.", 'error')
            return redirect(url_for('index'))

    except (ValueError, TypeError):
        flash("Please enter valid numeric values.", 'error')
        return redirect(url_for('index'))

    inputs = {
        "city":             city,
        "state":            state,
        "bhk":              bhk,
        "size_in_sqft":     sqft,
        "property_type":    prop,
        "furnished_status": furnish,
        "floor_no":         _parse_float(request.form.get('floor_no', '3')),
        "total_floors":     _parse_float(request.form.get('total_floors', '10')),
        "age_of_property":  _parse_float(request.form.get('age_of_property', '5')),
        "transport":        request.form.get('transport', 'Medium'),
        "parking":          request.form.get('parking', 'Yes'),
        "security":         request.form.get('security', 'Yes'),
        "availability":     request.form.get('availability', 'Ready_to_Move'),
        "owner_type":       request.form.get('owner_type', 'Builder'),
        "amenities":        request.form.get('amenities', ''),
    }

    try:
        result = predict_price(inputs)
        if "error" in result:
            flash(f"Prediction Error: {result['error']}", 'error')
            return redirect(url_for('index'))
    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        flash(f"Prediction Error: {exc}", 'error')
        return redirect(url_for('index'))

    if db_available and predictions_col is not None:
        try:
            predictions_col.insert_one({
                **inputs,
                'predicted_price_lakhs': result['predicted_price_lakhs'],
                'prediction_source':     result['prediction_source'],
                'city_tier':             result['city_tier'],
                'created_at':            datetime.now(timezone.utc),
            })
        except PyMongoError as exc:
            logger.warning("Could not save to MongoDB: %s", exc)
            flash("Prediction succeeded, but history save failed (DB unavailable).", 'warning')
    else:
        flash("Prediction succeeded, but history is unavailable (no DB).", 'warning')

    src = result['prediction_source']
    if src == "state_tier_fallback":
        flash(
            f"City '{city}' is not in our dataset. Used {result['city_tier']} pricing model "
            f"with {result['tier_multiplier']}× state-level adjustment.",
            'warning'
        )
    flash(result['formatted_price'], 'prediction')
    return redirect(url_for('index'))


# --------------------------------------------------------------------------
# Error Handlers
# --------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(_e):
    return render_template(
        'index.html',
        db_available=db_available,
        model_available=model_available,
        locations=known_cities,
        known_cities=known_cities,
        known_states=known_states,
        model_metrics=model_metrics,
    ), 404


@app.errorhandler(500)
def server_error(_e):
    flash("An unexpected server error occurred. Please try again.", 'error')
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)