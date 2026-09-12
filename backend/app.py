"""
app.py
-------
Flask web server for the House Price Prediction portfolio project.

- Loads the trained scikit-learn pipeline (backend/model.pkl) directly into
  memory with joblib so predictions are served without any per-request I/O.
- Connects to a local/remote MongoDB instance ('house_price_db.predictions')
  using PyMongo, with a bounded server-selection timeout so the app degrades
  gracefully (rather than hanging) if the database is unreachable.
- Provides both server-side rendered pages (with PRG pattern) and REST API
  endpoints for AJAX-powered frontend interactions.
"""

import os
import logging
from datetime import datetime, timezone

from flask import (
Flask, render_template, request, redirect, url_for,
flash, jsonify
)
# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# App / path configuration
# --------------------------------------------------------------------------
base_dir = os.path.abspath(os.path.dirname(__file__))
frontend_dir = os.path.abspath(os.path.join(base_dir, '..', 'frontend'))

app = Flask(
    __name__,
    template_folder=frontend_dir,
    static_folder=frontend_dir,
    static_url_path=''
)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'house_price_prediction_secret_key')

# --------------------------------------------------------------------------
# MongoDB connection (with timeout handling so the app never hangs)
# --------------------------------------------------------------------------
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
MONGO_TIMEOUT_MS = 3000

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=MONGO_TIMEOUT_MS)
    client.admin.command('ping')  # Force a round-trip to verify connectivity early
    db = client['house_price_db']
    predictions_col = db['predictions']
    db_available = True
    logger.info("Connected to MongoDB at %s", MONGO_URI)
except (ServerSelectionTimeoutError, PyMongoError) as exc:
    logger.warning("MongoDB connection failed (%s). Running without persistence/history.", exc)
    client = None
    predictions_col = None
    db_available = False

# --------------------------------------------------------------------------
# Load the trained model pipeline (with error handling for a missing file)
# --------------------------------------------------------------------------
model_path = os.path.join(base_dir, 'model.pkl')
model = None
model_load_error = None
known_locations = []

try:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"model.pkl not found at {model_path}. Run 'python train_model.py' first."
        )
    model = joblib.load(model_path)
    logger.info("Model loaded successfully from %s", model_path)

    # Extract known locations from the trained pipeline's OneHotEncoder
    try:
        preprocessor = model.named_steps.get('preprocessor')
        if preprocessor:
            cat_transformer = preprocessor.named_transformers_.get('cat')
            if cat_transformer:
                encoder = cat_transformer.named_steps.get('encoder')
                if encoder and hasattr(encoder, 'categories_'):
                    known_locations = sorted(encoder.categories_[0].tolist())
                    logger.info("Extracted %d known locations from model", len(known_locations))
    except Exception as loc_exc:
        logger.warning("Could not extract locations from model: %s", loc_exc)

except Exception as exc:
    model_load_error = str(exc)
    logger.error("Failed to load model: %s", model_load_error)


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def format_price(price_val):
    """Format a raw numeric price into a human-friendly Indian currency string."""
    try:
        price_val = float(price_val)
    except (ValueError, TypeError):
        return "₹N/A"

    if price_val >= 1_00_00_000:  # >= 1 Crore
        crore = price_val / 1_00_00_000
        return f"₹{crore:,.2f} Cr"
    elif price_val >= 1_00_000:  # >= 1 Lakh
        lakh = price_val / 1_00_000
        return f"₹{lakh:,.2f} Lakhs"
    else:
        return f"₹{price_val:,.2f}"


def get_recent_history(limit=5):
    """Safely fetch the most recent prediction records from MongoDB."""
    if not db_available or predictions_col is None:
        return []
    try:
        docs = list(predictions_col.find().sort('created_at', -1).limit(limit))
        for doc in docs:
            doc['_id'] = str(doc['_id'])
            created_at = doc.get('created_at')
            if isinstance(created_at, datetime):
                doc['timestamp'] = created_at.strftime('%Y-%m-%d %H:%M UTC')
            else:
                doc['timestamp'] = 'N/A'
            # Add formatted price
            doc['formatted_price'] = format_price(doc.get('predicted_price', 0))
        return docs
    except PyMongoError as exc:
        logger.warning("Could not fetch prediction history: %s", exc)
        return []


# --------------------------------------------------------------------------
# API Endpoints (JSON) — used by the AJAX frontend
# --------------------------------------------------------------------------
@app.route('/api/locations', methods=['GET'])
def api_locations():
    """Return the list of city/location names the model was trained on."""
    return jsonify({
        'success': True,
        'locations': known_locations
    })


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Accept JSON prediction request, return JSON result."""
    if model is None:
        return jsonify({
            'success': False,
            'error': f"Model unavailable: {model_load_error}"
        }), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'success': False, 'error': 'Invalid JSON payload'}), 400

    try:
        area = float(data.get('area', 0))
        bedrooms = int(float(data.get('bedrooms', 0)))
        bathrooms = int(float(data.get('bathrooms', 0)))
        location = str(data.get('location', '')).strip()

        if not location:
            return jsonify({'success': False, 'error': 'Location is required.'}), 400
        if area <= 0 or bedrooms <= 0 or bathrooms <= 0:
            return jsonify({'success': False, 'error': 'Area, bedrooms, and bathrooms must be positive.'}), 400
        if area > 1_000_000:
            return jsonify({'success': False, 'error': 'Area value seems unrealistic.'}), 400

    except (ValueError, TypeError):
        return jsonify({'success': False, 'error': 'Invalid numeric values.'}), 400

    try:
        fallback_used = False
        if location and known_locations and location not in known_locations:
            preds = []
            for loc in known_locations:
                df_in = pd.DataFrame([{
                    'area': area,
                    'bedrooms': bedrooms,
                    'bathrooms': bathrooms,
                    'location': loc
                }])
                preds.append(float(model.predict(df_in)[0]))
            pred_val = float(np.median(preds))
            fallback_used = True
        else:
            input_data = pd.DataFrame([{
                'area': area,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'location': location
            }])
            pred_val = float(model.predict(input_data)[0])
        pred_val = max(pred_val, 0)
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        return jsonify({'success': False, 'error': f'Prediction error: {exc}'}), 500

    # Persist to MongoDB (non-fatal)
    saved = False
    if db_available and predictions_col is not None:
        try:
            predictions_col.insert_one({
                'area': area,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'location': location,
                'predicted_price': pred_val,
                'created_at': datetime.now(timezone.utc)
            })
            saved = True
        except PyMongoError as exc:
            logger.warning("Could not save prediction to MongoDB: %s", exc)

    return jsonify({
        'success': True,
        'predicted_price': pred_val,
        'formatted_price': format_price(pred_val),
        'saved_to_db': saved,
        'fallback_used': fallback_used,
        'inputs': {
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'location': location
        }
    })


@app.route('/api/history', methods=['GET'])
def api_history():
    """Return recent prediction history as JSON."""
    limit = request.args.get('limit', 50, type=int)
    limit = min(limit, 200)
    history = get_recent_history(limit)
    return jsonify({
        'success': True,
        'db_available': db_available,
        'predictions': history
    })


# --------------------------------------------------------------------------
# Page Routes (server-side rendered)
# --------------------------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    return render_template(
        'index.html',
        db_available=db_available,
        model_available=model is not None,
        locations=known_locations
    )


@app.route('/history', methods=['GET'])
def view_history():
    """Route to view full search history on a dedicated page."""
    predictions = get_recent_history(50)
    return render_template('history.html', predictions=predictions, db_available=db_available)


@app.route('/compare', methods=['GET'])
def view_compare():
    """Route to view the dedicated property comparison matrix."""
    return render_template('compare.html', db_available=db_available)


@app.route('/predict', methods=['POST'])
def predict():
    """Server-side form POST handler (PRG pattern fallback)."""
    # ---- Guard: model must be loaded to serve predictions ----
    if model is None:
        flash(f"Model unavailable: {model_load_error}", 'error')
        return redirect(url_for('index'))

    # ---- Input validation ----
    try:
        area_raw = request.form.get('area', '').strip()
        bedrooms_raw = request.form.get('bedrooms', '').strip()
        bathrooms_raw = request.form.get('bathrooms', '').strip()
        location = request.form.get('location', '').strip()

        if not all([area_raw, bedrooms_raw, bathrooms_raw, location]):
            flash("All fields are required. Please fill in every field.", 'error')
            return redirect(url_for('index'))

        area = float(area_raw)
        bedrooms = int(float(bedrooms_raw))
        bathrooms = int(float(bathrooms_raw))

        if area <= 0 or bedrooms <= 0 or bathrooms <= 0:
            flash("Area, bedrooms, and bathrooms must be positive numbers.", 'error')
            return redirect(url_for('index'))

        if area > 1_000_000:
            flash("Area value seems unrealistic. Please double-check your input.", 'error')
            return redirect(url_for('index'))

    except (ValueError, TypeError):
        flash("Please enter valid numeric values for area, bedrooms, and bathrooms.", 'error')
        return redirect(url_for('index'))

    # ---- Model prediction ----
    try:
        fallback_used = False
        if location and known_locations and location not in known_locations:
            preds = []
            for loc in known_locations:
                df_in = pd.DataFrame([{
                    'area': area,
                    'bedrooms': bedrooms,
                    'bathrooms': bathrooms,
                    'location': loc
                }])
                preds.append(float(model.predict(df_in)[0]))
            pred_val = float(np.median(preds))
            fallback_used = True
        else:
            input_data = pd.DataFrame([{
                'area': area,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'location': location
            }])
            pred_val = float(model.predict(input_data)[0])
        pred_val = max(pred_val, 0)
        formatted_price = format_price(pred_val)
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        flash(f"Prediction Error: {exc}", 'error')
        return redirect(url_for('index'))

    # ---- Persist to MongoDB (non-fatal if it fails) ----
    if db_available and predictions_col is not None:
        try:
            predictions_col.insert_one({
                'area': area,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'location': location,
                'predicted_price': pred_val,
                'created_at': datetime.now(timezone.utc)
            })
        except PyMongoError as exc:
            logger.warning("Could not save prediction to MongoDB: %s", exc)
            flash("Prediction succeeded, but saving to history failed (DB unavailable).", 'warning')
    else:
        flash("Prediction succeeded, but history is unavailable (no database connection).", 'warning')

    if fallback_used:
        flash(f"City '{location}' not in dataset. Using an intelligent baseline estimate.", 'warning')
    flash(formatted_price, 'prediction')
    # PRG Pattern: always redirect after POST to avoid duplicate form resubmission on refresh
    return redirect(url_for('index'))


@app.errorhandler(404)
def not_found(_e):
    return render_template(
        'index.html',
        db_available=db_available,
        model_available=model is not None,
        locations=known_locations
    ), 404


@app.errorhandler(500)
def server_error(_e):
    flash("An unexpected server error occurred. Please try again.", 'error')
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True, port=5000)