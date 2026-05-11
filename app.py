from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Allow requests from your website

# ── Load model files once at startup ──────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
model      = joblib.load(os.path.join(BASE_DIR, 'maize_yield_model.pkl'))
scaler     = joblib.load(os.path.join(BASE_DIR, 'maize_scaler.pkl'))

with open(os.path.join(BASE_DIR, 'model_metadata.json'), 'r') as f:
    meta = json.load(f)

FEATURES   = meta['features']
MODEL_NAME = meta['model_name']

print(f"✅ Model loaded: {MODEL_NAME}")

# ── Helper: validate input ranges ─────────────────────────
def validate_inputs(data):
    ranges = meta['feature_ranges']
    errors = []
    for feat in FEATURES:
        val = data.get(feat)
        if val is None:
            errors.append(f"Missing field: {feat}")
            continue
        low, high = ranges[feat]
        if not (low <= float(val) <= high):
            errors.append(f"{feat}={val} is out of range [{low} – {high}]")
    return errors

# ── Helper: generate alerts ───────────────────────────────
def get_alerts(N, P, K, pH, moisture, temperature, humidity):
    alerts = []
    if N < 150:           alerts.append("N is LOW — nitrogen deficiency (optimal: 190–280 mg/kg)")
    if N > 300:           alerts.append("N is HIGH — excess nitrogen (optimal: 190–280 mg/kg)")
    if P < 20:            alerts.append("P is LOW — phosphorus deficiency (optimal: 25–50 mg/kg)")
    if P > 60:            alerts.append(f"P is HIGH ({P} mg/kg) — NUTRIENT LOCKOUT RISK! (optimal: 25–50)")
    if K < 150:           alerts.append("K is LOW — potassium deficiency (optimal: 180–280 mg/kg)")
    if K > 300:           alerts.append("K is HIGH — excess potassium (optimal: 180–280 mg/kg)")
    if pH < 5.8:          alerts.append("pH too ACIDIC — may cause aluminium toxicity (optimal: 6.0–6.5)")
    if pH > 7.0:          alerts.append("pH too ALKALINE — nutrient lockout risk (optimal: 6.0–6.5)")
    if moisture < 50:     alerts.append("Moisture too LOW — drought stress risk")
    if moisture > 85:     alerts.append("Moisture too HIGH — waterlogging risk")
    if temperature < 20:  alerts.append("Temperature too LOW for maize growth")
    if temperature > 35:  alerts.append("Temperature too HIGH — heat stress risk")
    if humidity > 85:     alerts.append("Humidity too HIGH — fungal disease risk")
    return alerts

# ── Main prediction endpoint ───────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400

        # Validate
        errors = validate_inputs(data)
        if errors:
            return jsonify({'error': 'Validation failed', 'details': errors}), 422

        # Extract values
        N           = float(data['N'])
        P           = float(data['P'])
        K           = float(data['K'])
        pH          = float(data['pH'])
        moisture    = float(data['moisture'])
        temperature = float(data['temperature'])
        humidity    = float(data['humidity'])

        # Build input dataframe
        input_df = pd.DataFrame(
            [[N, P, K, pH, moisture, temperature, humidity]],
            columns=FEATURES
        )

        # Scale if linear model
        if MODEL_NAME in ['Linear Regression', 'Ridge Regression']:
            inp = scaler.transform(input_df)
        else:
            inp = input_df

        # Run inference
        prediction = float(model.predict(inp)[0])
        prediction = max(200.0, prediction)

        # Yield category
        if prediction < 2000:
            category    = 'Very Low'
            category_color = 'red'
            advice      = 'Critical failure. Multiple parameters need urgent correction.'
        elif prediction < 4000:
            category    = 'Low'
            category_color = 'orange'
            advice      = 'Poor yield. Review deficient or toxic nutrient levels.'
        elif prediction < 8000:
            category    = 'Medium'
            category_color = 'yellow'
            advice      = 'Moderate yield. Optimise nutrients and environmental conditions.'
        else:
            category    = 'High'
            category_color = 'green'
            advice      = 'Excellent conditions! Maintain current practices.'

        alerts = get_alerts(N, P, K, pH, moisture, temperature, humidity)

        return jsonify({
            'success'         : True,
            'prediction'      : round(prediction, 2),
            'prediction_tons' : round(prediction / 1000, 3),
            'category'        : category,
            'category_color'  : category_color,
            'advice'          : advice,
            'alerts'          : alerts,
            'model_used'      : MODEL_NAME,
            'inputs'          : {
                'N': N, 'P': P, 'K': K,
                'pH': pH, 'moisture': moisture,
                'temperature': temperature, 'humidity': humidity
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Health check endpoint ──────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status'     : 'online',
        'model'      : MODEL_NAME,
        'version'    : meta.get('version', 'v1'),
        'test_r2'    : meta.get('test_r2'),
        'test_mae'   : meta.get('test_mae'),
    })


# ── Root ───────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def root():
    return jsonify({'message': '🌽 Maize Yield Prediction API is running!'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
