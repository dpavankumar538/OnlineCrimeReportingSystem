# predict.py
# ==============================================================
# Utility functions to load the model and make predictions.
# Imported by app.py for the /admin/prediction route.
# ==============================================================

import joblib
import numpy as np
import os

# Paths to saved model files
MODEL_DIR     = os.path.dirname(__file__)
MODEL_PATH    = os.path.join(MODEL_DIR, 'crime_model.pkl')
ENCODER_PATH  = os.path.join(MODEL_DIR, 'label_encoder.pkl')
DISTRICT_PATH = os.path.join(MODEL_DIR, 'district_encoder.pkl')


def load_model():
    """Load and return the trained model and encoders."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run ml_model/train_model.py first."
        )
    model            = joblib.load(MODEL_PATH)
    crime_encoder    = joblib.load(ENCODER_PATH)
    district_encoder = joblib.load(DISTRICT_PATH)
    return model, crime_encoder, district_encoder


def predict_crime(district, hour, month, report_count):
    """
    Predict the most likely crime type for given parameters.

    Parameters:
        district     (str): Name of the district
        hour         (int): Hour of day (0–23)
        month        (int): Month (1–12)
        report_count (int): Historical report count for that district

    Returns:
        str: Predicted crime type label
    """
    model, crime_encoder, district_encoder = load_model()

    # Encode district text → number
    # If district not seen during training, use 0 as fallback
    try:
        district_enc = district_encoder.transform([district])[0]
    except ValueError:
        district_enc = 0

    # Build feature array (must match training column order)
    features = np.array([[district_enc, hour, month, report_count]])

    # Make prediction
    pred_encoded = model.predict(features)[0]
    pred_label   = crime_encoder.inverse_transform([pred_encoded])[0]

    # Get prediction probabilities for each class
    proba        = model.predict_proba(features)[0]
    max_conf     = round(max(proba) * 100, 1)

    return {
        'crime_type'  : pred_label,
        'confidence'  : max_conf,
        'district'    : district
    }


# Quick test when run directly
if __name__ == '__main__':
    result = predict_crime('Bengaluru Urban', hour=22, month=1, report_count=45)
    print(f"\nTest Prediction:")
    print(f"  District   : {result['district']}")
    print(f"  Likely Crime: {result['crime_type']}")
    print(f"  Confidence : {result['confidence']}%")