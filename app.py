import os
import joblib
import pandas as pd
import json
from flask import Flask, render_template, request

# ------------------------
# Load trained pipeline
# ------------------------
pipeline = joblib.load("models/final_pipeline.pkl")

# Load default values (for missing features)
with open("models/default_values.json", "r") as f:
    DEFAULT_VALUES = json.load(f)

ALL_FEATURES = list(DEFAULT_VALUES.keys())  # same order as training

# Only expose 12 fields in the form
FORM_FEATURES = [
    "ProductName", "OSVersion", "ProcessorCoreCount", "TotalPhysicalRAMMB",
    "PrimaryDiskCapacityMB", "PrimaryDiskType", "IsSecureBootEnabled",
    "IsVirtualDevice", "IsTouchEnabled", "IsPenCapable", "IsGamer",
    "RegionIdentifier"
]

app = Flask(__name__)

# ------------------------
# Routes
# ------------------------
@app.route("/")
def home():
    return render_template("index.html", features=FORM_FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    # Start with defaults
    input_data = DEFAULT_VALUES.copy()

    # Update with user inputs
    for feature in FORM_FEATURES:
        value = request.form.get(feature)
        try:
            input_data[feature] = float(value)
        except:
            input_data[feature] = value

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Make prediction
    prediction = pipeline.predict(df)[0]
    pred_proba = pipeline.predict_proba(df)[0][1]

    # Result message
    result = "⚠️ Threat Detected" if prediction == 1 else "✅ Safe"

    return render_template(
        "index.html",
        features=FORM_FEATURES,
        result=f"{result} (Probability: {pred_proba:.2f})"
    )

# ------------------------
# Run Flask
# ------------------------
if __name__ == "__main__":
    app.run(debug=True)
