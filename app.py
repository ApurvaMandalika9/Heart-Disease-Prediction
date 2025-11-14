from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os, json



app = Flask(__name__)
MODEL_PATH = os.getenv("MODEL_PATH", "models/heart_model.joblib")
model = joblib.load(MODEL_PATH)

with open("model_meta.json", "r") as f:
    meta = json.load(f)

THRESHOLD = float(meta["threshold"])
EXPECTED = meta["expected_feature_order"]

# Expected fields (same as training)
#EXPECTED = [
#    "age","sex","cp","trestbps","chol","fbs","restecg",
#    "thalach","exang","oldpeak","slope","ca","thal"
#]

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json()
        if isinstance(payload, dict) and "payload" in payload:
            row = payload["payload"]                 # single record
        elif isinstance(payload, dict):
            row = payload                            # also allow raw dict
        else:
            return jsonify({"error": "Invalid JSON body"}), 400

        # Ensure all expected keys exist
        missing = [c for c in EXPECTED if c not in row]
        if missing:
            return jsonify({"error": f"Missing keys: {missing}"}), 400

        X = pd.DataFrame([row], columns=EXPECTED)
        if hasattr(model, "predict_proba"):
            p = float(model.predict_proba(X)[0,1])
            y = int(p >= THRESHOLD)
            return jsonify({"prediction": y, "probability": p})
        else:
            y = model.predict(X)[0]
            return jsonify({"prediction": int(y)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # For local dev; Docker will use gunicorn
    app.run(host="0.0.0.0", port=8000, debug=True)
