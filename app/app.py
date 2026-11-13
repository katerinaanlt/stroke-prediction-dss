import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static")
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "stroke_pipeline.pkl")
pipeline = joblib.load(MODEL_PATH)

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([{
            "gender": str(data["gender"]),
            "age": float(data["age"]),
            "hypertension": int(data["hypertension"]),
            "heart_disease": int(data["heart_disease"]),
            "ever_married": str(data["ever_married"]),
            "work_type": str(data["work_type"]),
            "Residence_type": str(data["Residence_type"]),
            "avg_glucose_level": float(data["avg_glucose_level"]),
            "bmi": float(data["bmi"]),
            "smoking_status": str(data["smoking_status"])
        }])

        prediction = int(pipeline.predict(input_df)[0])
        probability = float(pipeline.predict_proba(input_df)[0][1])

        return jsonify({
            "stroke_prediction": "Yes" if prediction == 1 else "No",
            "stroke_probability": round(probability * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
