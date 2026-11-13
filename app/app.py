import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "stroke_pipeline.pkl")

# Train automatically if stroke_pipeline.pkl hasn't been generated yet
if not os.path.exists(MODEL_PATH):
    import train
    train.main()

pipeline = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return send_file(os.path.join(BASE_DIR, "index.html"))


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame(
            [
                {
                    "gender": str(data.get("gender", "Male")),
                    "age": float(data.get("age", 50.0)),
                    "hypertension": int(data.get("hypertension", 0)),
                    "heart_disease": int(data.get("heart_disease", 0)),
                    "ever_married": str(data.get("ever_married", "Yes")),
                    "work_type": str(data.get("work_type", "Private")),
                    "Residence_type": str(data.get("Residence_type", "Urban")),
                    "avg_glucose_level": float(data.get("avg_glucose_level", 100.0)),
                    "bmi": float(data.get("bmi", 28.0)),
                    "smoking_status": str(data.get("smoking_status", "never smoked")),
                }
            ]
        )

        prediction = int(pipeline.predict(input_df)[0])
        probability = float(pipeline.predict_proba(input_df)[0][1])

        return jsonify(
            {
                "stroke_prediction": "Yes" if prediction == 1 else "No",
                "stroke_probability": round(probability * 100, 2),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
