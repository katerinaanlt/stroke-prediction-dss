import sys
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Automatically detect project root whether app.py is in root or app/ subfolder
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name in ("app", "src") else SCRIPT_DIR

# Add src/ and root to sys.path so 'import train' works from anywhere
for folder in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if folder.is_dir() and str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


def resolve_model_path() -> Path:
    """Finds stroke_pipeline.pkl or triggers training if it doesn't exist yet."""
    candidates = [
        PROJECT_ROOT / "stroke_pipeline.pkl",
        PROJECT_ROOT / "models" / "stroke_pipeline.pkl",
        SCRIPT_DIR / "stroke_pipeline.pkl",
    ]
    for path in candidates:
        if path.is_file():
            return path

    # Train automatically if model file is missing
    print("Model not found. Running training pipeline...")
    import train
    train.main()
    return PROJECT_ROOT / "stroke_pipeline.pkl"


def resolve_html_path() -> Path:
    """Finds index.html across root, app/, or app/static/."""
    candidates = [
        SCRIPT_DIR / "index.html",
        SCRIPT_DIR / "static" / "index.html",
        PROJECT_ROOT / "index.html",
        PROJECT_ROOT / "app" / "index.html",
        PROJECT_ROOT / "app" / "static" / "index.html",
    ]
    for path in candidates:
        if path.is_file():
            return path

    searched = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find 'index.html'. Searched in:\n  - {searched}")


MODEL_PATH = resolve_model_path()
pipeline = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return send_file(resolve_html_path())


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
