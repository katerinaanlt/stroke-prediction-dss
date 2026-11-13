# Clinical Decision Support System: Stroke Risk Prediction

An end-to-end Machine Learning pipeline and containerized web-based Decision Support System (DSS) designed to predict patient stroke risk from clinical and demographic records.

## Overview
Medical datasets frequently suffer from missing values and severe class imbalance, which can cause standard classifiers to default to majority-class predictions. This project implements an automated preprocessing and class-balancing pipeline, benchmarks four supervised learning algorithms, generates comparative diagnostic plots, and serves the trained pipeline through a **Flask REST API** and interactive frontend containerized with **Docker**.

## Key Features
* **Automated Data Cleaning (`src/preprocess.py`):**
  * Parses raw clinical records (`5,110` instances, `12` attributes) from `healthcare-dataset-stroke-data.csv`.
  * Handles `201` missing/corrupted `bmi` entries (`N/A`) via numeric coercion and mean imputation (`28.89`).
  * Drops non-predictive identifiers (`id`) and exports `cleaned_stroke_data.csv`.
* **Class Imbalance Mitigation (`spread_subsample_csv`):**
  * Resolves severe target class skew (`249` positive stroke cases vs. `4,861` negative cases) using reproducible random undersampling (`SpreadSubsample`) to produce a balanced `1:1` training dataset (`498` total instances: `249` Stroke, `249` No Stroke) saved to `balanced_stroke_data.csv`.
* **Unified `scikit-learn` Pipeline & Benchmarking (`src/train.py`):**
  * Combines `SimpleImputer`, `StandardScaler` (for `age`, `avg_glucose_level`, `bmi`), and `OneHotEncoder` (for categorical clinical attributes) inside a single `ColumnTransformer` pipeline.
  * Evaluates **Logistic Regression**, **Random Forest**, **Decision Tree**, and **K-Nearest Neighbors (KNN)** using 10-fold Stratified Cross-Validation and an 80/20 stratified train/test split (`398` train, `100` test instances).
  * Automatically generates a 2x2 diagnostic figure (`model_evaluation.png`) displaying Confusion Matrices and ROC Curves, and exports the serialized model to `stroke_pipeline.pkl`.
* **Full-Stack Web Application & Docker Deployment (`app.py`, `index.html`, `Dockerfile`):**
  * **Backend:** Flask server exposing a `/predict` `POST` endpoint with CORS support and automatic path discovery.
  * **Frontend:** Responsive HTML/CSS/JavaScript clinical form using the Fetch API to display real-time stroke risk classifications and probability percentages.
  * **Containerization:** Production-ready `Dockerfile` using `python:3.11-slim` and `gunicorn` that runs the training pipeline during image build and serves the application on port `5000`.

---

## Model Performance & Benchmarks

### 1. Python `scikit-learn` Pipeline Evaluation (Stratified Test Split)
| Model | Test Accuracy | 10-Fold CV ROC-AUC | Test ROC-AUC |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | **81.00%** | **0.8071** | **0.8664** |
| **Decision Tree (CART)** | 77.00% | 0.6717 | 0.7700 |
| **Random Forest (100 trees)** | 75.00% | 0.7811 | 0.8142 |
| **K-Nearest Neighbors ($k=5$)** | 72.00% | 0.7560 | 0.7752 |

### 2. Full-Dataset 10-Fold Cross-Validation (Weka Benchmark: Random Forest vs. J48)
During exploratory benchmarking across all `498` balanced instances using 10-fold stratified cross-validation in Weka:
| Metric | J48 Decision Tree (C4.5) | Random Forest (100 Trees) |
| :--- | :---: | :---: |
| **Accuracy** | 72.29% | **73.69%** |
| **Kappa Statistic** | 0.4458 | **0.4739** |
| **Precision (Weighted)** | 0.723 | **0.740** |
| **Recall (Stroke Class)** | 0.723 | **0.795** |
| **F1-Score (Weighted)** | 0.723 | **0.736** |
| **ROC Area** | 0.753 | **0.811** |
| **PRC Area** | 0.698 | **0.806** |

> **Clinical Model Selection Note:** While J48 provides transparent rule-tree interpretability (highlighting `age > 47`, `heart_disease`, `bmi`, `hypertension`, and `avg_glucose_level` as primary split nodes), **Random Forest** was selected for deployment in `stroke_pipeline.pkl` due to its higher sensitivity/recall on positive stroke cases and stronger resistance to overfitting.

![Model Evaluation Plots](model_evalua<img width="2100" height="1500" alt="model_evaluation" src="https://github.com/user-attachments/assets/5fc14fe1-bde6-4311-95f4-598287222f77" />
tion.png)

---

## Project Structure

```text
stroke-prediction-dss/
├── src/
│   ├── preprocess.py                      # Data cleaning & SpreadSubsample class balancing
│   └── train.py                           # Model benchmarking, plotting & pipeline export
├── app/
│   ├── app.py                             # Flask REST API (/predict) & UI server
│   └── index.html                         # Clinical frontend interface (Fetch API)
├── healthcare-dataset-stroke-data.csv     # Raw clinical dataset (5,110 records)
├── cleaned_stroke_data.csv                # Generated cleaned dataset
├── balanced_stroke_data.csv               # Generated 1:1 balanced dataset (498 records)
├── stroke_pipeline.pkl                    # Serialized scikit-learn preprocessing + RF model
├── model_evaluation.png                   # Generated Confusion Matrices & ROC curves
├── Dockerfile                             # Automated build & Gunicorn server setup
├── .dockerignore                          # Excludes local caches & venv from Docker build
├── .gitignore                             # Git exclusions
└── requirements.txt                       # Python dependencies
```
*(Note: Path resolvers in `preprocess.py`, `train.py`, and `app.py` support both flat root layouts and `src/` + `app/` subdirectory layouts automatically.)*

---

## Getting Started

### Option 1: Run with Docker (Recommended)
Build the image (which automatically preprocesses the dataset and trains `stroke_pipeline.pkl`) and start the container:

```bash
docker build -t stroke-dss .
docker run -p 5000:5000 stroke-dss
```
Open **`http://localhost:5000`** in your browser.

### Option 2: Run Locally with Python
1. **Clone the repository and install dependencies:**
   ```bash
   git clone [https://github.com/YOUR-USERNAME/stroke-prediction-dss.git](https://github.com/YOUR-USERNAME/stroke-prediction-dss.git)
   cd stroke-prediction-dss
   pip install -r requirements.txt
   ```

2. **Run preprocessing and model training:**
   ```bash
   python src/train.py
   ```
   *(This generates `cleaned_stroke_data.csv`, `balanced_stroke_data.csv`, `model_evaluation.png`, and `stroke_pipeline.pkl`.)*

3. **Launch the Flask web server:**
   ```bash
   python app/app.py
   ```
   Navigate to **`http://localhost:5000`** in your browser.

---

## API Usage Example

You can also query the `/predict` endpoint directly via `curl` or any HTTP client:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 67,
    "gender": "Male",
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
  }'
```

**Sample JSON Response:**
```json
{
  "stroke_prediction": "Yes",
  "stroke_probability": 84.0
}
```
