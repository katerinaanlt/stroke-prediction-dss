import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from preprocess import clean_stroke_data, spread_subsample_csv

def build_preprocessor():
    numeric_features = ['age', 'avg_glucose_level', 'bmi']
    categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

def main():
    df = clean_stroke_data("data/healthcare-dataset-stroke-data.csv")
    balanced_df = spread_subsample_csv(df, class_attribute='stroke')

    X = balanced_df.drop(columns=['stroke'])
    y = balanced_df['stroke']

    # 80/20 split on the 498 rows gives exactly 100 test instances (50 False, 50 True), matching your report!
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=100, stratify=y, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    trained_pipelines = {}

    for name, classifier in models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', build_preprocessor()),
            ('classifier', classifier)
        ])
        
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc').mean()
        pipe.fit(X_train, y_train)
        
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        
        print(f"=== {name} Results ===")
        print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"  Mean CV AUC: {cv_auc:.4f}")
        print(f"  AUC: {roc_auc_score(y_test, y_prob):.4f}")
        print("  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['False', 'True']))
        
        trained_pipelines[name] = pipe

    # Save Random Forest pipeline for the Flask app
    os.makedirs("models", exist_ok=True)
    joblib.dump(trained_pipelines["Random Forest"], "models/stroke_pipeline.pkl")
    print("Saved Random Forest pipeline to models/stroke_pipeline.pkl")

if __name__ == "__main__":
    main()
