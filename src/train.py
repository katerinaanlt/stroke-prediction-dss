import joblib
import matplotlib
matplotlib.use("Agg")  # Headless backend so Docker builds never crash on plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)

from preprocess import clean_stroke_data, spread_subsample_csv


def build_preprocessor() -> ColumnTransformer:
    numeric_features = ["age", "avg_glucose_level", "bmi"]
    categorical_features = [
        "gender",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def main():
    # 1. Clean and balance the dataset (498 rows: 249 Stroke, 249 No Stroke)
    cleaned_df = clean_stroke_data(
        input_filename="healthcare-dataset-stroke-data.csv",
        output_filename="cleaned_stroke_data.csv",
        weka_nominal_prefixes=False,
    )
    balanced_df = spread_subsample_csv(cleaned_df, class_attribute="stroke")
    balanced_df.to_csv("balanced_stroke_data.csv", index=False)

    X = balanced_df.drop(columns=["stroke"])
    y = balanced_df["stroke"].astype(int)

    # 80/20 stratified split (398 train, 100 test: 50 False, 50 True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    trained_pipelines = {}
    confusion_matrices = {}
    roc_data = {}

    for name, clf in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", clf),
            ]
        )

        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        print(f"\n{name} Results:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Mean CV AUC: {cv_auc:.4f}")
        print(f"  AUC: {auc:.4f}")
        print("  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["False", "True"]))

        trained_pipelines[name] = pipe
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, auc)

    # 2. Save evaluation plot (Confusion Matrices + ROC Curves)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_models = ["Logistic Regression", "Decision Tree", "Random Forest"]
    coords = [(0, 0), (0, 1), (1, 0)]

    for (r, c), m_name in zip(coords, plot_models):
        sns.heatmap(
            confusion_matrices[m_name],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Stroke", "Stroke"],
            yticklabels=["No Stroke", "Stroke"],
            ax=axes[r, c],
        )
        axes[r, c].set_title(f"{m_name} Confusion Matrix")
        axes[r, c].set_xlabel("Predicted")
        axes[r, c].set_ylabel("Actual")

    # Bottom-right subplot: ROC Curves
    ax_roc = axes[1, 1]
    for m_name, (fpr, tpr, auc_val) in roc_data.items():
        ax_roc.plot(fpr, tpr, label=f"{m_name} (AUC = {auc_val:.2f})")
    ax_roc.plot([0, 1], [0, 1], "k--")
    ax_roc.set_title("ROC Curves")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("model_evaluation.png", dpi=150)
    print("Saved evaluation plots to model_evaluation.png")

    # 3. Export the Random Forest pipeline for the Flask web application
    joblib.dump(trained_pipelines["Random Forest"], "stroke_pipeline.pkl")
    print("Saved Random Forest pipeline to stroke_pipeline.pkl")


if __name__ == "__main__":
    main()
