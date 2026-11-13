from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
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

from preprocess import PROJECT_ROOT, clean_stroke_data, spread_subsample_csv


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
    cleaned_df = clean_stroke_data(
        input_filename="healthcare-dataset-stroke-data.csv",
        output_filename="cleaned_stroke_data.csv",
        weka_nominal_prefixes=False,
    )
    balanced_df = spread_subsample_csv(cleaned_df, class_attribute="stroke")

    balanced_path = PROJECT_ROOT / "balanced_stroke_data.csv"
    balanced_df.to_csv(balanced_path, index=False)

    X = balanced_df.drop(columns=["stroke"])
    y = balanced_df["stroke"].astype(int)

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

    ax_roc = axes[1, 1]
    for m_name, (fpr, tpr, auc_val) in roc_data.items():
        ax_roc.plot(fpr, tpr, label=f"{m_name} (AUC = {auc_val:.2f})")
    ax_roc.plot([0, 1], [0, 1], "k--")
    ax_roc.set_title("ROC Curves")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")

    plt.tight_layout()
    plot_path = PROJECT_ROOT / "model_evaluation.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved evaluation plots to {plot_path}")

    model_path = PROJECT_ROOT / "stroke_pipeline.pkl"
    joblib.dump(trained_pipelines["Random Forest"], model_path)
    print(f"Saved Random Forest pipeline to {model_path}")


if __name__ == "__main__":
    main()
