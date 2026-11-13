from pathlib import Path
import pandas as pd
import numpy as np

# Automatically detect project directories whether script is in root or src/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR


def resolve_input_csv(filename: str = "healthcare-dataset-stroke-data.csv") -> Path:
    """Searches common project locations for the dataset CSV."""
    candidates = [
        Path(filename),                        # Current working directory
        PROJECT_ROOT / filename,               # Project root folder
        PROJECT_ROOT / "data" / filename,      # Project data/ subfolder
        SCRIPT_DIR / filename,                 # Same folder as preprocess.py
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()

    searched = "\n  - ".join(str(p.resolve()) for p in candidates)
    raise FileNotFoundError(
        f"Could not find '{filename}'. Searched in:\n  - {searched}"
    )


def clean_stroke_data(
    input_filename: str = "healthcare-dataset-stroke-data.csv",
    output_filename: str = "cleaned_stroke_data.csv",
    weka_nominal_prefixes: bool = False,
) -> pd.DataFrame:
    """Cleans the raw stroke dataset: imputes missing BMI values and removes ID."""
    input_path = resolve_input_csv(input_filename)
    df = pd.read_csv(input_path)
    print(f"Successfully read data from {input_path} (Shape: {df.shape})")

    # Handle 'N/A' values in 'bmi' and convert to numeric
    df["bmi"] = df["bmi"].replace("N/A", np.nan)
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

    # Impute missing 'bmi' values with the column mean
    bmi_mean = df["bmi"].mean()
    df["bmi"] = df["bmi"].fillna(bmi_mean)
    print(f"Handled missing 'bmi' values by filling with the mean ({bmi_mean:.2f}).")

    # Optional: Add prefix to ensure nominal interpretation in Weka
    if weka_nominal_prefixes:
        columns_to_nominal = ["heart_disease", "hypertension", "stroke"]
        for col in columns_to_nominal:
            df[col] = df[col].astype(str).apply(lambda x: f"{col}_{x}")
            print(f"Converted column '{col}' to nominal (categorical) type with prefix.")

    # Remove the 'id' column
    if "id" in df.columns:
        df = df.drop(columns=["id"])
        print("Removed the 'id' column.")

    # Save cleaned CSV in the same directory as the input dataset
    if output_filename:
        output_path = input_path.parent / Path(output_filename).name
        df.to_csv(output_path, index=False)
        print(f"Successfully saved cleaned data to {output_path}")

    return df


def spread_subsample_csv(df: pd.DataFrame, class_attribute: str = "stroke") -> pd.DataFrame:
    """Balances the target class (1:1 ratio) using random undersampling (SpreadSubsample)."""
    if df.empty or class_attribute not in df.columns:
        return df

    class_counts = df[class_attribute].value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    n_minority = class_counts[minority_class]
    n_majority = class_counts[majority_class]

    if n_majority <= n_minority:
        return df

    majority_indices = df[df[class_attribute] == majority_class].index
    np.random.seed(42)
    sampled_indices = np.random.choice(majority_indices, size=n_minority, replace=False)

    minority_df = df[df[class_attribute] == minority_class]
    sampled_majority_df = df.loc[sampled_indices]

    balanced_df = pd.concat([minority_df, sampled_majority_df])
    return balanced_df.reset_index(drop=True)


if __name__ == "__main__":
    cleaned_df = clean_stroke_data()
    balanced_df = spread_subsample_csv(cleaned_df, class_attribute="stroke")
    out_bal = PROJECT_ROOT / "balanced_stroke_data.csv"
    balanced_df.to_csv(out_bal, index=False)
    print(f"Saved balanced dataset to {out_bal} (Shape: {balanced_df.shape})")
