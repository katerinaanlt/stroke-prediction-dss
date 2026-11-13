import pandas as pd
import numpy as np


def clean_stroke_data(
    input_filename: str = "healthcare-dataset-stroke-data.csv",
    output_filename: str = "cleaned_stroke_data.csv",
    weka_nominal_prefixes: bool = False,
) -> pd.DataFrame:
    """Cleans the raw stroke dataset: imputes missing BMI values and removes ID."""
    # 1. Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_filename)
    print(f"Successfully read data from {input_filename} (Shape: {df.shape})")

    # 2. Handle potential 'N/A' values in the 'bmi' column
    df["bmi"] = df["bmi"].replace("N/A", np.nan)

    # 3. Convert the 'bmi' column to numeric, handling errors
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

    # 4. Impute missing 'bmi' values with the mean
    bmi_mean = df["bmi"].mean()
    df["bmi"] = df["bmi"].fillna(bmi_mean)
    print(f"Handled missing 'bmi' values by filling with the mean ({bmi_mean:.2f}).")

    # 5. Optional: Add prefix to ensure nominal interpretation in Weka
    if weka_nominal_prefixes:
        columns_to_nominal = ["heart_disease", "hypertension", "stroke"]
        for col in columns_to_nominal:
            df[col] = df[col].astype(str).apply(lambda x: f"{col}_{x}")
            print(f"Converted column '{col}' to nominal (categorical) type with prefix.")

    # 6. Remove the 'id' column
    if "id" in df.columns:
        df = df.drop(columns=["id"])
        print("Removed the 'id' column.")

    # 7. Save the cleaned DataFrame to a new CSV file
    if output_filename:
        df.to_csv(output_filename, index=False)
        print(f"Successfully saved cleaned data to {output_filename}")

    return df


def spread_subsample_csv(df: pd.DataFrame, class_attribute: str = "stroke") -> pd.DataFrame:
    """Balances the target class (1:1 ratio) using random undersampling (SpreadSubsample)."""
    if df.empty:
        return pd.DataFrame()

    if class_attribute not in df.columns:
        print(f"Error: Class attribute '{class_attribute}' not found.")
        return df

    class_counts = df[class_attribute].value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    n_minority = class_counts[minority_class]
    n_majority = class_counts[majority_class]

    if n_majority <= n_minority:
        print("Class distribution is already balanced or the minority class is larger.")
        return df

    majority_indices = df[df[class_attribute] == majority_class].index
    np.random.seed(42)  # For reproducibility
    sampled_indices = np.random.choice(majority_indices, size=n_minority, replace=False)

    minority_df = df[df[class_attribute] == minority_class]
    sampled_majority_df = df.loc[sampled_indices]

    balanced_df = pd.concat([minority_df, sampled_majority_df])
    balanced_df = balanced_df.reset_index(drop=True)
    return balanced_df


if __name__ == "__main__":
    cleaned_df = clean_stroke_data(
        input_filename="healthcare-dataset-stroke-data.csv",
        output_filename="cleaned_stroke_data.csv",
        weka_nominal_prefixes=False,
    )
    balanced_df = spread_subsample_csv(cleaned_df, class_attribute="stroke")
    balanced_df.to_csv("balanced_stroke_data.csv", index=False)
    print(f"Saved balanced dataset to balanced_stroke_data.csv (Shape: {balanced_df.shape})")
    print(balanced_df["stroke"].value_counts())
