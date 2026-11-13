import os
import pandas as pd
import numpy as np

def clean_stroke_data(input_filepath: str, output_filepath: str = None) -> pd.DataFrame:
    """Cleans raw stroke dataset: drops id, imputes BMI with mean, formats columns."""
    df = pd.read_csv(input_filepath)
    
    # 1. Handle 'N/A' values in 'bmi' and convert to numeric
    df['bmi'] = df['bmi'].replace('N/A', np.nan)
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    
    # 2. Impute missing 'bmi' values with the column mean
    bmi_mean = df['bmi'].mean()
    df['bmi'] = df['bmi'].fillna(bmi_mean)
    
    # 3. Drop non-predictive 'id' column (and rare 'Other' gender row if present)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    df = df[df['gender'] != 'Other']
    
    if output_filepath:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        df.to_csv(output_filepath, index=False)
        
    return df

def spread_subsample_csv(df: pd.DataFrame, class_attribute: str = 'stroke') -> pd.DataFrame:
    """Balances dataset using random undersampling (1:1 majority-to-minority ratio)."""
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
    np.random.seed(42)  # For reproducibility
    sampled_indices = np.random.choice(majority_indices, size=n_minority, replace=False)

    minority_df = df[df[class_attribute] == minority_class]
    sampled_majority_df = df.loc[sampled_indices]

    balanced_df = pd.concat([minority_df, sampled_majority_df]).sample(frac=1, random_state=42)
    return balanced_df.reset_index(drop=True)

if __name__ == "__main__":
    cleaned_df = clean_stroke_data("data/healthcare-dataset-stroke-data.csv", "data/cleaned_stroke_data.csv")
    balanced_df = spread_subsample_csv(cleaned_df, class_attribute="stroke")
    balanced_df.to_csv("data/balanced_stroke_data.csv", index=False)
    print(f"Balanced dataset saved. Shape: {balanced_df.shape}")
    print(balanced_df['stroke'].value_counts())
