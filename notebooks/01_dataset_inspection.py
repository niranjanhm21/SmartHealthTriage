import os
import glob
import pandas as pd
import numpy as np

# Locate CSV inside datasets/
DATASET_DIR = os.path.join(os.getcwd(), 'datasets')
csv_files = glob.glob(os.path.join(DATASET_DIR, '*.csv'))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV file found in '{DATASET_DIR}'. "
        "Please make sure your Kaggle dataset CSV is placed inside the datasets/ folder."
    )

csv_path = csv_files[0]
print(f"Loading dataset: {os.path.basename(csv_path)}...")
df = pd.read_csv(csv_path)

# Automatically identify the disease target column
target_candidates = ['diseases', 'Disease', 'disease', 'prognosis', 'label']
target_col = next((c for c in target_candidates if c in df.columns), df.select_dtypes(include=['object']).columns[0])
symptom_cols = [c for c in df.columns if c != target_col]

print("\n" + "=" * 65)
print("             DATASET AUDIT REPORT: SMARTHEALTH TRIAGE")
print("=" * 65)
print(f"Total Patient Records (Rows): {df.shape[0]:,}")
print(f"Total Columns:                {df.shape[1]}")
print(f"Identified Disease Column:    '{target_col}'")
print(f"Unique Disease Classes:       {df[target_col].nunique()}")
print(f"Total Symptom Features:       {len(symptom_cols)}")
print(f"Missing Values (NaNs):        {df.isnull().sum().sum()}")

# Duplicate Analysis
exact_dupes = df.duplicated().sum()
conflicting_dupes = df.duplicated(subset=symptom_cols).sum() - exact_dupes
print(f"Exact Duplicate Rows:         {exact_dupes:,} ({(exact_dupes / len(df)) * 100:.2f}%)")
print(f"Conflicting Symptom Vectors:  {conflicting_dupes:,}")

# Matrix Sparsity and Active Symptoms
symptom_matrix = df[symptom_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
symptoms_per_row = symptom_matrix.sum(axis=1)
sparsity = 1.0 - (symptom_matrix.values.sum() / (df.shape[0] * len(symptom_cols)))
print(f"Matrix Sparsity:              {sparsity * 100:.2f}% (Percentage of zeros)")
print(f"Mean Symptoms per Record:     {symptoms_per_row.mean():.2f}")
print(f"Min Symptoms per Record:      {symptoms_per_row.min():.0f}")
print(f"Max Symptoms per Record:      {symptoms_per_row.max():.0f}")
print("=" * 65)

print("\nTop 10 Most Frequent Diseases:")
print(df[target_col].value_counts().head(10).to_string())

print("\nTop 10 Least Frequent Diseases:")
print(df[target_col].value_counts().tail(10).to_string())
print("=" * 65 + "\n")