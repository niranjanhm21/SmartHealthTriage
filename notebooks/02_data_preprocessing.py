import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print("=" * 65)
print("     PHASE 3: DATA PREPROCESSING & CLEANING")
print("=" * 65)

# 1. Locate and Load Dataset
DATASET_DIR = os.path.join(os.getcwd(), 'datasets')
csv_files = glob.glob(os.path.join(DATASET_DIR, '*.csv'))
df = pd.read_csv(csv_files[0])

target_candidates = ['diseases', 'Disease', 'disease', 'prognosis', 'label']
target_col = next((c for c in target_candidates if c in df.columns), df.select_dtypes(include=['object', 'str']).columns[0])

print(f"Initial raw rows: {len(df):,}")

# 2. Drop Exact Duplicate Rows
df_clean = df.drop_duplicates().copy()
print(f"Rows after removing exact duplicates: {len(df_clean):,}")

# 3. Filter for Top-N Clinical Conditions (Ensures computational speed & valid SHAP)
TOP_N_CLASSES = 40
top_diseases = df_clean[target_col].value_counts().head(TOP_N_CLASSES).index.tolist()
df_filtered = df_clean[df_clean[target_col].isin(top_diseases)].copy()

print(f"Selected Top {TOP_N_CLASSES} conditions.")
print(f"Rows in filtered subset: {len(df_filtered):,}")

# 4. Remove all-zero symptom columns in the filtered subset
symptom_cols = [c for c in df_filtered.columns if c != target_col]
active_symptoms = [c for c in symptom_cols if df_filtered[c].sum() > 0]
dropped_symptoms = len(symptom_cols) - len(active_symptoms)

df_final = df_filtered[[target_col] + active_symptoms].copy()
print(f"Retained active symptoms: {len(active_symptoms)} (dropped {dropped_symptoms} inactive columns)")

# 5. Stratified Train/Test Split (80% Train, 20% Test)
X = df_final[active_symptoms]
y = df_final[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("-" * 65)
print(f"Training set records: {X_train.shape[0]:,} (Features: {X_train.shape[1]})")
print(f"Testing set records:  {X_test.shape[0]:,} (Features: {X_test.shape[1]})")

# 6. Save Processed Splits to datasets/
X_train.to_parquet(os.path.join(DATASET_DIR, 'X_train.parquet'))
X_test.to_parquet(os.path.join(DATASET_DIR, 'X_test.parquet'))
y_train.to_frame().to_parquet(os.path.join(DATASET_DIR, 'y_train.parquet'))
y_test.to_frame().to_parquet(os.path.join(DATASET_DIR, 'y_test.parquet'))

# Save symptom feature list
with open(os.path.join(DATASET_DIR, 'symptom_features.txt'), 'w') as f:
    for sym in active_symptoms:
        f.write(f"{sym}\n")

print("=" * 65)
print("SUCCESS: Preprocessed splits saved as Parquet in 'datasets/'")
print("=" * 65)