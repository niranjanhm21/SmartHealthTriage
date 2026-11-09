import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from src.explainability import explain_prediction, save_patient_shap_plot

print("=" * 65)
print("     VERIFYING TASK 1: SHAP EXPLAINABILITY")
print("=" * 65)

# Load test set
DATASET_DIR = os.path.join(os.getcwd(), 'datasets')
X_test = pd.read_parquet(os.path.join(DATASET_DIR, 'X_test.parquet'))

# Pick a patient with active symptoms
sample_idx = 0
for idx in range(len(X_test)):
    if X_test.iloc[idx].sum() >= 3:
        sample_idx = idx
        break

patient_sample = X_test.iloc[[sample_idx]]
active_symptoms = [col for col in patient_sample.columns if patient_sample.iloc[0][col] == 1]
print(f"Test Patient Index: {sample_idx}")
print(f"Active Symptoms ({len(active_symptoms)}): {', '.join(active_symptoms[:6])}")

# Compute explanation
explanation = explain_prediction(patient_sample)

print("\n--- SHAP Prediction Explanation ---")
print(f"Predicted Condition: {explanation['top_disease']}")
print(f"Model Confidence:    {explanation['disease_prob']*100:.2f}%")

print("\nTop Contributing Symptoms (Pushing TOWARDS Condition):")
for feat, val, state in explanation['positive_contributors']:
    status = "(Present)" if state == 1 else "(Absent)"
    print(f"  + {feat:<30} {status:<10} SHAP: +{val:.4f}")

print("\nTop Counter Symptoms (Pushing AWAY from Condition):")
for feat, val, state in explanation['negative_contributors']:
    status = "(Present)" if state == 1 else "(Absent)"
    print(f"  - {feat:<30} {status:<10} SHAP: {val:.4f}")

# Save visualization
MODELS_DIR = os.path.join(os.getcwd(), 'models')
output_plot = os.path.join(MODELS_DIR, 'shap_test_patient.png')
save_patient_shap_plot(explanation, output_plot)

print(f"\n[Artifact Saved] SHAP bar chart saved to: {output_plot}")
print("=" * 65)