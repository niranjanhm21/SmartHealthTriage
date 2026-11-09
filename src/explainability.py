import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to save plots cleanly
import matplotlib.pyplot as plt

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'random_forest_model.joblib')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'label_encoder.joblib')
FEATURE_PATH = os.path.join(BASE_DIR, 'datasets', 'symptom_features.txt')

# Load artifacts
_rf_model = None
_label_encoder = None
_explainer = None
_features = None

def get_explainer():
    global _rf_model, _label_encoder, _explainer, _features
    if _rf_model is None:
        _rf_model = joblib.load(MODEL_PATH)
        _label_encoder = joblib.load(ENCODER_PATH)
        with open(FEATURE_PATH, 'r') as f:
            _features = [line.strip() for line in f if line.strip()]
        # TreeExplainer is fast and exact for tree ensembles
        _explainer = shap.TreeExplainer(_rf_model)
    return _rf_model, _label_encoder, _explainer, _features

def explain_prediction(patient_df, top_disease_idx=None, top_k=5):
    """
    Computes SHAP values for a single patient vector.
    Returns:
      dict with:
        - top_disease: disease name
        - positive_contributors: list of (feature_name, shap_value) pushing towards this disease
        - negative_contributors: list of (feature_name, shap_value) pushing away from this disease
        - shap_values: raw array for plotting
    """
    model, encoder, explainer, feature_names = get_explainer()
    
    # Ensure patient_df columns match training feature names
    if isinstance(patient_df, pd.DataFrame):
        patient_df_matched = patient_df[feature_names]
        patient_vector = patient_df_matched.values
    else:
        patient_vector = np.array(patient_df).reshape(1, -1)
        patient_df_matched = pd.DataFrame(patient_vector, columns=feature_names)
        
    probs = model.predict_proba(patient_df_matched)[0]
    if top_disease_idx is None:
        top_disease_idx = int(np.argmax(probs))
        
    disease_name = encoder.inverse_transform([top_disease_idx])[0]
    
    # Calculate TreeSHAP values
    shap_vals = explainer.shap_values(patient_vector)
    
    # Handle SHAP output format (list of arrays per class, or 3D array [samples, features, classes])
    if isinstance(shap_vals, list):
        class_shap = shap_vals[top_disease_idx][0]
    elif len(shap_vals.shape) == 3:
        class_shap = shap_vals[0, :, top_disease_idx]
    else:
        class_shap = shap_vals[0]
        
    # Pair feature names with SHAP contributions
    contribs = list(zip(feature_names, class_shap, patient_vector[0]))
    
    # Positive contributors: features present and supporting this condition
    pos = sorted([c for c in contribs if c[1] > 0], key=lambda x: x[1], reverse=True)[:top_k]
    # Negative contributors: absent features that would otherwise support it, or features arguing against it
    neg = sorted([c for c in contribs if c[1] < 0], key=lambda x: x[1])[:top_k]
    
    return {
        "top_disease": disease_name,
        "disease_prob": float(probs[top_disease_idx]),
        "positive_contributors": [(item[0], float(item[1]), int(item[2])) for item in pos],
        "negative_contributors": [(item[0], float(item[1]), int(item[2])) for item in neg],
        "raw_shap": class_shap,
        "features": feature_names,
        "patient_vector": patient_vector[0]
    }

def save_patient_shap_plot(explanation, output_filepath):
    """Generates a clear horizontal bar chart showing top SHAP symptom contributions."""
    pos = explanation["positive_contributors"]
    neg = explanation["negative_contributors"]
    
    combined = pos + neg
    if not combined:
        return
    
    # Sort for bar plot
    combined = sorted(combined, key=lambda x: x[1])
    names = [c[0] for c in combined]
    values = [c[1] for c in combined]
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
    
    plt.figure(figsize=(9, 5))
    y_pos = np.arange(len(names))
    plt.barh(y_pos, values, color=colors, align='center')
    plt.yticks(y_pos, names, fontsize=9)
    plt.xlabel('SHAP Impact on Disease Prediction (Log-Odds / Contribution)', fontsize=10)
    plt.title(f'Feature Attribution for: {explanation["top_disease"]} (Confidence: {explanation["disease_prob"]*100:.1f}%)', fontsize=11, fontweight='bold')
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=200)
    plt.close()