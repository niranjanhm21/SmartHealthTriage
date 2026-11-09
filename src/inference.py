"""
End-to-End Triage Inference Pipeline.
SmartHealth Triage System.

Coordinates:
1. Feature vectorization (178 binary symptom features)
2. ML Model inference (Top-K conditions + probabilities)
3. Clinical Safety Layer & Red-Flag deterministic override
4. SHAP Model Explainability
"""

import os
from typing import List, Dict, Any, Optional, Union, Tuple
import joblib
import numpy as np
import pandas as pd

from src.safety_layer import evaluate_triage, EMERGENCY_RED_FLAGS, TRIAGE_GUIDANCE, MEDICAL_DISCLAIMER
from src.explainability import explain_prediction

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.joblib')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'label_encoder.joblib')
FEATURE_PATH = os.path.join(BASE_DIR, 'datasets', 'symptom_features.txt')

_model = None
_label_encoder = None
_symptom_features = None
_feature_to_idx = None


def load_artifacts():
    """Lazy loads ML model, label encoder, and feature names."""
    global _model, _label_encoder, _symptom_features, _feature_to_idx
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _label_encoder = joblib.load(ENCODER_PATH)
        with open(FEATURE_PATH, 'r', encoding='utf-8') as f:
            _symptom_features = [line.strip() for line in f if line.strip()]
        
        # Build flexible mapping for symptom lookups
        _feature_to_idx = {}
        for idx, feat in enumerate(_symptom_features):
            norm_space = feat.lower().replace("_", " ").strip()
            norm_under = feat.lower().replace(" ", "_").strip()
            _feature_to_idx[feat] = idx
            _feature_to_idx[norm_space] = idx
            _feature_to_idx[norm_under] = idx
            
    return _model, _label_encoder, _symptom_features, _feature_to_idx


def get_available_symptoms() -> List[str]:
    """Returns the list of 178 standardized symptom features."""
    _, _, symptom_features, _ = load_artifacts()
    return list(symptom_features)


def get_emergency_red_flags() -> Dict[str, str]:
    """Returns the dictionary of standardized clinical red-flag triggers."""
    return dict(EMERGENCY_RED_FLAGS)


def build_feature_vector(reported_symptoms: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Transforms a list of reported symptom strings into a 178-dimensional binary vector.
    """
    _, _, symptom_features, feature_to_idx = load_artifacts()
    n_features = len(symptom_features)
    vector = np.zeros(n_features, dtype=int)
    
    for sym in reported_symptoms:
        if not sym:
            continue
        raw_str = str(sym).strip()
        norm_space = raw_str.lower().replace("_", " ")
        norm_under = raw_str.lower().replace(" ", "_")
        
        if raw_str in feature_to_idx:
            vector[feature_to_idx[raw_str]] = 1
        elif norm_space in feature_to_idx:
            vector[feature_to_idx[norm_space]] = 1
        elif norm_under in feature_to_idx:
            vector[feature_to_idx[norm_under]] = 1
            
    df_vector = pd.DataFrame([vector], columns=symptom_features)
    return vector, df_vector


def predict_triage_assessment(
    reported_symptoms: List[str],
    user_red_flags: Optional[List[str]] = None,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Executes full clinical triage assessment:
    1. Vectorizes reported symptoms to the 178 binary feature dimensions.
    2. Computes top-k conditions with probability distributions.
    3. Evaluates deterministic clinical safety overrides.
    4. Computes TreeSHAP local feature attribution for explainability.
    5. Returns unified structured clinical assessment.

    Parameters:
    -----------
    reported_symptoms:
        List of reported symptom names.
    user_red_flags:
        List of red-flag symptoms or emergency complaints.
    top_k:
        Number of top conditions to retrieve (default: 3).

    Returns:
    --------
    dict containing:
        - top_conditions (list of dicts with condition, probability, confidence_percent)
        - primary_condition (str)
        - primary_confidence (float)
        - triage_level ('LOW', 'MODERATE', 'HIGH', 'EMERGENCY')
        - safety_override (bool)
        - red_flag_alerts (list of str)
        - clinical_guidance (str)
        - clinical_disclaimer (str)
        - shap_explanation (dict with positive and negative contributors)
        - active_features_count (int)
    """
    user_red_flags = user_red_flags or []
    reported_symptoms = reported_symptoms or []
    
    model, label_encoder, symptom_features, _ = load_artifacts()
    
    # 1. Transform symptoms into 178-dimensional binary model vector
    vector, df_vector = build_feature_vector(reported_symptoms)
    
    # 2. Predict Top-K probable conditions with confidence percentages
    probabilities = model.predict_proba(df_vector)[0]
    sorted_class_indices = np.argsort(probabilities)[::-1][:top_k]
    
    top_conditions = []
    preds_for_safety = []
    
    for cls_idx in sorted_class_indices:
        condition_name = label_encoder.inverse_transform([cls_idx])[0]
        prob = float(probabilities[cls_idx])
        conf_pct = round(prob * 100, 2)
        top_conditions.append({
            "condition": condition_name,
            "probability": prob,
            "confidence_percent": conf_pct,
            "confidence_formatted": f"{conf_pct:.1f}%"
        })
        preds_for_safety.append((condition_name, prob))
        
    primary_idx = int(sorted_class_indices[0])
    primary_condition = top_conditions[0]["condition"]
    primary_confidence = top_conditions[0]["confidence_percent"]
    
    # 3. Pass through clinical safety layer
    triage_eval = evaluate_triage(
        predicted_conditions_with_probs=preds_for_safety,
        reported_symptoms=reported_symptoms,
        user_red_flags=user_red_flags
    )
    
    # 4. Extract SHAP feature explanations
    shap_raw = explain_prediction(df_vector, top_disease_idx=primary_idx, top_k=5)
    
    positive_contributors = [
        {
            "symptom": feat,
            "shap_value": round(float(val), 4),
            "is_present": bool(state == 1)
        }
        for feat, val, state in shap_raw.get("positive_contributors", [])
    ]
    
    negative_contributors = [
        {
            "symptom": feat,
            "shap_value": round(float(val), 4),
            "is_present": bool(state == 1)
        }
        for feat, val, state in shap_raw.get("negative_contributors", [])
    ]
    
    return {
        "top_conditions": top_conditions,
        "primary_condition": primary_condition,
        "primary_confidence": primary_confidence,
        "triage_level": triage_eval["triage_level"],
        "safety_override": triage_eval["safety_override"],
        "red_flag_alerts": triage_eval["override_reasons"],
        "clinical_guidance": triage_eval["guidance"],
        "clinical_disclaimer": triage_eval["disclaimer"],
        "shap_explanation": {
            "top_disease": shap_raw.get("top_disease"),
            "disease_prob": shap_raw.get("disease_prob"),
            "positive_contributors": positive_contributors,
            "negative_contributors": negative_contributors,
        },
        "active_features_count": int(np.sum(vector)),
        "reported_symptoms": reported_symptoms,
        "user_red_flags": user_red_flags
    }
