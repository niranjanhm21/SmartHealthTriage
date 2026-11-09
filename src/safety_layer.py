"""
Clinical Safety Layer & Deterministic Triage Rule Engine.
SmartHealth Triage System.

Enforces deterministic clinical red-flag overrides to guarantee patient safety
regardless of machine learning model probability distributions.
"""

from typing import List, Tuple, Dict, Any, Union, Optional

# 1. Definitive Emergency Clinical Red Flags
EMERGENCY_RED_FLAGS: Dict[str, str] = {
    "severe_chest_pain": "Severe crushing chest pain, pressure, or tightness radiating to left arm or jaw (suspected acute coronary syndrome).",
    "difficulty_breathing": "Severe shortness of breath, respiratory distress, gasping, or inability to speak in sentences.",
    "cyanosis": "Bluish or grayish discoloration of the lips, face, or nail beds (acute hypoxia / poor perfusion).",
    "loss_of_consciousness": "Loss of consciousness, syncope, sudden unresponsiveness, or severely altered mental status.",
    "sudden_paralysis": "Sudden weakness, facial drooping, numbness, or paralysis on one side of the body (suspected stroke / CVA).",
    "hemoptysis": "Coughing up blood or vomiting significant amounts of fresh/coffee-ground blood (gastrointestinal or pulmonary hemorrhage).",
    "severe_abdominal_trauma": "Severe acute abdominal trauma, rigid/board-like abdomen, or sudden excruciating abdominal agony."
}

# 2. Baseline Clinical Urgency by Condition (for ML model predictions when no red flags are present)
CONDITION_SEVERITY_MAP: Dict[str, str] = {
    # Emergency conditions (acute life/organ threats)
    "gastrointestinal hemorrhage": "EMERGENCY",
    "acute pancreatitis": "EMERGENCY",
    "spontaneous abortion": "EMERGENCY",

    # High urgency conditions (requires rapid clinical intervention within 12-24h)
    "pneumonia": "HIGH",
    "cholecystitis": "HIGH",
    "diverticulitis": "HIGH",
    "concussion": "HIGH",
    "hypoglycemia": "HIGH",
    "liver disease": "HIGH",
    "acute bronchiolitis": "HIGH",

    # Moderate urgency conditions (outpatient medical evaluation within 2-3 days)
    "acute bronchitis": "MODERATE",
    "strep throat": "MODERATE",
    "cystitis": "MODERATE",
    "infectious gastroenteritis": "MODERATE",
    "noninfectious gastroenteritis": "MODERATE",
    "pyogenic skin infection": "MODERATE",
    "esophagitis": "MODERATE",
    "injury to the arm": "MODERATE",
    "injury to the leg": "MODERATE",
    "sprain or strain": "MODERATE",
    "peripheral nerve disorder": "MODERATE",
    "spinal stenosis": "MODERATE",
    "spondylosis": "MODERATE",
    "complex regional pain syndrome": "MODERATE",
    "arthritis of the hip": "MODERATE",
    "bursitis": "MODERATE",
    "gout": "MODERATE",
    "benign prostatic hyperplasia (bph)": "MODERATE",
    "obstructive sleep apnea (osa)": "MODERATE",
    "idiopathic excessive menstruation": "MODERATE",
    "vaginal cyst": "MODERATE",
    "anxiety": "MODERATE",
    "marijuana abuse": "MODERATE",
    "dental caries": "MODERATE",

    # Low urgency conditions (mild, routine, or self-limiting)
    "conjunctivitis due to allergy": "LOW",
    "nose disorder": "LOW",
    "eczema": "LOW",
    "fungal infection of the hair": "LOW",
    "stye": "LOW",
    "vulvodynia": "LOW",
}

# Actionable clinical recommendations by triage tier
TRIAGE_GUIDANCE: Dict[str, str] = {
    "EMERGENCY": (
        "CRITICAL EMERGENCY ALERT: Immediate medical intervention required. "
        "Call emergency services (911) or proceed immediately to the nearest Emergency Department. "
        "Do not drive yourself."
    ),
    "HIGH": (
        "HIGH PRIORITY: Prompt medical evaluation recommended within 12–24 hours at an Urgent Care center "
        "or same-day medical clinic. Monitor closely for acute escalation."
    ),
    "MODERATE": (
        "MODERATE CLINICAL PRIORITY: Schedule an appointment with a primary care physician or outpatient "
        "clinic within 2–3 days. Seek immediate medical attention if symptoms worsen."
    ),
    "LOW": (
        "ROUTINE / SELF-CARE: Symptoms appear consistent with non-urgent, self-limiting or mild conditions. "
        "Follow supportive self-care (hydration, rest) and consult a primary care provider or pharmacist "
        "if symptoms persist beyond one week."
    )
}

MEDICAL_DISCLAIMER: str = (
    "DISCLAIMER: SmartHealth Triage is an AI decision-support and triage prioritization tool "
    "designed for informational purposes only. It is not an automated medical diagnostic tool and does not "
    "replace evaluation by a licensed healthcare professional. In the event of a medical emergency, "
    "call 911 or visit the nearest emergency room immediately."
)


def _normalize_flag(flag: str) -> str:
    """Helper to convert user inputs (e.g. 'Difficulty Breathing', 'severe-chest-pain') to standard key."""
    return flag.strip().lower().replace(" ", "_").replace("-", "_")


def evaluate_triage(
    predicted_conditions_with_probs: Union[List[Tuple[str, float]], List[List[Any]], Dict[str, float]],
    reported_symptoms: Optional[List[str]] = None,
    user_red_flags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluates patient triage level applying deterministic clinical safety overrides.

    Parameters:
    -----------
    predicted_conditions_with_probs:
        List of (condition_name, probability) tuples sorted descending by probability, or a dict.
    reported_symptoms:
        List of symptom strings reported by the user.
    user_red_flags:
        List of explicit red-flag indicators selected or reported by the user.

    Returns:
    --------
    dict containing:
        - top_condition (str)
        - confidence (float)
        - triage_level (str: 'LOW', 'MODERATE', 'HIGH', 'EMERGENCY')
        - safety_override (bool)
        - override_reasons (list of str)
        - guidance (str)
        - disclaimer (str)
    """
    reported_symptoms = reported_symptoms or []
    user_red_flags = user_red_flags or []

    # 1. Parse top ML condition and confidence
    top_condition = "Undetermined"
    confidence = 0.0

    if isinstance(predicted_conditions_with_probs, dict):
        if predicted_conditions_with_probs:
            sorted_preds = sorted(predicted_conditions_with_probs.items(), key=lambda x: x[1], reverse=True)
            top_condition, confidence = sorted_preds[0]
    elif isinstance(predicted_conditions_with_probs, (list, tuple)):
        if len(predicted_conditions_with_probs) > 0:
            item = predicted_conditions_with_probs[0]
            top_condition = str(item[0])
            confidence = float(item[1])

    # 2. Check for Red-Flag Safety Overrides
    triggered_red_flags: List[str] = []
    
    # Check explicitly flagged red flags
    for flag in user_red_flags:
        if not flag or not str(flag).strip():
            continue
        norm = _normalize_flag(str(flag))
        if norm in EMERGENCY_RED_FLAGS:
            triggered_red_flags.append(f"{norm}: {EMERGENCY_RED_FLAGS[norm]}")
        else:
            # Check exact or extended phrase matches (e.g., "acute difficulty_breathing at rest")
            matched = False
            for k, desc in EMERGENCY_RED_FLAGS.items():
                if norm == k or (len(norm) >= len(k) and k in norm):
                    triggered_red_flags.append(f"{k}: {desc}")
                    matched = True
                    break
            if not matched and len(norm) > 1:
                triggered_red_flags.append(str(flag))

    # Also inspect reported symptoms for direct emergency red flags
    for sym in reported_symptoms:
        if not sym or not str(sym).strip():
            continue
        norm_sym = _normalize_flag(str(sym))
        for k, desc in EMERGENCY_RED_FLAGS.items():
            if norm_sym == k and f"{k}: {desc}" not in triggered_red_flags:
                triggered_red_flags.append(f"{k}: {desc}")

    # Deduplicate while preserving order
    triggered_red_flags = list(dict.fromkeys(triggered_red_flags))

    # 3. Determine Triage Urgency and Safety Override
    if len(triggered_red_flags) > 0:
        # HARD OVERRIDE: Any red flag automatically forces EMERGENCY level
        triage_level = "EMERGENCY"
        safety_override = True
        override_reasons = triggered_red_flags
        guidance = TRIAGE_GUIDANCE["EMERGENCY"]
    else:
        # No red flag: Derive triage level based on clinical mapping of the predicted condition
        safety_override = False
        override_reasons = []

        norm_cond = top_condition.lower().strip()
        if norm_cond in CONDITION_SEVERITY_MAP:
            triage_level = CONDITION_SEVERITY_MAP[norm_cond]
        else:
            # Fallback heuristic based on confidence and keywords
            if any(term in norm_cond for term in ["hemorrhage", "infarct", "stroke", "failure", "shock", "arrest"]):
                triage_level = "EMERGENCY"
            elif any(term in norm_cond for term in ["acute", "fracture", "infection", "bronchiolitis", "pneumonia"]):
                triage_level = "HIGH"
            elif any(term in norm_cond for term in ["allergy", "cold", "rhinitis", "mild", "dermatitis", "eczema"]):
                triage_level = "LOW"
            else:
                triage_level = "MODERATE"

        guidance = TRIAGE_GUIDANCE.get(triage_level, TRIAGE_GUIDANCE["MODERATE"])

    return {
        "top_condition": top_condition,
        "confidence": confidence,
        "triage_level": triage_level,
        "safety_override": safety_override,
        "override_reasons": override_reasons,
        "guidance": guidance,
        "disclaimer": MEDICAL_DISCLAIMER,
    }
