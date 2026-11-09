
"""
Safety Layer and Rule-Based Triage Assessment.
SmartHealth Triage Academic Prototype.

This module checks configured emergency warning signs and applies
rule-based urgency overrides. It is an educational prototype and
has not been clinically validated.
"""

from typing import List, Tuple, Dict, Any, Union, Optional


# 1. Configured emergency warning signs
EMERGENCY_RED_FLAGS: Dict[str, str] = {
    "severe_chest_pain": (
        "Severe crushing chest pain, pressure, or tightness "
        "radiating to the arm or jaw."
    ),
    "difficulty_breathing": (
        "Severe shortness of breath, respiratory distress, "
        "gasping, or inability to speak in sentences."
    ),
    "cyanosis": (
        "Bluish or grayish discoloration of the lips, face, or nail beds."
    ),
    "loss_of_consciousness": (
        "Loss of consciousness, sudden unresponsiveness, "
        "or severely altered mental status."
    ),
    "sudden_paralysis": (
        "Sudden weakness, facial drooping, numbness, "
        "or paralysis on one side of the body."
    ),
    "hemoptysis": (
        "Coughing up blood or vomiting significant amounts of blood."
    ),
    "severe_abdominal_trauma": (
        "Severe abdominal trauma, rigid abdomen, "
        "or sudden severe abdominal pain."
    )
}


# 2. Baseline urgency mapping for predicted conditions
CONDITION_SEVERITY_MAP: Dict[str, str] = {
    "gastrointestinal hemorrhage": "EMERGENCY",
    "acute pancreatitis": "EMERGENCY",
    "spontaneous abortion": "EMERGENCY",

    "pneumonia": "HIGH",
    "cholecystitis": "HIGH",
    "diverticulitis": "HIGH",
    "concussion": "HIGH",
    "hypoglycemia": "HIGH",
    "liver disease": "HIGH",
    "acute bronchiolitis": "HIGH",

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

    "conjunctivitis due to allergy": "LOW",
    "nose disorder": "LOW",
    "eczema": "LOW",
    "fungal infection of the hair": "LOW",
    "stye": "LOW",
    "vulvodynia": "LOW",
}


# 3. General guidance by urgency level
TRIAGE_GUIDANCE: Dict[str, str] = {
    "EMERGENCY": (
        "EMERGENCY WARNING: Seek immediate professional medical help. "
        "Call local emergency services (112 in India) or go to the "
        "nearest emergency department. Do not drive yourself if "
        "you may be seriously unwell."
    ),
    "HIGH": (
        "HIGH PRIORITY: Prompt medical evaluation is recommended. "
        "Seek professional advice as soon as possible, preferably "
        "the same day. Seek emergency help if symptoms become severe."
    ),
    "MODERATE": (
        "MODERATE PRIORITY: Arrange an appointment with a healthcare "
        "professional. Seek earlier care if symptoms worsen."
    ),
    "LOW": (
        "LOWER PRIORITY: Consider general supportive measures such as "
        "rest and hydration. Consult a healthcare professional if "
        "symptoms persist, worsen, or cause concern."
    )
}


# 4. Disclaimer
MEDICAL_DISCLAIMER: str = (
    "DISCLAIMER: SmartHealth Triage is an academic symptom-assessment "
    "prototype. It is not a medical diagnostic system and does not "
    "replace evaluation by a qualified healthcare professional. "
    "Its predictions and urgency assessments have not been clinically "
    "validated. In a possible medical emergency, seek immediate "
    "professional help through local emergency services (112 in India)."
)


def _normalize_flag(flag: str) -> str:
    """Convert a red-flag label into a comparable format."""
    return (
        str(flag)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def evaluate_triage(
    predicted_conditions_with_probs: Union[
        List[Tuple[str, float]],
        List[List[Any]],
        Dict[str, float]
    ],
    reported_symptoms: Optional[List[str]] = None,
    user_red_flags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluate urgency using configured warning signs and condition mapping.

    Returns a dictionary containing the predicted condition, confidence,
    triage level, safety override status, reasons, guidance and disclaimer.
    """

    reported_symptoms = reported_symptoms or []
    user_red_flags = user_red_flags or []

    # 1. Read the highest-ranked predicted condition
    top_condition = "Undetermined"
    confidence = 0.0

    if isinstance(predicted_conditions_with_probs, dict):
        if predicted_conditions_with_probs:
            sorted_predictions = sorted(
                predicted_conditions_with_probs.items(),
                key=lambda item: item[1],
                reverse=True
            )
            top_condition, confidence = sorted_predictions[0]

    elif isinstance(predicted_conditions_with_probs, (list, tuple)):
        if len(predicted_conditions_with_probs) > 0:
            item = predicted_conditions_with_probs[0]
            top_condition = str(item[0])
            confidence = float(item[1])

    # 2. Check explicitly selected red flags
    triggered_red_flags: List[str] = []

    for flag in user_red_flags:
        if not flag or not str(flag).strip():
            continue

        norm = _normalize_flag(str(flag))

        # Only configured warning signs can activate this override.
        for key, description in EMERGENCY_RED_FLAGS.items():
            if norm == key or (
                len(norm) >= len(key) and key in norm
            ):
                triggered_red_flags.append(
                    f"{key}: {description}"
                )
                break

    # 3. Check reported symptoms for exact warning-sign matches
    for symptom in reported_symptoms:
        if not symptom or not str(symptom).strip():
            continue

        norm_symptom = _normalize_flag(str(symptom))

        if norm_symptom in EMERGENCY_RED_FLAGS:
            description = EMERGENCY_RED_FLAGS[norm_symptom]
            triggered_red_flags.append(
                f"{norm_symptom}: {description}"
            )

    # Remove duplicate warning signs
    triggered_red_flags = list(dict.fromkeys(triggered_red_flags))

    # 4. Determine urgency
    if triggered_red_flags:
        triage_level = "EMERGENCY"
        safety_override = True
        override_reasons = triggered_red_flags
        guidance = TRIAGE_GUIDANCE["EMERGENCY"]

    else:
        safety_override = False
        override_reasons = []

        norm_condition = str(top_condition).lower().strip()

        if norm_condition in CONDITION_SEVERITY_MAP:
            triage_level = CONDITION_SEVERITY_MAP[norm_condition]

        else:
            # Fallback keyword-based urgency estimation
            if any(
                term in norm_condition
                for term in [
                    "hemorrhage", "infarct", "stroke",
                    "failure", "shock", "arrest"
                ]
            ):
                triage_level = "EMERGENCY"

            elif any(
                term in norm_condition
                for term in [
                    "acute", "fracture", "infection",
                    "bronchiolitis", "pneumonia"
                ]
            ):
                triage_level = "HIGH"

            elif any(
                term in norm_condition
                for term in [
                    "allergy", "cold", "rhinitis",
                    "mild", "dermatitis", "eczema"
                ]
            ):
                triage_level = "LOW"

            else:
                triage_level = "MODERATE"

        guidance = TRIAGE_GUIDANCE.get(
            triage_level,
            TRIAGE_GUIDANCE["MODERATE"]
        )

    # 5. Return assessment
    return {
        "top_condition": top_condition,
        "confidence": confidence,
        "triage_level": triage_level,
        "safety_override": safety_override,
        "override_reasons": override_reasons,
        "guidance": guidance,
        "disclaimer": MEDICAL_DISCLAIMER,
    }
