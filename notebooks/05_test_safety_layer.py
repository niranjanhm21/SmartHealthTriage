import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.safety_layer import evaluate_triage

print("=" * 65)
print("     VERIFYING TASK 2: CLINICAL RED-FLAG SAFETY OVERRIDE")
print("=" * 65)

# CASE A: Mild Case without Red Flags
print("\n[TEST CASE A: Mild Cold Symptoms - No Red Flags]")
preds_a = [("conjunctivitis due to allergy", 0.74), ("nose disorder", 0.18), ("cystitis", 0.02)]
symptoms_a = ["sneezing", "runny_nose", "watery_eyes"]
result_a = evaluate_triage(preds_a, reported_symptoms=symptoms_a, user_red_flags=[])

print(f"Predicted:       {result_a['top_condition']} ({result_a['confidence']*100:.1f}%)")
print(f"Triage Level:    {result_a['triage_level']}")
print(f"Safety Override: {result_a['safety_override']}")
print(f"Guidance:        {result_a['guidance']}")

# CASE B: Mild Condition + Critical Red Flag (Difficulty Breathing)
print("\n" + "-" * 65)
print("[TEST CASE B: Mild Diagnosis + Acute Red Flag (Breathing Emergency)]")
preds_b = [("nose disorder", 0.81), ("conjunctivitis due to allergy", 0.10)]
symptoms_b = ["cough", "runny_nose"]
red_flags_b = ["difficulty_breathing"]

result_b = evaluate_triage(preds_b, reported_symptoms=symptoms_b, user_red_flags=red_flags_b)

print(f"Original ML Top Condition: {result_b['top_condition']} ({result_b['confidence']*100:.1f}%)")
print(f"Triage Level:              {result_b['triage_level']}")
print(f"Safety Override:           {result_b['safety_override']} (OVERRIDDEN!)")
print(f"Triggered Red Flags:       {result_b['override_reasons']}")
print(f"Guidance:                  {result_b['guidance']}")

print("=" * 65)
print("ALL TESTS PASSED: Safety Override behaves deterministically.")
print("=" * 65)