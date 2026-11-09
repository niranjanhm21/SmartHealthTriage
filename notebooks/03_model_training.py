import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

print("=" * 70)
print("     PHASE 4: MULTI-MODEL TRAINING & COMPARATIVE BENCHMARK")
print("=" * 70)

# 1. Load Processed Parquet Data
DATASET_DIR = os.path.join(os.getcwd(), 'datasets')
MODELS_DIR = os.path.join(os.getcwd(), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

X_train = pd.read_parquet(os.path.join(DATASET_DIR, 'X_train.parquet'))
X_test = pd.read_parquet(os.path.join(DATASET_DIR, 'X_test.parquet'))
y_train_df = pd.read_parquet(os.path.join(DATASET_DIR, 'y_train.parquet'))
y_test_df = pd.read_parquet(os.path.join(DATASET_DIR, 'y_test.parquet'))

target_col = y_train_df.columns[0]
y_train_raw = y_train_df[target_col].values
y_test_raw = y_test_df[target_col].values

# 2. Encode Labels to Integers (0 to 39)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)
y_test = label_encoder.transform(y_test_raw)

# Save Label Encoder for Streamlit app
joblib.dump(label_encoder, os.path.join(MODELS_DIR, 'label_encoder.joblib'))

# 3. Define Candidate Models (All native scikit-learn: zero DLL issues)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42),
    "Gradient Boosting (GBM)": HistGradientBoostingClassifier(max_iter=100, max_depth=10, random_state=42)
}

results = []
trained_models = {}

# 4. Train, Predict, and Evaluate
for name, clf in models.items():
    print(f"\n[Training] {name}...")
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_duration = time.time() - start_time
    
    # Probabilities for Top-1 and Top-3 accuracy
    probs = clf.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)
    
    top1 = accuracy_score(y_test, preds)
    top3 = top_k_accuracy_score(y_test, probs, k=3, labels=np.arange(len(label_encoder.classes_)))
    f1 = f1_score(y_test, preds, average='macro')
    
    results.append({
        "Model": name,
        "Top-1 Accuracy": f"{top1 * 100:.2f}%",
        "Top-3 Accuracy": f"{top3 * 100:.2f}%",
        "Macro F1-Score": f"{f1 * 100:.2f}%",
        "Training Time (s)": f"{train_duration:.2f}s"
    })
    trained_models[name] = (clf, top3)

# 5. Display Benchmark Results Table
results_df = pd.DataFrame(results)
print("\n" + "=" * 70)
print("                   BENCHMARK EVALUATION RESULTS")
print("=" * 70)
print(results_df.to_string(index=False))
print("=" * 70)

# 6. Save the Best Performing Model (Based on Top-3 Accuracy)
best_model_name = max(trained_models, key=lambda k: trained_models[k][1])
best_clf = trained_models[best_model_name][0]
joblib.dump(best_clf, os.path.join(MODELS_DIR, 'best_model.joblib'))

# Also save Random Forest explicitly for fast SHAP analysis
rf_clf = trained_models["Random Forest"][0]
joblib.dump(rf_clf, os.path.join(MODELS_DIR, 'random_forest_model.joblib'))

print(f"\n[Artifact Saved] Selected '{best_model_name}' as best model.")
print(f"Saved artifacts to '{MODELS_DIR}/'")
print("=" * 70 + "\n")