# Academic Project Report: SmartHealth Triage

**Degree Programme:** Master of Computer Applications (MCA)  
**Project Title:** SmartHealth Triage: Explainable Symptom-Based Health Issue Prediction and Risk Stratification System  
**Academic Domain:** Data Science, Health Informatics, Machine Learning and Explainable Artificial Intelligence (XAI)  
**Submission Year:** 2026

---

## Executive Summary / Abstract

**SmartHealth Triage** is an academic project that explores explainable machine learning for symptom-based health issue prediction and urgency assessment. The system combines a machine learning classifier with a separate rule-based safety layer that checks for selected emergency red flags. It also uses SHAP-based explanations to show which symptom features contribute to model predictions.

The application accepts user-reported symptoms and displays up to three possible health conditions, an urgency category, relevant red-flag alerts, and an explanation of the model output. The safety layer can override the model's normal triage result when a configured emergency red flag is detected.

The project uses Python, Scikit-learn, Streamlit, and SHAP. Its purpose is to demonstrate how machine learning, explainability, and explicit safety rules can be combined in an educational symptom-assessment prototype.

The system has important limitations. Its predictions and urgency categories have not been clinically validated, and its model confidence scores should not be interpreted as medically calibrated probabilities. The red-flag rules are limited to predefined conditions and do not guarantee that every emergency will be detected. The application is not a medical diagnostic tool and must not replace professional medical assessment.

---

## 1. Introduction and Problem Statement

### 1.1 Background and Context

People often experience symptoms without knowing which health condition may be associated with them or how urgently they should seek medical attention. Digital symptom-assessment applications attempt to help users understand possible health conditions and decide what level of care may be appropriate.

Machine learning provides techniques for identifying patterns in symptom data and predicting possible health conditions. However, a system that focuses only on predicting the most likely condition may overlook important warning signs that require urgent attention.

SmartHealth Triage explores how machine learning can be combined with a separate safety layer and explainable AI to create a more transparent symptom-assessment prototype.

### 1.2 Problem Statement

A conventional symptom-based classification model primarily learns relationships between symptoms and disease labels. Its prediction may not always reflect the urgency of a user's condition.

For example, some serious conditions may initially share symptoms with common illnesses. A model may therefore predict a common condition even when a reported warning sign requires urgent attention.

Another challenge is model transparency. Users may receive a prediction without understanding which input features influenced the result.

The project addresses these challenges by combining possible-condition prediction, urgency assessment, predefined emergency red-flag checks, and model explanations in one application.

### 1.3 Project Objectives

1. **Symptom-Based Condition Prediction:** Develop a machine learning model that predicts possible health conditions from reported symptoms.
2. **Top-3 Predictions:** Display up to three possible conditions instead of presenting one prediction as a definitive diagnosis.
3. **Safety Override:** Implement a separate rule-based safety layer that checks configured emergency warning signs and can override the normal model assessment.
4. **Urgency Assessment:** Provide an urgency category based on the implemented safety rules and condition-to-urgency mapping.
5. **Explainable AI:** Use SHAP to explain the symptom features contributing to model predictions.
6. **Interactive Application:** Develop a Streamlit interface for symptom input and presentation of assessment results.
7. **Safety Evaluation:** Examine model performance and test whether configured red-flag rules behave as expected.

### 1.4 Scope of the Project

The project focuses on an educational prototype for symptom-based condition prediction and urgency assessment.

The current implementation includes:
- User-reported symptom selection.
- Machine learning-based condition prediction.
- Top-3 possible condition suggestions.
- A rule-based emergency red-flag layer.
- Urgency classification.
- SHAP-based feature explanations.
- A Streamlit user interface.

The application does not provide a confirmed diagnosis, prescribe medication, or replace professional medical assessment.

---

## 2. Literature Review and Theoretical Grounding

### 2.1 Symptom Checker Accuracy Reporting Framework (Kopka and Feufel, 2026)

Kopka and Feufel (2026) proposed the **Symptom Checker Accuracy Reporting Framework (SCARF)** to improve the quality, transparency, and comparability of evaluations of symptom checkers and AI-based diagnostic decision-support systems.

The framework emphasizes:
- Clearly defining the intended use case and target population.
- Selecting representative clinical cases or symptom scenarios.
- Establishing appropriate reference standards for evaluating system recommendations.
- Reporting overall accuracy and performance across individual triage levels.
- Evaluating under-triage and over-triage.
- Improving consistency between evaluations so that results can be compared across studies.

**Relevance to SmartHealth Triage:**

SCARF provides methodological guidance for evaluating symptom-assessment systems. It highlights that evaluating a system only through condition prediction accuracy is insufficient. Urgency recommendations and the safety of the system must also be considered.

This framework supports the evaluation approach adopted in SmartHealth Triage, particularly the importance of assessing emergency-related performance, under-triage, and over-triage.

### 2.2 Patient-Facing AI System for Symptom Guidance (Patwardhan et al., 2026)

Patwardhan et al. (2026) conducted an analytical validation study of a patient-facing AI system designed to provide symptom guidance and care-seeking recommendations.

The researchers evaluated a locked configuration of the Personal Health Assistant using 772 synthetic patient cases. Five independent physicians reviewed the cases to establish reference recommendations for the appropriate level of care.

The study evaluated:
- Urgent under-triage, where a system recommends a less urgent level of care than required.
- Nonurgent under-triage, where the recommended level of care is less urgent than the reference recommendation.
- Over-triage, where the system recommends a more urgent level of care than necessary.
- The appropriateness of possible-condition suggestions and additional patient guidance.

The study reported urgent under-triage in 6.9% of evaluated cases, nonurgent under-triage in 18.4%, and over-triage in 11.0%. These results belong to the evaluated Personal Health Assistant, not SmartHealth Triage. The study used synthetic cases and simulated conversations, and its authors noted that the results may not represent real-world performance.

**Relevance to SmartHealth Triage:**

This research demonstrates the importance of evaluating patient-facing AI systems using structured test cases and expert reference assessments. It also highlights that under-triage and over-triage should be measured separately rather than relying only on overall accuracy.

The study provides a useful reference for the safety evaluation strategy of SmartHealth Triage.

### 2.3 Synthesis of Research Gaps

The reviewed research highlights several important considerations when developing and evaluating symptom-assessment systems:

- Condition prediction accuracy alone does not establish the safety of a triage recommendation.
- Under-triage and over-triage should be evaluated separately.
- Emergency warning signs require explicit attention during system design and testing.
- Explainability can help users understand which input features influence model predictions, but it does not independently establish clinical correctness.
- Transparent reporting of datasets, evaluation methods, results, and limitations is necessary.
- Systems evaluated using synthetic data require further validation before they can be considered suitable for real-world clinical use.

**Proposed Approach in SmartHealth Triage:**

SmartHealth Triage explores these considerations through machine learning-based condition prediction, a separate rule-based red-flag safety layer, and SHAP-based model explanations.

The machine learning component predicts possible health conditions from reported symptoms. The safety layer checks for configured emergency warning signs and can override the normal assessment when a recognised red flag is detected. SHAP is used to explain the features contributing to model predictions.

The current implementation is an academic prototype. Its predictions, urgency categories, and safety rules have not undergone clinical validation and must not be used as a substitute for professional medical assessment.

---

## 3. Methodology and System Architecture

### 3.1 Overall System Architecture

The system follows a modular architecture in which condition prediction and safety assessment are handled as separate components.

```text
+------------------------------------------------------+
|                    USER INPUT                        |
|                                                      |
|       Reported Symptoms and Red-Flag Selections      |
+---------------------------+--------------------------+
                            |
                            v
+------------------------------------------------------+
|               INPUT PROCESSING                       |
|                                                      |
|       Symptom Normalization and Feature Vector       |
+---------------------------+--------------------------+
                            |
              +-------------+-------------+
              |                           |
              v                           v
+---------------------------+  +-----------------------+
|   MACHINE LEARNING MODEL  |  |    SAFETY LAYER       |
|                           |  |                       |
|   Possible Conditions     |  |   Red-Flag Detection  |
|   Top-3 Predictions       |  |   Rule-Based Checks   |
+-------------+-------------+  +-----------+-----------+
              |                            |
              +-------------+--------------+
                            |
                            v
+------------------------------------------------------+
|             URGENCY ASSESSMENT                       |
|                                                      |
|  Safety Override or Condition-Based Urgency Mapping  |
+---------------------------+--------------------------+
                            |
                            v
+------------------------------------------------------+
|                EXPLAINABILITY                        |
|                                                      |
|       SHAP-Based Feature Contributions               |
+---------------------------+--------------------------+
                            |
                            v
+------------------------------------------------------+
|                STREAMLIT INTERFACE                   |
|                                                      |
|       Predictions, Alerts and Explanations           |
+------------------------------------------------------+
```

The safety layer is separate from the machine learning classifier. When a configured emergency red flag is detected, it can override the normal assessment.

### 3.2 Dataset Description

The project uses a symptom-and-disease dataset for supervised machine learning.

The dataset is used to train a model that learns relationships between symptom features and disease labels. The available data is artificially generated and should not be treated as a clinically validated patient dataset.

The project uses processed symptom features and disease labels for model development.

Important dataset characteristics to document include:
- Total number of records.
- Number of original disease categories.
- Number of selected disease categories.
- Number of symptom features.
- Missing values and duplicate records.
- Class distribution.
- Training and testing sample counts.

The final values should be taken directly from the dataset inspection and preprocessing outputs. They should not be estimated or assumed.

### 3.3 Data Preprocessing

The preprocessing stage prepares the dataset for machine learning.

The main activities include:

1. **Dataset Inspection:** Examine the dataset structure, columns, target variable, and data types.
2. **Missing Value Handling:** Identify and handle missing or incomplete records where required.
3. **Duplicate Checking:** Identify duplicate records and decide how they should be handled.
4. **Feature Selection:** Select the symptom features used by the model.
5. **Target Preparation:** Prepare disease labels for supervised classification.
6. **Feature Encoding:** Represent symptom information in a format suitable for machine learning.
7. **Train-Test Split:** Divide the processed data into training and testing sets.

The actual split ratio and sample counts should be reported from the executed preprocessing pipeline.

### 3.4 Candidate Machine Learning Models

The project considers the following classification approaches:

**1. Logistic Regression**

Logistic Regression is used as a baseline classification model. It provides a relatively simple approach for learning relationships between input features and target classes.

**2. Random Forest Classifier**

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees. It can model nonlinear relationships and interactions between symptom features.

**3. Histogram Gradient Boosting**

Histogram Gradient Boosting is a boosting-based approach that builds decision trees sequentially to improve prediction performance.

The candidate models should be evaluated using the same test dataset and appropriate classification metrics. The final selected model must be based on measured results rather than assumed performance.

### 3.5 Safety Layer and Red-Flag Detection

The safety layer is implemented separately from the machine learning prediction component.

It checks user-selected red flags against predefined emergency warning-sign rules.

The configured red-flag categories include:
- Severe chest pain.
- Difficulty breathing.
- Coughing up blood.
- Sudden paralysis or serious neurological symptoms.
- Loss of consciousness.
- Cyanosis.
- Severe abdominal trauma or related warning signs.

When a configured red flag is recognised, the safety layer can override the normal urgency assessment and return an emergency-level result.

**Safety Override Logic:**

```text
IF a configured emergency red flag is detected:
       Set urgency to EMERGENCY
       Activate safety override
       Display emergency guidance

ELSE:
       Use the normal condition-based urgency assessment
```

The safety layer is a predefined rule-based mechanism. It is not a complete clinical decision system and cannot guarantee detection of every emergency condition.

### 3.6 Explainable AI Using SHAP

SHAP stands for **SHapley Additive exPlanations**. It is a model explanation technique based on Shapley values from cooperative game theory.

SHAP helps explain how individual input features contribute to a model prediction.

The general additive explanation is represented as:

\[
f(x) = \phi_0 + \sum_{i=1}^{M}\phi_i
\]

Where:

- \(f(x)\) represents the model output being explained.
- \(\phi_0\) represents the baseline model output.
- \(\phi_i\) represents the contribution of feature \(i\).
- \(M\) represents the number of features.

Positive and negative SHAP values indicate the direction in which a feature influences the model output being explained.

**Application in SmartHealth Triage:**

SHAP is used to provide explanations for model predictions by identifying symptom features that contributed to the selected prediction.

These explanations describe model behaviour. They do not prove that a predicted condition is medically correct.

---

## 4. Model Evaluation and Safety Testing

### 4.1 Evaluation Strategy

The machine learning models should be evaluated using a separate test dataset that was not used during model training.

The following metrics are relevant:

| Evaluation Metric | Purpose |
|---|---|
| Accuracy | Measures the overall proportion of correct predictions. |
| Precision | Measures how many predicted class instances are correct. |
| Recall | Measures how many actual class instances are correctly identified. |
| F1-score | Combines precision and recall. |
| Confusion Matrix | Shows correct and incorrect predictions across classes. |
| Top-3 Accuracy | Checks whether the correct class appears among the three highest-ranked predictions. |
| Macro F1-score | Gives equal weight to each class when calculating the F1-score. |

The actual values must be generated from the final model evaluation. No unverified performance figures are presented in this report.

### 4.2 Safety Evaluation

The safety layer should be tested independently and as part of the complete inference pipeline.

The evaluation should include:

**Test Case 1: Routine Symptoms Without Red Flags**

- Provide a set of ordinary reported symptoms.
- Ensure that no configured emergency red flag is selected.
- Check the predicted conditions and normal urgency assessment.
- Confirm that the safety override is not activated unless a configured rule is triggered.

**Test Case 2: Emergency Red-Flag Input**

- Provide symptoms along with a configured emergency red flag, such as difficulty breathing.
- Check whether the safety layer recognises the configured warning sign.
- Confirm whether the normal assessment is overridden.
- Check whether the application displays emergency guidance.

**Test Case 3: Multiple Red Flags**

- Provide more than one configured emergency warning sign.
- Check whether the safety layer recognises the applicable rules.
- Verify the resulting urgency assessment and displayed alerts.

**Test Case 4: Unrecognised Input**

- Provide an unrecognised symptom or red-flag description.
- Check how the application handles unsupported input.
- Ensure that the application does not silently imply that an unrecognised warning sign is safe.

### 4.3 Results Reporting

The final report should include actual results obtained by executing the model evaluation and safety tests.

The following information should be recorded:

- Selected model and model configuration.
- Training and testing dataset sizes.
- Accuracy, precision, recall, and F1-score.
- Top-3 accuracy.
- Confusion matrix.
- Class-wise performance.
- Under-triage and over-triage results, where a suitable reference-labelled safety test set is available.
- Red-flag test cases and their observed outcomes.
- SHAP explanation behaviour and measured execution time, if measured.

A successful rule execution in a small test set should not be described as proof of complete emergency detection or clinical safety.

---

## 5. Software Architecture and Implementation

The application is organised into separate modules to make the project easier to understand, maintain, and extend.

### 5.1 Inference Pipeline (`src/inference.py`)

The inference module handles model prediction.

Its responsibilities include:
- Preparing the input symptom feature vector.
- Loading and using the trained model.
- Generating possible-condition predictions.
- Producing the information required by the application.
- Coordinating the safety and explainability components.

The current inference flow uses reported symptoms and red-flag inputs for assessment. Although the interface collects other information such as age, sex, symptom duration, and severity, these fields should not be described as model features unless they are explicitly passed into and used by the prediction pipeline.

### 5.2 Safety Rule Engine (`src/safety_layer.py`)

The safety module contains predefined emergency red-flag rules and condition-to-urgency mappings.

Its responsibilities include:
- Checking configured emergency warning signs.
- Determining whether a safety override should be activated.
- Returning the applicable urgency result.
- Providing relevant safety information to the application.

The rules are limited to the conditions explicitly implemented in the source code.

### 5.3 Explainability Engine (`src/explainability.py`)

The explainability module uses SHAP to explain the model's predictions.

It helps identify the input features that contribute to the selected model output and prepares the explanation data for display in the application.

### 5.4 Streamlit Application (`app/app.py`)

The application provides the user interface.

Its main responsibilities include:
- Collecting user inputs.
- Displaying possible-condition predictions.
- Showing urgency information.
- Presenting red-flag alerts.
- Displaying model explanations.
- Providing safety disclaimers and next-step guidance.

### 5.5 Main Application Entry Point (`main.py`)

The root `main.py` file provides the application entry point and launches the Streamlit application.

### 5.6 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Dataset handling and preprocessing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and evaluation |
| SHAP | Model explainability |
| Streamlit | Interactive web application |
| Altair | Data visualisation |
| Joblib | Saving and loading model artifacts |
| PyArrow | Parquet data handling |

---

## 6. Discussion, Limitations and Future Scope

### 6.1 Discussion

SmartHealth Triage demonstrates an approach to combining machine learning, rule-based safety checks, and explainability in a symptom-assessment application.

The machine learning component provides possible-condition predictions, while the separate safety layer checks for configured emergency warning signs. SHAP provides an additional explanation of how the model uses symptom features.

The project also highlights an important distinction between classification performance and safety performance. A model can achieve good classification results without necessarily producing safe urgency recommendations. Therefore, safety evaluation should be treated as a separate and important part of the system.

### 6.2 Project Limitations

1. **Synthetic Dataset:** The project dataset is artificially generated and does not establish performance on real patient populations.
2. **No Clinical Validation:** The application has not been clinically validated by medical professionals or tested in a real healthcare setting.
3. **Limited Red-Flag Rules:** The safety layer checks only predefined warning signs. It cannot guarantee that all emergency situations will be detected.
4. **Limited Input Features:** The model's prediction pipeline is based on the symptom features it receives. Information collected by the interface but not passed to the model does not influence its prediction.
5. **Binary Symptom Representation:** Symptom features represented as present or absent may not capture intensity, progression, or other clinical details.
6. **Model Confidence:** Classifier scores are model outputs and should not be interpreted as medically calibrated probabilities.
7. **Possible Data Bias:** Artificially generated data may contain patterns or limitations that affect model generalisation.
8. **Uncertainty in Predictions:** Similar symptoms can be associated with different health conditions, and the model may not always identify the correct condition.

### 6.3 Future Scope

1. **Improved Dataset Quality:** Evaluate the system using carefully selected, appropriately governed, and clinically reviewed datasets where permitted.
2. **Adaptive Symptom Questioning:** Introduce follow-up questions based on previously reported symptoms.
3. **Additional Input Features:** Explore the use of symptom duration, severity, age, relevant medical history, and vital signs in the prediction pipeline, subject to suitable data availability and validation.
4. **Improved Safety Evaluation:** Develop a larger, independently reviewed red-flag test set and measure under-triage and over-triage.
5. **Enhanced Explainability:** Improve the presentation of global and individual SHAP explanations.
6. **Model Calibration and Uncertainty:** Investigate whether model outputs can be calibrated and how uncertain predictions should be communicated.
7. **Deployment:** Deploy the application through a suitable hosting platform for academic demonstration.
8. **Prospective Evaluation:** Conduct appropriate evaluation with expert oversight before considering any real-world healthcare application.

---

## 7. Conclusion

**SmartHealth Triage** is an academic project that explores explainable machine learning for symptom-based condition prediction and urgency assessment.

The project combines a machine learning classifier, a separate rule-based safety layer, SHAP-based explanations, and an interactive Streamlit interface. This architecture demonstrates how different software components can work together to provide possible-condition predictions, identify configured emergency warning signs, and explain model behaviour.

The project also emphasises that classification accuracy alone is not sufficient for evaluating symptom-assessment systems. Safety-related evaluation, including under-triage and over-triage, is an important part of assessing such applications.

The current implementation is an educational prototype based on artificially generated data and predefined rules. It has not been clinically validated and must not be used to diagnose medical conditions or replace professional medical advice.

---

## 8. Academic References

1. **Kopka, M., & Feufel, M. A. (2026).** How to Evaluate the Accuracy of Symptom Checkers and Diagnostic Decision Support Systems: Symptom Checker Accuracy Reporting Framework (SCARF). *JMIR Human Factors, 13*, e76168. https://doi.org/10.2196/76168

2. **Patwardhan, A., Verma, N., Barman, P., Mao, B., Li, N., Varghese, P., Peri, V., & Lehmann, L. S. (2026).** Patient-Facing AI System for Symptom Guidance Using Simulated Encounters and Physician Review: Analytical Validation Study. *JMIR Formative Research, 10*, e103419. https://doi.org/10.2196/103419

3. **Lundberg, S. M., & Lee, S. I. (2017).** A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems, 30*.

4. **Gilboy, N., Tanabe, T., Travers, D., & Rosenau, A. M. (2012).** Emergency Severity Index (ESI): A Triage Tool for Emergency Department Care, Version 4: Implementation Handbook. *Agency for Healthcare Research and Quality (AHRQ)*, Publication No. 12-0014.

5. **Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011).** Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

---

## Project Disclaimer

SmartHealth Triage is an academic software prototype developed for educational and research purposes. It is not a medical diagnostic system and has not been clinically validated. Its predictions, model scores, explanations, and urgency categories must not be treated as confirmed medical conclusions.

If a person experiences a possible medical emergency, they should seek immediate professional help through their local emergency services. In India, the national emergency response number is 112.
