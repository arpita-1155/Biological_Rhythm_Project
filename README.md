# 🧬 Automated Biological Rhythm Analysis & Health State Classification

## 📌 Project Overview

Automated Biological Rhythm Analysis & Health State Classification is a final-year B.Tech project that analyzes wearable activity data to identify biological rhythm and behavioral patterns.

The system combines:

- Data preprocessing and resampling
- 24-hour Cosinor analysis
- Wavelet Transform analysis
- Sample Entropy
- Permutation Entropy
- Detrended Fluctuation Analysis (DFA)
- Random Forest classification
- XGBoost classification
- Interactive Streamlit visualization

The project is designed as an integrated pipeline that transforms noisy wearable activity signals into interpretable rhythm and complexity features and then uses machine-learning models for health-state classification.

---

## 🎯 Objectives

The major objectives of the project are:

1. Process and clean wearable actigraphy data.
2. Resample activity data into a consistent 5-minute representation.
3. Extract 24-hour biological rhythm characteristics using Cosinor analysis.
4. Analyze time-frequency activity patterns using Wavelet Transform.
5. Quantify signal complexity using Sample Entropy and Permutation Entropy.
6. Analyze long-range fluctuations using Detrended Fluctuation Analysis.
7. Combine extracted features into a machine-learning feature table.
8. Classify subjects using Random Forest and XGBoost.
9. Prevent data leakage using subject-level separation during model validation.
10. Provide an interactive Streamlit dashboard for visualization and demonstration.

---

## 🏗️ System Pipeline

```text
Raw Wearable Activity Data
            ↓
     Data Preprocessing
            ↓
     5-Minute Resampling
            ↓
 ┌─────────────────────────────┐
 │   Biological Rhythm Analysis│
 │                             │
 │  • Cosinor Analysis         │
 │  • Wavelet Transform        │
 └─────────────────────────────┘
            ↓
 ┌─────────────────────────────┐
 │ Complexity Analysis         │
 │                             │
 │  • Sample Entropy           │
 │  • Permutation Entropy      │
 │  • DFA                      │
 └─────────────────────────────┘
            ↓
      Feature Combination
            ↓
 ┌─────────────────────────────┐
 │ Machine Learning            │
 │                             │
 │  • Random Forest            │
 │  • XGBoost                  │
 └─────────────────────────────┘
            ↓
      Subject Classification
            ↓
     Streamlit Dashboard