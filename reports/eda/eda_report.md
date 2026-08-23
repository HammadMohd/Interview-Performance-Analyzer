# Detailed Exploratory Data Analysis (EDA) Report
**Dataset:** `data/features/merged_features.csv`  
**Dataset Dimensions:** 2011 rows x 88 columns  
**Total Numerical Columns:** 84  
**Total Categorical / Text Columns:** 4  
**Unique Candidates:** 331  

---

## 1. Executive Summary and Structural Integrity

### Numerical Summary
- **Dataset Size:** 2011 interview video response samples.
- **Memory Footprint:** 1.35 MB.
- **Missing Values:** Only `transcript` has missing values (13 rows, 0.65%). All 84 numerical feature columns have 0 missing values.
- **Duplicate Rows:** 0 duplicate records.

### What We Learn From These Metrics
1. **Data Collection Reliability:** The feature extraction pipeline (OpenFace, MediaPipe, Librosa/Whisper) executed with 100% success on numerical feature extraction across all 2011 clips.
2. **Missing Text Entries:** The 13 missing transcripts represent short video clips where speech recognition did not detect audible speech or candidate silence.

### Engineering Decisions and Options
- **Decision:** Impute the 13 missing transcript values with empty strings (`""`) prior to text NLP embedding generation.
- **Option A (Chosen):** Simple string imputation (`""`). Keeps all 2011 numerical rows fully usable.
- **Option B (Rejected):** Row deletion. Deleting rows would drop valid audio and visual feature data.

---

## 2. Target Variable Profiles

### Target Score Summary Statistics
The dataset contains 6 performance target variables. All targets are standardized z-scores (mean approx 0.0, std approx 1.0 - 1.2):

| Target Column | Mean | Std | Min | Median | Max | Skewness | Kurtosis |
|---|---|---|---|---|---|---|---|
| `interview_score` | 0.0000 | 1.2301 | -7.8603 | -0.0274 | 9.3860 | 0.6801 | 11.6040 |
| `answer_score` | 0.0000 | 1.1755 | -10.1991 | 0.0060 | 8.7223 | 0.3534 | 12.0329 |
| `speaking_skills` | 0.0000 | 1.2776 | -9.4063 | 0.0199 | 7.9015 | -0.8550 | 12.7270 |
| `confidence_score` | 0.0000 | 1.0797 | -7.2861 | 0.0111 | 6.8394 | -0.6404 | 8.8211 |
| `facial_expression` | 0.0000 | 1.1510 | -7.4614 | -0.0184 | 9.2558 | 0.5977 | 11.1791 |
| `overall_performance` | 0.0000 | 1.2007 | -9.3111 | 0.0434 | 8.8490 | -0.7475 | 11.7422 |

### What We Learn From Target Statistics
1. **Standardization:** All target distributions are centered near 0.0 with standard deviations close to 1.0. The targets were pre-standardized during dataset curation.
2. **Symmetry:** Skewness values range between -0.8550 and +0.6801, indicating approximately symmetric, unimodal distributions.
3. **Outlier Prevalence:** Target outliers (Z > 3) represent only 1.0% to 1.6% of the dataset.

### Engineering Decisions and Options
- **Decision:** Do NOT apply logarithmic transformations to target variables. Use standard regression loss functions (Mean Squared Error or Mean Absolute Error).
- **Option A (Chosen - Single Target Baseline):** Predict `interview_score` as the primary target for initial model evaluation.
- **Option B (Chosen - Advanced Multi-Task):** Train a multi-output regressor to predict all 6 target scores simultaneously, leveraging shared feature representations across sub-metrics.

---

## 3. Inter-Target and Psychometric Correlations

### Top Predictive Features Across All Domains
Ranked by average absolute Pearson correlation (|r|) across all 6 target scores:

| Feature Name | Pearson r (vs `interview_score`) | Avg |r| (All 6 Targets) |
|---|---|---|
| `openness` | 0.6837 | 0.6397 |
| `overall_personality` | 0.6471 | 0.6786 |
| `extraversion` | 0.6412 | 0.6269 |
| `agreeableness` | 0.6249 | 0.6507 |
| `conscientiousness` | 0.5235 | 0.5872 |
| `Total_Words` | -0.3736 | 0.3700 |
| `Duration_Sec` | -0.3486 | 0.3443 |
| `Pause_Count` | -0.3388 | 0.3342 |
| `Total_Silence_Sec` | -0.2681 | 0.2684 |
| `Spectral_Contrast` | -0.2601 | 0.2564 |

### Bottom 10 Weakest Predictors
| Feature Name | Pearson r (vs `interview_score`) | Avg |r| (All 6 Targets) |
|---|---|---|
| `hand_speed_mean` | 0.0121 | 0.0109 |
| `head_centering_score_mean` | -0.0118 | 0.0153 |
| `mouth_frown_std` | 0.0108 | 0.0196 |
| `MFCC_2` | -0.0079 | 0.0169 |
| `posture_shift_count` | -0.0043 | 0.0136 |
| `agitation_score` | -0.0036 | 0.0128 |
| `eye_openness_mean` | 0.0034 | 0.0116 |
| `shoulder_slope_var` | 0.0027 | 0.0043 |
| `Mean_ZCR` | 0.0015 | 0.0180 |
| `Mean_Pitch_Hz` | -0.0009 | 0.0277 |

### What We Learn From Correlation Analysis
1. **Target Coherence:** All 6 targets correlate moderately to strongly with each other (r = 0.60 to 0.77). This proves evaluator scores across different criteria reflect an underlying common quality signal.
2. **Psychometrics Dominance:** Big Five personality scores (`openness`, `extraversion`, `agreeableness`, `conscientiousness`) are the strongest linear predictors in the dataset (r = +0.52 to +0.68). Candidates evaluated as highly open, extraverted, and agreeable receive higher interview scores. `neuroticism` correlates negatively (r = -0.2305).
3. **Brevity and Fluency Signal:** Audio temporal and fluency metrics show consistent negative correlations with performance scores: `Total_Words` (r = -0.3736), `Duration_Sec` (r = -0.3486), `Pause_Count` (r = -0.3388), and `Total_Silence_Sec` (r = -0.2681). Candidates who give concise, structured answers with fewer hesitations score higher. Verbose, rambling responses receive lower ratings.
4. **Vocal & Facial Signals:** Higher vocal energy (`Mean_Energy` r = -0.1900) and higher eyebrow variation (`brow_raise_std` r = -0.1679) correlate with lower ratings, reflecting potential tension or over-exertion. Maintaining a calm neutral expression (`emotion_neutral_mean` r = +0.1512) correlates positively with scores.

---

## 4. Multicollinearity and Redundancy Audit (|r| > 0.85)

| Feature A | Feature B | Correlation (|r|) |
|---|---|---|
| `shoulder_width_mean` | `engagement_score` | 0.9998 |
| `posture_shift_count` | `agitation_score` | 0.9998 |
| `smile_score_mean` | `emotion_happy_mean` | 0.9964 |
| `emotion_sad_mean` | `emotion_fear_mean` | 0.9871 |
| `frown_score_mean` | `emotion_angry_mean` | 0.9844 |
| `brow_raise_mean` | `emotion_surprise_mean` | 0.9767 |
| `Spectral_Centroid` | `Spectral_Rolloff` | 0.9743 |
| `Duration_Sec` | `Total_Words` | 0.9351 |
| `Mean_Energy` | `Std_Energy` | 0.9329 |
| `gaze_ratio_mean` | `gaze_deviation_mean` | 0.9132 |
| `mouth_frown_mean` | `mouth_frown_std` | 0.9025 |
| `Duration_Sec` | `Pause_Count` | 0.8795 |
| `jaw_open_mean` | `jaw_open_std` | 0.8782 |
| `crossed_arms_score` | `engagement_score` | 0.8616 |
| `shoulder_width_mean` | `crossed_arms_score` | 0.8601 |
| `Total_Words` | `Pause_Count` | 0.8585 |
| `brow_raise_mean` | `emotion_fear_mean` | 0.8538 |

### What We Learn From Multicollinearity
1. **Synthetic Derived Features:** `engagement_score` and `agitation_score` are exact linear transformations of `shoulder_width_mean` and `posture_shift_count` (r = 0.9998). Including both adds zero new information and causes perfect multicollinearity.
2. **Facial AU vs Emotion Duplication:** OpenFace Action Units (`smile_score_mean`, `frown_score_mean`, `brow_raise_mean`) correlate almost perfectly (r = 0.97 to 0.99) with deep-learning emotion probabilities (`emotion_happy_mean`, `emotion_angry_mean`, `emotion_surprise_mean`).

### Engineering Decisions and Options
- **Decision:** Drop 10 redundant features (`engagement_score`, `agitation_score`, `emotion_happy_mean`, `emotion_fear_mean`, `emotion_angry_mean`, `emotion_surprise_mean`, `Spectral_Rolloff`, `gaze_deviation_mean`, `mouth_frown_std`, `jaw_open_std`).
- **Impact:** Eliminates matrix singularity risk in linear models (Ridge/Lasso), stabilizes feature importance in tree models, and reduces dimensionality from 75 to 65 predictive features without losing predictive signal.

---

## 5. Low-Variance and Near-Constant Feature Audit (< 1e-4)

| Feature Name | Variance | Action |
|---|---|---|
| `mouth_frown_mean` | 5.31e-05 | DROP (near-zero variance) |
| `shoulder_width_var` | 5.37e-07 | DROP (near-zero variance) |
| `nose_shoulder_dist_var` | 3.54e-07 | DROP (near-zero variance) |
| `core_speed_mean` | 1.63e-05 | DROP (near-zero variance) |

### What We Learn From Low Variance
1. **Static Metrics:** Features like `mouth_frown_mean` and `shoulder_width_var` remain virtually constant across almost all candidates.
2. **Zero Discriminative Power:** Near-constant features add noise to distance-based models (KNN, SVM) and waste splitting nodes in tree models.

### Engineering Decisions and Options
- **Decision:** Drop all 4 near-zero variance features prior to scaling and model training.

---

## 6. Actionable Feature Engineering and Preprocessing Roadmap

### Step 1: Missing Value Imputation
- Impute missing `transcript` entries with empty strings (`""`).

### Step 2: Feature Elimination List (14 Features Total)
- **Drop Near-Zero Variance (4):** `mouth_frown_mean`, `shoulder_width_var`, `nose_shoulder_dist_var`, `core_speed_mean`.
- **Drop High Collinear Pairs (10):** `engagement_score`, `agitation_score`, `emotion_happy_mean`, `emotion_fear_mean`, `emotion_angry_mean`, `emotion_surprise_mean`, `Spectral_Rolloff`, `gaze_deviation_mean`, `mouth_frown_std`, `jaw_open_std`.
- **Exclude Metadata (7):** `id`, `file_name`, `question_id`, `question`, `user_no`, `duration_label`, `transcript`.

### Step 3: Feature Transformation List (8 Features)
- Apply `np.log1p` transformation to: `Total_Silence_Sec`, `Silence_Ratio`, `Pause_Count`, `Avg_Pause_Duration_Sec`, `Mean_Energy`, `Filler_Word_Count`, `hand_speed_mean`, `posture_shift_count`.

### Step 4: Scaling and Normalization
- Scale the clean 48-feature set using `RobustScaler` (or `StandardScaler`).

### Step 5: Model Experimentation Roadmap
1. **Baseline 1 (Linear):** Ridge / Lasso Regression on clean structured features.
2. **Baseline 2 (Tree Ensembles):** XGBoost / LightGBM Regressors with hyperparameter tuning.
3. **Multi-Output Model:** Multi-output LightGBM Regressor predicting all 6 performance targets simultaneously.
4. **Multimodal Deep Learning:** Concatenate structured multimodal features with BERT transcript embeddings for late-fusion performance prediction.
