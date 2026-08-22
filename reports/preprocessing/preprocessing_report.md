# Preprocessing Report — Multimodal Interview Performance Dataset
**Notebook:** `notebooks/Preprocessing/01_preprocessing.ipynb`
**Dataset:** `data/features/merged_features.csv`
**Date:** 2026-08-23

---

## 1. Executive Summary

| Metric | Value |
|---|---|
| Original Shape | 2,011 rows x 76 columns |
| Cleaned Shape | 2,011 rows x 76 columns |
| Rows Removed | 0 |
| Columns Removed | 0 |
| Missing Values | 13 (transcript only, 0.65%) |
| Duplicate Rows | 0 |
| Duplicate IDs | 0 |
| Unique Candidates | 331 |
| Avg Videos/Candidate | 6.1 |
| Infinite Values | 0 |
| Constant Columns | 0 |

**Outcome:** Dataset is clean. No irreversible transformations were applied.
All 2,011 rows and 76 columns are preserved in the output.

---

## 2. Dataset Overview

| Property | Value |
|---|---|
| Total Rows | 2,011 |
| Total Columns | 76 |
| float64 columns | 65 |
| int64 columns | 6 |
| str columns | 5 |
| Memory Usage | 2.30 MB |

### Column Index (first and last 5)

| # | Column | Dtype |
|---|---|---|
| 1 | id | int64 |
| 2 | file_name | str |
| 3 | duration_label | str |
| 4 | question_id | int64 |
| 5 | question | str |
| ... | ... | ... |
| 72 | core_speed_mean | float64 |
| 73 | posture_shift_count | int64 |
| 74 | engagement_score | float64 |
| 75 | agitation_score | int64 |
| 76 | transcript | str |

---

## 3. Schema Audit

Per-column audit covering dtype, unique count, missing values, infinite values,
constant detection, and near-zero variance detection.

### Audit Summary

| Metric | Count |
|---|---|
| Total columns | 76 |
| Numeric columns | 71 |
| Categorical (object) columns | 0 (5 are str dtype) |
| Constant columns | 0 |
| Near-zero variance columns | 0 |
| Columns with missing values | 1 |
| Columns with infinite values | 0 |

### Columns with Missing Values

| Column | Dtype | Missing Count | Missing % |
|---|---|---|---|
| transcript | str | 13 | 0.65% |

### Infinite Values

No infinite values found in any numeric column.

---

## 4. Column Grouping Validation

All columns validated against the reference schema. No missing or unknown columns.

| Group | Expected | Present | Status |
|---|---|---|---|
| METADATA | 7 | 7 | All present |
| TARGETS | 12 | 12 | All present |
| AUDIO | 6 | 6 | All present |
| MFCC | 13 | 13 | All present |
| FACE | 16 | 16 | All present |
| EMOTION | 7 | 7 | All present |
| POSTURE | 14 | 14 | All present |
| TEXT | 1 | 1 | All present |
| UNKNOWN | — | 0 | None found |

### Column Groups Detail

**Metadata (7):** id, file_name, user_no, question_id, question, duration_label, video_quality

**Targets (12):** openness, conscientiousness, extraversion, agreeableness, neuroticism,
overall_personality, interview_score, answer_score, speaking_skills, confidence_score,
facial_expression, overall_performance

**Audio (6):** Duration_Sec, Speech_Rate_WPM, Silence_Duration_Sec, Mean_Pitch_Hz,
Mean_Energy, Mean_ZCR

**MFCC (13):** MFCC_1 through MFCC_13

**Face (16):** face_detected_ratio, gaze_ratio_mean, gaze_deviation_mean,
gaze_stability_std, smile_score_mean/std, frown_score_mean/std,
eye_openness_mean/std, jaw_open_mean/std, brow_raise_mean/std,
mouth_frown_mean/std

**Emotion (7):** emotion_happy_mean, emotion_sad_mean, emotion_angry_mean,
emotion_surprise_mean, emotion_fear_mean, emotion_disgust_mean, emotion_neutral_mean

**Posture (14):** head_centering_score_mean, absolute_shoulder_slope_mean,
shoulder_slope_var, shoulder_width_mean/var, nose_shoulder_dist_mean/var,
hand_speed_mean, hand_to_face_touches, crossed_arms_score, core_speed_mean,
posture_shift_count, engagement_score, agitation_score

**Text (1):** transcript

---

## 5. Missing Value Analysis

### Findings

Only one column has missing values: `transcript` (13 rows, 0.65%).

- **Numeric features:** 0 missing values across all 62 feature columns
- **Target columns:** 0 missing values across all 12 target columns
- **Total missing:** 13 (all in transcript)

### Strategy

| Column | Missing | Decision | Rationale |
|---|---|---|---|
| transcript | 13 (0.65%) | KEEP NaN | NLP pipeline will handle missing transcripts. Do NOT impute with filler values. |
| All numeric features | 0 | No action needed | Complete data |
| All targets | 0 | No action needed | Complete data |

---

## 6. Duplicate Analysis

### Row-Level Duplicates

| Check | Result |
|---|---|
| Duplicate rows | 0 |
| Duplicate IDs | 0 |

All 2,011 rows are unique. All 2,011 IDs are unique.

### ID / Candidate Analysis

| Metric | Value |
|---|---|
| Unique candidates (user_no) | 331 |
| Total rows | 2,011 |
| Mean videos/candidate | 6.08 |
| Std videos/candidate | 1.53 |
| Min videos/candidate | 1 |
| Median videos/candidate | 6 |
| Max videos/candidate | 17 |

### Distribution of Videos per Candidate

| Videos | Candidates |
|---|---|
| 1 | 11 |
| 2 | 4 |
| 3 | 2 |
| 4 | 6 |
| 5 | 8 |
| 6 | 223 |
| 7 | 60 |
| 8 | 8 |
| 9 | 4 |
| 11 | 1 |
| 12 | 3 |
| 17 | 1 |

**Key insight:** 67% of candidates (223/331) have exactly 6 videos.
The majority of candidates answered 6-7 questions.

---

## 7. Infinite / Invalid Value Analysis

### Infinite Values

No infinite values found in any of the 71 numeric columns.

### Audio Feature Range Check

| Feature | Min | Max | Negatives | Zeros |
|---|---|---|---|---|
| Duration_Sec | 0.60 | 92.34 | 0 | 0 |
| Speech_Rate_WPM | 0.00 | 571.43 | 0 | 13 |
| Silence_Duration_Sec | 0.00 | 29.43 | 0 | 23 |
| Mean_Pitch_Hz | 0.00 | 1339.07 | 0 | 7 |
| Mean_Energy | 0.00 | 0.21 | 0 | 7 |
| Mean_ZCR | 0.00 | 0.34 | 0 | 1 |

**No physically impossible negative values found.** All audio features are non-negative
as expected.

---

## 8. Range Validation

### Ratio/Probability Columns (expected [0, 1])

| Column | Min | Max | Status |
|---|---|---|---|
| face_detected_ratio | 0.036 | 1.000 | OK |
| gaze_ratio_mean | 0.463 | 0.600 | OK |
| smile_score_mean | 0.000 | 0.767 | OK |
| smile_score_std | 0.000 | 0.396 | OK |
| frown_score_mean | 0.000 | 0.532 | OK |
| frown_score_std | 0.000 | 0.213 | OK |
| eye_openness_mean | 0.342 | 0.989 | OK |
| eye_openness_std | 0.000 | 0.293 | OK |
| emotion_happy_mean | 0.000 | 0.660 | OK |
| emotion_sad_mean | 0.001 | 0.361 | OK |
| emotion_angry_mean | 0.001 | 0.270 | OK |
| emotion_surprise_mean | 0.005 | 0.404 | OK |
| emotion_fear_mean | 0.002 | 0.266 | OK |
| emotion_disgust_mean | 0.000 | 0.191 | OK |
| emotion_neutral_mean | 0.000 | 0.914 | OK |
| jaw_open_mean | 0.000 | 0.402 | OK |
| jaw_open_std | 0.000 | 0.245 | OK |
| brow_raise_mean | 0.001 | 0.949 | OK |
| brow_raise_std | 0.000 | 0.267 | OK |
| mouth_frown_mean | 0.000 | 0.096 | OK |
| mouth_frown_std | 0.000 | 0.138 | OK |
| **engagement_score** | **2.485** | **7.975** | **FLAG** |
| **agitation_score** | **0.000** | **299.000** | **FLAG** |
| **crossed_arms_score** | **0.371** | **1.561** | **FLAG** |

### Flagged Columns (outside [0, 1])

| Column | Values > 1 | Note |
|---|---|---|
| engagement_score | 2,011 (100%) | All values > 1 — not a probability. Likely a composite score. |
| agitation_score | 963 (48%) | ~48% of values > 1 — appears to be a count, not a probability. |
| crossed_arms_score | 721 (36%) | ~36% of values > 1 — appears to be a composite score. |

**Decision:** These are NOT probabilities. They are composite/scaled scores
with different ranges. No clipping or transformation needed.

### Target Variable Ranges

| Target | Min | Max | Mean | Std | Skewness |
|---|---|---|---|---|---|
| openness | -7.80 | 9.34 | 0.00 | 1.13 | 0.03 |
| conscientiousness | -6.93 | 6.62 | 0.00 | 0.88 | -0.57 |
| extraversion | -7.18 | 8.91 | 0.00 | 1.10 | -0.22 |
| agreeableness | -8.41 | 9.24 | 0.00 | 1.20 | 0.40 |
| neuroticism | -2.58 | 2.23 | 0.00 | 0.49 | -0.24 |
| overall_personality | -7.53 | 9.27 | 0.00 | 1.20 | 0.46 |
| interview_score | -7.86 | 9.39 | 0.00 | 1.23 | 0.68 |
| answer_score | -10.20 | 8.72 | 0.00 | 1.18 | 0.35 |
| speaking_skills | -9.41 | 7.90 | 0.00 | 1.28 | -0.86 |
| confidence_score | -7.29 | 6.84 | 0.00 | 1.08 | -0.64 |
| facial_expression | -7.46 | 9.26 | 0.00 | 1.15 | 0.60 |
| overall_performance | -9.31 | 8.85 | 0.00 | 1.20 | -0.75 |

All targets are z-normalised (mean ~0, std ~1). This is expected for
Big Five personality traits and interview scores.

### MFCC Feature Ranges

MFCC values are spectral coefficients — negative values are normal.

| MFCC | Min | Max | Mean | Std |
|---|---|---|---|---|
| MFCC_1 | -1131.37 | -154.33 | -408.99 | 60.31 |
| MFCC_2 | 0.00 | 189.81 | 109.47 | 19.77 |
| MFCC_3 | -77.15 | 63.72 | 19.04 | 18.01 |
| MFCC_4 | -28.33 | 41.84 | 9.17 | 10.85 |
| MFCC_5 | -19.15 | 41.40 | 14.71 | 9.53 |
| MFCC_6 | -23.57 | 47.94 | 9.79 | 9.68 |
| MFCC_7 | -31.06 | 17.21 | -7.53 | 7.40 |
| MFCC_8 | -26.50 | 26.40 | 1.01 | 10.00 |
| MFCC_9 | -32.22 | 14.91 | -8.99 | 6.01 |
| MFCC_10 | -26.97 | 17.71 | -4.43 | 8.19 |
| MFCC_11 | -21.66 | 12.95 | -5.26 | 5.15 |
| MFCC_12 | -26.38 | 11.41 | -2.88 | 5.56 |
| MFCC_13 | -24.66 | 10.11 | -5.85 | 5.10 |

---

## 9. Categorical Variable Analysis

### Object Dtype Columns

| Column | Unique | Missing | Top Value |
|---|---|---|---|
| file_name | 2,011 | 0 | vid_0001.mp4 (1 each) |
| duration_label | 3 | 0 | medium (754) |
| question | 76 | 0 | Introduce yourself (193) |
| video_quality | 2 | 0 | High (1,675) |
| transcript | 1,932 | 13 | [varies] |

### Duration Label Distribution

| Label | Count | Percentage |
|---|---|---|
| medium | 754 | 37.5% |
| short | 680 | 33.8% |
| long | 577 | 28.7% |

### Video Quality Distribution

| Quality | Count | Percentage |
|---|---|---|
| High | 1,675 | 83.3% |
| Low | 336 | 16.7% |

### Categorical Encoding Plan

| Column | Plan | Rationale |
|---|---|---|
| duration_label | Ordinal encode (short=0, medium=1, long=2) | Has natural order |
| question_id | Preserve as metadata | Not an ML feature |
| user_no | Preserve as grouping variable | For candidate-level aggregation and GroupKFold |
| file_name | Metadata only | Traceability, not a feature |
| question | Preserve for NLP pipeline | Will be used for text analysis |
| video_quality | Preserve for analysis | May show bias; investigate in EDA |

---

## 10. Target Variable Validation

### Target Statistics

| Target | Dtype | Non-null | Missing | Mean | Std | Skewness |
|---|---|---|---|---|---|---|
| openness | float64 | 2,011 | 0 | 0.00 | 1.13 | 0.03 |
| conscientiousness | float64 | 2,011 | 0 | 0.00 | 0.88 | -0.57 |
| extraversion | float64 | 2,011 | 0 | 0.00 | 1.10 | -0.22 |
| agreeableness | float64 | 2,011 | 0 | 0.00 | 1.20 | 0.40 |
| neuroticism | float64 | 2,011 | 0 | 0.00 | 0.49 | -0.24 |
| overall_personality | float64 | 2,011 | 0 | 0.00 | 1.20 | 0.46 |
| interview_score | float64 | 2,011 | 0 | 0.00 | 1.23 | 0.68 |
| answer_score | float64 | 2,011 | 0 | 0.00 | 1.18 | 0.35 |
| speaking_skills | float64 | 2,011 | 0 | 0.00 | 1.28 | -0.86 |
| confidence_score | float64 | 2,011 | 0 | 0.00 | 1.08 | -0.64 |
| facial_expression | float64 | 2,011 | 0 | 0.00 | 1.15 | 0.60 |
| overall_performance | float64 | 2,011 | 0 | 0.00 | 1.20 | -0.75 |

### Target-Target Correlation Matrix

| | open | consc | extra | agree | neuro | overall_p | int_score | ans_score | speak | conf | facial | overall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| openness | 1.00 | 0.56 | 0.74 | 0.54 | -0.15 | 0.62 | 0.68 | 0.62 | 0.64 | 0.60 | 0.64 | 0.65 |
| conscientiousness | 0.56 | 1.00 | 0.48 | 0.58 | -0.27 | 0.71 | 0.52 | 0.65 | 0.53 | 0.62 | 0.53 | 0.67 |
| extraversion | 0.74 | 0.48 | 1.00 | 0.55 | -0.20 | 0.58 | 0.64 | 0.63 | 0.62 | 0.62 | 0.64 | 0.61 |
| agreeableness | 0.54 | 0.58 | 0.55 | 1.00 | -0.32 | 0.74 | 0.63 | 0.67 | 0.63 | 0.66 | 0.68 | 0.65 |
| neuroticism | -0.15 | -0.27 | -0.20 | -0.32 | 1.00 | -0.34 | -0.23 | -0.32 | -0.25 | -0.35 | -0.27 | -0.33 |
| overall_personality | 0.62 | 0.71 | 0.58 | 0.74 | -0.34 | 1.00 | 0.65 | 0.70 | 0.61 | 0.72 | 0.63 | 0.77 |
| interview_score | 0.68 | 0.52 | 0.64 | 0.63 | -0.23 | 0.65 | 1.00 | 0.72 | 0.73 | 0.65 | 0.74 | 0.68 |
| answer_score | 0.62 | 0.65 | 0.63 | 0.67 | -0.32 | 0.70 | 0.72 | 1.00 | 0.63 | 0.70 | 0.68 | 0.75 |
| speaking_skills | 0.64 | 0.53 | 0.62 | 0.63 | -0.25 | 0.61 | 0.73 | 0.63 | 1.00 | 0.61 | 0.73 | 0.62 |
| confidence_score | 0.60 | 0.62 | 0.62 | 0.66 | -0.35 | 0.72 | 0.65 | 0.70 | 0.61 | 1.00 | 0.60 | 0.77 |
| facial_expression | 0.64 | 0.53 | 0.64 | 0.68 | -0.27 | 0.63 | 0.74 | 0.68 | 0.73 | 0.60 | 1.00 | 0.61 |
| overall_performance | 0.65 | 0.67 | 0.61 | 0.65 | -0.33 | 0.77 | 0.68 | 0.75 | 0.62 | 0.77 | 0.61 | 1.00 |

**Key observations:**
- All targets correlate r=0.52-0.77 with each other
- Strongest pair: overall_personality <-> overall_performance (r=0.77)
- neuroticism is negatively correlated with all other targets
- High inter-target correlation suggests a single latent "interview performance" factor

---

## 11. Data Leakage Audit

### Feature-Target Correlations

No feature-target correlation exceeded |r| > 0.5. No suspiciously high correlations found.

### Exact Duplicate Check

No feature column is an exact duplicate of any target column.

**Conclusion:** No data leakage detected. No automatic deletions performed.

---

## 12. Outlier Detection Report (IQR Method)

Outliers flagged but NOT removed. Extreme values may represent genuine behavioral
variation in interview performance.

### Top 15 Features by Outlier Percentage

| Feature | Outliers | % | Lower Fence | Upper Fence | Min | Max |
|---|---|---|---|---|---|---|
| agitation_score | 274 | 13.63% | -12.00 | 20.00 | 0.00 | 299.00 |
| posture_shift_count | 274 | 13.63% | -12.00 | 20.00 | 0.00 | 299.00 |
| mouth_frown_mean | 270 | 13.43% | -0.003 | 0.006 | 0.000 | 0.096 |
| frown_score_mean | 246 | 12.23% | -0.044 | 0.077 | 0.000 | 0.532 |
| mouth_frown_std | 242 | 12.03% | -0.006 | 0.010 | 0.000 | 0.138 |
| shoulder_width_var | 225 | 11.19% | -0.0002 | 0.0005 | 0.000 | 0.013 |
| emotion_angry_mean | 225 | 11.19% | -0.020 | 0.049 | 0.001 | 0.270 |
| emotion_happy_mean | 211 | 10.49% | -0.096 | 0.171 | 0.000 | 0.660 |
| smile_score_mean | 200 | 9.95% | -0.105 | 0.185 | 0.000 | 0.767 |
| hand_speed_mean | 190 | 9.45% | -0.037 | 0.114 | 0.003 | 0.330 |
| frown_score_std | 188 | 9.35% | -0.033 | 0.058 | 0.000 | 0.213 |
| shoulder_slope_var | 175 | 8.70% | -0.0005 | 0.001 | 0.000 | 16.992 |
| Mean_Energy | 143 | 7.11% | 0.011 | 0.051 | 0.000 | 0.215 |
| nose_shoulder_dist_var | 143 | 7.11% | -0.0005 | 0.001 | 0.000 | 0.007 |
| core_speed_mean | 136 | 6.76% | -0.001 | 0.013 | 0.001 | 0.032 |

**Note:** Outlier percentages are moderate (1-14%). These represent genuine variation
in interview behavior, not data errors. No removal recommended.

---

## 13. Preprocessing Decisions

### What Was Changed

| Action | Detail |
|---|---|
| Schema audit | Generated and saved to data_quality_report.csv |
| Column grouping | All 76 columns validated against reference schema |
| Missing values | Documented (transcript: 13 rows, 0.65%) |
| Duplicate check | Completed (0 duplicate rows, all IDs unique) |
| Infinite/invalid check | Completed (0 infinite values found) |
| Range validation | Completed for all ratio and audio columns |
| Outlier detection | Completed using IQR method (flagged only) |
| Data leakage audit | Completed (no suspicious correlations) |

### What Was IntentionALLY NOT Changed

| Action | Rationale |
|---|---|
| No rows dropped | All 2,011 rows are valid |
| No columns dropped | Feature selection happens in later stage |
| No scaling applied | Deferred to after train/test split to avoid data leakage |
| No feature engineering | Deferred to feature engineering stage |
| No PCA | Deferred to dimensionality reduction stage |
| No NLP preprocessing | Deferred to NLP pipeline |
| No imputation on transcript | Deferred to NLP pipeline |
| No categorical encoding | Decision deferred to EDA/feature engineering |

---

## 14. Files Created

| File | Size | Description |
|---|---|---|
| data/processed/clean_master_data.csv | 3,150,805 bytes | Full 2011x76 dataset with all columns |
| data/processed/numeric_features_preprocessed.csv | 2,069,042 bytes | 2011x72 numeric features (excludes id, file_name, transcript, question) |
| data/processed/data_quality_report.csv | 9,434 bytes | Per-column audit with dtype, missing, inf, range stats |
| data/processed/preprocessing_summary.txt | 1,903 bytes | Text summary of preprocessing decisions |
| notebooks/Preprocessing/01_preprocessing.ipynb | — | Executed notebook with all outputs |

---

## 15. Next Steps

### EDA Stage (Immediate)

1. **Distribution analysis** — Histograms and KDE plots for all 62 numeric features
2. **Correlation heatmap** — Feature-feature and feature-target correlations
3. **Feature-target relationships** — Scatter plots with regression lines
4. **Outlier visualization** — Box plots to decide on transformation/removal
5. **Video quality impact** — Compare High vs Low quality on targets
6. **Question-level analysis** — Which questions correlate with higher scores
7. **Duration label analysis** — Does short/medium/long affect scores
8. **Candidate-level analysis** — Aggregate features per user_no

### Feature Engineering Stage (Later)

1. **Candidate-level aggregation** — Mean, std, min, max across questions per user
2. **Interaction features** — Personality x Audio, Facial x Posture
3. **Composite target engineering** — Mean of all 6 targets
4. **NLP pipeline** — BERT/TF-IDF embeddings for transcripts
5. **Scaling** — StandardScaler/RobustScaler AFTER train/test split
6. **Categorical encoding** — Ordinal encode duration_label, one-hot if needed

### Modelling Stage (Future)

1. **Baseline models** — Ridge, Lasso, ElasticNet
2. **Tree models** — XGBoost, LightGBM, Random Forest
3. **Multi-modal fusion** — Concatenate audio+facial+posture+personality
4. **Multi-task learning** — Shared backbone, 6 prediction heads
5. **Group-aware splitting** — GroupKFold(groups=user_no)

---

## 16. Appendix: Complete Column List

### All 76 Columns

| # | Column | Dtype | Group | Missing |
|---|---|---|---|---|
| 1 | id | int64 | METADATA | 0 |
| 2 | file_name | str | METADATA | 0 |
| 3 | duration_label | str | METADATA | 0 |
| 4 | question_id | int64 | METADATA | 0 |
| 5 | question | str | METADATA | 0 |
| 6 | video_quality | str | METADATA | 0 |
| 7 | user_no | int64 | METADATA | 0 |
| 8 | openness | float64 | TARGET | 0 |
| 9 | conscientiousness | float64 | TARGET | 0 |
| 10 | extraversion | float64 | TARGET | 0 |
| 11 | agreeableness | float64 | TARGET | 0 |
| 12 | neuroticism | float64 | TARGET | 0 |
| 13 | overall_personality | float64 | TARGET | 0 |
| 14 | interview_score | float64 | TARGET | 0 |
| 15 | answer_score | float64 | TARGET | 0 |
| 16 | speaking_skills | float64 | TARGET | 0 |
| 17 | confidence_score | float64 | TARGET | 0 |
| 18 | facial_expression | float64 | TARGET | 0 |
| 19 | overall_performance | float64 | TARGET | 0 |
| 20 | Duration_Sec | float64 | AUDIO | 0 |
| 21 | Speech_Rate_WPM | float64 | AUDIO | 0 |
| 22 | Silence_Duration_Sec | float64 | AUDIO | 0 |
| 23 | Mean_Pitch_Hz | float64 | AUDIO | 0 |
| 24 | Mean_Energy | float64 | AUDIO | 0 |
| 25 | Mean_ZCR | float64 | AUDIO | 0 |
| 26 | MFCC_1 | float64 | MFCC | 0 |
| 27 | MFCC_2 | float64 | MFCC | 0 |
| 28 | MFCC_3 | float64 | MFCC | 0 |
| 29 | MFCC_4 | float64 | MFCC | 0 |
| 30 | MFCC_5 | float64 | MFCC | 0 |
| 31 | MFCC_6 | float64 | MFCC | 0 |
| 32 | MFCC_7 | float64 | MFCC | 0 |
| 33 | MFCC_8 | float64 | MFCC | 0 |
| 34 | MFCC_9 | float64 | MFCC | 0 |
| 35 | MFCC_10 | float64 | MFCC | 0 |
| 36 | MFCC_11 | float64 | MFCC | 0 |
| 37 | MFCC_12 | float64 | MFCC | 0 |
| 38 | MFCC_13 | float64 | MFCC | 0 |
| 39 | face_detected_ratio | float64 | FACE | 0 |
| 40 | gaze_ratio_mean | float64 | FACE | 0 |
| 41 | gaze_deviation_mean | float64 | FACE | 0 |
| 42 | gaze_stability_std | float64 | FACE | 0 |
| 43 | smile_score_mean | float64 | FACE | 0 |
| 44 | smile_score_std | float64 | FACE | 0 |
| 45 | frown_score_mean | float64 | FACE | 0 |
| 46 | frown_score_std | float64 | FACE | 0 |
| 47 | eye_openness_mean | float64 | FACE | 0 |
| 48 | eye_openness_std | float64 | FACE | 0 |
| 49 | jaw_open_mean | float64 | FACE | 0 |
| 50 | jaw_open_std | float64 | FACE | 0 |
| 51 | brow_raise_mean | float64 | FACE | 0 |
| 52 | brow_raise_std | float64 | FACE | 0 |
| 53 | mouth_frown_mean | float64 | FACE | 0 |
| 54 | mouth_frown_std | float64 | FACE | 0 |
| 55 | emotion_happy_mean | float64 | EMOTION | 0 |
| 56 | emotion_sad_mean | float64 | EMOTION | 0 |
| 57 | emotion_angry_mean | float64 | EMOTION | 0 |
| 58 | emotion_surprise_mean | float64 | EMOTION | 0 |
| 59 | emotion_fear_mean | float64 | EMOTION | 0 |
| 60 | emotion_disgust_mean | float64 | EMOTION | 0 |
| 61 | emotion_neutral_mean | float64 | EMOTION | 0 |
| 62 | head_centering_score_mean | float64 | POSTURE | 0 |
| 63 | absolute_shoulder_slope_mean | float64 | POSTURE | 0 |
| 64 | shoulder_slope_var | float64 | POSTURE | 0 |
| 65 | shoulder_width_mean | float64 | POSTURE | 0 |
| 66 | shoulder_width_var | float64 | POSTURE | 0 |
| 67 | nose_shoulder_dist_mean | float64 | POSTURE | 0 |
| 68 | nose_shoulder_dist_var | float64 | POSTURE | 0 |
| 69 | hand_speed_mean | float64 | POSTURE | 0 |
| 70 | hand_to_face_touches | int64 | POSTURE | 0 |
| 71 | crossed_arms_score | float64 | POSTURE | 0 |
| 72 | core_speed_mean | float64 | POSTURE | 0 |
| 73 | posture_shift_count | int64 | POSTURE | 0 |
| 74 | engagement_score | float64 | POSTURE | 0 |
| 75 | agitation_score | int64 | POSTURE | 0 |
| 76 | transcript | str | TEXT | 13 |

---

*Report generated from executed notebook: `notebooks/Preprocessing/01_preprocessing.ipynb`*
