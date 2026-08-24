# EDA Report — Multimodal Interview Performance Dataset (v2)
**Notebook:** `notebooks/eda/interview_analyzer_eda.ipynb`
**Dataset:** `data/features/merged_features.csv`
**Date:** 2026-08-25

---

## 1. Executive Summary

| Metric | Value |
|---|---|
| Dataset Shape | 2,011 rows x 88 columns |
| Unique Candidates | 331 |
| Avg Responses/Candidate | 6.1 |
| Predictive Features | 75 |
| Performance Targets | 6 |
| Missing Values | 13 (transcript only, 0.65%) |
| Duplicate Rows | 0 |
| Collinear Pairs (|r| > 0.85) | 17 |
| Low-Variance Features | 4 |
| High-Skewness Features | 29 (39%) |

**Outcome:** Dataset is clean and suitable for modelling. Key findings: conciseness is the dominant predictor, personality traits are the strongest features, and 14 features should be dropped due to redundancy or near-zero variance.

---

## 2. Notebook Structure

21 sections organized into 7 phases:

| Phase | Sections | Description |
|---|---|---|
| Setup & Loading | 1-2 | Environment config, structural audit |
| Column Taxonomy | 3-5 | Feature grouping, target profiling, inter-target correlation |
| Modality Analysis | 6-11 | Audio, MFCC, facial gaze, facial AUs, emotions, posture |
| Feature Quality | 12-15 | Correlation ranking, multicollinearity, low-variance, outliers |
| Categorical & Text | 16-18 | Duration labels, candidate distribution, transcript analysis |
| Transformation | 19-20 | Skewness guide, feature engineering decision matrix |
| Summary | 21 | Findings and next steps |

---

## 3. Key Findings

### 3.1 Conciseness Dominates Performance

| Feature | Correlation with interview_score |
|---|---|
| Total_Words | r = -0.374 |
| Duration_Sec | r = -0.349 |
| Pause_Count | r = -0.339 |
| Total_Silence_Sec | r = -0.268 |

Shorter, more concise answers systematically receive higher scores across all 6 performance dimensions. Verbose responses with more pauses and silence are penalized.

### 3.2 Personality Traits Are Strongest Predictors

| Feature | Avg |r| with All Targets |
|---|---|
| openness | 0.68 |
| agreeableness | 0.65 |
| extraversion | 0.63 |
| overall_personality | 0.62 |
| conscientiousness | 0.58 |

Psychometric Big Five features dominate the feature-target correlation ranking.

### 3.3 Target Correlation Matrix

All 6 targets are strongly correlated (r = 0.62 to 0.77):
- `overall_performance` <-> `confidence_score`: r = 0.77
- `overall_performance` <-> `answer_score`: r = 0.75
- `interview_score` <-> `facial_expression`: r = 0.74

Multi-task/multi-output model architectures are feasible.

### 3.4 Dataset Quality

- **Zero duplicates** across all 2,011 rows
- **99.76% face detection rate** — high video tracking quality
- **Only 13 missing values** (all in transcript, 0.65%)
- **84 numerical columns** with zero missing values

---

## 4. Feature Redundancy Analysis

### 4.1 Collinear Feature Pairs (|r| > 0.85)

| Feature 1 | Feature 2 | Correlation |
|---|---|---|
| shoulder_width_mean | engagement_score | 0.9998 |
| posture_shift_count | agitation_score | 0.9998 |
| smile_score_mean | emotion_happy_mean | 0.9964 |
| gaze_ratio_mean | gaze_deviation_mean | 0.9867 |
| mouth_frown_mean | mouth_frown_std | 0.9589 |
| jaw_open_mean | jaw_open_std | 0.9123 |
| Spectral_Centroid | Spectral_Rolloff | 0.8934 |
| emotion_fear_mean | emotion_sad_mean | 0.8756 |
| emotion_angry_mean | emotion_disgust_mean | 0.8634 |
| emotion_surprise_mean | emotion_neutral_mean | 0.8512 |

### 4.2 Low-Variance Features (< 1e-4)

| Feature | Variance | Decision |
|---|---|---|
| mouth_frown_mean | 0.000053 | DROP |
| shoulder_width_var | 0.000012 | DROP |
| nose_shoulder_dist_var | 0.000008 | DROP |
| core_speed_mean | 0.000003 | DROP |

---

## 5. Outlier Analysis

### 5.1 Severe Outlier Rates (>10%)

| Feature | Outlier % | Method |
|---|---|---|
| Pitch_Range_Hz | 15.86% | IQR |
| agitation_score | 13.63% | IQR |
| posture_shift_count | 13.63% | IQR |
| mouth_frown_mean | 13.43% | IQR |
| frown_score_mean | 12.23% | IQR |
| mouth_frown_std | 12.03% | IQR |
| shoulder_width_var | 11.19% | IQR |
| emotion_angry_mean | 11.19% | IQR |
| emotion_happy_mean | 10.49% | IQR |
| Std_Energy | 10.49% | IQR |

### 5.2 Moderate Outlier Rates (5-10%)

14 additional features with moderate outlier rates including `smile_score_mean`, `Filler_Rate_Per_Min`, `hand_speed_mean`, `Std_Pitch_Hz`, and others.

---

## 6. Skewness Analysis

29 features (39%) exhibit high skewness (|skewness| > 2.0):

| Feature | Skewness |
|---|---|
| shoulder_slope_var | 42.24 |
| gaze_stability_std | 21.80 |
| face_detected_ratio | -21.14 |
| hand_to_face_touches | 13.87 |
| Mean_Pitch_Hz | 11.33 |
| Pitch_Range_Hz | 2.31 |
| Total_Words | 0.71 |
| Speech_Rate_WPM | 6.60 |

**Recommendation:** Apply `np.log1p` transformation to right-skewed features.

---

## 7. Categorical Analysis

### Duration Label Impact

| Duration | Avg interview_score | Count |
|---|---|---|
| short | +0.547 | 680 |
| medium | -0.132 | 754 |
| long | -0.472 | 577 |

Short responses consistently score higher across all 6 performance targets.

### Candidate Distribution

- 331 unique candidates
- Range: 1 to 17 responses per candidate
- Median: 6 responses
- **GroupKFold cross-validation by `user_no` is mandatory** to prevent data leakage

---

## 8. Feature Engineering Decision Matrix

### 8.1 Features to DROP (14 features)

**Near-zero variance (4):**
- `mouth_frown_mean`
- `shoulder_width_var`
- `nose_shoulder_dist_var`
- `core_speed_mean`

**Highly collinear (10):**
- `engagement_score` (collinear with shoulder_width_mean)
- `agitation_score` (collinear with posture_shift_count)
- `emotion_happy_mean` (collinear with smile_score_mean)
- `emotion_fear_mean` (collinear with emotion_sad_mean)
- `emotion_angry_mean` (collinear with emotion_disgust_mean)
- `emotion_surprise_mean` (collinear with emotion_neutral_mean)
- `Spectral_Rolloff` (collinear with Spectral_Centroid)
- `gaze_deviation_mean` (collinear with gaze_ratio_mean)
- `mouth_frown_std` (collinear with mouth_frown_mean)
- `jaw_open_std` (collinear with jaw_open_mean)

### 8.2 Features to TRANSFORM (log1p)

- `Total_Silence_Sec`
- `Silence_Ratio`
- `Pause_Count`
- `Avg_Pause_Duration_Sec`
- `Mean_Energy`
- `Filler_Word_Count`
- `hand_speed_mean`
- `posture_shift_count`

### 8.3 Scaling Strategy

- **Tree models (XGBoost, LightGBM, RF):** No scaling needed
- **Linear models (Ridge, Lasso):** Apply `RobustScaler` after train/test split
- **Neural networks:** Apply `StandardScaler` after train/test split

### 8.4 Cross-Validation Strategy

```python
GroupKFold(n_splits=5, groups=df['user_no'])
```

Mandatory to prevent candidate data leakage.

---

## 9. Top Feature Predictors (Ranked by Avg |r|)

| Rank | Feature | Avg |r| | Domain |
|---|---|---|---|
| 1 | openness | 0.68 | Personality |
| 2 | agreeableness | 0.65 | Personality |
| 3 | extraversion | 0.63 | Personality |
| 4 | overall_personality | 0.62 | Personality |
| 5 | conscientiousness | 0.58 | Personality |
| 6 | Total_Words | 0.37 | Audio |
| 7 | Duration_Sec | 0.35 | Audio |
| 8 | Pause_Count | 0.34 | Audio |
| 9 | Total_Silence_Sec | 0.27 | Audio |
| 10 | Spectral_Contrast | 0.19 | Audio |

---

## 10. Recommendations and Next Steps

### Immediate (Preprocessing)

1. **Drop 14 redundant/low-variance features** as outlined in Section 8.1
2. **Apply log1p transformation** to 8 right-skewed features
3. **Encode duration_label** as ordinal (short=0, medium=1, long=2)
4. **Impute transcript** missing values with empty string for NLP pipeline

### Feature Engineering

1. **Candidate-level aggregation** — Mean, std, min, max across questions per user_no
2. **Interaction features** — Personality x Audio, Facial x Posture
3. **NLP embeddings** — TF-IDF or BERT for transcript semantic quality
4. **Composite targets** — Mean of all 6 targets for single-output baseline

### Modelling

1. **Baseline models** — Ridge, Lasso, ElasticNet
2. **Tree models** — XGBoost, LightGBM, Random Forest
3. **Multi-modal fusion** — Concatenate audio+facial+posture+personality
4. **Multi-task learning** — Shared backbone, 6 prediction heads
5. **Group-aware splitting** — GroupKFold(groups=user_no)

---

*Report generated from executed notebook: `notebooks/eda/interview_analyzer_eda.ipynb`*
