# EDA Report — Multimodal Interview Performance Dataset
**Script:** `src/eda_merged_features.py` v2.0  
**Dataset:** `data/features/merged_features.csv`  
**Schema:** 2011 rows x 76 columns  

---
## 1. Dataset Summary

| Metric | Value |
|---|---|
| Total Records | 2,011 |
| Total Columns | 76 |
| Numerical Columns | 71 |
| Text/Cat Columns | 5 |
| Unique Candidates | 331 |
| Missing Values | transcript only (13 rows, 0.65%) |
| Duplicate Rows | 0 |

---
## 2. New Columns in v2 Schema (vs v1)

| Column | Type | Description |
|---|---|---|
| `video_quality` | Categorical | High/Low — metadata tag for video resolution |
| `answer_score` | Float | Standardised score for answer content quality |
| `speaking_skills` | Float | Standardised score for verbal delivery |
| `confidence_score` | Float | Standardised score for exhibited confidence |
| `facial_expression` | Float | Standardised score for facial expression quality |
| `overall_performance` | Float | Composite standardised overall performance score |

---
## 3. Target / Score Variable Summary

| Target | Mean | Std | Min | Max | Skew |
|---|---|---|---|---|---|
| `interview_score` | 0.0000 | 1.2301 | -7.8603 | 9.3860 | 0.6801 |
| `answer_score` | 0.0000 | 1.1755 | -10.1991 | 8.7223 | 0.3534 |
| `speaking_skills` | 0.0000 | 1.2776 | -9.4063 | 7.9015 | -0.8550 |
| `confidence_score` | 0.0000 | 1.0797 | -7.2861 | 6.8394 | -0.6404 |
| `facial_expression` | 0.0000 | 1.1510 | -7.4614 | 9.2558 | 0.5977 |
| `overall_performance` | 0.0000 | 1.2007 | -9.3111 | 8.8490 | -0.7475 |

> All targets are approximately standard-normal scaled (mean~0, std~1.0-1.3)

---
## 4. Top Predictive Features (by |Pearson r| with `interview_score`)

| Feature | |r| vs interview_score | Avg |r| (all 6 targets) |
|---|---|---|
| `Duration_Sec` | 0.3486 | 0.3443 |
| `Silence_Duration_Sec` | 0.2681 | 0.2684 |
| `Mean_Energy` | 0.1900 | 0.1954 |
| `MFCC_1` | 0.1846 | 0.1822 |
| `brow_raise_std` | 0.1679 | 0.1704 |
| `emotion_sad_mean` | 0.1557 | 0.1498 |
| `emotion_neutral_mean` | 0.1512 | 0.1482 |
| `eye_openness_std` | 0.1491 | 0.1575 |
| `emotion_fear_mean` | 0.1441 | 0.1352 |
| `brow_raise_mean` | 0.1377 | 0.1316 |

**Bottom 10 weakest predictors:**

| Feature | |r| vs interview_score | Avg |r| (all 6 targets) |
|---|---|---|
| `mouth_frown_std` | 0.0108 | 0.0196 |
| `MFCC_2` | 0.0079 | 0.0169 |
| `posture_shift_count` | 0.0043 | 0.0136 |
| `agitation_score` | 0.0036 | 0.0128 |
| `eye_openness_mean` | 0.0034 | 0.0116 |
| `shoulder_slope_var` | 0.0027 | 0.0043 |
| `user_no` | 0.0018 | 0.0105 |
| `Mean_ZCR` | 0.0015 | 0.0180 |
| `Mean_Pitch_Hz` | 0.0009 | 0.0277 |
| `question_id` | 0.0000 | 0.0000 |

---
## 5. Collinear Feature Pairs (|r| > 0.85) — DROP Candidates

| Feature A | Feature B | |r| |
|---|---|---|
| `shoulder_width_mean` | `engagement_score` | 0.9998 |
| `posture_shift_count` | `agitation_score` | 0.9998 |
| `smile_score_mean` | `emotion_happy_mean` | 0.9964 |
| `emotion_sad_mean` | `emotion_fear_mean` | 0.9871 |
| `frown_score_mean` | `emotion_angry_mean` | 0.9844 |
| `brow_raise_mean` | `emotion_surprise_mean` | 0.9767 |
| `gaze_ratio_mean` | `gaze_deviation_mean` | 0.9132 |
| `mouth_frown_mean` | `mouth_frown_std` | 0.9025 |
| `jaw_open_mean` | `jaw_open_std` | 0.8782 |
| `crossed_arms_score` | `engagement_score` | 0.8616 |
| `shoulder_width_mean` | `crossed_arms_score` | 0.8601 |
| `brow_raise_mean` | `emotion_fear_mean` | 0.8538 |

---
## 6. Near-Zero Variance Features — DROP

| Feature | Variance | Action |
|---|---|---|
| `mouth_frown_mean` | ~5.3e-05 | **DROP** |
| `shoulder_width_var` | ~5.4e-07 | **DROP** |
| `nose_shoulder_dist_var` | ~3.5e-07 | **DROP** |
| `core_speed_mean` | ~1.6e-05 | **DROP** |

---
## 7. Key Findings

### Duration Effect
- Shorter answers score higher (Duration_Sec r=-0.349)
- Silence_Duration_Sec r=-0.268
- `duration_label` redundant — Duration_Sec captures this

### Psychometrics Dominate
- Big Five traits have |r| = 0.52-0.68 with interview_score — strongest predictors
- overall_personality is a composite — consider excluding to avoid leakage from sub-traits

### Audio Signals
- `Mean_Energy` r~=-0.19 — very loud/intense speech correlates with lower scores
- `Speech_Rate_WPM` near zero correlation — pace alone is not discriminative

### Facial Signals
- `brow_raise_std` top facial predictor (r~=-0.17)
- Many facial features heavily collinear with emotions — prune pairs

### Video Quality
- High: -0.0064 | Low: 0.0318
- Negligible difference — safely exclude

### Transcript Word Count
- r = -0.35 to -0.38 with ALL 6 targets
- Longer rambling answers -> lower performance across all dimensions

### Candidate Leakage Risk
- 331 unique candidates answering multiple questions
- **MUST use GroupKFold(groups=user_no)** to prevent leakage

---
## 8. ML Recommendations

| Issue | Recommendation |
|---|---|
| Cross-validation | `GroupKFold(n_splits=5, groups=user_no)` |
| Zero-var features | Drop `mouth_frown_mean`, `shoulder_width_var`, `nose_shoulder_dist_var`, `core_speed_mean` |
| Redundant features | Drop one of each collinear pair (|r|>0.90) — see Section 5 |
| Skewed features | Log-transform `Silence_Duration_Sec`, `hand_speed_mean`, `hand_to_face_touches` |
| Metadata exclusion | Exclude: `id`, `file_name`, `question`, `question_id`, `user_no`, `video_quality`, `duration_label` |
| Text transcript | Build NLP embedding (BERT/TF-IDF) separately — not usable raw |
| Target leakage | Never use score columns as features when predicting another score |
| Multi-target strategy | Consider multi-output regression or separate models per target |

---
## 9. Plots Generated (18 total)

| # | File | Description |
|---|---|---|
| 1 | `01_target_distributions.png` | Histogram + normal overlay for all 6 targets |
| 2 | `02_target_boxplots.png` | Boxplots of all 6 target scores |
| 3 | `03_inter_target_heatmap.png` | Score x score correlation heatmap |
| 4 | `04_top_feature_correlations.png` | Top 20 features vs interview_score |
| 5 | `05_avg_correlation_all_targets.png` | Avg abs correlation across all 6 targets |
| 6 | `06_personality_vs_score.png` | Big Five scatter vs interview_score |
| 7 | `07_audio_prosodic_vs_score.png` | Audio prosodic feature scatter |
| 8 | `08_mfcc_correlations.png` | MFCC Pearson r bar chart |
| 9 | `09_emotion_analysis.png` | Emotion mean probabilities + correlations |
| 10 | `10_duration_analysis.png` | Duration label boxplots + duration scatter |
| 11 | `11_video_quality_analysis.png` | Score distributions by video quality |
| 12 | `12_features_by_perf_tier.png` | Feature distributions Low/Med/High tier |
| 13 | `13_feature_collinearity_heatmap.png` | All-feature collinearity heatmap |
| 14 | `14_candidate_score_dist.png` | Per-candidate mean + std scores |
| 15 | `15_samples_per_candidate.png` | Sample count per candidate |
| 16 | `16_outlier_prevalence.png` | Outlier % per feature |
| 17 | `17_transcript_analysis.png` | Word count distribution + score scatter |
| 18 | `18_pairwise_target_correlations.png` | Pairwise correlations between all 6 targets |

---
*Generated by `src/eda_merged_features.py` v2.0*
