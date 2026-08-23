# Executive EDA Summary & Feature Engineering Strategy
**Dataset:** `data/features/merged_features.csv` (88 columns, 2,011 rows, 331 candidates)  

---

## 1. Overview & Data Quality
- **Complete Feature Coverage:** 88 total columns spanning 7 multimodal feature domains (Personality, Audio Prosodic/Fluency, MFCCs, Facial Gaze, Facial Expressions, Facial Emotions, Posture/Pose).
- **Clean Numerical Fields:** 0 missing values across all numerical features. Only 13 records (0.65%) are missing text transcripts.

---

## 2. Decisional Matrix & Action Items for Preprocessing

| Decision Area | Analytical Finding | Action Item for Pipeline |
|---|---|---|
| **Cross-Validation** | 331 candidates present with multiple interview answers each. | Use `GroupKFold(n_splits=5, groups=df['user_no'])` to prevent train-test data leakage. |
| **Zero Variance** | 4 features (`mouth_frown_mean`, `shoulder_width_var`, `nose_shoulder_dist_var`, `core_speed_mean`) have variance < 0.0001. | **DROP** all 4 zero-variance columns. |
| **Collinearity** | 17 feature pairs show |r| > 0.85 (e.g. `engagement_score` vs `shoulder_width_mean` r=0.9998). | **DROP** 10 redundant collinear features to simplify model space and prevent multicollinearity. |
| **Skewed Features** | Fluency and silence features (`Total_Silence_Sec`, `Pause_Count`, `Mean_Energy`, `Filler_Word_Count`) are heavily right-skewed. | Apply `np.log1p` or `np.sqrt` transformation before linear/neural model training. |
| **Target Selection** | 6 performance targets are all z-score standardized and correlate strongly with each other (r = 0.62 - 0.77). | Train primary model on `interview_score` or build multi-output regressor for all 6 scores. |

---

## 3. Recommended Clean Feature Set (48 Features)
Removing metadata, zero-variance features, and redundant collinear pairs reduces the feature space from 75 raw features down to 48 highly informative features:
- **Psychometrics (6):** `openness`, `conscientiousness`, `extraversion`, `agreeableness`, `neuroticism`, `overall_personality`
- **Audio Prosody & Fluency (12):** `Duration_Sec`, `Total_Words`, `Speech_Rate_WPM`, `Articulation_Rate_WPM`, `Total_Silence_Sec`, `Silence_Ratio`, `Pause_Count`, `Avg_Pause_Duration_Sec`, `Filler_Word_Count`, `Filler_Rate_Per_Min`, `Mean_Pitch_Hz`, `Pitch_Range_Hz`, `Mean_Energy`, `Spectral_Centroid`, `Spectral_Contrast`, `Mean_ZCR`
- **Audio MFCCs (13):** `MFCC_1` through `MFCC_13`
- **Facial Gaze & Eyes (5):** `face_detected_ratio`, `gaze_ratio_mean`, `gaze_stability_std`, `eye_openness_mean`, `eye_openness_std`
- **Facial Expressions (6):** `smile_score_mean`, `smile_score_std`, `frown_score_mean`, `frown_score_std`, `jaw_open_mean`, `brow_raise_mean`, `brow_raise_std`
- **Facial Emotions (3):** `emotion_neutral_mean`, `emotion_sad_mean`, `emotion_disgust_mean`
- **Posture & Pose (8):** `head_centering_score_mean`, `absolute_shoulder_slope_mean`, `shoulder_slope_var`, `shoulder_width_mean`, `nose_shoulder_dist_mean`, `hand_speed_mean`, `hand_to_face_touches`, `posture_shift_count`

---
*Report generated automatically by `src/eda_merged_features.py`.*
