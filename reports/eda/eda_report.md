================================================================================
           INTERVIEW PERFORMANCE ANALYZER — EDA REPORT
                          v1.0 | Generated: 2026-08-23
================================================================================

Table of Contents
-----------------
  1. Executive Summary
  2. Dataset Overview
  3. Target Variables (6 scores)
  4. Personality Features (Big Five / OCEAN)
  5. Audio Features (Prosodic + MFCC)
  6. Facial Features (Gaze, Smile, Frown, Emotions)
  7. Posture Features (Shoulder, Hand, Core)
  8. Transcript Analysis
  9. Inter-Target Correlations
 10. Feature-Target Correlations (Top Predictors)
 11. Data Quality Issues
 12. Skewness & Transformation Recommendations
 13. Feature Drop Recommendations (Collinearity / Zero-Variance)
 14. Outlier Analysis
 15. Cross-Correlation Heatmap Observations
 16. Video Quality Impact
 17. Key Findings Summary
 18. Before-Modelling Checklist
 19. What More Is Required (Next Steps)
 20. Appendix: Recommended Clean Feature Set (48 features)

================================================================================
1. EXECUTIVE SUMMARY
================================================================================

The dataset contains 2,011 interview segments from 331 unique candidates.
Each segment represents one question-answer pair with multimodal features
extracted from audio, video (facial + posture), and transcript data.

6 target variables measure interview performance on different dimensions.
All targets are z-normalised (mean=0, std ~1.1-1.3) and highly correlated
(r=0.62 to 0.77), suggesting a strong underlying general factor.

Strongest predictors are personality traits (OCEAN) — particularly
openness, extraversion, and agreeableness (r=0.60-0.68). Audio duration
and word count are negatively correlated with all targets (longer, rambling
answers score lower).

Feature engineering must account for user-level grouping (multiple questions
per candidate) to avoid data leakage. A recommended clean feature set of
48 features (down from 62) is proposed after removing zero-variance,
collinear, and metadata columns.

================================================================================
2. DATASET OVERVIEW
================================================================================

  Shape:              2,011 rows x 75 columns (after merge)
  Unique candidates:  331 (user_no)
  Questions per user: median=6, range=1-17
  Missing transcripts: 13 rows (0.65%)

  Column categories:
    - Metadata:        8 columns  (id, file_name, question_id, question,
                                    user_no, video_quality, duration_label, transcript)
    - Personality:     6 columns  (openness, conscientiousness, extraversion,
                                    agreeableness, neuroticism, overall_personality)
    - Audio prosodic:  7 columns  (Duration_Sec, Speech_Rate_WPM, Silence_Duration_Sec,
                                    Mean_Pitch_Hz, Mean_Energy, Mean_ZCR)
    - Audio MFCC:     13 columns  (MFCC_1 through MFCC_13)
    - Facial:         21 columns  (face_detected_ratio, gaze_ratio_mean,
                                    gaze_stability_std, eye_openness_mean/std,
                                    smile_score_mean/std, frown_score_mean/std,
                                    jaw_open_mean, brow_raise_mean/std,
                                    emotion_neutral_mean, emotion_happy_mean,
                                    emotion_sad_mean, emotion_angry_mean,
                                    emotion_surprise_mean, emotion_fear_mean,
                                    emotion_disgust_mean, head_centering_score_mean,
                                    absolute_shoulder_slope_mean, crossed_arms_score,
                                    engagement_score, agitation_score)
    - Posture:         8 columns  (shoulder_slope_var, shoulder_width_mean/var,
                                    nose_shoulder_dist_mean/var, hand_speed_mean,
                                    hand_to_face_touches, posture_shift_count, core_speed_mean)
    - Targets:         6 columns  (interview_score, answer_score, speaking_skills,
                                    confidence_score, facial_expression, overall_performance)

================================================================================
3. TARGET VARIABLES (6 SCORES)
================================================================================

All targets are z-normalised (mean ~0, std ~1.1-1.3).

  interview_score:     Overall interview rating
  answer_score:        Quality of answers given
  speaking_skills:     Verbal communication ability
  confidence_score:    Apparent confidence level
  facial_expression:   Facial expressiveness/engagement
  overall_performance: Composite performance metric

  INTER-TARGET CORRELATIONS (Pearson r):

                        interview  answer  speaking  confidence  facial  overall
  interview_score         1.00
  answer_score            0.77      1.00
  speaking_skills         0.69      0.72    1.00
  confidence_score        0.70      0.73    0.65      1.00
  facial_expression       0.64      0.66    0.63      0.62        1.00
  overall_performance     0.77      0.77    0.69      0.73        0.66    1.00

  Key insight: All targets correlate r=0.62-0.77, suggesting a strong
  underlying "general interview performance" factor. This enables:
    - Multi-task learning across all 6 targets
    - Single-target modelling (e.g., interview_score) as proxy
    - Potential composite target engineering

================================================================================
4. PERSONALITY FEATURES (Big Five / OCEAN)
================================================================================

Standardised Z-scores from psychometric assessment.

  TRAIT                  MEAN    STD     RANGE         SKEWNESS
  openness              -0.01    0.99    [-3.0, +3.0]   -0.18
  conscientiousness     -0.02    0.99    [-3.0, +3.0]   -0.06
  extraversion          -0.01    0.99    [-3.0, +3.0]   -0.12
  agreeableness         -0.01    0.99    [-3.0, +3.0]   -0.08
  neuroticism           -0.01    0.99    [-3.0, +3.0]   -0.04
  overall_personality    0.00    0.99    [-3.0, +3.0]   -0.03

  PERSONALITY-TO-TARGET CORRELATIONS (sorted by avg_abs_r):

  FEATURE                interview  answer  speaking  confidence  facial  overall  AVG
  overall_personality      0.647    0.699    0.608     0.723     0.629    0.765   0.679
  agreeableness           0.625    0.672    0.628     0.655     0.677    0.648   0.651
  openness                0.684    0.621    0.642     0.601     0.639    0.651   0.640
  extraversion            0.641    0.626    0.624     0.622     0.638    0.610   0.627
  conscientiousness       0.524    0.651    0.533     0.615     0.533    0.669   0.587
  neuroticism            -0.231   -0.317   -0.245    -0.353    -0.265   -0.329   0.290

  KEY INSIGHT: Personality traits are the STRONGEST predictors (r=0.59-0.77).
  This is because the targets were likely rated by humans who were influenced
  by candidates' demonstrated personality. overall_personality (r=0.77 with
  overall_performance) is the single best predictor in the dataset.

================================================================================
5. AUDIO FEATURES (Prosodic + MFCC)
================================================================================

  --- Prosodic Features ---

  FEATURE                RANGE           SKEWNESS   NOTES
  Duration_Sec           [2.6, 197.3]     2.07     Highly right-skewed
  Speech_Rate_WPM        [36.8, 308.6]    1.58     Right-skewed
  Silence_Duration_Sec   [0.0, 97.4]      2.70     Highly right-skewed
  Mean_Pitch_Hz          [74.7, 523.1]   11.33     Extremely right-skewed
  Mean_Energy            [0.0, 0.5]       3.82     Right-skewed
  Mean_ZCR               [0.0, 0.2]       1.47     Right-skewed

  --- MFCC Features (13 coefficients) ---

  MFCC coefficients represent the spectral envelope of speech.
  MFCC_1:  skewness = -4.95 (heavily left-skewed)
  MFCC_2-13: moderate skewness

  AUDIO-TO-TARGET CORRELATIONS (top performers):

  FEATURE                interview  answer  speaking  confidence  facial  overall  AVG
  Duration_Sec           -0.349   -0.371   -0.329    -0.321    -0.328   -0.367   0.344
  Silence_Duration_Sec   -0.268   -0.290   -0.260    -0.251    -0.250   -0.292   0.268
  Mean_Energy            -0.190   -0.191   -0.185    -0.224    -0.187   -0.195   0.195
  MFCC_1                 -0.185   -0.188   -0.180    -0.182    -0.177   -0.180   0.182
  MFCC_7                  0.133    0.153    0.140     0.177     0.142    0.142   0.147
  MFCC_10                 0.118    0.131    0.153     0.170     0.148    0.129   0.142
  Speech_Rate_WPM        -0.114   -0.090   -0.117    -0.148    -0.142   -0.135   0.124

  KEY INSIGHT: Duration and Silence_Duration are negatively correlated —
  shorter, more concise answers score higher. Mean_Pitch_Hz has near-zero
  correlation (r=0.00 to -0.04) and can be dropped.

================================================================================
6. FACIAL FEATURES (Gaze, Smile, Frown, Emotions)
================================================================================

  --- Gaze Features ---
  gaze_ratio_mean:       [0.0, 1.0],     avg ~0.65   (face-to-camera ratio)
  gaze_stability_std:    [0.0, 0.5],     right-skewed (gaze jitter)
  eye_openness_mean:     [0.0, 1.0],     avg ~0.82
  eye_openness_std:      [0.0, 0.3],     avg ~0.06

  --- Smile/Frown Features (Action Units) ---
  smile_score_mean:      [0.0, 0.9],     skewness=2.86
  smile_score_std:       [0.0, 0.4],     skewness=2.15
  frown_score_mean:      [0.0, 0.3],     skewness=3.50
  frown_score_std:       [0.0, 0.2],     skewness=2.61
  jaw_open_mean:         [0.0, 0.5],     avg ~0.05
  brow_raise_mean:       [0.0, 0.8],     avg ~0.04
  brow_raise_std:        [0.0, 0.3],     avg ~0.03

  --- Emotion Probabilities ---
  emotion_neutral_mean:  avg ~0.55       (dominant emotion)
  emotion_happy_mean:    avg ~0.20
  emotion_sad_mean:      avg ~0.08
  emotion_angry_mean:    avg ~0.04
  emotion_surprise_mean: avg ~0.06
  emotion_fear_mean:     avg ~0.03
  emotion_disgust_mean:  avg ~0.04

  --- Other Facial ---
  face_detected_ratio:   [0.0, 1.0],     skewness=-21.14 (many 1.0 values)
  head_centering_score_mean: avg ~0.85
  absolute_shoulder_slope_mean: avg ~0.04

  FACIAL-TO-TARGET CORRELATIONS (top performers):

  FEATURE                interview  answer  speaking  confidence  facial  overall  AVG
  brow_raise_std        -0.168   -0.171   -0.160    -0.186    -0.165   -0.172   0.170
  eye_openness_std      -0.149   -0.162   -0.152    -0.171    -0.150   -0.161   0.158
  emotion_sad_mean      -0.156   -0.132   -0.143    -0.167    -0.160   -0.141   0.150
  emotion_neutral_mean   0.151    0.138    0.139     0.173     0.152    0.136   0.148
  brow_raise_mean       -0.138   -0.117   -0.129    -0.144    -0.136   -0.126   0.132
  smile_score_std       -0.130   -0.134   -0.115    -0.136    -0.141   -0.122   0.130

  KEY INSIGHT: brow_raise_std is the top facial predictor (r=0.17).
  Facial features have weaker correlations than personality but still
  contribute meaningful signal. Stable gaze and neutral emotion correlate
  positively with scores.

================================================================================
7. POSTURE FEATURES (Shoulder, Hand, Core)
================================================================================

  FEATURE                RANGE           SKEWNESS   NOTES
  shoulder_slope_var     [0.0, 0.05]    42.24      Extremely skewed, near-zero variance
  shoulder_width_mean    [0.0, 1.0]      0.20      Near-perfect r=0.9998 with engagement_score
  shoulder_width_var     [0.0, 0.001]    9.23      Near-zero variance
  nose_shoulder_dist_mean [0.0, 1.0]     0.80      Moderate
  nose_shoulder_dist_var [0.0, 0.001]    3.73      Near-zero variance
  hand_speed_mean        [0.0, 1.0]      2.13      Right-skewed
  hand_to_face_touches   [0.0, 50]      13.87      Highly right-skewed
  posture_shift_count    [0, 15]         4.71      Right-skewed
  core_speed_mean        [0.0, 0.01]    2.38       Near-zero variance

  POSTURE-TO-TARGET CORRELATIONS:
  Most posture features have very weak correlations (r < 0.05).
  The strongest is nose_shoulder_dist_mean (r=0.09 avg_abs).

  KEY INSIGHT: Posture features are the weakest predictors. Several have
  near-zero variance and should be dropped. engagement_score and
  agitation_score are perfectly collinear with other features.

================================================================================
8. TRANSCRIPT ANALYSIS
================================================================================

  Missing transcripts: 13 rows (0.65%)
  Valid transcript rows: 1,998

  Word count stats:
    mean:    ~85 words
    median:  ~70 words
    min:     ~5 words
    max:     ~500+ words

  WORD COUNT → TARGET CORRELATIONS:

  TARGET               Pearson_r  Spearman_rho
  interview_score       -0.366     -0.438
  answer_score          -0.382     -0.451
  speaking_skills       -0.348     -0.421
  confidence_score      -0.349     -0.397
  facial_expression     -0.351     -0.401
  overall_performance   -0.380     -0.464

  KEY INSIGHT: Word count NEGATIVELY correlates with ALL targets
  (r=-0.35 to -0.38). Longer, rambling answers score LOWER across
  every assessment dimension. This is one of the most actionable
  insights — conciseness matters.

  Spearman correlations are stronger than Pearson, suggesting a
  monotonic (not necessarily linear) relationship.

  RECOMMENDATION: Build NLP pipeline for transcripts separately:
    - TF-IDF for baseline
    - BERT embeddings for richer representation
    - Consider sentiment analysis, keyword extraction
    - Word count itself is a useful engineered feature

================================================================================
9. INTER-TARGET CORRELATIONS
================================================================================

  All 6 targets are highly correlated:

  STRONGEST PAIRS:
    interview_score  ↔ overall_performance:  r = 0.77
    answer_score     ↔ overall_performance:  r = 0.77
    answer_score     ↔ interview_score:      r = 0.77
    confidence_score ↔ overall_performance:  r = 0.73
    confidence_score ↔ answer_score:         r = 0.73

  WEAKEST PAIRS:
    facial_expression ↔ interview_score:     r = 0.64
    facial_expression ↔ speaking_skills:     r = 0.63
    facial_expression ↔ confidence_score:    r = 0.62

  KEY INSIGHT: The high inter-target correlation (r=0.62-0.77) suggests
  these scores measure a single latent construct (interview performance).
  This supports:
    1. Multi-task learning across all 6 targets
    2. Using interview_score or overall_performance as primary target
    3. Target engineering: composite score = mean of all 6

================================================================================
10. FEATURE-TARGET CORRELATIONS (TOP PREDICTORS)
================================================================================

  RANK  FEATURE                  AVG_ABS_R  DOMAIN
  ----  -----------------------  ---------  --------
   1.   overall_personality        0.679     Personality
   2.   agreeableness              0.651     Personality
   3.   openness                   0.640     Personality
   4.   extraversion               0.627     Personality
   5.   conscientiousness          0.587     Personality
   6.   Duration_Sec               0.344     Audio
   7.   neuroticism                0.290     Personality
   8.   Silence_Duration_Sec       0.268     Audio
   9.   Mean_Energy                0.195     Audio
  10.   MFCC_1                     0.182     Audio
  11.   brow_raise_std             0.170     Facial
  12.   eye_openness_std           0.158     Facial
  13.   emotion_sad_mean           0.150     Facial
  14.   emotion_neutral_mean       0.148     Facial
  15.   MFCC_7                     0.147     Audio

  FEATURES WITH NEAR-ZERO CORRELATION (can drop):
    - Mean_Pitch_Hz          avg_abs_r = 0.028
    - shoulder_width_mean     avg_abs_r = 0.022
    - Mean_ZCR                avg_abs_r = 0.018
    - MFCC_2                  avg_abs_r = 0.017
    - head_centering_score_mean avg_abs_r = 0.015
    - MFCC_6                  avg_abs_r = 0.015
    - posture_shift_count     avg_abs_r = 0.014
    - eye_openness_mean       avg_abs_r = 0.012
    - hand_speed_mean         avg_abs_r = 0.011
    - shoulder_slope_var      avg_abs_r = 0.004

================================================================================
11. DATA QUALITY ISSUES
================================================================================

  ISSUE                          COUNT    PERCENTAGE   SEVERITY
  ----                           -----    ----------   --------
  Missing transcripts            13       0.65%        LOW
  Near-zero variance features    4        —            MEDIUM
  Perfect collinear pairs        2        —            HIGH
  High collinear pairs (r>0.9)   10       —            HIGH
  Extreme skewness (|s|>2)       21       —            MEDIUM
  Metadata columns               8        —            LOW (exclude)

  MISSING DATA:
  - 13 rows have missing transcripts — can be handled by imputation
    or exclusion during NLP pipeline processing.
  - No other missing values detected in numeric features.

  ZERO-VARIANCE FEATURES (var < 1e-4):
  - mouth_frown_mean:     var = 5.31e-05
  - shoulder_width_var:   var = 5.37e-07
  - nose_shoulder_dist_var: var = 3.54e-07
  - core_speed_mean:      var = 1.63e-05

================================================================================
12. SKEWNESS & TRANSFORMATION RECOMMENDATIONS
================================================================================

  Categories based on |skewness|:

  SEVERITY   RECOMMENDATION              FEATURES (count)
  --------   -------------------------   ----------------
  |s| > 2    Log/sqrt transform          21 features
  |s| 1-2    Consider Box-Cox            ~8 features
  |s| 0.5-1  Mildly skewed (ok for trees) ~5 features
  |s| < 0.5  Approx normal               ~15 features

  TOP 10 MOST SKEWED FEATURES:
    1. shoulder_slope_var         skewness =  42.24
    2. gaze_stability_std         skewness =  21.80
    3. face_detected_ratio        skewness = -21.14
    4. hand_to_face_touches       skewness =  13.87
    5. Mean_Pitch_Hz              skewness =  11.33
    6. shoulder_width_var         skewness =   9.23
    7. mouth_frown_mean           skewness =   5.35
    8. MFCC_1                     skewness =  -4.95
    9. mouth_frown_std            skewness =   4.78
   10. agitation_score            skewness =   4.73

  RECOMMENDATION:
  - Tree-based models (XGBoost, LightGBM, RF): NO transformation needed
  - Linear models (Ridge, Lasso, SVM): Apply log/sqrt to top 21 features
  - Scale-invariant models can skip standardisation entirely

================================================================================
13. FEATURE DROP RECOMMENDATIONS
================================================================================

  CATEGORY A — DROP: Near-Zero Variance (var < 1e-4)
  ---------------------------------------------------
    mouth_frown_mean        var = 5.31e-05
    shoulder_width_var      var = 5.37e-07
    nose_shoulder_dist_var  var = 3.54e-07
    core_speed_mean         var = 1.63e-05

  CATEGORY B — DROP: High Collinearity (|r| > 0.90)
  ---------------------------------------------------
    DROP emotion_happy_mean    (keep smile_score_mean   r=0.996)
    DROP emotion_angry_mean    (keep frown_score_mean   r=0.984)
    DROP emotion_surprise_mean (keep brow_raise_mean    r=0.977)
    DROP emotion_fear_mean     (keep emotion_sad_mean   r=0.987)
    DROP gaze_deviation_mean   (keep gaze_ratio_mean    r=0.913)
    DROP engagement_score      (keep shoulder_width_mean r=0.9998 — PERFECT)
    DROP agitation_score       (keep posture_shift_count r=0.9998 — PERFECT)
    DROP mouth_frown_std       (keep mouth_frown_mean   r=0.903 — already Cat A)
    DROP jaw_open_std          (keep jaw_open_mean      r=0.878)
    DROP crossed_arms_score    (keep shoulder_width_mean r=0.862)

  CATEGORY C — EXCLUDE: Metadata (never model features)
  ------------------------------------------------------
    id, file_name, question_id, question, user_no,
    video_quality, duration_label, transcript

  TOTAL FEATURES REMOVED: 14 (4 zero-var + 10 collinear)
  RECOMMENDED CLEAN SET:   48 features (down from 62)

================================================================================
14. OUTLIER ANALYSIS
================================================================================

  IQR method applied to all 62 numeric features.

  Features with highest outlier percentage:
    - hand_to_face_touches:     ~15% outliers
    - shoulder_slope_var:       ~12% outliers
    - gaze_stability_std:       ~10% outliers
    - Mean_Pitch_Hz:            ~8% outliers
    - Mean_Energy:              ~7% outliers

  APPROACH:
  - For tree-based models: outliers are handled automatically
  - For linear models: consider robust scaling or winsorisation
  - Do NOT remove outliers blindly — they may represent real
    high/low performers

================================================================================
15. CROSS-CORRELATION HEATMAP OBSERVATIONS
================================================================================

  FEATURE GROUP CORRELATIONS:

  Personality traits:     moderate inter-correlation (r=0.3-0.6)
  MFCC coefficients:      low inter-correlation (r=0.0-0.3)
  Facial AUs:             high inter-correlation (smile-happy r=0.996)
  Posture features:       moderate inter-correlation (r=0.3-0.6)
  Audio prosodic:         low-moderate inter-correlation

  CROSS-DOMAIN CORRELATIONS:
  - Personality ↔ Audio:   weak (r < 0.10)
  - Personality ↔ Facial:  weak (r < 0.10)
  - Personality ↔ Posture: weak (r < 0.05)
  - Audio ↔ Facial:        weak (r < 0.08)
  - Audio ↔ Posture:       weak (r < 0.05)
  - Facial ↔ Posture:      weak (r < 0.05)

  KEY INSIGHT: Cross-domain correlations are very weak, meaning each
  modality provides independent signal. This supports multi-modal
  fusion approaches.

================================================================================
16. VIDEO QUALITY IMPACT
================================================================================

  Video quality levels: "High" and "Low"

  Comparison of targets by video quality:
  - No significant difference between High and Low quality videos
  - Target score distributions are similar across quality levels

  CONCLUSION: Video quality is safe to exclude from models. It does
  not introduce systematic bias in scoring.

================================================================================
17. KEY FINDINGS SUMMARY
================================================================================

  1. PERSONALITY DOMINATES: OCEAN traits are the strongest predictors
     (r=0.59-0.77). overall_personality alone explains ~59% of variance
     in overall_performance (R² ≈ 0.59).

  2. CONCISENESS MATTERS: Word count and duration negatively correlate
     with ALL targets. Shorter, focused answers score higher.

  3. MULTIMODAL INDEPENDENCE: Cross-domain feature correlations are
     weak (r<0.10), meaning audio, facial, posture, and transcript
     provide complementary signal.

  4. TARGET REDUNDANCY: All 6 targets correlate r=0.62-0.77, suggesting
     a single latent "interview performance" factor.

  5. DATA LEAKAGE RISK: Multiple rows per user require GroupKFold
     splitting by user_no — never split by row.

  6. CLEAN SET REDUCTION: 14 features can be safely dropped (zero-variance
     + collinear), reducing from 62 to 48 features.

  7. SKEWNESS: 21 features have |skewness| > 2 and need transformation
     for linear models. Tree models are robust to this.

  8. NEAR-ZERO SIGNAL: ~10 features have avg_abs_r < 0.02 with all
     targets and can be considered for removal.

================================================================================
18. BEFORE-MODELLING CHECKLIST
================================================================================

  [ ] Drop zero-variance features: mouth_frown_mean, shoulder_width_var,
      nose_shoulder_dist_var, core_speed_mean

  [ ] Drop collinear duplicates: engagement_score, agitation_score,
      emotion_happy_mean, emotion_angry_mean, emotion_surprise_mean,
      emotion_fear_mean, gaze_deviation_mean, jaw_open_std,
      mouth_frown_std, crossed_arms_score

  [ ] Exclude metadata: id, file_name, question, question_id, user_no,
      video_quality, duration_label, transcript

  [ ] Log-transform for linear models: Silence_Duration_Sec,
      hand_speed_mean, hand_to_face_touches, Mean_Pitch_Hz

  [ ] Use GroupKFold(groups=user_no) for ALL cross-validation — no exceptions

  [ ] Choose target(s): Single target (interview_score) or multi-output

  [ ] Build NLP pipeline for transcripts separately (BERT / TF-IDF)

  [ ] Handle 13 missing transcripts (impute or exclude)

  [ ] Consider user-level aggregation (mean, std, min, max across questions)

================================================================================
19. WHAT MORE IS REQUIRED (NEXT STEPS)
================================================================================

  A. FEATURE ENGINEERING (Priority: HIGH)
  ----------------------------------------
  [ ] Aggregate features per user (mean, std, min, max across questions)
      - Current dataset has 1 row per question; models need 1 row per user
      - Create user-level features: mean_openness, std_speech_rate, etc.
      - This will reduce rows from 2,011 to 331 (one per candidate)

  [ ] Create composite target variable (mean of all 6 targets)
      - Useful as primary target for single-target modelling

  [ ] Engineer interaction features:
      - personality × audio interactions
      - facial × posture interactions
      - word_count × speech_rate (efficiency metric)

  [ ] Binary features:
      - is_high_performer (top 25% on overall_performance)
      - is_confident (confidence_score > median)

  B. NLP PIPELINE (Priority: HIGH)
  ---------------------------------
  [ ] TF-IDF vectorisation of transcripts
  [ ] BERT/MiniLM embeddings for richer semantic features
  [ ] Sentiment analysis (positive/negative/neutral)
  [ ] Keyword extraction (filler words, action verbs, technical terms)
  [ ] Readability scores (Flesch-Kincaid, etc.)

  C. MODEL DEVELOPMENT (Priority: HIGH)
  --------------------------------------
  [ ] Baseline models: Ridge, Lasso, ElasticNet
  [ ] Tree models: XGBoost, LightGBM, Random Forest
  [ ] Multi-modal fusion: concatenate audio+facial+posture+personality
  [ ] Deep learning: neural network with modality-specific encoders
  [ ] Multi-task learning: shared backbone, 6 prediction heads
  [ ] Ensemble methods: stacking, blending

  D. EVALUATION FRAMEWORK (Priority: HIGH)
  -----------------------------------------
  [ ] GroupKFold CV (groups=user_no, k=5)
  [ ] Metrics: RMSE, MAE, R², Spearman correlation
  [ ] Per-target evaluation
  [ ] Feature importance analysis (SHAP values)
  [ ] Learning curves to assess data adequacy

  E. PREPROCESSING PIPELINE (Priority: MEDIUM)
  ---------------------------------------------
  [ ] StandardScaler/RobustScaler for linear models
  [ ] Log transforms for skewed features
  [ ] Imputation strategy for missing transcripts
  [ ] Pipeline serialization (sklearn Pipeline / joblib)

  F. DOCUMENTATION & DEPLOYMENT (Priority: LOW)
  ----------------------------------------------
  [ ] Model card documenting performance, limitations, biases
  [ ] Inference pipeline for new interview videos
  [ ] API endpoint for real-time scoring
  [ ] Dashboard for visualising predictions

================================================================================
20. APPENDIX: RECOMMENDED CLEAN FEATURE SET (48 FEATURES)
================================================================================

  After applying all drop recommendations, the clean feature set is:

  PERSONALITY (6):
    1.  openness
    2.  conscientiousness
    3.  extraversion
    4.  agreeableness
    5.  neuroticism
    6.  overall_personality

  AUDIO PROSODIC (6):
    7.  Duration_Sec
    8.  Speech_Rate_WPM
    9.  Silence_Duration_Sec
    10. Mean_Pitch_Hz
    11. Mean_Energy
    12. Mean_ZCR

  AUDIO MFCC (13):
    13. MFCC_1
    14. MFCC_2
    15. MFCC_3
    16. MFCC_4
    17. MFCC_5
    18. MFCC_6
    19. MFCC_7
    20. MFCC_8
    21. MFCC_9
    22. MFCC_10
    23. MFCC_11
    24. MFCC_12
    25. MFCC_13

  FACIAL (14):
    26. face_detected_ratio
    27. gaze_ratio_mean
    28. gaze_stability_std
    29. eye_openness_mean
    30. eye_openness_std
    31. smile_score_mean
    32. smile_score_std
    33. frown_score_mean
    34. frown_score_std
    35. jaw_open_mean
    36. brow_raise_mean
    37. brow_raise_std
    38. emotion_neutral_mean
    39. emotion_sad_mean
    40. emotion_disgust_mean

  POSTURE (9):
    41. head_centering_score_mean
    42. absolute_shoulder_slope_mean
    43. shoulder_slope_var
    44. shoulder_width_mean
    45. nose_shoulder_dist_mean
    46. hand_speed_mean
    47. hand_to_face_touches
    48. posture_shift_count

================================================================================
                          END OF EDA REPORT
================================================================================

  REPORT VERSION: 1.0
  GENERATED FROM: interview_analyzer_eda.ipynb (v2.0)
  DATA SOURCE:    data/features/merged_all.csv (2,011 rows x 75 cols)
  BRANCH:         shibli-eda
  STATUS:         LOCAL ONLY — awaiting user review before commit/push

================================================================================
