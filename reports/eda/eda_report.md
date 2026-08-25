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

## 10. Problem-Solution Matrix

This section maps each identified problem to a concrete, implementable solution.

---

### 10.1 MFCC Features (MFCC_1 to MFCC_13) — High Correlation, Redundancy

**Problem:** 13 MFCC coefficients capture overlapping spectral information. MFCC_1 alone has r=-0.185 with interview_score, but the remaining 12 add diminishing returns and introduce multicollinearity.

**Solution: PCA Dimensionality Reduction**

```python
from sklearn.decomposition import PCA

mfcc_cols = [f"MFCC_{i}" for i in range(1, 14)]
pca = PCA(n_components=0.95)  # Retain 95% variance
mfcc_pca = pca.fit_transform(df[mfcc_cols])
# Creates MFCC_PC1, MFCC_PC2, ... (typically 5-7 components)
```

**Why PCA for MFCC:**
- MFCCs are spectral coefficients designed to be decorrelated
- PCA further compresses redundant information
- Reduces 13 features to ~5-7 components while retaining 95% variance
- Eliminates multicollinearity that hurts linear models

**Alternative:** Keep only MFCC_1, MFCC_7, MFCC_10 (highest individual correlations) and drop the rest.

---

### 10.2 Collinear Feature Pairs — Redundancy (17 pairs with |r| > 0.85)

**Problem:** 10 features are nearly identical to other features (r > 0.99), adding noise without information.

**Solution: Drop One Feature from Each Collinear Pair**

| Keep | Drop | Reason |
|---|---|---|
| shoulder_width_mean | engagement_score | r=0.9998, engagement is composite |
| posture_shift_count | agitation_score | r=0.9998, agitation is composite |
| smile_score_mean | emotion_happy_mean | r=0.9964, AU is more granular |
| emotion_sad_mean | emotion_fear_mean | r=0.8756, sad is stronger signal |
| emotion_disgust_mean | emotion_angry_mean | r=0.8634, disgust is stronger |
| emotion_neutral_mean | emotion_surprise_mean | r=0.8512, neutral is dominant |
| Spectral_Centroid | Spectral_Rolloff | r=0.8934, centroid is more stable |
| gaze_ratio_mean | gaze_deviation_mean | r=0.9867, ratio is more interpretable |
| mouth_frown_mean | mouth_frown_std | r=0.9589, mean is more stable |
| jaw_open_mean | jaw_open_std | r=0.9123, mean is more stable |

```python
drop_collinear = [
    'engagement_score', 'agitation_score', 'emotion_happy_mean',
    'emotion_fear_mean', 'emotion_angry_mean', 'emotion_surprise_mean',
    'Spectral_Rolloff', 'gaze_deviation_mean', 'mouth_frown_std', 'jaw_open_std'
]
df_clean = df.drop(columns=drop_collinear)
```

---

### 10.3 Low-Variance Features — Near-Zero Discriminative Power

**Problem:** 4 features have variance < 1e-4, meaning they are essentially constant across all 2,011 samples.

**Solution: Drop These Features**

```python
drop_low_var = ['mouth_frown_mean', 'shoulder_width_var',
                'nose_shoulder_dist_var', 'core_speed_mean']
df_clean = df.drop(columns=drop_low_var)
```

**Why drop instead of transform:**
- Near-zero variance means the feature cannot distinguish between candidates
- No transformation can create signal from noise
- These features would only add computational cost and overfitting risk

---

### 10.4 Skewed Features — 29 Features with |skewness| > 2.0

**Problem:** Right-skewed features violate linear model assumptions and hurt tree model split decisions.

**Solution: Log1p Transformation**

```python
import numpy as np

skewed_cols = [
    'Total_Silence_Sec', 'Silence_Ratio', 'Pause_Count',
    'Avg_Pause_Duration_Sec', 'Mean_Energy', 'Filler_Word_Count',
    'hand_speed_mean', 'posture_shift_count', 'Pitch_Range_Hz',
    'Std_Pitch_Hz', 'Speech_Rate_WPM', 'Mean_Pitch_Hz'
]

for col in skewed_cols:
    df[col] = np.log1p(df[col])  # log(1 + x) handles zeros safely
```

**Why log1p:**
- Compresses right tail without distorting relative ordering
- Handles zero values safely (log1p(0) = 0)
- Makes distributions more symmetric for linear models
- Helps tree models find better split points

**Note:** Do NOT apply to left-skewed features (use sqrt or Box-Cox for those).

---

### 10.5 Outliers — 24 Features with 5-16% Outlier Rates

**Problem:** Extreme values can skew model training, especially for linear models and neural networks.

**Solution: RobustScaler (NOT StandardScaler)**

```python
from sklearn.preprocessing import RobustScaler

# Apply AFTER train/test split to prevent data leakage
scaler = RobustScaler()  # Uses IQR, resistant to outliers
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Why RobustScaler over StandardScaler:**
- StandardScaler uses mean±std, which are themselves affected by outliers
- RobustScaler uses median±IQR, which are resistant to outliers
- Preserves the relative distance between normal data points
- Essential for Ridge, Lasso, ElasticNet, and neural networks

**For tree models (XGBoost, LightGBM, RF):** No scaling needed — trees are invariant to monotonic transformations.

---

### 10.6 Categorical Features — Duration Label Ordinal Encoding

**Problem:** `duration_label` has natural order (short < medium < long) but is stored as string.

**Solution: Ordinal Encoding**

```python
from sklearn.preprocessing import OrdinalEncoder

duration_map = {'short': 0, 'medium': 1, 'long': 2}
df['duration_label_encoded'] = df['duration_label'].map(duration_map)
```

**Why ordinal (not one-hot):**
- The order matters: short < medium < long
- One-hot would lose the ordinal relationship
- Only 3 categories, so ordinal is simpler and more interpretable

---

### 10.7 Missing Values — Transcript (13 rows, 0.65%)

**Problem:** 13 transcripts are missing, which could bias NLP analysis.

**Solution: Empty String Imputation (Defer to NLP Pipeline)**

```python
df['transcript'] = df['transcript'].fillna('')
```

**Why empty string:**
- These rows still have valid audio, facial, and posture features
- Dropping them would lose valuable multimodal data
- The NLP pipeline will handle empty transcripts naturally (zero TF-IDF, zero BERT embeddings)
- Can optionally flag these rows with a `transcript_missing` binary feature

---

### 10.8 Text Transcripts — Raw Text Not Model-Ready

**Problem:** Transcripts are raw text strings, not numerical features.

**Solution: TF-IDF or BERT Embeddings**

**Option A: TF-IDF (Fast, Interpretable)**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=500, stop_words='english')
X_text = tfidf.fit_transform(df['transcript']).toarray()
# Creates 500-dimensional sparse matrix
```

**Option B: BERT Embeddings (Semantic, Slower)**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
X_text = model.encode(df['transcript'].tolist())
# Creates 384-dimensional dense matrix
```

**Recommendation:** Start with TF-IDF for快速 iteration, switch to BERT if performance is insufficient.

---

### 10.9 Data Leakage — Candidate Overlap Between Train/Test

**Problem:** Each candidate has ~6 responses. Random splitting would put some responses in train and others in test for the same candidate, leaking information.

**Solution: GroupKFold Cross-Validation**

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups=df['user_no']):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    # Train and evaluate here
```

**Why GroupKFold:**
- Ensures no candidate appears in both train and test
- Provides realistic generalization estimates
- Mandatory for this dataset due to candidate-level clustering

---

### 10.10 Feature Scaling — Linear vs Tree Models

**Problem:** Different model families require different preprocessing.

**Solution: Conditional Scaling Pipeline**

```python
# For tree models (NO scaling needed)
from xgboost import XGBRegressor
model_tree = XGBRegressor()

# For linear models (scaling required AFTER split)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

pipeline_linear = Pipeline([
    ('scaler', RobustScaler()),
    ('model', Ridge(alpha=1.0))
])

# For neural networks (scaling required AFTER split)
from sklearn.neural_network import MLPRegressor
pipeline_nn = Pipeline([
    ('scaler', StandardScaler()),
    ('model', MLPRegressor(hidden_layer_sizes=(128, 64)))
])
```

---

### 10.11 Candidate-Level Aggregation — Multiple Responses per User

**Problem:** Each candidate has ~6 responses. Models may overfit to individual response noise instead of capturing candidate-level patterns.

**Solution: Aggregate Features per Candidate**

```python
# Aggregate numeric features per user_no
agg_funcs = ['mean', 'std', 'min', 'max']
feature_cols = [c for c in df.columns if c not in METADATA_COLS + TARGET_COLS + TEXT_COLS]

df_agg = df.groupby('user_no')[feature_cols].agg(agg_funcs)
df_agg.columns = ['_'.join(col) for col in df_agg.columns]
# Creates: Duration_Sec_mean, Duration_Sec_std, Duration_Sec_min, Duration_Sec_max, ...
```

**Why aggregate:**
- Reduces noise from individual responses
- Creates candidate-level feature vectors
- Enables candidate-level train/test splitting
- Can be combined with response-level features for multi-level modelling

---

### 10.12 Personality Features — Strongest Predictors, But May Overfit

**Problem:** Big Five personality traits (openness, agreeableness, etc.) are the strongest predictors (r=0.58-0.68), but they are external metadata, not derived from the interview itself.

**Solution: Two-Model Strategy**

```python
# Model 1: Interview-only features (audio + facial + posture)
features_interview = audio_cols + mfcc_cols + face_cols + emotion_cols + posture_cols

# Model 2: All features (interview + personality)
features_all = features_interview + personality_cols

# Compare performance to assess personality contribution
```

**Why two models:**
- Model 1 tests what can be predicted FROM the interview itself
- Model 2 tests what can be predicted WITH personality metadata
- The gap between them quantifies how much personality information helps
- In production, you may not have personality data upfront

---

## 11. Summary: Recommended Preprocessing Pipeline

```python
def preprocess_pipeline(df):
    """Complete preprocessing pipeline based on EDA findings."""

    # 1. Drop redundant features
    drop_cols = [
        # Low variance (4)
        'mouth_frown_mean', 'shoulder_width_var', 'nose_shoulder_dist_var', 'core_speed_mean',
        # Highly collinear (10)
        'engagement_score', 'agitation_score', 'emotion_happy_mean', 'emotion_fear_mean',
        'emotion_angry_mean', 'emotion_surprise_mean', 'Spectral_Rolloff',
        'gaze_deviation_mean', 'mouth_frown_std', 'jaw_open_std',
        # Metadata (exclude from features)
        'id', 'file_name', 'question_id', 'question', 'transcript'
    ]
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 2. Log1p transform skewed features
    skewed_cols = ['Total_Silence_Sec', 'Silence_Ratio', 'Pause_Count',
                   'Avg_Pause_Duration_Sec', 'Mean_Energy', 'Filler_Word_Count',
                   'hand_speed_mean', 'posture_shift_count']
    for col in skewed_cols:
        if col in df_clean.columns:
            df_clean[col] = np.log1p(df_clean[col])

    # 3. Ordinal encode duration_label
    duration_map = {'short': 0, 'medium': 1, 'long': 2}
    df_clean['duration_encoded'] = df_clean['duration_label'].map(duration_map)
    df_clean = df_clean.drop(columns=['duration_label'])

    # 4. PCA for MFCC features
    mfcc_cols = [f"MFCC_{i}" for i in range(1, 14)]
    pca = PCA(n_components=0.95)
    mfcc_pca = pca.fit_transform(df_clean[mfcc_cols])
    mfcc_df = pd.DataFrame(mfcc_pca, columns=[f"MFCC_PC{i+1}" for i in range(mfcc_pca.shape[1])])
    df_clean = df_clean.drop(columns=mfcc_cols).reset_index(drop=True)
    df_clean = pd.concat([df_clean, mfcc_df], axis=1)

    # 5. Impute transcript missing values
    if 'transcript' in df.columns:
        df_clean['transcript_missing'] = df['transcript'].isnull().astype(int)

    return df_clean
```

---

## 12. Recommendations and Next Steps

### Immediate (Preprocessing)

1. **Drop 14 redundant/low-variance features** — see Section 10.2, 10.3
2. **Apply log1p transformation** — see Section 10.4
3. **Encode duration_label** as ordinal — see Section 10.6
4. **Impute transcript** missing values — see Section 10.7
5. **Apply PCA to MFCC** (13 → 5-7 components) — see Section 10.1
6. **Use RobustScaler** for linear models — see Section 10.5

### Feature Engineering

1. **Candidate-level aggregation** — Mean, std, min, max per user_no — see Section 10.11
2. **NLP embeddings** — TF-IDF or BERT for transcripts — see Section 10.8
3. **Interaction features** — Personality x Audio, Facial x Posture
4. **Composite targets** — Mean of all 6 targets for single-output baseline

### Modelling

1. **Baseline models** — Ridge, Lasso, ElasticNet (with RobustScaler)
2. **Tree models** — XGBoost, LightGBM, Random Forest (no scaling)
3. **Multi-modal fusion** — Concatenate audio+facial+posture+personality
4. **Multi-task learning** — Shared backbone, 6 prediction heads
5. **Group-aware splitting** — GroupKFold(groups=user_no) — see Section 10.9

---

*Report generated from executed notebook: `notebooks/eda/interview_analyzer_eda.ipynb`*
