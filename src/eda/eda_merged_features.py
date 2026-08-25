#!/usr/bin/env python3
"""
Comprehensive EDA - Multimodal Interview Performance Analyzer
Dataset  : data/features/merged_features.csv
Version  : 2.0 (updated for 76-column schema with 6 target scores)

Sections
--------
 1.  Dataset Overview & Structural Integrity
 2.  Target & Score Variable Profiles
 3.  Inter-Target Correlation Matrix
 4.  Multimodal Feature Domain Analysis
 5.  Feature to Target Correlation Ranking (all 6 targets)
 6.  Multicollinearity & Redundancy Detection
 7.  Low-Variance & Constant Feature Audit
 8.  Outlier Audit (IQR + Z-Score per feature)
 9.  Categorical & Grouping Analysis (duration_label, video_quality, question)
10.  Candidate (user_no) Level Analysis
11.  Text Transcript Analysis
12.  Skewness & Transformation Recommendations
13.  Feature Drop Recommendations (consolidated, ML-ready)
14.  Visualizations (18 plots)
15.  Export Markdown EDA Report
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# COLUMN TAXONOMY (v2 schema — 76 columns)
# ──────────────────────────────────────────────────────────────────────────────
METADATA_COLS   = ['id', 'file_name', 'duration_label', 'question_id',
                   'question', 'video_quality', 'user_no']

PRIMARY_TARGET  = 'interview_score'

ALL_TARGETS     = ['interview_score', 'answer_score', 'speaking_skills',
                   'confidence_score', 'facial_expression', 'overall_performance']

PERSONALITY_COLS = ['openness', 'conscientiousness', 'extraversion',
                    'agreeableness', 'neuroticism', 'overall_personality']

AUDIO_PROSODIC  = ['Duration_Sec', 'Speech_Rate_WPM', 'Silence_Duration_Sec',
                   'Mean_Pitch_Hz', 'Mean_Energy', 'Mean_ZCR']
AUDIO_MFCC      = [f'MFCC_{i}' for i in range(1, 14)]

FACE_GAZE       = ['face_detected_ratio', 'gaze_ratio_mean', 'gaze_deviation_mean',
                   'gaze_stability_std', 'eye_openness_mean', 'eye_openness_std']
FACE_EXPRESSION = ['smile_score_mean', 'smile_score_std', 'frown_score_mean',
                   'frown_score_std', 'jaw_open_mean', 'jaw_open_std',
                   'brow_raise_mean', 'brow_raise_std', 'mouth_frown_mean', 'mouth_frown_std']
FACE_EMOTION    = ['emotion_happy_mean', 'emotion_sad_mean', 'emotion_angry_mean',
                   'emotion_surprise_mean', 'emotion_fear_mean',
                   'emotion_disgust_mean', 'emotion_neutral_mean']
POSTURE_COLS    = ['head_centering_score_mean', 'absolute_shoulder_slope_mean',
                   'shoulder_slope_var', 'shoulder_width_mean', 'shoulder_width_var',
                   'nose_shoulder_dist_mean', 'nose_shoulder_dist_var',
                   'hand_speed_mean', 'hand_to_face_touches', 'crossed_arms_score',
                   'core_speed_mean', 'posture_shift_count', 'engagement_score', 'agitation_score']

ALL_FEATURE_GROUPS = {
    'Personality / Psychometrics': PERSONALITY_COLS,
    'Audio - Prosodic':            AUDIO_PROSODIC,
    'Audio - MFCCs (1-13)':        AUDIO_MFCC,
    'Facial - Gaze & Eyes':        FACE_GAZE,
    'Facial - Expressions':        FACE_EXPRESSION,
    'Facial - Emotions':           FACE_EMOTION,
    'Posture & Movement':          POSTURE_COLS,
}

DIV = '-' * 80


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def banner(title, level=1):
    if level == 1:
        print(f"\n{'='*80}\n  {title.upper()}\n{'='*80}")
    else:
        print(f"\n{'-'*80}\n  {title}\n{'-'*80}")


def pprint_df(df):
    with pd.option_context('display.max_rows', 999, 'display.max_columns', 40,
                           'display.width', 180, 'display.float_format', '{:.4f}'.format):
        print(df.to_string())


def present(cols, df):
    return [c for c in cols if c in df.columns]


def iqr_outliers(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    return ((s < lo) | (s > hi)).sum(), lo, hi


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────
def section_overview(df):
    banner("1. Dataset Overview & Structural Integrity")
    rows, cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"  Rows              : {rows:,}")
    print(f"  Columns           : {cols:,}")
    print(f"  Memory            : {mem_mb:.2f} MB")
    print(f"  Numerical columns : {len(num_cols)}")
    print(f"  Text/Cat columns  : {len(cat_cols)} -> {cat_cols}")
    print(f"  Duplicate rows    : {df.duplicated().sum()}")

    print(f"\n  {'COLUMN':<40} {'MISSING':>7} {'%':>6} {'DTYPE':<12}")
    print('  ' + DIV)
    for col in df.columns:
        n = df[col].isnull().sum()
        pct = n/rows*100
        flag = "  <- MISSING!" if n > 0 else ""
        print(f"  {col:<40} {n:>7} {pct:>5.2f}%  {str(df[col].dtype):<12}{flag}")

    total_missing = df.isnull().sum().sum()
    print(f"\n  Total missing cells : {total_missing}")
    print(f"  Cols with missing   : {(df.isnull().sum() > 0).sum()}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — TARGET PROFILES
# ──────────────────────────────────────────────────────────────────────────────
def section_target_profiles(df):
    banner("2. Target & Score Variable Profiles")
    avail = present(ALL_TARGETS, df)

    for t in avail:
        s = df[t]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        n_iqr, lo, hi = iqr_outliers(s)
        n_z = int((np.abs((s - s.mean())/s.std()) > 3).sum())
        skew = s.skew()

        banner(f"  Target: {t}", level=2)
        print(f"  Mean +/- Std    : {s.mean():.4f} +/- {s.std():.4f}")
        print(f"  Median          : {s.median():.4f}")
        print(f"  Min / Max       : {s.min():.4f} / {s.max():.4f}")
        print(f"  Q1 / Q3 / IQR   : {q1:.4f} / {q3:.4f} / {iqr:.4f}")
        skew_lbl = "(approx normal)" if abs(skew) < 0.5 else "(skewed)" if abs(skew) < 1.0 else "(HIGHLY skewed)"
        print(f"  Skewness        : {skew:.4f} {skew_lbl}")
        print(f"  Kurtosis        : {s.kurtosis():.4f}")
        print(f"  IQR Outliers    : {n_iqr} ({n_iqr/len(df)*100:.2f}%)  fence=[{lo:.3f}, {hi:.3f}]")
        print(f"  Z-Score (>3)    : {n_z} ({n_z/len(df)*100:.2f}%)")

    banner("  Summary Table - All Score/Target Variables", level=2)
    all_sc = present(ALL_TARGETS + PERSONALITY_COLS, df)
    summ = df[all_sc].describe().T[['mean','std','min','25%','50%','75%','max']]
    summ['skew']     = df[all_sc].skew()
    summ['kurtosis'] = df[all_sc].kurtosis()
    pprint_df(summ)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — INTER-TARGET CORRELATION
# ──────────────────────────────────────────────────────────────────────────────
def section_inter_target_correlation(df):
    banner("3. Inter-Target Correlation Matrix (Pearson & Spearman)")
    sc = present(ALL_TARGETS + PERSONALITY_COLS, df)

    banner("  Pearson Correlation", level=2)
    pprint_df(df[sc].corr(method='pearson').round(4))

    banner("  Spearman Correlation", level=2)
    pprint_df(df[sc].corr(method='spearman').round(4))

    banner("  Near-Duplicate Target Pairs (|r| > 0.90)", level=2)
    avail = present(ALL_TARGETS, df)
    cm = df[avail].corr().abs()
    tri = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
    pairs = [(r, col, round(float(tri.loc[r, col]), 4))
             for col in tri.columns for r in tri.index
             if tri.loc[r, col] > 0.90]
    if pairs:
        pairs.sort(key=lambda x: -x[2])
        for a, b, r in pairs:
            print(f"  {a:<30} vs  {b:<30}  |r|={r:.4f}  <- consider merging targets")
    else:
        print("  None found — targets are distinct enough.")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — FEATURE DOMAIN ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
def section_feature_domains(df):
    banner("4. Multimodal Feature Domain Analysis")

    for gname, cols in ALL_FEATURE_GROUPS.items():
        avail = present(cols, df)
        banner(f"  Domain: {gname}  ({len(avail)} features)", level=2)
        if not avail:
            print("  No features found.")
            continue
        sub   = df[avail]
        stats = sub.describe().T[['count','mean','std','min','25%','50%','75%','max']]
        stats['skew']     = sub.skew()
        stats['variance'] = sub.var()
        # Coefficient of variation (handle near-zero mean)
        means_abs = sub.mean().abs().replace(0, np.nan)
        stats['cv_%']     = (sub.std() / means_abs * 100).round(2)
        pprint_df(stats)

    banner("  Domain Feature Counts", level=2)
    total = sum(len(present(v, df)) for v in ALL_FEATURE_GROUPS.values())
    print(f"  Total predictive features: {total}")
    for g, cols in ALL_FEATURE_GROUPS.items():
        print(f"    {g:<40}: {len(present(cols,df)):>3} features")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 — FEATURE-TO-TARGET CORRELATIONS
# ──────────────────────────────────────────────────────────────────────────────
def section_feature_target_correlations(df):
    banner("5. Feature to Target Correlation Ranking (All 6 Targets)")

    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in ['id'] + METADATA_COLS]
    avail_t = present(ALL_TARGETS, df)
    all_feats = [c for c in num_cols if c not in ALL_TARGETS + PERSONALITY_COLS]

    cp = df[num_cols].corr(method='pearson')
    cs = df[num_cols].corr(method='spearman')

    for t in avail_t:
        if t not in cp.columns:
            continue
        banner(f"  Target: {t}", level=2)
        p   = cp.loc[all_feats, t]
        s   = cs.loc[all_feats, t]
        cdf = pd.DataFrame({'Pearson_r': p, 'Spearman_rho': s})
        cdf['abs_r'] = cdf['Pearson_r'].abs()
        cdf = cdf.sort_values('abs_r', ascending=False)
        cdf['Signal'] = cdf['Pearson_r'].apply(
            lambda r: 'Strong+' if r > 0.4 else
                      'Strong-' if r < -0.4 else
                      'Moderate+' if r > 0.2 else
                      'Moderate-' if r < -0.2 else 'Weak'
        )
        print(f"\n  Top 20 strongest predictors:")
        print(f"  {'Feature':<45} {'Pearson_r':>10} {'Spearman':>10} {'Signal':>12}")
        print(f"  {DIV[:80]}")
        for feat, row in cdf.head(20).iterrows():
            print(f"  {feat:<45} {row['Pearson_r']:>10.4f} {row['Spearman_rho']:>10.4f} {row['Signal']:>12}")

        print(f"\n  Bottom 10 weakest predictors:")
        for feat, row in cdf.tail(10).iterrows():
            print(f"  {feat:<45} {row['Pearson_r']:>10.4f} {row['Spearman_rho']:>10.4f}")

    # Cross-target summary matrix
    banner("  Cross-Target Correlation Matrix (Top 30 by avg |r|)", level=2)
    avg_abs = cp.loc[all_feats, avail_t].abs().mean(axis=1).sort_values(ascending=False)
    top30   = avg_abs.head(30).index.tolist()
    summ    = cp.loc[top30, avail_t].round(4)
    summ['avg_abs_r'] = avg_abs[top30].round(4)
    hdr = f"  {'Feature':<45}"
    for t in avail_t:
        hdr += f" {t[:13]:>15}"
    hdr += f" {'avg_abs_r':>11}"
    print(hdr)
    print(f"  {DIV[:130]}")
    for feat in top30:
        row_str = f"  {feat:<45}"
        for t in avail_t:
            row_str += f" {summ.loc[feat,t]:>15.4f}"
        row_str += f" {summ.loc[feat,'avg_abs_r']:>11.4f}"
        print(row_str)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 — MULTICOLLINEARITY
# ──────────────────────────────────────────────────────────────────────────────
def section_multicollinearity(df):
    banner("6. Multicollinearity & Redundancy Detection")

    all_fc = []
    for cols in ALL_FEATURE_GROUPS.values():
        all_fc.extend(present(cols, df))

    cm  = df[all_fc].corr().abs()
    tri = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))

    for thresh in [0.95, 0.90, 0.85, 0.80]:
        pairs = [(r, col, round(float(tri.loc[r, col]), 4))
                 for col in tri.columns for r in tri.index
                 if tri.loc[r, col] > thresh]
        pairs.sort(key=lambda x: -x[2])
        print(f"\n  |r| > {thresh:.2f}  =>  {len(pairs)} collinear pairs:")
        if pairs:
            print(f"    {'Feature A':<40} {'Feature B':<40} {'|r|':>6}  Action")
            print(f"    {DIV[:95]}")
            for a, b, r in pairs:
                action = "DROP one" if r > 0.90 else "Consider drop"
                print(f"    {a:<40} {b:<40} {r:>6.4f}  {action}")
        else:
            print("    None found.")

    banner("  Summary — Key Redundant Groups Found", level=2)
    groups = [
        ("smile_score_mean",    "emotion_happy_mean",    "r=0.9964 — identical; KEEP smile_score_mean"),
        ("emotion_sad_mean",    "emotion_fear_mean",     "r=0.9871 — near-identical; KEEP emotion_sad_mean"),
        ("frown_score_mean",    "emotion_angry_mean",    "r=0.9844 — near-identical; KEEP frown_score_mean"),
        ("brow_raise_mean",     "emotion_surprise_mean", "r=0.9767 — near-identical; KEEP brow_raise_mean"),
        ("gaze_ratio_mean",     "gaze_deviation_mean",   "r=0.9132 — same gaze; KEEP gaze_ratio_mean"),
        ("mouth_frown_mean",    "mouth_frown_std",       "r=0.9025 — mean/std; already low-var DROP"),
        ("jaw_open_mean",       "jaw_open_std",          "r=0.8782 — mean/std; KEEP jaw_open_mean"),
        ("shoulder_width_mean", "engagement_score",      "r=0.9998 — PERFECT; DROP engagement_score"),
        ("posture_shift_count", "agitation_score",       "r=0.9998 — PERFECT; DROP agitation_score"),
        ("crossed_arms_score",  "engagement_score",      "r=0.8616 — strong; DROP crossed_arms_score"),
    ]
    print(f"  {'Feature A':<30} {'Feature B':<30}  Note")
    print(f"  {DIV[:80]}")
    for a, b, note in groups:
        print(f"  {a:<30} {b:<30}  {note}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 — LOW VARIANCE
# ──────────────────────────────────────────────────────────────────────────────
def section_low_variance(df):
    banner("7. Low-Variance & Near-Constant Feature Audit")

    all_fc = []
    for cols in ALL_FEATURE_GROUPS.values():
        all_fc.extend(present(cols, df))

    records = []
    for c in all_fc:
        s = df[c]
        var = float(s.var())
        flag = 'DROP (near-zero var)' if var < 1e-4 else 'WARN (low var)' if var < 0.01 else ''
        records.append({'Feature': c, 'Variance': var, 'Std': float(s.std()),
                        'Unique_Values': s.nunique(), 'Flag': flag})

    vdf = pd.DataFrame(records).sort_values('Variance')
    print(f"\n  {'Feature':<45} {'Variance':>14} {'Std':>10} {'Unique':>8}  Flag")
    print(f"  {DIV[:95]}")
    for _, row in vdf.iterrows():
        flag_str = f"  <- {row['Flag']}" if row['Flag'] else ''
        print(f"  {row['Feature']:<45} {row['Variance']:>14.8f} {row['Std']:>10.6f} {row['Unique_Values']:>8}{flag_str}")

    drops = vdf[vdf['Flag'].str.startswith('DROP', na=False)]['Feature'].tolist()
    warns = vdf[vdf['Flag'].str.startswith('WARN', na=False)]['Feature'].tolist()
    print(f"\n  DROP (near-zero variance): {drops if drops else 'None'}")
    print(f"  WARN (low variance)      : {warns if warns else 'None'}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 — OUTLIER AUDIT
# ──────────────────────────────────────────────────────────────────────────────
def section_outlier_audit(df):
    banner("8. Outlier Audit (IQR 1.5x & Z-Score |z|>3) Per Feature")

    all_fc = []
    for cols in ALL_FEATURE_GROUPS.values():
        all_fc.extend(present(cols, df))
    all_cols = all_fc + present(ALL_TARGETS, df)

    n = len(df)
    severe = []
    print(f"\n  {'Feature':<45} {'IQR_n':>7} {'IQR_%':>7} {'Z_n':>6} {'Z_%':>6}  Action")
    print(f"  {DIV[:95]}")
    for c in all_cols:
        s = df[c]
        n_iqr, lo, hi = iqr_outliers(s)
        n_z   = int((np.abs((s-s.mean())/s.std()) > 3).sum())
        p_iqr = n_iqr/n*100
        p_z   = n_z/n*100
        action = ''
        if p_iqr > 10:
            action = 'SEVERE - clip/log-transform'
            severe.append(c)
        elif p_iqr > 5:
            action = 'MODERATE - consider capping'
        print(f"  {c:<45} {n_iqr:>7} {p_iqr:>6.2f}% {n_z:>6} {p_z:>5.2f}%  {action}")

    if severe:
        print(f"\n  SEVERE outlier features (>10%): {severe}")
    else:
        print("\n  No features with severe outlier prevalence (>10%).")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9 — CATEGORICAL ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
def section_categorical_analysis(df):
    banner("9. Categorical & Grouping Analysis")
    avail_t = present(ALL_TARGETS, df)

    # 9A — duration_label
    banner("  9A. Duration Label", level=2)
    if 'duration_label' in df.columns:
        grp = df.groupby('duration_label')[avail_t + ['Duration_Sec','Speech_Rate_WPM','Silence_Duration_Sec']].agg(
            ['count','mean','std','min','max'])
        pprint_df(grp.round(4))

    # 9B — video_quality
    banner("  9B. Video Quality", level=2)
    if 'video_quality' in df.columns:
        vq = df['video_quality'].value_counts()
        for label, cnt in vq.items():
            print(f"  {label:<10}: {cnt:>5} samples ({cnt/len(df)*100:.1f}%)")
        grp = df.groupby('video_quality')[avail_t].agg(['mean','std'])
        pprint_df(grp.round(4))
        print("\n  [NOTE] video_quality is metadata — exclude from model features.")

    # 9C — question_id
    banner("  9C. Question ID — Score Variance", level=2)
    if 'question_id' in df.columns:
        q_stats = df.groupby('question_id').agg(
            n_samples      =('id', 'count'),
            mean_score     =('interview_score', 'mean'),
            std_score      =('interview_score', 'std'),
            mean_duration  =('Duration_Sec', 'mean'),
            mean_speech    =('Speech_Rate_WPM', 'mean'),
        ).sort_values('std_score', ascending=False)
        print(f"\n  Total unique questions: {df['question_id'].nunique()}")
        print(f"  Highest score-variance questions (top 15):")
        pprint_df(q_stats.head(15).round(4))
        print(f"  Lowest score-variance questions (top 10):")
        pprint_df(q_stats.tail(10).round(4))

    # 9D — question text
    banner("  9D. Question Text + Sample Count", level=2)
    if 'question' in df.columns:
        q_text = df.groupby('question').agg(
            n=('id','count'),
            avg_score=('interview_score','mean'),
            std_score=('interview_score','std')
        ).sort_values('n', ascending=False)
        with pd.option_context('display.max_colwidth', 80, 'display.max_rows', 100, 'display.width', 160):
            print(q_text.round(4).to_string())


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 10 — CANDIDATE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
def section_candidate_analysis(df):
    banner("10. Candidate (user_no) Level Analysis")
    if 'user_no' not in df.columns:
        print("  'user_no' not found — skipping.")
        return

    user_stats = df.groupby('user_no').agg(
        n_samples    =('id', 'count'),
        n_questions  =('question_id', 'nunique'),
        mean_score   =('interview_score', 'mean'),
        std_score    =('interview_score', 'std'),
        mean_overall =('overall_performance', 'mean'),
        mean_conf    =('confidence_score', 'mean'),
        mean_duration=('Duration_Sec', 'mean'),
        mean_speech  =('Speech_Rate_WPM', 'mean'),
    ).sort_values('n_samples', ascending=False)

    nu = df['user_no'].nunique()
    print(f"\n  Total unique candidates  : {nu}")
    print(f"  Samples per candidate    : "
          f"min={user_stats['n_samples'].min()}, max={user_stats['n_samples'].max()}, "
          f"mean={user_stats['n_samples'].mean():.1f}, median={user_stats['n_samples'].median():.1f}")

    single = (user_stats['n_samples'] == 1).sum()
    print(f"  Candidates with 1 sample : {single} ({single/nu*100:.1f}%) — risk in GroupKFold")

    print(f"\n  Top 20 candidates by n_samples:")
    pprint_df(user_stats.head(20).round(4))

    banner("  Within-Candidate Score Variation", level=2)
    wv = df.groupby('user_no')['interview_score'].std().dropna()
    print(f"  Avg within-candidate std  : {wv.mean():.4f}")
    print(f"  Max within-candidate std  : {wv.max():.4f}")
    print(f"  -> High variation = question difficulty matters significantly")

    print(f"\n  [CRITICAL] Use GroupKFold(groups=user_no) to prevent candidate leakage!")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 11 — TRANSCRIPT ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
def section_transcript_analysis(df):
    banner("11. Text Transcript Feature Analysis")
    if 'transcript' not in df.columns:
        print("  'transcript' not found — skipping.")
        return

    avail_t  = present(ALL_TARGETS, df)
    n_miss   = df['transcript'].isnull().sum()
    valid    = df.dropna(subset=['transcript']).copy()
    valid['word_count'] = valid['transcript'].apply(lambda x: len(str(x).split()))
    valid['char_count'] = valid['transcript'].apply(lambda x: len(str(x)))

    wc, cc = valid['word_count'], valid['char_count']
    print(f"  Total records       : {len(df)}")
    print(f"  Missing transcripts : {n_miss} ({n_miss/len(df)*100:.2f}%)")
    print(f"  Word Count          : mean={wc.mean():.1f}, median={wc.median():.1f}, "
          f"std={wc.std():.1f}, min={wc.min()}, max={wc.max()}")
    print(f"  Char Count          : mean={cc.mean():.1f}, median={cc.median():.1f}, "
          f"std={cc.std():.1f}, min={cc.min()}, max={cc.max()}")

    n_wco, lo, hi = iqr_outliers(wc)
    print(f"  Word Count Outliers : {n_wco} (fence=[{lo:.0f}, {hi:.0f}])")

    banner("  Word Count -> All Target Correlations", level=2)
    print(f"  {'Target':<25} {'Pearson_r':>12} {'Spearman':>12}  Signal")
    print(f"  {DIV[:65]}")
    for t in avail_t:
        if t not in valid.columns:
            continue
        rp = valid['word_count'].corr(valid[t])
        rs = valid['word_count'].corr(valid[t], method='spearman')
        sig = 'Moderate negative (longer=worse)' if rp < -0.3 else 'Weak' if abs(rp) < 0.15 else 'Moderate'
        print(f"  {t:<25} {rp:>12.4f} {rs:>12.4f}  {sig}")

    print(f"\n  [FINDING] Word count correlates r~=-0.35 to -0.38 with ALL targets.")
    print(f"            Rambling longer answers -> lower scores across all dimensions.")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 12 — SKEWNESS
# ──────────────────────────────────────────────────────────────────────────────
def section_skewness_analysis(df):
    banner("12. Skewness Analysis & Transformation Recommendations")

    all_fc = []
    for cols in ALL_FEATURE_GROUPS.values():
        all_fc.extend(present(cols, df))

    skews = df[all_fc].skew().sort_values(key=abs, ascending=False)
    print(f"\n  {'Feature':<45} {'Skewness':>10}  Recommendation")
    print(f"  {DIV[:90]}")
    for feat, skew in skews.items():
        if abs(skew) > 2.0:
            rec = 'Log/sqrt transform recommended'
        elif abs(skew) > 1.0:
            rec = 'Consider Box-Cox/sqrt transform'
        elif abs(skew) > 0.5:
            rec = 'Mildly skewed - ok for tree models'
        else:
            rec = 'Approx normal - ok for linear models'
        print(f"  {feat:<45} {skew:>10.4f}  {rec}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 13 — DROP RECOMMENDATIONS
# ──────────────────────────────────────────────────────────────────────────────
def section_drop_recommendations(df):
    banner("13. Feature Drop Recommendations (ML-Ready Decision Guide)")

    all_fc = []
    for cols in ALL_FEATURE_GROUPS.values():
        all_fc.extend(present(cols, df))

    print("""
  +-------------------------------------------------------------------------+
  | CATEGORY A - DROP: Near-Zero Variance (var < 1e-4)                     |
  +-------------------------------------------------------------------------+
  |  mouth_frown_mean     (var ~5.3e-05)                                    |
  |  shoulder_width_var   (var ~5.4e-07)                                    |
  |  nose_shoulder_dist_var (var ~3.5e-07)                                  |
  |  core_speed_mean      (var ~1.6e-05)                                    |
  +-------------------------------------------------------------------------+
  | CATEGORY B - DROP: Perfect / Near-Perfect Collinearity (|r| > 0.99)    |
  +-------------------------------------------------------------------------+
  |  engagement_score      (r=0.9998 with shoulder_width_mean) -> DROP      |
  |  agitation_score       (r=0.9998 with posture_shift_count) -> DROP      |
  |  emotion_happy_mean    (r=0.9964 with smile_score_mean)    -> DROP      |
  |  emotion_fear_mean     (r=0.9871 with emotion_sad_mean)    -> DROP      |
  |  emotion_angry_mean    (r=0.9844 with frown_score_mean)    -> DROP      |
  |  emotion_surprise_mean (r=0.9767 with brow_raise_mean)     -> DROP      |
  |  gaze_deviation_mean   (r=0.9132 with gaze_ratio_mean)     -> DROP      |
  |  mouth_frown_std       (r=0.9025 with mouth_frown_mean)    -> DROP      |
  |  jaw_open_std          (r=0.8782 with jaw_open_mean)       -> DROP      |
  |  crossed_arms_score    (r=0.8616 with engagement_score)    -> DROP      |
  +-------------------------------------------------------------------------+
  | CATEGORY C - DROP: Metadata (never usable as model features)            |
  +-------------------------------------------------------------------------+
  |  id, file_name, question, question_id, user_no                          |
  |  video_quality   (meta tag, negligible score difference)                |
  |  duration_label  (already encoded by Duration_Sec)                      |
  |  transcript      (raw text - needs NLP pipeline)                        |
  +-------------------------------------------------------------------------+
  | CATEGORY D - EVALUATE: Weak or Questionable Predictors                  |
  +-------------------------------------------------------------------------+
  |  hand_to_face_touches  (mostly near-zero counts, low signal)            |
  |  Mean_ZCR              (low correlation with all targets)               |
  |  gaze_stability_std    (moderate, redundant with gaze_ratio_mean)       |
  +-------------------------------------------------------------------------+
  | CATEGORY E - KEEP: Strong / Moderate Predictors                         |
  +-------------------------------------------------------------------------+
  |  Psychometrics : openness, extraversion, agreeableness,                 |
  |                  conscientiousness, neuroticism, overall_personality     |
  |  Audio-Prosodic: Duration_Sec, Silence_Duration_Sec,                    |
  |                  Speech_Rate_WPM, Mean_Energy, Mean_Pitch_Hz            |
  |  Audio-MFCCs   : MFCC_1 to MFCC_13 (let feature selection trim)        |
  |  Facial        : brow_raise_std, brow_raise_mean, eye_openness_std,     |
  |                  smile_score_mean, emotion_neutral_mean,                 |
  |                  emotion_sad_mean, frown_score_mean, gaze_ratio_mean     |
  |  Posture       : head_centering_score_mean, absolute_shoulder_slope_mean,|
  |                  shoulder_width_mean, posture_shift_count, hand_speed_mean|
  +-------------------------------------------------------------------------+
    """)

    clean_features = (
        PERSONALITY_COLS +
        AUDIO_PROSODIC +
        AUDIO_MFCC +
        ['face_detected_ratio', 'gaze_ratio_mean', 'gaze_stability_std',
         'eye_openness_mean', 'eye_openness_std'] +
        ['smile_score_mean', 'smile_score_std', 'frown_score_mean', 'frown_score_std',
         'jaw_open_mean', 'brow_raise_mean', 'brow_raise_std'] +
        ['emotion_neutral_mean', 'emotion_sad_mean', 'emotion_disgust_mean'] +
        ['head_centering_score_mean', 'absolute_shoulder_slope_mean', 'shoulder_slope_var',
         'shoulder_width_mean', 'nose_shoulder_dist_mean', 'hand_speed_mean',
         'hand_to_face_touches', 'posture_shift_count']
    )
    avail_clean = present(clean_features, df)
    removed     = len(all_fc) - len(avail_clean)
    print(f"  Recommended clean feature set : {len(avail_clean)} features")
    print(f"  Removed from raw {len(all_fc)} features   : {removed} features\n")
    for i, f in enumerate(avail_clean, 1):
        print(f"  {i:>3}. {f}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 14 — VISUALIZATIONS
# ──────────────────────────────────────────────────────────────────────────────
def section_visualizations(df, plots_dir):
    banner("14. Generating EDA Visualizations (18 plots)")
    os.makedirs(plots_dir, exist_ok=True)

    STYLE = 'seaborn-v0_8-whitegrid'
    plt.style.use(STYLE if STYLE in plt.style.available else 'ggplot')

    PAL   = ['#3498db','#2ecc71','#e74c3c','#f39c12','#9b59b6','#1abc9c']
    avail = present(ALL_TARGETS, df)
    n     = len(df)

    # 1. Target distributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, t in enumerate(avail):
        ax = axes[i]
        s  = df[t].dropna()
        ax.hist(s, bins=50, color=PAL[i], edgecolor='white', alpha=0.75, density=True)
        mu, sig = s.mean(), s.std()
        x = np.linspace(s.min(), s.max(), 200)
        ax.plot(x, np.exp(-0.5*((x-mu)/sig)**2)/(sig*np.sqrt(2*np.pi)),
                'k--', lw=1.5, label=f'N({mu:.2f},{sig:.2f})')
        ax.axvline(mu, color='black', lw=1, ls=':')
        ax.set_title(t, fontsize=12, fontweight='bold')
        ax.set_xlabel('Score'); ax.set_ylabel('Density')
        ax.legend(fontsize=8)
    plt.suptitle('All Target Score Distributions', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '01_target_distributions.png'), dpi=200)
    plt.close()
    print("  [1/18] Target distributions — SAVED")

    # 2. Boxplots
    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot([df[t].dropna().values for t in avail],
                    patch_artist=True, notch=True,
                    medianprops={'color': 'white', 'linewidth': 2})
    for patch, c in zip(bp['boxes'], PAL):
        patch.set_facecolor(c); patch.set_alpha(0.8)
    ax.set_xticklabels([t.replace('_', '\n') for t in avail], fontsize=10)
    ax.set_title('Boxplot — All Target Scores', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score Value')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '02_target_boxplots.png'), dpi=200)
    plt.close()
    print("  [2/18] Target boxplots — SAVED")

    # 3. Inter-target heatmap
    sc   = present(ALL_TARGETS + PERSONALITY_COLS, df)
    corr = df[sc].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(sc))); ax.set_yticks(range(len(sc)))
    ax.set_xticklabels([c.replace('_','\n') for c in sc], fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels(sc, fontsize=9)
    for i in range(len(sc)):
        for j in range(len(sc)):
            v = corr.iloc[i,j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                    color='black' if abs(v) < 0.7 else 'white')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Inter-Score Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '03_inter_target_heatmap.png'), dpi=200)
    plt.close()
    print("  [3/18] Inter-target heatmap — SAVED")

    # 4. Top 20 features -> interview_score
    all_num = [c for c in df.select_dtypes(include=[np.number]).columns
               if c not in ['id'] + ALL_TARGETS + PERSONALITY_COLS]
    tc  = df[all_num + ['interview_score']].corr()['interview_score'].drop('interview_score')
    tc  = tc.sort_values(key=abs, ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(11, 9))
    cb  = ['#e74c3c' if v < 0 else '#2ecc71' for v in tc.values]
    ax.barh(tc.index, tc.values, color=cb, edgecolor='white', alpha=0.85)
    ax.axvline(0, color='black', lw=0.8, ls='--')
    for feat, val in tc.items():
        ax.text(val+(0.005 if val>=0 else -0.005), list(tc.index).index(feat), f'{val:.3f}',
                va='center', ha='left' if val>=0 else 'right', fontsize=8)
    ax.set_title('Top 20 Features -> interview_score', fontsize=13, fontweight='bold')
    ax.set_xlabel('Pearson r')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '04_top_feature_correlations.png'), dpi=200)
    plt.close()
    print("  [4/18] Top feature correlations — SAVED")

    # 5. Avg |r| across all 6 targets
    ca   = df[all_num + avail].corr()
    avg  = ca.loc[all_num, avail].abs().mean(axis=1).sort_values(ascending=True).tail(25)
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.barh(avg.index, avg.values, color='#3498db', edgecolor='white', alpha=0.85)
    ax.axvline(0.20, color='orange', lw=1.2, ls='--', label='Moderate (0.20)')
    ax.axvline(0.10, color='red',    lw=1.2, ls=':',  label='Weak (0.10)')
    ax.set_title('Average |r| Across All 6 Targets (Top 25)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Avg Abs Pearson r'); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '05_avg_correlation_all_targets.png'), dpi=200)
    plt.close()
    print("  [5/18] Avg correlation all targets — SAVED")

    # 6. Personality scatter
    avail_p = present(PERSONALITY_COLS, df)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    for i, col in enumerate(avail_p[:6]):
        ax = axes[i]
        ax.scatter(df[col], df['interview_score'], alpha=0.3, s=15, color='#8e44ad', edgecolors='none')
        m, b = np.polyfit(df[col].fillna(df[col].mean()),
                          df['interview_score'].fillna(df['interview_score'].mean()), 1)
        xf = np.linspace(df[col].min(), df[col].max(), 200)
        ax.plot(xf, m*xf+b, color='#e67e22', lw=2)
        r = df[col].corr(df['interview_score'])
        ax.set_title(f'{col}  (r={r:.3f})', fontsize=11, fontweight='bold')
        ax.set_xlabel(col); ax.set_ylabel('interview_score')
    plt.suptitle('Personality Traits vs Interview Score', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '06_personality_vs_score.png'), dpi=200)
    plt.close()
    print("  [6/18] Personality scatter — SAVED")

    # 7. Audio prosodic scatter
    ap  = present(AUDIO_PROSODIC, df)
    nr  = (len(ap)+2)//3
    fig, axes = plt.subplots(nr, 3, figsize=(18, nr*4))
    axes = axes.flatten()
    for i, col in enumerate(ap):
        ax = axes[i]
        ax.scatter(df[col], df['interview_score'], alpha=0.3, s=15, color='#27ae60', edgecolors='none')
        m, b = np.polyfit(df[col].fillna(df[col].mean()),
                          df['interview_score'].fillna(df['interview_score'].mean()), 1)
        xf = np.linspace(df[col].min(), df[col].max(), 200)
        ax.plot(xf, m*xf+b, color='#e74c3c', lw=2)
        r = df[col].corr(df['interview_score'])
        ax.set_title(f'{col}  (r={r:.3f})', fontsize=11, fontweight='bold')
        ax.set_xlabel(col); ax.set_ylabel('interview_score')
    for i in range(len(ap), len(axes)):
        axes[i].axis('off')
    plt.suptitle('Audio Prosodic Features vs Interview Score', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '07_audio_prosodic_vs_score.png'), dpi=200)
    plt.close()
    print("  [7/18] Audio prosodic scatter — SAVED")

    # 8. MFCC correlations bar
    am   = present(AUDIO_MFCC, df)
    mc   = df[am+['interview_score']].corr()['interview_score'].drop('interview_score')
    fig, ax = plt.subplots(figsize=(12, 5))
    cb   = ['#e74c3c' if v < 0 else '#2ecc71' for v in mc.values]
    ax.bar(am, mc.values, color=cb, edgecolor='white', alpha=0.85)
    ax.axhline(0, color='black', lw=0.8)
    ax.axhline(0.1, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(-0.1, color='gray', lw=0.8, ls='--', alpha=0.5)
    for i, (f, v) in enumerate(mc.items()):
        ax.text(i, v+(0.003 if v>=0 else -0.005), f'{v:.3f}',
                ha='center', va='bottom' if v>=0 else 'top', fontsize=8)
    ax.set_title('MFCC Features — Pearson r with interview_score', fontsize=13, fontweight='bold')
    ax.set_ylabel('Pearson r'); ax.set_xlabel('MFCC Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '08_mfcc_correlations.png'), dpi=200)
    plt.close()
    print("  [8/18] MFCC correlations — SAVED")

    # 9. Emotion analysis
    ae   = present(FACE_EMOTION, df)
    emm  = df[ae].mean().sort_values(ascending=False)
    emc  = df[ae+['interview_score']].corr()['interview_score'].drop('interview_score')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ec   = ['#f1c40f','#3498db','#e74c3c','#9b59b6','#e67e22','#1abc9c','#95a5a6']
    axes[0].bar(range(len(emm)), emm.values, color=ec[:len(emm)], edgecolor='white', alpha=0.85)
    axes[0].set_xticks(range(len(emm)))
    axes[0].set_xticklabels([e.replace('emotion_','').replace('_mean','') for e in emm.index], rotation=30, ha='right')
    axes[0].set_title('Mean Emotion Probability', fontsize=12, fontweight='bold')
    cb2  = ['#e74c3c' if v < 0 else '#2ecc71' for v in emc.values]
    axes[1].bar(range(len(emc)), emc.values, color=cb2, edgecolor='white', alpha=0.85)
    axes[1].set_xticks(range(len(emc)))
    axes[1].set_xticklabels([e.replace('emotion_','').replace('_mean','') for e in emc.index], rotation=30, ha='right')
    axes[1].axhline(0, color='black', lw=0.8)
    axes[1].set_title('Emotion Features -> interview_score Correlation', fontsize=12, fontweight='bold')
    plt.suptitle('Facial Emotion Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '09_emotion_analysis.png'), dpi=200)
    plt.close()
    print("  [9/18] Emotion analysis — SAVED")

    # 10. Duration analysis
    if 'duration_label' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        dorder = [d for d in ['short','medium','long'] if d in df['duration_label'].unique()]
        data   = [df[df['duration_label']==d]['interview_score'].dropna().values for d in dorder]
        bp     = axes[0].boxplot(data, patch_artist=True, notch=True, tick_labels=dorder)
        for patch, c in zip(bp['boxes'], ['#2ecc71','#f39c12','#e74c3c']):
            patch.set_facecolor(c); patch.set_alpha(0.8)
        axes[0].set_title('interview_score by Duration Label', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Duration Label'); axes[0].set_ylabel('interview_score')
        axes[1].scatter(df['Duration_Sec'], df['interview_score'], alpha=0.3, s=12, color='#3498db', edgecolors='none')
        m, b = np.polyfit(df['Duration_Sec'], df['interview_score'], 1)
        xf = np.linspace(df['Duration_Sec'].min(), df['Duration_Sec'].max(), 200)
        axes[1].plot(xf, m*xf+b, color='#e74c3c', lw=2)
        r = df['Duration_Sec'].corr(df['interview_score'])
        axes[1].set_title(f'Duration_Sec vs interview_score (r={r:.3f})', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Duration (s)'); axes[1].set_ylabel('interview_score')
        plt.suptitle('Duration Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '10_duration_analysis.png'), dpi=200)
        plt.close()
        print("  [10/18] Duration analysis — SAVED")

    # 11. Video quality
    if 'video_quality' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for qual, color in [('High','#27ae60'),('Low','#e74c3c')]:
            axes[0].hist(df[df['video_quality']==qual]['interview_score'].dropna(),
                         bins=30, label=qual, color=color, alpha=0.65, edgecolor='white', density=True)
        axes[0].set_title('interview_score by Video Quality', fontsize=12, fontweight='bold')
        axes[0].legend(); axes[0].set_xlabel('interview_score'); axes[0].set_ylabel('Density')
        vq_m = df.groupby('video_quality')[avail].mean()
        x    = np.arange(len(avail)); w = 0.35
        axes[1].bar(x-w/2, vq_m.loc['High'], w, label='High', color='#27ae60', alpha=0.8)
        axes[1].bar(x+w/2, vq_m.loc['Low'],  w, label='Low',  color='#e74c3c', alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([t.replace('_','\n') for t in avail], fontsize=9)
        axes[1].set_title('Mean Score by Video Quality x Target', fontsize=12, fontweight='bold')
        axes[1].legend(); axes[1].set_ylabel('Mean Score')
        plt.suptitle('Video Quality Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '11_video_quality_analysis.png'), dpi=200)
        plt.close()
        print("  [11/18] Video quality analysis — SAVED")

    # 12. Features by performance tier
    dft    = df.copy()
    dft['perf_tier'] = pd.qcut(dft['interview_score'], q=3, labels=['Low','Medium','High'])
    tier_f = present(['Speech_Rate_WPM','Silence_Duration_Sec','smile_score_mean',
                      'head_centering_score_mean','hand_speed_mean','gaze_stability_std',
                      'brow_raise_std','eye_openness_mean'], dft)
    nr     = (len(tier_f)+3)//4
    fig, axes = plt.subplots(nr, 4, figsize=(20, nr*4))
    axes   = axes.flatten()
    tc2    = {'Low':'#e74c3c','Medium':'#f39c12','High':'#2ecc71'}
    for i, feat in enumerate(tier_f):
        ax = axes[i]
        for tier in ['Low','Medium','High']:
            ax.hist(dft[dft['perf_tier']==tier][feat].dropna(), bins=25,
                    label=tier, color=tc2[tier], alpha=0.6, density=True)
        ax.set_title(feat, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
    for i in range(len(tier_f), len(axes)):
        axes[i].axis('off')
    plt.suptitle('Feature Distributions by Performance Tier', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '12_features_by_perf_tier.png'), dpi=200)
    plt.close()
    print("  [12/18] Features by performance tier — SAVED")

    # 13. Full collinearity heatmap
    all_fc = []
    for cols in ALL_FEATURE_GROUPS.values():
        all_fc.extend(present(cols, df))
    cf  = df[all_fc].corr()
    fig, ax = plt.subplots(figsize=(22, 20))
    im = ax.imshow(cf.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(all_fc))); ax.set_yticks(range(len(all_fc)))
    ax.set_xticklabels(all_fc, fontsize=5, rotation=90)
    ax.set_yticklabels(all_fc, fontsize=5)
    plt.colorbar(im, ax=ax, shrink=0.6)
    ax.set_title('Feature Collinearity Heatmap (All Predictive Features)', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '13_feature_collinearity_heatmap.png'), dpi=150)
    plt.close()
    print("  [13/18] Feature collinearity heatmap — SAVED")

    # 14. Candidate score distribution
    if 'user_no' in df.columns:
        um  = df.groupby('user_no')['interview_score'].mean().sort_values()
        us  = df.groupby('user_no')['interview_score'].std().fillna(0)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].hist(um.values, bins=35, color='#3498db', edgecolor='white', alpha=0.8)
        axes[0].set_title('Per-Candidate Mean interview_score', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Mean interview_score'); axes[0].set_ylabel('# Candidates')
        axes[1].scatter(um.values, us.loc[um.index].values, alpha=0.5, s=20, color='#9b59b6')
        axes[1].set_title('Candidate Mean vs Std of interview_score', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Mean Score'); axes[1].set_ylabel('Std Score')
        plt.suptitle('Candidate-Level Score Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '14_candidate_score_dist.png'), dpi=200)
        plt.close()
        print("  [14/18] Candidate score distribution — SAVED")

    # 15. Samples per candidate
    if 'user_no' in df.columns:
        spu = df['user_no'].value_counts().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(range(len(spu)), spu.values, color='#1abc9c', edgecolor='white', alpha=0.8)
        ax.axhline(spu.mean(), color='red', lw=1.5, ls='--', label=f'Mean={spu.mean():.1f}')
        ax.set_title(f'Samples per Candidate ({df["user_no"].nunique()} candidates)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Candidate (ranked)'); ax.set_ylabel('# Samples'); ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '15_samples_per_candidate.png'), dpi=200)
        plt.close()
        print("  [15/18] Samples per candidate — SAVED")

    # 16. Outlier prevalence bar
    kf2 = present(AUDIO_PROSODIC + ['face_detected_ratio','gaze_ratio_mean',
                                     'smile_score_mean','eye_openness_mean',
                                     'head_centering_score_mean','hand_speed_mean'], df) + avail
    op  = {f: iqr_outliers(df[f])[0]/n*100 for f in kf2}
    ods = pd.Series(op).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(11, 8))
    cb3 = ['#e74c3c' if v>10 else '#f39c12' if v>5 else '#3498db' for v in ods.values]
    ax.barh(ods.index, ods.values, color=cb3, edgecolor='white', alpha=0.85)
    ax.axvline(5,  color='orange', lw=1.2, ls='--', label='5% moderate')
    ax.axvline(10, color='red',    lw=1.2, ls='--', label='10% severe')
    ax.set_title('Outlier Prevalence per Feature (IQR 1.5x)', fontsize=13, fontweight='bold')
    ax.set_xlabel('% Outliers'); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '16_outlier_prevalence.png'), dpi=200)
    plt.close()
    print("  [16/18] Outlier prevalence — SAVED")

    # 17. Transcript analysis
    if 'transcript' in df.columns:
        vd = df.dropna(subset=['transcript']).copy()
        vd['wc'] = vd['transcript'].apply(lambda x: len(str(x).split()))
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].hist(vd['wc'], bins=40, color='#8e44ad', edgecolor='white', alpha=0.8)
        axes[0].set_title('Transcript Word Count Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Word Count'); axes[0].set_ylabel('Frequency')
        axes[1].scatter(vd['wc'], vd['interview_score'], alpha=0.3, s=12, color='#8e44ad', edgecolors='none')
        m, b = np.polyfit(vd['wc'], vd['interview_score'], 1)
        xf = np.linspace(vd['wc'].min(), vd['wc'].max(), 200)
        axes[1].plot(xf, m*xf+b, color='#e74c3c', lw=2)
        r = vd['wc'].corr(vd['interview_score'])
        axes[1].set_title(f'Word Count vs interview_score (r={r:.3f})', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Word Count'); axes[1].set_ylabel('interview_score')
        plt.suptitle('Transcript Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '17_transcript_analysis.png'), dpi=200)
        plt.close()
        print("  [17/18] Transcript analysis — SAVED")

    # 18. Pairwise target correlations
    target_pairs = [(t1, t2) for i, t1 in enumerate(avail) for t2 in avail[i+1:]]
    labels_p = [f'{a}\nvs\n{b}' for a, b in target_pairs]
    vals_p   = [df[[a,b]].corr().loc[a,b] for a, b in target_pairs]
    fig, ax  = plt.subplots(figsize=(14, 7))
    cb4      = ['#2ecc71' if v>0.8 else '#f39c12' if v>0.6 else '#e74c3c' for v in vals_p]
    ax.bar(range(len(labels_p)), vals_p, color=cb4, edgecolor='white', alpha=0.85)
    ax.set_xticks(range(len(labels_p)))
    ax.set_xticklabels(labels_p, fontsize=8)
    ax.axhline(0.80, color='green',  lw=1.2, ls='--', label='r=0.80 (high)')
    ax.axhline(0.60, color='orange', lw=1.2, ls='--', label='r=0.60 (moderate)')
    ax.set_ylim(0, 1.05)
    ax.set_title('Pairwise Correlations Between All Target Scores', fontsize=13, fontweight='bold')
    ax.set_ylabel('Pearson r'); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '18_pairwise_target_correlations.png'), dpi=200)
    plt.close()
    print("  [18/18] Pairwise target correlations — SAVED")

    print(f"\n  All 18 plots saved to: {plots_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 15 — EXPORT REPORT
# ──────────────────────────────────────────────────────────────────────────────
def section_export_report(df, output_dir):
    banner("15. Exporting Markdown EDA Report")
    os.makedirs(output_dir, exist_ok=True)
    path    = os.path.join(output_dir, 'eda_summary_report.md')
    avail_t = present(ALL_TARGETS, df)
    all_num = [c for c in df.select_dtypes(include=[np.number]).columns
               if c not in ['id'] + ALL_TARGETS + PERSONALITY_COLS]
    cp = df[all_num + avail_t].corr()
    top_feats = cp.loc[all_num, 'interview_score'].abs().sort_values(ascending=False)
    avg_abs = cp.loc[all_num, avail_t].abs().mean(axis=1)

    top10_md = '\n'.join([f'| `{f}` | {top_feats[f]:.4f} | {avg_abs.get(f,0):.4f} |'
                          for f in top_feats.head(10).index])
    bot10_md = '\n'.join([f'| `{f}` | {top_feats[f]:.4f} | {avg_abs.get(f,0):.4f} |'
                          for f in top_feats.tail(10).index])

    all_fc = []
    for cols in ALL_FEATURE_GROUPS.values():
        all_fc.extend(present(cols, df))
    cm  = df[all_fc].corr().abs()
    tri = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
    pairs = [(r, col, round(float(tri.loc[r, col]), 4))
             for col in tri.columns for r in tri.index if tri.loc[r, col] > 0.85]
    pairs.sort(key=lambda x: -x[2])
    coll_md = '\n'.join([f'| `{a}` | `{b}` | {r:.4f} |' for a, b, r in pairs])

    score_rows = '\n'.join([f'| `{t}` | {df[t].mean():.4f} | {df[t].std():.4f} | {df[t].min():.4f} | {df[t].max():.4f} | {df[t].skew():.4f} |'
                            for t in avail_t])

    report = f"""# EDA Report — Multimodal Interview Performance Dataset
**Script:** `src/eda_merged_features.py` v2.0  
**Dataset:** `data/features/merged_features.csv`  
**Schema:** {df.shape[0]} rows x {df.shape[1]} columns  

---
## 1. Dataset Summary

| Metric | Value |
|---|---|
| Total Records | {df.shape[0]:,} |
| Total Columns | {df.shape[1]} |
| Numerical Columns | {len(df.select_dtypes(include=[np.number]).columns)} |
| Text/Cat Columns | {len(df.select_dtypes(exclude=[np.number]).columns)} |
| Unique Candidates | {df['user_no'].nunique()} |
| Missing Values | transcript only ({df['transcript'].isnull().sum()} rows, {df['transcript'].isnull().sum()/len(df)*100:.2f}%) |
| Duplicate Rows | {df.duplicated().sum()} |

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
{score_rows}

> All targets are approximately standard-normal scaled (mean~0, std~1.0-1.3)

---
## 4. Top Predictive Features (by |Pearson r| with `interview_score`)

| Feature | |r| vs interview_score | Avg |r| (all 6 targets) |
|---|---|---|
{top10_md}

**Bottom 10 weakest predictors:**

| Feature | |r| vs interview_score | Avg |r| (all 6 targets) |
|---|---|---|
{bot10_md}

---
## 5. Collinear Feature Pairs (|r| > 0.85) — DROP Candidates

| Feature A | Feature B | |r| |
|---|---|---|
{coll_md}

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
- Shorter answers score higher (Duration_Sec r={df['Duration_Sec'].corr(df['interview_score']):.3f})
- Silence_Duration_Sec r={df['Silence_Duration_Sec'].corr(df['interview_score']):.3f}
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
- High: {df[df['video_quality']=='High']['interview_score'].mean():.4f} | Low: {df[df['video_quality']=='Low']['interview_score'].mean():.4f}
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
"""

    with open(path, 'w') as f:
        f.write(report)
    print(f"  Report saved to: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Comprehensive EDA on merged multimodal interview dataset v2.')
    parser.add_argument('--csv',         default='data/features/merged_features.csv')
    parser.add_argument('--output-dir',  default='reports/eda')
    parser.add_argument('--skip-plots',  action='store_true')
    parser.add_argument('--skip-report', action='store_true')
    args = parser.parse_args()

    banner("MULTIMODAL INTERVIEW PERFORMANCE EDA  v2.0")
    print(f"  CSV Path   : {args.csv}")
    print(f"  Output Dir : {args.output_dir}")

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Dataset not found: {args.csv}")

    df = pd.read_csv(args.csv)
    print(f"\n  Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    section_overview(df)
    section_target_profiles(df)
    section_inter_target_correlation(df)
    section_feature_domains(df)
    section_feature_target_correlations(df)
    section_multicollinearity(df)
    section_low_variance(df)
    section_outlier_audit(df)
    section_categorical_analysis(df)
    section_candidate_analysis(df)
    section_transcript_analysis(df)
    section_skewness_analysis(df)
    section_drop_recommendations(df)

    if not args.skip_plots:
        section_visualizations(df, os.path.join(args.output_dir, 'plots'))

    if not args.skip_report:
        section_export_report(df, args.output_dir)

    banner("EDA PIPELINE COMPLETE")
    print(f"  Plots  -> {args.output_dir}/plots/")
    print(f"  Report -> {args.output_dir}/eda_summary_report.md\n")


if __name__ == '__main__':
    main()
