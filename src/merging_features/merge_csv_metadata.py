
#!/usr/bin/env python3
"""
Script to merge data/features/merged_features.csv and data/features/metadata.jsonl
into a single unified CSV file without column duplication or alignment drift.
"""

import os
import json
import pandas as pd

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(repo_root, "data", "features", "merged_features.csv")
    jsonl_path = os.path.join(repo_root, "data", "features", "metadata.jsonl")

    print(f"Reading CSV dataset: {csv_path}")
    df_csv = pd.read_csv(csv_path)
    print(f"CSV shape: {df_csv.shape}")

    print(f"Reading JSONL dataset: {jsonl_path}")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        jsonl_data = [json.loads(line) for line in f if line.strip()]
    df_jsonl = pd.DataFrame(jsonl_data)
    print(f"JSONL shape: {df_jsonl.shape}")

    # Create temporary normalized ID for alignment
    df_csv["_norm_id"] = df_csv["id"].astype(str).str.zfill(4)
    df_jsonl["_norm_id"] = df_jsonl["id"].astype(str).str.zfill(4)

    # Verify 1-to-1 row alignment
    if not (df_csv["_norm_id"] == df_jsonl["_norm_id"]).all():
        print("Warning: Row order mismatch detected! Merging on _norm_id explicitly.")

    # Unique columns in JSONL not present in original CSV
    jsonl_new_fields = [
        "video_quality", 
        "answer_score", 
        "speaking_skills", 
        "confidence_score", 
        "facial_expression", 
        "overall_performance"
    ]

    # Drop jsonl_new_fields from df_csv if already present (e.g. from previous runs)
    df_csv = df_csv.drop(columns=[c for c in jsonl_new_fields if c in df_csv.columns])
    if "gemini_summary" in df_csv.columns:
        df_csv = df_csv.drop(columns=["gemini_summary"])

    # Subset JSONL with _norm_id and new fields
    df_jsonl_sub = df_jsonl[["_norm_id"] + jsonl_new_fields]

    # Perform left join to preserve CSV structure and append new fields
    merged_df = pd.merge(df_csv, df_jsonl_sub, on="_norm_id", how="left")
    merged_df.drop(columns=["_norm_id"], inplace=True)

    # Standardize column taxonomy order
    metadata_cols = ["id", "file_name", "duration_label", "question_id", "question", "video_quality", "user_no"]
    score_cols = [
        "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism", "overall_personality",
        "interview_score", "answer_score", "speaking_skills", "confidence_score", "facial_expression", "overall_performance"
    ]
    audio_cols = ["Duration_Sec", "Speech_Rate_WPM", "Silence_Duration_Sec", "Mean_Pitch_Hz", "Mean_Energy", "Mean_ZCR"] + [f"MFCC_{i}" for i in range(1, 14)]
    face_cols = [
        "face_detected_ratio", "gaze_ratio_mean", "gaze_deviation_mean", "gaze_stability_std",
        "smile_score_mean", "smile_score_std", "frown_score_mean", "frown_score_std",
        "eye_openness_mean", "eye_openness_std", "jaw_open_mean", "jaw_open_std",
        "brow_raise_mean", "brow_raise_std", "mouth_frown_mean", "mouth_frown_std",
        "emotion_happy_mean", "emotion_sad_mean", "emotion_angry_mean", "emotion_surprise_mean",
        "emotion_fear_mean", "emotion_disgust_mean", "emotion_neutral_mean"
    ]
    posture_cols = [
        "head_centering_score_mean", "absolute_shoulder_slope_mean", "shoulder_slope_var",
        "shoulder_width_mean", "shoulder_width_var", "nose_shoulder_dist_mean", "nose_shoulder_dist_var",
        "hand_speed_mean", "hand_to_face_touches", "crossed_arms_score", "core_speed_mean",
        "posture_shift_count", "engagement_score", "agitation_score"
    ]
    text_cols = ["transcript"]

    ordered_cols = metadata_cols + score_cols + audio_cols + face_cols + posture_cols + text_cols

    # Reorder columns
    merged_df = merged_df[ordered_cols]

    print(f"Final merged shape: {merged_df.shape}")
    print(f"Duplicate columns count: {len(merged_df.columns) - len(set(merged_df.columns))}")

    # Overwrite merged_features.csv
    merged_df.to_csv(csv_path, index=False)
    print(f"Successfully saved merged features to: {csv_path}")

if __name__ == "__main__":
    main()
