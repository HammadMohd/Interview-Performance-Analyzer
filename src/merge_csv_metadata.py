#!/usr/bin/env python3
"""
Script to merge data/features/merged_features.csv and data/features/metadata.jsonl
into a single unified CSV file without column duplication or alignment drift.
Skips video_quality and gemini_summary as requested.
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
    print(f"CSV initial shape: {df_csv.shape}")

    print(f"Reading JSONL dataset: {jsonl_path}")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        jsonl_data = [json.loads(line) for line in f if line.strip()]
    df_jsonl = pd.DataFrame(jsonl_data)
    print(f"JSONL initial shape: {df_jsonl.shape}")

    # Fields to bring in from metadata.jsonl (skipping video_quality and gemini_summary)
    new_fields = [
        "answer_score", 
        "speaking_skills", 
        "confidence_score", 
        "facial_expression", 
        "overall_performance"
    ]

    # Drop new fields or excluded fields if already present in df_csv to prevent suffixes
    excluded_fields = ["video_quality", "gemini_summary"]
    for col in new_fields + excluded_fields:
        if col in df_csv.columns:
            df_csv.drop(columns=[col], inplace=True)

    # Standardize normalized ID for alignment
    df_csv["_norm_id"] = df_csv["id"].astype(str).str.zfill(4)
    df_jsonl["_norm_id"] = df_jsonl["id"].astype(str).str.zfill(4)

    # Verify 1-to-1 row alignment
    if not (df_csv["_norm_id"] == df_jsonl["_norm_id"]).all():
        print("Warning: Row order mismatch detected! Merging on _norm_id explicitly.")

    # Subset JSONL with _norm_id and new_fields
    df_jsonl_sub = df_jsonl[["_norm_id"] + new_fields]

    # Left join to preserve row alignment
    merged_df = pd.merge(df_csv, df_jsonl_sub, on="_norm_id", how="left")
    merged_df.drop(columns=["_norm_id"], inplace=True)

    # Reorder columns: place new sub-scores right after interview_score
    cols = merged_df.columns.tolist()
    base_cols = [c for c in cols if c not in new_fields]

    if "interview_score" in base_cols:
        idx = base_cols.index("interview_score")
        ordered_cols = base_cols[:idx+1] + new_fields + base_cols[idx+1:]
    else:
        ordered_cols = base_cols + new_fields

    merged_df = merged_df[ordered_cols]

    print(f"Final merged shape: {merged_df.shape}")
    print(f"Duplicate columns count: {len(merged_df.columns) - len(set(merged_df.columns))}")
    print(f"Contains video_quality: {'video_quality' in merged_df.columns}")
    print(f"Contains gemini_summary: {'gemini_summary' in merged_df.columns}")

    # Overwrite merged_features.csv
    merged_df.to_csv(csv_path, index=False)
    print(f"Successfully saved merged features to: {csv_path}")

if __name__ == "__main__":
    main()
