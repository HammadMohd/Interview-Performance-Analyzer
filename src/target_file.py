import json
import pandas as pd


def process_target_data(csv_filepath, jsonl_filepath, output_filepath="target.csv"):
    """Extracts target features from both CSV and JSONL files to generate a unified,

    formatted CSV file.

    IDs are formatted and padded into a standard 4-digit '0001' string format.
    """

    # 1. Define the target features/columns
    target_columns = [
        "id",
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
        "overall_personality",
        "interview_score",
        "answer_score",
        "speaking_skills",
        "confidence_score",
        "facial_expression",
        "overall_performance",
    ]

    records = []

    # -------------------------------------------------------------
    # 2. Process JSONL File (Line-by-Line)
    # -------------------------------------------------------------
    try:
        jsonl_count = 0
        with open(jsonl_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # Blank lines skip karna

                item = json.loads(line)

                # Filter and extract target columns only
                record = {col: item.get(col, None) for col in target_columns}

                # Standardize ID to '0001' format
                if record["id"] is not None:
                    record["id"] = f"{int(record['id']):04d}"

                records.append(record)
                jsonl_count += 1

        print(f"Successfully processed {jsonl_count} records from JSONL file.")
    except Exception as e:
        print(f"Error reading JSONL file: {e}")

    # -------------------------------------------------------------
    # 3. Process CSV File
    # -------------------------------------------------------------
    try:
        df_csv = pd.read_csv(csv_filepath)

        # Select target columns present in the CSV file
        existing_targets = [
            col for col in target_columns if col in df_csv.columns
        ]
        df_csv_targets = df_csv[existing_targets].copy()

        # Homogenize CSV IDs to '0001' format
        if "id" in df_csv_targets.columns:
            df_csv_targets["id"] = (
                df_csv_targets["id"].astype(int).apply(lambda x: f"{x:04d}")
            )

        # Convert dataframe to a list of dictionaries and add to records
        records.extend(df_csv_targets.to_dict(orient="records"))

        print(
            f"Successfully processed {len(df_csv_targets)} records from CSV file."
        )
    except Exception as e:
        print(f"Error reading CSV file: {e}")

    # -------------------------------------------------------------
    # 4. Merge Data and Export Final Target CSV
    # -------------------------------------------------------------
    if records:
        df_final = pd.DataFrame(records)

        # Handle missing columns
        df_final = df_final.reindex(columns=target_columns)

        # Drop duplicate IDs (retaining the first occurrence)
        df_final = df_final.drop_duplicates(subset=["id"], keep="first")

        # Sort values by the ID column
        df_final = df_final.sort_values(by="id").reset_index(drop=True)

        # Save to output CSV file
        df_final.to_csv(output_filepath, index=False)
        print(f"\nSuccess! Target CSV file generated: '{output_filepath}'")
        print(f"Total processed entries: {len(df_final)}")
        print("\nFirst few rows preview:")
        print(df_final.head())
    else:
        print("No records were processed.")



process_target_data(
    csv_filepath="final_merged_data_fixed.csv",  
    jsonl_filepath="notebooks/data/raw/AI4A-lab/RecruitView/metadata.jsonl",  
    output_filepath="target.csv",  
)