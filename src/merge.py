import pandas as pd

# 1. Load the three source files
df_audio = pd.read_csv("notebooks/data/recruitview_final_extracted_features.csv")
df_member1 = pd.read_csv("member1_data.csv")
df_member2 = pd.read_csv("member2_data.csv")

# 2. Correct Horizontal Merge using 'id'
df_correct = df_audio.merge(df_member1, on='id', how='inner') \
                     .merge(df_member2, on='id', how='inner')

# 3. Save the clean file
df_correct.to_csv("final_merged_data_fixed.csv", index=False)

print(f"✅ Fixed CSV Created Successfully!")
print(f"Shape: {df_correct.shape}")  # Should print (2011, 82)