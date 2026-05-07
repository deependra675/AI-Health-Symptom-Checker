import pandas as pd
import joblib

# loading dataset
df = pd.read_csv(r"C:\Users\SiS\Desktop\np\dataset\symptom_description.csv")

# clean column names
df.columns = df.columns.str.strip()

# check required columns
if "Disease" not in df.columns or "Description" not in df.columns:
    raise ValueError("CSV must have Disease and Description columns")

# Remove duplicates
df = df.drop_duplicates(subset=["Disease"])

# Convert to Dict.
description_dict = dict(zip(df["Disease"], df["Description"]))

# Saving processed file
joblib.dump(description_dict, "description.pkl")
print("Successfuly saved")